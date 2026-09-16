"""Parent-conditioned QM delta for the IPMR-Q anchor.

``IPMRQPCQM`` leaves the complete legacy IPMR-Q predictor in place and adds a
small, initially zero, correction to its native-80 middle coefficients.  The
correction is generated independently inside each native-40 parent from four
native-80 child tokens.  A shared deep D40 state conditions all four children
through FiLM, while a single relation-aware attention/MLP block mixes them.

The mixer is D4 equivariant by construction: it has no absolute child-position
embedding, every token transform is shared, and its only pairwise bias
distinguishes self, edge-neighbour, and diagonal-neighbour relations.  These
three relations are invariant under every rotation and reflection of a 2x2
child square.  The final scalar projection is zero initialized, so construction
from a given random seed preserves the inherited IPMR-Q function exactly at
update zero.  The inherited high-frequency proposal and all Q projection and
closure operations are unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_ipmr_q import IPMRQ, IPMRQComponents
from g246_q_bands import support_block_mean


__all__ = ["IPMRQPCQM"]


_CHILDREN_PER_PARENT = 4
_FINE_TO_CHILD_SCALE = 2
_FINE_TO_PARENT_SCALE = 4


def _relation_types() -> Tensor:
    """Return the D4-invariant relation class for every ordered child pair."""

    coordinates = ((0, 0), (0, 1), (1, 0), (1, 1))
    relation = torch.empty(
        _CHILDREN_PER_PARENT,
        _CHILDREN_PER_PARENT,
        dtype=torch.long,
    )
    for query, (qy, qx) in enumerate(coordinates):
        for key, (ky, kx) in enumerate(coordinates):
            distance = abs(qy - ky) + abs(qx - kx)
            # 0=self, 1=edge neighbour, 2=diagonal neighbour.
            relation[query, key] = distance
    return relation


def _pack_children(value80: Tensor) -> Tensor:
    """Pack a native-80 NCHW map into four tokens per native-40 parent."""

    if value80.ndim != 4:
        raise ValueError("native-80 child features must have NCHW rank 4")
    batch, channels, height, width = value80.shape
    if height % 2 or width % 2:
        raise ValueError("native-80 geometry must be divisible by two")
    parent_height, parent_width = height // 2, width // 2
    return (
        value80.reshape(batch, channels, parent_height, 2, parent_width, 2)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, parent_height, parent_width, _CHILDREN_PER_PARENT, channels)
    )


def _unpack_child_scalars(tokens: Tensor) -> Tensor:
    """Invert :func:`_pack_children` for one scalar per child token."""

    if tokens.ndim != 5 or tokens.shape[-2:] != (_CHILDREN_PER_PARENT, 1):
        raise ValueError("child scalars must have shape [B,H40,W40,4,1]")
    batch, parent_height, parent_width, _, _ = tokens.shape
    return (
        tokens[..., 0]
        .reshape(batch, parent_height, parent_width, 2, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, 1, 2 * parent_height, 2 * parent_width)
    )


class _D4ParentTokenBlock(nn.Module):
    """One D4-equivariant four-child attention and shared-token MLP block."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("token channels must be a positive integer")
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise ValueError("attention heads must be a positive integer")
        if channels % heads:
            raise ValueError("token channels must be divisible by attention heads")
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, int) or mlp_ratio < 1:
            raise ValueError("mlp_ratio must be a positive integer")

        self.channels = channels
        self.heads = heads
        self.head_channels = channels // heads
        self.scale = self.head_channels ** -0.5
        self.attention_norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(heads, 3))
        self.attention_output = nn.Linear(channels, channels, bias=False)
        self.mlp_norm = nn.LayerNorm(channels)
        hidden = mlp_ratio * channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.register_buffer(
            "relation_type",
            _relation_types(),
            persistent=True,
        )

    def forward(self, tokens: Tensor, child_active: Tensor) -> Tensor:
        if tokens.ndim != 5 or tokens.shape[-2:] != (
            _CHILDREN_PER_PARENT,
            self.channels,
        ):
            raise ValueError(
                "tokens must have shape [B,H40,W40,4,token_channels]"
            )
        if child_active.shape != tokens.shape[:-1]:
            raise ValueError("child_active must have shape [B,H40,W40,4]")
        if child_active.dtype != torch.bool:
            raise TypeError("child_active must be boolean")
        if tokens.device != child_active.device:
            raise ValueError("tokens and child_active must share a device")

        batch, parent_height, parent_width, children, channels = tokens.shape
        parents = batch * parent_height * parent_width
        active = child_active.reshape(parents, children)
        values = tokens.reshape(parents, children, channels)
        values = values * active[..., None].to(dtype=values.dtype)

        qkv = self.qkv(self.attention_norm(values)).reshape(
            parents,
            children,
            3,
            self.heads,
            self.head_channels,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        relation_bias = self.relation_bias[:, self.relation_type]
        logits = logits + relation_bias[None].to(dtype=logits.dtype)

        # Empty parents are legal outside physical support.  Give their first
        # token a numerical-only key, then mask every output by the true support
        # below.  The externally visible all-empty result remains exactly zero.
        first = torch.zeros(children, dtype=torch.bool, device=active.device)
        first[0] = True
        fallback = (~active.any(dim=-1, keepdim=True)) & first[None]
        safe_active = active | fallback
        minimum = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~safe_active[:, None, None, :], minimum)
        weights = F.softmax(logits, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(parents, children, channels)
        values = values + self.attention_output(attended)
        values = values * active[..., None].to(dtype=values.dtype)
        values = values + self.mlp(self.mlp_norm(values))
        values = values * active[..., None].to(dtype=values.dtype)
        return values.reshape(
            batch,
            parent_height,
            parent_width,
            children,
            channels,
        )


class IPMRQPCQM(IPMRQ):
    """IPMR-Q plus a zero-initialized parent-conditioned QM coefficient delta."""

    schema_version = "g246-ipmr-q-pcqm-v1"
    pcqm_contract = (
        "Fine52[2:]+support -> narrow shared 1x1 stem -> support-aware P80 "
        "children -> D40 FiLM -> one D4-equivariant four-token block -> "
        "zero-initialized delta added to legacy middle coefficients"
    )
    anchor_contract = (
        "legacy IPMR-Q middle head retained; PCQM delta is exact zero at "
        "initialization; inherited QH proposal path and Q closure unchanged"
    )
    equivariance_contract = (
        "no absolute child position; shared token maps; attention relation bias "
        "has self/edge/diagonal classes only"
    )
    guidance_channel_slice = (2, 52)

    def __init__(
        self,
        *,
        fine_channels: int = 52,
        context_dim: int = 19,
        width: int = 48,
        activation_checkpointing: bool = True,
        channels: Sequence[int] | None = None,
        blocks_per_scale: int = 2,
        selector_rank: int = 32,
        pcqm_width: int = 32,
        pcqm_heads: int = 4,
        pcqm_mlp_ratio: int = 2,
        film_bound: float = 0.25,
    ) -> None:
        super().__init__(
            fine_channels=fine_channels,
            context_dim=context_dim,
            width=width,
            activation_checkpointing=activation_checkpointing,
            channels=channels,
            blocks_per_scale=blocks_per_scale,
            selector_rank=selector_rank,
        )
        if fine_channels != 52:
            raise ValueError("IPMRQPCQM requires the exact Fine52 contract")
        if isinstance(pcqm_width, bool) or not isinstance(pcqm_width, int) \
                or pcqm_width < 1:
            raise ValueError("pcqm_width must be a positive integer")
        if not math.isfinite(film_bound) or film_bound < 0.0:
            raise ValueError("film_bound must be finite and nonnegative")

        self.pcqm_width = pcqm_width
        self.pcqm_heads = int(pcqm_heads)
        self.pcqm_mlp_ratio = int(pcqm_mlp_ratio)
        self.pcqm_film_bound = float(film_bound)
        guidance_inputs = fine_channels - self.guidance_channel_slice[0] + 1
        self.pcqm_guidance_stem = nn.Sequential(
            nn.Conv2d(guidance_inputs, pcqm_width, 1, bias=False),
            nn.SiLU(inplace=False),
        )
        self.pcqm_deep_film = nn.Conv2d(
            self.channels["parent"],
            2 * pcqm_width,
            1,
        )
        self.pcqm_token_block = _D4ParentTokenBlock(
            pcqm_width,
            heads=pcqm_heads,
            mlp_ratio=pcqm_mlp_ratio,
        )
        self.pcqm_output = nn.Linear(pcqm_width, 1)
        self._initialize_pcqm()

    def _initialize_pcqm(self) -> None:
        """Initialize only new modules, leaving inherited anchor weights intact."""

        for module in (
            self.pcqm_guidance_stem,
            self.pcqm_deep_film,
            self.pcqm_token_block,
        ):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        child.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.LayerNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)
        nn.init.zeros_(self.pcqm_output.weight)
        nn.init.zeros_(self.pcqm_output.bias)

    def pcqm_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over the disjoint PCQM parameter partition."""

        modules = (
            self.pcqm_guidance_stem,
            self.pcqm_deep_film,
            self.pcqm_token_block,
            self.pcqm_output,
        )
        for module in modules:
            yield from module.parameters()

    def freeze_anchor(self) -> "IPMRQPCQM":
        """Freeze inherited IPMR-Q parameters and leave the PCQM delta trainable."""

        pcqm_ids = {id(parameter) for parameter in self.pcqm_parameters()}
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in pcqm_ids)
        return self

    @property
    def pcqm_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.pcqm_parameters())

    @property
    def anchor_parameter_count(self) -> int:
        return self.parameter_count - self.pcqm_parameter_count

    @property
    def parameter_partition(self) -> dict[str, int]:
        return {
            "anchor": self.anchor_parameter_count,
            "pcqm": self.pcqm_parameter_count,
            "total": self.parameter_count,
        }

    def _pcqm_delta(
        self,
        fine: Tensor,
        support: Tensor,
        deep40: Tensor,
    ) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != self.fine_channels:
            raise ValueError("PCQM fine input must have shape [B,52,H,W]")
        if support.shape != fine[:, :1].shape:
            raise ValueError("PCQM support must have shape [B,1,H,W]")
        if support.dtype == torch.bool:
            support_bool = support
        else:
            if support.is_floating_point() and bool(
                torch.any(~torch.isfinite(support)).detach().cpu().item()
            ):
                raise ValueError("PCQM support must be finite and binary")
            if bool(torch.any((support != 0) & (support != 1)).detach().cpu().item()):
                raise ValueError("PCQM support must contain only 0/1 values")
            support_bool = support.to(dtype=torch.bool)
        if fine.shape[-2] % _FINE_TO_PARENT_SCALE \
                or fine.shape[-1] % _FINE_TO_PARENT_SCALE:
            raise ValueError("PCQM fine geometry must be divisible by four")
        expected_deep = (
            fine.shape[0],
            self.channels["parent"],
            fine.shape[-2] // _FINE_TO_PARENT_SCALE,
            fine.shape[-1] // _FINE_TO_PARENT_SCALE,
        )
        if deep40.shape != expected_deep:
            raise ValueError(
                f"PCQM D40 state must have shape {expected_deep}, got {tuple(deep40.shape)}"
            )
        if len({fine.device, support.device, deep40.device}) != 1:
            raise ValueError("PCQM inputs must share a device")
        if not fine.is_floating_point() or not deep40.is_floating_point():
            raise TypeError("PCQM fine and D40 inputs must be floating tensors")

        support_value = support_bool.to(dtype=fine.dtype)
        guidance = torch.cat(
            (fine[:, self.guidance_channel_slice[0] :], support_value),
            dim=1,
        )
        embedded160 = self.pcqm_guidance_stem(guidance)
        child80, child_active80 = support_block_mean(
            embedded160,
            support_bool,
            _FINE_TO_CHILD_SCALE,
        )
        tokens = _pack_children(child80)
        child_active = _pack_children(
            child_active80.to(dtype=child80.dtype)
        )[..., 0].to(dtype=torch.bool)

        film = self.pcqm_deep_film(deep40).permute(0, 2, 3, 1)
        scale, shift = film.chunk(2, dim=-1)
        scale = self.pcqm_film_bound * torch.tanh(scale)
        shift = self.pcqm_film_bound * shift
        tokens = tokens * (1.0 + scale[..., None, :]) + shift[..., None, :]
        tokens = tokens * child_active[..., None].to(dtype=tokens.dtype)
        tokens = self.pcqm_token_block(tokens, child_active)
        delta_tokens = self.pcqm_output(tokens)
        delta_tokens = delta_tokens * child_active[..., None].to(
            dtype=delta_tokens.dtype
        )
        return _unpack_child_scalars(delta_tokens)

    def _band_proposals(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context15: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the inherited trunk, adding PCQM only to its middle coefficients."""

        local = torch.cat(
            (((fine[:, :1] - 300.0) / 20.0), fine[:, 1:], support),
            dim=1,
        )
        f160 = self.fine_stem(local)
        f80 = self.encode80(self.down80(f160))
        content40 = self.encode40(self.down40(f80))
        merged40 = self.merge40(
            content40,
            self._physical_token40(fine, coarse_k, support, context15),
        )
        f20 = self.encode20(self.down20(merged40))
        f10 = self.bottleneck10(self.down10(f20))
        d20_up = self.up20(
            F.interpolate(
                f10,
                size=f20.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        d20 = self.decode20(f20, d20_up)
        d40_up = self.up40(
            F.interpolate(
                d20,
                size=merged40.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        d40 = self.decode40(merged40, d40_up)
        d80_up = self.up80(
            F.interpolate(
                d40,
                size=f80.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        d80 = self.decode80(f80, d80_up)

        legacy_middle = self.middle_head(d80)
        # Keep the inherited high-frequency path byte-for-byte in structure.
        high = self.high_head(self._condition_high(f160, d80))
        middle = legacy_middle + self._pcqm_delta(fine, support, d40)
        return middle, high


# The inherited method returns this exact component tuple.  Keeping the alias in
# the module makes the public compatibility contract explicit to importers and
# static tooling without wrapping or altering ``forward_components``.
IPMRQPCQMComponents = IPMRQComponents

