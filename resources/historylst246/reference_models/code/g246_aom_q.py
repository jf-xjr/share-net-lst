"""Identifiable support-aware building blocks for the G246 AOM-Q model.

This module intentionally contains only the two mathematical components whose
contract does not depend on the final r6a/r9 exchange topology:

* :class:`SupportAwareDualState` separates a fine predictor into a local
  absolute state and an exactly parent-centred contrast state.
* :class:`AnchorOrthogonalBand` separates a Q-band prediction into a bounded
  gain of an existing anchor and a new morphology that is exactly orthogonal
  to that anchor on every support block.

Neither component accepts coordinates, targets, city labels or region labels.
The eventual AOM-Q backbone can therefore reuse them without weakening the
locked-test or no-geolocation contracts.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_calibrated_continuous_q import SceneCalibratedContinuousQ
from g246_ipmr_q import IPMRQ
from g246_q_bands import (
    lift_p80_coefficients,
    orthogonal_q_bands,
    support_block_projection,
)
from ocnir import support_project


__all__ = [
    "AnchorOrthogonalBand",
    "AnchorOrthogonalBandComponents",
    "AOMQ",
    "AOMQComponents",
    "SupportAwareDualState",
    "SupportAwareDualStateComponents",
]


def _contains_true(value: Tensor) -> bool:
    return bool(torch.any(value).detach().cpu().item())


def _binary(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if not value.is_floating_point() and value.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise TypeError(f"{name} must be bool or a numeric binary tensor")
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


class SupportAwareDualStateComponents(NamedTuple):
    absolute: Tensor
    contrast: Tensor
    parent_mean: Tensor
    parent_active: Tensor


class SupportAwareDualState(nn.Module):
    """Exact local-absolute/parent-contrast split on supported 4x4 cells.

    For a supported field ``x``, the returned states satisfy

    ``absolute = P40^s x`` and ``contrast = s*x - absolute``.

    Consequently ``absolute + contrast == s*x`` and the support-weighted mean
    of ``contrast`` is exactly zero in every non-empty parent.  Empty parents
    remain zero rather than acquiring an epsilon-dependent artificial state.
    """

    parent_size = 4

    def forward(self, field: Tensor, support: Tensor) -> SupportAwareDualStateComponents:
        if not isinstance(field, Tensor) or field.ndim != 4:
            raise ValueError("field must have shape [B,C,H,W]")
        if not field.is_floating_point() or _contains_true(~torch.isfinite(field)):
            raise ValueError("field must be finite and floating")
        if not isinstance(support, Tensor) or support.shape != (
            field.shape[0], 1, *field.shape[-2:]
        ):
            raise ValueError("support must have shape [B,1,H,W]")
        if field.device != support.device:
            raise ValueError("field and support must be on one device")
        support_bool = _binary(support, "support")
        if field.shape[-2] % self.parent_size or field.shape[-1] % self.parent_size:
            raise ValueError("fine geometry must be divisible by four")

        absolute = support_block_projection(field, support_bool, self.parent_size)
        restricted = field * support_bool.to(dtype=field.dtype)
        contrast = restricted - absolute
        parent_mean = absolute[..., ::self.parent_size, ::self.parent_size]
        parent_active = support_bool.reshape(
            field.shape[0], 1,
            field.shape[-2] // self.parent_size, self.parent_size,
            field.shape[-1] // self.parent_size, self.parent_size,
        ).any(dim=(3, 5))
        return SupportAwareDualStateComponents(
            absolute=absolute,
            contrast=contrast,
            parent_mean=parent_mean,
            parent_active=parent_active,
        )


class AnchorOrthogonalBandComponents(NamedTuple):
    value: Tensor
    gained_anchor: Tensor
    morphology: Tensor
    anchor_band: Tensor
    proposal_band: Tensor
    gain: Tensor
    anchor_energy: Tensor


class AnchorOrthogonalBand(nn.Module):
    """Identifiable bounded-anchor-gain plus orthogonal-morphology layer.

    Middle-band quantities are resolved independently in each 4x4 observation
    parent; high-band quantities are resolved in each 2x2 fine child.  The gain
    is a scalar on that same block, so multiplying the anchor cannot leave its
    registered Q band.  On non-degenerate blocks the morphology is projected
    exactly perpendicular to the anchor.  If the anchor has zero energy, the
    perpendicular constraint is vacuous and the proposal is retained.
    """

    max_log_gain = 0.35
    energy_epsilon = 1.0e-12

    def __init__(self, band: Literal["middle", "high"]) -> None:
        super().__init__()
        if band not in ("middle", "high"):
            raise ValueError("band must be 'middle' or 'high'")
        self.band = band
        self.block_size = 4 if band == "middle" else 2

    def _select_band(
        self, field: Tensor, support: Tensor, coarse_valid: Tensor
    ) -> Tensor:
        bands = orthogonal_q_bands(field, support, coarse_valid)
        return bands.q_middle if self.band == "middle" else bands.q_high

    def _block_inner(self, left: Tensor, right: Tensor, support: Tensor) -> Tensor:
        size = self.block_size
        batch, channels, height, width = left.shape
        weights = support.to(dtype=left.dtype).reshape(
            batch, 1, height // size, size, width // size, size
        )
        product = (left * right).reshape(
            batch, channels, height // size, size, width // size, size
        )
        return (product * weights).sum(dim=(3, 5))

    def _lift(self, blocks: Tensor) -> Tensor:
        return blocks.repeat_interleave(self.block_size, -2).repeat_interleave(
            self.block_size, -1
        )

    def forward(
        self,
        anchor_q: Tensor,
        proposal: Tensor,
        gain_logits: Tensor,
        support: Tensor,
        coarse_valid: Tensor,
    ) -> AnchorOrthogonalBandComponents:
        if not isinstance(anchor_q, Tensor) or anchor_q.ndim != 4 \
                or anchor_q.shape[1] != 1 or not anchor_q.is_floating_point():
            raise ValueError("anchor_q must be floating [B,1,H,W]")
        if not isinstance(proposal, Tensor) or proposal.shape != anchor_q.shape \
                or not proposal.is_floating_point():
            raise ValueError("proposal must match anchor_q")
        if _contains_true(~torch.isfinite(anchor_q)) \
                or _contains_true(~torch.isfinite(proposal)):
            raise ValueError("anchor_q and proposal must be finite")
        if not isinstance(support, Tensor) or support.shape != anchor_q.shape:
            raise ValueError("support must match anchor_q")
        support_bool = _binary(support, "support")
        expected_gain = (
            anchor_q.shape[0], 1,
            anchor_q.shape[-2] // self.block_size,
            anchor_q.shape[-1] // self.block_size,
        )
        if not isinstance(gain_logits, Tensor) or gain_logits.shape != expected_gain:
            raise ValueError(f"gain_logits must have shape {expected_gain}")
        if not gain_logits.is_floating_point() \
                or _contains_true(~torch.isfinite(gain_logits)):
            raise ValueError("gain_logits must be finite and floating")
        expected_valid = (
            anchor_q.shape[0], 1,
            anchor_q.shape[-2] // 4, anchor_q.shape[-1] // 4,
        )
        if not isinstance(coarse_valid, Tensor) or coarse_valid.shape != expected_valid:
            raise ValueError("coarse_valid must have shape [B,1,H/4,W/4]")
        coarse_bool = _binary(coarse_valid, "coarse_valid")
        if len({anchor_q.device, proposal.device, gain_logits.device,
                support.device, coarse_valid.device}) != 1:
            raise ValueError("all inputs must be on one device")

        anchor_band = self._select_band(anchor_q, support_bool, coarse_bool)
        proposal_band = self._select_band(proposal, support_bool, coarse_bool)
        anchor_energy = self._block_inner(
            anchor_band, anchor_band, support_bool
        )
        cross = self._block_inner(anchor_band, proposal_band, support_bool)
        nondegenerate = anchor_energy > self.energy_epsilon
        coefficient = torch.where(
            nondegenerate,
            cross / torch.where(
                nondegenerate, anchor_energy, torch.ones_like(anchor_energy)
            ),
            torch.zeros_like(cross),
        )
        morphology = proposal_band - anchor_band * self._lift(coefficient)
        morphology = morphology * support_bool.to(dtype=morphology.dtype)

        gain = torch.exp(
            self.max_log_gain * torch.tanh(gain_logits.float())
        ).to(dtype=anchor_band.dtype)
        gained_anchor = anchor_band * self._lift(gain)
        value = gained_anchor + morphology
        return AnchorOrthogonalBandComponents(
            value=value,
            gained_anchor=gained_anchor,
            morphology=morphology,
            anchor_band=anchor_band,
            proposal_band=proposal_band,
            gain=gain,
            anchor_energy=anchor_energy,
        )


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Project(nn.Sequential):
    def __init__(
        self, inputs: int, outputs: int, *, kernel: int = 1, stride: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(inputs, outputs, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(inplace=False),
        )


class _Residual(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel, padding=kernel // 2,
            groups=channels, bias=False,
        )
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1), nn.SiLU(inplace=False),
            nn.Conv2d(2 * channels, channels, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.mix(F.silu(self.norm(self.depthwise(value))))


class _ZeroExchange(nn.Conv2d):
    """Zero-start, bounded cross-route that cannot overwrite a native stream."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__(inputs, outputs, 1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, value: Tensor) -> Tensor:
        # Preserve the two deliberately different inductive biases.  Exchange
        # can supply missing evidence but cannot grow into an unbounded shortcut
        # that numerically erases the receiving backbone's native state.
        return 0.25 * torch.tanh(super().forward(value))


class _DualMorphologyEncoder(nn.Module):
    """Raw Fine52 A/C encoder with an explicit 40->80->160 synthesis path."""

    def __init__(self) -> None:
        super().__init__()
        self.absolute160 = _Project(52, 48, kernel=3)
        self.contrast160 = nn.Sequential(
            _Project(52, 64, kernel=3), _Residual(64, 3), _Residual(64, 5)
        )
        self.down_a80 = _Project(48, 64, kernel=3, stride=2)
        self.down_c80 = _Project(64, 64, kernel=3, stride=2)
        self.down_a40 = _Project(64, 96, kernel=3, stride=2)
        self.down_c40 = _Project(64, 96, kernel=3, stride=2)
        self.fuse40 = nn.Sequential(
            _Project(96 + 96 + 144 + 144 + 2, 112, kernel=3),
            _Residual(112, 3), _Residual(112, 5),
        )
        self.fuse80 = nn.Sequential(
            _Project(64 + 64 + 96 + 112 + 3, 96, kernel=3),
            _Residual(96, 3), _Residual(96, 5),
        )
        self.fuse160 = nn.Sequential(
            _Project(48 + 64 + 64 + 64 + 96 + 3, 64, kernel=3),
            _Residual(64, 3), _Residual(64, 5),
        )
        self.middle_proposal = nn.Conv2d(96, 1, 3, padding=1)
        self.high_proposal = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.middle_gain = nn.Conv2d(112, 1, 1)
        self.high_gain = nn.Conv2d(96, 1, 1)
        for head in (
            self.middle_proposal, self.high_proposal,
            self.middle_gain, self.high_gain,
        ):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        absolute: Tensor,
        contrast: Tensor,
        shallow160: Tensor,
        shallow40: Tensor,
        deep160: Tensor,
        deep80: Tensor,
        deep40: Tensor,
        anchor_q: Tensor,
        support: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a160 = self.absolute160(absolute)
        c160 = self.contrast160(contrast)
        a80, c80 = self.down_a80(a160), self.down_c80(c160)
        a40, c40 = self.down_a40(a80), self.down_c40(c80)
        support40 = F.avg_pool2d(support.float(), 4, 4)
        anchor40 = F.avg_pool2d(anchor_q.abs(), 4, 4)
        g40 = self.fuse40(torch.cat(
            (a40, c40, shallow40, deep40, support40, anchor40), dim=1
        ))
        support80 = F.max_pool2d(support.float(), 2, 2)
        anchor80 = F.avg_pool2d(anchor_q.abs(), 2, 2)
        g80 = self.fuse80(torch.cat((
            a80, c80, deep80,
            F.interpolate(g40, size=deep80.shape[-2:], mode="bilinear",
                          align_corners=False),
            support80, anchor80,
            F.avg_pool2d(anchor_q.square(), 2, 2).sqrt(),
        ), dim=1))
        g160 = self.fuse160(torch.cat((
            a160, c160, shallow160, deep160,
            F.interpolate(g80, size=shallow160.shape[-2:], mode="bilinear",
                          align_corners=False),
            support.float(), anchor_q, anchor_q.abs(),
        ), dim=1))
        return (
            self.middle_proposal(g80), self.high_proposal(g160),
            self.middle_gain(g40), self.high_gain(g80),
        )


class _PairInteraction(nn.Module):
    """Align two backbone states and expose agreement and complementarity."""

    def __init__(self, shallow_channels: int, deep_channels: int, width: int) -> None:
        super().__init__()
        self.shallow = nn.Conv2d(shallow_channels, width, 1, bias=False)
        self.deep = nn.Conv2d(deep_channels, width, 1, bias=False)
        self.merge = nn.Sequential(
            _Project(4 * width, width), _Residual(width, 3), _Residual(width, 5)
        )

    def forward(self, shallow: Tensor, deep: Tensor) -> Tensor:
        s = self.shallow(shallow)
        d = self.deep(deep)
        # Difference preserves complementary signed evidence; the bounded
        # product represents feature agreement without an expert-selection gate.
        return self.merge(torch.cat((s, d, s - d, torch.tanh(s) * torch.tanh(d)), 1))


class _LightBandFusion(nn.Module):
    """Sub-0.6M band decoder over the two exchanged native backbones only."""

    def __init__(self) -> None:
        super().__init__()
        self.pair40 = _PairInteraction(144, 144, 48)
        self.pair80 = _PairInteraction(64, 96, 48)
        self.pair160 = _PairInteraction(64, 64, 40)
        self.fuse40 = nn.Sequential(_Project(48, 48), _Residual(48, 3))
        self.fuse80 = nn.Sequential(
            _Project(48 + 48, 56, kernel=3), _Residual(56, 3)
        )
        self.fuse160 = nn.Sequential(
            _Project(40 + 56, 48, kernel=3), _Residual(48, 3), _Residual(48, 5)
        )
        self.middle_proposal = nn.Conv2d(56, 1, 3, padding=1)
        self.high_proposal = nn.Conv2d(48, 1, 3, padding=1, bias=False)
        self.middle_gain = nn.Conv2d(48, 1, 1)
        self.high_gain = nn.Conv2d(56, 1, 1)
        for head in (
            self.middle_proposal, self.high_proposal,
            self.middle_gain, self.high_gain,
        ):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        shallow160: Tensor,
        shallow80: Tensor,
        shallow40: Tensor,
        deep160: Tensor,
        deep80: Tensor,
        deep40: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        g40 = self.fuse40(self.pair40(shallow40, deep40))
        p80 = self.pair80(shallow80, deep80)
        g80 = self.fuse80(torch.cat((
            p80, F.interpolate(g40, size=p80.shape[-2:], mode="bilinear",
                               align_corners=False),
        ), 1))
        p160 = self.pair160(shallow160, deep160)
        g160 = self.fuse160(torch.cat((
            p160, F.interpolate(g80, size=p160.shape[-2:], mode="bilinear",
                                align_corners=False),
        ), 1))
        return (
            self.middle_proposal(g80), self.high_proposal(g160),
            self.middle_gain(g40), self.high_gain(g80),
        )


class AOMQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_k: Tensor
    q_middle_k: Tensor
    q_high_k: Tensor
    shallow_prediction_k: Tensor
    deep_prediction_k: Tensor
    shallow_q_k: Tensor
    deep_q_k: Tensor
    half_prediction_k: Tensor
    half_q_k: Tensor
    gain_only_prediction_k: Tensor
    shape_only_prediction_k: Tensor
    gain_q_k: Tensor
    morphology_q_k: Tensor
    middle: AnchorOrthogonalBandComponents
    high: AnchorOrthogonalBandComponents


class AOMQ(nn.Module):
    """Jointly trainable r6a/r9 backbones with identifiable AOM-Q output.

    The source branches exchange features in both directions at 40, 80 and 160
    through zero-initialised adapters.  Thus u0 is the exact per-band half
    anchor, while every source and exchange parameter remains trainable.  The
    raw Fine52 dual-state morphology encoder never consumes either source's
    hidden feature *instead of* the predictors; source features only provide
    explicit semantic/detail context at its three synthesis lattices.
    """

    schema_version = "g246-aom-q-v1"
    geolocation_context_indices = (5, 6, 7, 8)
    source_expert_names = ("r6a_context15_continuous", "r9_ipmr_q")
    scale_path = "coupled(r6a,r9):40<->80<->160 + dual-state 40->80->160"

    def __init__(self) -> None:
        super().__init__()
        self.shallow_expert = SceneCalibratedContinuousQ(
            fine_channels=52, context_dim=19, width=48,
            activation_checkpointing=False, no_geo_core=True,
        )
        self.deep_expert = IPMRQ(
            fine_channels=52, context_dim=19, width=48,
            activation_checkpointing=False,
        )
        self.s_from_d40 = _ZeroExchange(144, 144)
        self.d_from_s40 = _ZeroExchange(144, 144)
        self.s_from_d80 = _ZeroExchange(96, 64)
        self.d_from_s80 = _ZeroExchange(64, 96)
        self.s_from_d160 = _ZeroExchange(64, 64)
        self.d_from_s160 = _ZeroExchange(64, 64)
        self.shallow_down80 = _Project(64, 64, kernel=3, stride=2)
        # The support-aware mathematical split remains public for future input
        # studies, but the evidence-backed default does not instantiate a third
        # raw-predictor encoder alongside the two complete source backbones.
        self.morphology_encoder = _LightBandFusion()
        self.middle_band = AnchorOrthogonalBand("middle")
        self.high_band = AnchorOrthogonalBand("high")
        self.register_buffer(
            "experts_initialized", torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_expert_state_dicts(self, shallow: dict[str, Tensor], deep: dict[str, Tensor]) -> None:
        if bool(self.experts_initialized.item()):
            raise RuntimeError("AOM-Q experts may be initialized only once")
        self.shallow_expert.load_state_dict(shallow, strict=True)
        self.deep_expert.load_state_dict(deep, strict=True)
        self.experts_initialized.fill_(True)

    @staticmethod
    def _context15(context: Tensor) -> Tensor:
        return torch.cat((context[..., :5], context[..., 9:]), dim=-1)

    def _validate(
        self, fine: Tensor, coarse: Tensor, support: Tensor, context: Tensor,
        available: Tensor, query: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not bool(self.experts_initialized.item()):
            raise RuntimeError("AOM-Q experts have not been initialized")
        if fine.ndim != 5 or fine.shape[1:3] != (1, 52):
            raise ValueError("fine must have shape [B,1,52,H,W]")
        b, _, _, h, w = fine.shape
        if h % 16 or w % 16:
            raise ValueError("fine geometry must be divisible by sixteen")
        if coarse.shape != (b, 1, 1, h // 4, w // 4):
            raise ValueError("coarse_k geometry differs")
        if support.shape != (b, 1, 1, h, w) or context.shape != (b, 1, 19):
            raise ValueError("support/context geometry differs")
        if available.shape != (b, 1) or query.shape != (b,) \
                or _contains_true(~_binary(available, "temporal_available")) \
                or _contains_true(query != 0):
            raise ValueError("AOM-Q requires one available query at index zero")
        return fine[:, 0], coarse[:, 0], _binary(support[:, 0], "support"), context[:, 0]

    def _coupled_features(
        self, fine: Tensor, coarse: Tensor, support: Tensor, context15: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        s = self.shallow_expert
        d = self.deep_expert
        batch = fine.shape[0]
        support_float = support.to(dtype=fine.dtype)
        available = torch.ones(batch, dtype=torch.bool, device=fine.device)
        query = torch.zeros(batch, dtype=torch.long, device=fine.device)
        s160_native, token40, descriptor = s.date_encoder(
            fine, coarse, support_float, context15, available
        )
        s40_native = s.context_unet(s.temporal_fusion(
            token40[:, None], available[:, None], query
        ))
        scene = s.scene_fusion(torch.cat(
            (descriptor, descriptor, torch.zeros_like(descriptor)), dim=1
        ))

        local = torch.cat(
            (((fine[:, :1] - 300.0) / 20.0), fine[:, 1:], support_float), dim=1
        )
        d160_native = d.fine_stem(local)
        f80 = d.encode80(d.down80(d160_native))
        content40 = d.encode40(d.down40(f80))
        merged40 = d.merge40(content40, d._physical_token40(
            fine, coarse, support_float, context15
        ))
        f20 = d.encode20(d.down20(merged40))
        f10 = d.bottleneck10(d.down10(f20))
        d20 = d.decode20(f20, d.up20(F.interpolate(
            f10, size=f20.shape[-2:], mode="bilinear", align_corners=False
        )))
        d40_native = d.decode40(merged40, d.up40(F.interpolate(
            d20, size=merged40.shape[-2:], mode="bilinear", align_corners=False
        )))

        s40 = s40_native + self.s_from_d40(d40_native)
        d40 = d40_native + self.d_from_s40(s40_native)
        d80_native = d.decode80(f80, d.up80(F.interpolate(
            d40, size=f80.shape[-2:], mode="bilinear", align_corners=False
        )))

        head = s.continuous_head
        parent_up = head.fine_projection(F.interpolate(
            s40, size=s160_native.shape[-2:], mode="bilinear", align_corners=False
        ))
        shallow160 = head.fuse(torch.cat((s160_native, parent_up), dim=1))
        scale, shift = head.scene_film(scene).chunk(2, dim=1)
        shallow160 = shallow160 * (1.0 + 0.15 * torch.tanh(scale)[..., None, None])
        shallow160 = shallow160 + 0.15 * shift[..., None, None]
        shallow160 = head.blocks(shallow160)
        shallow80 = self.shallow_down80(shallow160)
        d80 = d80_native + self.d_from_s80(shallow80)
        shallow160 = shallow160 + F.interpolate(
            self.s_from_d80(d80_native), size=shallow160.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        s160_pre = shallow160
        shallow160 = shallow160 + self.s_from_d160(d160_native)
        deep160 = d160_native + self.d_from_s160(s160_pre)
        return (
            shallow160, shallow80, s40, scene, deep160, d80, d40,
            parent_up, support_float,
        )

    def _branch_q(
        self, fine: Tensor, coarse: Tensor, support: Tensor,
        shallow160: Tensor, scene: Tensor, deep160: Tensor, deep80: Tensor,
        parent_up: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sh = self.shallow_expert.continuous_head
        expert_values = torch.cat((
            sh.local_expert(shallow160), sh.edge_expert(shallow160),
            sh.context_expert(parent_up),
        ), dim=1)
        expert_values = expert_values * (
            1.0 + 0.25 * torch.tanh(sh.scene_gain(scene))
        )[..., None, None]
        weights = torch.softmax(
            (sh.spatial_gate(shallow160) + sh.scene_gate(scene)[..., None, None]).float(),
            dim=1,
        ).to(dtype=shallow160.dtype)
        shallow_raw = (expert_values * weights).sum(dim=1, keepdim=True)
        valid = torch.isfinite(coarse)
        shallow_q = support_project(
            shallow_raw, torch.zeros_like(coarse), support, valid
        ) * support
        base = support_project(fine[:, :1], coarse, support)

        deep = self.deep_expert
        middle = deep.middle_head(deep80)
        high = deep.high_head(deep._condition_high(deep160, deep80))
        deep_q = orthogonal_q_bands(
            lift_p80_coefficients(middle, support).float(), support, valid
        ).q_middle + orthogonal_q_bands(high.float(), support, valid).q_high
        deep_q = support_project(
            deep_q, torch.zeros_like(coarse, dtype=deep_q.dtype), support, valid
        ) * support
        return shallow_q.float(), deep_q.float(), base.float()

    def forward_components(
        self, fine: Tensor, coarse_k: Tensor, support: Tensor, context: Tensor,
        temporal_available: Tensor, query_index: Tensor,
    ) -> AOMQComponents:
        x, coarse, support_bool, ctx = self._validate(
            fine, coarse_k, support, context, temporal_available, query_index
        )
        context15 = self._context15(ctx)
        features = self._coupled_features(x, coarse, support_bool, context15)
        (shallow160, shallow80, shallow40, scene, deep160, deep80, deep40,
         parent_up, sfloat) = features
        shallow_q, deep_q, base = self._branch_q(
            x, coarse, support_bool, shallow160, scene, deep160, deep80, parent_up
        )
        valid = torch.isfinite(coarse)
        shallow_bands = orthogonal_q_bands(shallow_q, support_bool, valid)
        deep_bands = orthogonal_q_bands(deep_q, support_bool, valid)
        anchor_middle = 0.5 * (shallow_bands.q_middle + deep_bands.q_middle)
        anchor_high = 0.5 * (shallow_bands.q_high + deep_bands.q_high)
        anchor_q = anchor_middle + anchor_high

        middle_raw, high_raw, middle_gain, high_gain = self.morphology_encoder(
            shallow160, shallow80, shallow40, deep160, deep80, deep40,
        )
        middle_proposal = lift_p80_coefficients(middle_raw, support_bool)
        middle = self.middle_band(
            anchor_middle, middle_proposal, middle_gain, support_bool, valid
        )
        high = self.high_band(
            anchor_high, high_raw, high_gain, support_bool, valid
        )
        q_pre = middle.value + high.value
        q = support_project(
            q_pre, torch.zeros_like(coarse, dtype=q_pre.dtype), support_bool, valid
        ) * sfloat
        gained = middle.gained_anchor + high.gained_anchor
        morphology = middle.morphology + high.morphology
        half_q = anchor_q
        return AOMQComponents(
            prediction_k=base + q, base_k=base, q_k=q,
            q_middle_k=middle.value, q_high_k=high.value,
            shallow_prediction_k=base + shallow_q,
            deep_prediction_k=base + deep_q,
            shallow_q_k=shallow_q, deep_q_k=deep_q,
            half_prediction_k=base + half_q, half_q_k=half_q,
            gain_only_prediction_k=base + gained,
            shape_only_prediction_k=base + half_q + morphology,
            gain_q_k=gained, morphology_q_k=morphology,
            middle=middle, high=high,
        )

    def forward(
        self, fine: Tensor, coarse_k: Tensor, support: Tensor, context: Tensor,
        temporal_available: Tensor, query_index: Tensor,
    ) -> Tensor:
        return self.forward_components(
            fine, coarse_k, support, context, temporal_available, query_index
        ).prediction_k
