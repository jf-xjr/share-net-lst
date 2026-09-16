"""Observable-conditioned nullspace information reconstruction (OCNIR).

The public surface is intentionally small.  ``support_project`` implements a
target-free, differentiable 4x support-aware data-consistency layer, while
``OCNIR`` predicts a free Kelvin field and applies that layer before returning
it.  The first fine input channel is therefore kept in physical Kelvin units;
the remaining channels may be normalized by the training runner.

Invalid coarse parents are represented by non-finite ``coarse_k`` values when
``coarse_valid`` is omitted.  They are not projected.  A parent declared valid
must have a finite coarse value and at least one genuinely supported fine
pixel; fabricating support is deliberately not a fallback.
"""

from __future__ import annotations

from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


__all__ = ["OCNIR", "support_project"]

_SCALE = 4
_WIDTH_PRESETS = (48, 64, 96)


def _is_floating(tensor: Tensor) -> bool:
    return tensor.is_floating_point()


def _contains_true(value: Tensor) -> bool:
    """Return a scalar predicate with one deliberate device synchronization."""
    return bool(torch.any(value).detach().cpu().item())


def _binary_mask(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if _is_floating(value) and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


def _projection_shapes(
    raw_k: Tensor,
    coarse_k: Tensor,
    support120: Tensor,
    coarse_valid: Tensor | None,
) -> None:
    for value, name in (
        (raw_k, "raw_k"),
        (coarse_k, "coarse_k"),
        (support120, "support120"),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4:
            raise ValueError(f"{name} must have NCHW rank 4")
    if raw_k.shape[1] != 1 or coarse_k.shape[1] != 1 or support120.shape[1] != 1:
        raise ValueError("raw_k, coarse_k, and support120 must each have one channel")
    if support120.shape != raw_k.shape:
        raise ValueError("support120 must have exactly the raw_k shape")
    if raw_k.shape[0] != coarse_k.shape[0]:
        raise ValueError("raw_k and coarse_k batch sizes differ")
    expected = (_SCALE * coarse_k.shape[-2], _SCALE * coarse_k.shape[-1])
    if raw_k.shape[-2:] != expected:
        raise ValueError("fine/coarse geometry must be an exact 4x relation")
    if coarse_valid is not None:
        if not isinstance(coarse_valid, Tensor):
            raise TypeError("coarse_valid must be a torch.Tensor or None")
        if coarse_valid.shape != coarse_k.shape:
            raise ValueError("coarse_valid must have exactly the coarse_k shape")
    devices = {raw_k.device, coarse_k.device, support120.device}
    if coarse_valid is not None:
        devices.add(coarse_valid.device)
    if len(devices) != 1:
        raise ValueError("projection tensors must be on one device")
    if not _is_floating(raw_k) or not _is_floating(coarse_k):
        raise TypeError("raw_k and coarse_k must have floating dtypes")


def support_project(
    raw_k: Tensor,
    coarse_k: Tensor,
    support120: Tensor,
    coarse_valid: Tensor | None = None,
) -> Tensor:
    """Project a raw Kelvin field onto its observable coarse constraints.

    The operator is a support-restricted Euclidean projection independently in
    every 4x4 parent.  For a valid parent with support ``s`` and direct value
    ``c``, it adds ``c - sum(s * raw) / sum(s)`` to supported pixels.  Thus its
    supported parent mean is ``c`` and the learned within-parent nullspace
    component is unchanged.  Unsupported pixels, and all pixels in invalid
    parents, are returned unchanged.

    When ``coarse_valid`` is ``None``, finite entries of ``coarse_k`` define
    valid parents.  Otherwise the explicit binary mask is authoritative;
    coarse values outside it may be finite or non-finite.  A valid parent with
    zero support is an inconsistent observation and raises ``ValueError``.
    """

    _projection_shapes(raw_k, coarse_k, support120, coarse_valid)
    support = _binary_mask(support120, "support120")
    valid = (
        torch.isfinite(coarse_k)
        if coarse_valid is None
        else _binary_mask(coarse_valid, "coarse_valid")
    )

    if _contains_true(~torch.isfinite(raw_k)):
        raise ValueError("raw_k must be finite")
    if _contains_true(valid & ~torch.isfinite(coarse_k)):
        raise ValueError("coarse_k must be finite wherever coarse_valid is true")

    batch, _, fine_height, fine_width = raw_k.shape
    coarse_height, coarse_width = coarse_k.shape[-2:]
    support_blocks = support.reshape(
        batch, 1, coarse_height, _SCALE, coarse_width, _SCALE
    )
    support_weight = support_blocks.to(dtype=raw_k.dtype)
    counts = support_weight.sum(dim=(3, 5))
    if _contains_true(valid & (counts == 0)):
        raise ValueError("a valid coarse parent has zero fine support")

    raw_blocks = raw_k.reshape(
        batch, 1, coarse_height, _SCALE, coarse_width, _SCALE
    )
    supported_sum = torch.where(
        support_blocks, raw_blocks, torch.zeros((), dtype=raw_k.dtype, device=raw_k.device)
    ).sum(dim=(3, 5))
    safe_counts = torch.where(valid, counts, torch.ones_like(counts))
    supported_mean = supported_sum / safe_counts

    coarse = coarse_k.to(dtype=raw_k.dtype)
    # Avoid forming NaN arithmetic on invalid parents: this matters to backward
    # even though torch.where would hide the invalid forward branch.
    safe_coarse = torch.where(valid, coarse, supported_mean)
    correction = torch.where(valid, safe_coarse - supported_mean, torch.zeros_like(safe_coarse))
    correction_fine = correction.repeat_interleave(_SCALE, dim=-2).repeat_interleave(
        _SCALE, dim=-1
    )
    if correction_fine.shape[-2:] != (fine_height, fine_width):  # defensive
        raise AssertionError("internal projection expansion changed geometry")
    return raw_k + correction_fine * support.to(dtype=raw_k.dtype)


def _groups(channels: int) -> int:
    return gcd(channels, min(8, channels))


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels), channels)


class _SpatialResidual(nn.Module):
    """Compute-efficient spatial residual block used at every fine scale."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = expansion * channels
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.norm = _normalization(channels)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.contract = nn.Conv2d(hidden, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.depthwise(inputs)
        residual = self.activation(self.norm(residual))
        residual = self.activation(self.expand(residual))
        residual = self.contract(residual)
        return inputs + residual


class _Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            _normalization(out_channels),
            nn.SiLU(inplace=False),
        )


class _Project(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _normalization(out_channels),
            nn.SiLU(inplace=False),
        )


class _GatedFusion(nn.Module):
    """Add only the observable-conditioned part selected by a learned gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, local: Tensor, observable: Tensor) -> Tensor:
        if local.shape != observable.shape:
            raise ValueError("gated fusion inputs must have identical shapes")
        weight = torch.sigmoid(self.gate(torch.cat((local, observable), dim=1)))
        return local + weight * observable


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class OCNIR(nn.Module):
    """Three-scale observable-conditioned deterministic Kelvin reconstructor.

    Registered OCNIR-S/M/L runs use ``width`` 48/64/96, respectively. Other
    positive widths remain available for smoke and memory qualification. A
    fine guidance encoder runs at H, H/2, and H/4 while a coarse thermal encoder runs from
    H/4 upward.  Learned gates fuse the two routes at all three scales.  The
    head predicts a residual from ``fine[:, 0:1]`` in Kelvin, after which
    :func:`support_project` enforces the direct coarse observation.
    """

    widths = _WIDTH_PRESETS

    def __init__(
        self,
        width: int,
        fine_channels: int = 11,
        context_dim: int = 5,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if isinstance(fine_channels, bool) or not isinstance(fine_channels, int) or fine_channels < 1:
            raise ValueError("fine_channels must be a positive integer")
        if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 1:
            raise ValueError("context_dim must be a positive integer")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be bool")

        self.width = width
        self.fine_channels = fine_channels
        self.context_dim = context_dim
        self.activation_checkpointing = activation_checkpointing
        channels = (width, 2 * width, 4 * width)

        # Support is appended explicitly even if a runner also includes it in
        # GlobalCore: the physical operator must remain visible to the model.
        self.fine_stem = nn.Sequential(
            nn.Conv2d(fine_channels + 1, channels[0], 3, padding=1, bias=False),
            _normalization(channels[0]),
            nn.SiLU(inplace=False),
        )
        self.fine_blocks = nn.ModuleList(
            [
                nn.Sequential(*(_SpatialResidual(channels[0]) for _ in range(2))),
                nn.Sequential(*(_SpatialResidual(channels[1]) for _ in range(2))),
                nn.Sequential(*(_SpatialResidual(channels[2]) for _ in range(4))),
            ]
        )
        self.fine_down = nn.ModuleList(
            [_Downsample(channels[0], channels[1]), _Downsample(channels[1], channels[2])]
        )

        # Inputs: fixed-scaled direct Kelvin, observed support fraction, and
        # direct-parent validity.  Non-finite invalid coarse entries are filled
        # before reaching convolution.
        self.coarse_stem = nn.Sequential(
            nn.Conv2d(3, channels[2], 3, padding=1, bias=False),
            _normalization(channels[2]),
            nn.SiLU(inplace=False),
        )
        self.coarse_blocks = nn.ModuleList(
            [
                nn.Sequential(*(_SpatialResidual(channels[2]) for _ in range(2))),
                nn.Sequential(*(_SpatialResidual(channels[1]) for _ in range(2))),
                nn.Sequential(*(_SpatialResidual(channels[0]) for _ in range(2))),
            ]
        )
        self.coarse_up = nn.ModuleList(
            [_Project(channels[2], channels[1]), _Project(channels[1], channels[0])]
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 2 * width),
            nn.SiLU(inplace=False),
            nn.Linear(2 * width, sum(channels)),
        )
        self.fusions = nn.ModuleList(_GatedFusion(value) for value in channels)

        self.decode_up = nn.ModuleList(
            [_Project(channels[2], channels[1]), _Project(channels[1], channels[0])]
        )
        self.decode_merge = nn.ModuleList(
            [_Project(2 * channels[1], channels[1]), _Project(2 * channels[0], channels[0])]
        )
        self.decode_blocks = nn.ModuleList(
            [
                nn.Sequential(*(_SpatialResidual(channels[2]) for _ in range(2))),
                nn.Sequential(*(_SpatialResidual(channels[1]) for _ in range(2))),
                nn.Sequential(*(_SpatialResidual(channels[0]) for _ in range(2))),
            ]
        )
        self.head = nn.Sequential(
            _normalization(channels[0]),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels[0], 1, 3, padding=1),
        )

        self.apply(_initialize)
        # Start from the physically meaningful low-resolution Kelvin base.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _run(self, module: nn.Module, *values: Tensor) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return gradient_checkpoint(module, *values, use_reentrant=False)
        return module(*values)

    def _check_inputs(
        self, fine: Tensor, coarse_k: Tensor, support120: Tensor, context: Tensor
    ) -> None:
        for value, name in (
            (fine, "fine"),
            (coarse_k, "coarse_k"),
            (support120, "support120"),
            (context, "context"),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if fine.ndim != 4 or fine.shape[1] != self.fine_channels:
            raise ValueError(
                f"fine must have shape [B,{self.fine_channels},H,W]"
            )
        if coarse_k.ndim != 4 or coarse_k.shape[1] != 1:
            raise ValueError("coarse_k must have shape [B,1,H/4,W/4]")
        if support120.shape != fine[:, :1].shape:
            raise ValueError("support120 must have shape [B,1,H,W]")
        if context.ndim != 2 or context.shape != (fine.shape[0], self.context_dim):
            raise ValueError(f"context must have shape [B,{self.context_dim}]")
        if coarse_k.shape[0] != fine.shape[0]:
            raise ValueError("fine and coarse_k batch sizes differ")
        if fine.shape[-2:] != (
            _SCALE * coarse_k.shape[-2],
            _SCALE * coarse_k.shape[-1],
        ):
            raise ValueError("fine/coarse geometry must be an exact 4x relation")
        if not _is_floating(fine) or not _is_floating(coarse_k) or not _is_floating(context):
            raise TypeError("fine, coarse_k, and context must have floating dtypes")
        if len({fine.device, coarse_k.device, support120.device, context.device}) != 1:
            raise ValueError("all OCNIR inputs must be on one device")
        if _contains_true(~torch.isfinite(fine)):
            raise ValueError("fine must be finite")
        if _contains_true(~torch.isfinite(context)):
            raise ValueError("context must be finite")
        _binary_mask(support120, "support120")

    def forward(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support120: Tensor,
        context: Tensor,
    ) -> Tensor:
        self._check_inputs(fine, coarse_k, support120, context)
        support = support120.to(dtype=fine.dtype)

        base_k = fine[:, :1]
        # Fixed physical scaling protects the first raw-Kelvin channel from
        # dominating runner-standardized auxiliary channels.
        fine_encoded_input = torch.cat(
            ((base_k - 300.0) / 20.0, fine[:, 1:], support), dim=1
        )

        fine_scales: list[Tensor] = []
        value = self._run(self.fine_stem, fine_encoded_input)
        value = self._run(self.fine_blocks[0], value)
        fine_scales.append(value)
        for index, down in enumerate(self.fine_down, start=1):
            value = self._run(down, value)
            value = self._run(self.fine_blocks[index], value)
            fine_scales.append(value)

        valid = torch.isfinite(coarse_k)
        coarse_safe = torch.where(valid, coarse_k, torch.full_like(coarse_k, 300.0))
        coarse_height, coarse_width = coarse_k.shape[-2:]
        support_fraction = support.reshape(
            fine.shape[0], 1, coarse_height, _SCALE, coarse_width, _SCALE
        ).mean(dim=(3, 5))
        coarse_inputs = torch.cat(
            ((coarse_safe.to(fine.dtype) - 300.0) / 20.0, support_fraction, valid.to(fine.dtype)),
            dim=1,
        )

        context_bias = self.context_encoder(context.to(dtype=fine.dtype))
        context_parts = context_bias.split(
            (self.width, 2 * self.width, 4 * self.width), dim=1
        )
        coarse_low = self._run(self.coarse_stem, coarse_inputs)
        coarse_low = coarse_low + context_parts[2][..., None, None]
        coarse_low = self._run(self.coarse_blocks[0], coarse_low)
        coarse_mid = F.interpolate(coarse_low, size=fine_scales[1].shape[-2:], mode="bilinear", align_corners=False)
        coarse_mid = self._run(self.coarse_up[0], coarse_mid)
        coarse_mid = coarse_mid + context_parts[1][..., None, None]
        coarse_mid = self._run(self.coarse_blocks[1], coarse_mid)
        coarse_high = F.interpolate(coarse_mid, size=fine_scales[0].shape[-2:], mode="bilinear", align_corners=False)
        coarse_high = self._run(self.coarse_up[1], coarse_high)
        coarse_high = coarse_high + context_parts[0][..., None, None]
        coarse_high = self._run(self.coarse_blocks[2], coarse_high)
        coarse_scales = (coarse_high, coarse_mid, coarse_low)

        decoded = self._run(self.fusions[2], fine_scales[2], coarse_scales[2])
        decoded = self._run(self.decode_blocks[0], decoded)
        for decoder_index, scale_index in enumerate((1, 0)):
            decoded = F.interpolate(
                decoded,
                size=fine_scales[scale_index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = self._run(self.decode_up[decoder_index], decoded)
            decoded = self._run(
                self.decode_merge[decoder_index],
                torch.cat((decoded, fine_scales[scale_index]), dim=1),
            )
            decoded = self._run(
                self.fusions[scale_index], decoded, coarse_scales[scale_index]
            )
            decoded = self._run(self.decode_blocks[2 - scale_index], decoded)

        raw_k = base_k + self.head(decoded)
        return support_project(raw_k, coarse_k, support120)
