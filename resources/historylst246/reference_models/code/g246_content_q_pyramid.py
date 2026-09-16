"""Content-only, support-aware Q correction pyramid for G246.

The module in this file deliberately has no metadata input: it cannot consume
latitude, longitude, a region/city identifier, a scene code, or the registered
``Context19`` vector.  Its only conditioning is the observable fine-grid
predictor, spatial features already produced by the backbone, the delivered Q
field, and physical support/coarse-availability masks.

``ContentOnlyQPyramid`` is an identity-initialized *additive branch*.  It
returns a proposal (not ``delivered_q0 + proposal``), and all three frequency
heads start at exact zero.  This statement applies only to the branch: the
integrated no-geolocation candidate intentionally masks four Context19
coordinates before the inherited core and therefore is a no-geo rebase, not a
numerically identical r2k migration.  The proposal is projected to zero
supported mean in every 4x4 parent, including parents without a coarse
observation; parents with fewer than two supported children return exact zero.
"""

from __future__ import annotations

from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = ["ContentOnlyQPyramid"]

_SCALE = 4
_MAX_Q_K = 4.0
_RMS_EPSILON = 1.0e-6


def _groups(channels: int) -> int:
    return gcd(channels, min(8, channels))


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels), channels)


def _has_true(value: Tensor) -> bool:
    """Reduce a validation predicate without retaining it in autograd."""

    return bool(torch.any(value).detach().cpu().item())


def _require_finite(value: Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if _has_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _binary(value: Tensor, name: str) -> Tensor:
    if value.dtype == torch.bool:
        return value
    if not value.is_floating_point() and value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} must be bool or a numeric binary tensor")
    if value.is_floating_point() and _has_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _has_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class _D4SymmetricDepthwiseConv(nn.Module):
    """Depthwise convolution with an exactly D4-symmetrized learned kernel."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("D4 kernel size must be a positive odd integer")
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(channels, 1, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    @staticmethod
    def _symmetrize(weight: Tensor) -> Tensor:
        reflected = torch.flip(weight, dims=(-1,))
        transforms = tuple(
            torch.rot90(weight, turns, dims=(-2, -1)) for turns in range(4)
        )
        transforms += tuple(
            torch.rot90(reflected, turns, dims=(-2, -1)) for turns in range(4)
        )
        return torch.stack(transforms, dim=0).mean(dim=0)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self._symmetrize(self.weight),
            self.bias,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )


class _D4LargeKernelResidual(nn.Module):
    """Orientation-free large-kernel residual block."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        expanded = 2 * channels
        self.depthwise = _D4SymmetricDepthwiseConv(channels, kernel_size)
        self.norm = _normalization(channels)
        self.expand = nn.Conv2d(channels, expanded, 1)
        self.contract = nn.Conv2d(expanded, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1.0e-2))
        self.norm.apply(_initialize)
        self.expand.apply(_initialize)
        self.contract.apply(_initialize)

    def forward(self, inputs: Tensor) -> Tensor:
        value = F.silu(self.norm(self.depthwise(inputs)))
        value = self.contract(F.silu(self.expand(value)))
        return inputs + self.scale * value


class _MaskedD4Stage(nn.Module):
    """Two D4 blocks that keep physically unsupported tokens exactly zero."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _D4LargeKernelResidual(channels, kernel_size) for _ in range(2)
        )

    def forward(self, inputs: Tensor, support: Tensor) -> Tensor:
        mask = support.to(dtype=inputs.dtype)
        value = inputs * mask
        for block in self.blocks:
            value = block(value) * mask
        return value


def _support_pool2(values: Tensor, support_mass: Tensor) -> tuple[Tensor, Tensor]:
    """2x support-mass-normalized pooling and propagated child mass.

    Propagating the count, rather than only an any-supported boolean, ensures
    the 40-grid token is the mean over original fine pixels.  Otherwise a
    sparse 80-grid child with one supported pixel would receive the same
    weight as a fully supported child with four pixels.
    """

    weights = support_mass.to(dtype=values.dtype)
    counts = F.avg_pool2d(weights, kernel_size=2, stride=2) * 4.0
    sums = F.avg_pool2d(values * weights, kernel_size=2, stride=2) * 4.0
    available = counts > 0
    pooled = torch.where(
        available,
        sums / counts.clamp_min(1.0),
        torch.zeros_like(sums),
    )
    return pooled, counts


def _parent_statistics(
    values: Tensor, support: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return supported 4x4 contrast, parent RMS, and support count."""

    batch, channels, height, width = values.shape
    parent_height, parent_width = height // _SCALE, width // _SCALE
    blocks = values.float().reshape(
        batch, channels, parent_height, _SCALE, parent_width, _SCALE
    )
    weights = support.float().reshape(
        batch, 1, parent_height, _SCALE, parent_width, _SCALE
    )
    counts = weights.sum(dim=(3, 5))
    means = (blocks * weights).sum(dim=(3, 5)) / counts.clamp_min(1.0)
    means = torch.where(counts > 0, means, torch.zeros_like(means))
    expanded_mean = means.repeat_interleave(_SCALE, dim=-2).repeat_interleave(
        _SCALE, dim=-1
    )
    contrast = (values.float() - expanded_mean) * support.float()
    contrast_blocks = contrast.reshape(
        batch, channels, parent_height, _SCALE, parent_width, _SCALE
    )
    variance = contrast_blocks.square().sum(dim=(3, 5)) / counts.clamp_min(1.0)
    rms = variance.clamp_min(_RMS_EPSILON**2).sqrt()
    rms = torch.where(counts >= 2, rms, torch.zeros_like(rms))
    return contrast, rms, counts


def _binomial_gaussian(size: int) -> Tensor:
    """Return a fixed separable binomial approximation to a Gaussian."""

    if size not in (5, 13):
        raise ValueError("the registered Gaussian sizes are G5 and G13")
    coefficients = [1]
    for _ in range(size - 1):
        coefficients = [1] + [
            coefficients[index - 1] + coefficients[index]
            for index in range(1, len(coefficients))
        ] + [1]
    vector = torch.tensor(coefficients, dtype=torch.float32)
    vector = vector / vector.sum()
    kernel = vector[:, None] * vector[None, :]
    return kernel[None, None]


def _masked_gaussian(
    values: Tensor,
    support: Tensor,
    kernel: Tensor,
) -> Tensor:
    """Apply a fixed Gaussian without treating unsupported holes as zero data."""

    mask = support.to(dtype=values.dtype)
    fixed = kernel.to(device=values.device, dtype=values.dtype)
    padding = fixed.shape[-1] // 2
    numerator = F.conv2d(values * mask, fixed, padding=padding)
    denominator = F.conv2d(mask, fixed, padding=padding)
    smoothed = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).eps),
        torch.zeros_like(numerator),
    )
    return smoothed * mask


def _project_parent_q(values: Tensor, support: Tensor) -> Tensor:
    """Project to supported parent-zero-mean Q, zeroing counts below two."""

    contrast, _, counts = _parent_statistics(values, support)
    usable = (counts >= 2).to(dtype=contrast.dtype)
    usable_fine = usable.repeat_interleave(_SCALE, dim=-2).repeat_interleave(
        _SCALE, dim=-1
    )
    return contrast * usable_fine * support.to(dtype=contrast.dtype)


class ContentOnlyQPyramid(nn.Module):
    """Observable-content Q pyramid with no geographic/identity interface.

    Geometry names 160/80/40 denote the full-scene registration.  The same
    operations apply to any fine lattice divisible by four (e.g. 96/48/24
    training patches).

    Parameters
    ----------
    fine_channels:
        Channel count of the directly observable fine predictor (FineC).
    fine_width:
        Channel count of the backbone fine-grid feature.
    parent_width:
        Channel count of the backbone 4x4-parent spatial feature.  This is a
        feature map, not a metadata/context vector.
    hidden_width:
        Shared pyramid width; the registered candidate uses 64.
    """

    schema_version = "g246-content-only-q-pyramid-v1"

    def __init__(
        self,
        fine_channels: int,
        fine_width: int,
        parent_width: int,
        *,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        for value, name in (
            (fine_channels, "fine_channels"),
            (fine_width, "fine_width"),
            (parent_width, "parent_width"),
            (hidden_width, "hidden_width"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.fine_channels = fine_channels
        self.fine_width = fine_width
        self.parent_width = parent_width
        self.hidden_width = hidden_width

        # q0, supported parent contrast, parent RMS, support, coarse validity.
        self.q_projection = nn.Conv2d(5, hidden_width, 1, bias=False)
        self.predictor_projection = nn.Conv2d(
            fine_channels, hidden_width, 1, bias=False
        )
        self.fine_projection = nn.Conv2d(
            fine_width, hidden_width, 1, bias=False
        )
        self.parent_projection = nn.Conv2d(
            parent_width, hidden_width, 1, bias=False
        )
        self.input_norm = _normalization(hidden_width)

        # Registered 160/80/40 large-kernel allocation: 7/9/7, two each.
        self.stage_40 = _MaskedD4Stage(hidden_width, kernel_size=7)
        self.stage_80 = _MaskedD4Stage(hidden_width, kernel_size=9)
        self.stage_160 = _MaskedD4Stage(hidden_width, kernel_size=7)

        self.signed_head_40 = nn.Conv2d(hidden_width, 1, 1)
        self.signed_head_80 = nn.Conv2d(hidden_width, 1, 1)
        self.signed_head_160 = nn.Conv2d(hidden_width, 1, 1)

        self.register_buffer("gaussian_g5", _binomial_gaussian(5), persistent=True)
        self.register_buffer(
            "gaussian_g13", _binomial_gaussian(13), persistent=True
        )

        for module in (
            self.q_projection,
            self.predictor_projection,
            self.fine_projection,
            self.parent_projection,
            self.input_norm,
        ):
            module.apply(_initialize)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Set all three signed frequency heads to exact zero."""

        for head in (
            self.signed_head_40,
            self.signed_head_80,
            self.signed_head_160,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _validate(
        self,
        delivered_q0: Tensor,
        predictor: Tensor,
        support: Tensor,
        coarse_valid: Tensor,
        fine_features: Tensor,
        parent_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if delivered_q0.ndim != 4 or delivered_q0.shape[1] != 1:
            raise ValueError("delivered_q0 must have shape [B,1,H,W]")
        batch, _, height, width = delivered_q0.shape
        if batch < 1 or height < _SCALE or width < _SCALE:
            raise ValueError("delivered_q0 batch and spatial dimensions are too small")
        if height % _SCALE or width % _SCALE:
            raise ValueError("fine geometry must be divisible by four")
        expected_fine = (batch, height, width)
        if predictor.ndim != 4 or predictor.shape != (
            batch, self.fine_channels, height, width
        ):
            raise ValueError(
                f"predictor must have shape [B,{self.fine_channels},H,W]"
            )
        if fine_features.ndim != 4 or fine_features.shape != (
            batch, self.fine_width, height, width
        ):
            raise ValueError(
                f"fine_features must have shape [B,{self.fine_width},H,W]"
            )
        if support.shape != delivered_q0.shape:
            raise ValueError("support must match delivered_q0 geometry")
        parent_geometry = (batch, 1, height // _SCALE, width // _SCALE)
        if coarse_valid.shape != parent_geometry:
            raise ValueError("coarse_valid must have shape [B,1,H/4,W/4]")
        if parent_context.ndim != 4 or parent_context.shape != (
            batch, self.parent_width, height // _SCALE, width // _SCALE
        ):
            raise ValueError(
                f"parent_context must have shape [B,{self.parent_width},H/4,W/4]"
            )
        del expected_fine

        floating = {
            "delivered_q0": delivered_q0,
            "predictor": predictor,
            "fine_features": fine_features,
            "parent_context": parent_context,
        }
        for name, value in floating.items():
            _require_finite(value, name)
            if value.device != delivered_q0.device:
                raise ValueError("all pyramid inputs must be on one device")
        if support.device != delivered_q0.device \
                or coarse_valid.device != delivered_q0.device:
            raise ValueError("all pyramid inputs must be on one device")
        return _binary(support, "support"), _binary(
            coarse_valid, "coarse_valid"
        )

    def forward(
        self,
        delivered_q0: Tensor,
        predictor: Tensor,
        support: Tensor,
        coarse_valid: Tensor,
        fine_features: Tensor,
        parent_context: Tensor,
    ) -> Tensor:
        support_bool, coarse_bool = self._validate(
            delivered_q0,
            predictor,
            support,
            coarse_valid,
            fine_features,
            parent_context,
        )
        support_float = support_bool.to(dtype=delivered_q0.dtype)
        contrast, parent_rms, _ = _parent_statistics(
            delivered_q0, support_float
        )
        rms_up = parent_rms.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)
        coarse_up = coarse_bool.repeat_interleave(
            _SCALE, dim=-2
        ).repeat_interleave(_SCALE, dim=-1)
        q_inputs = torch.cat(
            (
                delivered_q0 / _MAX_Q_K,
                contrast.to(dtype=delivered_q0.dtype) / _MAX_Q_K,
                rms_up.to(dtype=delivered_q0.dtype) / _MAX_Q_K,
                support_float,
                coarse_up.to(dtype=delivered_q0.dtype),
            ),
            dim=1,
        )

        # Only the physical temperature channel has a registered Kelvin
        # centring.  Remaining FineC channels retain their supplied contract.
        centred_predictor = torch.cat(
            ((predictor[:, :1] - 300.0) / 20.0, predictor[:, 1:]), dim=1
        )
        # Predictor tensors remain fp32 under CUDA autocast while convolution
        # features may be fp16/bf16.  Cast at the module boundary rather than
        # rejecting the trainer's normal mixed-precision execution contract.
        feature_dtype = fine_features.dtype
        lateral_160 = (
            self.q_projection(q_inputs.to(dtype=feature_dtype))
            + self.predictor_projection(centred_predictor.to(dtype=feature_dtype))
            + self.fine_projection(fine_features)
        )
        lateral_160 = F.silu(self.input_norm(lateral_160)) * support_float
        lateral_80, support_mass_80 = _support_pool2(
            lateral_160, support_float
        )
        support_80 = support_mass_80 > 0
        lateral_40, support_mass_40 = _support_pool2(
            lateral_80, support_mass_80
        )
        support_40 = support_mass_40 > 0
        lateral_40 = (
            lateral_40 + self.parent_projection(
                parent_context.to(dtype=lateral_40.dtype)
            )
        ) * support_40.to(dtype=lateral_40.dtype)

        feature_40 = self.stage_40(lateral_40, support_40)
        feature_80 = self.stage_80(
            lateral_80 + F.interpolate(feature_40, scale_factor=2, mode="nearest"),
            support_80,
        )
        feature_160 = self.stage_160(
            lateral_160 + F.interpolate(feature_80, scale_factor=2, mode="nearest"),
            support_bool,
        )

        signed_40 = F.interpolate(
            _MAX_Q_K * torch.tanh(self.signed_head_40(feature_40)),
            scale_factor=4,
            mode="nearest",
        )
        signed_80 = F.interpolate(
            _MAX_Q_K * torch.tanh(self.signed_head_80(feature_80)),
            scale_factor=2,
            mode="nearest",
        )
        signed_160 = _MAX_Q_K * torch.tanh(
            self.signed_head_160(feature_160)
        )

        # Fixed, mask-normalized frequency partition: G13,
        # (G5 - G13), and (I - G5).
        low_band = _masked_gaussian(
            signed_40, support_bool, self.gaussian_g13
        )
        mid_band = _masked_gaussian(
            signed_80, support_bool, self.gaussian_g5
        ) - _masked_gaussian(signed_80, support_bool, self.gaussian_g13)
        high_band = signed_160 * support_float - _masked_gaussian(
            signed_160, support_bool, self.gaussian_g5
        )
        proposal = _project_parent_q(
            low_band + mid_band + high_band, support_bool
        )
        proposal = proposal.to(dtype=delivered_q0.dtype) * support_float
        if _has_true(~torch.isfinite(proposal)):
            raise FloatingPointError("content Q pyramid produced non-finite output")
        return proposal
