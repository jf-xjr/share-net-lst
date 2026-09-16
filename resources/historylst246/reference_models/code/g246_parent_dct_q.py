"""Content-only Parent-DCT15 decoder for the G246 Q field.

This is the deliberately narrow fallback for the no-geolocation G246 route.
Its public interface contains only the registered ``Fine52`` tensor, the
delivered fine-grid Q field, physical fine support, and coarse availability.
It cannot consume Context19, latitude/longitude, region/city labels, a scene
identifier, or a learned scene router.

Fine inputs are compressed to 20 channels with a pointwise convolution and
then packed, without losing 4x4 phase, onto the parent lattice.  A small
40 -> 20 -> 10 parent-grid U-Net predicts the 15 non-DC coefficients of a
fixed orthonormal 4x4 DCT basis.  Fixed synthesis maps those coefficients back
to fine pixels.  Because masking a full-support non-DC basis can reintroduce a
DC component, the synthesized correction is itself projected by the real
support in *every* parent, independent of coarse availability.  It is thus a
shape-only correction and is zero for fewer than two supported children.
After it is added to ``delivered_q0``, only parents with a valid coarse
observation receive the overall Q closure.  Parents without one preserve the
inherited support-weighted mean.  Unsupported pixels always return zero.

The coefficient head is initialized to exact zero, so migration initially
adds no learned DCT correction.  The final Q projection remains active because
it is a scientific invariant, rather than a learned branch.

D4 equivariance is **not guaranteed**.  Pixel-unshuffle phases and ordinary
parent-grid convolutions have orientation-specific channels/weights, while a
rotation or reflection mixes signed DCT coefficients.  D4 augmentation may be
used by a future trainer, but this module intentionally makes no architectural
equivariance claim.
"""

from __future__ import annotations

import math
from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = ["ParentDCT15QDecoder"]

_SCALE = 4
_DCT_COMPONENTS = _SCALE * _SCALE - 1
_FINE_CHANNELS = 52
_MAX_COEFFICIENT_K = 4.0


def _has_true(value: Tensor) -> bool:
    """Reduce a validation predicate without retaining it in autograd."""

    return bool(torch.any(value).detach().cpu().item())


def _groups(channels: int) -> int:
    return gcd(channels, min(8, channels))


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels), channels)


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


def _dct15_basis() -> Tensor:
    """Return the 15 non-DC rows of the orthonormal 4x4 DCT-II basis."""

    coordinates = torch.arange(_SCALE, dtype=torch.float64)
    frequencies = torch.arange(_SCALE, dtype=torch.float64)[:, None]
    alpha = torch.full((_SCALE, 1), math.sqrt(2.0 / _SCALE), dtype=torch.float64)
    alpha[0] = math.sqrt(1.0 / _SCALE)
    one_dimensional = alpha * torch.cos(
        math.pi * (2.0 * coordinates[None, :] + 1.0) * frequencies
        / (2.0 * _SCALE)
    )
    two_dimensional = torch.einsum(
        "ui,vj->uvij", one_dimensional, one_dimensional
    ).reshape(_SCALE * _SCALE, _SCALE, _SCALE)
    return two_dimensional[1:].to(dtype=torch.float32)


def _project_parent_q(
    values: Tensor,
    support: Tensor,
    coarse_valid: Tensor,
) -> Tensor:
    """Apply Q closure only where a direct coarse observation exists.

    Coarse-valid parents are support-centred; a valid singleton has no
    identifiable contrast and therefore returns zero.  A valid parent with no
    support is an inconsistent scientific observation and fails closed.
    Coarse-invalid parents have no mean constraint and retain ``values`` on
    supported pixels.  This distinction is required for an exact zero-head
    migration of the delivered field outside the observable coarse domain.
    """

    batch, channels, height, width = values.shape
    parent_height, parent_width = height // _SCALE, width // _SCALE
    work_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    blocks = values.to(dtype=work_dtype).reshape(
        batch, channels, parent_height, _SCALE, parent_width, _SCALE
    )
    weights = support.to(dtype=work_dtype).reshape(
        batch, 1, parent_height, _SCALE, parent_width, _SCALE
    )
    counts = weights.sum(dim=(3, 5))
    valid = coarse_valid.reshape(
        batch, 1, parent_height, 1, parent_width, 1
    )
    if _has_true(coarse_valid & (counts == 0)):
        raise ValueError("a coarse-valid parent has zero fine support")
    means = (blocks * weights).sum(dim=(3, 5)) / counts.clamp_min(1.0)
    centred = (blocks - means[:, :, :, None, :, None]) * weights
    usable = (counts >= 2).to(dtype=work_dtype)
    centred = centred * usable[:, :, :, None, :, None]
    supported = blocks * weights
    result = torch.where(valid, centred, supported)
    return result.reshape(batch, channels, height, width).to(dtype=values.dtype)


def _project_correction_q(values: Tensor, support: Tensor) -> Tensor:
    """Project a learned correction to shape-only ``Pi_s`` on every parent.

    A fixed non-DC DCT atom is zero-mean over all 16 phases, but not generally
    over an arbitrary supported subset.  This projection removes that support
    DC component without consulting ``coarse_valid``.  Count-zero and
    count-one parents cannot carry an identifiable shape and return zero.
    """

    batch, channels, height, width = values.shape
    parent_height, parent_width = height // _SCALE, width // _SCALE
    work_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    blocks = values.to(dtype=work_dtype).reshape(
        batch, channels, parent_height, _SCALE, parent_width, _SCALE
    )
    weights = support.to(dtype=work_dtype).reshape(
        batch, 1, parent_height, _SCALE, parent_width, _SCALE
    )
    counts = weights.sum(dim=(3, 5))
    means = (blocks * weights).sum(dim=(3, 5)) / counts.clamp_min(1.0)
    centred = (blocks - means[:, :, :, None, :, None]) * weights
    usable = (counts >= 2).to(dtype=work_dtype)
    centred = centred * usable[:, :, :, None, :, None]
    return centred.reshape(batch, channels, height, width).to(dtype=values.dtype)


class _MaskedResidualStage(nn.Module):
    """One or more compact residual blocks on a supported parent lattice."""

    def __init__(self, channels: int, *, blocks: int = 1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "conv1": nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    "norm1": _norm(channels),
                    "conv2": nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    "norm2": _norm(channels),
                }
            )
            for _ in range(blocks)
        )
        self.apply(_initialize)

    def forward(self, inputs: Tensor, available: Tensor) -> Tensor:
        mask = available.to(dtype=inputs.dtype)
        value = inputs * mask
        for block in self.blocks:
            residual = block["conv1"](value)
            residual = F.silu(block["norm1"](residual))
            residual = block["norm2"](block["conv2"](residual))
            value = F.silu(value + residual) * mask
        return value


class ParentDCT15QDecoder(nn.Module):
    """Predict parent-local non-DC DCT coefficients from observable content.

    The registered geometry is 160 fine pixels -> 40 parents -> 20 -> 10.
    Patch geometries use the same two exact parent-grid halvings, so the fine
    height and width must each be divisible by 16.

    Parameters are intentionally below 0.8M at the registered widths.  The
    input compression width is fixed at 20, within the pre-registered 16--24
    range; changing it would define a different candidate.
    """

    schema_version = "g246-parent-dct15-q-v1"
    fine_channels = _FINE_CHANNELS
    compression_width = 20
    parent_widths = (48, 72, 96)
    d4_equivariant = False
    d4_status = (
        "not guaranteed: phase packing, ordinary convolutions, and signed "
        "DCT coefficients are orientation-specific"
    )
    correction_role = "shape-only support-projected Pi_s"

    def __init__(self) -> None:
        super().__init__()
        width_40, width_20, width_10 = self.parent_widths

        # Fine52, delivered q0, physical support -> registered 20 channels.
        self.fine_compress = nn.Sequential(
            nn.Conv2d(_FINE_CHANNELS + 2, self.compression_width, 1, bias=False),
            _norm(self.compression_width),
            nn.SiLU(inplace=False),
        )
        # Preserve all 16 fine phases.  Explicit phase support, support
        # fraction, and coarse availability prevent missingness from being
        # encoded as an ordinary physical zero.
        packed_channels = self.compression_width * _SCALE**2
        parent_inputs = packed_channels + _SCALE**2 + 2
        self.parent_pack = nn.Sequential(
            nn.Conv2d(parent_inputs, width_40, 1, bias=False),
            _norm(width_40),
            nn.SiLU(inplace=False),
        )

        self.parent_stage_40 = _MaskedResidualStage(width_40)
        self.down_40_to_20 = nn.Sequential(
            nn.Conv2d(width_40, width_20, 3, stride=2, padding=1, bias=False),
            _norm(width_20),
            nn.SiLU(inplace=False),
        )
        self.parent_stage_20 = _MaskedResidualStage(width_20)
        self.down_20_to_10 = nn.Sequential(
            nn.Conv2d(width_20, width_10, 3, stride=2, padding=1, bias=False),
            _norm(width_10),
            nn.SiLU(inplace=False),
        )
        self.parent_stage_10 = _MaskedResidualStage(width_10)

        self.merge_10_to_20 = nn.Sequential(
            nn.Conv2d(width_10 + width_20, width_20, 1, bias=False),
            _norm(width_20),
            nn.SiLU(inplace=False),
        )
        self.decode_stage_20 = _MaskedResidualStage(width_20)
        self.merge_20_to_40 = nn.Sequential(
            nn.Conv2d(width_20 + width_40, width_40, 1, bias=False),
            _norm(width_40),
            nn.SiLU(inplace=False),
        )
        self.decode_stage_40 = _MaskedResidualStage(width_40)
        self.coefficient_head = nn.Conv2d(width_40, _DCT_COMPONENTS, 1)

        self.register_buffer("dct15_basis", _dct15_basis(), persistent=True)
        for module in (
            self.fine_compress,
            self.parent_pack,
            self.down_40_to_20,
            self.down_20_to_10,
            self.merge_10_to_20,
            self.merge_20_to_40,
        ):
            module.apply(_initialize)
        self.reset_identity_initialization()

    def reset_identity_initialization(self) -> None:
        """Set all 15 learned DCT coefficient outputs to exact zero."""

        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    def _validate(
        self,
        fine52: Tensor,
        delivered_q0: Tensor,
        support: Tensor,
        coarse_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for value, name in (
            (fine52, "fine52"),
            (delivered_q0, "delivered_q0"),
            (support, "support"),
            (coarse_valid, "coarse_valid"),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if delivered_q0.ndim != 4 or delivered_q0.shape[1] != 1:
            raise ValueError("delivered_q0 must have shape [B,1,H,W]")
        batch, _, height, width = delivered_q0.shape
        if batch < 1 or height < 16 or width < 16:
            raise ValueError("fine batch must be nonempty and geometry at least 16x16")
        if height % 16 or width % 16:
            raise ValueError("fine height and width must be divisible by 16")
        if fine52.shape != (batch, _FINE_CHANNELS, height, width):
            raise ValueError("fine52 must have exact shape [B,52,H,W]")
        if support.shape != delivered_q0.shape:
            raise ValueError("support must exactly match delivered_q0")
        expected_parent = (batch, 1, height // _SCALE, width // _SCALE)
        if coarse_valid.shape != expected_parent:
            raise ValueError("coarse_valid must have shape [B,1,H/4,W/4]")
        if not fine52.is_floating_point() or not delivered_q0.is_floating_point():
            raise TypeError("fine52 and delivered_q0 must be floating-point tensors")
        if _has_true(~torch.isfinite(fine52)):
            raise ValueError("fine52 must contain only finite values")
        if _has_true(~torch.isfinite(delivered_q0)):
            raise ValueError("delivered_q0 must contain only finite values")
        devices = {
            fine52.device,
            delivered_q0.device,
            support.device,
            coarse_valid.device,
        }
        if len(devices) != 1:
            raise ValueError("all Parent-DCT15 inputs must be on one device")
        return _binary(support, "support"), _binary(
            coarse_valid, "coarse_valid"
        )

    @staticmethod
    def _coarser_mask(mask: Tensor) -> Tensor:
        return F.max_pool2d(mask.to(dtype=torch.float32), 2, stride=2) > 0

    def _synthesize(self, coefficients: Tensor) -> Tensor:
        basis = self.dct15_basis.to(
            device=coefficients.device, dtype=coefficients.dtype
        )
        # [B,15,Hp,Wp] x [15,4,4] -> [B,Hp,4,Wp,4].
        blocks = torch.einsum("bkpq,kij->bpiqj", coefficients, basis)
        return blocks.unsqueeze(1).reshape(
            coefficients.shape[0],
            1,
            coefficients.shape[-2] * _SCALE,
            coefficients.shape[-1] * _SCALE,
        )

    def forward(
        self,
        fine52: Tensor,
        delivered_q0: Tensor,
        support: Tensor,
        coarse_valid: Tensor,
    ) -> Tensor:
        support_bool, coarse_bool = self._validate(
            fine52, delivered_q0, support, coarse_valid
        )
        support_float = support_bool.to(dtype=delivered_q0.dtype)

        # Only channel zero is Kelvin; Fine52 sidecars already normalize the
        # other 51 channels under their registered contracts.
        centred_fine = torch.cat(
            ((fine52[:, :1] - 300.0) / 20.0, fine52[:, 1:]), dim=1
        )
        fine_inputs = torch.cat(
            (
                centred_fine * support_float,
                (delivered_q0 / _MAX_COEFFICIENT_K) * support_float,
                support_float,
            ),
            dim=1,
        )
        compressed = self.fine_compress(fine_inputs) * support_float
        packed = F.pixel_unshuffle(compressed, _SCALE)
        phase_support = F.pixel_unshuffle(support_float, _SCALE)
        support_fraction = phase_support.mean(dim=1, keepdim=True)
        parent_inputs = torch.cat(
            (
                packed,
                phase_support.to(dtype=packed.dtype),
                support_fraction.to(dtype=packed.dtype),
                coarse_bool.to(dtype=packed.dtype),
            ),
            dim=1,
        )

        mask_40 = support_fraction > 0
        mask_20 = self._coarser_mask(mask_40)
        mask_10 = self._coarser_mask(mask_20)
        features_40 = self.parent_pack(parent_inputs) * mask_40.to(
            dtype=packed.dtype
        )
        features_40 = self.parent_stage_40(features_40, mask_40)
        features_20 = self.down_40_to_20(features_40) * mask_20.to(
            dtype=features_40.dtype
        )
        features_20 = self.parent_stage_20(features_20, mask_20)
        features_10 = self.down_20_to_10(features_20) * mask_10.to(
            dtype=features_20.dtype
        )
        features_10 = self.parent_stage_10(features_10, mask_10)

        up_20 = F.interpolate(
            features_10, size=features_20.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        decoded_20 = self.merge_10_to_20(
            torch.cat((up_20, features_20), dim=1)
        ) * mask_20.to(dtype=features_20.dtype)
        decoded_20 = self.decode_stage_20(decoded_20, mask_20)
        up_40 = F.interpolate(
            decoded_20, size=features_40.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        decoded_40 = self.merge_20_to_40(
            torch.cat((up_40, features_40), dim=1)
        ) * mask_40.to(dtype=features_40.dtype)
        decoded_40 = self.decode_stage_40(decoded_40, mask_40)

        coefficients = _MAX_COEFFICIENT_K * torch.tanh(
            self.coefficient_head(decoded_40)
        )
        counts = phase_support.sum(dim=1, keepdim=True)
        coefficients = coefficients * (counts >= 2).to(dtype=coefficients.dtype)
        correction = self._synthesize(coefficients).to(dtype=delivered_q0.dtype)
        # Non-DC in the complete 4x4 DCT basis is not sufficient under partial
        # support.  Pi_s makes the branch shape-only before it can interact
        # with the inherited q0 mean or the coarse-valid closure contract.
        correction = _project_correction_q(correction, support_bool)
        result = _project_parent_q(
            delivered_q0 + correction, support_bool, coarse_bool
        )
        if _has_true(~torch.isfinite(result)):
            raise FloatingPointError("Parent-DCT15 produced non-finite output")
        return result
