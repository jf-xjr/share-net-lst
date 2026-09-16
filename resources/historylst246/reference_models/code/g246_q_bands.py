"""Support-aware nested orthogonal bands for G246 fine-grid Q fields.

The physical observation is a support-weighted mean on each 4x4 fine-grid
parent.  This module decomposes any support-restricted fine field into three
mutually orthogonal pieces:

``P40``
    the observed-parent constant component (zero on invalid parents),
``QM = P80 - P40``
    variation between active 2x2 children inside a 4x4 parent, and
``QH = I - P80``
    within-child fine variation.

Consequently ``QM + QH = I - P40``.  On a valid coarse parent this is exactly
the nullspace of the delivered observation; on an invalid parent ``P40=0`` so
the two learned bands retain the complete support-restricted field.  All
operators use the same support-weighted Euclidean inner product.  Empty 2x2
children and unsupported pixels are represented by zero rather than by a
fabricated observation.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


__all__ = [
    "OrthogonalQBands",
    "support_block_mean",
    "support_block_projection",
    "project_p40",
    "project_p80",
    "lift_p80_coefficients",
    "orthogonal_q_bands",
]


class OrthogonalQBands(NamedTuple):
    """Nested projections and the two learnable Q bands."""

    p40: Tensor
    p80: Tensor
    q_middle: Tensor
    q_high: Tensor
    q_total: Tensor


def _contains_true(value: Tensor) -> bool:
    return bool(torch.any(value).detach().cpu().item())


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
        raise TypeError(f"{name} must be boolean or numeric binary")
    if value.is_floating_point() and _contains_true(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite and binary")
    if _contains_true((value != 0) & (value != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return value.to(dtype=torch.bool)


def _validate_field_support(
    field: Tensor,
    support: Tensor,
    block_size: int,
) -> Tensor:
    if not isinstance(field, Tensor) or not isinstance(support, Tensor):
        raise TypeError("field and support must be torch.Tensor instances")
    if field.ndim != 4 or support.ndim != 4:
        raise ValueError("field and support must have NCHW rank 4")
    if not field.is_floating_point():
        raise TypeError("field must have a floating dtype")
    if support.shape != (field.shape[0], 1, *field.shape[-2:]):
        raise ValueError("support must have shape [B,1,H,W] matching field")
    if field.device != support.device:
        raise ValueError("field and support must be on one device")
    if isinstance(block_size, bool) or not isinstance(block_size, int) \
            or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if field.shape[-2] % block_size or field.shape[-1] % block_size:
        raise ValueError("field geometry must be divisible by block_size")
    return _binary(support, "support")


def support_block_mean(
    field: Tensor,
    support: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """Return supported block means and a boolean non-empty-block mask.

    Means have shape ``[B,C,H/block,W/block]``.  An empty block has mean zero
    and ``active=False``.  Reductions are promoted to float32 for half and
    bfloat16 inputs so this operator remains numerically stable under AMP.
    """

    support_bool = _validate_field_support(field, support, block_size)
    batch, channels, height, width = field.shape
    blocks_y, blocks_x = height // block_size, width // block_size
    work_dtype = (
        torch.float32
        if field.dtype in (torch.float16, torch.bfloat16)
        else field.dtype
    )
    values = field.to(dtype=work_dtype).reshape(
        batch, channels, blocks_y, block_size, blocks_x, block_size
    )
    weights = support_bool.to(dtype=work_dtype).reshape(
        batch, 1, blocks_y, block_size, blocks_x, block_size
    )
    counts = weights.sum(dim=(3, 5))
    sums = (values * weights).sum(dim=(3, 5))
    means = sums / counts.clamp_min(1.0)
    active = counts > 0
    means = torch.where(active, means, torch.zeros_like(means))
    return means.to(dtype=field.dtype), active


def support_block_projection(
    field: Tensor,
    support: Tensor,
    block_size: int,
    active_blocks: Tensor | None = None,
) -> Tensor:
    """Orthogonally project onto support-constant active blocks.

    ``active_blocks`` may remove whole blocks from the projection subspace.
    It has shape ``[B,1,H/block,W/block]``.  Removed/empty blocks and
    unsupported coordinates are zero in the returned restricted-space field.
    """

    means, nonempty = support_block_mean(field, support, block_size)
    support_bool = _binary(support, "support")
    if active_blocks is None:
        active = nonempty
    else:
        if not isinstance(active_blocks, Tensor):
            raise TypeError("active_blocks must be a torch.Tensor or None")
        expected = (
            field.shape[0],
            1,
            field.shape[-2] // block_size,
            field.shape[-1] // block_size,
        )
        if active_blocks.shape != expected:
            raise ValueError(
                "active_blocks must have shape [B,1,H/block,W/block]"
            )
        if active_blocks.device != field.device:
            raise ValueError("active_blocks must be on the field device")
        requested = _binary(active_blocks, "active_blocks")
        if _contains_true(requested & ~nonempty):
            raise ValueError("an active block has zero fine support")
        active = requested & nonempty
    restricted_means = means * active.to(dtype=means.dtype)
    expanded = restricted_means.repeat_interleave(
        block_size, dim=-2
    ).repeat_interleave(block_size, dim=-1)
    return expanded * support_bool.to(dtype=field.dtype)


def project_p40(field: Tensor, support: Tensor, coarse_valid: Tensor) -> Tensor:
    """Project onto valid, support-constant 4x4 observation parents."""

    return support_block_projection(
        field, support, block_size=4, active_blocks=coarse_valid
    )


def project_p80(field: Tensor, support: Tensor) -> Tensor:
    """Project onto support-constant 2x2 child cells."""

    return support_block_projection(field, support, block_size=2)


def lift_p80_coefficients(coefficients: Tensor, support: Tensor) -> Tensor:
    """Lift one coefficient per 2x2 cell to its supported fine pixels.

    Coefficients belonging to empty cells disappear naturally.  This is the
    adjoint-style synthesis map used by IPMR-Q before applying ``QM``.
    """

    if not isinstance(coefficients, Tensor) or not isinstance(support, Tensor):
        raise TypeError("coefficients and support must be torch.Tensor instances")
    if coefficients.ndim != 4 or support.ndim != 4:
        raise ValueError("coefficients and support must have NCHW rank 4")
    if not coefficients.is_floating_point():
        raise TypeError("coefficients must have a floating dtype")
    if support.shape != (
        coefficients.shape[0], 1,
        2 * coefficients.shape[-2], 2 * coefficients.shape[-1],
    ):
        raise ValueError("support must be the exact 2x fine geometry")
    if coefficients.device != support.device:
        raise ValueError("coefficients and support must be on one device")
    support_bool = _binary(support, "support")
    lifted = coefficients.repeat_interleave(2, dim=-2).repeat_interleave(
        2, dim=-1
    )
    return lifted * support_bool.to(dtype=coefficients.dtype)


def orthogonal_q_bands(
    field: Tensor,
    support: Tensor,
    coarse_valid: Tensor,
) -> OrthogonalQBands:
    """Decompose ``field`` into the valid-parent mean and two Q bands."""

    support_bool = _validate_field_support(field, support, block_size=4)
    expected_valid = (
        field.shape[0], 1, field.shape[-2] // 4, field.shape[-1] // 4
    )
    if not isinstance(coarse_valid, Tensor):
        raise TypeError("coarse_valid must be a torch.Tensor")
    if coarse_valid.shape != expected_valid:
        raise ValueError("coarse_valid must have shape [B,1,H/4,W/4]")
    if coarse_valid.device != field.device:
        raise ValueError("coarse_valid must be on the field device")
    valid = _binary(coarse_valid, "coarse_valid")
    restricted = field * support_bool.to(dtype=field.dtype)
    p40 = project_p40(restricted, support_bool, valid)
    p80 = project_p80(restricted, support_bool)
    q_middle = p80 - p40
    q_high = restricted - p80
    q_total = q_middle + q_high
    return OrthogonalQBands(p40, p80, q_middle, q_high, q_total)
