"""Registered D4 transforms for G246 full-scene training and eight-view TTA.

Codes 0--7 match ``g246_r2_data``: counterclockwise ``rot90(code & 3)``,
then a horizontal flip when ``code & 4``.  Context15 components 6 and 7
are solar east/north and transform with the raster.  Metadata is untouched.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


SPATIAL_KEYS = ("fine", "coarse", "support", "target", "formal", "valid", "emissivity", "history")
SOLAR_EAST_INDEX = 6
SOLAR_NORTH_INDEX = 7


def _code(code: int) -> int:
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 8:
        raise ValueError("D4 code must be an integer in [0,7]")
    return code


def _field(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim not in (4, 5):
        raise ValueError(f"{name} must be a BCHW or BTCHW torch tensor")


def transform_field(value: Tensor, code: int) -> Tensor:
    """Transform raster dimensions only; preserve channels, dtype and device."""
    code = _code(code)
    _field(value, "field")
    result = torch.rot90(value, code & 3, dims=(-2, -1))
    if code & 4:
        result = torch.flip(result, dims=(-1,))
    return result.contiguous()


def inverse_field(prediction: Tensor, code: int) -> Tensor:
    """Undo one registered transform for a scalar field or ordinary raster."""
    code = _code(code)
    _field(prediction, "prediction")
    result = torch.flip(prediction, dims=(-1,)) if code & 4 else prediction
    return torch.rot90(result, -(code & 3), dims=(-2, -1)).contiguous()


def inverse_code(code: int) -> int:
    """Every reflected D4 element is self-inverse; rotations reverse sign."""
    code = _code(code)
    return code if code & 4 else (-code) & 3


def transform_context(context: Tensor, code: int) -> Tensor:
    """Rotate/reflect physical Context15 solar azimuth; preserve other slots."""
    code = _code(code)
    if not isinstance(context, Tensor) or context.ndim not in (2, 3) or context.shape[-1] != 15:
        raise ValueError("context must be [B,15] or [B,T,15], with geographic columns removed")
    if not context.is_floating_point():
        raise TypeError("physical Context15 must have a floating dtype")
    result = context.clone()
    east, north = context[..., SOLAR_EAST_INDEX], context[..., SOLAR_NORTH_INDEX]
    rotation = code & 3
    if rotation == 0:
        transformed_east, transformed_north = east, north
    elif rotation == 1:
        transformed_east, transformed_north = -north, east
    elif rotation == 2:
        transformed_east, transformed_north = -east, -north
    else:
        transformed_east, transformed_north = north, -east
    if code & 4:
        transformed_east = -transformed_east
    result[..., SOLAR_EAST_INDEX] = transformed_east
    result[..., SOLAR_NORTH_INDEX] = transformed_north
    return result


def transform_batch(batch: dict[str, Any], code: int) -> dict[str, Any]:
    """Transform all registered predictors and supervision without mutation.

    One code applies to the complete batch and all dates.  Packed 30 m optical
    detail needs both spatial and within-cell phase permutations, delegated to
    its registered encoder.  Hourly solar directions use their registered
    physical-vector transform without changing the order of hours.
    Missing metadata keys and unrelated values retain their original objects.
    """
    code = _code(code)
    if not isinstance(batch, dict):
        raise TypeError("batch must be a dictionary")
    if "context" not in batch:
        raise ValueError("D4 augmentation requires Context15 to transform solar azimuth")
    result = dict(batch)
    for name in SPATIAL_KEYS:
        if name in batch:
            _field(batch[name], name)
            if name == "history" and (batch[name].ndim != 5 or batch[name].shape[1] not in (3, 6, 7, 9)
                                      or batch[name].shape[2] != 9):
                raise ValueError("history must contain three, six, seven or nine dates and nine scalar rasters [B,T,9,H,W]")
            result[name] = transform_field(batch[name], code)
    result["context"] = transform_context(batch["context"], code)
    if "detail" in batch:
        _field(batch["detail"], "detail")
        if batch["detail"].shape[-3] != 128:
            raise ValueError("detail must use the registered 128-channel optical phase encoding")
        try:
            from build_g246_8h_optical_detail import transform_detail_d4
        except ImportError as error:
            raise RuntimeError("optical detail D4 support is unavailable; augmentation cannot continue") from error
        result["detail"] = transform_detail_d4(batch["detail"], code)
    if "hourly" in batch:
        hourly = batch["hourly"]
        if not isinstance(hourly, Tensor) or hourly.ndim != 3 or hourly.shape[-2:] != (48, 10):
            raise ValueError("hourly must be a [B,48,10] torch tensor")
        if not hourly.is_floating_point():
            raise TypeError("hourly physical features must have a floating dtype")
        try:
            from g246_8h_hourly_network import transform_hourly_d4
        except ImportError as error:
            raise RuntimeError("hourly D4 support is unavailable; augmentation cannot continue") from error
        result["hourly"] = transform_hourly_d4(hourly, code)
    return result


__all__ = ["transform_batch", "transform_field", "transform_context", "inverse_field", "inverse_code"]
