#!/usr/bin/env python3
"""Immutable, transactional builder for the delivered-120 m thermal corpus."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import requests
import scipy
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy.ndimage import binary_dilation, zoom


STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
LANDSAT_COLLECTION = "landsat-c2-l2"
OPTICAL_KEYS = ("blue", "green", "red", "nir08", "swir16", "swir22")
DEFAULT_CONTRACT = Path("data/v2/build_contract.json")
DEFAULT_PROTOCOL = Path("research/protocol/confirmatory_v2.md")
DEFAULT_EXPERIMENT_PLAN = Path("research/protocol/experiment_plan_v2_1.json")
DEFAULT_RESERVE_CONFIG = Path("data/v2/reserve_cities.json")
DEFAULT_SEAL_POLICY = Path("data/v2/seal_policy.json")
REQUEST_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


class CityIneligibleError(RuntimeError):
    """A deterministic scientific/data-coverage failure that may consume a candidate."""


class RemoteAssetReadError(RuntimeError):
    """A required Landsat asset failed before its content could be interpreted."""

    def __init__(self, item_id: str, asset_key: str, href: str):
        super().__init__(f"Required Landsat asset read failed: {item_id}/{asset_key}")
        self.item_id = item_id
        self.asset_key = asset_key
        self.href = href


class TemporalEligibilityError(RuntimeError):
    """A scene would use land-cover information from a future year."""

QA_REASON_BITS = {
    "QA_PIXEL_RAW_REJECT": 0,
    "QA_PIXEL_DILATION_ONLY": 1,
    "QA_RADSAT_NONZERO": 2,
    "AEROSOL_FILL": 3,
    "AEROSOL_INVALID_STATE": 4,
    "AEROSOL_HIGH": 5,
    "ST_QA_ABOVE_3K": 6,
    "TARGET_ZERO": 7,
    "TARGET_OUTSIDE_240_380K": 8,
    "OPTICAL_ZERO": 9,
}

QA_SENSITIVITY = {
    "primary_mask": "qa_valid_buffer1_30",
    "cloud_buffer_masks": [
        "qa_valid_buffer0_30", "qa_valid_buffer1_30", "qa_valid_buffer2_30"
    ],
    "direct_aerosol_mask": "qa_valid_direct_aerosol30",
    "aerosol_fill_policy": "reject",
}

CANONICAL_GRID_CONTRACT = {
    "crs_rule": "WGS84 UTM zone floor((longitude+180)/6)+1, clamped to 1..60",
    "fine_grid_m": 30,
    "shape30": [640, 640],
    "axis_anchor": "integer multiples of 30 m in canonical UTM",
    "center_rounding": "half-up to nearest canonical grid corner",
    "resampling": "nearest",
    "asset_scope": "all Landsat numeric and QA assets before QA masking",
    "destination_fill": 0,
    "construction_order": "reproject, QA-mask, valid-support block aggregate",
}

REPLACEMENT_CONTRACT = {
    "cli_flag": "--replace-ineligible",
    "candidate_order": "requested primary, then unused split reserves in file order",
    "fixed_cardinality": True,
    "failed_attempt_location": "failure_history and attempt_provenance",
    "final_identity_location": "selected_city_identities and replacement_assignments",
    "active_manifest_scope": "selected cities only",
}

CANDIDATE_POOL_CONTRACT = {
    "cli_flag": "--select-first-complete",
    "allowed_splits": ["source", "validation"],
    "candidate_order": "configured split cities in file order, then split reserves in file order",
    "quota_source": "config.required_city_counts[split]",
    "eligibility_rule": "only deterministic CityIneligibleError from frozen QA/support/completeness, including twice-confirmed non-TIFF content for a required GeoTIFF asset, may consume a candidate; transport/read/OS/internal errors abort without advancing candidate order",
    "selection_information": "QA/support/completeness only; no temperature distribution or model result",
    "active_manifest_scope": "first quota-complete cities only",
}

NPZ_ARRAY_NAMES = (
    "optical", "lowres_lst", "target_lst", "valid", "eligible",
    "built_fraction", "water_fraction", "lulc_coverage_fraction120",
    "target_support_count120", "target_support_fraction120",
    "pre_lowres_valid120", "coarse_lst480", "coarse_support_count480",
    "coarse_valid480", "interpolation_weight120",
    "interpolation_value_numerator120", "qa_valid_buffer0_30",
    "qa_valid_buffer1_30", "qa_valid_buffer2_30",
    "qa_valid_direct_aerosol30", "qa_reason30", "metadata",
)

CONSTRUCTION_CONFIG_KEYS = (
    "fine_delivery_m", "target_grid_m", "synthetic_coarse_grid_m", "aoi_size_m",
    "fine_to_target_factor", "target_to_input_factor", "target_block_minimum_count",
    "coarse_block_minimum_count", "full_support_count", "interpolation_weight_min",
    "lulc_cell_coverage_min", "lulc_aoi_coverage_fraction_min", "scene_years",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8"))


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json_bytes(value))


@contextmanager
def builder_lock(output_dir: Path):
    """Exclude concurrent writers without adding a file inside the output."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.builder.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another builder owns output lock {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonicalize_provenance(value: Any, *, provider: str,
                            _pointer: str = "") -> Any:
    if isinstance(value, list):
        return [canonicalize_provenance(item, provider=provider,
                                        _pointer=f"{_pointer}/{index}")
                for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    return {
        key: canonicalize_provenance(nested, provider=provider,
                                     _pointer=f"{_pointer}/{key}")
        for key, nested in value.items()
        if not (provider == "nasa_power" and _pointer == "" and key == "times")
    }


def provenance_hashes(value: Any, *, provider: str) -> dict[str, str]:
    return {
        "raw_sha256": sha256_bytes(json_bytes(value)),
        "canonical_sha256": canonical_json_sha256(
            canonicalize_provenance(value, provider=provider)
        ),
    }


def request_json(url: str, *, payload: dict[str, Any] | None = None,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
    attempts = len(REQUEST_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(attempts):
        try:
            response = (requests.post(url, json=payload, timeout=120)
                        if payload is not None
                        else requests.get(url, params=params, timeout=120))
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(REQUEST_RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("unreachable request retry state")


def _range_probe(url: str) -> tuple[bytes, str | None, int]:
    attempts = len(REQUEST_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(attempts):
        try:
            response = requests.get(
                url, headers={"Range": "bytes=0-15"}, timeout=120)
            response.raise_for_status()
            return (response.content[:16], response.headers.get("Content-Range"),
                    len(response.content))
        except requests.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(REQUEST_RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("unreachable range-probe retry state")


def confirmed_non_tiff_payload(url: str) -> bool:
    """Return true only for two identical successful non-TIFF range responses."""
    tiff_signatures = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
    first = _range_probe(url)
    if first[0].startswith(tiff_signatures):
        return False
    time.sleep(REQUEST_RETRY_DELAYS_SECONDS[0])
    second = _range_probe(url)
    return first == second and not second[0].startswith(tiff_signatures)


def sas_token(collection: str) -> str:
    endpoint = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
    return request_json(endpoint)["token"]


def signed_href(item: dict[str, Any], key: str, token: str) -> str:
    href = item["assets"][key]["href"]
    return href if not token else f"{href}?{token}"


def normalized_construction_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in CONSTRUCTION_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing construction configuration keys: {missing}")
    normalized = {key: config[key] for key in CONSTRUCTION_CONFIG_KEYS}
    normalized["scene_years"] = [int(year) for year in normalized["scene_years"]]
    return normalized


def assert_construction_geometry(config: dict[str, Any]) -> None:
    fine, target = int(config["fine_delivery_m"]), int(config["target_grid_m"])
    coarse = int(config["synthetic_coarse_grid_m"])
    f1, f2 = int(config["fine_to_target_factor"]), int(config["target_to_input_factor"])
    years = [int(year) for year in config["scene_years"]]
    if target != fine * f1 or coarse != target * f2:
        raise ValueError("Configured grid scales and factors are inconsistent")
    if int(config["aoi_size_m"]) % coarse:
        raise ValueError("aoi_size_m must be divisible by synthetic_coarse_grid_m")
    if len(years) != 3 or len(set(years)) != 3:
        raise ValueError("Exactly three distinct scene years are required")
    block1, block2 = f1 * f1, f2 * f2
    if not 1 <= int(config["target_block_minimum_count"]) <= block1:
        raise ValueError("Invalid target_block_minimum_count")
    if not 1 <= int(config["coarse_block_minimum_count"]) <= block2:
        raise ValueError("Invalid coarse_block_minimum_count")
    if int(config["full_support_count"]) != block1 or block1 != block2:
        raise ValueError("full_support_count must equal both block areas")
    for key in ("interpolation_weight_min", "lulc_cell_coverage_min",
                "lulc_aoi_coverage_fraction_min"):
        if not 0.0 < float(config[key]) <= 1.0:
            raise ValueError(f"Invalid {key}")


def required_npz_arrays(config: dict[str, Any]) -> dict[str, list[int]]:
    fine = int(config["aoi_size_m"]) // int(config["fine_delivery_m"])
    target = int(config["aoi_size_m"]) // int(config["target_grid_m"])
    coarse = int(config["aoi_size_m"]) // int(config["synthetic_coarse_grid_m"])
    expected = {
        name: ([6, target, target] if name == "optical"
               else ([fine, fine] if name.startswith("qa_")
                     else ([coarse, coarse] if name.startswith("coarse_")
                           else [target, target])))
        for name in NPZ_ARRAY_NAMES if name != "metadata"
    }
    expected["metadata"] = []
    return expected


def validate_contract(config: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    construction = normalized_construction_config(config)
    if contract.get("schema_version") != config.get("schema_version"):
        raise ValueError("Build-contract schema does not match config schema")
    if contract.get("construction_config") != construction:
        raise ValueError("Build-contract construction_config does not match config")
    if contract.get("qa_reason_bits") != QA_REASON_BITS:
        raise ValueError("Build-contract QA reason dictionary does not match builder")
    assert_construction_geometry(construction)
    if contract.get("qa_sensitivity") != QA_SENSITIVITY:
        raise ValueError("Build-contract QA sensitivity dictionary does not match builder")
    if contract.get("required_npz_arrays") != required_npz_arrays(construction):
        raise ValueError("Build-contract NPZ array names/shapes do not match builder")
    if contract.get("canonical_grid_contract") != CANONICAL_GRID_CONTRACT:
        raise ValueError("Build-contract canonical-grid rule does not match builder")
    if contract.get("replacement_contract") != REPLACEMENT_CONTRACT:
        raise ValueError("Build-contract reserve-replacement rule does not match builder")
    if contract.get("candidate_pool_contract") != CANDIDATE_POOL_CONTRACT:
        raise ValueError("Build-contract candidate-pool rule does not match builder")
    required_counts = config.get("required_city_counts")
    if (
        not isinstance(required_counts, dict)
        or set(required_counts) != {"source", "validation"}
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 2
               for value in required_counts.values())
        or contract.get("required_city_counts") != required_counts
    ):
        raise ValueError("Build-contract required city counts do not match config")
    return construction


def city_aoi_bbox_wgs84(city: dict[str, Any], config: dict[str, Any]) -> list[float]:
    zone = int(math.floor((float(city["lon"]) + 180.0) / 6.0)) + 1
    epsg = (32600 if float(city["lat"]) >= 0.0 else 32700) + zone
    xs, ys = transform("EPSG:4326", f"EPSG:{epsg}", [city["lon"]], [city["lat"]])
    half = float(config["aoi_size_m"]) / 2.0 + float(config["target_grid_m"])
    return list(transform_bounds(f"EPSG:{epsg}", "EPSG:4326",
                                 xs[0] - half, ys[0] - half,
                                 xs[0] + half, ys[0] + half, densify_pts=21))


def query_landsat(city: dict[str, Any], year: int,
                   config: dict[str, Any]) -> dict[str, Any]:
    query = {
        "collections": [LANDSAT_COLLECTION],
        "bbox": city_aoi_bbox_wgs84(city, config),
        "datetime": f"{year}-{config['season_start']}/{year}-{config['season_end']}",
        "query": {
            "platform": {"in": ["landsat-8", "landsat-9"]},
            "landsat:collection_category": {"eq": "T1"},
            "eo:cloud_cover": {"lte": config["candidate_catalog_cloud_max"]},
        },
        "limit": 100,
    }
    return request_json(STAC_SEARCH, payload=query)


def select_lulc_items(response: dict[str, Any], config: dict[str, Any],
                      city_name: str) -> list[dict[str, Any]]:
    year = int(config["lulc_year"])
    matches = sorted(
        (feature for feature in response.get("features", [])
         if feature["id"].endswith(f"-{year}") and "data" in feature.get("assets", {})),
        key=lambda feature: feature["id"],
    )
    if not matches:
        raise CityIneligibleError(
            f"No {year} LULC tile covers full-AOI query for {city_name}")
    if len({feature["id"] for feature in matches}) != len(matches):
        raise RuntimeError(f"Duplicate LULC item IDs for {city_name}")
    return matches


def query_lulc(city: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    year = int(config["lulc_year"])
    query = {
        "collections": [config["lulc_collection"]],
        "bbox": city_aoi_bbox_wgs84(city, config),
        "datetime": f"{year}-01-01/{year}-12-31",
        "limit": 100,
    }
    return request_json(STAC_SEARCH, payload=query)


def query_power(city: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    start_year, end_year = min(config["scene_years"]), max(config["scene_years"])
    params = {
        "parameters": "T2M_MAX", "community": "AG",
        "longitude": city["lon"], "latitude": city["lat"],
        "start": f"{start_year}{config['season_start'].replace('-', '')}",
        "end": f"{end_year}{config['season_end'].replace('-', '')}", "format": "JSON",
    }
    return request_json(POWER_API, params=params)


def grid_signature(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "crs": str(dataset.crs), "transform": list(tuple(dataset.transform)),
        "shape": [int(dataset.height), int(dataset.width)],
        "width": int(dataset.width), "height": int(dataset.height),
        "dtype": str(dataset.dtypes[0]), "nodata": dataset.nodata,
    }


def assert_delivered_grid(dataset: rasterio.io.DatasetReader, config: dict[str, Any],
                          reference: dict[str, Any] | None = None,
                          label: str = "asset") -> dict[str, Any]:
    affine, expected = dataset.transform, float(config["fine_delivery_m"])
    if dataset.crs is None or not dataset.crs.is_projected:
        raise ValueError(f"{label} must use a projected CRS")
    if not (math.isclose(affine.a, expected, abs_tol=1e-8)
            and math.isclose(affine.e, -expected, abs_tol=1e-8)
            and math.isclose(affine.b, 0.0, abs_tol=1e-10)
            and math.isclose(affine.d, 0.0, abs_tol=1e-10)):
        raise ValueError(f"{label} is not on the required north-up {expected:g} m grid")
    signature = grid_signature(dataset)
    if reference is not None:
        same = (signature["crs"] == reference["crs"]
                and signature["width"] == reference["width"]
                and signature["height"] == reference["height"]
                and Affine(*signature["transform"]).almost_equals(
                    Affine(*reference["transform"]), precision=1e-10))
        if not same:
            raise ValueError(f"{label} grid is not co-registered with lwir11")
    return signature


def centered_window(dataset: rasterio.io.DatasetReader, lon: float, lat: float,
                    size_m: int) -> Window:
    affine = dataset.transform
    if not (math.isclose(affine.b, 0.0, abs_tol=1e-10)
            and math.isclose(affine.d, 0.0, abs_tol=1e-10)
            and affine.a > 0 and affine.e < 0):
        raise ValueError("centered_window requires a north-up projected grid")
    width_float, height_float = float(size_m) / affine.a, float(size_m) / abs(affine.e)
    width, height = int(round(width_float)), int(round(height_float))
    if not (math.isclose(width_float, width, abs_tol=1e-9)
            and math.isclose(height_float, height, abs_tol=1e-9)):
        raise ValueError("AOI size is not an integer number of delivered pixels")
    if width % 16 or height % 16:
        raise ValueError("AOI delivered-pixel shape must be divisible by 16")
    xs, ys = transform("EPSG:4326", dataset.crs, [lon], [lat])
    fractional_col, fractional_row = (~affine) * (xs[0], ys[0])
    center_col = int(math.floor(fractional_col + 0.5))
    center_row = int(math.floor(fractional_row + 0.5))
    return Window(center_col - width // 2, center_row - height // 2, width, height)


def window_center_metadata(dataset: rasterio.io.DatasetReader, window: Window,
                           lon: float, lat: float) -> dict[str, Any]:
    x, y = transform("EPSG:4326", dataset.crs, [lon], [lat])
    rx, ry = dataset.transform * (window.col_off + window.width / 2.0,
                                  window.row_off + window.height / 2.0)
    dx, dy = rx - x[0], ry - y[0]
    return {
        "requested_center_lonlat": [float(lon), float(lat)],
        "requested_center_projected": [float(x[0]), float(y[0])],
        "realized_grid_center_projected": [float(rx), float(ry)],
        "center_offset_xy_m": [float(dx), float(dy)],
        "center_offset_distance_m": float(math.hypot(dx, dy)),
        "center_rule": "nearest delivered-grid corner for even-sized pixel-aligned AOI",
    }


def coverage_fraction(window: Window, dataset: rasterio.io.DatasetReader) -> float:
    try:
        intersection = window.intersection(Window(0, 0, dataset.width, dataset.height))
    except rasterio.errors.WindowError:
        return 0.0
    return float(intersection.width * intersection.height / (window.width * window.height))


def canonical_city_grid(city: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Return a scene-independent, globally anchored local UTM delivery grid."""
    lon, lat = float(city["lon"]), float(city["lat"])
    zone = min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))
    epsg = (32600 if lat >= 0.0 else 32700) + zone
    crs = f"EPSG:{epsg}"
    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    pixel = float(config["fine_delivery_m"])
    size = int(config["aoi_size_m"]) // int(config["fine_delivery_m"])
    if size * int(config["fine_delivery_m"]) != int(config["aoi_size_m"]) or size % 16:
        raise ValueError("Canonical city grid has invalid AOI/pixel geometry")
    center_x = math.floor(xs[0] / pixel + 0.5) * pixel
    center_y = math.floor(ys[0] / pixel + 0.5) * pixel
    affine = Affine(pixel, 0.0, center_x - size * pixel / 2.0,
                    0.0, -pixel, center_y + size * pixel / 2.0)
    dx, dy = center_x - xs[0], center_y - ys[0]
    return {
        "crs": crs, "transform": affine, "shape": (size, size),
        "metadata": {
            "requested_center_lonlat": [lon, lat],
            "requested_center_projected": [float(xs[0]), float(ys[0])],
            "realized_grid_center_projected": [float(center_x), float(center_y)],
            "center_offset_xy_m": [float(dx), float(dy)],
            "center_offset_distance_m": float(math.hypot(dx, dy)),
            "center_rule": ("nearest 30 m corner on the city-longitude UTM-zone grid; "
                            "even 640-pixel AOI; grid independent of scene/path"),
            "canonical_grid_crs": crs,
            "canonical_grid_rule": ("WGS84 UTM zone floor((lon+180)/6)+1; 30 m axes "
                                    "anchored at integer multiples of 30 m; requested "
                                    "center rounded half-up to nearest grid corner"),
            "reprojection_rule": ("nearest-neighbor reprojection of every delivered "
                                  "Landsat numeric/QA asset to the canonical 30 m grid "
                                  "before QA masking and block aggregation; destination "
                                  "fill is zero and therefore rejected"),
        },
    }


def _source_window_for_grid(dataset: rasterio.io.DatasetReader,
                            destination_grid: dict[str, Any]) -> Window:
    height, width = destination_grid["shape"]
    left, bottom, right, top = array_bounds(height, width,
                                             destination_grid["transform"])
    bounds = transform_bounds(destination_grid["crs"], dataset.crs,
                              left, bottom, right, top, densify_pts=21)
    window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
    return Window(window.col_off - 2, window.row_off - 2,
                  window.width + 4, window.height + 4)


def reproject_asset_to_canonical(dataset: rasterio.io.DatasetReader,
                                 destination_grid: dict[str, Any],
                                 fill_value: int = 0) -> np.ndarray:
    """Nearest-sample a delivered asset onto the explicitly frozen city grid."""
    window = _source_window_for_grid(dataset, destination_grid)
    source = dataset.read(1, window=window, boundless=True, fill_value=fill_value)
    destination = np.full(destination_grid["shape"], fill_value, dtype=source.dtype)
    reproject(source=source, destination=destination,
              src_transform=dataset.window_transform(window), src_crs=dataset.crs,
              dst_transform=destination_grid["transform"], dst_crs=destination_grid["crs"],
              resampling=Resampling.nearest, dst_nodata=fill_value,
              init_dest_nodata=True)
    return destination


def canonical_footprint_fraction(dataset: rasterio.io.DatasetReader,
                                 destination_grid: dict[str, Any]) -> float:
    window = _source_window_for_grid(dataset, destination_grid)
    try:
        intersection = window.intersection(Window(0, 0, dataset.width, dataset.height))
    except rasterio.errors.WindowError:
        return 0.0
    intersection = intersection.round_offsets().round_lengths()
    if intersection.width <= 0 or intersection.height <= 0:
        return 0.0
    source = np.ones((int(intersection.height), int(intersection.width)), dtype=np.uint8)
    destination = np.zeros(destination_grid["shape"], dtype=np.uint8)
    reproject(source=source, destination=destination,
              src_transform=dataset.window_transform(intersection), src_crs=dataset.crs,
              dst_transform=destination_grid["transform"], dst_crs=destination_grid["crs"],
              resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
              init_dest_nodata=True)
    return float((destination != 0).mean())


def read_asset(item: dict[str, Any], key: str, token: str, window: Window | None,
               fill_value: int = 0, *, config: dict[str, Any] | None = None,
               reference_grid: dict[str, Any] | None = None,
               signatures: dict[str, Any] | None = None,
               destination_grid: dict[str, Any] | None = None) -> np.ndarray:
    href = signed_href(item, key, token)
    try:
        with rasterio.open(href) as dataset:
            if config is not None:
                signature = assert_delivered_grid(dataset, config, reference_grid, key)
                if signatures is not None:
                    signatures[key] = signature
            if destination_grid is not None:
                return reproject_asset_to_canonical(dataset, destination_grid, fill_value)
            if window is None:
                raise ValueError("Native asset read requires a window")
            return dataset.read(1, window=window, boundless=True, fill_value=fill_value)
    except rasterio.errors.RasterioIOError as error:
        raise RemoteAssetReadError(item["id"], key, href) from error


def delivered_validity(item: dict[str, Any], token: str, window: Window | None,
                       cloud_buffer_pixels: int = 1, include_optical: bool = True,
                       *, config: dict[str, Any] | None = None,
                       reference_grid: dict[str, Any] | None = None,
                       destination_grid: dict[str, Any] | None = None,
                       return_details: bool = False
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | dict[str, Any]:
    signatures: dict[str, Any] = {}
    def read(key: str) -> np.ndarray:
        return read_asset(item, key, token, window, config=config,
                          reference_grid=reference_grid, signatures=signatures,
                          destination_grid=destination_grid)
    target_raw, qa_pixel, qa_radsat = read("lwir11"), read("qa_pixel"), read("qa_radsat")
    st_qa, aerosol = read("qa").astype(np.int32), read("qa_aerosol")
    optical_raw = np.stack([read(key) for key in OPTICAL_KEYS]) if include_optical else None

    raw_bad = (qa_pixel & sum(1 << bit for bit in range(6))) != 0
    bad1, bad2 = binary_dilation(raw_bad, iterations=1), binary_dilation(raw_bad, iterations=2)
    dilation_only = bad1 & ~raw_bad
    radsat_bad = qa_radsat != 0
    aerosol_fill = (aerosol & 1) != 0
    aerosol_valid = (aerosol & (1 << 1)) != 0
    aerosol_interpolated = (aerosol & (1 << 5)) != 0
    aerosol_invalid = ~(aerosol_valid | aerosol_interpolated)
    aerosol_high = ((aerosol >> 6) & 0b11) == 3
    aerosol_ok = ~aerosol_fill & ~aerosol_invalid & ~aerosol_high
    direct_ok = ~aerosol_fill & aerosol_valid & ~aerosol_interpolated & ~aerosol_high
    target = target_raw.astype(np.float32) * np.float32(0.00341802) + np.float32(149.0)
    stqa_bad, target_zero = st_qa > 300, target_raw == 0
    range_bad = (target < 240.0) | (target > 380.0)
    optical_zero = (np.any(optical_raw == 0, axis=0) if optical_raw is not None
                    else np.zeros(target.shape, dtype=bool))
    other_bad = radsat_bad | ~aerosol_ok | stqa_bad | target_zero | range_bad | optical_zero
    cloud_bad = {0: raw_bad, 1: bad1, 2: bad2}
    if cloud_buffer_pixels not in cloud_bad:
        cloud_bad[cloud_buffer_pixels] = binary_dilation(raw_bad, iterations=cloud_buffer_pixels)
    valid_by_buffer = {key: ~value & ~other_bad for key, value in cloud_bad.items()}
    direct = ~bad1 & ~radsat_bad & direct_ok & ~stqa_bad & ~target_zero & ~range_bad & ~optical_zero

    reason = np.zeros(target.shape, dtype=np.uint16)
    for mask, name in ((raw_bad, "QA_PIXEL_RAW_REJECT"),
                       (dilation_only, "QA_PIXEL_DILATION_ONLY"),
                       (radsat_bad, "QA_RADSAT_NONZERO"),
                       (aerosol_fill, "AEROSOL_FILL"),
                       (aerosol_invalid, "AEROSOL_INVALID_STATE"),
                       (aerosol_high, "AEROSOL_HIGH"),
                       (stqa_bad, "ST_QA_ABOVE_3K"),
                       (target_zero, "TARGET_ZERO"),
                       (range_bad, "TARGET_OUTSIDE_240_380K"),
                       (optical_zero, "OPTICAL_ZERO")):
        reason[mask] |= np.uint16(1 << QA_REASON_BITS[name])
    if not return_details:
        return target_raw, valid_by_buffer[cloud_buffer_pixels], optical_raw
    return {"target_raw": target_raw, "optical_raw": optical_raw,
            "valid_primary": valid_by_buffer[cloud_buffer_pixels],
            "valid_by_buffer": valid_by_buffer, "valid_direct_aerosol": direct,
            "qa_reason": reason, "asset_grid_signatures": signatures}


def aggregate_blocks_state(values: np.ndarray, valid: np.ndarray, factor: int,
                           minimum_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim == 2:
        values, squeeze = values[None], True
    else:
        squeeze = False
    channels, height, width = values.shape
    if valid.shape != (height, width) or height % factor or width % factor:
        raise ValueError("Aggregation shape/mask is invalid")
    shaped_values = values.reshape(channels, height // factor, factor, width // factor, factor)
    shaped_valid = valid.reshape(height // factor, factor, width // factor, factor)
    counts = shaped_valid.sum((1, 3)).astype(np.uint8)
    sums = np.where(shaped_valid[None], shaped_values, 0.0).sum((2, 4))
    means = sums / np.maximum(counts[None], 1)
    out_valid = counts >= minimum_count
    means[:, ~out_valid] = np.nan
    return (means[0] if squeeze else means).astype(np.float32), out_valid, counts


def aggregate_blocks(values: np.ndarray, valid: np.ndarray, factor: int,
                     minimum_count: int) -> tuple[np.ndarray, np.ndarray]:
    means, out_valid, _ = aggregate_blocks_state(values, valid, factor, minimum_count)
    return means, out_valid


def degrade_and_upsample_state(target: np.ndarray, valid: np.ndarray, factor: int = 4,
                               minimum_count: int = 12,
                               interpolation_weight_min: float = 0.5) -> dict[str, np.ndarray]:
    coarse, coarse_valid, coarse_count = aggregate_blocks_state(
        target, valid, factor, minimum_count)
    filled, weight = np.where(coarse_valid, coarse, 0.0), coarse_valid.astype(np.float32)
    numerator = zoom(filled, factor, order=1, mode="nearest", prefilter=False, grid_mode=True)
    up_weight = zoom(weight, factor, order=1, mode="nearest", prefilter=False, grid_mode=True)
    lowres = numerator / np.maximum(up_weight, 1e-6)
    low_valid = up_weight >= interpolation_weight_min
    lowres[~low_valid] = np.nan
    return {"lowres": lowres.astype(np.float32), "low_valid": low_valid,
            "coarse": coarse.astype(np.float32), "coarse_valid": coarse_valid,
            "coarse_count": coarse_count,
            "interpolation_weight": up_weight.astype(np.float32),
            "interpolation_numerator": numerator.astype(np.float32)}


def degrade_and_upsample(target: np.ndarray, valid: np.ndarray, factor: int = 4,
                         minimum_count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    state = degrade_and_upsample_state(target, valid, factor, minimum_count, 0.5)
    return state["lowres"], state["low_valid"]


def _reproject_lulc_indicator(source_array: np.ndarray, source_transform: Affine,
                              source_crs: Any, target_transform: Affine,
                              target_crs: Any, shape: tuple[int, int]) -> np.ndarray:
    destination = np.zeros(shape, dtype=np.float32)
    reproject(source=source_array.astype(np.float32), destination=destination,
              src_transform=source_transform, src_crs=source_crs,
              dst_transform=target_transform, dst_crs=target_crs,
              resampling=Resampling.average, src_nodata=None, dst_nodata=0,
              init_dest_nodata=True)
    return destination


def landcover_fractions(items: list[dict[str, Any]], token: str, target_crs: Any,
                        target_transform: Affine,
                        shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mosaic every intersecting tile with explicit NoData-area accounting."""
    height, width = shape
    left, top = target_transform * (0, 0)
    right, bottom = target_transform * (width, height)
    coverage = np.zeros(shape, dtype=np.float32)
    built_numerator = np.zeros(shape, dtype=np.float32)
    water_numerator = np.zeros(shape, dtype=np.float32)
    for item in sorted(items, key=lambda feature: feature["id"]):
        with rasterio.open(signed_href(item, "data", token)) as source:
            bounds = transform_bounds(target_crs, source.crs, left, bottom, right, top,
                                      densify_pts=21)
            source_window = from_bounds(*bounds, transform=source.transform).round_offsets().round_lengths()
            source_window = Window(source_window.col_off - 2, source_window.row_off - 2,
                                   source_window.width + 4, source_window.height + 4)
            nodata = source.nodata
            if nodata is None:
                bands = item.get("assets", {}).get("data", {}).get("raster:bands", [{}])
                nodata = bands[0].get("nodata", 0)
            classes = source.read(1, window=source_window, boundless=True, fill_value=nodata)
            source_transform = source.window_transform(source_window)
            source_valid = classes != nodata
            tile_coverage = _reproject_lulc_indicator(
                source_valid, source_transform, source.crs, target_transform, target_crs, shape)
            tile_built = _reproject_lulc_indicator(
                source_valid & (classes == 7), source_transform, source.crs,
                target_transform, target_crs, shape)
            tile_water = _reproject_lulc_indicator(
                source_valid & (classes == 1), source_transform, source.crs,
                target_transform, target_crs, shape)
        remaining = np.clip(1.0 - coverage, 0.0, 1.0)
        take = np.minimum(np.clip(tile_coverage, 0.0, 1.0), remaining)
        built_fraction = np.divide(tile_built, tile_coverage, out=np.zeros_like(tile_built),
                                   where=tile_coverage > 1e-6)
        water_fraction = np.divide(tile_water, tile_coverage, out=np.zeros_like(tile_water),
                                   where=tile_coverage > 1e-6)
        built_numerator += built_fraction * take
        water_numerator += water_fraction * take
        coverage += take
    coverage = np.clip(coverage, 0.0, 1.0)
    built = np.divide(built_numerator, coverage, out=np.zeros_like(built_numerator),
                      where=coverage > 1e-6)
    water = np.divide(water_numerator, coverage, out=np.zeros_like(water_numerator),
                      where=coverage > 1e-6)
    return built.astype(np.float32), water.astype(np.float32), coverage.astype(np.float32)


def enforce_lulc_coverage(coverage: np.ndarray,
                          config: dict[str, Any]) -> tuple[np.ndarray, float]:
    complete = coverage >= float(config["lulc_cell_coverage_min"])
    fraction = float(complete.mean())
    required = float(config["lulc_aoi_coverage_fraction_min"])
    if fraction < required:
        raise CityIneligibleError(
            f"LULC coverage gate failed: {fraction:.6f} < {required:.6f}")
    return complete, fraction


def power_by_date(response: dict[str, Any]) -> dict[str, float]:
    values = response["properties"]["parameter"]["T2M_MAX"]
    return {datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d"): float(value)
            for date, value in values.items() if float(value) > -900}


def candidate_record(item: dict[str, Any], tmax: float | None) -> dict[str, Any]:
    return {"item_id": item["id"], "datetime": item["properties"].get("datetime"),
            "platform": item["properties"].get("platform"),
            "catalog_cloud_cover": item["properties"].get("eo:cloud_cover"),
            "proj_epsg": item["properties"].get("proj:epsg"),
            "power_t2m_max_c": tmax}


def select_and_build_scene(city: dict[str, Any], year: int, config: dict[str, Any],
                           contract: dict[str, Any], landsat_response: dict[str, Any],
                           power: dict[str, float], landsat_token: str,
                           lulc_items: list[dict[str, Any]], lulc_token: str,
                           output_dir: Path) -> dict[str, Any]:
    lulc_year = int(config["lulc_year"])
    if int(year) < lulc_year:
        raise TemporalEligibilityError(
            f"Scene year {year} precedes LULC year {lulc_year}; future-year "
            "eligibility information is forbidden")
    required = (*OPTICAL_KEYS, "lwir11", "qa_pixel", "qa_radsat", "qa", "qa_aerosol")
    candidates = []
    for item in landsat_response.get("features", []):
        if "L2SP" not in item["id"] or not all(key in item.get("assets", {}) for key in required):
            continue
        date = item["properties"].get("datetime", "")[:10]
        if date in power:
            candidates.append((item, power[date]))
    candidates.sort(key=lambda pair: (-pair[1],
                                      float(pair[0]["properties"].get("eo:cloud_cover", 1000.0)),
                                      pair[0]["properties"].get("datetime", ""), pair[0]["id"]))
    rejections: list[dict[str, Any]] = []
    canonical_grid = canonical_city_grid(city, config)
    target_crs = canonical_grid["crs"]
    transform30 = canonical_grid["transform"]
    center_metadata = canonical_grid["metadata"]
    for item, tmax in candidates:
        record = candidate_record(item, tmax)
        target_href = signed_href(item, "lwir11", landsat_token)
        try:
            with rasterio.open(target_href) as target_ds:
                reference_grid = assert_delivered_grid(target_ds, config, label="lwir11")
                scene_coverage = canonical_footprint_fraction(target_ds, canonical_grid)
                source_window = _source_window_for_grid(target_ds, canonical_grid)
        except rasterio.errors.RasterioIOError as error:
            failure = RemoteAssetReadError(item["id"], "lwir11", target_href)
            if confirmed_non_tiff_payload(failure.href):
                record.update({"accepted": False,
                               "reason": "required_asset_not_geotiff",
                               "asset_key": failure.asset_key, "fraction": -1.0})
                rejections.append(record)
                continue
            raise failure from error
        if scene_coverage < 0.999:
            record.update({"accepted": False, "reason": "aoi_coverage",
                           "fraction": scene_coverage})
            rejections.append(record)
            continue
        try:
            qa = delivered_validity(item, landsat_token, None, cloud_buffer_pixels=1,
                                    config=config, reference_grid=reference_grid,
                                    destination_grid=canonical_grid,
                                    return_details=True)
        except RemoteAssetReadError as error:
            if confirmed_non_tiff_payload(error.href):
                record.update({"accepted": False,
                               "reason": "required_asset_not_geotiff",
                               "asset_key": error.asset_key, "fraction": -1.0})
                rejections.append(record)
                continue
            raise
        target_raw, optical_raw, valid30 = qa["target_raw"], qa["optical_raw"], qa["valid_primary"]
        target30 = target_raw.astype(np.float32) * np.float32(0.00341802) + np.float32(149.0)
        target120, target_valid120, target_count120 = aggregate_blocks_state(
            target30, valid30, int(config["fine_to_target_factor"]),
            int(config["target_block_minimum_count"]))
        if float(target_valid120.mean()) < float(config["candidate_aoi_qa_fraction_min"]):
            record.update({"accepted": False, "reason": "target_support",
                           "fraction": float(target_valid120.mean())})
            rejections.append(record)
            continue
        optical30 = optical_raw.astype(np.float32) * np.float32(2.75e-5) - np.float32(0.2)
        optical120, optical_valid120, _ = aggregate_blocks_state(
            optical30, valid30, int(config["fine_to_target_factor"]),
            int(config["target_block_minimum_count"]))
        pre_lowres_valid120 = target_valid120 & optical_valid120
        coarse_state = degrade_and_upsample_state(
            target120, pre_lowres_valid120, int(config["target_to_input_factor"]),
            int(config["coarse_block_minimum_count"]),
            float(config["interpolation_weight_min"]))
        valid120 = pre_lowres_valid120 & coarse_state["low_valid"]
        final_availability = float(valid120.mean())
        if final_availability < float(config["candidate_aoi_qa_fraction_min"]):
            record.update({"accepted": False, "reason": "final_model_support",
                           "fraction": final_availability})
            rejections.append(record)
            continue
        transform120 = transform30 * Affine.scale(int(config["fine_to_target_factor"]))
        transform480 = transform120 * Affine.scale(int(config["target_to_input_factor"]))
        built, water, lulc_coverage = landcover_fractions(
            lulc_items, lulc_token, target_crs, transform120, target120.shape)
        lulc_complete, lulc_fraction = enforce_lulc_coverage(lulc_coverage, config)
        eligible = (lulc_complete
                    & (built >= float(config["eligible_built_fraction_min"]))
                    & (water <= float(config["eligible_water_fraction_max"])))
        evaluation = valid120 & eligible
        if int(evaluation.sum()) < 500:
            record.update({"accepted": False, "reason": "insufficient_eligible_cells",
                           "fraction": float(evaluation.mean()), "count": int(evaluation.sum())})
            rejections.append(record)
            continue
        reason_counts = {name: int(((qa["qa_reason"] & (1 << bit)) != 0).sum())
                         for name, bit in QA_REASON_BITS.items()}
        metadata = {
            **record, **center_metadata, "accepted": True, "city": city["name"],
            "split": city["split"], "year": int(year), "crs": str(target_crs),
            "target_definition": contract["target_definition"],
            "coarse_definition": contract["coarse_definition"],
            "transform30": list(tuple(transform30)),
            "transform120": list(tuple(transform120)),
            "transform480": list(tuple(transform480)),
            "window30": [0, 0, int(canonical_grid["shape"][1]),
                         int(canonical_grid["shape"][0])],
            "source_window30": [int(source_window.col_off), int(source_window.row_off),
                                int(source_window.width), int(source_window.height)],
            "source_crs": reference_grid["crs"],
            "source_transform30": reference_grid["transform"],
            "source_to_canonical_reprojected": True,
            "canonical_grid_contract": CANONICAL_GRID_CONTRACT,
            "shape30": list(target30.shape), "shape120": list(target120.shape),
            "shape480": list(coarse_state["coarse"].shape),
            "valid_fraction120": final_availability,
            "eligible_fraction120": float(eligible.mean()),
            "evaluation_cell_count": int(evaluation.sum()),
            "target_support_fraction_mean120": float(
                (target_count120.astype(np.float32) / float(config["full_support_count"])).mean()),
            "lulc_item_ids": [feature["id"] for feature in lulc_items],
            "lulc_item_id": lulc_items[0]["id"] if len(lulc_items) == 1 else None,
            "lulc_coverage_fraction": lulc_fraction,
            "lulc_mean_coverage": float(lulc_coverage.mean()),
            "asset_grid_signatures": {"lwir11": reference_grid,
                                      **qa["asset_grid_signatures"]},
            "qa_reason_bits": QA_REASON_BITS, "qa_reason_counts30": reason_counts,
            "qa_sensitivity": contract["qa_sensitivity"],
            "qa_rule": ("QA_PIXEL bits0:5 +1px dilation; QA_RADSAT=0; aerosol fill "
                        "rejected; aerosol valid-or-interpolated and not-high; ST_QA<=3K; "
                        "assets nonzero; 240<=ST<=380K; configured support thresholds"),
            "rejected_higher_ranked_candidates": rejections,
        }
        arrays = {
            "optical": optical120.astype(np.float32),
            "lowres_lst": coarse_state["lowres"].astype(np.float32),
            "target_lst": target120.astype(np.float32),
            "valid": valid120.astype(np.uint8),
            "eligible": eligible.astype(np.uint8),
            "built_fraction": built.astype(np.float32),
            "water_fraction": water.astype(np.float32),
            "lulc_coverage_fraction120": lulc_coverage.astype(np.float32),
            "target_support_count120": target_count120.astype(np.uint8),
            "target_support_fraction120": (target_count120.astype(np.float32)
                                           / np.float32(config["full_support_count"])),
            "pre_lowres_valid120": pre_lowres_valid120.astype(np.uint8),
            "coarse_lst480": coarse_state["coarse"].astype(np.float32),
            "coarse_support_count480": coarse_state["coarse_count"].astype(np.uint8),
            "coarse_valid480": coarse_state["coarse_valid"].astype(np.uint8),
            "interpolation_weight120": coarse_state["interpolation_weight"].astype(np.float32),
            "interpolation_value_numerator120": coarse_state["interpolation_numerator"].astype(np.float32),
            "qa_valid_buffer0_30": qa["valid_by_buffer"][0].astype(np.uint8),
            "qa_valid_buffer1_30": qa["valid_by_buffer"][1].astype(np.uint8),
            "qa_valid_buffer2_30": qa["valid_by_buffer"][2].astype(np.uint8),
            "qa_valid_direct_aerosol30": qa["valid_direct_aerosol"].astype(np.uint8),
            "qa_reason30": qa["qa_reason"].astype(np.uint16),
            "metadata": np.array(json.dumps(metadata, sort_keys=True)),
        }
        expected_shapes = required_npz_arrays(config)
        actual_shapes = {name: list(array.shape) for name, array in arrays.items()}
        if list(arrays) != list(expected_shapes) or actual_shapes != expected_shapes:
            raise RuntimeError("Scene arrays do not satisfy the frozen NPZ contract")
        missing_metadata = set(contract["required_scene_metadata"]) - set(metadata)
        if missing_metadata:
            raise RuntimeError(f"Scene metadata contract missing {sorted(missing_metadata)}")
        path = output_dir / f"{city['name']}_{year}.npz"
        np.savez_compressed(path, **arrays)
        metadata["file"], metadata["sha256"] = path.name, sha256_file(path)
        return metadata
    summary = "; ".join(f"{r['item_id']}:{r['reason']}:{r.get('fraction', -1):.3f}"
                        for r in rejections)
    raise CityIneligibleError(f"No usable {city['name']} {year} scene. {summary}")


def construction_identity(config_path: Path, contract_path: Path = DEFAULT_CONTRACT,
                          protocol_path: Path = DEFAULT_PROTOCOL,
                          experiment_plan_path: Path = DEFAULT_EXPERIMENT_PLAN,
                          reserve_path: Path = DEFAULT_RESERVE_CONFIG,
                          seal_policy_path: Path = DEFAULT_SEAL_POLICY,
                          ) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {"config_sha256": sha256_file(config_path),
            "contract_sha256": sha256_file(contract_path),
            "protocol_sha256": sha256_file(protocol_path),
            "experiment_plan_sha256": sha256_file(experiment_plan_path),
            "reserve_sha256": sha256_file(reserve_path),
            "seal_policy_sha256": sha256_file(seal_policy_path),
            "code_sha256": sha256_file(script_path), "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "rasterio": rasterio.__version__,
            "requests": requests.__version__}


def prepare_output(output_dir: Path, identity: dict[str, Any], config: dict[str, Any],
                   contract: dict[str, Any], split: str, resume: bool,
                   requested_primary_cities: list[str], candidate_cities: list[str],
                   replace_ineligible: bool = False,
                   selection_mode: str = "fixed_primary_replacement",
                   required_city_count: int | None = None) -> dict[str, Any]:
    required_count = (len(requested_primary_cities) if required_city_count is None
                      else int(required_city_count))
    if not 1 <= required_count <= len(candidate_cities):
        raise ValueError("Required city count is outside the candidate scope")
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume or not manifest_path.exists():
            raise RuntimeError(f"Refusing nonempty output {output_dir}; use a new directory or --resume")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["construction_identity"] != identity
                or manifest["schema_version"] != config["schema_version"]
                or manifest["split"] != split
                or manifest.get("requested_primary_cities") != requested_primary_cities
                or manifest.get("candidate_cities") != candidate_cities
                or manifest.get("replace_ineligible") is not replace_ineligible
                or manifest.get("selection_mode") != selection_mode
                or manifest.get("required_city_count") != required_count
                or manifest.get("construction_config") != normalized_construction_config(config)):
            raise RuntimeError("Resume identity/scope mismatch; a clean output is mandatory")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance").mkdir()
    (output_dir / ".staging").mkdir()
    manifest = {
        "schema_version": config["schema_version"], "split": split,
        "requested_primary_cities": requested_primary_cities,
        "candidate_cities": candidate_cities,
        "selected_cities": [],
        "replace_ineligible": replace_ineligible,
        "selection_mode": selection_mode,
        "required_city_count": required_count,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "construction_identity": identity,
        "construction_config": normalized_construction_config(config),
        "target_definition": contract["target_definition"],
        "coarse_definition": contract["coarse_definition"],
        "qa_reason_bits": contract["qa_reason_bits"],
        "qa_sensitivity": contract["qa_sensitivity"],
        "canonical_grid_contract": contract["canonical_grid_contract"],
        "replacement_contract": contract["replacement_contract"],
        "candidate_pool_contract": contract["candidate_pool_contract"],
        "provenance": {},
        "snapshot_phase_complete": False,
        "city_transactions": {
            city: {"status": "snapshot_pending",
                   "required_years": [int(year) for year in config["scene_years"]]}
            for city in candidate_cities},
        "replacement_assignments": [], "candidate_selection": [],
        "attempt_provenance": {},
        "scenes": [], "failures": [], "failure_history": [], "build_complete": False,
        "sealed_test_unlocked": split == "sealed_test",
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _snapshot_relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _snapshot_registry(manifest: dict[str, Any]) -> dict[str, Any]:
    combined = dict(manifest.get("attempt_provenance", {}))
    combined.update(manifest.get("provenance", {}))
    return combined


def save_snapshot(path: Path, output_dir: Path, manifest: dict[str, Any],
                  manifest_path: Path, response: dict[str, Any],
                  provider: str) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"Unregistered provenance snapshot already exists: {path}")
    raw, hashes = json_bytes(response), provenance_hashes(response, provider=provider)
    atomic_write_bytes(path, raw)
    if sha256_file(path) != hashes["raw_sha256"]:
        raise RuntimeError(f"Snapshot write verification failed: {path}")
    manifest["provenance"][_snapshot_relative(path, output_dir)] = hashes
    atomic_write_json(manifest_path, manifest)
    return response


def load_snapshot(path: Path, output_dir: Path, manifest: dict[str, Any],
                  provider: str) -> dict[str, Any]:
    relative = _snapshot_relative(path, output_dir)
    recorded = _snapshot_registry(manifest).get(relative)
    if recorded is None or not path.is_file():
        raise RuntimeError(f"Missing registered provenance snapshot: {relative}")
    response = json.loads(path.read_text())
    if provenance_hashes(response, provider=provider) != recorded:
        raise RuntimeError(f"Provenance hash mismatch: {relative}")
    return response


def load_or_query_snapshot(path: Path, output_dir: Path, manifest: dict[str, Any],
                           manifest_path: Path, provider: str,
                           fetch: Callable[[], dict[str, Any]],
                           allow_missing_query: bool) -> dict[str, Any]:
    relative = _snapshot_relative(path, output_dir)
    if path.exists() or relative in _snapshot_registry(manifest):
        return load_snapshot(path, output_dir, manifest, provider)
    if not allow_missing_query:
        raise RuntimeError(f"Resume refuses to query missing snapshot: {path.name}")
    return save_snapshot(path, output_dir, manifest, manifest_path, fetch(), provider)


def freeze_query_snapshots(cities: list[dict[str, Any]], config: dict[str, Any],
                           output_dir: Path, manifest: dict[str, Any],
                           manifest_path: Path, *, is_resume: bool) -> None:
    # A resumed run verifies and consumes frozen responses only. An interruption
    # during the initial query phase therefore requires a clean output directory.
    allow_missing_query = not is_resume and not manifest.get("snapshot_phase_complete", False)
    if manifest.get("scenes") and allow_missing_query:
        raise RuntimeError("Cannot query missing snapshots after scene publication")
    for city in cities:
        name = city["name"]
        load_or_query_snapshot(
            output_dir / "provenance" / f"{name}_nasa_power.json", output_dir,
            manifest, manifest_path, "nasa_power",
            lambda city=city: query_power(city, config), allow_missing_query)
        load_or_query_snapshot(
            output_dir / "provenance" / f"{name}_lulc_stac.json", output_dir,
            manifest, manifest_path, "stac",
            lambda city=city: query_lulc(city, config), allow_missing_query)
        for year in config["scene_years"]:
            load_or_query_snapshot(
                output_dir / "provenance" / f"{name}_{year}_landsat_stac.json",
                output_dir, manifest, manifest_path, "stac",
                lambda city=city, year=year: query_landsat(city, int(year), config),
                allow_missing_query)
        status = manifest["city_transactions"][name]["status"]
        if status in {"snapshot_pending", "snapshots_frozen", "building"}:
            manifest["city_transactions"][name] = {
                "status": "snapshots_frozen",
                "required_years": list(config["scene_years"]),
            }
        elif status not in {"failed", "committed"}:
            raise RuntimeError(f"Unknown city transaction status for {name}: {status}")
    manifest["snapshot_phase_complete"] = True
    atomic_write_json(manifest_path, manifest)


def load_city_snapshots(city: dict[str, Any], config: dict[str, Any], output_dir: Path,
                        manifest: dict[str, Any]) -> tuple[dict[str, Any],
                                                          list[dict[str, Any]],
                                                          dict[int, dict[str, Any]]]:
    name = city["name"]
    power = load_snapshot(output_dir / "provenance" / f"{name}_nasa_power.json",
                          output_dir, manifest, "nasa_power")
    lulc_response = load_snapshot(output_dir / "provenance" / f"{name}_lulc_stac.json",
                                  output_dir, manifest, "stac")
    lulc_items = select_lulc_items(lulc_response, config, name)
    landsat = {int(year): load_snapshot(
        output_dir / "provenance" / f"{name}_{year}_landsat_stac.json",
        output_dir, manifest, "stac") for year in config["scene_years"]}
    return power, lulc_items, landsat


def verify_committed_city(city_name: str, config: dict[str, Any], output_dir: Path,
                          manifest: dict[str, Any]) -> None:
    expected = sorted(int(year) for year in config["scene_years"])
    entries = sorted((scene for scene in manifest["scenes"] if scene["city"] == city_name),
                     key=lambda scene: int(scene["year"]))
    if [int(scene["year"]) for scene in entries] != expected or len(entries) != 3:
        raise RuntimeError(f"Committed city {city_name} lacks exactly three years")
    for entry in entries:
        path = output_dir / entry["file"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Committed city hash failure: {path}")


def publish_city_transaction(city_name: str, entries: list[dict[str, Any]],
                             staging_dir: Path, output_dir: Path,
                             manifest: dict[str, Any], manifest_path: Path,
                             required_years: list[int], split: str) -> None:
    expected = sorted(int(year) for year in required_years)
    actual = sorted(int(entry["year"]) for entry in entries)
    if len(entries) != 3 or actual != expected:
        raise RuntimeError(f"City transaction requires exactly {expected}, got {actual}")
    ordered = sorted(entries, key=lambda item: int(item["year"]))
    if any(entry["city"] != city_name or entry["split"] != split for entry in ordered):
        raise RuntimeError("City transaction entry identity/split mismatch")
    expected_files = [f"{city_name}_{year}.npz" for year in expected]
    if [entry.get("file") for entry in ordered] != expected_files:
        raise RuntimeError("City transaction scene filenames are not canonical")
    if len({entry.get("item_id") for entry in ordered}) != len(ordered):
        raise RuntimeError("City transaction item identities are not unique")
    if any(scene["city"] == city_name for scene in manifest["scenes"]):
        raise RuntimeError(f"City {city_name} already has published entries")
    # Validate every source and possible orphan destination before moving any byte.
    for entry in entries:
        source, destination = staging_dir / entry["file"], output_dir / entry["file"]
        if not source.is_file() or sha256_file(source) != entry["sha256"]:
            raise RuntimeError(f"Staged scene hash failure: {source}")
        if destination.exists() and sha256_file(destination) != entry["sha256"]:
            raise RuntimeError(f"Orphan destination conflicts: {destination}")
    for entry in entries:
        source, destination = staging_dir / entry["file"], output_dir / entry["file"]
        if destination.exists():
            source.unlink()
        else:
            os.replace(source, destination)
    # This single atomic manifest replacement publishes all three entries together.
    manifest["scenes"].extend(entries)
    manifest["scenes"].sort(key=lambda scene: (scene["city"], int(scene["year"])))
    transaction_payload = {
        "city": city_name,
        "split": split,
        "years": [int(entry["year"]) for entry in ordered],
        "scene_files": [entry["file"] for entry in ordered],
        "scene_sha256": [entry["sha256"] for entry in ordered],
        "item_ids": [entry["item_id"] for entry in ordered],
    }
    manifest["city_transactions"][city_name] = {
        "status": "committed", **transaction_payload,
        "transaction_sha256": canonical_json_sha256(transaction_payload),
    }
    manifest["failures"] = [failure for failure in manifest.get("failures", [])
                            if failure.get("city") != city_name]
    atomic_write_json(manifest_path, manifest)


def _attempt_city(city: dict[str, Any], requested_primary: str,
                  args: argparse.Namespace, config: dict[str, Any],
                  contract: dict[str, Any], manifest: dict[str, Any],
                  manifest_path: Path) -> bool:
    name = city["name"]
    status = manifest["city_transactions"][name]["status"]
    if status == "committed":
        verify_committed_city(name, config, args.output, manifest)
        return True
    if status == "failed":
        return False
    manifest_before_attempt = copy.deepcopy(manifest)
    print(f"city {name} ({args.split}; requested={requested_primary})", flush=True)
    staging = args.output / ".staging" / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest["city_transactions"][name]["status"] = "building"
    atomic_write_json(manifest_path, manifest)
    entries: list[dict[str, Any]] = []
    try:
        power_response, lulc_items, landsat = load_city_snapshots(
            city, config, args.output, manifest)
        power = power_by_date(power_response)
        landsat_token = sas_token(LANDSAT_COLLECTION)
        lulc_token = sas_token(config["lulc_collection"])
        for year in config["scene_years"]:
            entry = select_and_build_scene(
                city, int(year), config, contract, landsat[int(year)], power,
                landsat_token, lulc_items, lulc_token, staging)
            entries.append(entry)
            print(f"  staged {year}: {entry['item_id']} "
                  f"T2M_MAX={entry['power_t2m_max_c']:.2f}C "
                  f"valid={entry['valid_fraction120']:.3f} "
                  f"eligible={entry['evaluation_cell_count']}", flush=True)
        publish_city_transaction(name, entries, staging, args.output, manifest,
                                 manifest_path, [int(y) for y in config["scene_years"]],
                                 args.split)
        print("  committed 3 scenes", flush=True)
        return True
    except CityIneligibleError as error:
        manifest.clear()
        manifest.update(manifest_before_attempt)
        failure = {
            "city": name, "requested_primary": requested_primary,
            "candidate_role": ("reserve" if city.get("_identity_source") == "reserve"
                               else "primary"),
            "error": repr(error),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["city_transactions"][name] = {
            "status": "failed", "required_years": list(config["scene_years"]),
            "staged_years": [int(entry["year"]) for entry in entries],
            "error": repr(error)}
        manifest["failures"] = [item for item in manifest.get("failures", [])
                                if item.get("city") != name] + [failure]
        if not any(item.get("city") == name and item.get("error") == repr(error)
                   for item in manifest.setdefault("failure_history", [])):
            manifest["failure_history"].append(failure)
        if staging.exists():
            shutil.rmtree(staging)
        atomic_write_json(manifest_path, manifest)
        return False
    except BaseException:
        manifest.clear()
        manifest.update(manifest_before_attempt)
        manifest["city_transactions"][name] = {
            "status": "snapshots_frozen",
            "required_years": list(config["scene_years"]),
        }
        if staging.exists():
            shutil.rmtree(staging)
        atomic_write_json(manifest_path, manifest)
        raise


def _finalize_selected_manifest(manifest: dict[str, Any], selected: list[dict[str, Any]],
                                assignments: list[dict[str, Any]], output_dir: Path,
                                manifest_path: Path,
                                candidate_selection: list[dict[str, Any]] | None = None) -> None:
    selected_names = [city["name"] for city in selected]
    if len(selected_names) != int(manifest["required_city_count"]):
        raise RuntimeError("Final selected-city cardinality is incomplete")
    if len(set(selected_names)) != len(selected_names):
        raise RuntimeError("Final selected-city identities are not unique")
    if len(manifest["scenes"]) != 3 * len(selected_names):
        raise RuntimeError("Final scene cardinality is not exactly three per city")
    all_provenance = _snapshot_registry(manifest)
    selected_prefixes = tuple(f"provenance/{name}_" for name in selected_names)
    manifest["provenance"] = {
        key: value for key, value in all_provenance.items()
        if key.startswith(selected_prefixes)
    }
    manifest["attempt_provenance"] = {
        key: value for key, value in all_provenance.items()
        if not key.startswith(selected_prefixes)
    }
    manifest["city_transactions"] = {
        name: manifest["city_transactions"][name] for name in selected_names
    }
    manifest["selected_cities"] = selected_names
    manifest["selected_city_identities"] = [
        {"name": city["name"], "lon": city["lon"], "lat": city["lat"],
         "split": city["split"],
         "identity_source": ("cities.json" if city.get("_identity_source") == "primary"
                             else "reserve_cities.json")}
        for city in selected
    ]
    manifest["replacement_assignments"] = assignments
    manifest["candidate_selection"] = list(candidate_selection or [])
    manifest["failures"] = []
    manifest["build_complete"] = True
    atomic_write_json(manifest_path, manifest)


def _run_builder_locked(args: argparse.Namespace, config: dict[str, Any],
                        contract: dict[str, Any], identity: dict[str, Any],
                        reserve_config: dict[str, Any],
                        unlock: dict[str, Any] | None = None) -> None:
    primaries = [dict(city, _identity_source="primary") for city in config["cities"]
                 if city["split"] == args.split]
    if args.city:
        if len(set(args.city)) != len(args.city):
            raise RuntimeError("Duplicate --city identity")
        requested = set(args.city)
        primaries = [city for city in primaries if city["name"] in requested]
        missing = requested - {city["name"] for city in primaries}
        if missing:
            raise RuntimeError(f"Requested cities are absent from split: {sorted(missing)}")
    if not primaries:
        raise RuntimeError("No cities selected")
    replace_ineligible = bool(getattr(args, "replace_ineligible", False))
    select_first_complete = bool(getattr(args, "select_first_complete", False))
    if select_first_complete:
        if args.split not in CANDIDATE_POOL_CONTRACT["allowed_splits"]:
            raise RuntimeError("Candidate-pool quota selection is forbidden for sealed_test")
        if args.city:
            raise RuntimeError("Candidate-pool quota selection forbids caller-selected cities")
        if not replace_ineligible:
            raise RuntimeError("Candidate-pool quota selection requires frozen split reserves")
    reserve_rows = reserve_config.get(args.split, []) if replace_ineligible else []
    if replace_ineligible and not isinstance(reserve_rows, list):
        raise RuntimeError(f"Reserve list for {args.split} is malformed")
    reserves = [dict(city, split=args.split, _identity_source="reserve")
                for city in reserve_rows]
    candidate_names = [city["name"] for city in [*primaries, *reserves]]
    if len(set(candidate_names)) != len(candidate_names):
        raise RuntimeError("Primary/reserve candidate identities overlap or repeat")
    requested_names = [city["name"] for city in primaries]
    selection_mode = ("ordered_complete_quota" if select_first_complete
                      else "fixed_primary_replacement")
    required_city_count = (int(config["required_city_counts"][args.split])
                           if select_first_complete else len(requested_names))
    manifest = prepare_output(
        args.output, identity, config, contract, args.split, args.resume,
        requested_names, candidate_names, replace_ineligible, selection_mode,
        required_city_count)
    manifest_path = args.output / "manifest.json"
    if manifest.get("build_complete"):
        for name in manifest["selected_cities"]:
            verify_committed_city(name, config, args.output, manifest)
        print("verified complete fixed-cardinality build", flush=True)
        return
    if unlock is not None:
        manifest["sealed_authorization"] = unlock
        atomic_write_json(manifest_path, manifest)
    all_candidates = [*primaries, *reserves]
    freeze_query_snapshots(all_candidates, config, args.output, manifest, manifest_path,
                           is_resume=args.resume)
    selected: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    if select_first_complete:
        candidate_selection: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(all_candidates):
            if len(selected) == required_city_count:
                break
            if _attempt_city(candidate, candidate["name"], args, config, contract,
                             manifest, manifest_path):
                selected.append(candidate)
                candidate_selection.append({
                    "selection_rank": len(selected) - 1,
                    "candidate_order_index": candidate_index,
                    "selected_city": candidate["name"],
                    "identity_source": ("cities.json"
                                        if candidate.get("_identity_source") == "primary"
                                        else "reserve_cities.json"),
                })
        if len(selected) != required_city_count:
            raise RuntimeError(
                f"Only {len(selected)} complete cities for required quota "
                f"{required_city_count} under frozen candidate order")
        _finalize_selected_manifest(
            manifest, selected, [], args.output, manifest_path, candidate_selection)
        return
    reserve_cursor = 0
    for primary in primaries:
        chosen: dict[str, Any] | None = None
        if _attempt_city(primary, primary["name"], args, config, contract,
                         manifest, manifest_path):
            chosen = primary
        elif replace_ineligible:
            while reserve_cursor < len(reserves):
                reserve = reserves[reserve_cursor]
                reserve_cursor += 1
                if _attempt_city(reserve, primary["name"], args, config, contract,
                                 manifest, manifest_path):
                    chosen = reserve
                    assignments.append({
                        "requested_primary": primary["name"],
                        "selected_city": reserve["name"],
                        "reserve_order_index": reserve_cursor - 1,
                    })
                    break
        if chosen is None:
            raise RuntimeError(
                f"No eligible city for requested primary {primary['name']} under frozen order")
        selected.append(chosen)
    _finalize_selected_manifest(manifest, selected, assignments,
                                args.output, manifest_path)


def run_builder(args: argparse.Namespace, config: dict[str, Any],
                contract: dict[str, Any], identity: dict[str, Any],
                reserve_config: dict[str, Any],
                unlock: dict[str, Any] | None = None) -> None:
    with builder_lock(args.output):
        _run_builder_locked(args, config, contract, identity, reserve_config, unlock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data/v2/cities.json"))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--experiment-plan", type=Path, default=DEFAULT_EXPERIMENT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("source", "validation", "sealed_test"), required=True)
    parser.add_argument("--city", action="append")
    parser.add_argument("--replace-ineligible", action="store_true")
    parser.add_argument("--select-first-complete", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unlock-manifest", type=Path)
    parser.add_argument("--seal-audit-path", type=Path)
    parser.add_argument("--reserve-config", type=Path, default=DEFAULT_RESERVE_CONFIG)
    parser.add_argument("--seal-policy", type=Path, default=DEFAULT_SEAL_POLICY)
    parser.add_argument("--source-artifact", action="append", type=Path, default=[])
    parser.add_argument("--validation-artifact", action="append", type=Path, default=[])
    parser.add_argument("--baseline-artifact", action="append", type=Path, default=[])
    parser.add_argument("--candidate-model-code", type=Path)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--evaluator-code", type=Path)
    parser.add_argument("--metrics-code", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, contract = json.loads(args.config.read_text()), json.loads(args.contract.read_text())
    reserve_config = json.loads(args.reserve_config.read_text())
    validate_contract(config, contract)
    identity = construction_identity(
        args.config, args.contract, args.protocol, args.experiment_plan,
        args.reserve_config, args.seal_policy)
    if args.split != "sealed_test":
        run_builder(args, config, contract, identity, reserve_config)
        return

    # Sealed acquisition is available only through the independently pinned,
    # one-shot seal_v2 preflight. No network call occurs before this callback.
    from seal_v2 import EvaluationArtifacts, run_after_sealed_preflight
    required_paths = (args.candidate_model_code, args.candidate_checkpoint,
                      args.evaluator_code, args.metrics_code, args.seal_audit_path)
    if any(path is None for path in required_paths):
        raise RuntimeError("Sealed acquisition requires every frozen artifact path and audit path")
    artifacts = EvaluationArtifacts.from_paths(
        source_artifacts=args.source_artifact,
        validation_artifacts=args.validation_artifact,
        baseline_artifacts=args.baseline_artifact,
        candidate_model_code=args.candidate_model_code,
        candidate_checkpoint=args.candidate_checkpoint,
        evaluator_code=args.evaluator_code,
        metrics_code=args.metrics_code)
    run_after_sealed_preflight(
        lambda authorization: run_builder(args, config, contract, identity,
                                          reserve_config, authorization.to_dict()),
        args.unlock_manifest, artifacts, audit_path=args.seal_audit_path,
        action_kind="acquisition", config_path=args.config,
        protocol_path=args.protocol, reserve_path=args.reserve_config,
        build_contract_path=args.contract, experiment_plan_path=args.experiment_plan,
        policy_path=args.seal_policy)


if __name__ == "__main__":
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF")
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
    main()
