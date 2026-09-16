#!/usr/bin/env python3
"""Build ORQ 48-hour causal forcing sidecars for public G246 scenes.

The raw hourly download and completed-interval interpretation are delegated to
the already-audited v1 builder.  This module adds the per-hour physical solar
state required by ORQ and writes a fit-only normalized model sequence.  It
never opens target arrays or the locked-test descriptor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import build_g246_r2_power_hourly_causal_sidecars as hourly


SCHEMA = "uhi-cdc-g246-r2-power-hourly-sequence-manifest-v2"
SIDECAR_SCHEMA = "uhi-cdc-g246-r2-power-hourly-sequence-sidecar-v2"
NORMALIZATION_SCHEMA = "uhi-cdc-g246-r2-power-hourly-sequence-normalization-v2"
TOKEN_NAMES = (
    "tair_fit_z", "rh_fit_z", "wind_fit_z", "shortwave_fit_z",
    "solar_cos_zenith", "solar_azimuth_east", "solar_azimuth_north",
    "sunrise_exposure_0_1", "solar_phase_0_1", "delta_t_over_48",
)
RAW_ORDER = ("T2M", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN")
DEFAULT_OUTPUT = Path("artifacts/g246_r2_power_hourly_sequence_sidecars_v2")
DEFAULT_V1_SOURCE = hourly.DEFAULT_OUTPUT


class SequenceSidecarError(RuntimeError):
    pass


@dataclass(frozen=True)
class _RawSequence:
    query: hourly.HourlySceneQuery
    values: np.ndarray  # [48, 4], T/RH/W/SW in physical provider units
    valid: np.ndarray   # [48, 4]
    solar: np.ndarray   # [48, 3]
    sunrise: np.ndarray # [48]
    phase: np.ndarray   # [48]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SequenceSidecarError("timestamp must carry a timezone")
    return parsed.astimezone(timezone.utc)


def solar3(timestamp: datetime, longitude: float, latitude: float) -> np.ndarray:
    """Return physical Solar3 without exposing coordinates to the model."""

    timestamp = timestamp.astimezone(timezone.utc)
    day = timestamp.timetuple().tm_yday
    hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    equation = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma) - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma))
    declination = (0.006918 - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma) - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma))
    solar_minutes = (hour * 60.0 + equation + 4.0 * longitude) % 1440.0
    angle = math.radians(solar_minutes / 4.0 - 180.0)
    lat = math.radians(latitude)
    cos_zenith = float(np.clip(
        math.sin(lat) * math.sin(declination)
        + math.cos(lat) * math.cos(declination) * math.cos(angle), -1.0, 1.0,
    ))
    sin_zenith = max(math.sqrt(max(1.0 - cos_zenith ** 2, 0.0)), 1.0e-8)
    east = -math.sin(angle) * math.cos(declination) / sin_zenith
    north = (math.sin(declination) - math.sin(lat) * cos_zenith) / max(
        math.cos(lat) * sin_zenith, 1.0e-8
    )
    norm = max(math.hypot(east, north), 1.0e-8)
    return np.asarray((cos_zenith, east / norm, north / norm), dtype=np.float32)


def _prefix_solar_state(shortwave: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal sunrise exposure and phase from each prefix only.

    A missing shortwave bin makes the entire sequence unsuitable for dynamic
    updates; finite placeholder states are still emitted for robust loading.
    """

    exposure = np.zeros(48, np.float32)
    phase = np.zeros(48, np.float32)
    current = 0.0
    daylight = 0.0
    for index in range(48):
        if not valid[index]:
            current = 0.0
            daylight = 0.0
        elif shortwave[index] > hourly.RADIOMETRIC_DAYLIGHT_THRESHOLD_MJ_PER_HOUR:
            current = min(current + 1.0, hourly.RADIOMETRIC_EXPOSURE_CLIP_HOURS)
            daylight = min(daylight + 1.0, 12.0)
        else:
            current = 0.0
            daylight = 0.0
        exposure[index] = current / hourly.RADIOMETRIC_EXPOSURE_CLIP_HOURS
        phase[index] = min(1.0, daylight / 12.0)
    return exposure, phase


def derive_sequence_from_v1(
    payload: Mapping[str, Any], query: hourly.HourlySceneQuery,
) -> _RawSequence:
    """Convert one validated v1 lag tensor into the ORQ raw sequence."""

    hourly.validate_sidecar(dict(payload))
    lag = payload["sanitized_complete_lag_tensor"]
    source_order = tuple(lag["parameter_order"])
    if source_order != hourly.PARAMETERS:
        raise SequenceSidecarError("v1 hourly parameter order differs")
    values = np.asarray(lag["values_missing_filled_zero"], np.float64)
    valid = np.asarray(lag["valid_mask"], bool)
    gaps = np.asarray(lag["lag_interval_end_hours_before_t_c"], int)
    if values.shape != (4, 48) or valid.shape != values.shape or not np.array_equal(gaps, np.arange(47, -1, -1)):
        raise SequenceSidecarError("v1 hourly lag tensor geometry differs")
    row = {name: index for index, name in enumerate(source_order)}
    ordered = np.stack([values[row[name]] for name in RAW_ORDER], axis=1).astype(np.float32)
    ordered_valid = np.stack([valid[row[name]] for name in RAW_ORDER], axis=1)
    t_c = _parse_utc(lag["t_c_latest_complete_interval_end_utc"]) if "t_c_latest_complete_interval_end_utc" in lag else None
    # v1 stores t_c in its audit, not the model sidecar; infer it from query
    # conservatively as the last complete hourly boundary no later than t0.
    if t_c is None:
        t0 = query.t0
        t_c = t0.replace(minute=0, second=0, microsecond=0)
        if t_c > t0:
            t_c -= timedelta(hours=1)
    ends = [t_c - timedelta(hours=int(gap)) for gap in gaps]
    solar = np.stack([solar3(end - timedelta(minutes=30), query.longitude, query.latitude) for end in ends])
    exposure, phase = _prefix_solar_state(ordered[:, 3], ordered_valid[:, 3])
    return _RawSequence(query, ordered, ordered_valid, solar, exposure, phase)


def fit_transform(rows: Sequence[_RawSequence]) -> dict[str, Any]:
    """Fit equal-region/city/scene moments from public fit predictors only."""

    grouped: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        region: {} for region in ("us", "china", "europe")
    }
    for row in rows:
        if row.query.role != "fit":
            continue
        first = np.zeros(4, np.float64)
        second = np.zeros(4, np.float64)
        for column in range(4):
            selected = row.values[row.valid[:, column], column]
            if selected.size == 0:
                raise SequenceSidecarError(f"fit scene has no valid {RAW_ORDER[column]} forcing")
            first[column] = selected.mean()
            second[column] = np.square(selected).mean()
        grouped[row.query.region].setdefault(row.query.city, []).append((first, second))
    if any(not grouped[region] for region in grouped):
        raise SequenceSidecarError("fit forcing scope lacks a macro-region")
    region_first, region_second = [], []
    for region in ("us", "china", "europe"):
        city_first = [np.mean(np.stack([x[0] for x in scenes]), axis=0) for scenes in grouped[region].values()]
        city_second = [np.mean(np.stack([x[1] for x in scenes]), axis=0) for scenes in grouped[region].values()]
        region_first.append(np.mean(np.stack(city_first), axis=0))
        region_second.append(np.mean(np.stack(city_second), axis=0))
    mean = np.mean(np.stack(region_first), axis=0)
    std = np.sqrt(np.maximum(np.mean(np.stack(region_second), axis=0) - mean ** 2, 1e-12))
    return {
        "schema": NORMALIZATION_SCHEMA, "raw_feature_names": list(RAW_ORDER),
        "mean": mean.tolist(), "std": std.tolist(),
        "weighting": "equal-region -> equal-city -> equal-scene -> valid-hour",
        "fit_scene_count": sum(row.query.role == "fit" for row in rows),
        "target_arrays_opened": False, "locked_test_opened": False,
    }


def model_tokens(row: _RawSequence, transform: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    mean = np.asarray(transform["mean"], np.float32)
    std = np.asarray(transform["std"], np.float32)
    if mean.shape != (4,) or std.shape != (4,) or np.any(~np.isfinite(mean)) or np.any(std <= 0):
        raise SequenceSidecarError("forcing transform is invalid")
    ready = bool(np.all(row.valid))
    normalized = np.zeros_like(row.values, np.float32)
    normalized[row.valid] = ((row.values - mean) / std)[row.valid]
    delta = np.linspace(-47.0 / 48.0, 0.0, 48, dtype=np.float32)
    tokens = np.concatenate((normalized, row.solar, row.sunrise[:, None], row.phase[:, None], delta[:, None]), axis=1)
    if tokens.shape != (48, 10) or not np.all(np.isfinite(tokens)):
        raise SequenceSidecarError("ORQ forcing token geometry is invalid")
    if not ready:
        tokens[:] = 0.0
    return tokens, ready


def _sequence_payload(row: _RawSequence, transform: Mapping[str, Any]) -> dict[str, Any]:
    tokens, ready = model_tokens(row, transform)
    return {
        "schema": SIDECAR_SCHEMA,
        "scene": {
            "scene_id": row.query.scene_id, "role": row.query.role,
            "region": row.query.region, "city": row.query.city,
            "item_id": row.query.item_id, "source_scene_sha256": row.query.source_scene_sha256,
        },
        "tokens": {"names": list(TOKEN_NAMES), "values": tokens.tolist(), "forcing_ready": ready},
        "quality": {"forcing_valid": row.valid.tolist(), "complete_48h_required": True},
        "solar_at_overpass": solar3(row.query.t0, row.query.longitude, row.query.latitude).tolist(),
        "coordinates_or_absolute_time_exposed_to_model": False,
        "target_arrays_opened": False, "locked_test_opened": False,
    }


def _existing_v1_sidecar(root: Path, query: hourly.HourlySceneQuery) -> Mapping[str, Any] | None:
    """Reuse a verified v1 predictor sidecar rather than refetching Fit603."""

    path = root / "sidecars" / query.role / f"{hourly._artifact_stem(query)}.json"  # noqa: SLF001
    if not path.is_file() or path.is_symlink():
        return None
    payload = hourly.daily.read_json(path, "existing v1 forcing sidecar")
    hourly.validate_sidecar(payload)
    return payload


def build(
    output: Path, *, allow_full_public_build: bool = False,
    v1_source_root: Path = DEFAULT_V1_SOURCE, fetcher=hourly.fetch_power_hourly,
) -> dict[str, Any]:
    """Materialise all public ORQ sequence sidecars after an explicit bulk gate."""

    if not allow_full_public_build:
        raise SequenceSidecarError("full public ORQ sequence build requires explicit approval")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    v1_source_root = Path(v1_source_root).resolve()
    v1_cache_root = output / "v1_acquisition_cache"
    all_queries: list[hourly.HourlySceneQuery] = []
    bindings: list[dict[str, Any]] = []
    for role in ("fit", "validation"):
        binding, queries = hourly.collect_queries(
            role=role, allow_validation_predictors=(role == "validation"),
        )
        bindings.append(binding)
        all_queries.extend(queries)
    if len(all_queries) != 648 or len({q.scene_id for q in all_queries}) != 648:
        raise SequenceSidecarError("public ORQ sequence scope must contain exactly 648 scenes")
    raw_rows: list[_RawSequence] = []
    for query in all_queries:
        source = _existing_v1_sidecar(v1_source_root, query)
        if source is None:
            transaction = hourly.process_one(query, v1_cache_root, fetcher=fetcher)
            source = hourly.daily.read_json(v1_cache_root / transaction["sidecar_path"], "v1 forcing sidecar")
        raw_rows.append(derive_sequence_from_v1(source, query))
    transform = fit_transform(raw_rows)
    normalization_path = output / "normalization.json"
    hourly.daily.atomic_json(normalization_path, transform)
    records = []
    for row in raw_rows:
        safe = hourly._artifact_stem(row.query)  # noqa: SLF001 - shared filename contract
        relative = Path("sequence_sidecars") / row.query.role / f"{safe}.json"
        path = output / relative
        payload = _sequence_payload(row, transform)
        hourly.daily.atomic_json(path, payload)
        records.append({
            "scene_id": row.query.scene_id, "role": row.query.role, "region": row.query.region,
            "city": row.query.city, "item_id": row.query.item_id,
            "source_scene_sha256": row.query.source_scene_sha256,
            "path": relative.as_posix(), "sha256": hourly.daily.sha256_file(path),
        })
    manifest = {
        "schema": SCHEMA, "status": "complete", "scene_count": len(records),
        "token_names": list(TOKEN_NAMES), "lookback_hours": 48,
        "normalization": {"path": "normalization.json", "sha256": hourly.daily.sha256_file(normalization_path)},
        "source_bindings": bindings, "v1_source_reuse_root": str(v1_source_root), "records": records,
        "target_arrays_opened": False, "locked_test_opened": False,
    }
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    hourly.daily.atomic_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v1-source-root", type=Path, default=DEFAULT_V1_SOURCE)
    parser.add_argument("--allow-full-public-build", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, allow_full_public_build=args.allow_full_public_build, v1_source_root=args.v1_source_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
