#!/usr/bin/env python3
"""Build target-free, strictly causal NASA POWER HOURLY G246 sidecars.

The default acquisition role is Fit603.  Only Fit identities and the small
``metadata.npy`` member of their registered NPZ files are used.  Scientific
arrays, target/formal masks, Validation NPZ metadata/targets and the locked-test
descriptor are never opened.  Validation predictor acquisition has a separate
explicit gate and is not authorised by the default CLI.

POWER hourly keys are interpreted conservatively as the *start* of the whole
hour, following the NASA POWER FAQ.  A record keyed by ``YYYYMMDDHH`` covers
``[key, key + 1 hour)`` and is usable only when its interval end is no later
than the Landsat acquisition time ``t0``.  Coordinates, UTC keys and scene
identity remain in private audit artifacts; ``load_model_predictor`` accepts
only public ``sidecars/`` and returns the registered physical feature vector,
units, and non-learnable quality metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import requests

try:
    from . import build_g246_r2_weather_sidecars as daily
except ImportError:  # pragma: no cover - direct script entry point
    import build_g246_r2_weather_sidecars as daily


WORKSPACE = Path(__file__).resolve().parents[1]
POWER_HOURLY_API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = ("ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M")
TAUS_HOURS = (1, 3, 6, 12, 24)
LOOKBACK_HOURS = 48
MIN_COMPLETE_HOURS = 30
MIN_FEATURE_COVERAGE = 0.80
TIME_STANDARD = "UTC"
COMMUNITY = "AG"
MISSING_VALUE_CUTOFF = -900.0
RADIOMETRIC_DAYLIGHT_THRESHOLD_MJ_PER_HOUR = 0.036
RADIOMETRIC_EXPOSURE_CLIP_HOURS = 18.0
RADIOMETRIC_PHASE_FALLBACK_HOURS = 12.0
FIT_SCENE_COUNT = 603
VALIDATION_SCENE_COUNT = 45
MAX_CONCURRENCY = 5
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)

EXPECTED_PROVIDER_UNITS = {
    "ALLSKY_SFC_SW_DWN": "MJ/hr",
    "T2M": "C",
    "RH2M": "%",
    "WS2M": "m/s",
}

RAW_SCHEMA = "uhi-cdc-g246-r2-power-hourly-raw-v1"
SIDECAR_SCHEMA = "uhi-cdc-g246-r2-power-hourly-causal-sidecar-v1"
STATE_SCHEMA = "uhi-cdc-g246-r2-power-hourly-causal-state-v1"
MANIFEST_SCHEMA = "uhi-cdc-g246-r2-power-hourly-causal-manifest-v1"
LOADER_SCHEMA = "uhi-cdc-g246-r2-power-hourly-model-vector-v1"
LAG_TENSOR_SCHEMA = "uhi-cdc-g246-r2-power-hourly-complete-lag-tensor-v1"
TRANSFORM_SCHEMA = "uhi-cdc-g246-r2-power-hourly-train014-transform-v1"
DEFAULT_OUTPUT = WORKSPACE / "artifacts/g246_r2_power_hourly_causal_sidecars_v1"

TIMESTAMP_SEMANTICS = {
    "official_hourly_api_url": (
        "https://power.larc.nasa.gov/docs/services/api/temporal/hourly/"
    ),
    "official_time_faq_url": "https://power.larc.nasa.gov/docs/faqs/other/",
    "provider_documentation": (
        "NASA POWER FAQ: an hourly timestamp represents the start of the "
        "hour for the whole hour"
    ),
    "documentation_reviewed_utc_date": "2026-09-01",
    "documentation_content_hash_frozen": True,
    "documentation_receipt_path": (
        "artifacts/g246_r2/diagnostics/"
        "g246_power_hourly_timestamp_semantics_receipt_20260901.json"
    ),
    "documentation_receipt_sha256": (
        "cc5fb9666ff34444f3926603836599cbd235d7089a92ed07bb7ed0b44dce200e"
    ),
    "bulk_gate": "local content-hash receipt frozen before Fit603 acquisition",
    "interval_interpretation": "half-open [timestamp, timestamp + 1 hour)",
    "causal_rule": "select only intervals with interval_end_utc <= t0_utc",
    "future_or_partial_hour_used": False,
}

FEATURE_NAMES = (
    "hours_since_radiometric_sunrise_clipped18h",
    "normalized_radiometric_solar_day_phase",
    *(f"{parameter}_causal_exp_mean_window{tau}h"
      for parameter in PARAMETERS for tau in TAUS_HOURS),
)
FORBIDDEN_LOADER_TOKENS = (
    "latitude", "longitude", "coord", "city", "region", "scene", "item",
    "identity", "utc_hour", "timestamp", "datetime", "target", "formal",
)


class HourlyWeatherError(RuntimeError):
    """Raised when a causal or provenance contract cannot be proved."""


@dataclass(frozen=True)
class HourlySceneQuery:
    scene_id: str
    role: str
    region: str
    city: str
    item_id: str
    source_scene_sha256: str
    t0_utc: str
    longitude: float
    latitude: float

    @classmethod
    def from_scene(cls, scene: daily.SceneRequest) -> "HourlySceneQuery":
        t0 = parse_utc(scene.datetime)
        return cls(
            scene_id=scene.scene_id, role=scene.role, region=scene.region,
            city=scene.city, item_id=scene.item_id,
            source_scene_sha256=scene.source_scene_sha256,
            t0_utc=t0.isoformat().replace("+00:00", "Z"),
            longitude=float(scene.longitude), latitude=float(scene.latitude),
        )

    @property
    def t0(self) -> datetime:
        return parse_utc(self.t0_utc)

    @property
    def request_start_date(self) -> date:
        return (self.t0 - timedelta(hours=LOOKBACK_HOURS)).date()

    @property
    def request_end_date(self) -> date:
        return self.t0.date()

    def api_params(self) -> dict[str, Any]:
        return {
            "parameters": ",".join(PARAMETERS), "community": COMMUNITY,
            "longitude": self.longitude, "latitude": self.latitude,
            "start": self.request_start_date.strftime("%Y%m%d"),
            "end": self.request_end_date.strftime("%Y%m%d"),
            "format": "JSON", "time-standard": TIME_STANDARD,
        }

    def audit_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "request_start_date": self.request_start_date.isoformat(),
            "request_end_date": self.request_end_date.isoformat(),
            "lookback_hours": LOOKBACK_HOURS,
            "parameters": list(PARAMETERS), "community": COMMUNITY,
            "time_standard": TIME_STANDARD,
        }

    @property
    def query_sha256(self) -> str:
        return daily.canonical_sha256(self.audit_record())


@dataclass(frozen=True)
class HourInterval:
    key: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ModelPredictor:
    """The entire loader return type; deliberately contains no audit identity."""

    schema: str
    names: tuple[str, ...]
    values: np.ndarray
    units: tuple[str, ...]


@dataclass(frozen=True)
class PredictorQuality:
    """Availability metadata, separate from the 22 physical predictor values."""

    feature_coverage: tuple[float, ...]
    feature_valid: tuple[bool, ...]
    parameter_valid_interval_fraction: tuple[float, ...]
    parameter_causal_gap_hours: tuple[float, ...]
    minimum_required_feature_coverage: float


@dataclass(frozen=True)
class PredictorTransform:
    """Frozen train014-only equal-hierarchy standardization."""

    schema: str
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    valid_hierarchy_mass: tuple[float, ...]
    feature_names: tuple[str, ...]
    scope: str
    fitted_scene_count: int
    fitted_city_count: int
    fitted_region_count: int


@dataclass(frozen=True)
class TransformScene:
    """Predictor-only grouping record supplied by a frozen train014 split."""

    sidecar_path: Path
    region: str
    city: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def verify_timestamp_semantics_receipt() -> dict[str, Any]:
    """Verify the frozen source-semantics receipt before any bulk request."""

    relative = Path(str(TIMESTAMP_SEMANTICS["documentation_receipt_path"]))
    path = WORKSPACE / relative
    if not path.is_file() or path.is_symlink():
        raise HourlyWeatherError("timestamp semantics receipt is absent or unsafe")
    expected = str(TIMESTAMP_SEMANTICS["documentation_receipt_sha256"])
    if daily.sha256_file(path) != expected:
        raise HourlyWeatherError("timestamp semantics receipt hash changed")
    payload = daily.read_json(path, "timestamp semantics receipt")
    if payload.get("schema") != "g246-power-hourly-timestamp-semantics-receipt-v1" \
            or payload.get("physical_post_overpass_used") is not False \
            or payload.get("deployment_scope") != "offline_retrospective_only" \
            or payload.get("target_arrays_opened") is not False \
            or payload.get("validation_target_opened") is not False \
            or payload.get("locked_test_opened") is not False:
        raise HourlyWeatherError("timestamp semantics receipt contract differs")
    return payload


def parse_utc(value: str) -> datetime:
    raw = str(value).strip()
    try:
        parsed = datetime.fromisoformat(
            raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        )
    except ValueError as exc:
        raise HourlyWeatherError(f"invalid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        raise HourlyWeatherError("datetime must carry a timezone")
    return parsed.astimezone(timezone.utc)


def parse_hour_key(key: str) -> HourInterval:
    if not re.fullmatch(r"\d{10}", str(key)):
        raise HourlyWeatherError(f"POWER hourly key is not YYYYMMDDHH: {key!r}")
    try:
        start = datetime.strptime(str(key), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HourlyWeatherError(f"POWER hourly key is invalid: {key!r}") from exc
    return HourInterval(str(key), start, start + timedelta(hours=1))


def _finite_or_missing(value: Any, parameter: str, key: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HourlyWeatherError(f"POWER {parameter}/{key} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HourlyWeatherError(f"POWER {parameter}/{key} is nonfinite")
    return None if result <= MISSING_VALUE_CUTOFF else result


def _radiometric_state(
    intervals: Sequence[HourInterval], sw_values: Sequence[float],
) -> dict[str, Any] | None:
    """Return the frozen coordinate-free state from completed past SW only.

    The 10 W/m2 threshold is 0.036 MJ/m2 over one hour in POWER's frozen
    ``MJ/hr`` unit.  A rise requires four consecutive strictly sub-threshold
    bins followed by two consecutive strictly super-threshold bins.  It is
    timestamped conservatively at the *end* of the first bright bin.  No
    coordinate, date, UTC phase, or future forcing enters the state.
    """

    if len(intervals) != len(sw_values):
        raise HourlyWeatherError("radiometric sunrise inputs have unequal length")
    values = np.asarray(sw_values, np.float64)
    threshold = RADIOMETRIC_DAYLIGHT_THRESHOLD_MJ_PER_HOUR
    rise_indices: list[int] = []
    for index in range(4, len(intervals) - 1):
        local = values[index - 4:index + 2]
        if not np.all(np.isfinite(local)):
            continue
        if np.all(local[:4] < threshold) and np.all(local[4:] > threshold):
            rise_indices.append(index)
    if not rise_indices:
        return None
    latest_index = rise_indices[-1]
    latest_sunrise = intervals[latest_index].end
    previous_episode_duration: float | None = None
    previous_episode_start: datetime | None = None
    previous_episode_end: datetime | None = None
    for rise_index in reversed(rise_indices[:-1]):
        end_index: int | None = None
        for index in range(rise_index + 2, latest_index):
            if not math.isfinite(values[index]):
                break
            if values[index] < threshold:
                end_index = index
                break
        if end_index is None:
            continue
        episode_start = intervals[rise_index].end
        episode_end = intervals[end_index].start
        duration = (episode_end - episode_start).total_seconds() / 3600.0
        if 0.0 < duration <= 24.0:
            previous_episode_duration = duration
            previous_episode_start = episode_start
            previous_episode_end = episode_end
            break
    return {
        "latest_sunrise": latest_sunrise,
        "previous_episode_duration_hours": previous_episode_duration,
        "previous_episode_start": previous_episode_start,
        "previous_episode_end": previous_episode_end,
        "rise_count": len(rise_indices),
    }


def fetch_power_hourly(
    query: HourlySceneQuery, *, get: Callable[..., Any] = requests.get,
    sleep: Callable[[float], None] = time.sleep,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    attempts = len(retry_delays) + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        status: int | None = None
        try:
            response = get(
                POWER_HOURLY_API, params=query.api_params(),
                headers={"User-Agent": "uhi-cdc-g246-r2-hourly-causal/1.0"},
                timeout=float(timeout_seconds),
            )
            status = int(getattr(response, "status_code", 200))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("POWER response root is not an object")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status in {408, 425, 429} or status >= 500
            if not retryable or attempt == attempts - 1:
                break
            delay = float(retry_delays[attempt])
            if not math.isfinite(delay) or delay < 0:
                raise HourlyWeatherError("retry delay is invalid")
            sleep(delay)
    raise HourlyWeatherError(
        f"NASA POWER hourly request failed for {query.scene_id} after {attempts} attempts"
    ) from last_error


def _response_blocks(
    response: Mapping[str, Any], query: HourlySceneQuery,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], Mapping[str, Any]]:
    properties = response.get("properties")
    series = properties.get("parameter") if isinstance(properties, Mapping) else None
    descriptors = response.get("parameters")
    header = response.get("header")
    if not isinstance(series, Mapping) or not isinstance(descriptors, Mapping) \
            or not isinstance(header, Mapping):
        raise HourlyWeatherError("POWER response lacks hourly parameter metadata")
    if str(header.get("time_standard", "")).upper() != TIME_STANDARD:
        raise HourlyWeatherError("POWER response is not explicitly UTC")
    units: dict[str, str] = {}
    selected: dict[str, Mapping[str, Any]] = {}
    key_set: set[str] | None = None
    for parameter in PARAMETERS:
        values, descriptor = series.get(parameter), descriptors.get(parameter)
        if not isinstance(values, Mapping) or not isinstance(descriptor, Mapping):
            raise HourlyWeatherError(f"POWER response lacks {parameter}")
        unit = descriptor.get("units")
        if not isinstance(unit, str) or not unit.strip():
            raise HourlyWeatherError(f"POWER response lacks units for {parameter}")
        if unit.strip() != EXPECTED_PROVIDER_UNITS[parameter]:
            raise HourlyWeatherError(
                f"POWER {parameter} units changed: expected "
                f"{EXPECTED_PROVIDER_UNITS[parameter]!r}, got {unit.strip()!r}"
            )
        keys = {str(key) for key in values}
        if key_set is None:
            key_set = keys
        elif keys != key_set:
            raise HourlyWeatherError("POWER parameters have unequal hourly keys")
        units[parameter] = unit.strip()
        selected[parameter] = values
    if not key_set:
        raise HourlyWeatherError("POWER response has no hourly values")
    start = query.request_start_date.strftime("%Y%m%d")
    end = query.request_end_date.strftime("%Y%m%d")
    if str(header.get("start")) != start or str(header.get("end")) != end:
        raise HourlyWeatherError("POWER response request dates differ")
    return selected, units, header


def derive_causal_predictor(
    response: Mapping[str, Any], query: HourlySceneQuery,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    series, units, header = _response_blocks(response, query)
    t0 = query.t0
    response_intervals = sorted(
        (parse_hour_key(key) for key in series[PARAMETERS[0]]),
        key=lambda row: row.start,
    )
    # The latest *nominal* whole-hour boundary is determined by t0, never by
    # provider key presence.  Otherwise a missing latest key would silently
    # shift every finite window one hour into the past and masquerade as fresh.
    t_c = t0.replace(minute=0, second=0, microsecond=0)
    lookback_start = t_c - timedelta(hours=LOOKBACK_HOURS)
    response_by_start = {
        interval.start: interval for interval in response_intervals
        if lookback_start <= interval.start < t_c
    }
    expected_intervals = tuple(
        HourInterval(
            key=(lookback_start + timedelta(hours=offset)).strftime("%Y%m%d%H"),
            start=lookback_start + timedelta(hours=offset),
            end=lookback_start + timedelta(hours=offset + 1),
        )
        for offset in range(LOOKBACK_HOURS)
    )
    present_count = sum(interval.start in response_by_start for interval in expected_intervals)
    if present_count < MIN_COMPLETE_HOURS:
        raise HourlyWeatherError(
            f"only {present_count} complete causal hourly keys; "
            f"require {MIN_COMPLETE_HOURS}"
        )
    matrix = np.full((len(PARAMETERS), LOOKBACK_HOURS), np.nan, np.float64)
    for column, expected in enumerate(expected_intervals):
        interval = response_by_start.get(expected.start)
        if interval is None:
            continue
        if interval.key != expected.key or interval.end != expected.end:
            raise HourlyWeatherError("POWER hourly cadence is not an exact UTC hour")
        for row, parameter in enumerate(PARAMETERS):
            value = _finite_or_missing(series[parameter][interval.key], parameter, interval.key)
            if value is not None:
                matrix[row, column] = value
    valid_by_parameter = np.isfinite(matrix)
    if np.any(valid_by_parameter.sum(axis=1) == 0):
        missing = [parameter for parameter, valid in zip(
            PARAMETERS, valid_by_parameter.sum(axis=1)
        ) if valid == 0]
        raise HourlyWeatherError(f"no valid complete intervals for {missing}")
    parameter_causal_gap = []
    parameter_valid_fraction = []
    latest_valid_end_by_parameter: dict[str, str] = {}
    for row, parameter in enumerate(PARAMETERS):
        valid_columns = np.flatnonzero(valid_by_parameter[row])
        latest_valid_end = max(
            expected_intervals[int(column)].end for column in valid_columns
        )
        gap = (t0 - latest_valid_end).total_seconds() / 3600.0
        if gap < 0:
            raise HourlyWeatherError("negative causal gap would imply future leakage")
        parameter_causal_gap.append(gap)
        parameter_valid_fraction.append(float(np.mean(valid_by_parameter[row])))
        latest_valid_end_by_parameter[parameter] = latest_valid_end.isoformat()

    sw_row = PARAMETERS.index("ALLSKY_SFC_SW_DWN")
    sw_coverage_48h = float(np.mean(valid_by_parameter[sw_row]))
    radiometric_state = _radiometric_state(
        expected_intervals, matrix[sw_row],
    )
    sunrise_valid = False
    sunrise_reason = "no_confirmed_four_dark_two_bright_transition"
    latest_sunrise: datetime | None = None
    previous_episode_start: datetime | None = None
    previous_episode_end: datetime | None = None
    previous_episode_duration: float | None = None
    phase_denominator_hours = RADIOMETRIC_PHASE_FALLBACK_HOURS
    phase_fallback_used = True
    hours_since_sunrise = 0.0
    solar_day_phase = 0.0
    rise_count = 0
    if radiometric_state is not None:
        latest_sunrise = radiometric_state["latest_sunrise"]
        previous_episode_start = radiometric_state["previous_episode_start"]
        previous_episode_end = radiometric_state["previous_episode_end"]
        previous_episode_duration = radiometric_state[
            "previous_episode_duration_hours"
        ]
        rise_count = int(radiometric_state["rise_count"])
        raw_exposure = (t_c - latest_sunrise).total_seconds() / 3600.0
        if sw_coverage_48h < MIN_FEATURE_COVERAGE:
            sunrise_reason = "shortwave_48h_coverage_below_threshold"
        elif raw_exposure < 0:
            sunrise_reason = "confirmed_transition_ends_after_t_c"
        else:
            sunrise_valid = True
            sunrise_reason = "valid"
            hours_since_sunrise = min(
                raw_exposure, RADIOMETRIC_EXPOSURE_CLIP_HOURS,
            )
            if previous_episode_duration is not None:
                phase_denominator_hours = previous_episode_duration
                phase_fallback_used = False
            solar_day_phase = min(
                1.0, max(0.0, hours_since_sunrise / phase_denominator_hours),
            )

    features: list[float] = [hours_since_sunrise, solar_day_phase]
    feature_units: list[str] = ["h", "1"]
    feature_coverage: list[float] = [sw_coverage_48h, sw_coverage_48h]
    feature_valid: list[bool] = [sunrise_valid, sunrise_valid]
    kernel_receipt: dict[str, Any] = {}
    for parameter_row, parameter in enumerate(PARAMETERS):
        for horizon in TAUS_HOURS:
            window_start = t_c - timedelta(hours=horizon)
            rho_hours = horizon / 2.0
            full_mass = float(rho_hours * (1.0 - math.exp(-horizon / rho_hours)))
            observed_mass = 0.0
            weighted_sum = 0.0
            expected_slot_count = 0
            observed_slot_count = 0
            for column, interval in enumerate(expected_intervals):
                if interval.start < window_start or interval.end > t_c:
                    continue
                expected_slot_count += 1
                age_start = (interval.start - t_c).total_seconds() / 3600.0
                age_end = (interval.end - t_c).total_seconds() / 3600.0
                weight = rho_hours * (
                    math.exp(age_end / rho_hours) - math.exp(age_start / rho_hours)
                )
                if weight <= 0 or not math.isfinite(weight):
                    raise HourlyWeatherError("invalid causal exponential kernel weight")
                if valid_by_parameter[parameter_row, column]:
                    observed_slot_count += 1
                    observed_mass += weight
                    weighted_sum += weight * matrix[parameter_row, column]
            if expected_slot_count != horizon:
                raise HourlyWeatherError("completed-window hourly geometry differs")
            if not full_mass > 0:
                raise HourlyWeatherError("causal exponential kernel has zero full mass")
            coverage = min(1.0, max(0.0, observed_mass / full_mass))
            is_valid = observed_mass > 0 and coverage >= MIN_FEATURE_COVERAGE
            # The physical feature is the finite-window exponential integral
            # divided by the parameter-specific observed kernel mass.  Missing
            # values are therefore never silently treated as physical zero.
            causal_mean = float(weighted_sum / observed_mass) if is_valid else 0.0
            features.append(causal_mean)
            feature_units.append(units[parameter])
            feature_coverage.append(coverage)
            feature_valid.append(is_valid)
            kernel_receipt[f"{parameter}_window{horizon}h"] = {
                "window_start_utc": window_start.isoformat(),
                "window_end_t_c_utc": t_c.isoformat(),
                "exponential_scale_rho_hours": rho_hours,
                "expected_hourly_slot_count": expected_slot_count,
                "observed_parameter_slot_count": observed_slot_count,
                "full_expected_kernel_mass_hours": full_mass,
                "observed_parameter_kernel_mass": observed_mass,
                "coverage_fraction": coverage,
                "normalization": (
                    "finite-window exponential integral divided by this "
                    "parameter's observed kernel mass; no dataset standardization"
                ),
                "invalid_value_policy": "zero placeholder; runner must use no-dynamic-correction fallback",
            }
    vector = np.asarray(features, np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise HourlyWeatherError("derived model vector is malformed or nonfinite")
    if len(feature_coverage) != len(FEATURE_NAMES) or len(feature_valid) != len(FEATURE_NAMES):
        raise HourlyWeatherError("feature quality geometry differs")
    model_payload = {
        "names": list(FEATURE_NAMES), "values": vector.tolist(),
        "units": feature_units,
    }
    quality_payload = {
        "feature_coverage": feature_coverage,
        "feature_valid": feature_valid,
        "parameter_order": list(PARAMETERS),
        "parameter_valid_interval_fraction": parameter_valid_fraction,
        "parameter_causal_gap_hours": parameter_causal_gap,
        "minimum_required_feature_coverage": MIN_FEATURE_COVERAGE,
        "all_features_valid": all(feature_valid),
        "raw_vector_direct_forward_authorized": False,
        "invalid_value_policy": (
            "frozen train014 transform maps invalid dimensions to standardized zero"
        ),
        "quality_is_not_part_of_physical_vector": True,
    }
    lag_tensor_payload = {
        "schema": LAG_TENSOR_SCHEMA,
        "parameter_order": list(PARAMETERS),
        "values_missing_filled_zero": np.where(
            valid_by_parameter, matrix, 0.0,
        ).astype(np.float64).tolist(),
        "valid_mask": valid_by_parameter.tolist(),
        "lag_interval_end_hours_before_t_c": list(
            range(LOOKBACK_HOURS - 1, -1, -1)
        ),
        "units": [units[parameter] for parameter in PARAMETERS],
        "parameter_causal_gap_hours_before_t0": parameter_causal_gap,
        "state_increment_valid": bool(np.all(valid_by_parameter)),
        "invalid_state_policy": "disable only the 4x48 state increment",
        "absolute_time_coordinates_identity_exposed": False,
        "not_loaded_by_22d_model_loader": True,
    }
    audit = {
        "timestamp_semantics": dict(TIMESTAMP_SEMANTICS),
        "requested_header": {
            "start": header.get("start"), "end": header.get("end"),
            "time_standard": header.get("time_standard"),
            "api": header.get("api"), "sources": header.get("sources"),
            "fill_value": header.get("fill_value"),
        },
        "provider_units": units,
        "complete_candidate_interval_count": present_count,
        "expected_lookback_interval_count": LOOKBACK_HOURS,
        "missing_hourly_key_count": LOOKBACK_HOURS - present_count,
        "valid_interval_count_by_parameter": {
            parameter: int(valid_by_parameter[row].sum())
            for row, parameter in enumerate(PARAMETERS)
        },
        "first_selected_interval_start_utc": lookback_start.isoformat(),
        "last_selected_interval_end_utc": t_c.isoformat(),
        "t_c_latest_complete_interval_end_utc": t_c.isoformat(),
        "latest_valid_interval_end_utc_by_parameter": latest_valid_end_by_parameter,
        "t0_utc": t0.isoformat(),
        "parameter_causal_gap_hours": dict(zip(PARAMETERS, parameter_causal_gap)),
        "radiometric_sunrise_detection": {
            "source": "past_completed_ALLSKY_SFC_SW_DWN_only",
            "threshold_mj_per_hour": RADIOMETRIC_DAYLIGHT_THRESHOLD_MJ_PER_HOUR,
            "threshold_physical_origin": "10 W/m2 integrated over one hour",
            "rise_rule": "four valid bins < threshold then two valid bins > threshold",
            "rise_timestamp_rule": "right endpoint of first bright bin",
            "confirmed_rise_count": rise_count,
            "previous_completed_episode_start_utc": (
                previous_episode_start.isoformat()
                if previous_episode_start is not None else None
            ),
            "previous_completed_episode_end_utc": (
                previous_episode_end.isoformat()
                if previous_episode_end is not None else None
            ),
            "previous_completed_episode_duration_hours": previous_episode_duration,
            "phase_denominator_hours": phase_denominator_hours,
            "phase_fixed_12h_fallback_used": phase_fallback_used,
            "elapsed_exposure_clip_hours": RADIOMETRIC_EXPOSURE_CLIP_HOURS,
            "latest_sunrise_utc": (
                latest_sunrise.isoformat() if latest_sunrise is not None else None
            ),
            "valid": sunrise_valid,
            "reason": sunrise_reason,
            "coordinates_or_calendar_used": False,
            "future_forcing_used": False,
        },
        "retrospective_batch_deployment_only": True,
        "near_real_time_or_operational_forecast_claim": False,
        "dual_time_axis": {
            "physical_causality": "every used interval_end_utc <= t0_utc",
            "deployment_availability": (
                "provider response must be acquired and cached by retrospective t_dep"
            ),
            "provider_publication_by_original_t0_required": False,
        },
        "kernel_receipt": kernel_receipt,
        "partial_or_future_interval_used": False,
        "raw_response_future_interval_count_ignored": sum(
            interval.end > t0 for interval in response_intervals
        ),
    }
    return model_payload, quality_payload, lag_tensor_payload, audit


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "_", value.casefold()).strip("._-")
    if not slug:
        raise HourlyWeatherError("cannot derive safe artifact filename")
    return slug


def _artifact_stem(query: HourlySceneQuery) -> str:
    suffix = hashlib.sha256(query.scene_id.encode()).hexdigest()[:12]
    return f"{_safe_slug(query.city)}_{query.t0.strftime('%Y%m%dT%H%M%S')}_{suffix}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    daily.atomic_json(path, payload)


def _sidecar_payload(
    query: HourlySceneQuery, model: Mapping[str, Any], quality: Mapping[str, Any],
    lag_tensor: Mapping[str, Any], audit_sha256: str,
    response_canonical_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SIDECAR_SCHEMA,
        "scene_binding": {
            "scene_id": query.scene_id, "role": query.role,
            "source_scene_sha256": query.source_scene_sha256,
        },
        "model_predictor": dict(model),
        "predictor_quality": dict(quality),
        "sanitized_complete_lag_tensor": dict(lag_tensor),
        "private_provenance_digest": {
            "audit_sha256": audit_sha256,
            "response_canonical_sha256": response_canonical_sha256,
            "query_sha256": query.query_sha256,
        },
        "private_provenance_path_exposed": False,
        "target_free": True, "validation_target_opened": False,
        "locked_test_opened": False,
    }


def validate_sidecar(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SIDECAR_SCHEMA \
            or payload.get("target_free") is not True \
            or payload.get("validation_target_opened") is not False \
            or payload.get("locked_test_opened") is not False:
        raise HourlyWeatherError("hourly causal sidecar contract differs")
    model = payload.get("model_predictor")
    quality = payload.get("predictor_quality")
    lag_tensor = payload.get("sanitized_complete_lag_tensor")
    if not isinstance(model, Mapping) or model.get("names") != list(FEATURE_NAMES):
        raise HourlyWeatherError("sidecar model predictor names differ")
    if not isinstance(quality, Mapping) \
            or quality.get("parameter_order") != list(PARAMETERS) \
            or quality.get("quality_is_not_part_of_physical_vector") is not True \
            or quality.get("raw_vector_direct_forward_authorized") is not False:
        raise HourlyWeatherError("sidecar predictor quality differs")
    if not isinstance(lag_tensor, Mapping) \
            or lag_tensor.get("schema") != LAG_TENSOR_SCHEMA \
            or lag_tensor.get("parameter_order") != list(PARAMETERS) \
            or lag_tensor.get("absolute_time_coordinates_identity_exposed") is not False \
            or lag_tensor.get("not_loaded_by_22d_model_loader") is not True:
        raise HourlyWeatherError("sidecar sanitized lag tensor contract differs")
    names, values, units = model.get("names"), model.get("values"), model.get("units")
    if not isinstance(names, list) or not isinstance(values, list) or not isinstance(units, list) \
            or len(values) != len(FEATURE_NAMES) or len(units) != len(FEATURE_NAMES):
        raise HourlyWeatherError("sidecar model vector geometry differs")
    if any(any(token in str(name).casefold() for token in FORBIDDEN_LOADER_TOKENS)
           for name in names):
        raise HourlyWeatherError("model feature name exposes forbidden identity/time data")
    vector = np.asarray(values, np.float64)
    if not np.all(np.isfinite(vector)):
        raise HourlyWeatherError("sidecar model vector is nonfinite")
    coverage = np.asarray(quality.get("feature_coverage"), np.float64)
    valid = np.asarray(quality.get("feature_valid"), bool)
    parameter_fraction = np.asarray(
        quality.get("parameter_valid_interval_fraction"), np.float64,
    )
    parameter_gap = np.asarray(quality.get("parameter_causal_gap_hours"), np.float64)
    if coverage.shape != (len(FEATURE_NAMES),) or valid.shape != coverage.shape \
            or parameter_fraction.shape != (len(PARAMETERS),) \
            or parameter_gap.shape != (len(PARAMETERS),) \
            or np.any(~np.isfinite(coverage)) or np.any(~np.isfinite(parameter_fraction)) \
            or np.any(~np.isfinite(parameter_gap)) or np.any((coverage < 0) | (coverage > 1)) \
            or np.any((parameter_fraction < 0) | (parameter_fraction > 1)) \
            or np.any(parameter_gap < 0):
        raise HourlyWeatherError("sidecar predictor quality geometry/value differs")
    if np.any(vector[~valid] != 0.0):
        raise HourlyWeatherError(
            "invalid raw predictor dimensions must be zero placeholders"
        )
    if payload.get("private_provenance_path_exposed") is not False:
        raise HourlyWeatherError("sidecar exposes a private provenance path")
    lag_values = np.asarray(lag_tensor.get("values_missing_filled_zero"), np.float64)
    lag_valid = np.asarray(lag_tensor.get("valid_mask"), bool)
    lag_hours = np.asarray(lag_tensor.get("lag_interval_end_hours_before_t_c"), int)
    if lag_values.shape != (len(PARAMETERS), LOOKBACK_HOURS) \
            or lag_valid.shape != lag_values.shape \
            or lag_hours.shape != (LOOKBACK_HOURS,) \
            or not np.array_equal(lag_hours, np.arange(LOOKBACK_HOURS - 1, -1, -1)) \
            or np.any(~np.isfinite(lag_values)):
        raise HourlyWeatherError("sidecar sanitized lag tensor geometry differs")


def _read_validated_sidecar(path: Path) -> dict[str, Any]:
    guarded = daily._reject_forbidden_path(path, "hourly causal sidecar")  # noqa: SLF001
    if "sidecars" not in guarded.resolve().parts:
        raise HourlyWeatherError("model loader accepts only a sidecars/ artifact")
    payload = daily.read_json(guarded, "hourly causal sidecar")
    validate_sidecar(payload)
    return payload


def audit_predictor_quality(path: Path) -> PredictorQuality:
    """Return non-learnable route quality for a pre-forward barrier."""

    payload = _read_validated_sidecar(path)
    quality = payload["predictor_quality"]
    return PredictorQuality(
        feature_coverage=tuple(float(value) for value in quality["feature_coverage"]),
        feature_valid=tuple(bool(value) for value in quality["feature_valid"]),
        parameter_valid_interval_fraction=tuple(
            float(value) for value in quality["parameter_valid_interval_fraction"]
        ),
        parameter_causal_gap_hours=tuple(
            float(value) for value in quality["parameter_causal_gap_hours"]
        ),
        minimum_required_feature_coverage=float(
            quality["minimum_required_feature_coverage"]
        ),
    )


def fit_predictor_transform(
    scenes: Sequence[TransformScene],
) -> PredictorTransform:
    """Fit the frozen train014 predictor-only equal-hierarchy transform.

    The caller must supply only the preregistered train014 scene set.  Region,
    city, and scene identities determine weights here but are never retained in
    or returned by the model-facing transform.
    """

    if not scenes:
        raise HourlyWeatherError("train014 transform has no scenes")
    grouped: dict[str, dict[str, list[TransformScene]]] = {}
    seen_scene_ids: set[str] = set()
    rows: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    ordered_scenes = sorted(
        scenes, key=lambda scene: (scene.region, scene.city, str(scene.sidecar_path)),
    )
    for scene in ordered_scenes:
        region = daily.g246_data.canonical_region(scene.region)
        city = str(scene.city).strip()
        if not city:
            raise HourlyWeatherError("train014 transform has an empty city")
        payload = _read_validated_sidecar(scene.sidecar_path)
        binding = payload.get("scene_binding")
        if not isinstance(binding, Mapping) or binding.get("role") != "fit":
            raise HourlyWeatherError("train014 transform accepts fit sidecars only")
        scene_id = str(binding.get("scene_id", ""))
        if not scene_id or scene_id in seen_scene_ids:
            raise HourlyWeatherError("train014 transform scene identity is duplicate/empty")
        seen_scene_ids.add(scene_id)
        grouped.setdefault(region, {}).setdefault(city, []).append(scene)
        rows.append(np.asarray(payload["model_predictor"]["values"], np.float64))
        valids.append(np.asarray(payload["predictor_quality"]["feature_valid"], bool))
    regions = sorted(grouped)
    weights_by_path: dict[Path, float] = {}
    for region in regions:
        cities = grouped[region]
        for city in sorted(cities):
            city_scenes = cities[city]
            weight = 1.0 / (
                len(regions) * len(cities) * len(city_scenes)
            )
            for scene in city_scenes:
                weights_by_path[scene.sidecar_path.resolve()] = weight
    weights = np.asarray(
        [weights_by_path[scene.sidecar_path.resolve()] for scene in ordered_scenes],
        np.float64,
    )
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise HourlyWeatherError("equal-region/city/scene weights do not sum to one")
    matrix = np.stack(rows)
    valid_matrix = np.stack(valids)
    means = np.empty(len(FEATURE_NAMES), np.float64)
    standard_deviations = np.empty(len(FEATURE_NAMES), np.float64)
    valid_mass = np.empty(len(FEATURE_NAMES), np.float64)
    for index, name in enumerate(FEATURE_NAMES):
        selected = valid_matrix[:, index]
        mass = float(weights[selected].sum())
        valid_mass[index] = mass
        if mass < MIN_FEATURE_COVERAGE:
            raise HourlyWeatherError(
                f"train014 valid hierarchy mass for {name} is {mass:.6f} < "
                f"{MIN_FEATURE_COVERAGE:.2f}"
            )
        conditional_weights = weights[selected] / mass
        values = matrix[selected, index]
        mean = float(np.dot(conditional_weights, values))
        variance = float(np.dot(conditional_weights, np.square(values - mean)))
        scale = max(1.0, abs(mean))
        if not math.isfinite(variance) or variance <= np.finfo(np.float64).eps * scale * scale:
            raise HourlyWeatherError(f"train014 standard deviation degenerates for {name}")
        means[index] = mean
        standard_deviations[index] = math.sqrt(variance)
    return PredictorTransform(
        schema=TRANSFORM_SCHEMA,
        means=tuple(float(value) for value in means),
        standard_deviations=tuple(float(value) for value in standard_deviations),
        valid_hierarchy_mass=tuple(float(value) for value in valid_mass),
        feature_names=tuple(FEATURE_NAMES),
        scope="train014_predictor_only",
        fitted_scene_count=len(ordered_scenes),
        fitted_city_count=sum(len(cities) for cities in grouped.values()),
        fitted_region_count=len(grouped),
    )


def predictor_transform_payload(transform: PredictorTransform) -> dict[str, Any]:
    return {
        "schema": transform.schema,
        "scope": transform.scope,
        "feature_names": list(transform.feature_names),
        "means": list(transform.means),
        "standard_deviations": list(transform.standard_deviations),
        "valid_hierarchy_mass": list(transform.valid_hierarchy_mass),
        "fitted_scene_count": transform.fitted_scene_count,
        "fitted_city_count": transform.fitted_city_count,
        "fitted_region_count": transform.fitted_region_count,
        "weighting": "equal-region -> equal-city -> equal-scene",
        "invalid_dimension_forward_value": 0.0,
        "quality_or_identity_returned_to_model": False,
        "target_arrays_opened": False,
        "validation_target_opened": False,
        "locked_test_opened": False,
    }


def load_model_predictor(
    path: Path, *, transform: PredictorTransform,
) -> ModelPredictor:
    """Load exactly 22 standardized values, never quality or identity."""

    payload = _read_validated_sidecar(path)
    if transform.schema != TRANSFORM_SCHEMA \
            or transform.scope != "train014_predictor_only" \
            or transform.feature_names != tuple(FEATURE_NAMES):
        raise HourlyWeatherError("predictor transform contract differs")
    means = np.asarray(transform.means, np.float64)
    standard_deviations = np.asarray(transform.standard_deviations, np.float64)
    if means.shape != (len(FEATURE_NAMES),) \
            or standard_deviations.shape != means.shape \
            or np.any(~np.isfinite(means)) \
            or np.any(~np.isfinite(standard_deviations)) \
            or np.any(standard_deviations <= 0):
        raise HourlyWeatherError("predictor transform values differ")
    model = payload["model_predictor"]
    raw_values = np.asarray(model["values"], np.float64)
    valid = np.asarray(payload["predictor_quality"]["feature_valid"], bool)
    standardized = np.zeros(len(FEATURE_NAMES), np.float64)
    standardized[valid] = (
        raw_values[valid] - means[valid]
    ) / standard_deviations[valid]
    result = ModelPredictor(
        schema=LOADER_SCHEMA, names=tuple(model["names"]),
        values=standardized.astype(np.float32),
        units=tuple("standardized_1" for _ in FEATURE_NAMES),
    )
    if result.values.shape != (len(FEATURE_NAMES),):
        raise HourlyWeatherError("loader vector geometry differs")
    return result


def process_one(
    query: HourlySceneQuery, output: Path,
    *, fetcher: Callable[[HourlySceneQuery], Mapping[str, Any]] = fetch_power_hourly,
) -> dict[str, Any]:
    stem = _artifact_stem(query)
    raw_relative = Path("private_provenance/raw") / f"{stem}.json"
    audit_relative = Path("private_provenance/audit") / f"{stem}.json"
    sidecar_relative = Path("sidecars") / query.role / f"{stem}.json"
    raw_path, audit_path, sidecar_path = (
        output / raw_relative, output / audit_relative, output / sidecar_relative,
    )
    if sidecar_path.is_file() and audit_path.is_file() and raw_path.is_file():
        sidecar = daily.read_json(sidecar_path, "existing hourly causal sidecar")
        validate_sidecar(sidecar)
        audit_sha = daily.sha256_file(audit_path)
        raw_sha = daily.sha256_file(raw_path)
        if sidecar["private_provenance_digest"]["audit_sha256"] != audit_sha:
            raise HourlyWeatherError("existing audit binding changed")
        return {
            "scene_id": query.scene_id, "role": query.role,
            "sidecar_path": sidecar_relative.as_posix(),
            "sidecar_sha256": daily.sha256_file(sidecar_path),
            "audit_path": audit_relative.as_posix(), "audit_sha256": audit_sha,
            "raw_path": raw_relative.as_posix(), "raw_sha256": raw_sha,
            "network_request_issued": False,
        }
    if any(path.exists() or path.is_symlink() for path in (raw_path, audit_path, sidecar_path)):
        raise HourlyWeatherError("incomplete existing scene transaction requires manual audit")
    response = fetcher(query)
    raw_payload = {
        "schema": RAW_SCHEMA, "provider": "NASA POWER",
        "endpoint": POWER_HOURLY_API, "query": query.audit_record(),
        "query_sha256": query.query_sha256,
        "response": dict(response),
        "response_canonical_sha256": daily.canonical_sha256(response),
        "fetched_utc": utc_now(), "canonicalized_json_cache": True,
        "target_arrays_opened": False, "validation_target_opened": False,
        "locked_test_opened": False,
    }
    _atomic_json(raw_path, raw_payload)
    raw_sha = daily.sha256_file(raw_path)
    model, quality, lag_tensor, interval_audit = derive_causal_predictor(
        response, query,
    )
    audit_payload = {
        "schema": "uhi-cdc-g246-r2-power-hourly-causal-audit-v1",
        "provider": "NASA POWER", "endpoint": POWER_HOURLY_API,
        "query": query.audit_record(), "query_sha256": query.query_sha256,
        "raw_cache": {"path": raw_relative.as_posix(), "sha256": raw_sha},
        "interval_and_derivation": interval_audit,
        "predictor_quality": quality,
        "model_feature_names": list(FEATURE_NAMES),
        "coordinates_or_raw_time_exposed_by_loader": False,
        "target_arrays_opened": False, "validation_target_opened": False,
        "locked_test_opened": False,
    }
    _atomic_json(audit_path, audit_payload)
    audit_sha = daily.sha256_file(audit_path)
    sidecar = _sidecar_payload(
        query, model, quality, lag_tensor, audit_sha,
        str(raw_payload["response_canonical_sha256"]),
    )
    validate_sidecar(sidecar)
    _atomic_json(sidecar_path, sidecar)
    return {
        "scene_id": query.scene_id, "role": query.role,
        "sidecar_path": sidecar_relative.as_posix(),
        "sidecar_sha256": daily.sha256_file(sidecar_path),
        "audit_path": audit_relative.as_posix(), "audit_sha256": audit_sha,
        "raw_path": raw_relative.as_posix(), "raw_sha256": raw_sha,
        "network_request_issued": True,
    }


def collect_queries(
    split_receipt: Path = daily.DEFAULT_SPLIT_RECEIPT,
    *, role: str = "fit", allow_validation_predictors: bool = False,
) -> tuple[dict[str, Any], tuple[HourlySceneQuery, ...]]:
    selected_role = str(role).strip().casefold()
    if selected_role not in {"fit", "validation"}:
        raise HourlyWeatherError("hourly acquisition role must be fit or validation")
    if selected_role == "validation" and not allow_validation_predictors:
        raise HourlyWeatherError(
            "Validation predictor acquisition requires an explicit post-gate permission"
        )
    guarded = daily._reject_forbidden_path(  # noqa: SLF001
        split_receipt, "split receipt",
    )
    # The campaign loader verifies both public view descriptors and the common
    # source manifest, but only the selected role's NPZ metadata member is
    # opened below.  No scientific member or target is loaded.
    splits = daily.g246_data.load_splits(guarded, role="fit+validation")
    entries = splits.fit if selected_role == "fit" else splits.validation
    expected_count = (
        FIT_SCENE_COUNT if selected_role == "fit" else VALIDATION_SCENE_COUNT
    )
    if len(entries) != expected_count:
        raise HourlyWeatherError(
            f"{selected_role} view has {len(entries)} scenes; expected {expected_count}"
        )
    requests_: list[daily.SceneRequest] = []
    for entry in entries:
        if entry.view_role != selected_role:
            raise HourlyWeatherError("selected view contains another role")
        longitude, latitude = daily.read_registered_location(entry)
        requests_.append(daily.SceneRequest(
            city=entry.city, year=entry.year,
            region=daily.g246_data.canonical_region(entry.region),
            role=entry.view_role, item_id=entry.item_id,
            datetime=entry.datetime, date=daily._scene_date(entry.datetime),  # noqa: SLF001
            longitude=longitude, latitude=latitude,
            source_scene_sha256=entry.sha256,
        ))
    requests_.sort(
        key=lambda item: (item.region, item.city, item.date, item.item_id),
    )
    queries = tuple(HourlySceneQuery.from_scene(scene) for scene in requests_)
    if len({query.scene_id for query in queries}) != expected_count:
        raise HourlyWeatherError(
            f"{selected_role} query inventory must contain {expected_count} unique scenes"
        )
    public = splits.public_record()
    binding = {
        "split_receipt": {
            "path": public["receipt_path"], "sha256": public["receipt_sha256"],
        },
        "source_manifest": {
            "path": public["source_manifest"],
            "sha256": public["source_manifest_sha256"],
        },
        "campaign_id": public["campaign_id"],
        "selected_role": selected_role,
        "selected_view_sha256": (
            public["fit_view_sha256"] if selected_role == "fit"
            else public["validation_view_sha256"]
        ),
        "selected_scene_count": len(queries),
        "expected_selected_scene_count": expected_count,
        "view_descriptors_verified": ["fit", "validation"],
        "npz_metadata_role_opened": selected_role,
        "validation_npz_metadata_opened": selected_role == "validation",
        "validation_predictor_acquisition_explicitly_authorized": (
            selected_role == "validation" and allow_validation_predictors
        ),
        "target_arrays_opened": False, "validation_target_opened": False,
        "locked_test_opened": False,
    }
    return binding, queries


def build_partial(
    *, output: Path, split_receipt: Path = daily.DEFAULT_SPLIT_RECEIPT,
    role: str = "fit", max_new_scenes: int = 1,
    allow_bulk_acquisition: bool = False,
    allow_full_role: bool = False,
    allow_validation_predictors: bool = False,
    fetcher: Callable[[HourlySceneQuery], Mapping[str, Any]] = fetch_power_hourly,
) -> dict[str, Any]:
    if max_new_scenes <= 0:
        raise HourlyWeatherError("max_new_scenes must be positive")
    selected_role = str(role).strip().casefold()
    expected_count = (
        FIT_SCENE_COUNT if selected_role == "fit" else VALIDATION_SCENE_COUNT
    )
    if selected_role not in {"fit", "validation"}:
        raise HourlyWeatherError("hourly acquisition role must be fit or validation")
    if max_new_scenes > 1 and not allow_bulk_acquisition:
        raise HourlyWeatherError("multi-scene bulk acquisition requires explicit approval")
    if max_new_scenes > 1 \
            and TIMESTAMP_SEMANTICS["documentation_content_hash_frozen"] is not True:
        raise HourlyWeatherError(
            "bulk acquisition is blocked until timestamp semantics are hash-frozen"
        )
    if max_new_scenes > 1:
        verify_timestamp_semantics_receipt()
    if max_new_scenes >= expected_count and not allow_full_role:
        raise HourlyWeatherError(
            f"full {selected_role}{expected_count} acquisition requires explicit approval"
        )
    if max_new_scenes >= expected_count \
            and TIMESTAMP_SEMANTICS["documentation_content_hash_frozen"] is not True:
        raise HourlyWeatherError(
            "bulk acquisition is blocked until the official timestamp semantics "
            "document has a frozen local content-hash receipt"
        )
    output = daily._reject_forbidden_path(output, "hourly causal output").resolve()  # noqa: SLF001
    output.mkdir(parents=True, exist_ok=True)
    source_binding, queries = collect_queries(
        split_receipt, role=selected_role,
        allow_validation_predictors=allow_validation_predictors,
    )
    state_path = output / "private_provenance/build_state.json"
    if state_path.exists():
        state = daily.read_json(state_path, "hourly causal build state")
        if state.get("schema") != STATE_SCHEMA or state.get("source_binding") != source_binding:
            raise HourlyWeatherError("resume state source binding differs")
    else:
        state = {
            "schema": STATE_SCHEMA, "created_utc": utc_now(),
            "source_binding": source_binding,
            "contract": {
                "endpoint": POWER_HOURLY_API, "parameters": list(PARAMETERS),
                "lookback_hours": LOOKBACK_HOURS,
                "minimum_complete_hours": MIN_COMPLETE_HOURS,
                "timestamp_semantics": dict(TIMESTAMP_SEMANTICS),
                "model_feature_names": list(FEATURE_NAMES),
                "normalization": (
                    "no dataset-level standardization; every finite-window "
                    "exponential mean divides by parameter-specific observed "
                    "kernel mass and retains the provider physical unit"
                ),
                "exponential_scale": "rho_H = H/2",
                "acquisition_execution": (
                    "serial; private state committed atomically after each scene"
                ),
                "retrospective_batch_only": True,
                "near_real_time_authorized": False,
                "selected_role": selected_role,
                "full_role_requires_explicit_approval": True,
                "validation_predictors_require_separate_permission": True,
                "private_provenance_paths_exposed_to_model_loader": False,
            },
            "transactions": {}, "target_arrays_opened": False,
            "validation_target_opened": False, "locked_test_opened": False,
        }
    complete = set(state["transactions"])
    pending = [query for query in queries if query.scene_id not in complete]
    projected = len(complete) + min(max_new_scenes, len(pending))
    if projected > 1 and not allow_bulk_acquisition:
        raise HourlyWeatherError("resumed bulk acquisition requires explicit approval")
    if projected > 1 \
            and TIMESTAMP_SEMANTICS["documentation_content_hash_frozen"] is not True:
        raise HourlyWeatherError(
            "resumed bulk acquisition is blocked until timestamp semantics are hash-frozen"
        )
    if projected == len(queries) and not allow_full_role:
        raise HourlyWeatherError(
            f"completing full {selected_role}{len(queries)} acquisition requires approval"
        )
    if projected == len(queries) \
            and TIMESTAMP_SEMANTICS["documentation_content_hash_frozen"] is not True:
        raise HourlyWeatherError(
            "bulk acquisition is blocked until timestamp semantics are hash-frozen"
        )
    selected = pending[:max_new_scenes]
    for query in selected:
        result = process_one(query, output, fetcher=fetcher)
        state["transactions"][query.scene_id] = result
        state["updated_utc"] = utc_now()
        _atomic_json(state_path, state)
    remaining = len(queries) - len(state["transactions"])
    public_transactions = [
        {
            "scene_id": state["transactions"][key]["scene_id"],
            "role": state["transactions"][key]["role"],
            "sidecar_path": state["transactions"][key]["sidecar_path"],
            "sidecar_sha256": state["transactions"][key]["sidecar_sha256"],
        }
        for key in sorted(state["transactions"])
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete" if remaining == 0 else "partial",
        "created_utc": state["created_utc"], "updated_utc": utc_now(),
        "source_binding": source_binding, "contract": state["contract"],
        "completed_scene_count": len(state["transactions"]),
        "remaining_scene_count": remaining,
        "transactions": public_transactions,
        "raw_query_coordinates_and_times_audit_only": True,
        "private_provenance_paths_exposed": False,
        "loader_returns_exactly_22_physical_values": True,
        "quality_available_only_through_pre_forward_audit": True,
        "target_arrays_opened": False, "validation_target_opened": False,
        "locked_test_opened": False,
    }
    manifest["manifest_content_sha256"] = daily.canonical_sha256(manifest)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-receipt", type=Path, default=daily.DEFAULT_SPLIT_RECEIPT)
    parser.add_argument("--role", choices=("fit", "validation"), default="fit")
    parser.add_argument("--max-new-scenes", type=int, default=1)
    parser.add_argument("--allow-bulk-acquisition", action="store_true")
    parser.add_argument("--allow-full-role", action="store_true")
    parser.add_argument("--allow-validation-predictors", action="store_true")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    result = build_partial(
        output=args.output, split_receipt=args.split_receipt,
        role=args.role,
        max_new_scenes=args.max_new_scenes,
        allow_bulk_acquisition=args.allow_bulk_acquisition,
        allow_full_role=args.allow_full_role,
        allow_validation_predictors=args.allow_validation_predictors,
    )
    print(json.dumps({
        "status": result["status"],
        "completed_scene_count": result["completed_scene_count"],
        "remaining_scene_count": result["remaining_scene_count"],
        "locked_test_opened": result["locked_test_opened"],
        "validation_target_opened": result["validation_target_opened"],
    }, indent=2, sort_keys=True))
