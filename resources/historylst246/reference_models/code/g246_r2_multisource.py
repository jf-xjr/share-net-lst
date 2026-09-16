"""Strict offline Stage-C texture and weather adapter for G246 R2.

This module deliberately wraps :class:`g246_r2_data.R2TemporalDataset` rather
than changing its Core22/Context12 contract.  The first 22 fine channels and
the first 12 context values therefore remain byte-for-byte those produced by
the strict v2 optical adapter.  Stage C appends:

* 30 target-free Landsat 30 m texture summaries, normalised with the exact
  Core22 fitting scope under the scene -> city -> region hierarchy (the
  registered complete fit view for public validation, or a reproducible
  train-only recomputation for internal dev);
  ``valid_coverage30`` is a missing-data mask and is not an input channel.
* seven target-free NASA POWER DAILY values, normalised under that same
  fit-only equal-region/equal-city/equal-scene scope.

Only local immutable sidecars are read.  Source manifests, remote assets,
POWER endpoints, target arrays, and locked-test descriptors are never opened
by this module.  Every requested scene is bound to its public role, identity,
item, datetime, and frozen source SHA-256 before a sidecar can be materialised.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from . import g246_data
    from . import g246_r2_data as r2
except ImportError:  # pragma: no cover - direct code-path import
    import g246_data
    import g246_r2_data as r2


TEXTURE_MANIFEST_SCHEMA = "uhi-cdc-g246-r2-texture-sidecars-v2"
TEXTURE_SCENE_SCHEMA = "uhi-cdc-g246-r2-texture-scene-v2"
TEXTURE_NORMALIZATION_SCHEMA = "uhi-cdc-g246-r2-texture-normalization-v2"
WEATHER_MANIFEST_SCHEMA = "uhi-cdc-g246-r2-weather-manifest-v1"
WEATHER_SCENE_SCHEMA = "uhi-cdc-g246-r2-weather-sidecar-v1"
WEATHER_NORMALIZATION_SCHEMA = "uhi-cdc-g246-r2-weather-normalization-v1"
MULTISOURCE_SCHEMA = "uhi-cdc-g246-r2-multisource-binding-v1"

TEXTURE_RAW_CHANNEL_NAMES = (
    "blue_std", "blue_q25", "blue_q75",
    "green_std", "green_q25", "green_q75",
    "red_std", "red_q25", "red_q75",
    "nir08_std", "nir08_q25", "nir08_q75",
    "swir16_std", "swir16_q25", "swir16_q75",
    "swir22_std", "swir22_q25", "swir22_q75",
    "ndvi_mean", "ndvi_std", "ndvi_q25", "ndvi_q75",
    "ndbi_mean", "ndbi_std", "ndbi_q25", "ndbi_q75",
    "mndwi_mean", "mndwi_std", "mndwi_q25", "mndwi_q75",
)
TEXTURE_COVERAGE_NAME = "valid_coverage30"
TEXTURE_STORED_CHANNEL_NAMES = (*TEXTURE_RAW_CHANNEL_NAMES, TEXTURE_COVERAGE_NAME)
OPTICAL_MEAN_NAMES = (
    "blue_mean120", "green_mean120", "red_mean120",
    "nir08_mean120", "swir16_mean120", "swir22_mean120",
)
TEXTURE_FINE_CHANNEL_NAMES = tuple(
    f"texture_{name}_fit_z" for name in TEXTURE_RAW_CHANNEL_NAMES
)

WEATHER_PARAMETERS = (
    "T2M_MAX", "T2M_MIN", "RH2M", "WS2M",
    "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "GWETTOP",
)
WEATHER_CONTEXT_NAMES = (
    "power_daily_t2m_max_fit_z",
    "power_daily_t2m_min_fit_z",
    "power_daily_rh2m_fit_z",
    "power_daily_ws2m_fit_z",
    "power_daily_allsky_sfc_sw_dwn_fit_z",
    "power_daily_prectotcorr_fit_z",
    "power_daily_gwettop_fit_z",
)

MULTISOURCE_FINE_CHANNEL_NAMES = (*r2.FINE_CHANNEL_NAMES, *TEXTURE_FINE_CHANNEL_NAMES)
MULTISOURCE_CONTEXT_NAMES = (*r2.CONTEXT_NAMES, *WEATHER_CONTEXT_NAMES)
MULTISOURCE_FINE_CHANNELS = len(MULTISOURCE_FINE_CHANNEL_NAMES)
MULTISOURCE_CONTEXT_DIM = len(MULTISOURCE_CONTEXT_NAMES)
if MULTISOURCE_FINE_CHANNELS != 52 or MULTISOURCE_CONTEXT_DIM != 19:  # pragma: no cover
    raise RuntimeError("G246 Stage-C multisource feature contract drifted")


def _hex_digest(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _guard_entry(entry: g246_data.G246Scene) -> None:
    # Role is checked before any path operation, including manifest reads.
    if entry.view_role not in {"fit", "validation"}:
        raise ValueError(
            f"Stage-C accepts only public fit/validation scenes, got {entry.view_role!r}"
        )
    g246_data.reject_forbidden_path(entry.file)
    _hex_digest(entry.sha256, f"source scene {entry.scene_id} SHA-256")


def _entry_map(
    entries: Sequence[g246_data.G246Scene], *, label: str,
) -> dict[str, g246_data.G246Scene]:
    if not entries:
        raise ValueError(f"{label} requires at least one public scene")
    result: dict[str, g246_data.G246Scene] = {}
    for entry in entries:
        _guard_entry(entry)
        if entry.scene_id in result:
            raise ValueError(f"{label} received duplicate scene {entry.scene_id}")
        result[entry.scene_id] = entry
    return result


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = g246_data.reject_forbidden_path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular local file: {candidate}")
    return candidate.resolve()


def _resolve_member(root: Path, raw: Any, label: str) -> Path:
    """Resolve a manifest member without allowing absolute/traversal/symlink paths."""

    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label} has an unsafe relative path")
    g246_data.reject_forbidden_path(raw)
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} has an unsafe relative path")
    root = root.resolve()
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its manifest directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular local file: {resolved}")
    return resolved


def _read_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return value, raw


def _scene_date_utc(timestamp: str) -> str:
    raw = str(timestamp).strip()
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as exc:
        raise ValueError(f"scene datetime is not ISO-8601: {timestamp!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("scene datetime must carry a timezone")
    return parsed.astimezone(timezone.utc).date().isoformat()


def _safe_moments(document: Mapping[str, Any], label: str) -> tuple[float, float]:
    mean = float(document.get("mean", math.nan))
    std = float(document.get("std", math.nan))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError(f"{label} has invalid fit-only moments")
    return mean, std


def _validate_texture_hierarchy(document: Mapping[str, Any], channel: str) -> None:
    """Validate the registered scene -> city -> region mixture weights."""

    regions = document.get("regions")
    if not isinstance(regions, Mapping) or set(regions) != set(r2.MACRO_REGIONS):
        raise ValueError(f"texture normalization {channel} lacks all macro-regions")
    for region in r2.MACRO_REGIONS:
        region_doc = regions[region]
        if not isinstance(region_doc, Mapping) or not math.isclose(
            float(region_doc.get("normalization_weight", math.nan)),
            1.0 / len(r2.MACRO_REGIONS), rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"texture normalization {channel}/{region} weight is invalid")
        cities = region_doc.get("cities")
        if not isinstance(cities, Mapping) or not cities:
            raise ValueError(f"texture normalization {channel}/{region} has no fit cities")
        city_weight = 1.0 / len(cities)
        for city, city_doc in cities.items():
            if not city or not isinstance(city_doc, Mapping) or not math.isclose(
                float(city_doc.get("normalization_weight", math.nan)),
                city_weight, rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError(f"texture normalization {channel}/{region} city weight is invalid")
            scenes = city_doc.get("scenes")
            if not isinstance(scenes, Mapping) or not scenes:
                raise ValueError(f"texture normalization {channel}/{city} has no fit scenes")
            scene_weight = 1.0 / len(scenes)
            if any(
                not isinstance(scene_doc, Mapping)
                or not math.isclose(
                    float(scene_doc.get("normalization_weight", math.nan)),
                    scene_weight, rel_tol=0.0, abs_tol=1e-12,
                )
                for scene_doc in scenes.values()
            ):
                raise ValueError(f"texture normalization {channel}/{city} scene weight is invalid")


def _normalization_scope(
    rows: Mapping[str, Mapping[str, Any]],
    scene_ids: Sequence[str] | None,
    *,
    label: str,
) -> tuple[tuple[str, ...], str]:
    """Resolve a predictor-only fitting scope and bind it by canonical hash."""

    full_fit = tuple(sorted(
        scene_id for scene_id, row in rows.items()
        if (row.get("view_role") or row.get("role")) == "fit"
    ))
    selected = full_fit if scene_ids is None else tuple(sorted(map(str, scene_ids)))
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(scene_id not in rows for scene_id in selected)
        or any(
            (rows[scene_id].get("view_role") or rows[scene_id].get("role")) != "fit"
            for scene_id in selected
        )
    ):
        raise ValueError(f"{label} normalization scene scope is not a fit-only subset")
    regions = {str(rows[scene_id].get("region")) for scene_id in selected}
    if regions != set(r2.MACRO_REGIONS):
        raise ValueError(f"{label} normalization scene scope lacks all macro-regions")
    identity = _canonical_sha256({
        "schema": "uhi-cdc-g246-r2-predictor-normalization-scope-v1",
        "scene_ids": list(selected),
        "locked_test_opened": False,
        "target_arrays_opened": False,
    })
    return selected, identity


def _balanced_moments(
    components: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-scene/equal-city/equal-region moments for vector predictors."""

    grouped: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        region: {} for region in r2.MACRO_REGIONS
    }
    for region, city, first, second in components:
        if region not in grouped or not city:
            raise ValueError(f"{label} normalization has invalid city/region")
        grouped[region].setdefault(city, []).append((first, second))
    if any(not grouped[region] for region in r2.MACRO_REGIONS):
        raise ValueError(f"{label} normalization lacks a macro-region")
    regional_first: list[np.ndarray] = []
    regional_second: list[np.ndarray] = []
    for region in r2.MACRO_REGIONS:
        city_first = [
            np.mean(np.stack([value[0] for value in scenes]), axis=0)
            for scenes in grouped[region].values()
        ]
        city_second = [
            np.mean(np.stack([value[1] for value in scenes]), axis=0)
            for scenes in grouped[region].values()
        ]
        regional_first.append(np.mean(np.stack(city_first), axis=0))
        regional_second.append(np.mean(np.stack(city_second), axis=0))
    mean = np.mean(np.stack(regional_first), axis=0)
    second = np.mean(np.stack(regional_second), axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0)).clip(1e-8)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError(f"{label} normalization moments are not finite")
    return mean.astype(np.float32), std.astype(np.float32)


class Texture30Adapter:
    """Read and normalise the 30 texture features from a complete v2 manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        entries: Sequence[g246_data.G246Scene],
        *,
        expected_manifest_sha256: str | None = None,
        normalization_scene_ids: Sequence[str] | None = None,
        cache_size: int = 128,
    ) -> None:
        self._entries = _entry_map(entries, label="texture adapter")
        if cache_size <= 0:
            raise ValueError("texture cache_size must be positive")
        path = _regular_file(manifest_path, "texture manifest")
        raw = path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_manifest_sha256 is not None and self.manifest_sha256 != _hex_digest(
            expected_manifest_sha256, "expected texture manifest SHA-256"
        ):
            raise ValueError("texture manifest SHA-256 differs from the registered binding")
        try:
            values = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("texture manifest JSONL is malformed") from exc
        if not values or any(not isinstance(value, dict) for value in values):
            raise ValueError("texture manifest must contain JSON objects")
        header, rows = values[0], values[1:]
        if (
            header.get("kind") != "dataset"
            or header.get("schema") != TEXTURE_MANIFEST_SCHEMA
            or header.get("status") != "complete"
            or header.get("input_scope") != "g246_data.load_splits:fit+validation"
            or header.get("locked_test_opened") is not False
            or header.get("selected_cities") not in (None, [])
            or header.get("normalization_scope")
            != "fit-only equal-scene/equal-city/equal-region"
        ):
            raise ValueError("texture manifest is incomplete, partial, or unsafe")
        if int(header.get("scene_count", -1)) != len(rows):
            raise ValueError("texture manifest scene count differs from its rows")
        role_counts = {
            role: sum(row.get("view_role") == role for row in rows)
            for role in ("fit", "validation")
        }
        if any(
            int(header.get(f"{role}_scene_count", -1)) != role_counts[role]
            for role in role_counts
        ):
            raise ValueError("texture manifest public-role counts differ")
        arrays = header.get("arrays")
        texture_contract = arrays.get("texture31") if isinstance(arrays, Mapping) else None
        optical_contract = arrays.get("optical_mean6") if isinstance(arrays, Mapping) else None
        if (
            not isinstance(texture_contract, Mapping)
            or texture_contract.get("shape") != [31, 160, 160]
            or texture_contract.get("dtype") != "float16"
            or texture_contract.get("channels") != list(TEXTURE_STORED_CHANNEL_NAMES)
            or not isinstance(optical_contract, Mapping)
            or optical_contract.get("shape") != [6, 160, 160]
            or optical_contract.get("dtype") != "float16"
            or optical_contract.get("channels") != list(OPTICAL_MEAN_NAMES)
            or int(header.get("stored_channel_count", -1)) != 37
        ):
            raise ValueError("texture manifest feature-array contract differs")
        self.fit_view_sha256 = _hex_digest(
            header.get("fit_view_sha256"), "texture fit view SHA-256"
        )
        self.validation_view_sha256 = _hex_digest(
            header.get("validation_view_sha256"), "texture validation view SHA-256"
        )
        self.receipt_sha256 = _hex_digest(
            header.get("receipt_sha256"), "texture split receipt SHA-256"
        )
        self.source_manifest_sha256 = _hex_digest(
            header.get("source_manifest_sha256"), "texture source manifest SHA-256"
        )
        self.campaign_id = _hex_digest(
            header.get("campaign_id"), "texture campaign identity"
        )

        row_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            scene_id = str(row.get("scene_id", ""))
            if (
                row.get("kind") != "scene"
                or row.get("status") != "complete"
                or row.get("view_role") not in {"fit", "validation"}
                or row.get("locked_test_opened") is not False
                or not scene_id
                or scene_id in row_by_id
                or row.get("shape") != [31, 160, 160]
                or row.get("optical_mean_shape") != [6, 160, 160]
                or row.get("arrays") != ["texture31", "optical_mean6"]
                or row.get("dtype") != "float16"
            ):
                raise ValueError("texture manifest contains an unsafe/malformed scene row")
            if not isinstance(row.get("file"), str) or Path(str(row["file"])).name != row["file"]:
                raise ValueError("texture scene filename must be a local basename")
            g246_data.reject_forbidden_path(row["file"])
            _hex_digest(row.get("sha256"), f"texture sidecar {scene_id} SHA-256")
            _hex_digest(row.get("source_scene_sha256"), f"texture source {scene_id} SHA-256")
            if int(row.get("bytes", -1)) <= 0:
                raise ValueError(f"texture sidecar {scene_id} has invalid byte count")
            row_by_id[scene_id] = dict(row)
        missing = set(self._entries).difference(row_by_id)
        if missing:
            raise ValueError(f"texture manifest omits requested scenes: {sorted(missing)[:3]}")
        for scene_id, entry in self._entries.items():
            row = row_by_id[scene_id]
            expected = {
                "view_role": entry.view_role, "region": entry.region,
                "city": entry.city, "target_year": int(entry.year),
                "source_scene_sha256": entry.sha256, "item_id": entry.item_id,
                "datetime": entry.datetime,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"texture row differs from public scene: {scene_id}")

        normalization_path = _resolve_member(
            path.parent, header.get("normalization_file"), "texture normalization"
        )
        normalization, normalization_raw = _read_json_file(
            normalization_path, "texture normalization"
        )
        self.normalization_sha256 = hashlib.sha256(normalization_raw).hexdigest()
        if self.normalization_sha256 != _hex_digest(
            header.get("normalization_sha256"), "texture normalization SHA-256"
        ):
            raise ValueError("texture normalization SHA-256 differs from manifest")
        scope = normalization.get("scope")
        if (
            normalization.get("schema") != TEXTURE_NORMALIZATION_SCHEMA
            or not isinstance(scope, Mapping)
            or scope.get("view_role") != "fit"
            or scope.get("validation_included") is not False
            or scope.get("locked_test_opened") is not False
            or normalization.get("fit_view_sha256") != self.fit_view_sha256
            or int(normalization.get("fit_scene_count", -1)) != role_counts["fit"]
            or int(normalization.get("validation_scene_count_excluded", -1))
            != role_counts["validation"]
            or normalization.get("regions") != list(r2.MACRO_REGIONS)
            or normalization.get("coverage_channel") != TEXTURE_COVERAGE_NAME
            or normalization.get("excluded") != [TEXTURE_COVERAGE_NAME]
            or normalization.get("value_arrays", {}).get("texture31")
            != list(TEXTURE_RAW_CHANNEL_NAMES)
            or normalization.get("value_arrays", {}).get("optical_mean6")
            != list(OPTICAL_MEAN_NAMES)
            or int(normalization.get("normalized_channel_count", -1)) != 36
        ):
            raise ValueError("texture normalization is not the registered fit-only hierarchy")
        weighting = str(normalization.get("weighting", ""))
        if not all(token in weighting for token in (
            "weights cells only within each scene", "scenes are equal within city",
            "cities are equal within region", "global weight 1/3",
        )):
            raise ValueError("texture normalization weighting contract differs")
        channels = normalization.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != set(
            (*TEXTURE_RAW_CHANNEL_NAMES, *OPTICAL_MEAN_NAMES)
        ):
            raise ValueError("texture normalization channel inventory differs")
        means: list[float] = []
        stds: list[float] = []
        for name in TEXTURE_RAW_CHANNEL_NAMES:
            document = channels[name]
            if not isinstance(document, Mapping):
                raise ValueError(f"texture normalization channel {name} is malformed")
            mean, std = _safe_moments(document, f"texture normalization {name}")
            _validate_texture_hierarchy(document, name)
            means.append(mean)
            stds.append(std)

        self.manifest_path = path
        self.header = dict(header)
        self.normalization = dict(normalization)
        self._rows = row_by_id
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()
        self.normalization_scene_ids, self.normalization_scope_sha256 = (
            _normalization_scope(
                self._rows, normalization_scene_ids, label="texture"
            )
        )
        full_fit_ids = tuple(sorted(
            scene_id for scene_id, row in self._rows.items()
            if row.get("view_role") == "fit"
        ))
        registered_means = np.asarray(means, dtype=np.float32)
        registered_stds = np.asarray(stds, dtype=np.float32)
        if self.normalization_scene_ids == full_fit_ids:
            active_means, active_stds = registered_means, registered_stds
            self.normalization_scope = "complete_fit_view"
            self.active_normalization_sha256 = self.normalization_sha256
        else:
            active_means, active_stds = self._fit_subset_moments(
                self.normalization_scene_ids
            )
            self.normalization_scope = "internal_dev_train_only"
            self.active_normalization_sha256 = _canonical_sha256({
                "schema": "uhi-cdc-g246-r2-texture-runtime-normalization-v1",
                "source_normalization_sha256": self.normalization_sha256,
                "scope_sha256": self.normalization_scope_sha256,
                "mean": [float(value) for value in active_means],
                "std": [float(value) for value in active_stds],
                "locked_test_opened": False,
                "target_arrays_opened": False,
            })
        self._means = active_means[:, None, None]
        self._stds = active_stds[:, None, None]

    def _raw_arrays(self, scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if scene_id not in self._cache:
            row = self._rows[scene_id]
            path = _resolve_member(self.manifest_path.parent, row["file"], "texture sidecar")
            raw = path.read_bytes()
            if (
                len(raw) != int(row["bytes"])
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
            ):
                raise ValueError(f"texture sidecar hash/size mismatch: {scene_id}")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                expected_arrays = {
                    "texture31", "optical_mean6", "channel_names",
                    "optical_channel_names", "metadata",
                }
                if set(archive.files) != expected_arrays:
                    raise ValueError(f"texture sidecar arrays differ: {scene_id}")
                texture = np.asarray(archive["texture31"])
                optical = np.asarray(archive["optical_mean6"])
                names = tuple(np.asarray(archive["channel_names"]).astype(str).tolist())
                optical_names = tuple(
                    np.asarray(archive["optical_channel_names"]).astype(str).tolist()
                )
                try:
                    metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"texture sidecar metadata is malformed: {scene_id}"
                    ) from exc
            coverage = np.asarray(texture[-1], dtype=np.float32)
            expected_metadata = {
                "schema": TEXTURE_SCENE_SCHEMA,
                "view_role": row["view_role"], "region": row["region"],
                "city": row["city"], "target_year": int(row["target_year"]),
                "scene_id": scene_id,
                "source_scene_sha256": row["source_scene_sha256"],
                "item_id": row["item_id"], "datetime": row["datetime"],
                "stored_optical_used": False, "locked_test_opened": False,
            }
            if (
                texture.shape != (31, 160, 160)
                or texture.dtype != np.float16
                or optical.shape != (6, 160, 160)
                or optical.dtype != np.float16
                or names != TEXTURE_STORED_CHANNEL_NAMES
                or optical_names != OPTICAL_MEAN_NAMES
                or not np.all(np.isfinite(texture))
                or not np.all(np.isfinite(optical))
                or np.any(coverage < 0)
                or np.any(coverage > 1)
                or np.any(texture[:-1, coverage == 0] != 0)
                or np.any(optical[:, coverage == 0] != 0)
                or any(metadata.get(key) != value for key, value in expected_metadata.items())
                or metadata.get("texture_channels") != list(TEXTURE_STORED_CHANNEL_NAMES)
                or metadata.get("optical_mean_channels") != list(OPTICAL_MEAN_NAMES)
            ):
                raise ValueError(f"texture sidecar content is invalid: {scene_id}")
            self._cache[scene_id] = (
                np.array(texture[:30], copy=True),
                np.array(optical, copy=True),
                np.array(texture[-1], copy=True),
            )
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(scene_id)
        return self._cache[scene_id]

    def _fit_subset_moments(
        self, scene_ids: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        components: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for scene_id in scene_ids:
            texture, _optical, coverage = self._raw_arrays(scene_id)
            mask = np.isfinite(coverage) & (coverage > 0)
            if not np.any(mask):
                raise ValueError(f"texture normalization scene has no support: {scene_id}")
            weights = np.asarray(coverage[mask], dtype=np.float64)
            values = np.asarray(texture[:, mask], dtype=np.float64)
            weight = float(weights.sum())
            first = np.sum(values * weights[None], axis=1) / weight
            second = np.sum(np.square(values) * weights[None], axis=1) / weight
            row = self._rows[scene_id]
            components.append((str(row["region"]), str(row["city"]), first, second))
        return _balanced_moments(components, label="texture")

    def normalize_texture(
        self, texture: np.ndarray, coverage: np.ndarray,
    ) -> np.ndarray:
        normalized = (np.asarray(texture, dtype=np.float32) - self._means) / self._stds
        normalized[:, np.asarray(coverage) <= 0] = 0.0
        if not np.all(np.isfinite(normalized)):
            raise ValueError("normalised texture is not finite")
        return normalized

    def preload_requested(self) -> dict[str, int]:
        """Materialise each requested immutable sidecar exactly once.

        The explicit capacity check prevents a misleading "preload" that
        immediately evicts scenes and returns to repeated NPZ decompression.
        Predictor-only normalization scenes outside the dataset (for example,
        train scenes used by an internal-dev evaluator) are discarded first.
        """

        requested = tuple(sorted(self._entries))
        if self.cache_size < len(requested):
            raise ValueError(
                "texture cache is too small for a complete requested-scene preload: "
                f"capacity={self.cache_size}, required={len(requested)}"
            )
        for scene_id in tuple(self._cache):
            if scene_id not in self._entries:
                del self._cache[scene_id]
        for scene_id in requested:
            self._raw_arrays(scene_id)
        resident_bytes = sum(
            sum(array.nbytes for array in arrays)
            for arrays in self._cache.values()
        )
        return {
            "requested_scenes": len(requested),
            "resident_scenes": len(self._cache),
            "resident_bytes": int(resident_bytes),
            "cache_capacity": self.cache_size,
        }

    def load(self, entry: g246_data.G246Scene) -> dict[str, np.ndarray]:
        _guard_entry(entry)
        if self._entries.get(entry.scene_id) != entry:
            raise ValueError(f"scene is outside texture adapter scope: {entry.scene_id}")
        texture, optical, coverage = self._raw_arrays(entry.scene_id)
        normalized = self.normalize_texture(texture, coverage)
        return {
            "texture30": np.array(normalized, copy=True),
            "texture30_raw": np.array(texture, copy=True),
            "optical_mean6": np.array(optical, copy=True),
            "coverage": np.array(coverage, copy=True),
        }


class Weather7Adapter:
    """Read fit-normalised NASA POWER daily features from immutable JSON sidecars."""

    def __init__(
        self,
        manifest_path: str | Path,
        entries: Sequence[g246_data.G246Scene],
        *,
        expected_manifest_sha256: str | None = None,
        expected_fit_view_sha256: str | None = None,
        expected_validation_view_sha256: str | None = None,
        normalization_scene_ids: Sequence[str] | None = None,
        cache_size: int = 768,
    ) -> None:
        self._entries = _entry_map(entries, label="weather adapter")
        if cache_size <= 0:
            raise ValueError("weather cache_size must be positive")
        path = _regular_file(manifest_path, "weather manifest")
        manifest, raw = _read_json_file(path, "weather manifest")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_manifest_sha256 is not None and self.manifest_sha256 != _hex_digest(
            expected_manifest_sha256, "expected weather manifest SHA-256"
        ):
            raise ValueError("weather manifest SHA-256 differs from the registered binding")
        content = dict(manifest)
        registered_content_sha = content.pop("manifest_content_sha256", None)
        if _canonical_sha256(content) != _hex_digest(
            registered_content_sha, "weather manifest content SHA-256"
        ):
            raise ValueError("weather manifest content identity differs")
        binding = manifest.get("source_binding")
        contract = manifest.get("contract")
        if (
            manifest.get("schema") != WEATHER_MANIFEST_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("locked_test_opened") is not False
            or manifest.get("target_arrays_opened") is not False
            or not isinstance(binding, Mapping)
            or binding.get("roles_opened") != ["fit", "validation"]
            or binding.get("locked_test_opened") is not False
            or binding.get("target_arrays_opened") is not False
            or not isinstance(contract, Mapping)
            or contract.get("parameters") != list(WEATHER_PARAMETERS)
            or contract.get("normalization_scope")
            != "fit_only_equal_region_equal_city_equal_scene"
            or contract.get("locked_test_opened") is not False
            or contract.get("target_arrays_opened") is not False
        ):
            raise ValueError("weather manifest is incomplete or unsafe")
        self.fit_view_sha256 = _hex_digest(
            binding.get("fit_view_sha256"), "weather fit view SHA-256"
        )
        self.validation_view_sha256 = _hex_digest(
            binding.get("validation_view_sha256"), "weather validation view SHA-256"
        )
        split_receipt = binding.get("split_receipt")
        source_manifest = binding.get("source_manifest")
        if not isinstance(split_receipt, Mapping) or not isinstance(source_manifest, Mapping):
            raise ValueError("weather manifest lacks public source descriptors")
        for descriptor, label in (
            (split_receipt, "weather split receipt"),
            (source_manifest, "weather source manifest"),
        ):
            if not isinstance(descriptor.get("path"), str):
                raise ValueError(f"{label} path is malformed")
            # Provenance paths are never resolved or read here, but even their
            # lexical form must not point at a sealed/locked descriptor.
            g246_data.reject_forbidden_path(descriptor["path"])
            _hex_digest(descriptor.get("sha256"), f"{label} SHA-256")
        self.receipt_sha256 = str(split_receipt["sha256"])
        self.source_manifest_sha256 = str(source_manifest["sha256"])
        self.campaign_id = _hex_digest(
            binding.get("campaign_id"), "weather campaign identity"
        )
        if expected_fit_view_sha256 is not None and self.fit_view_sha256 != _hex_digest(
            expected_fit_view_sha256, "expected fit view SHA-256"
        ):
            raise ValueError("weather and texture fit views differ")
        if (
            expected_validation_view_sha256 is not None
            and self.validation_view_sha256 != _hex_digest(
                expected_validation_view_sha256, "expected validation view SHA-256"
            )
        ):
            raise ValueError("weather and texture validation views differ")
        units = manifest.get("units")
        if not isinstance(units, Mapping) or set(units) != set(WEATHER_PARAMETERS):
            raise ValueError("weather manifest units differ from the seven-feature contract")
        scenes = manifest.get("scenes")
        counts = manifest.get("counts")
        if (
            not isinstance(scenes, list)
            or not isinstance(counts, Mapping)
            or int(counts.get("scenes", -1)) != len(scenes)
        ):
            raise ValueError("weather manifest scene count differs")
        role_counts = {
            role: sum(
                isinstance(row, Mapping) and row.get("role") == role for row in scenes
            )
            for role in ("fit", "validation")
        }
        if (
            counts.get("scenes_by_role") != role_counts
            or int(binding.get("fit_scene_count", -1)) != role_counts["fit"]
            or int(binding.get("validation_scene_count", -1)) != role_counts["validation"]
        ):
            raise ValueError("weather manifest public-role counts differ")
        row_by_id: dict[str, dict[str, Any]] = {}
        for row in scenes:
            if not isinstance(row, Mapping):
                raise ValueError("weather manifest scene row is malformed")
            scene_id = str(row.get("scene_id", ""))
            if (
                not scene_id or scene_id in row_by_id
                or row.get("role") not in {"fit", "validation"}
            ):
                raise ValueError("weather manifest has duplicate/unsafe scene identity")
            _hex_digest(row.get("sha256"), f"weather sidecar {scene_id} SHA-256")
            _hex_digest(row.get("source_scene_sha256"), f"weather source {scene_id} SHA-256")
            # Validate relative paths now, without opening unrequested sidecars.
            raw_member = row.get("path")
            if not isinstance(raw_member, str) or "\\" in raw_member:
                raise ValueError("weather manifest sidecar path is unsafe")
            pure = PurePosixPath(raw_member)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise ValueError("weather manifest sidecar path is unsafe")
            g246_data.reject_forbidden_path(raw_member)
            row_by_id[scene_id] = dict(row)
        missing = set(self._entries).difference(row_by_id)
        if missing:
            raise ValueError(f"weather manifest omits requested scenes: {sorted(missing)[:3]}")
        for scene_id, entry in self._entries.items():
            row = row_by_id[scene_id]
            expected = {
                "role": entry.view_role, "region": entry.region, "city": entry.city,
                "year": int(entry.year), "source_scene_sha256": entry.sha256,
                "item_id": entry.item_id, "datetime": entry.datetime,
                "date_utc": _scene_date_utc(entry.datetime),
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"weather row differs from public scene: {scene_id}")

        normalization_record = manifest.get("normalization")
        if not isinstance(normalization_record, Mapping):
            raise ValueError("weather manifest lacks normalization binding")
        normalization_path = _resolve_member(
            path.parent, normalization_record.get("path"), "weather normalization"
        )
        normalization, normalization_raw = _read_json_file(
            normalization_path, "weather normalization"
        )
        self.normalization_sha256 = hashlib.sha256(normalization_raw).hexdigest()
        if (
            self.normalization_sha256
            != _hex_digest(normalization_record.get("sha256"), "weather normalization SHA-256")
            or normalization_record.get("schema") != WEATHER_NORMALIZATION_SCHEMA
            or normalization_record.get("scope") != "fit_only"
            or normalization.get("schema") != WEATHER_NORMALIZATION_SCHEMA
            or normalization.get("scope") != "fit_only"
            or normalization.get("weighting")
            != "equal_region_then_equal_city_then_equal_scene"
            or normalization.get("parameters") != list(WEATHER_PARAMETERS)
            or normalization.get("fit_view_sha256") != self.fit_view_sha256
            or int(normalization.get("fit_scene_count", -1)) != role_counts["fit"]
            or normalization.get("validation_values_used") is not False
            or normalization.get("locked_test_opened") is not False
            or normalization.get("target_arrays_opened") is not False
        ):
            raise ValueError("weather normalization is not fit-only or is unbound")
        mean_doc, std_doc = normalization.get("mean"), normalization.get("std")
        if (
            not isinstance(mean_doc, Mapping) or set(mean_doc) != set(WEATHER_PARAMETERS)
            or not isinstance(std_doc, Mapping) or set(std_doc) != set(WEATHER_PARAMETERS)
        ):
            raise ValueError("weather normalization moment inventory differs")
        means: list[float] = []
        stds: list[float] = []
        for parameter in WEATHER_PARAMETERS:
            mean, std = _safe_moments(
                {"mean": mean_doc[parameter], "std": std_doc[parameter]},
                f"weather normalization {parameter}",
            )
            means.append(mean)
            stds.append(std)

        self.manifest_path = path
        self.manifest = dict(manifest)
        self.normalization = dict(normalization)
        self._rows = row_by_id
        self._units = dict(units)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.normalization_scene_ids, self.normalization_scope_sha256 = (
            _normalization_scope(
                self._rows, normalization_scene_ids, label="weather"
            )
        )
        full_fit_ids = tuple(sorted(
            scene_id for scene_id, row in self._rows.items()
            if row.get("role") == "fit"
        ))
        registered_means = np.asarray(means, dtype=np.float32)
        registered_stds = np.asarray(stds, dtype=np.float32)
        if self.normalization_scene_ids == full_fit_ids:
            self._means, self._stds = registered_means, registered_stds
            self.normalization_scope = "complete_fit_view"
            self.active_normalization_sha256 = self.normalization_sha256
        else:
            self._means, self._stds = self._fit_subset_moments(
                self.normalization_scene_ids
            )
            self.normalization_scope = "internal_dev_train_only"
            self.active_normalization_sha256 = _canonical_sha256({
                "schema": "uhi-cdc-g246-r2-weather-runtime-normalization-v1",
                "source_normalization_sha256": self.normalization_sha256,
                "scope_sha256": self.normalization_scope_sha256,
                "mean": [float(value) for value in self._means],
                "std": [float(value) for value in self._stds],
                "locked_test_opened": False,
                "target_arrays_opened": False,
            })

    def _raw_values(self, scene_id: str) -> np.ndarray:
        if scene_id not in self._cache:
            row = self._rows[scene_id]
            path = _resolve_member(self.manifest_path.parent, row["path"], "weather sidecar")
            payload, raw = _read_json_file(path, f"weather sidecar {scene_id}")
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"weather sidecar SHA-256 mismatch: {scene_id}")
            expected = {
                "schema": WEATHER_SCENE_SCHEMA, "scene_id": scene_id,
                "city": row["city"], "year": int(row["year"]),
                "region": row["region"], "role": row["role"],
                "item_id": row["item_id"], "datetime": row["datetime"],
                "date_utc": row["date_utc"],
                "source_scene_sha256": row["source_scene_sha256"],
                "target_free": True, "locked_test_opened": False,
                "target_arrays_opened": False,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                raise ValueError(f"weather sidecar identity differs: {scene_id}")
            features = payload.get("features")
            source_snapshot = payload.get("source_snapshot")
            if (
                not isinstance(features, Mapping)
                or set(features) != set(WEATHER_PARAMETERS)
                or payload.get("units") != self._units
                or not isinstance(source_snapshot, Mapping)
                or not isinstance(source_snapshot.get("path"), str)
            ):
                raise ValueError(f"weather sidecar contract differs: {scene_id}")
            _hex_digest(source_snapshot.get("sha256"), "weather source snapshot SHA-256")
            _hex_digest(source_snapshot.get("query_sha256"), "weather query SHA-256")
            snapshot_path = PurePosixPath(str(source_snapshot["path"]))
            g246_data.reject_forbidden_path(str(source_snapshot["path"]))
            if (
                snapshot_path.is_absolute()
                or any(part in {"", ".", ".."} for part in snapshot_path.parts)
                or "\\" in str(source_snapshot["path"])
            ):
                raise ValueError(f"weather snapshot path is unsafe: {scene_id}")
            raw_values = np.asarray(
                [features[name] for name in WEATHER_PARAMETERS], dtype=np.float32
            )
            if not np.all(np.isfinite(raw_values)) or np.any(raw_values <= -900.0):
                raise ValueError(f"weather sidecar values are invalid: {scene_id}")
            self._cache[scene_id] = raw_values
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(scene_id)
        return self._cache[scene_id]

    def _fit_subset_moments(
        self, scene_ids: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        components: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for scene_id in scene_ids:
            values = np.asarray(self._raw_values(scene_id), dtype=np.float64)
            row = self._rows[scene_id]
            components.append((
                str(row["region"]), str(row["city"]), values, np.square(values)
            ))
        return _balanced_moments(components, label="weather")

    def preload_requested(self) -> dict[str, int]:
        requested = tuple(sorted(self._entries))
        if self.cache_size < len(requested):
            raise ValueError(
                "weather cache is too small for a complete requested-scene preload: "
                f"capacity={self.cache_size}, required={len(requested)}"
            )
        for scene_id in tuple(self._cache):
            if scene_id not in self._entries:
                del self._cache[scene_id]
        for scene_id in requested:
            self._raw_values(scene_id)
        resident_bytes = sum(array.nbytes for array in self._cache.values())
        return {
            "requested_scenes": len(requested),
            "resident_scenes": len(self._cache),
            "resident_bytes": int(resident_bytes),
            "cache_capacity": self.cache_size,
        }

    def load(self, entry: g246_data.G246Scene) -> np.ndarray:
        _guard_entry(entry)
        if self._entries.get(entry.scene_id) != entry:
            raise ValueError(f"scene is outside weather adapter scope: {entry.scene_id}")
        normalized = (self._raw_values(entry.scene_id) - self._means) / self._stds
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"normalised weather is not finite: {entry.scene_id}")
        return np.asarray(normalized, dtype=np.float32).copy()


def _spatial_transform(array: np.ndarray, code: int) -> np.ndarray:
    value = np.rot90(array, k=code & 3, axes=(-2, -1))
    if code & 4:
        value = value[..., ::-1]
    return np.ascontiguousarray(value)


def _parse_optical_augmentation(
    value: Any,
) -> tuple[np.ndarray, np.ndarray, int | None, bool]:
    if not isinstance(value, Mapping) or value.get("schema") != r2.OPTICAL_AUGMENTATION_SCHEMA:
        raise ValueError("Core22 batch lacks the registered optical augmentation contract")
    try:
        gain = np.asarray(value.get("gain"), dtype=np.float32)
        offset = np.asarray(value.get("offset_raw"), dtype=np.float32)
        applied = bool(value.get("applied"))
        dropped = value.get("dropped_band")
        dropped_band = None if dropped is None else int(dropped)
    except (TypeError, ValueError) as exc:
        raise ValueError("Core22 optical augmentation contract is malformed") from exc
    if (
        gain.shape != (6,) or offset.shape != (6,)
        or not np.all(np.isfinite(gain)) or not np.all(np.isfinite(offset))
        or np.any(gain <= 0.0)
        or (dropped_band is not None and not 0 <= dropped_band < 6)
        or (
            not applied and (
                not np.array_equal(gain, np.ones(6, dtype=np.float32))
                or not np.array_equal(offset, np.zeros(6, dtype=np.float32))
                or dropped_band is not None
            )
        )
    ):
        raise ValueError("Core22 optical augmentation contract is malformed")
    return gain, offset, dropped_band, applied


def _index_value_and_gradient(
    optical: np.ndarray,
    positive: Sequence[int],
    negative: Sequence[int],
    gains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised difference and gradient norm w.r.t. original bands."""

    a = np.sum(optical[list(positive)], axis=0)
    b = np.sum(optical[list(negative)], axis=0)
    denominator = a + b
    value = np.zeros_like(a, dtype=np.float32)
    np.divide(a - b, denominator, out=value, where=np.abs(denominator) > 1e-6)
    gradient_squared = np.zeros_like(a, dtype=np.float32)
    safe_squared = np.square(denominator)
    for index in positive:
        derivative = np.zeros_like(a, dtype=np.float32)
        np.divide(
            2.0 * b * float(gains[index]), safe_squared,
            out=derivative, where=np.abs(denominator) > 1e-6,
        )
        gradient_squared += np.square(derivative)
    for index in negative:
        derivative = np.zeros_like(a, dtype=np.float32)
        np.divide(
            -2.0 * a * float(gains[index]), safe_squared,
            out=derivative, where=np.abs(denominator) > 1e-6,
        )
        gradient_squared += np.square(derivative)
    return np.clip(value, -1.0, 1.0), np.sqrt(gradient_squared)


def _transport_texture_augmentation(
    texture: np.ndarray,
    optical_mean: np.ndarray,
    augmentation: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply Core22's affine draw to stored texture moments.

    Band standard deviations and quantiles have an exact affine transport.
    The three normalised-difference distributions are transported by the
    local value delta and Jacobian ratio evaluated at the target-free 120 m
    optical mean.  This keeps their location and spread responsive to the
    *same* six-band draw; it is a deterministic first-order moment transport,
    not a second independently sampled augmentation.
    """

    gain, offset, dropped_band, applied = _parse_optical_augmentation(augmentation)
    value = np.asarray(texture, dtype=np.float32)
    optical = np.asarray(optical_mean, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 30 or optical.shape != (6, *value.shape[1:]):
        raise ValueError("texture augmentation geometry is invalid")
    output = np.array(value, copy=True)
    if applied:
        for band in range(6):
            start = 3 * band
            output[start] *= gain[band]
            output[start + 1:start + 3] = (
                output[start + 1:start + 3] * gain[band] + offset[band]
            )
        augmented_optical = optical * gain[:, None, None] + offset[:, None, None]
        specs = (((3,), (2,)), ((4,), (3,)), ((1,), (4,)))
        unit_gains = np.ones(6, dtype=np.float32)
        for feature, (positive, negative) in enumerate(specs):
            original_index, original_gradient = _index_value_and_gradient(
                optical, positive, negative, unit_gains
            )
            augmented_index, augmented_gradient = _index_value_and_gradient(
                augmented_optical, positive, negative, gain
            )
            delta = augmented_index - original_index
            scale = np.ones_like(delta, dtype=np.float32)
            np.divide(
                augmented_gradient, original_gradient, out=scale,
                where=original_gradient > 1e-8,
            )
            scale = np.clip(scale, 0.25, 4.0)
            start = 18 + 4 * feature
            original_mean = np.array(output[start], copy=True)
            new_mean = np.clip(original_mean + delta, -1.0, 1.0)
            output[start] = new_mean
            output[start + 1] = np.maximum(output[start + 1] * scale, 0.0)
            output[start + 2:start + 4] = np.clip(
                new_mean[None]
                + scale[None] * (output[start + 2:start + 4] - original_mean[None]),
                -1.0, 1.0,
            )
    dropped_channels: set[int] = set()
    if dropped_band is not None:
        dropped_channels.update(range(3 * dropped_band, 3 * dropped_band + 3))
        dependencies = {
            0: (), 1: (2,), 2: (0,), 3: (0, 1), 4: (1, 2), 5: (),
        }
        for feature in dependencies[dropped_band]:
            dropped_channels.update(range(18 + 4 * feature, 22 + 4 * feature))
    if not np.all(np.isfinite(output)):
        raise ValueError("augmented texture moments are not finite")
    return output, tuple(sorted(dropped_channels))


class MultiSourceTemporalDataset:
    """API-compatible Stage-C wrapper around a strict Core22 temporal dataset."""

    fine_channels = MULTISOURCE_FINE_CHANNELS
    context_dim = MULTISOURCE_CONTEXT_DIM
    time_steps = r2.TIME_STEPS
    fine_channel_names = MULTISOURCE_FINE_CHANNEL_NAMES
    context_names = MULTISOURCE_CONTEXT_NAMES
    optical_source = r2.OPTICAL_SOURCE
    scientific_status = "TARGET_FREE_OPTICAL_TEXTURE_WEATHER_OFFLINE"

    def __init__(
        self,
        base_dataset: r2.R2TemporalDataset,
        *,
        texture_manifest: str | Path,
        weather_manifest: str | Path,
        expected_texture_manifest_sha256: str | None = None,
        expected_weather_manifest_sha256: str | None = None,
        texture_cache_size: int = 128,
        weather_cache_size: int = 768,
        preload_texture_cache: bool = False,
        preload_weather_cache: bool = False,
    ) -> None:
        entries = tuple(getattr(base_dataset, "entries", ()))
        # Fail before reading either manifest if a forged locked role is supplied.
        self._entry_by_id = _entry_map(entries, label="multisource dataset")
        normalization = getattr(base_dataset, "normalization", None)
        if (
            getattr(base_dataset, "fine_channels", None) != r2.FINE_CHANNELS
            or getattr(base_dataset, "context_dim", None) != r2.CONTEXT_DIM
            or tuple(getattr(base_dataset, "fine_channel_names", ()))
            != tuple(r2.FINE_CHANNEL_NAMES)
            or tuple(getattr(base_dataset, "context_names", ())) != tuple(r2.CONTEXT_NAMES)
            or normalization is None
            or getattr(normalization, "optical_source", None) != r2.OPTICAL_SOURCE
            or getattr(normalization, "scientific_status", None) != r2.SCIENTIFIC_STATUS
        ):
            raise ValueError("Stage-C requires the strict v2 Core22/Context12 base dataset")
        fit_scene_ids = tuple(getattr(normalization, "fit_scene_ids", ()))
        normalization_scene_ids: Sequence[str] | None = fit_scene_ids or None
        self.texture_adapter = Texture30Adapter(
            texture_manifest, entries,
            expected_manifest_sha256=expected_texture_manifest_sha256,
            normalization_scene_ids=normalization_scene_ids,
            cache_size=texture_cache_size,
        )
        normalization_fit_count = int(getattr(normalization, "fit_scene_count", -1))
        if (
            not fit_scene_ids
            and normalization_fit_count not in {
                -1, len(self.texture_adapter.normalization_scene_ids)
            }
        ):
            raise ValueError(
                "internal-dev multisource normalization lacks its train-only scene scope"
            )
        if (
            getattr(normalization, "optical_sidecar_manifest_sha256", None)
            != self.texture_adapter.manifest_sha256
            or getattr(normalization, "fit_view_sha256", None)
            != self.texture_adapter.fit_view_sha256
        ):
            raise ValueError("Core22 normalization and Stage-C texture binding differ")
        self.weather_adapter = Weather7Adapter(
            weather_manifest, entries,
            expected_manifest_sha256=expected_weather_manifest_sha256,
            expected_fit_view_sha256=self.texture_adapter.fit_view_sha256,
            expected_validation_view_sha256=self.texture_adapter.validation_view_sha256,
            normalization_scene_ids=self.texture_adapter.normalization_scene_ids,
            cache_size=weather_cache_size,
        )
        if (
            self.weather_adapter.receipt_sha256 != self.texture_adapter.receipt_sha256
            or self.weather_adapter.source_manifest_sha256
            != self.texture_adapter.source_manifest_sha256
            or self.weather_adapter.campaign_id != self.texture_adapter.campaign_id
            or self.weather_adapter.normalization_scope_sha256
            != self.texture_adapter.normalization_scope_sha256
        ):
            raise ValueError("texture and weather public source bindings differ")
        self.texture_cache_record = (
            self.texture_adapter.preload_requested()
            if preload_texture_cache else None
        )
        self.weather_cache_record = (
            self.weather_adapter.preload_requested()
            if preload_weather_cache else None
        )
        self.base_dataset = base_dataset
        self.entries = entries
        self.normalization = normalization
        self.evaluation_size = int(base_dataset.evaluation_size)
        self.seed = int(getattr(base_dataset, "seed", 0))
        self.augment = bool(getattr(base_dataset, "augment", False))
        binding = {
            "schema": MULTISOURCE_SCHEMA,
            "fine_channels": MULTISOURCE_FINE_CHANNELS,
            "context_dim": MULTISOURCE_CONTEXT_DIM,
            "core_fine_channels_preserved": r2.FINE_CHANNELS,
            "core_context_dim_preserved": r2.CONTEXT_DIM,
            "texture_manifest_sha256": self.texture_adapter.manifest_sha256,
            "texture_normalization_sha256": self.texture_adapter.normalization_sha256,
            "texture_active_normalization_sha256": (
                self.texture_adapter.active_normalization_sha256
            ),
            "weather_manifest_sha256": self.weather_adapter.manifest_sha256,
            "weather_normalization_sha256": self.weather_adapter.normalization_sha256,
            "weather_active_normalization_sha256": (
                self.weather_adapter.active_normalization_sha256
            ),
            "normalization_scope": self.texture_adapter.normalization_scope,
            "normalization_scene_count": len(
                self.texture_adapter.normalization_scene_ids
            ),
            "normalization_scope_sha256": (
                self.texture_adapter.normalization_scope_sha256
            ),
            "fit_view_sha256": self.texture_adapter.fit_view_sha256,
            "validation_view_sha256": self.texture_adapter.validation_view_sha256,
            "receipt_sha256": self.texture_adapter.receipt_sha256,
            "source_manifest_sha256": self.texture_adapter.source_manifest_sha256,
            "campaign_id": self.texture_adapter.campaign_id,
            "training_io": "local_sidecars_only_no_network",
            "locked_test_opened": False,
            "target_arrays_opened_by_multisource_adapter": False,
            "optical_texture_augmentation": "shared_affine_and_dropout_v1",
            "texture_index_transport": "local_delta_jacobian_v1",
        }
        binding["binding_sha256"] = _canonical_sha256(binding)
        self._provenance = binding

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[g246_data.G246Scene],
        normalization: r2.R2Normalization,
        *,
        seed: int,
        augment: bool,
        texture_manifest: str | Path,
        weather_manifest: str | Path,
        cache_size: int = 768,
        expected_texture_manifest_sha256: str | None = None,
        expected_weather_manifest_sha256: str | None = None,
        texture_cache_size: int = 128,
        weather_cache_size: int = 768,
        preload_texture_cache: bool = False,
        preload_weather_cache: bool = False,
    ) -> "MultiSourceTemporalDataset":
        base = r2.R2TemporalDataset(
            entries, normalization, seed=seed, augment=augment,
            cache_size=cache_size, texture_manifest=texture_manifest,
            allow_provisional=False,
        )
        return cls(
            base, texture_manifest=texture_manifest, weather_manifest=weather_manifest,
            expected_texture_manifest_sha256=expected_texture_manifest_sha256,
            expected_weather_manifest_sha256=expected_weather_manifest_sha256,
            texture_cache_size=texture_cache_size, weather_cache_size=weather_cache_size,
            preload_texture_cache=preload_texture_cache,
            preload_weather_cache=preload_weather_cache,
        )

    def __len__(self) -> int:
        return self.evaluation_size

    def provenance_record(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)

    def preload_caches(self) -> dict[str, dict[str, int]]:
        """Explicitly switch both local predictor adapters to full residency."""

        self.texture_cache_record = self.texture_adapter.preload_requested()
        self.weather_cache_record = self.weather_adapter.preload_requested()
        return {
            "texture": dict(self.texture_cache_record),
            "weather": dict(self.weather_cache_record),
        }

    def _attach(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(batch)
        fine = output.get("fine")
        context = output.get("context")
        available = output.get("temporal_available")
        scene_ids = output.get("temporal_scene_ids")
        origins = output.get("crop_origin")
        d4_codes = output.get("d4_code")
        optical_augmentations = output.get("optical_augmentation")
        if (
            not isinstance(fine, torch.Tensor) or fine.ndim != 5
            or fine.shape[1] not in {1, r2.TIME_STEPS}
            or fine.shape[2] != r2.FINE_CHANNELS
            or not isinstance(context, torch.Tensor) or context.ndim != 3
            or context.shape[1] != fine.shape[1]
            or context.shape[2] != r2.CONTEXT_DIM
            or not isinstance(available, torch.Tensor)
            or available.shape != fine.shape[:2]
            or not isinstance(scene_ids, list) or len(scene_ids) != fine.shape[0]
            or not isinstance(origins, list) or len(origins) != fine.shape[0]
            or not isinstance(d4_codes, list) or len(d4_codes) != fine.shape[0]
            or not isinstance(optical_augmentations, list)
            or len(optical_augmentations) != fine.shape[0]
        ):
            raise ValueError("Core22 batch lacks the registered Stage-C attachment metadata")
        height, width = int(fine.shape[-2]), int(fine.shape[-1])
        if height != width or height not in {r2.PATCH_SIZE, 160}:
            raise ValueError("Stage-C requires registered 96x96 patches or 160x160 scenes")
        texture_batch = np.empty(
            (fine.shape[0], fine.shape[1], 30, height, width), dtype=np.float32
        )
        weather_batch = np.empty(
            (fine.shape[0], fine.shape[1], len(WEATHER_PARAMETERS)), dtype=np.float32
        )
        availability = available.detach().cpu().numpy().astype(bool, copy=False)
        for sample_index in range(fine.shape[0]):
            if len(scene_ids[sample_index]) != fine.shape[1]:
                raise ValueError("temporal_scene_ids count must match the materialised dates")
            try:
                row, col = map(int, origins[sample_index])
                code = int(d4_codes[sample_index])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid Core22 crop/D4 metadata") from exc
            if row < 0 or col < 0 or row + height > 160 or col + width > 160:
                raise ValueError("Core22 crop lies outside the canonical 160x160 grid")
            if not 0 <= code < 8:
                raise ValueError("Core22 D4 code lies outside [0,7]")
            augmentation = optical_augmentations[sample_index]
            # Validate once per sample even if all dates were dropped.
            _parse_optical_augmentation(augmentation)
            for time_index, scene_id in enumerate(scene_ids[sample_index]):
                entry = self._entry_by_id.get(str(scene_id))
                if entry is None:
                    raise ValueError(f"batch scene is outside Stage-C scope: {scene_id}")
                texture_record = self.texture_adapter.load(entry)
                texture_raw = texture_record["texture30_raw"][
                    :, row:row + height, col:col + width
                ]
                optical_mean = texture_record["optical_mean6"][
                    :, row:row + height, col:col + width
                ]
                coverage = texture_record["coverage"][
                    row:row + height, col:col + width
                ]
                texture_raw = _spatial_transform(texture_raw, code)
                optical_mean = _spatial_transform(optical_mean, code)
                coverage = _spatial_transform(coverage, code)
                texture_raw, dropped_channels = _transport_texture_augmentation(
                    texture_raw, optical_mean, augmentation
                )
                texture = self.texture_adapter.normalize_texture(
                    texture_raw, coverage
                )
                if dropped_channels:
                    texture[np.asarray(dropped_channels, dtype=np.int64)] = 0.0
                weather = self.weather_adapter.load(entry)
                if not availability[sample_index, time_index]:
                    texture = np.zeros_like(texture)
                    weather = np.zeros_like(weather)
                texture_batch[sample_index, time_index] = texture
                weather_batch[sample_index, time_index] = weather
        texture_tensor = torch.from_numpy(texture_batch).to(
            device=fine.device, dtype=fine.dtype, non_blocking=True
        )
        weather_tensor = torch.from_numpy(weather_batch).to(
            device=context.device, dtype=context.dtype, non_blocking=True
        )
        output["fine"] = torch.cat((fine, texture_tensor), dim=2)
        output["context"] = torch.cat((context, weather_tensor), dim=2)
        output["multisource_binding_sha256"] = self._provenance["binding_sha256"]
        return output

    def batch(
        self, batch_index: int, batch_size: int = 32, *, full: bool | None = None,
        query_only: bool = False,
    ) -> dict[str, Any]:
        return self._attach(self.base_dataset.batch(
            batch_index, batch_size, full=full, query_only=query_only,
        ))

    def predictor_only_full_batch(
        self, batch_index: int, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Attach Stage-C sources to the label-free canonical full-scene path."""

        return self._attach(self.base_dataset.predictor_only_full_batch(
            batch_index, batch_size,
        ))

    def predictor_only_temporal_full_batch(
        self, batch_index: int, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Attach Stage-C sources to the label-free aligned T=3 full scenes."""

        return self._attach(
            self.base_dataset.predictor_only_temporal_full_batch(
                batch_index, batch_size,
            )
        )

    def predictor_only_full_evaluation_batch(
        self, start: int, batch_size: int,
    ) -> dict[str, Any]:
        """Attach Stage-C sources to label-free deterministic scene coverage."""

        return self._attach(
            self.base_dataset.predictor_only_full_evaluation_batch(
                start, batch_size,
            )
        )

    def predictor_only_temporal_full_evaluation_batch(
        self, start: int, batch_size: int,
    ) -> dict[str, Any]:
        """Attach Stage-C sources to deterministic label-free T=3 coverage."""

        return self._attach(
            self.base_dataset.predictor_only_temporal_full_evaluation_batch(
                start, batch_size,
            )
        )

    def batch_slice(
        self, batch_index: int, start: int, stop: int, *, full: bool | None = None,
        query_only: bool = False,
    ) -> dict[str, Any]:
        return self._attach(
            self.base_dataset.batch_slice(
                batch_index, start, stop, full=full, query_only=query_only,
            )
        )

    def evaluation_batch(
        self, start: int, batch_size: int, *, query_only: bool = False,
    ) -> dict[str, Any]:
        return self._attach(self.base_dataset.evaluation_batch(
            start, batch_size, query_only=query_only,
        ))


__all__ = [
    "TEXTURE_RAW_CHANNEL_NAMES", "TEXTURE_FINE_CHANNEL_NAMES",
    "WEATHER_PARAMETERS", "WEATHER_CONTEXT_NAMES",
    "MULTISOURCE_FINE_CHANNEL_NAMES", "MULTISOURCE_CONTEXT_NAMES",
    "MULTISOURCE_FINE_CHANNELS", "MULTISOURCE_CONTEXT_DIM",
    "Texture30Adapter", "Weather7Adapter", "MultiSourceTemporalDataset",
]
