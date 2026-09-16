"""Offline, locked-test-safe temporal data adapter for the G246 R2 models.

The module consumes only :class:`g246_data.G246Scene` objects returned by the
guarded public split loader.  It never accepts or resolves a locked-test path.
Auxiliary dates expose predictors only; supervision is loaded for the query
date in a separate code path.  The immediately available Core22 path is
explicitly marked provisional because the frozen ``optical`` array was
aggregated upstream with a target-QA-conditioned mask.  It must not be called
strictly target-free until fresh optical-only 120 m means are supplied.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from . import g246_data
except ImportError:
    import g246_data


MACRO_REGIONS = tuple(g246_data.MACRO_REGIONS)
REGION_TO_INDEX = {name: index for index, name in enumerate(MACRO_REGIONS)}
SCALE = 4
PATCH_SIZE = 96
TIME_STEPS = 3
OPTICAL_REJECT_BITS = (0, 1, 2, 3, 4, 5, 9)
OPTICAL_REJECT_MASK = np.uint16(sum(1 << bit for bit in OPTICAL_REJECT_BITS))
OPTICAL_NAMES = ("blue", "green", "red", "nir", "swir1", "swir2")
OPTICAL_SIDECAR_NAMES = (
    "blue_mean120", "green_mean120", "red_mean120",
    "nir08_mean120", "swir16_mean120", "swir22_mean120",
)
OPTICAL_SIDECAR_MANIFEST_SCHEMA = "uhi-cdc-g246-r2-texture-sidecars-v2"
OPTICAL_SIDECAR_SCENE_SCHEMA = "uhi-cdc-g246-r2-texture-scene-v2"
OPTICAL_SIDECAR_NORMALIZATION_SCHEMA = "uhi-cdc-g246-r2-texture-normalization-v2"
FINE_CHANNEL_NAMES = (
    "base_lst_k", "interpolation_weight",
    *(f"{name}_global_z" for name in OPTICAL_NAMES),
    *(f"{name}_scene_iqr" for name in OPTICAL_NAMES),
    "ndvi", "ndbi", "mndwi", "bsi", "optical_valid",
    "built_fraction", "water_fraction", "lulc_coverage",
)
CONTEXT_NAMES = (
    "power_tmax_z", "doy_sin", "doy_cos", "platform_l8", "platform_l9",
    "latitude_sin", "latitude_cos", "longitude_sin", "longitude_cos",
    "cos_solar_zenith", "solar_azimuth_sin", "solar_azimuth_cos",
)
FINE_CHANNELS = len(FINE_CHANNEL_NAMES)
CONTEXT_DIM = len(CONTEXT_NAMES)
OPTICAL_SOURCE = "fresh_exact_item_b2_b7_optical_qa_sidecar_v2"
SCIENTIFIC_STATUS = "TARGET_FREE_OPTICAL_QA_ONLY"
PROVISIONAL_OPTICAL_SOURCE = "provisional_target_qa_aggregated"
PROVISIONAL_SCIENTIFIC_STATUS = "PROVISIONAL_TARGET_QA_CONDITIONED_OPTICAL"
OPTICAL_AUGMENTATION_SCHEMA = "uhi-cdc-g246-r2-optical-augmentation-v1"
AUXILIARY_MODALITY_AUGMENTATION_SCHEMA = (
    "uhi-cdc-g246-r2-auxiliary-modality-augmentation-v1"
)
AUXILIARY_MODALITY_DROPOUT_PROBABILITY = 0.10
AUXILIARY_MODALITY_DROPOUT_CHOICES = (
    "built", "water", "lulc", "power_tmax",
)
_AUXILIARY_MODALITY_CHANNEL_NAMES = {
    "built": {"fine": ("built_fraction",), "context": ()},
    "water": {"fine": ("water_fraction",), "context": ()},
    "lulc": {"fine": ("lulc_coverage",), "context": ()},
    "power_tmax": {"fine": (), "context": ("power_tmax_z",)},
}
_OPTICAL_GLOBAL_CHANNEL_NAMES = tuple(
    f"{name}_global_z" for name in OPTICAL_NAMES
)
_OPTICAL_IQR_CHANNEL_NAMES = tuple(
    f"{name}_scene_iqr" for name in OPTICAL_NAMES
)
_OPTICAL_INDEX_CHANNEL_NAMES = ("ndvi", "ndbi", "mndwi", "bsi")
PREDICTOR_PREFETCH_WORKERS = 8
if FINE_CHANNELS != 22 or CONTEXT_DIM != 12:  # pragma: no cover - contract guard
    raise RuntimeError("G246 R2 Core contract drifted")
if set(_AUXILIARY_MODALITY_CHANNEL_NAMES) != set(
    AUXILIARY_MODALITY_DROPOUT_CHOICES
):  # pragma: no cover - contract guard
    raise RuntimeError("auxiliary modality dropout choices drifted")
for _modality_channels in _AUXILIARY_MODALITY_CHANNEL_NAMES.values():
    if any(name not in FINE_CHANNEL_NAMES for name in _modality_channels["fine"]) \
            or any(
                name not in CONTEXT_NAMES
                for name in _modality_channels["context"]
            ):  # pragma: no cover - contract guard
        raise RuntimeError("auxiliary modality dropout channel contract drifted")
if any(
    name not in FINE_CHANNEL_NAMES
    for name in (
        *_OPTICAL_GLOBAL_CHANNEL_NAMES,
        *_OPTICAL_IQR_CHANNEL_NAMES,
        *_OPTICAL_INDEX_CHANNEL_NAMES,
        "optical_valid",
    )
):  # pragma: no cover - contract guard
    raise RuntimeError("optical augmentation channel contract drifted")


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _hex_digest(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _guard_public_entry(entry: g246_data.G246Scene) -> None:
    # Check the logical role before even inspecting its path.  This keeps a
    # forged descriptor whose filename omits "locked_test" fail-closed too.
    if entry.view_role not in {"fit", "validation"}:
        raise ValueError(f"R2 accepts only public fit/validation scenes, got {entry.view_role!r}")
    g246_data.reject_forbidden_path(entry.file)


@dataclass(frozen=True)
class R2Normalization:
    """Fit-only statistics with city -> region -> global equal weighting."""

    optical_mean: tuple[float, ...]
    optical_std: tuple[float, ...]
    power_tmax_mean_c: float
    power_tmax_std_c: float
    fit_city_count: int
    fit_scene_count: int
    fit_optical_cell_count: int
    fit_view_sha256: str
    optical_sidecar_manifest_sha256: str | None
    weighting: str = (
        "coverage_within_scene_then_equal_scene_within_city_then_"
        "equal_city_within_region_then_equal_region"
    )
    fine_channels: int = FINE_CHANNELS
    context_dim: int = CONTEXT_DIM
    optical_source: str = OPTICAL_SOURCE
    scientific_status: str = SCIENTIFIC_STATUS
    # Scene identities are predictors-only metadata.  Persisting the exact
    # fitting scope lets optional Stage-C adapters reproduce the Core22
    # internal-dev train-only normalization instead of silently falling back
    # to the complete fit-view moments embedded in their sidecar manifests.
    fit_scene_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.optical_mean) != 6 or len(self.optical_std) != 6:
            raise ValueError("R2 normalization requires six optical channels")
        values = (*self.optical_mean, *self.optical_std,
                  self.power_tmax_mean_c, self.power_tmax_std_c)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("R2 normalization values must be finite")
        if min(self.optical_std) <= 0 or self.power_tmax_std_c <= 0:
            raise ValueError("R2 normalization standard deviations must be positive")
        _hex_digest(self.fit_view_sha256, "fit view SHA-256")
        if self.optical_source == OPTICAL_SOURCE:
            if self.scientific_status != SCIENTIFIC_STATUS:
                raise ValueError("strict optical normalization has inconsistent status")
            _hex_digest(str(self.optical_sidecar_manifest_sha256),
                        "optical sidecar manifest SHA-256")
        elif self.optical_source == PROVISIONAL_OPTICAL_SOURCE:
            if (
                self.scientific_status != PROVISIONAL_SCIENTIFIC_STATUS
                or self.optical_sidecar_manifest_sha256 is not None
            ):
                raise ValueError("provisional optical normalization has inconsistent provenance")
        else:
            raise ValueError(f"unsupported optical normalization source: {self.optical_source}")
        if self.fit_scene_ids:
            if (
                len(self.fit_scene_ids) != self.fit_scene_count
                or len(set(self.fit_scene_ids)) != len(self.fit_scene_ids)
                or any(not str(scene_id).strip() for scene_id in self.fit_scene_ids)
            ):
                raise ValueError("R2 normalization fit-scene identity scope is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "uhi-cdc-g246-r2-normalization-v2",
            **asdict(self),
            "scope": (
                "fit_only_fresh_exact_item_optical_sidecar_v2"
                if self.optical_source == OPTICAL_SOURCE
                else "fit_only_provisional_target_qa_aggregated"
            ),
            "regions": list(MACRO_REGIONS),
            "locked_test_opened": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "R2Normalization":
        if payload.get("schema_version") != "uhi-cdc-g246-r2-normalization-v2":
            raise ValueError("unsupported G246 R2 normalization schema")
        if payload.get("locked_test_opened") not in (None, False):
            raise ValueError("normalization claims locked-test access")
        return cls(
            tuple(float(x) for x in payload["optical_mean"]),
            tuple(float(x) for x in payload["optical_std"]),
            float(payload["power_tmax_mean_c"]),
            float(payload["power_tmax_std_c"]),
            int(payload["fit_city_count"]), int(payload["fit_scene_count"]),
            int(payload["fit_optical_cell_count"]),
            str(payload["fit_view_sha256"]),
            (str(payload["optical_sidecar_manifest_sha256"])
             if payload.get("optical_sidecar_manifest_sha256") is not None else None),
            str(payload.get("weighting", cls.__dataclass_fields__["weighting"].default)),
            int(payload.get("fine_channels", FINE_CHANNELS)),
            int(payload.get("context_dim", CONTEXT_DIM)),
            str(payload.get("optical_source", OPTICAL_SOURCE)),
            str(payload.get("scientific_status", SCIENTIFIC_STATUS)),
            tuple(str(value) for value in payload.get("fit_scene_ids", ())),
        )


@dataclass(frozen=True)
class R2CityGroup:
    city: str
    region: str
    entries: tuple[g246_data.G246Scene, ...]

    def __post_init__(self) -> None:
        if len(self.entries) != TIME_STEPS:
            raise ValueError(f"{self.city} does not have exactly three dates")
        if len({entry.year for entry in self.entries}) != TIME_STEPS:
            raise ValueError(f"{self.city} does not have three distinct years")
        if any(entry.city != self.city or entry.region != self.region for entry in self.entries):
            raise ValueError(f"{self.city} temporal group metadata mismatch")


@dataclass(frozen=True)
class R2DevSplit:
    train_entries: tuple[g246_data.G246Scene, ...]
    dev_entries: tuple[g246_data.G246Scene, ...]
    train_cities: tuple[str, ...]
    dev_cities: tuple[str, ...]
    split_sha256: str


def build_city_groups(entries: Sequence[g246_data.G246Scene]) -> tuple[R2CityGroup, ...]:
    if not entries:
        raise ValueError("temporal grouping requires scenes")
    grouped: dict[str, list[g246_data.G246Scene]] = {}
    for entry in entries:
        _guard_public_entry(entry)
        grouped.setdefault(entry.city, []).append(entry)
    result: list[R2CityGroup] = []
    for city in sorted(grouped):
        rows = sorted(grouped[city], key=lambda x: (x.year, x.datetime, x.item_id))
        region = rows[0].region
        if len({row.item_id for row in rows}) != len(rows):
            raise ValueError(f"{city} repeats a Landsat item")
        result.append(R2CityGroup(city, region, tuple(rows)))
    return tuple(result)


def build_fit201_dev12(
    entries: Sequence[g246_data.G246Scene], fit_view_sha256: str,
    *, salt: str = "g246-r2-dev12-v1",
) -> R2DevSplit:
    """Choose four item-component-isolated fit cities per region by SHA-256."""

    digest = _hex_digest(fit_view_sha256, "fit view SHA-256")
    groups = build_city_groups(entries)
    if any(row.view_role != "fit" for row in entries):
        raise ValueError("dev12 may be derived only from fit-role scenes")
    item_cities: dict[str, set[str]] = {}
    for row in entries:
        item_cities.setdefault(row.item_id, set()).add(row.city)
    # A city is a singleton connected component iff every item used by it is
    # used by that city alone.  The frozen views contain no duplicate scene IDs.
    candidates = {
        region: [
            group.city for group in groups if group.region == region
            and all(item_cities[row.item_id] == {group.city} for row in group.entries)
        ]
        for region in MACRO_REGIONS
    }
    dev: list[str] = []
    for region in MACRO_REGIONS:
        ranked = sorted(
            candidates[region],
            key=lambda city: hashlib.sha256(
                f"{salt}|{digest}|{region}|{city}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < 4:
            raise ValueError(f"{region} lacks four item-isolated dev cities")
        dev.extend(ranked[:4])
    dev_set = set(dev)
    train_entries = tuple(row for row in entries if row.city not in dev_set)
    dev_entries = tuple(row for row in entries if row.city in dev_set)
    document = {
        "schema": "uhi-cdc-g246-r2-fit-dev12-v1", "salt": salt,
        "fit_view_sha256": digest, "dev_cities": sorted(dev),
        "locked_test_opened": False,
    }
    split_sha = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return R2DevSplit(
        train_entries, dev_entries,
        tuple(sorted({x.city for x in train_entries})), tuple(sorted(dev)), split_sha,
    )


def _read_archive(entry: g246_data.G246Scene, required: Sequence[str]) -> dict[str, Any]:
    _guard_public_entry(entry)
    path = g246_data.reject_forbidden_path(entry.file)
    payload = path.read_bytes()
    if g246_data.sha256_bytes(payload) != entry.sha256:
        raise ValueError(f"scene SHA-256 mismatch: {entry.scene_id}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"{entry.scene_id} missing arrays {sorted(missing)}")
        result = {key: np.array(archive[key], copy=True) for key in required if key != "metadata"}
        if "metadata" in required:
            try:
                result["metadata"] = json.loads(str(np.asarray(archive["metadata"]).item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{entry.scene_id} metadata is malformed") from exc
    metadata = result.get("metadata")
    if metadata is not None and (
        metadata.get("item_id") != entry.item_id
        or str(metadata.get("datetime")) != entry.datetime
    ):
        raise ValueError(f"{entry.scene_id} metadata differs from public view")
    return result


class R2OpticalSidecarAdapter:
    """Explicit, offline adapter for fresh target-free B2--B7 means.

    The adapter accepts only the v2 fit+validation manifest produced by
    ``build_g246_r2_texture_sidecars.py``.  It never opens a source manifest
    or resolves a locked-test descriptor.  Requested entries must already be
    public ``G246Scene`` objects, and every loaded sidecar is bound to their
    scene identity and frozen source SHA-256.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        entries: Sequence[g246_data.G246Scene],
        *,
        cache_size: int = 768,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("optical sidecar cache_size must be positive")
        if not entries:
            raise ValueError("optical sidecar adapter requires public entries")
        for entry in entries:
            _guard_public_entry(entry)
        path = g246_data.reject_forbidden_path(manifest_path).resolve()
        g246_data.reject_forbidden_path(path)
        raw = path.read_bytes()
        try:
            documents = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("optical sidecar manifest JSONL is malformed") from exc
        if not documents or any(not isinstance(value, Mapping) for value in documents):
            raise ValueError("optical sidecar manifest must contain JSON objects")
        header, rows = dict(documents[0]), [dict(value) for value in documents[1:]]
        if (
            header.get("kind") != "dataset"
            or header.get("schema") != OPTICAL_SIDECAR_MANIFEST_SCHEMA
            or header.get("status") != "complete"
            or header.get("input_scope") != "g246_data.load_splits:fit+validation"
            or header.get("locked_test_opened") is not False
        ):
            raise ValueError("unsupported or unsafe G246 R2 optical sidecar manifest")
        if int(header.get("scene_count", -1)) != len(rows):
            raise ValueError("optical sidecar manifest scene count mismatch")
        arrays = header.get("arrays")
        optical_contract = arrays.get("optical_mean6") if isinstance(arrays, Mapping) else None
        if (
            not isinstance(optical_contract, Mapping)
            or optical_contract.get("shape") != [6, 160, 160]
            or optical_contract.get("dtype") != "float16"
            or optical_contract.get("channels") != list(OPTICAL_SIDECAR_NAMES)
        ):
            raise ValueError("optical sidecar manifest lacks the v2 optical_mean6 contract")

        row_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("view_role") not in {"fit", "validation"}:
                raise ValueError("optical sidecar manifest contains a non-public role")
            if row.get("locked_test_opened") is not False:
                raise ValueError("optical sidecar record claims locked-test access")
            scene_id = str(row.get("scene_id", ""))
            if not scene_id or scene_id in row_by_id:
                raise ValueError("optical sidecar manifest has duplicate/empty scene identity")
            file_name = row.get("file")
            if not isinstance(file_name, str) or Path(file_name).name != file_name:
                raise ValueError(f"unsafe optical sidecar filename: {file_name!r}")
            g246_data.reject_forbidden_path(file_name)
            _hex_digest(str(row.get("sha256", "")), "optical sidecar SHA-256")
            if int(row.get("bytes", -1)) <= 0:
                raise ValueError("optical sidecar record has invalid byte count")
            row_by_id[scene_id] = row

        requested = {entry.scene_id: entry for entry in entries}
        if len(requested) != len(entries):
            raise ValueError("optical adapter received duplicate public scenes")
        missing = set(requested).difference(row_by_id)
        if missing:
            raise ValueError(f"optical sidecar manifest omits requested scenes: {sorted(missing)[:3]}")
        for scene_id, entry in requested.items():
            row = row_by_id[scene_id]
            expected = {
                "view_role": entry.view_role,
                "region": entry.region,
                "city": entry.city,
                "target_year": int(entry.year),
                "source_scene_sha256": entry.sha256,
                "item_id": entry.item_id,
                "datetime": entry.datetime,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"optical sidecar record differs from public view: {scene_id}")

        normalization_name = header.get("normalization_file")
        if (
            not isinstance(normalization_name, str)
            or Path(normalization_name).name != normalization_name
        ):
            raise ValueError("optical sidecar manifest has unsafe normalization path")
        g246_data.reject_forbidden_path(normalization_name)
        normalization_path = path.parent / normalization_name
        normalization_raw = normalization_path.read_bytes()
        normalization_sha = hashlib.sha256(normalization_raw).hexdigest()
        if normalization_sha != header.get("normalization_sha256"):
            raise ValueError("optical sidecar normalization SHA-256 mismatch")
        try:
            normalization = json.loads(normalization_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("optical sidecar normalization JSON is malformed") from exc
        if (
            not isinstance(normalization, Mapping)
            or normalization.get("schema") != OPTICAL_SIDECAR_NORMALIZATION_SCHEMA
            or normalization.get("scope", {}).get("view_role") != "fit"
            or normalization.get("scope", {}).get("validation_included") is not False
            or normalization.get("scope", {}).get("locked_test_opened") is not False
        ):
            raise ValueError("optical sidecar normalization is not fit-only v2")
        channels = normalization.get("channels")
        if not isinstance(channels, Mapping):
            raise ValueError("optical sidecar normalization has no channels")
        for name in OPTICAL_SIDECAR_NAMES:
            document = channels.get(name)
            if not isinstance(document, Mapping):
                raise ValueError(f"optical sidecar normalization lacks {name}")
            mean, std = float(document.get("mean", math.nan)), float(
                document.get("std", math.nan)
            )
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(f"optical sidecar normalization {name} is invalid")

        self.manifest_path = path
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.header = header
        self.normalization = dict(normalization)
        self._rows = row_by_id
        self._entries = requested
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def normalization_stats(
        self, fit_view_sha256: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        expected = _hex_digest(fit_view_sha256, "fit view SHA-256")
        if self.normalization.get("fit_view_sha256") != expected:
            raise ValueError("optical sidecar normalization fit view differs")
        channels = self.normalization["channels"]
        means = tuple(float(channels[name]["mean"]) for name in OPTICAL_SIDECAR_NAMES)
        stds = tuple(float(channels[name]["std"]) for name in OPTICAL_SIDECAR_NAMES)
        return means, stds

    def load(self, entry: g246_data.G246Scene) -> dict[str, np.ndarray]:
        _guard_public_entry(entry)
        registered = self._entries.get(entry.scene_id)
        if registered != entry:
            raise ValueError(f"scene is outside this optical adapter scope: {entry.scene_id}")
        if entry.scene_id not in self._cache:
            row = self._rows[entry.scene_id]
            path = self.manifest_path.parent / str(row["file"])
            g246_data.reject_forbidden_path(path)
            raw = path.read_bytes()
            if len(raw) != int(row["bytes"]) or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"optical sidecar hash/size mismatch: {entry.scene_id}")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                expected_arrays = {
                    "texture31", "optical_mean6", "channel_names",
                    "optical_channel_names", "metadata",
                }
                if set(archive.files) != expected_arrays:
                    raise ValueError(f"optical sidecar array contract mismatch: {entry.scene_id}")
                texture31 = np.asarray(archive["texture31"])
                optical = np.asarray(archive["optical_mean6"])
                optical_names = tuple(
                    np.asarray(archive["optical_channel_names"]).astype(str).tolist()
                )
                try:
                    metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"optical sidecar metadata is malformed: {entry.scene_id}") from exc
            coverage = np.asarray(texture31[-1], dtype=np.float32)
            expected_metadata = {
                "schema": OPTICAL_SIDECAR_SCENE_SCHEMA,
                "view_role": entry.view_role,
                "region": entry.region,
                "city": entry.city,
                "target_year": int(entry.year),
                "scene_id": entry.scene_id,
                "source_scene_sha256": entry.sha256,
                "item_id": entry.item_id,
                "datetime": entry.datetime,
                "stored_optical_used": False,
                "locked_test_opened": False,
            }
            if (
                texture31.shape != (31, 160, 160)
                or texture31.dtype != np.float16
                or optical.shape != (6, 160, 160)
                or optical.dtype != np.float16
                or optical_names != OPTICAL_SIDECAR_NAMES
                or not np.all(np.isfinite(texture31))
                or not np.all(np.isfinite(optical))
                or np.any(coverage < 0)
                or np.any(coverage > 1)
                or np.any(optical[:, coverage == 0] != 0)
                or any(metadata.get(key) != value for key, value in expected_metadata.items())
            ):
                raise ValueError(f"optical sidecar content is invalid: {entry.scene_id}")
            mask_source = str(metadata.get("optical_mask_source", ""))
            if "bits 0,1,2,3,4,5,9" not in mask_source or "bits 6,7,8 ignored" not in mask_source:
                raise ValueError(f"optical sidecar QA provenance is invalid: {entry.scene_id}")
            self._cache[entry.scene_id] = {
                "optical": optical.astype(np.float32),
                "optical_coverage120": coverage,
                "optical_valid120": (coverage > 0),
            }
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(entry.scene_id)
        cached = self._cache[entry.scene_id]
        return {name: np.array(value, copy=True) for name, value in cached.items()}


class R2ProvisionalOpticalAdapter:
    """Explicit internal-smoke adapter for the target-QA-conditioned optical array."""

    manifest_sha256: None = None

    def __init__(self, entries: Sequence[g246_data.G246Scene]) -> None:
        if not entries:
            raise ValueError("provisional optical adapter requires public entries")
        self._entries = {entry.scene_id: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("provisional optical adapter received duplicate scenes")
        for entry in entries:
            _guard_public_entry(entry)
        self.normalization: dict[str, Any] = {"fit_scene_count": -1}

    def _validate_entry(self, entry: g246_data.G246Scene) -> None:
        if self._entries.get(entry.scene_id) != entry:
            raise ValueError(f"scene is outside provisional adapter scope: {entry.scene_id}")

    def load_from_archive(
        self, entry: g246_data.G246Scene, scene: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Encode optical members already hash-checked with the base predictors."""

        self._validate_entry(entry)
        if "optical" not in scene or "qa_reason30" not in scene:
            raise ValueError(f"{entry.scene_id} provisional archive lacks optical members")
        optical = np.asarray(scene["optical"], dtype=np.float32)
        valid, coverage = _optical_mask(optical, scene["qa_reason30"])
        return {
            "optical": optical,
            "optical_coverage120": coverage,
            "optical_valid120": valid,
        }

    def load(self, entry: g246_data.G246Scene) -> dict[str, np.ndarray]:
        self._validate_entry(entry)
        scene = _read_archive(entry, ("optical", "qa_reason30"))
        return self.load_from_archive(entry, scene)


def _optical_mask(optical: np.ndarray, reason30: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(optical, dtype=np.float32)
    reason = np.asarray(reason30)
    if values.shape != (6, 160, 160) or reason.shape != (640, 640):
        raise ValueError("R2 optical/QA geometry must be [6,160,160] and [640,640]")
    if not np.issubdtype(reason.dtype, np.integer):
        raise ValueError("qa_reason30 must be integral")
    valid30 = (reason.astype(np.uint16) & OPTICAL_REJECT_MASK) == 0
    coverage = valid30.reshape(160, 4, 160, 4).mean((1, 3), dtype=np.float32)
    valid120 = (coverage > 0) & np.all(np.isfinite(values), axis=0)
    return valid120, coverage


def _predictor_raw(
    entry: g246_data.G246Scene,
    optical_adapter: R2OpticalSidecarAdapter | R2ProvisionalOpticalAdapter,
) -> dict[str, Any]:
    """Load source predictors while replacing the upstream optical member.

    Strict mode intentionally omits ``optical`` and ``qa_reason30`` because
    fresh means come from the v2 sidecar.  Provisional mode loads those members
    in the *same* hash-checked NPZ pass as the remaining predictors, avoiding a
    duplicate read without weakening its explicitly provisional provenance.
    """

    names = (
        "lowres_lst", "built_fraction", "water_fraction",
        "lulc_coverage_fraction120", "interpolation_weight120",
        "pre_lowres_valid120", "coarse_lst480", "coarse_valid480",
        "metadata",
    )
    if isinstance(optical_adapter, R2ProvisionalOpticalAdapter):
        combined = _read_archive(entry, (*names, "optical", "qa_reason30"))
        optical = optical_adapter.load_from_archive(entry, combined)
        # Keep the returned predictor contract identical to the former
        # two-read implementation; the raw QA member is not retained.
        scene = {name: combined[name] for name in names}
    else:
        scene = _read_archive(entry, names)
        optical = optical_adapter.load(entry)
    metadata = scene["metadata"]
    center = metadata.get("requested_center_lonlat")
    if not isinstance(center, list) or len(center) != 2 or not all(
        math.isfinite(float(x)) for x in center
    ):
        raise ValueError(f"{entry.scene_id} requested center is invalid")
    power = float(metadata.get("power_t2m_max_c", math.nan))
    if not math.isfinite(power):
        raise ValueError(f"{entry.scene_id} POWER Tmax is invalid")
    return {**scene, **optical, "power_tmax_c": power}


def fit_r2_normalization(
    entries: Sequence[g246_data.G246Scene], fit_view_sha256: str,
    *,
    texture_manifest: str | Path | None = None,
    optical_adapter: R2OpticalSidecarAdapter | None = None,
    allow_provisional: bool = False,
) -> R2Normalization:
    """Compute strict statistics, or explicit internal-smoke provisional ones."""

    groups = build_city_groups(entries)
    if any(row.view_role != "fit" for row in entries):
        raise ValueError("R2 normalization may use fit-role scenes only")
    if allow_provisional:
        if texture_manifest is not None or optical_adapter is not None:
            raise ValueError("provisional mode cannot receive strict optical sidecars")
        adapter: R2OpticalSidecarAdapter | R2ProvisionalOpticalAdapter = (
            R2ProvisionalOpticalAdapter(entries)
        )
    else:
        if (texture_manifest is None) == (optical_adapter is None):
            raise ValueError("strict mode requires exactly one sidecar manifest/adapter")
        adapter = (
            optical_adapter
            if optical_adapter is not None
            else R2OpticalSidecarAdapter(texture_manifest, entries)
        )
    assert adapter is not None
    by_region: dict[str, list[tuple[np.ndarray, np.ndarray, float, float]]] = {
        region: [] for region in MACRO_REGIONS
    }
    cells = 0
    for group in groups:
        scene_means: list[np.ndarray] = []
        scene_seconds: list[np.ndarray] = []
        powers: list[float] = []
        for entry in group.entries:
            scene = _predictor_raw(entry, adapter)
            optical = np.asarray(scene["optical"], dtype=np.float64)
            coverage = np.asarray(scene["optical_coverage120"], dtype=np.float64)
            mask = (
                np.asarray(scene["optical_valid120"], dtype=bool)
                & np.all(np.isfinite(optical), axis=0)
                & np.isfinite(coverage)
                & (coverage > 0)
            )
            if optical.shape != (6, 160, 160) or coverage.shape != (160, 160):
                raise ValueError(f"{entry.scene_id} optical sidecar geometry mismatch")
            if not np.any(mask):
                raise ValueError(f"{entry.scene_id} has no target-free optical support")
            values = optical[:, mask]
            weights = coverage[mask]
            weight = float(weights.sum())
            scene_means.append(np.sum(values * weights[None], axis=1) / weight)
            scene_seconds.append(
                np.sum(np.square(values) * weights[None], axis=1) / weight
            )
            cells += int(values.shape[1])
            powers.append(float(scene["power_tmax_c"]))
        # The three dates are an equal mixture even when cloud-free coverage
        # differs; this matches the sidecar normalization contract.
        mean = np.mean(np.stack(scene_means), axis=0)
        second = np.mean(np.stack(scene_seconds), axis=0)
        pmean = float(np.mean(powers))
        psecond = float(np.mean(np.square(powers)))
        by_region[group.region].append((mean, second, pmean, psecond))
    if any(not by_region[region] for region in MACRO_REGIONS):
        raise ValueError("R2 normalization requires fit cities in every region")
    region_mean = [np.mean([x[0] for x in by_region[r]], axis=0) for r in MACRO_REGIONS]
    region_second = [np.mean([x[1] for x in by_region[r]], axis=0) for r in MACRO_REGIONS]
    mean = np.mean(region_mean, axis=0)
    second = np.mean(region_second, axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0)).clip(1e-6)
    # The published normalization covers the complete fit view.  Internal-dev
    # training deliberately excludes held-out fit cities and therefore keeps
    # its independently recomputed train-only moments.
    if (
        isinstance(adapter, R2OpticalSidecarAdapter)
        and len(entries) == int(adapter.normalization.get("fit_scene_count", -1))
    ):
        registered_mean, registered_std = adapter.normalization_stats(fit_view_sha256)
        if not np.allclose(mean, np.asarray(registered_mean), rtol=0.0, atol=1e-6):
            raise ValueError("recomputed optical means differ from sidecar normalization")
        if not np.allclose(std, np.asarray(registered_std), rtol=0.0, atol=1e-6):
            raise ValueError("recomputed optical stds differ from sidecar normalization")
    power_mean = float(np.mean([
        np.mean([x[2] for x in by_region[r]]) for r in MACRO_REGIONS
    ]))
    power_second = float(np.mean([
        np.mean([x[3] for x in by_region[r]]) for r in MACRO_REGIONS
    ]))
    power_std = max(math.sqrt(max(power_second - power_mean * power_mean, 0.0)), 1e-6)
    return R2Normalization(
        tuple(map(float, mean)), tuple(map(float, std)), power_mean, power_std,
        len(groups), len(entries), cells,
        _hex_digest(fit_view_sha256, "fit view SHA-256"),
        adapter.manifest_sha256,
        optical_source=(
            PROVISIONAL_OPTICAL_SOURCE
            if isinstance(adapter, R2ProvisionalOpticalAdapter) else OPTICAL_SOURCE
        ),
        scientific_status=(
            PROVISIONAL_SCIENTIFIC_STATUS
            if isinstance(adapter, R2ProvisionalOpticalAdapter) else SCIENTIFIC_STATUS
        ),
        fit_scene_ids=tuple(sorted(entry.scene_id for entry in entries)),
    )


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    np.divide(a - b, a + b, out=out, where=np.abs(a + b) > 1e-6)
    return np.clip(out, -1.0, 1.0)


def _solar_features(timestamp: datetime, longitude: float, latitude: float) -> tuple[float, ...]:
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
    hour_angle = math.radians(solar_minutes / 4.0 - 180.0)
    lat = math.radians(latitude)
    cos_zenith = float(np.clip(
        math.sin(lat) * math.sin(declination)
        + math.cos(lat) * math.cos(declination) * math.cos(hour_angle), -1.0, 1.0
    ))
    sin_z = max(math.sqrt(max(1.0 - cos_zenith * cos_zenith, 0.0)), 1e-8)
    sin_az = -math.sin(hour_angle) * math.cos(declination) / sin_z
    cos_az = (
        (math.sin(declination) - math.sin(lat) * cos_zenith)
        / max(math.cos(lat) * sin_z, 1e-8)
    )
    norm = max(math.hypot(sin_az, cos_az), 1e-8)
    return cos_zenith, sin_az / norm, cos_az / norm


def _parent_fill(lowres: np.ndarray, coarse: np.ndarray) -> tuple[np.ndarray, float]:
    base = np.asarray(lowres, dtype=np.float32)
    parent = np.asarray(coarse, dtype=np.float32)
    if base.shape != (160, 160) or parent.shape != (40, 40):
        raise ValueError("R2 base/coarse geometry mismatch")
    finite = base[np.isfinite(base)]
    if finite.size == 0:
        raise ValueError("R2 scene has no finite low-resolution temperatures")
    fallback = float(np.median(finite))
    up = np.repeat(np.repeat(parent, SCALE, axis=0), SCALE, axis=1)
    filled = np.where(np.isfinite(base), base, np.where(np.isfinite(up), up, fallback))
    return filled.astype(np.float32), fallback


def encode_r2_predictors(
    entry: g246_data.G246Scene,
    normalization: R2Normalization,
    optical_adapter: R2OpticalSidecarAdapter | R2ProvisionalOpticalAdapter,
) -> dict[str, Any]:
    """Encode one date without reading target, valid, or eligible arrays."""

    if normalization.optical_sidecar_manifest_sha256 != optical_adapter.manifest_sha256:
        raise ValueError("normalization and optical sidecar manifest differ")
    scene = _predictor_raw(entry, optical_adapter)
    optical = np.asarray(scene["optical"], dtype=np.float32)
    optical_valid = np.asarray(scene["optical_valid120"], dtype=bool)
    support = np.asarray(scene["pre_lowres_valid120"], dtype=np.float32)
    if support.shape != (160, 160) or not np.all((support == 0) | (support == 1)):
        raise ValueError(f"{entry.scene_id} support must be binary [160,160]")
    coarse = np.asarray(scene["coarse_lst480"], dtype=np.float32)
    coarse_valid = np.asarray(scene["coarse_valid480"], dtype=bool)
    if coarse.shape != (40, 40) or coarse_valid.shape != (40, 40):
        raise ValueError(f"{entry.scene_id} coarse geometry mismatch")
    coarse = np.where(coarse_valid, coarse, np.nan).astype(np.float32)
    if np.any(np.isinf(coarse)):
        raise ValueError(f"{entry.scene_id} coarse values contain infinity")
    support_parent = support.reshape(40, 4, 40, 4).sum((1, 3))
    if np.any(coarse_valid & (support_parent <= 0)):
        raise ValueError(f"{entry.scene_id} observed parent has no fine support")
    base, fallback = _parent_fill(scene["lowres_lst"], coarse)
    interpolation = np.asarray(scene["interpolation_weight120"], dtype=np.float32)
    if (
        interpolation.shape != (160, 160)
        or not np.all(np.isfinite(interpolation))
        or np.any(interpolation < 0)
        or np.any(interpolation > 1.000001)
    ):
        raise ValueError(f"{entry.scene_id} interpolation weight is invalid")

    means = np.asarray(normalization.optical_mean, dtype=np.float32)[:, None, None]
    stds = np.asarray(normalization.optical_std, dtype=np.float32)[:, None, None]
    global_z = (optical - means) / stds
    global_z[:, ~optical_valid] = 0.0
    global_z = np.nan_to_num(global_z, nan=0.0, posinf=0.0, neginf=0.0)

    scene_anomaly = np.zeros_like(optical, dtype=np.float32)
    for channel in range(6):
        values = optical[channel, optical_valid]
        if values.size == 0:
            raise ValueError(f"{entry.scene_id} optical channel lacks target-free support")
        median = float(np.median(values))
        q25, q75 = np.quantile(values, (0.25, 0.75))
        scale = max(float(q75 - q25), 1e-6)
        scene_anomaly[channel, optical_valid] = np.clip(
            (values - median) / scale, -8.0, 8.0
        )

    blue, green, red, nir, swir1, _swir2 = optical
    indices = np.stack(
        (
            _normalized_difference(nir, red),
            _normalized_difference(swir1, nir),
            _normalized_difference(green, swir1),
            _normalized_difference(swir1 + red, nir + blue),
        ),
        axis=0,
    )
    indices[:, ~optical_valid] = 0.0

    coverage_arrays: list[np.ndarray] = []
    for name in ("built_fraction", "water_fraction", "lulc_coverage_fraction120"):
        value = np.asarray(scene[name], dtype=np.float32)
        if value.shape != (160, 160):
            raise ValueError(f"{entry.scene_id} {name} geometry mismatch")
        coverage_arrays.append(np.clip(np.nan_to_num(value), 0.0, 1.0))
    fine = np.concatenate(
        (
            base[None], interpolation[None], global_z, scene_anomaly, indices,
            optical_valid.astype(np.float32)[None],
            *(value[None] for value in coverage_arrays),
        ),
        axis=0,
    ).astype(np.float32)
    if fine.shape != (FINE_CHANNELS, 160, 160) or not np.all(np.isfinite(fine)):
        raise ValueError(f"{entry.scene_id} did not encode finite Core22 features")

    metadata = scene["metadata"]
    timestamp = datetime.fromisoformat(entry.datetime.replace("Z", "+00:00"))
    doy_angle = 2.0 * math.pi * (timestamp.timetuple().tm_yday - 1) / 365.2425
    longitude, latitude = map(float, metadata["requested_center_lonlat"])
    lon_rad, lat_rad = math.radians(longitude), math.radians(latitude)
    platform = str(metadata.get("platform", "")).casefold()
    if platform not in {"landsat-8", "landsat-9"}:
        raise ValueError(f"{entry.scene_id} unsupported platform {platform!r}")
    context = np.asarray(
        (
            (float(scene["power_tmax_c"]) - normalization.power_tmax_mean_c)
            / normalization.power_tmax_std_c,
            math.sin(doy_angle), math.cos(doy_angle),
            float(platform == "landsat-8"), float(platform == "landsat-9"),
            math.sin(lat_rad), math.cos(lat_rad),
            math.sin(lon_rad), math.cos(lon_rad),
            *_solar_features(timestamp, longitude, latitude),
        ),
        dtype=np.float32,
    )
    if context.shape != (CONTEXT_DIM,) or not np.all(np.isfinite(context)):
        raise ValueError(f"{entry.scene_id} context is invalid")
    transform = tuple(float(x) for x in metadata.get("transform120", ()))
    crs = str(metadata.get("canonical_grid_crs", ""))
    if len(transform) != 9 or not crs:
        raise ValueError(f"{entry.scene_id} canonical grid metadata is invalid")
    return {
        "fine": fine, "coarse_k": coarse[None], "support": support[None],
        "context": context, "fallback_k": fallback,
        "grid_signature": (crs, transform, (160, 160)),
    }


def _query_supervision(entry: g246_data.G246Scene, fallback_k: float) -> dict[str, np.ndarray]:
    """The only function allowed to materialise query supervision arrays."""

    values = _read_archive(entry, ("target_lst", "valid", "eligible"))
    target = np.asarray(values["target_lst"], dtype=np.float32)
    valid = np.asarray(values["valid"], dtype=bool)
    eligible = np.asarray(values["eligible"], dtype=bool)
    if target.shape != (160, 160) or valid.shape != target.shape or eligible.shape != target.shape:
        raise ValueError(f"{entry.scene_id} supervision geometry mismatch")
    valid &= np.isfinite(target)
    if not np.any(valid) or not np.any(valid & eligible):
        raise ValueError(f"{entry.scene_id} lacks query supervision")
    target = np.where(valid, target, np.float32(fallback_k)).astype(np.float32)
    return {"target_k": target[None], "valid": valid[None], "eligible": eligible[None]}


def _spatial_transform(array: np.ndarray, code: int) -> np.ndarray:
    value = np.rot90(array, k=code & 3, axes=(-2, -1))
    if code & 4:
        value = value[..., ::-1]
    return np.ascontiguousarray(value)


def _transform_solar_azimuth(context: np.ndarray, code: int) -> np.ndarray:
    """Transform east/north solar azimuth components with the raster D4."""

    value = np.array(context, dtype=np.float32, copy=True)
    sin_index = CONTEXT_NAMES.index("solar_azimuth_sin")
    cos_index = CONTEXT_NAMES.index("solar_azimuth_cos")
    east = np.array(value[..., sin_index], copy=True)
    north = np.array(value[..., cos_index], copy=True)
    rotation = code & 3
    if rotation == 0:
        transformed_east, transformed_north = east, north
    elif rotation == 1:
        transformed_east, transformed_north = -north, east
    elif rotation == 2:
        transformed_east, transformed_north = -east, -north
    else:
        transformed_east, transformed_north = north, -east
    if code & 4:  # final horizontal (column/east-axis) flip
        transformed_east = -transformed_east
    value[..., sin_index] = transformed_east
    value[..., cos_index] = transformed_north
    return value


def _augment_optical_core22(
    fine: np.ndarray, rng: np.random.Generator, normalization: R2Normalization,
) -> dict[str, Any]:
    """Perturb reflectance and return the exact reusable augmentation contract.

    Stage-C texture summaries are loaded after the Core22 batch has been
    materialised.  Returning, and later collating, the sampled affine
    parameters is therefore part of the data contract: downstream features
    can apply the same perturbation without consuming another RNG stream or
    inferring parameters from already-normalised tensors.
    """

    global_indices = tuple(
        FINE_CHANNEL_NAMES.index(name) for name in _OPTICAL_GLOBAL_CHANNEL_NAMES
    )
    iqr_indices = tuple(
        FINE_CHANNEL_NAMES.index(name) for name in _OPTICAL_IQR_CHANNEL_NAMES
    )
    index_indices = tuple(
        FINE_CHANNEL_NAMES.index(name) for name in _OPTICAL_INDEX_CHANNEL_NAMES
    )
    optical_valid_index = FINE_CHANNEL_NAMES.index("optical_valid")
    means = np.asarray(normalization.optical_mean, np.float32)[None, :, None, None]
    stds = np.asarray(normalization.optical_std, np.float32)[None, :, None, None]
    valid = fine[:, [optical_valid_index]] > 0.5
    raw = fine[:, global_indices] * stds + means
    affine_shape = (1, len(OPTICAL_NAMES), 1, 1)
    gain = rng.uniform(0.97, 1.03, size=affine_shape).astype(np.float32)
    offset_z = rng.uniform(-0.05, 0.05, size=affine_shape).astype(np.float32)
    offset_raw = offset_z * stds
    augmented = raw * gain + offset_raw
    fine[:, global_indices] = np.where(valid, (augmented - means) / stds, 0.0)
    # Per-scene median/IQR anomalies are invariant to a positive affine
    # transform applied uniformly across the scene, so named IQR channels remain.
    blue, green, red, nir, swir1, _swir2 = (
        augmented[:, index] for index in range(len(OPTICAL_NAMES))
    )
    indices = np.stack(
        (
            _normalized_difference(nir, red),
            _normalized_difference(swir1, nir),
            _normalized_difference(green, swir1),
            _normalized_difference(swir1 + red, nir + blue),
        ),
        axis=1,
    )
    fine[:, index_indices] = np.where(valid, indices, 0.0)
    dropped_band: int | None = None
    if rng.random() < 0.05:
        band = int(rng.integers(0, len(OPTICAL_NAMES)))
        dropped_band = band
        fine[:, global_indices[band]] = 0.0
        fine[:, iqr_indices[band]] = 0.0
        # Remove every nonlinear feature that could reveal the dropped band.
        dependencies = {
            "blue": ("bsi",),
            "green": ("mndwi",),
            "red": ("ndvi", "bsi"),
            "nir": ("ndvi", "ndbi", "bsi"),
            "swir1": ("ndbi", "mndwi", "bsi"),
            "swir2": (),
        }
        for name in dependencies[OPTICAL_NAMES[band]]:
            fine[:, FINE_CHANNEL_NAMES.index(name)] = 0.0
    return {
        "schema": OPTICAL_AUGMENTATION_SCHEMA,
        "applied": True,
        "gain": tuple(float(value) for value in gain.reshape(-1)),
        "offset_raw": tuple(float(value) for value in offset_raw.reshape(-1)),
        "dropped_band": dropped_band,
    }


def _identity_optical_augmentation() -> dict[str, Any]:
    return {
        "schema": OPTICAL_AUGMENTATION_SCHEMA,
        "applied": False,
        "gain": (1.0,) * len(OPTICAL_NAMES),
        "offset_raw": (0.0,) * len(OPTICAL_NAMES),
        "dropped_band": None,
    }


def _identity_auxiliary_modality_augmentation() -> dict[str, Any]:
    """Return the traceable no-op contract used by validation and inference."""

    return {
        "schema": AUXILIARY_MODALITY_AUGMENTATION_SCHEMA,
        "applied": False,
        "dropout_probability": AUXILIARY_MODALITY_DROPOUT_PROBABILITY,
        "dropped_modality": None,
        "fine_channel_names": (),
        "context_channel_names": (),
    }


def _augment_auxiliary_modalities(
    fine: np.ndarray,
    context: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Drop one registered auxiliary modality with probability 0.10.

    The conditional choice is uniform over built, water, LULC, and POWER Tmax.
    Channel positions are always resolved from the public name contracts rather
    than duplicated integer offsets.  A selected modality is removed from every
    materialised date in the sample, including the single query date when
    ``query_only=True``.
    """

    if fine.ndim != 4 or fine.shape[1] != len(FINE_CHANNEL_NAMES):
        raise ValueError("auxiliary modality augmentation requires TCHW Fine22")
    if context.ndim != 2 or context.shape[1] != len(CONTEXT_NAMES) \
            or context.shape[0] != fine.shape[0]:
        raise ValueError("auxiliary modality augmentation requires aligned TC Context12")
    if float(rng.random()) >= AUXILIARY_MODALITY_DROPOUT_PROBABILITY:
        return _identity_auxiliary_modality_augmentation()

    choice_index = int(rng.integers(0, len(AUXILIARY_MODALITY_DROPOUT_CHOICES)))
    modality = AUXILIARY_MODALITY_DROPOUT_CHOICES[choice_index]
    channels = _AUXILIARY_MODALITY_CHANNEL_NAMES[modality]
    fine_names = tuple(channels["fine"])
    context_names = tuple(channels["context"])
    for name in fine_names:
        fine[:, FINE_CHANNEL_NAMES.index(name)] = 0.0
    for name in context_names:
        context[:, CONTEXT_NAMES.index(name)] = 0.0
    return {
        "schema": AUXILIARY_MODALITY_AUGMENTATION_SCHEMA,
        "applied": True,
        "dropout_probability": AUXILIARY_MODALITY_DROPOUT_PROBABILITY,
        "dropped_modality": modality,
        "fine_channel_names": fine_names,
        "context_channel_names": context_names,
    }


class R2TemporalDataset:
    """T=3 query/aux adapter with state-free sampling and deterministic eval."""

    fine_channels = FINE_CHANNELS
    context_dim = CONTEXT_DIM
    time_steps = TIME_STEPS
    fine_channel_names = FINE_CHANNEL_NAMES
    context_names = CONTEXT_NAMES
    optical_source = OPTICAL_SOURCE
    scientific_status = SCIENTIFIC_STATUS

    def __init__(
        self,
        entries: Sequence[g246_data.G246Scene],
        normalization: R2Normalization,
        *,
        seed: int,
        augment: bool,
        cache_size: int = 768,
        texture_manifest: str | Path | None = None,
        allow_provisional: bool = False,
        region_sampling: str = "dataset_proportional",
    ) -> None:
        if texture_manifest is not None and allow_provisional:
            raise ValueError("strict sidecars and provisional optical are mutually exclusive")
        if texture_manifest is None and not allow_provisional:
            raise ValueError("Core22 requires the explicit target-free optical v2 manifest")
        if normalization.fine_channels != FINE_CHANNELS or normalization.context_dim != CONTEXT_DIM:
            raise ValueError("normalization feature dimensions differ from Core22")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.entries = tuple(entries)
        roles = {entry.view_role for entry in self.entries}
        if len(roles) != 1:
            raise ValueError("R2TemporalDataset requires a single public role per instance")
        self.normalization = normalization
        self.optical_adapter: R2OpticalSidecarAdapter | R2ProvisionalOpticalAdapter = (
            R2ProvisionalOpticalAdapter(self.entries)
            if allow_provisional
            else R2OpticalSidecarAdapter(
                texture_manifest, self.entries, cache_size=cache_size
            )
        )
        if (
            normalization.optical_sidecar_manifest_sha256
            != self.optical_adapter.manifest_sha256
        ):
            raise ValueError("normalization is bound to another optical sidecar manifest")
        expected_source = (
            PROVISIONAL_OPTICAL_SOURCE
            if allow_provisional else OPTICAL_SOURCE
        )
        if normalization.optical_source != expected_source:
            raise ValueError("normalization optical provenance differs from dataset mode")
        self.optical_source = normalization.optical_source
        self.scientific_status = normalization.scientific_status
        self.seed = int(seed)
        self.augment = bool(augment)
        self.region_sampling = str(region_sampling)
        if self.region_sampling not in {"equal_region", "dataset_proportional"}:
            raise ValueError("unsupported region sampling contract")
        self.groups = build_city_groups(self.entries)
        self.group_by_city = {group.city: group for group in self.groups}
        self.region_cities = {
            region: tuple(group.city for group in self.groups if group.region == region)
            for region in MACRO_REGIONS
        }
        if any(not self.region_cities[region] for region in MACRO_REGIONS):
            raise ValueError("R2 sampler requires cities in all three regions")
        self._evaluation = tuple(
            (group, query_slot) for group in self.groups for query_slot in range(TIME_STEPS)
        )
        self.evaluation_size = len(self._evaluation)
        self.cache_size = int(cache_size)
        self._predictor_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._predictor_cache_lock = threading.RLock()
        self._label_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._origin_cache: OrderedDict[str, tuple[tuple[int, int], ...]] = OrderedDict()

    def __len__(self) -> int:
        return self.evaluation_size

    @staticmethod
    def registered_region_counts(
        batch_index: int, batch_size: int = 32,
        mode: str = "dataset_proportional",
    ) -> dict[str, int]:
        if batch_index < 0 or batch_size <= 0:
            raise ValueError("invalid batch index/size")
        if mode == "dataset_proportional":
            weights = np.asarray((131.0, 35.0, 35.0))
            tie_order = (0, 1, 2) if batch_index % 2 == 0 else (0, 2, 1)
        elif mode == "equal_region":
            weights = np.ones(len(MACRO_REGIONS), dtype=np.float64)
            offset = batch_index % len(MACRO_REGIONS)
            tie_order = tuple(
                (offset + index) % len(MACRO_REGIONS)
                for index in range(len(MACRO_REGIONS))
            )
        else:
            raise ValueError("unsupported region sampling contract")
        quotas = weights / weights.sum() * batch_size
        counts = np.floor(quotas).astype(int)
        if batch_size >= 3:
            counts = np.maximum(counts, 1)
        remaining = batch_size - int(counts.sum())
        fractions = quotas - np.floor(quotas)
        rank = sorted(tie_order, key=lambda i: (-fractions[i], tie_order.index(i)))
        cursor = 0
        while remaining > 0:
            counts[rank[cursor % 3]] += 1
            remaining -= 1
            cursor += 1
        while remaining < 0:
            choices = [i for i in reversed(rank) if counts[i] > (1 if batch_size >= 3 else 0)]
            counts[choices[0]] -= 1
            remaining += 1
        return {region: int(counts[index]) for index, region in enumerate(MACRO_REGIONS)}

    def _prior_count(self, region: str, batch_index: int, batch_size: int) -> int:
        period = 3 if self.region_sampling == "equal_region" else 2
        cycle_counts = [
            self.registered_region_counts(index, batch_size, self.region_sampling)[region]
            for index in range(period)
        ]
        cycles, extra = divmod(batch_index, period)
        return cycles * sum(cycle_counts) + sum(cycle_counts[:extra])

    def _schedule(self, batch_index: int, batch_size: int) -> list[tuple[R2CityGroup, int]]:
        counts = self.registered_region_counts(
            batch_index, batch_size, self.region_sampling
        )
        scheduled: list[tuple[R2CityGroup, int]] = []
        for region in MACRO_REGIONS:
            start = self._prior_count(region, batch_index, batch_size)
            cities = self.region_cities[region]
            for local in range(counts[region]):
                occurrence = start + local
                epoch, offset = divmod(occurrence, len(cities))
                rng = np.random.default_rng(_stable_seed(self.seed, region, epoch, "city-order"))
                city = cities[int(rng.permutation(len(cities))[offset])]
                scheduled.append((self.group_by_city[city], epoch % TIME_STEPS))
        rng = np.random.default_rng(_stable_seed(self.seed, batch_index, "slot-order"))
        return [scheduled[int(i)] for i in rng.permutation(len(scheduled))]

    def _cached_predictor(self, entry: g246_data.G246Scene) -> dict[str, Any]:
        key = entry.scene_id
        with self._predictor_cache_lock:
            if key not in self._predictor_cache:
                self._predictor_cache[key] = encode_r2_predictors(
                    entry, self.normalization, self.optical_adapter
                )
                if len(self._predictor_cache) > self.cache_size:
                    self._predictor_cache.popitem(last=False)
            self._predictor_cache.move_to_end(key)
            return self._predictor_cache[key]

    @staticmethod
    def _predictor_entries(
        group: R2CityGroup, query_slot: int, query_only: bool,
    ) -> tuple[g246_data.G246Scene, ...]:
        if not 0 <= query_slot < TIME_STEPS:
            raise ValueError("query slot is outside T=3")
        ordered = (group.entries[query_slot],) + tuple(
            entry for index, entry in enumerate(group.entries) if index != query_slot
        )
        return ordered[:1] if query_only else ordered

    def _prefetch_predictors(
        self, entries: Sequence[g246_data.G246Scene],
    ) -> None:
        """Warm unique provisional predictors concurrently, then commit serially.

        Worker threads never mutate either LRU cache.  Strict sidecar loading is
        deliberately excluded because its adapter owns a separate mutable LRU.
        The capacity guard preserves the established serial eviction semantics
        for deliberately tiny caches.
        """

        if not isinstance(self.optical_adapter, R2ProvisionalOpticalAdapter):
            return
        unique: list[g246_data.G246Scene] = []
        seen: set[str] = set()
        for entry in entries:
            if entry.scene_id not in seen:
                seen.add(entry.scene_id)
                unique.append(entry)
        with self._predictor_cache_lock:
            missing = [
                entry for entry in unique
                if entry.scene_id not in self._predictor_cache
            ]
            if (
                not missing
                or len(self._predictor_cache) + len(missing) > self.cache_size
            ):
                return

            def encode(entry: g246_data.G246Scene) -> dict[str, Any]:
                return encode_r2_predictors(
                    entry, self.normalization, self.optical_adapter
                )

            workers = min(PREDICTOR_PREFETCH_WORKERS, len(missing))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="g246-r2-prefetch"
            ) as executor:
                encoded = list(executor.map(encode, missing))
            # executor.map preserves the first-use order.  Commit only after
            # every worker succeeds so a failed preflight leaves no half-cache.
            for entry, value in zip(missing, encoded):
                self._predictor_cache[entry.scene_id] = value

    def _prefetch_schedule(
        self, schedule: Sequence[tuple[R2CityGroup, int]], *, query_only: bool,
    ) -> None:
        entries = [
            entry
            for group, query_slot in schedule
            for entry in self._predictor_entries(group, query_slot, query_only)
        ]
        self._prefetch_predictors(entries)

    def _cached_labels(self, entry: g246_data.G246Scene, fallback: float) -> dict[str, np.ndarray]:
        key = entry.scene_id
        if key not in self._label_cache:
            self._label_cache[key] = _query_supervision(entry, fallback)
            if len(self._label_cache) > self.cache_size:
                self._label_cache.popitem(last=False)
        self._label_cache.move_to_end(key)
        return self._label_cache[key]

    def _origins(self, entry: g246_data.G246Scene, labels: Mapping[str, np.ndarray]) -> tuple[tuple[int, int], ...]:
        key = entry.scene_id
        if key not in self._origin_cache:
            mask = np.asarray(labels["valid"], bool) & np.asarray(labels["eligible"], bool)
            origins = tuple(
                (row, col)
                for row in range(0, 160 - PATCH_SIZE + 1, SCALE)
                for col in range(0, 160 - PATCH_SIZE + 1, SCALE)
                if np.any(mask[..., row:row + PATCH_SIZE, col:col + PATCH_SIZE])
            )
            if not origins:
                raise ValueError(f"{entry.scene_id} has no parent-aligned supervised patch")
            self._origin_cache[key] = origins
            if len(self._origin_cache) > self.cache_size:
                self._origin_cache.popitem(last=False)
        self._origin_cache.move_to_end(key)
        return self._origin_cache[key]

    def _sample(
        self, group: R2CityGroup, query_slot: int, *, full: bool,
        token: Any, do_augment: bool, sample_loss_weight: float,
        query_only: bool = False, include_supervision: bool = True,
    ) -> dict[str, Any]:
        if not include_supervision and (not full or do_augment):
            raise ValueError(
                "predictor-only sampling requires canonical full160 without augmentation"
            )
        ordered_entries = self._predictor_entries(group, query_slot, False)
        # Single-date controls must not pay the I/O/encoding cost of dates that
        # are discarded immediately by ``apply_temporal_mode``.  Keep the
        # query-first ordering (and therefore every sampling/augmentation RNG
        # token) identical to the T=3 path, but never touch the two aux files.
        predictor_entries = ordered_entries[:1] if query_only else ordered_entries
        predictors = [self._cached_predictor(entry) for entry in predictor_entries]
        if len({value["grid_signature"] for value in predictors}) != 1:
            raise ValueError(f"{group.city} dates do not share a canonical grid")
        labels = (
            self._cached_labels(ordered_entries[0], predictors[0]["fallback_k"])
            if include_supervision else None
        )
        rng = np.random.default_rng(_stable_seed(self.seed, token, group.city, query_slot))
        if full:
            row = col = 0
            size = 160
        else:
            if labels is None:  # guarded above; keeps the type contract explicit
                raise RuntimeError("patch sampling requires supervision metadata")
            origins = self._origins(ordered_entries[0], labels)
            row, col = origins[int(rng.integers(0, len(origins)))]
            size = PATCH_SIZE
        parent_row, parent_col, parent_size = row // SCALE, col // SCALE, size // SCALE
        fine = np.stack([x["fine"][..., row:row + size, col:col + size] for x in predictors])
        support = np.stack([x["support"][..., row:row + size, col:col + size] for x in predictors])
        coarse = np.stack([
            x["coarse_k"][..., parent_row:parent_row + parent_size,
                            parent_col:parent_col + parent_size]
            for x in predictors
        ])
        context = np.stack([x["context"] for x in predictors])
        result: dict[str, Any] = {
            "fine": fine, "coarse_k": coarse, "support": support, "context": context,
            "temporal_available": np.ones(len(predictors), dtype=bool),
            "query_index": np.int64(0),
            "sample_loss_weight": np.float32(sample_loss_weight),
            "optical_augmentation": _identity_optical_augmentation(),
            "auxiliary_modality_augmentation": (
                _identity_auxiliary_modality_augmentation()
            ),
        }
        if labels is not None:
            result.update({
                key: np.array(
                    value[..., row:row + size, col:col + size], copy=True,
                )
                for key, value in labels.items()
            })
        if do_augment:
            code = int(rng.integers(0, 8))
            for key in ("fine", "coarse_k", "support", "target_k", "valid", "eligible"):
                result[key] = _spatial_transform(result[key], code)
            result["context"] = _transform_solar_azimuth(result["context"], code)
            result["optical_augmentation"] = _augment_optical_core22(
                result["fine"], rng, self.normalization
            )
            # Keep this independent from the established patch/D4/optical and
            # auxiliary-date RNG stream.  Adding modality dropout must not move
            # any pre-existing temporal-dropout decision after a code upgrade.
            modality_rng = np.random.default_rng(_stable_seed(
                self.seed, token, group.city, query_slot,
                "auxiliary-modality-augmentation",
            ))
            result["auxiliary_modality_augmentation"] = (
                _augment_auxiliary_modalities(
                    result["fine"], result["context"], modality_rng
                )
            )
            if not query_only and rng.random() < 0.10:
                aux = int(rng.integers(1, TIME_STEPS))
                result["fine"][aux] = 0.0
                result["coarse_k"][aux] = np.nan
                result["support"][aux] = 0.0
                result["context"][aux] = 0.0
                result["temporal_available"][aux] = False
        result.update(
            city=group.city, region=group.region,
            region_index=np.int64(REGION_TO_INDEX[group.region]),
            query_year=int(ordered_entries[0].year),
            aux_years=(
                () if query_only
                else tuple(int(entry.year) for entry in ordered_entries[1:])
            ),
            scene_id=ordered_entries[0].scene_id,
            temporal_scene_ids=tuple(entry.scene_id for entry in predictor_entries),
            crop_origin=(row, col), d4_code=(code if do_augment else 0),
        )
        return result

    @staticmethod
    def _collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty R2 batch")
        supervised = "target_k" in samples[0]
        if any(("target_k" in sample) != supervised for sample in samples):
            raise ValueError("cannot mix supervised and predictor-only samples")
        output: dict[str, Any] = {}
        float_keys = ("fine", "coarse_k", "support", "context", "sample_loss_weight")
        bool_keys = ("temporal_available",)
        if supervised:
            float_keys = (*float_keys, "target_k")
            bool_keys = (*bool_keys, "valid", "eligible")
        int_keys = ("query_index", "region_index")
        for key in float_keys:
            output[key] = torch.from_numpy(np.stack([np.asarray(x[key]) for x in samples])).float()
        for key in bool_keys:
            output[key] = torch.from_numpy(np.stack([np.asarray(x[key]) for x in samples])).bool()
        for key in int_keys:
            output[key] = torch.from_numpy(np.stack([np.asarray(x[key]) for x in samples])).long()
        for key in (
            "city", "region", "query_year", "aux_years", "scene_id",
            "temporal_scene_ids", "crop_origin", "d4_code",
            "optical_augmentation",
            "auxiliary_modality_augmentation",
        ):
            output[key] = [x[key] for x in samples]
        return output

    def predictor_only_full_batch(
        self, batch_index: int, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Materialise canonical query predictors without opening label arrays.

        This narrow path exists for function distillation.  It is deliberately
        full160-only: patch selection otherwise depends on ``valid``/``eligible``
        labels, and a transformed/cropped teacher would not share the student's
        receptive-field view.
        """

        if batch_index < 0 or batch_size <= 0:
            raise ValueError("invalid batch index/size")
        schedule = self._schedule(batch_index, batch_size)
        self._prefetch_schedule(schedule, query_only=True)
        counts = {
            region: sum(group.region == region for group, _ in schedule)
            for region in MACRO_REGIONS
        }
        present = sum(count > 0 for count in counts.values())
        samples = [
            self._sample(
                group,
                query_slot,
                full=True,
                token=("predictor-only-full", batch_index, position),
                do_augment=False,
                sample_loss_weight=1.0 / (present * counts[group.region]),
                query_only=True,
                include_supervision=False,
            )
            for position, (group, query_slot) in enumerate(schedule)
        ]
        return self._collate(samples)

    def predictor_only_temporal_full_batch(
        self, batch_index: int, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Materialise aligned T=3 predictors without opening label arrays.

        The registered query-first triplet, region schedule and logical-batch
        weights are identical to the ordinary sampler.  This pretext-only path
        is deliberately canonical full160 with no D4 or stochastic
        augmentation, so all three dates share one exact spatial lattice and
        no patch decision can depend on target/valid/eligible arrays.
        """

        if batch_index < 0 or batch_size <= 0:
            raise ValueError("invalid batch index/size")
        schedule = self._schedule(batch_index, batch_size)
        self._prefetch_schedule(schedule, query_only=False)
        counts = {
            region: sum(group.region == region for group, _ in schedule)
            for region in MACRO_REGIONS
        }
        present = sum(count > 0 for count in counts.values())
        samples = [
            self._sample(
                group,
                query_slot,
                full=True,
                token=("predictor-only-temporal-full", batch_index, position),
                do_augment=False,
                sample_loss_weight=1.0 / (present * counts[group.region]),
                query_only=False,
                include_supervision=False,
            )
            for position, (group, query_slot) in enumerate(schedule)
        ]
        return self._collate(samples)

    def predictor_only_full_evaluation_batch(
        self, start: int, batch_size: int,
    ) -> dict[str, Any]:
        """Cover each public scene once without opening any supervision array."""

        if start < 0 or batch_size <= 0 or start >= self.evaluation_size:
            raise ValueError("invalid evaluation range")
        evaluation_counts = {
            region: sum(group.region == region for group, _ in self._evaluation)
            for region in MACRO_REGIONS
        }
        selected = self._evaluation[
            start:min(start + batch_size, self.evaluation_size)
        ]
        self._prefetch_schedule(selected, query_only=True)
        samples = [
            self._sample(
                group,
                query_slot,
                full=True,
                token=("predictor-only-evaluation", index),
                do_augment=False,
                sample_loss_weight=(
                    1.0 / (3.0 * evaluation_counts[group.region])
                ),
                query_only=True,
                include_supervision=False,
            )
            for index, (group, query_slot) in enumerate(
                selected, start=start,
            )
        ]
        return self._collate(samples)

    def predictor_only_temporal_full_evaluation_batch(
        self, start: int, batch_size: int,
    ) -> dict[str, Any]:
        """Cover each public query scene once with its label-free T=3 triplet."""

        if start < 0 or batch_size <= 0 or start >= self.evaluation_size:
            raise ValueError("invalid evaluation range")
        evaluation_counts = {
            region: sum(group.region == region for group, _ in self._evaluation)
            for region in MACRO_REGIONS
        }
        selected = self._evaluation[
            start:min(start + batch_size, self.evaluation_size)
        ]
        self._prefetch_schedule(selected, query_only=False)
        samples = [
            self._sample(
                group,
                query_slot,
                full=True,
                token=("predictor-only-temporal-evaluation", index),
                do_augment=False,
                sample_loss_weight=(
                    1.0 / (3.0 * evaluation_counts[group.region])
                ),
                query_only=False,
                include_supervision=False,
            )
            for index, (group, query_slot) in enumerate(
                selected, start=start,
            )
        ]
        return self._collate(samples)

    def batch(
        self, batch_index: int, batch_size: int = 32, *, full: bool | None = None,
        query_only: bool = False,
    ) -> dict[str, Any]:
        if batch_index < 0 or batch_size <= 0:
            raise ValueError("invalid batch index/size")
        if full is None:
            full = batch_index % 4 == 3
        schedule = self._schedule(batch_index, batch_size)
        self._prefetch_schedule(schedule, query_only=query_only)
        counts = {
            region: sum(group.region == region for group, _ in schedule)
            for region in MACRO_REGIONS
        }
        present = sum(count > 0 for count in counts.values())
        samples = [
            self._sample(group, query_slot, full=bool(full),
                         token=(batch_index, position), do_augment=self.augment,
                         sample_loss_weight=1.0 / (present * counts[group.region]),
                         query_only=query_only)
            for position, (group, query_slot) in enumerate(schedule)
        ]
        return self._collate(samples)

    def batch_slice(
        self, batch_index: int, start: int, stop: int, *, full: bool | None = None,
        query_only: bool = False,
    ) -> dict[str, Any]:
        """Materialise a slice of the registered logical batch of 32.

        ``sample_loss_weight`` remains normalized against the complete
        region-balanced logical batch.  Summing losses over all slices gives exactly
        the unsliced objective without retaining the full batch in GPU memory.
        """

        if batch_index < 0 or not 0 <= start < stop <= 32:
            raise ValueError("batch_slice requires 0 <= start < stop <= 32")
        if full is None:
            full = batch_index % 4 == 3
        schedule = self._schedule(batch_index, 32)
        sliced_schedule = schedule[start:stop]
        self._prefetch_schedule(sliced_schedule, query_only=query_only)
        counts = {
            region: sum(group.region == region for group, _ in schedule)
            for region in MACRO_REGIONS
        }
        samples = [
            self._sample(
                schedule[position][0], schedule[position][1], full=bool(full),
                token=(batch_index, position), do_augment=self.augment,
                sample_loss_weight=1.0 / (3.0 * counts[schedule[position][0].region]),
                query_only=query_only,
            )
            for position in range(start, stop)
        ]
        return self._collate(samples)

    def evaluation_batch(
        self, start: int, batch_size: int, *, query_only: bool = False,
    ) -> dict[str, Any]:
        """Cover each scene once as query; full-frame, stable order, no augmentation."""

        if start < 0 or batch_size <= 0 or start >= self.evaluation_size:
            raise ValueError("invalid evaluation range")
        evaluation_counts = {
            region: sum(group.region == region for group, _ in self._evaluation)
            for region in MACRO_REGIONS
        }
        selected = self._evaluation[
            start:min(start + batch_size, self.evaluation_size)
        ]
        self._prefetch_schedule(selected, query_only=query_only)
        samples = [
            self._sample(
                group, query_slot, full=True, token=("evaluation", index),
                do_augment=False,
                sample_loss_weight=1.0 / (3.0 * evaluation_counts[group.region]),
                query_only=query_only,
            )
            for index, (group, query_slot) in enumerate(
                selected, start=start
            )
        ]
        return self._collate(samples)


def region_equal_per_scene_loss(
    prediction_k: torch.Tensor,
    target_k: torch.Tensor,
    valid: torch.Tensor,
    eligible: torch.Tensor,
    region_index: torch.Tensor,
    *,
    eligible_weight: float = 0.8,
    sample_loss_weight: torch.Tensor | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """0.8 eligible-valid + 0.2 valid MSE, scene first and region equal."""

    if prediction_k.shape != target_k.shape or prediction_k.ndim != 4:
        raise ValueError("prediction/target must share [B,1,H,W]")
    if valid.shape != prediction_k.shape or eligible.shape != prediction_k.shape:
        raise ValueError("loss masks must match prediction shape")
    if region_index.shape != (prediction_k.shape[0],):
        raise ValueError("region_index must have shape [B]")
    if not 0.0 <= eligible_weight <= 1.0:
        raise ValueError("eligible_weight must be in [0,1]")
    batch = prediction_k.shape[0]
    # Autocast models may emit float16 Kelvin values.  Squaring a normal
    # 300 K initial error in float16 exceeds 65,504, so loss arithmetic is
    # unconditionally float32.
    squared = torch.square(
        prediction_k.float() - target_k.float()
    ).reshape(batch, -1)
    mask_all = valid.bool().reshape(batch, -1)
    mask_primary = (valid.bool() & eligible.bool()).reshape(batch, -1)
    count_all = mask_all.sum(dim=1)
    count_primary = mask_primary.sum(dim=1)
    invalid_batch = (
        (count_all <= 0).any()
        | (count_primary <= 0).any()
        | (region_index < 0).any()
        | (region_index >= len(MACRO_REGIONS)).any()
    )
    message = "each query requires valid+eligible pixels and a registered region index"
    if invalid_batch.device.type == "cuda" and hasattr(torch, "_assert_async"):
        # Device-side assertion keeps the ordinary CUDA hot path asynchronous.
        torch._assert_async(~invalid_batch, message)
    elif bool(invalid_batch):
        raise ValueError(message)
    primary_losses = (
        torch.where(mask_primary, squared, 0.0).sum(dim=1)
        / count_primary.to(squared.dtype)
    )
    all_valid_losses = (
        torch.where(mask_all, squared, 0.0).sum(dim=1)
        / count_all.to(squared.dtype)
    )
    stacked = eligible_weight * primary_losses + (1.0 - eligible_weight) * all_valid_losses
    region_sums = stacked.new_zeros(len(MACRO_REGIONS))
    region_counts = stacked.new_zeros(len(MACRO_REGIONS))
    region_sums.scatter_add_(0, region_index, stacked)
    region_counts.scatter_add_(0, region_index, torch.ones_like(stacked))
    present = region_counts > 0
    region_means = region_sums / region_counts.clamp_min(1.0)
    present_float = present.to(region_means.dtype)
    if sample_loss_weight is None:
        total = (region_means * present_float).sum() / present_float.sum().clamp_min(1.0)
    else:
        if sample_loss_weight.shape != (batch,):
            raise ValueError("sample_loss_weight must have shape [B]")
        weights = sample_loss_weight.to(device=stacked.device, dtype=stacked.dtype)
        if weights.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(torch.all(torch.isfinite(weights) & (weights >= 0)),
                                "sample_loss_weight must be finite and nonnegative")
        elif not bool(torch.all(torch.isfinite(weights) & (weights >= 0))):
            raise ValueError("sample_loss_weight must be finite and nonnegative")
        # Registered logical-batch weights sum to one.  Summing weighted
        # microbatch losses across gradient-accumulation steps exactly
        # reconstructs the logical region-equal objective.
        total = torch.sum(stacked * weights)
    if not return_details:
        return total
    return total, {
        "scene_loss": stacked,
        "eligible_mse": primary_losses,
        "valid_mse": all_valid_losses,
        "region_loss": torch.where(
            present, region_means, torch.full_like(region_means, torch.nan)
        ),
        "region_present": present,
    }


__all__ = [
    "FINE_CHANNELS", "CONTEXT_DIM", "FINE_CHANNEL_NAMES", "CONTEXT_NAMES",
    "OPTICAL_AUGMENTATION_SCHEMA", "AUXILIARY_MODALITY_AUGMENTATION_SCHEMA",
    "AUXILIARY_MODALITY_DROPOUT_PROBABILITY",
    "AUXILIARY_MODALITY_DROPOUT_CHOICES",
    "R2Normalization", "R2CityGroup", "R2DevSplit", "R2TemporalDataset",
    "build_city_groups", "build_fit201_dev12", "fit_r2_normalization",
    "encode_r2_predictors", "region_equal_per_scene_loss",
]
