"""Leakage-safe GlobalCore data loading for the G246 metric campaign.

Only the public fit and validation views are materialised.  The locked-test
descriptor may be present in the split receipt, but this module never resolves,
stats, hashes, or reads it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_RECEIPT = WORKSPACE / "data/v2/g246_metric_campaign_v1/split_receipt.json"
VIEW_SCHEMA = "uhi-cdc-g246-split-v1"
RECEIPT_SCHEMA = "uhi-cdc-g246-split-v1"
FINE_CHANNELS = 11
CONTEXT_DIM = 5
PATCH_SIZE = 96
SCALE = 4
MACRO_REGIONS = ("us", "china", "europe")


def reject_forbidden_path(path: str | os.PathLike[str]) -> Path:
    """Reject sealed/locked-test paths before any filesystem operation."""
    text = os.fspath(path).replace("\\", "/").casefold()
    if "sealed" in text or "locked_test" in text or "locked-test" in text:
        raise ValueError(f"sealed/locked-test access is forbidden: {path}")
    return Path(path)


def normalize_role(role: str) -> str:
    value = str(role).strip().casefold().replace("_", "+").replace("-", "+")
    if "sealed" in value or "locked" in value or "test" in value:
        raise ValueError(f"sealed/locked-test role is forbidden: {role!r}")
    if value not in {"fit+validation", "validation+fit"}:
        raise ValueError("the only supported role is 'fit+validation'")
    return "fit+validation"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    candidate = reject_forbidden_path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_guarded(path: str | os.PathLike[str]) -> tuple[dict[str, Any], bytes]:
    candidate = reject_forbidden_path(path)
    raw = candidate.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {candidate}")
    return value, raw


def _resolve(raw: str | os.PathLike[str], relative_to: Path) -> Path:
    candidate = reject_forbidden_path(raw)
    result = candidate if candidate.is_absolute() else relative_to / candidate
    result = result.resolve()
    reject_forbidden_path(result)
    return result


def _hex_digest(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} requires a lowercase SHA-256")
    return digest


def _descriptor(record: Any, label: str, root: Path) -> tuple[Path, str, str | None]:
    if not isinstance(record, Mapping):
        raise ValueError(f"split receipt is missing {label}")
    raw_path = record.get("path", record.get("file", record.get("manifest")))
    if isinstance(raw_path, Mapping):
        raw_path = raw_path.get("path", raw_path.get("file"))
    if not isinstance(raw_path, str):
        raise ValueError(f"{label} descriptor has no path")
    # Resolve/guard before hash or stat/read.
    path = _resolve(raw_path, root)
    digest = _hex_digest(record.get("sha256", record.get("manifest_sha256")), label)
    identity = record.get("view_content_sha256", record.get("tier_content_sha256"))
    if identity is not None:
        identity = _hex_digest(identity, f"{label} content identity")
    return path, digest, identity


def _view_descriptor(receipt: Mapping[str, Any], role: str, root: Path) -> tuple[Path, str, str | None]:
    split_files = receipt.get("split_files", receipt.get("views"))
    if not isinstance(split_files, Mapping):
        raise ValueError("split receipt has no split_files")
    file_name = f"{role}.json"
    record = split_files.get(role, split_files.get(file_name))
    if not isinstance(record, Mapping):
        raise ValueError(f"split receipt has no {role} descriptor")
    declared = str(record.get("role", role)).casefold()
    if declared != role:
        raise ValueError(f"{role} descriptor declares role {declared!r}")
    # The frozen receipt keys the descriptor by filename and therefore need
    # not redundantly carry a path inside the value.
    normalized = dict(record)
    normalized.setdefault("path", file_name)
    return _descriptor(normalized, f"{role} view", root)


def canonical_region(region: str) -> str:
    value = str(region).strip().casefold()
    if value == "base":
        return "us"
    if value not in MACRO_REGIONS:
        raise ValueError(f"unsupported G246 region: {region!r}")
    return value


@dataclass(frozen=True)
class G246Scene:
    city: str
    year: int
    region: str
    input_role: str
    item_id: str
    datetime: str
    file: Path
    sha256: str
    view_role: str

    @property
    def scene_id(self) -> str:
        return f"{self.city}:{self.year}:{self.item_id}"


@dataclass(frozen=True)
class G246Splits:
    receipt_path: Path
    receipt_sha256: str
    source_manifest: Path
    source_manifest_sha256: str
    campaign_id: str | None
    fit_view_sha256: str
    validation_view_sha256: str
    fit: tuple[G246Scene, ...]
    validation: tuple[G246Scene, ...]

    def public_record(self) -> dict[str, Any]:
        return {
            "receipt_path": str(self.receipt_path),
            "receipt_sha256": self.receipt_sha256,
            "source_manifest": str(self.source_manifest),
            "source_manifest_sha256": self.source_manifest_sha256,
            "campaign_id": self.campaign_id,
            "fit_view_sha256": self.fit_view_sha256,
            "validation_view_sha256": self.validation_view_sha256,
            "fit_city_count": len({x.city for x in self.fit}),
            "fit_scene_count": len(self.fit),
            "validation_city_count": len({x.city for x in self.validation}),
            "validation_scene_count": len(self.validation),
            "locked_test_opened": False,
        }


def _source_rows(payload: Mapping[str, Any], root: Path) -> dict[tuple[str, int], tuple[Path, str]]:
    if payload.get("sealed_test_opened") is not False or payload.get("sealed_test_unlocked") is not False:
        raise ValueError("source manifest does not keep locked test closed")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("source manifest has no scenes")
    result: dict[tuple[str, int], tuple[Path, str]] = {}
    for row in scenes:
        if not isinstance(row, Mapping):
            raise ValueError("source manifest contains malformed scene")
        key = (str(row.get("city", "")), int(row.get("year", 0)))
        path = _resolve(str(row.get("file", "")), root)
        digest = _hex_digest(row.get("sha256"), "source scene")
        if key in result:
            raise ValueError(f"duplicate source scene {key}")
        result[key] = (path, digest)
    return result


def _parse_view(
    path: Path,
    expected_sha: str,
    expected_identity: str | None,
    expected_role: str,
    source_path: Path,
    source_sha: str,
    source_rows: Mapping[tuple[str, int], tuple[Path, str]],
) -> tuple[G246Scene, ...]:
    payload, raw = _read_json_guarded(path)
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"{expected_role} view SHA-256 mismatch")
    if payload.get("schema_version") != VIEW_SCHEMA:
        raise ValueError(f"unsupported {expected_role} view schema")
    if payload.get("role") != expected_role:
        raise ValueError(f"{expected_role} view declares another role")
    if payload.get("locked") is not False or payload.get("arrays_allowed") is not True:
        raise ValueError(f"{expected_role} arrays are not explicitly allowed")
    if expected_identity is not None and payload.get("view_content_sha256") != expected_identity:
        raise ValueError(f"{expected_role} content identity mismatch")
    binding = payload.get("source_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError(f"{expected_role} view has no source binding")
    bound_path = _resolve(str(binding.get("path", "")), path.parent)
    if bound_path != source_path or binding.get("sha256") != source_sha:
        raise ValueError(f"{expected_role} source binding mismatch")
    rows = payload.get("scenes")
    cities = payload.get("selected_cities")
    if not isinstance(rows, list) or not isinstance(cities, list):
        raise ValueError(f"{expected_role} view requires scenes and selected_cities")
    entries: list[G246Scene] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"malformed {expected_role} scene {index}")
        city, year = str(row.get("city", "")), int(row.get("year", 0))
        key = (city, year)
        if not city or key in seen:
            raise ValueError(f"duplicate/empty {expected_role} scene {key}")
        seen.add(key)
        raw_file = row.get("source_file")
        if not isinstance(raw_file, str):
            raise ValueError(f"{expected_role} scene has no source_file")
        scene_file = _resolve(raw_file, source_path.parent)
        digest = _hex_digest(row.get("sha256"), f"{expected_role} scene")
        canonical = source_rows.get(key)
        if canonical != (scene_file, digest):
            raise ValueError(f"{expected_role} scene redirects or differs from G246: {key}")
        region = canonical_region(str(row.get("region", "")))
        item_id = str(row.get("item_id", "")).strip()
        timestamp = str(row.get("datetime", "")).strip()
        if not item_id or not timestamp:
            raise ValueError(f"{expected_role} scene lacks item_id/datetime")
        entries.append(G246Scene(city, year, region, str(row.get("input_role", "")), item_id,
                                 timestamp, scene_file, digest, expected_role))
    declared_cities = [str(x) for x in cities]
    if len(declared_cities) != len(set(declared_cities)) or {e.city for e in entries} != set(declared_cities):
        raise ValueError(f"{expected_role} selected_cities mismatch")
    if int(payload.get("city_count", -1)) != len(declared_cities) or int(payload.get("scene_count", -1)) != len(entries):
        raise ValueError(f"{expected_role} declared counts mismatch")
    years: dict[str, set[int]] = {}
    for entry in entries:
        years.setdefault(entry.city, set()).add(entry.year)
    if any(len(value) != 3 for value in years.values()):
        raise ValueError(f"{expected_role} requires three distinct years per city")
    return tuple(entries)


def load_splits(receipt_path: str | os.PathLike[str] = DEFAULT_SPLIT_RECEIPT,
                role: str = "fit+validation") -> G246Splits:
    """Load fit+validation, without resolving the locked-test descriptor."""
    normalize_role(role)  # Must precede every filesystem operation.
    receipt_path = reject_forbidden_path(receipt_path).resolve()
    reject_forbidden_path(receipt_path)
    receipt, receipt_raw = _read_json_guarded(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported G246 split receipt schema")
    locked_contract = receipt.get("locked_test")
    if (not isinstance(locked_contract, Mapping)
            or locked_contract.get("locked") is not True
            or locked_contract.get("arrays_allowed") is not False):
        raise ValueError("receipt does not keep locked-test arrays closed")
    root = receipt_path.parent
    source_path, source_sha, _ = _descriptor(receipt.get("source_manifest"), "source manifest", root)
    fit_path, fit_sha, fit_identity = _view_descriptor(receipt, "fit", root)
    val_path, val_sha, val_identity = _view_descriptor(receipt, "validation", root)
    # Intentionally never inspect/resolve the locked_test descriptor.
    source, source_raw = _read_json_guarded(source_path)
    if sha256_bytes(source_raw) != source_sha:
        raise ValueError("G246 source manifest SHA-256 mismatch")
    source_map = _source_rows(source, source_path.parent)
    fit = _parse_view(fit_path, fit_sha, fit_identity, "fit", source_path, source_sha, source_map)
    validation = _parse_view(val_path, val_sha, val_identity, "validation", source_path, source_sha, source_map)
    if {e.city for e in fit} & {e.city for e in validation}:
        raise ValueError("fit and validation cities overlap")
    if {e.item_id for e in fit} & {e.item_id for e in validation}:
        raise ValueError("fit and validation Landsat items overlap")
    campaign_id = receipt.get("campaign_id")
    if campaign_id is not None:
        campaign_id = _hex_digest(campaign_id, "campaign_id")
    return G246Splits(receipt_path, sha256_bytes(receipt_raw), source_path, source_sha,
                      campaign_id, fit_sha, val_sha, fit, validation)


@dataclass(frozen=True)
class GlobalNormalization:
    optical_mean: tuple[float, ...]
    optical_std: tuple[float, ...]
    power_tmax_mean_c: float
    power_tmax_std_c: float
    fit_city_count: int
    fit_scene_count: int
    fit_pixel_count: int
    fit_view_sha256: str

    def __post_init__(self) -> None:
        if len(self.optical_mean) != 6 or len(self.optical_std) != 6:
            raise ValueError("normalization requires six optical channels")
        if min(self.optical_std) <= 0 or self.power_tmax_std_c <= 0:
            raise ValueError("normalization standard deviations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "uhi-cdc-g246-globalcore-normalization-v1", **asdict(self),
                "scope": "fit201_only", "locked_test_opened": False}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalNormalization":
        if payload.get("schema_version") not in (None, "uhi-cdc-g246-globalcore-normalization-v1"):
            raise ValueError("unsupported normalization schema")
        return cls(tuple(float(x) for x in payload["optical_mean"]),
                   tuple(float(x) for x in payload["optical_std"]),
                   float(payload["power_tmax_mean_c"]), float(payload["power_tmax_std_c"]),
                   int(payload["fit_city_count"]), int(payload["fit_scene_count"]),
                   int(payload["fit_pixel_count"]), str(payload["fit_view_sha256"]))


def _load_npz(entry: G246Scene) -> dict[str, Any]:
    reject_forbidden_path(entry.file)
    raw = entry.file.read_bytes()
    if sha256_bytes(raw) != entry.sha256:
        raise ValueError(f"scene SHA-256 mismatch: {entry.scene_id}")
    required = {"optical", "lowres_lst", "target_lst", "valid", "eligible",
                "built_fraction", "water_fraction", "lulc_coverage_fraction120",
                "interpolation_weight120", "pre_lowres_valid120", "coarse_lst480",
                "coarse_valid480", "metadata"}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{entry.scene_id} missing arrays {sorted(missing)}")
        values = {name: np.array(archive[name], copy=True) for name in required if name != "metadata"}
        try:
            metadata = json.loads(str(archive["metadata"]))
        except Exception as exc:
            raise ValueError(f"{entry.scene_id} metadata is malformed") from exc
    optical = np.asarray(values["optical"], dtype=np.float32)
    lowres = np.asarray(values["lowres_lst"], dtype=np.float32)
    if optical.ndim != 3 or optical.shape[0] != 6 or lowres.shape != optical.shape[1:]:
        raise ValueError(f"{entry.scene_id} has invalid optical/lowres geometry")
    h, w = lowres.shape
    if h % SCALE or w % SCALE:
        raise ValueError(f"{entry.scene_id} geometry must be divisible by four")
    for name in ("target_lst", "valid", "eligible", "built_fraction", "water_fraction",
                 "lulc_coverage_fraction120", "interpolation_weight120", "pre_lowres_valid120"):
        if np.asarray(values[name]).shape != (h, w):
            raise ValueError(f"{entry.scene_id} {name} geometry mismatch")
    if metadata.get("item_id") != entry.item_id or str(metadata.get("datetime")) != entry.datetime:
        raise ValueError(f"{entry.scene_id} metadata differs from frozen view")
    power = float(metadata.get("power_t2m_max_c", math.nan))
    if not math.isfinite(power):
        raise ValueError(f"{entry.scene_id} has invalid POWER Tmax")
    return {**values, "metadata": metadata, "power_tmax_c": power}


def fit_normalization(entries: Sequence[G246Scene], fit_view_sha256: str) -> GlobalNormalization:
    if not entries or any(e.view_role != "fit" for e in entries):
        raise ValueError("normalization may use fit-role scenes only")
    count = 0
    total = np.zeros(6, dtype=np.float64)
    total_sq = np.zeros(6, dtype=np.float64)
    powers: list[float] = []
    for entry in entries:
        scene = _load_npz(entry)
        optical = np.asarray(scene["optical"], dtype=np.float64)
        mask = (np.asarray(scene["valid"], dtype=bool)
                & np.asarray(scene["eligible"], dtype=bool)
                & np.all(np.isfinite(optical), axis=0))
        if not np.any(mask):
            raise ValueError(f"fit scene has no finite valid optical pixels: {entry.scene_id}")
        values = optical[:, mask]
        count += values.shape[1]
        total += values.sum(axis=1)
        total_sq += np.square(values).sum(axis=1)
        powers.append(float(scene["power_tmax_c"]))
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0)), 1e-6)
    power_std = max(float(np.std(np.asarray(powers, dtype=np.float64))), 1e-6)
    return GlobalNormalization(tuple(map(float, mean)), tuple(map(float, std)),
                               float(np.mean(powers)), power_std,
                               len({e.city for e in entries}), len(entries), count,
                               _hex_digest(fit_view_sha256, "fit view SHA-256"))


def _parent_mean(lowres: np.ndarray, support: np.ndarray) -> np.ndarray:
    h, w = lowres.shape
    finite = np.isfinite(lowres)
    weights = np.where(finite, support, 0.0).astype(np.float32)
    numer = np.where(finite, lowres, 0.0).reshape(h // 4, 4, w // 4, 4)
    numer = (numer * weights.reshape(h // 4, 4, w // 4, 4)).sum((1, 3))
    denom = weights.reshape(h // 4, 4, w // 4, 4).sum((1, 3))
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)


def encode_globalcore(entry: G246Scene, normalization: GlobalNormalization) -> dict[str, Any]:
    scene = _load_npz(entry)
    optical = np.asarray(scene["optical"], dtype=np.float32)
    lowres = np.asarray(scene["lowres_lst"], dtype=np.float32)
    interpolation_weight = np.asarray(scene["interpolation_weight120"], dtype=np.float32)
    support = np.asarray(scene["pre_lowres_valid120"], dtype=np.float32)
    if not np.all(np.isfinite(interpolation_weight)) or np.any(interpolation_weight < 0) or np.any(interpolation_weight > 1.000001):
        raise ValueError(f"{entry.scene_id} interpolation weight must be finite in [0,1]")
    if not np.all((support == 0) | (support == 1)):
        raise ValueError(f"{entry.scene_id} physical support must be binary")
    coarse = np.asarray(scene["coarse_lst480"], dtype=np.float32)
    coarse_valid = np.asarray(scene["coarse_valid480"], dtype=bool)
    expected_coarse_shape = (lowres.shape[0] // SCALE, lowres.shape[1] // SCALE)
    if coarse.shape != expected_coarse_shape or coarse_valid.shape != expected_coarse_shape:
        raise ValueError(f"{entry.scene_id} coarse geometry mismatch")
    coarse = np.where(coarse_valid, coarse, np.nan).astype(np.float32)
    support_sum = support.reshape(expected_coarse_shape[0], SCALE,
                                  expected_coarse_shape[1], SCALE).sum((1, 3))
    if np.any(coarse_valid & (support_sum <= 0)):
        raise ValueError(f"{entry.scene_id} has an observed coarse parent without support")
    # Fill only the network's fine-grid base input.  The coarse observation
    # retains NaNs for unobserved parents and support retains real zeros.
    parent_up = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)
    finite_values = lowres[np.isfinite(lowres)]
    if finite_values.size == 0:
        raise ValueError(f"{entry.scene_id} has no finite low-resolution Kelvin value")
    fallback = float(np.median(finite_values))
    fine_lowres = np.where(np.isfinite(lowres), lowres,
                           np.where(np.isfinite(parent_up), parent_up, fallback)).astype(np.float32)
    optical_mean = np.asarray(normalization.optical_mean, np.float32)[:, None, None]
    optical_std = np.asarray(normalization.optical_std, np.float32)[:, None, None]
    optical = np.nan_to_num((optical - optical_mean) / optical_std, nan=0.0, posinf=0.0, neginf=0.0)
    aux = [np.nan_to_num(np.asarray(scene[name], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
           for name in ("built_fraction", "water_fraction", "lulc_coverage_fraction120")]
    fine = np.concatenate((fine_lowres[None], interpolation_weight[None], optical,
                           *(x[None] for x in aux)), axis=0)
    timestamp = datetime.fromisoformat(entry.datetime.replace("Z", "+00:00"))
    doy = timestamp.timetuple().tm_yday
    angle = 2.0 * math.pi * (doy - 1) / 365.2425
    platform = str(scene["metadata"].get("platform", "")).casefold()
    if platform not in {"landsat-8", "landsat-9"}:
        raise ValueError(f"{entry.scene_id} unsupported platform {platform!r}")
    context = np.array([(float(scene["power_tmax_c"]) - normalization.power_tmax_mean_c) /
                        normalization.power_tmax_std_c,
                        math.sin(angle), math.cos(angle), float(platform == "landsat-8"),
                        float(platform == "landsat-9")], dtype=np.float32)
    target = np.asarray(scene["target_lst"], np.float32)
    valid = np.asarray(scene["valid"], bool) & np.isfinite(target)
    eligible = np.asarray(scene["eligible"], bool)
    if not np.any(valid & eligible):
        raise ValueError(f"{entry.scene_id} has no primary evaluation pixels")
    return {"fine": fine.astype(np.float32), "coarse_k": coarse[None].astype(np.float32),
            "support120": support[None].astype(np.float32), "context": context,
            "target_k": np.nan_to_num(target, nan=fallback).astype(np.float32)[None],
            "valid": valid[None], "eligible": eligible[None], "city": entry.city,
            "region": entry.region, "year": entry.year, "scene_id": entry.scene_id}


def _stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _spatial_transform(array: np.ndarray, code: int) -> np.ndarray:
    value = np.rot90(array, k=code & 3, axes=(-2, -1))
    if code & 4:
        value = value[..., ::-1]
    return np.ascontiguousarray(value)


class G246Dataset:
    """In-memory deterministic hierarchical sampler for 96/full batches."""

    def __init__(self, entries: Sequence[G246Scene], normalization: GlobalNormalization,
                 *, seed: int, augment: bool) -> None:
        if not entries:
            raise ValueError("dataset requires scenes")
        self.entries = tuple(entries)
        self.normalization = normalization
        self.seed = int(seed)
        self.augment = bool(augment)
        self.scenes = tuple(encode_globalcore(e, normalization) for e in entries)
        self.by_city: dict[str, list[int]] = {}
        self.city_region: dict[str, str] = {}
        for index, entry in enumerate(entries):
            self.by_city.setdefault(entry.city, []).append(index)
            self.city_region[entry.city] = entry.region
        self.region_cities = {r: sorted(c for c, value in self.city_region.items() if value == r)
                              for r in MACRO_REGIONS}
        if augment and any(not cities for cities in self.region_cities.values()):
            raise ValueError("fit sampler requires US, China, and Europe cities")
        self.patch_origins: list[list[tuple[int, int]]] = []
        for scene in self.scenes:
            h, w = scene["target_k"].shape[-2:]
            origins = [(r, c) for r in range(0, h - PATCH_SIZE + 1, SCALE)
                       for c in range(0, w - PATCH_SIZE + 1, SCALE)
                       if np.any((scene["valid"] & scene["eligible"])[..., r:r+PATCH_SIZE, c:c+PATCH_SIZE])]
            if h >= PATCH_SIZE and w >= PATCH_SIZE and not origins:
                raise ValueError(f"{scene['scene_id']} has no eligible 96x96 parent-aligned patch")
            self.patch_origins.append(origins)

    def __len__(self) -> int:
        return len(self.scenes)

    def _city_for_draw(self, draw: int) -> str:
        region_index = (draw + self.seed) % len(MACRO_REGIONS)
        region = MACRO_REGIONS[region_index]
        cycle = (draw + self.seed) // len(MACRO_REGIONS)
        cities = self.region_cities[region]
        epoch, offset = divmod(cycle, len(cities))
        rng = np.random.default_rng(_stable_seed(self.seed, region, epoch, "city-order"))
        return cities[int(rng.permutation(len(cities))[offset])]

    def _sample(self, draw: int, full: bool) -> dict[str, Any]:
        city = self._city_for_draw(draw)
        indices = self.by_city[city]
        rng = np.random.default_rng(_stable_seed(self.seed, draw, city, "scene"))
        scene_index = indices[int(rng.integers(0, len(indices)))]
        source = self.scenes[scene_index]
        if full:
            row = column = 0
            size = source["target_k"].shape[-1]
        else:
            origins = self.patch_origins[scene_index]
            row, column = origins[int(rng.integers(0, len(origins)))]
            size = PATCH_SIZE
        cr, cc, cs = row // SCALE, column // SCALE, size // SCALE
        result = {
            key: np.array(source[key][..., row:row+size, column:column+size], copy=True)
            for key in ("fine", "support120", "target_k", "valid", "eligible")
        }
        result["coarse_k"] = np.array(source["coarse_k"][..., cr:cr+cs, cc:cc+cs], copy=True)
        result["context"] = np.array(source["context"], copy=True)
        if self.augment:
            aug = np.random.default_rng(_stable_seed(self.seed, draw, "augmentation"))
            code = int(aug.integers(0, 8))
            for key in ("fine", "support120", "target_k", "valid", "eligible", "coarse_k"):
                result[key] = _spatial_transform(result[key], code)
            # Jitter already normalized optical channels only.
            gain = aug.uniform(0.97, 1.03, size=(6, 1, 1)).astype(np.float32)
            offset = aug.uniform(-0.05, 0.05, size=(6, 1, 1)).astype(np.float32)
            result["fine"][2:8] = result["fine"][2:8] * gain + offset
            if aug.random() < 0.05:
                result["fine"][2 + int(aug.integers(0, 6))] = 0.0
            if aug.random() < 0.10:
                auxiliary = int(aug.integers(0, 4))
                if auxiliary < 3:
                    result["fine"][8 + auxiliary] = 0.0
                else:
                    result["context"][0] = 0.0
        result.update({key: source[key] for key in ("city", "region", "year", "scene_id")})
        return result

    @staticmethod
    def _collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key in ("fine", "coarse_k", "support120", "context", "target_k", "valid", "eligible"):
            array = np.stack([np.asarray(s[key]) for s in samples])
            tensor = torch.from_numpy(array)
            output[key] = tensor.bool() if key in {"valid", "eligible"} else tensor.float()
        for key in ("city", "region", "year", "scene_id"):
            output[key] = [s[key] for s in samples]
        return output

    def batch(self, start_draw: int, batch_size: int, *, full: bool | None = None) -> dict[str, Any]:
        if batch_size <= 0 or start_draw < 0:
            raise ValueError("invalid draw range")
        # Shape mode is batch-level: three patch microbatches then one full.
        if full is None:
            full = (start_draw // batch_size) % 4 == 3
        return self._collate([self._sample(start_draw + i, bool(full)) for i in range(batch_size)])

    def validation_batch(self, start: int, batch_size: int) -> dict[str, Any]:
        samples = []
        for index in range(start, min(start + batch_size, len(self.scenes))):
            source = self.scenes[index]
            samples.append({key: source[key] for key in ("fine", "coarse_k", "support120",
                                                          "context", "target_k", "valid", "eligible",
                                                          "city", "region", "year", "scene_id")})
        return self._collate(samples)
