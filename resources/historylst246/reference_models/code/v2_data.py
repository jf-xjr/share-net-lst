"""Leakage-safe data utilities for the delivered-product v2.1 thermal task.

The only supported manifest roles are ``source`` and ``validation``.  A role is
supplied by the caller and checked against any split declared in the manifest;
normalization can only be fit from entries carrying the source role.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


HEIGHT = 160
WIDTH = 160
OPTICAL_CHANNELS = 6
INPUT_CHANNELS = 8
PATCH_SIZE = 64
EXPECTED_SCENE_YEARS = (2021, 2022, 2023)
Role = Literal["source", "validation"]

_ROLE_ALIASES = {
    "source": {"source", "train", "training"},
    "validation": {"validation", "val", "valid"},
}


def reject_sealed_path(path: os.PathLike[str] | str) -> None:
    """Reject a sealed-test path before any stat, open, or manifest read."""
    text = os.fspath(path).replace("\\", "/").casefold()
    if "sealed_test" in text:
        raise ValueError(f"sealed-test access is forbidden: {path}")


def _normalise_role(value: str) -> Role:
    lowered = str(value).strip().casefold()
    for canonical, aliases in _ROLE_ALIASES.items():
        if lowered in aliases:
            return canonical  # type: ignore[return-value]
    if "sealed_test" in lowered:
        raise ValueError("sealed-test manifests/splits are forbidden")
    raise ValueError(f"unsupported v2 split/role: {value!r}")


@dataclass(frozen=True)
class SceneEntry:
    city: str
    year: int
    file: Path
    role: Role
    manifest: Path
    sha256: str | None = None

    @property
    def scene_id(self) -> str:
        return f"{self.city}:{self.year}:{self.file.stem}"


@dataclass(frozen=True)
class V2Normalization:
    optical_mean: tuple[float, ...]
    optical_std: tuple[float, ...]
    lowres_mean: float
    lowres_std: float
    residual_mean: float
    residual_std: float
    n_source_scenes: int
    n_source_pixels: int

    def __post_init__(self) -> None:
        if len(self.optical_mean) != OPTICAL_CHANNELS or len(self.optical_std) != OPTICAL_CHANNELS:
            raise ValueError("normalization must contain six optical channels")
        if min(self.optical_std) <= 0 or self.lowres_std <= 0 or self.residual_std <= 0:
            raise ValueError("all normalization standard deviations must be positive")
        if self.n_source_scenes <= 0 or self.n_source_pixels <= 0:
            raise ValueError("normalization provenance counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "V2Normalization":
        return cls(
            optical_mean=tuple(float(x) for x in values["optical_mean"]),
            optical_std=tuple(float(x) for x in values["optical_std"]),
            lowres_mean=float(values["lowres_mean"]),
            lowres_std=float(values["lowres_std"]),
            residual_mean=float(values["residual_mean"]),
            residual_std=float(values["residual_std"]),
            n_source_scenes=int(values["n_source_scenes"]),
            n_source_pixels=int(values["n_source_pixels"]),
        )


def _manifest_path(path: os.PathLike[str] | str) -> Path:
    reject_sealed_path(path)
    candidate = Path(path).resolve()
    reject_sealed_path(candidate)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    reject_sealed_path(candidate)
    return candidate


def _check_declared_role(value: Any, expected: Role, location: str) -> None:
    if value is None:
        return
    declared = _normalise_role(str(value))
    if declared != expected:
        raise ValueError(f"{location} declares {declared!r}, expected {expected!r}")


def read_manifest(
    path: os.PathLike[str] | str,
    role: Role,
    *,
    expected_manifest_sha256: str | None = None,
    expected_cities: Sequence[str] | None = None,
) -> list[SceneEntry]:
    """Read one explicitly source or validation manifest after sealed guards.

    Scene files must be relative descendants of the manifest directory.  This
    prevents a benign-looking source manifest from redirecting the loader into
    an unrelated or sealed directory.
    """
    expected = _normalise_role(role)
    manifest_path = _manifest_path(path)
    manifest_bytes = manifest_path.read_bytes()
    if expected_manifest_sha256 is not None:
        expected_hash = str(expected_manifest_sha256).strip().lower()
        if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
            raise ValueError("expected manifest SHA-256 must be 64 lowercase hex characters")
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
            raise ValueError("manifest bytes differ from the training-gate SHA-256")
    payload = json.loads(manifest_bytes)
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    if "sealed_test_unlocked" in payload and payload["sealed_test_unlocked"] is not False:
        raise ValueError("manifest must report sealed_test_unlocked=false")
    _check_declared_role(payload.get("split"), expected, "manifest")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain a non-empty scenes list")
    frozen_cities: list[str] | None = None
    if expected_cities is not None:
        frozen_cities = [str(city) for city in expected_cities]
        if not frozen_cities or len(frozen_cities) != len(set(frozen_cities)):
            raise ValueError("expected city list must be non-empty and unique")
        if payload.get("selected_cities") != frozen_cities:
            raise ValueError("manifest selected cities differ from the training gate")

    root = manifest_path.parent.resolve()
    entries: list[SceneEntry] = []
    seen: set[tuple[str, int, Path]] = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"scene {index} must be a JSON object")
        missing = {key for key in ("city", "year", "file") if key not in scene}
        if missing:
            raise ValueError(f"scene {index} is missing {sorted(missing)}")
        _check_declared_role(scene.get("split"), expected, f"scene {index}")
        if scene.get("accepted") is False:
            raise ValueError(f"scene {index} is explicitly marked unaccepted")

        city = str(scene["city"]).strip()
        if not city:
            raise ValueError(f"scene {index} has an empty city")
        if "sealed_test" in city.casefold():
            raise ValueError(f"scene {index} has a forbidden sealed-test city")
        try:
            year = int(scene["year"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scene {index} has an invalid year") from exc
        if str(year) != str(scene["year"]).strip() or not 1900 <= year <= 2200:
            raise ValueError(f"scene {index} has an invalid year: {scene['year']!r}")

        relative_file = Path(str(scene["file"]))
        reject_sealed_path(relative_file)
        if relative_file.is_absolute():
            raise ValueError(f"scene {index} file must be relative to its manifest")
        scene_file = (root / relative_file).resolve()
        reject_sealed_path(scene_file)
        try:
            scene_file.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"scene {index} escapes the manifest directory") from exc

        scene_sha = scene.get("sha256")
        if scene_sha is not None:
            scene_sha = str(scene_sha).strip().lower()
            if len(scene_sha) != 64 or any(
                char not in "0123456789abcdef" for char in scene_sha
            ):
                raise ValueError(f"scene {index} has an invalid SHA-256")
        if expected_manifest_sha256 is not None and scene_sha is None:
            raise ValueError(f"scene {index} is unhashed in a scientific manifest")

        identity = (city, year, scene_file)
        if identity in seen:
            raise ValueError(f"duplicate scene entry: {city} {year} {relative_file}")
        seen.add(identity)
        entries.append(SceneEntry(city, year, scene_file, expected, manifest_path, scene_sha))
    if frozen_cities is not None:
        scene_cities = {entry.city for entry in entries}
        if scene_cities != set(frozen_cities):
            raise ValueError("manifest scene cities differ from the training gate")
    return entries


def require_complete_city_triplets(entries: Sequence[SceneEntry]) -> None:
    """Reject any city not represented by exactly the three frozen years."""
    if not entries:
        raise ValueError("manifest contains no scene entries")
    grouped: dict[str, list[int]] = {}
    for entry in entries:
        grouped.setdefault(entry.city, []).append(entry.year)
    expected = list(EXPECTED_SCENE_YEARS)
    problems = {
        city: sorted(years)
        for city, years in grouped.items()
        if sorted(years) != expected
    }
    if problems:
        raise ValueError(
            f"every training city must have exactly years {expected}; invalid={problems}"
        )


def load_scene(entry: SceneEntry) -> dict[str, np.ndarray]:
    """Load and validate one v2 NPZ without permitting pickle payloads."""
    reject_sealed_path(entry.file)
    scene_bytes = entry.file.read_bytes()
    if entry.sha256 is not None and hashlib.sha256(scene_bytes).hexdigest() != entry.sha256:
        raise ValueError(f"{entry.file}: bytes differ from manifest SHA-256")
    required = {
        "optical", "lowres_lst", "interpolation_weight120", "target_lst", "valid", "eligible"
    }
    with np.load(io.BytesIO(scene_bytes), allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{entry.file} is missing arrays {sorted(missing)}")
        optical = np.asarray(archive["optical"], dtype=np.float32)
        lowres = np.asarray(archive["lowres_lst"], dtype=np.float32)
        input_support = np.asarray(archive["interpolation_weight120"], dtype=np.float32)
        target = np.asarray(archive["target_lst"], dtype=np.float32)
        valid = np.asarray(archive["valid"]).astype(bool, copy=False)
        eligible = np.asarray(archive["eligible"]).astype(bool, copy=False)

    expected_field = (HEIGHT, WIDTH)
    if optical.shape != (OPTICAL_CHANNELS, HEIGHT, WIDTH):
        raise ValueError(f"{entry.file}: optical shape {optical.shape}, expected (6, 160, 160)")
    for name, array in (
        ("lowres_lst", lowres),
        ("interpolation_weight120", input_support),
        ("target_lst", target),
        ("valid", valid),
        ("eligible", eligible),
    ):
        if array.shape != expected_field:
            raise ValueError(f"{entry.file}: {name} shape {array.shape}, expected (160, 160)")
    mask = valid & eligible
    if not np.any(mask):
        raise ValueError(f"{entry.file}: valid & eligible mask is empty")
    if not np.all(np.isfinite(input_support)) or np.any(input_support < 0.0) or np.any(input_support > 1.000001):
        raise ValueError(f"{entry.file}: interpolation_weight120 must be finite in [0,1]")
    if np.any(valid & (input_support < 0.5)):
        raise ValueError(f"{entry.file}: final valid mask includes unsupported coarse input")
    finite = np.isfinite(lowres) & np.isfinite(target) & np.all(np.isfinite(optical), axis=0)
    if not np.all(finite[mask]):
        raise ValueError(f"{entry.file}: non-finite value inside valid & eligible mask")
    return {
        "optical": optical,
        "lowres_lst": lowres,
        "input_support": input_support,
        "target_lst": target,
        "valid": valid,
        "eligible": eligible,
        "mask": mask,
    }


class _Moments:
    def __init__(self, channels: int = 1) -> None:
        self.count = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.total_sq = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[0] != self.total.size:
            raise ValueError("moment input must have shape [channels, values]")
        self.count += int(array.shape[1])
        self.total += np.sum(array, axis=1)
        self.total_sq += np.sum(array * array, axis=1)

    def finish(self, minimum_std: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise ValueError("cannot fit normalization without valid source pixels")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - mean * mean, 0.0)
        std = np.maximum(np.sqrt(variance), minimum_std)
        return mean, std


def fit_normalization(entries: Sequence[SceneEntry]) -> V2Normalization:
    """Fit channel statistics from source-role scenes and no other role."""
    if not entries:
        raise ValueError("at least one source scene is required")
    if any(entry.role != "source" for entry in entries):
        raise ValueError("normalization may only be fit on source-role scenes")
    optical_moments = _Moments(OPTICAL_CHANNELS)
    lowres_moments = _Moments()
    residual_moments = _Moments()
    n_pixels = 0
    for entry in entries:
        scene = load_scene(entry)
        mask = scene["mask"]
        optical_moments.update(scene["optical"][:, mask])
        lowres_moments.update(scene["lowres_lst"][mask])
        residual_moments.update((scene["target_lst"] - scene["lowres_lst"])[mask])
        n_pixels += int(mask.sum())
    optical_mean, optical_std = optical_moments.finish()
    lowres_mean, lowres_std = lowres_moments.finish()
    residual_mean, residual_std = residual_moments.finish()
    return V2Normalization(
        optical_mean=tuple(float(x) for x in optical_mean),
        optical_std=tuple(float(x) for x in optical_std),
        lowres_mean=float(lowres_mean[0]),
        lowres_std=float(lowres_std[0]),
        residual_mean=float(residual_mean[0]),
        residual_std=float(residual_std[0]),
        n_source_scenes=len(entries),
        n_source_pixels=n_pixels,
    )


def encode_scene(scene: Mapping[str, np.ndarray], normalization: V2Normalization) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 8-channel input, normalized residual target, and exact task mask."""
    optical_mean = np.asarray(normalization.optical_mean, dtype=np.float32)[:, None, None]
    optical_std = np.asarray(normalization.optical_std, dtype=np.float32)[:, None, None]
    optical = (scene["optical"] - optical_mean) / optical_std
    lowres = (scene["lowres_lst"] - normalization.lowres_mean) / normalization.lowres_std
    residual = scene["target_lst"] - scene["lowres_lst"]
    residual = (residual - normalization.residual_mean) / normalization.residual_std
    support = np.asarray(scene["input_support"], dtype=np.float32)
    inputs = np.concatenate((lowres[None, ...], support[None, ...], optical), axis=0)
    return (
        np.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        np.asarray(scene["mask"], dtype=bool),
    )


def decode_residual(normalized_residual: np.ndarray, normalization: V2Normalization) -> np.ndarray:
    return np.asarray(normalized_residual) * normalization.residual_std + normalization.residual_mean


def _origins(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > length:
        raise ValueError(f"patch size {patch_size} exceeds field size {length}")
    positions = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if positions[-1] != final:
        positions.append(final)
    return positions


class V2PatchDataset(Dataset[dict[str, Any]]):
    """Deterministic 64x64 source patches with hierarchical sampling support."""

    def __init__(
        self,
        entries: Sequence[SceneEntry],
        normalization: V2Normalization,
        *,
        stride: int = PATCH_SIZE,
        min_mask_fraction: float = 0.25,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        if not entries or any(entry.role != "source" for entry in entries):
            raise ValueError("V2PatchDataset accepts source-role entries only")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if not 0.0 <= min_mask_fraction <= 1.0:
            raise ValueError("min_mask_fraction must be in [0, 1]")
        self.entries = list(entries)
        self.normalization = normalization
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self._scenes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.samples: list[tuple[int, int, int]] = []
        rows = _origins(HEIGHT, PATCH_SIZE, stride)
        columns = _origins(WIDTH, PATCH_SIZE, stride)
        for scene_index, entry in enumerate(self.entries):
            encoded = encode_scene(load_scene(entry), normalization)
            self._scenes.append(encoded)
            mask = encoded[2]
            before = len(self.samples)
            for row in rows:
                for column in columns:
                    patch_mask = mask[row : row + PATCH_SIZE, column : column + PATCH_SIZE]
                    if float(patch_mask.mean()) >= min_mask_fraction:
                        self.samples.append((scene_index, row, column))
            if len(self.samples) == before:
                raise ValueError(
                    f"scene {entry.scene_id} has no 64x64 patch satisfying min_mask_fraction"
                )
        if not self.samples:
            raise ValueError("no 64x64 patches satisfy min_mask_fraction")

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _augmentation_code(self, index: int, draw: int = 0) -> int:
        scene_index, row, column = self.samples[index]
        identity = self.entries[scene_index].scene_id
        message = f"{self.seed}:{self.epoch}:{draw}:{identity}:{row}:{column}".encode("utf-8")
        return hashlib.sha256(message).digest()[0] & 7

    def sample_weights(self) -> torch.Tensor:
        """Weights giving equal city mass, then equal scene mass per city."""
        patches_per_scene = Counter(scene_index for scene_index, _, _ in self.samples)
        scenes_per_city = Counter(entry.city for entry in self.entries)
        values = []
        for scene_index, _, _ in self.samples:
            entry = self.entries[scene_index]
            values.append(1.0 / (scenes_per_city[entry.city] * patches_per_scene[scene_index]))
        return torch.tensor(values, dtype=torch.double)

    def make_sampler(
        self, num_samples: int, seed: int, *, start_draw: int = 0
    ) -> "DeterministicDrawSampler":
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if start_draw < 0:
            raise ValueError("start_draw must be nonnegative")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        weighted = WeightedRandomSampler(
            self.sample_weights(),
            num_samples=int(start_draw + num_samples),
            replacement=True,
            generator=generator,
        )
        return DeterministicDrawSampler(weighted, start_draw=start_draw, num_samples=num_samples)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            index, draw = index
        else:
            draw = 0
        scene_index, row, column = self.samples[index]
        inputs, target, mask = self._scenes[scene_index]
        spatial = np.concatenate(
            (
                inputs[:, row : row + PATCH_SIZE, column : column + PATCH_SIZE],
                target[None, row : row + PATCH_SIZE, column : column + PATCH_SIZE],
                mask[None, row : row + PATCH_SIZE, column : column + PATCH_SIZE],
            ),
            axis=0,
        )
        if self.augment:
            code = self._augmentation_code(index, draw)
            spatial = np.rot90(spatial, k=code & 3, axes=(-2, -1))
            if code & 4:
                spatial = spatial[..., ::-1]
        spatial = np.ascontiguousarray(spatial)
        entry = self.entries[scene_index]
        return {
            "input": torch.from_numpy(spatial[:INPUT_CHANNELS]).float(),
            "target": torch.from_numpy(spatial[INPUT_CHANNELS : INPUT_CHANNELS + 1]).float(),
            "mask": torch.from_numpy(spatial[INPUT_CHANNELS + 1 :]).bool(),
            "city": entry.city,
            "year": entry.year,
            "scene_id": entry.scene_id,
        }


class DeterministicDrawSampler(Sampler[tuple[int, int]]):
    """Attach a stable draw number so replacement draws can augment differently."""

    def __init__(
        self, sampler: WeightedRandomSampler, *, start_draw: int = 0, num_samples: int | None = None
    ) -> None:
        self.sampler = sampler
        self.start_draw = int(start_draw)
        self.num_samples = len(sampler) - self.start_draw if num_samples is None else int(num_samples)
        if self.start_draw < 0 or self.num_samples < 0:
            raise ValueError("invalid deterministic sampler slice")
        if self.start_draw + self.num_samples > len(sampler):
            raise ValueError("deterministic sampler slice exceeds weighted sampler")

    def __iter__(self):
        for draw, index in enumerate(self.sampler):
            if draw < self.start_draw:
                continue
            if draw >= self.start_draw + self.num_samples:
                break
            yield int(index), draw

    def __len__(self) -> int:
        return self.num_samples


def source_validation_overlap(source: Iterable[SceneEntry], validation: Iterable[SceneEntry]) -> set[Path]:
    """Return exact scene-file overlap for an explicit train/validation gate."""
    return {entry.file for entry in source}.intersection(entry.file for entry in validation)
