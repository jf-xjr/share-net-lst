#!/usr/bin/env python3
"""Audit and encode existing optical-only 30 m cache on canonical 120 m cells.

No network or thermal/target array is read. Optical QA and exact canonical
registration are checked against the original sidecar builder. Missing raw
scenes remain in the common cache order with a zero optional input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any

from affine import Affine
import numpy as np

import build_g246_8h_cache as main_cache
import build_g246_half_q_cache as source
import build_g246_r2_texture_sidecars as texture
import g246_data
import g246_r2_optical_cache as optical_cache

SCHEMA = "g246-8h-existing-optical-detail-v1"
ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/g246_r2_optical_cache_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/g246_8h/optical_detail_v1"
AUDIT_OUTPUT = ROOT / "artifacts/g246_8h/optical_detail_audit.json"
BANDS = tuple(texture.BAND_NAMES)
PAIRS = tuple(itertools.combinations(range(6), 2))
CHANNEL_NAMES = tuple(
    [f"{band}_phase_r{r}_c{c}_centred_fit_std" for band in BANDS
     for r in range(4) for c in range(4)]
    + [f"cov_{BANDS[i]}_{BANDS[j]}_fit_std" for i, j in PAIRS]
    + [f"optical_valid_phase_r{r}_c{c}" for r in range(4) for c in range(4)]
    + ["raw_optical_available"]
)


def _manifest() -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    records = [json.loads(line) for line in (RAW_ROOT / "manifest.jsonl").read_text().splitlines()]
    header, rows = records[0], records[1:]
    if header.get("schema") != optical_cache.CACHE_SCHEMA \
            or header.get("locked_test_opened") is not False \
            or header.get("target_arrays_opened") is not False:
        raise ValueError("raw optical cache has unsafe provenance")
    mapped = {}
    for row in rows:
        key = row["scene_id"], row["band"]
        if key in mapped or row["view_role"] not in ("fit", "validation"):
            raise ValueError("duplicate or non-public raw optical row")
        mapped[key] = row
    return header, mapped


def audit_existing(cache_root: Path) -> dict[str, Any]:
    baseline = main_cache.verify_cache(cache_root, full_hash=False)
    if baseline.get("smoke_only"):
        raise ValueError("optical detail requires the complete public baseline cache")
    header, records = _manifest()
    known_ids = set()
    roles = {}
    for role in ("fit", "validation"):
        metadata = json.loads((cache_root / role / "metadata.json").read_text())
        scenes = metadata["scenes"]
        available = [row for row in scenes if all((row["scene_id"], band) in records for band in BANDS)]
        known_ids.update(row["scene_id"] for row in scenes)
        coverage = Counter(row["region"] for row in available)
        total = Counter(row["region"] for row in scenes)
        roles[role] = {
            "total_scenes": len(scenes), "available_scenes": len(available),
            "scene_coverage_fraction": len(available) / len(scenes),
            "by_region": {region: {"available": coverage[region], "total": count}
                          for region, count in total.items()},
            "available_scene_ids": [row["scene_id"] for row in available],
            "missing_scene_ids": [row["scene_id"] for row in scenes if row not in available],
        }
    if any(scene_id not in known_ids for scene_id, _ in records):
        raise ValueError("raw manifest contains scene outside public baseline")
    result = {
        "schema_version": SCHEMA, "phase": "metadata-only-coverage-budget-audit",
        "raw_manifest_sha256": main_cache.sha256(RAW_ROOT / "manifest.jsonl"),
        "baseline_manifest_sha256": main_cache.sha256(cache_root / "manifest.json"),
        "raw_cache_status": header["status"], "roles": roles,
        "input_channel_count": len(CHANNEL_NAMES),
        "estimated_dense_array_bytes": 648 * 128 * 160 * 160 * 2,
        "estimated_incremental_working_memory_bytes": 80 * 1024**2,
        "compute": "CPU; one 640x640 six-band scene at a time; no GPU",
        "qa_rule": "original optical_valid_mask: bits 0-5 and 9 only; bits 6-8 ignored",
        "alignment_rule": "raw canonical-grid SHA + source transform30 * scale(4) == transform120",
        "baseline_sanity": "raw-derived float16 optical_mean6 and coverage must exactly match v2 sidecar",
        "coverage_warning": "Fit and Validation raw availability differ; keep all scenes and train modality dropout",
        "locked_test_opened": False, "target_arrays_opened": False,
        "raw_cache_modified": False, "network_used": False,
    }
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # A rerun replaces only this builder's own report, never source evidence.
    main_cache.write_json(AUDIT_OUTPUT, result)
    return result


def encode_detail(raw: np.ndarray, valid: np.ndarray, band_stds: np.ndarray) -> np.ndarray:
    """Channels: 96 canonical phases, 15 covariances, 16 QA masks, availability."""
    optical = raw.astype(np.float32) * texture.SR_SCALE + texture.SR_OFFSET
    phases = optical.reshape(6, 160, 4, 160, 4).transpose(0, 2, 4, 1, 3).reshape(6, 16, 160, 160)
    mask = valid.reshape(160, 4, 160, 4).transpose(1, 3, 0, 2).reshape(16, 160, 160)
    count = mask.sum(axis=0)
    mean = np.divide(np.where(mask[None], phases, 0).sum(axis=1, dtype=np.float64),
                     count[None], out=np.zeros((6, 160, 160), np.float64),
                     where=count[None] > 0).astype(np.float32)
    centred = np.where(mask[None], (phases - mean[:, None]) / band_stds[:, None, None, None], 0)
    covariance = np.stack([
        np.divide((centred[i] * centred[j]).sum(axis=0, dtype=np.float64), count,
                  out=np.zeros((160, 160), np.float64), where=count > 0)
        for i, j in PAIRS
    ]).astype(np.float32)
    packed = np.concatenate((centred.reshape(96, 160, 160), covariance,
                             mask.astype(np.float32), np.ones((1, 160, 160), np.float32)))
    if packed.shape != (128, 160, 160) or not np.isfinite(packed).all() \
            or np.max(np.abs(packed)) > np.finfo(np.float16).max:
        raise ValueError("optical detail cannot be represented as finite float16")
    return packed.astype(np.float16)


def _verify_one(entry: Any, records: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = texture.load_scene_spec(entry)  # only metadata + qa_reason30 members
    transform30 = Affine(*spec.metadata["transform30"])
    transform120 = Affine(*spec.metadata["transform120"])
    if not (transform30 * Affine.scale(4)).almost_equals(transform120, precision=1e-8):
        raise ValueError(f"30m/120m phase registration is inconsistent: {entry.scene_id}")
    raw_arrays = []
    for band in BANDS:
        row = records[(entry.scene_id, band)]
        path = RAW_ROOT / row["file"]
        if Path(row["file"]).name != row["file"]:
            raise ValueError("raw member path escapes cache")
        expected = optical_cache.record_from_existing(path, spec, band)
        if expected != row:
            raise ValueError(f"raw grid/content provenance mismatch: {entry.scene_id}/{band}")
        raw_arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
    raw = np.stack(raw_arrays)
    valid = texture.optical_valid_mask(spec.qa_reason30, raw)
    optical = raw.astype(np.float32) * texture.SR_SCALE + texture.SR_OFFSET
    means = texture.aggregate_optical_mean(optical, valid)
    coverage = valid.reshape(160, 4, 160, 4).mean((1, 3), dtype=np.float32)
    sidecar = source.DEFAULT_TEXTURE_MANIFEST.parent / texture.sidecar_name(entry)
    with np.load(sidecar, allow_pickle=False) as data:
        old_mean = data["optical_mean6"]
        old_coverage = data["texture31"][-1]
    if not np.array_equal(means, old_mean) or not np.array_equal(coverage, old_coverage):
        raise ValueError(f"raw/QA differs from baseline Fine52 optical: {entry.scene_id}")
    return raw, valid, {
        "optical_valid_fraction30": float(valid.mean()),
        "optical_nonempty_fraction120": float((coverage > 0).mean()),
        "fine52_optical_mean_exact_match": True,
        "fine52_optical_coverage_exact_match": True,
        "canonical_grid_sha256": optical_cache.canonical_sha256(optical_cache._canonical_grid(spec)),
    }


def build_detail(cache_root: Path, output: Path) -> dict[str, Any]:
    audit = audit_existing(cache_root)
    output = main_cache.guard(output)
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("status") == "complete" and manifest.get("schema_version") == SCHEMA:
            return manifest
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    header, records = _manifest()
    splits = g246_data.load_splits(role="fit+validation")
    entries = {entry.scene_id: entry for entry in (*splits.fit, *splits.validation)}
    norm = source._load_registered_normalization(
        source.DEFAULT_NORMALIZATION, expected_sha256=source.REGISTERED_NORMALIZATION_SHA256,
        expected_fit_view_sha256=source.REGISTERED_FIT_VIEW_SHA256,
        expected_scene_ids=[entry.scene_id for entry in splits.fit])
    stds = np.asarray(norm.optical_std, np.float32)
    manifest: dict[str, Any] = {
        **audit, "phase": "validated-incremental-cache", "status": "building", "roles": {},
        "channel_names": list(CHANNEL_NAMES), "channel_slices": {
            "phase_residual": [0, 96], "cross_band_covariance": [96, 111],
            "phase_optical_valid": [111, 127], "raw_available": [127, 128]},
        "normalization": "within-120m support-centred reflectance / registered Fit optical std; covariance population mean",
        "normalization_sha256": source.REGISTERED_NORMALIZATION_SHA256,
        "recommended_training": "drop all 128 channels together for unavailable/simulated missing modality; no complete-case filtering",
        "code_sha256": main_cache.sha256(Path(__file__)),
    }
    started = time.monotonic()
    for role in ("fit", "validation"):
        directory = staging / role
        directory.mkdir()
        metadata = json.loads((cache_root / role / "metadata.json").read_text())
        scenes = metadata["scenes"]
        detail = np.lib.format.open_memmap(directory / "detail.npy", mode="w+", dtype=np.float16,
                                           shape=(len(scenes), 128, 160, 160))
        # New files are zero-filled; explicitly set each missing row as well.
        output_scenes = []
        for index, row in enumerate(scenes):
            present = all((row["scene_id"], band) in records for band in BANDS)
            record = {**row, "raw_available": present}
            if present:
                raw, valid, sanity = _verify_one(entries[row["scene_id"]], records)
                detail[index] = encode_detail(raw, valid, stds)
                record.update(sanity)
            else:
                detail[index] = 0
            output_scenes.append(record)
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(scenes):
                print(f"detail {role} {index+1}/{len(scenes)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        detail.flush()
        del detail
        output_metadata = {
            "schema_version": SCHEMA, "role": role, "scenes": output_scenes,
            "channel_names": list(CHANNEL_NAMES), "locked_test_opened": False,
            "target_arrays_opened": False,
        }
        main_cache.write_json(directory / "metadata.json", output_metadata)
        fractions = [row["optical_valid_fraction30"] for row in output_scenes if row["raw_available"]]
        manifest["roles"][role] = {
            **audit["roles"][role],
            "shape": [len(scenes), 128, 160, 160], "dtype": "float16",
            "array_sha256": main_cache.sha256(directory / "detail.npy"),
            "array_bytes": (directory / "detail.npy").stat().st_size,
            "metadata_sha256": main_cache.sha256(directory / "metadata.json"),
            "valid_fraction30_mean_covered": float(np.mean(fractions)),
            "valid_fraction30_min_covered": float(np.min(fractions)),
            "valid_fraction30_mean_all": float(sum(fractions) / len(scenes)),
            "baseline_sanity_passed_scenes": len(fractions),
        }
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.monotonic() - started
    main_cache.write_json(staging / "manifest.json", manifest)
    os.rename(staging, output)
    return manifest


class OpticalDetailCache:
    """Small mmap loader. Callers may zero the whole returned tensor for dropout."""
    channels = 128

    def __init__(self, root: str | Path = DEFAULT_OUTPUT, role: str = "fit", *,
                 expected_scene_ids: list[str] | None = None, verify_hash: bool = False):
        if role not in ("fit", "validation"):
            raise ValueError("only public fit/validation detail is supported")
        self.root = main_cache.guard(Path(root))
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("status") != "complete" or self.manifest.get("schema_version") != SCHEMA:
            raise ValueError("detail cache not complete")
        meta_path = self.root / role / "metadata.json"
        expected = self.manifest["roles"][role]
        if main_cache.sha256(meta_path) != expected["metadata_sha256"]:
            raise ValueError("detail metadata hash mismatch")
        self.metadata = json.loads(meta_path.read_text())
        self.scene_ids = [row["scene_id"] for row in self.metadata["scenes"]]
        if expected_scene_ids is not None and self.scene_ids != list(expected_scene_ids):
            raise ValueError("detail scene order differs from base cache")
        self.available = np.asarray([row["raw_available"] for row in self.metadata["scenes"]], dtype=bool)
        path = self.root / role / "detail.npy"
        if verify_hash and main_cache.sha256(path) != expected["array_sha256"]:
            raise ValueError("detail array hash mismatch")
        self.array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(self.array.shape) != expected["shape"] or self.array.dtype != np.float16:
            raise ValueError("detail array shape/dtype mismatch")

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> np.ndarray:
        return np.asarray(self.array[index])


def transform_detail_d4(value: Any, code: int) -> Any:
    """Apply the registered D4 to both 120m cells and their true 30m phases.

    Accepts numpy arrays or torch tensors with final dimensions [128,H,W].
    Code 0..3 is rot90; bit 4 adds a horizontal flip after rotation, matching
    g246_r2_data._spatial_transform. Standard spatial-only augmentation is
    incorrect for this cache because its channels have subpixel positions.
    """
    if code not in range(8) or value.shape[-3] != 128:
        raise ValueError("optical detail D4 needs code 0..7 and 128 channels")
    phase = np.rot90(np.arange(16).reshape(4, 4), k=code & 3)
    if code & 4:
        phase = phase[:, ::-1]
    index = np.arange(128)
    index[:96] = (np.arange(6)[:, None] * 16 + phase.reshape(1, 16)).reshape(-1)
    index[111:127] = 111 + phase.reshape(-1)
    if isinstance(value, np.ndarray):
        transformed = np.rot90(value, k=code & 3, axes=(-2, -1))
        if code & 4:
            transformed = transformed[..., ::-1]
        return np.ascontiguousarray(np.take(transformed, index, axis=-3))
    import torch
    if not isinstance(value, torch.Tensor):
        raise TypeError("optical detail must be a numpy array or torch tensor")
    transformed = torch.rot90(value, k=code & 3, dims=(-2, -1))
    if code & 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed.index_select(-3, torch.as_tensor(index, device=value.device)).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=main_cache.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build", action="store_true", help="otherwise write coverage/budget audit only")
    args = parser.parse_args()
    result = build_detail(args.cache_root, args.output) if args.build else audit_existing(args.cache_root)
    print(json.dumps({"phase": result["phase"], "status": result.get("status", "audit_complete"),
                      "output": str(args.output),
                      "roles": {r: {k:v[k] for k in ("available_scenes", "total_scenes")}
                                for r, v in result["roles"].items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
