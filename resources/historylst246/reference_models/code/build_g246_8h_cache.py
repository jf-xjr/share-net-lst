#!/usr/bin/env python3
"""Materialize registered, unaugmented G246 public Fine52 single-date arrays.

Only Fit603 and public Validation45 are accepted. Predictors and supervision
are separate mmap files; metadata identities never occur in predictor arrays.
The public cache reuses the existing Fit603 normalization and is not a claim
of train-fold-only normalization. Kelvin-valued arrays retain float32.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

import build_g246_half_q_cache as source
import g246_data
import g246_r2_data as r2
import g246_r2_multisource as multi

SCHEMA = "g246-8h-public-fine52-fullscene-cache-v1"
WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WORKSPACE / "artifacts/g246_8h/cache_v1"
CONTEXT_KEEP = tuple(i for i in range(19) if i not in (5, 6, 7, 8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def guard(path: Path) -> Path:
    checked = g246_data.reject_forbidden_path(path).resolve()
    if checked.is_symlink():
        raise ValueError("cache path must not be a symlink")
    return checked


def verify_cache(root: Path, *, full_hash: bool = True) -> dict[str, Any]:
    root = guard(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("not a complete G246 8h cache")
    if manifest.get("locked_test_opened") is not False:
        raise ValueError("invalid locked-test contract")
    for role, detail in manifest["roles"].items():
        if role not in ("fit", "validation"):
            raise ValueError("unexpected cache role")
        directory = root / role
        if sha256(directory / "metadata.json") != detail["metadata_sha256"]:
            raise ValueError(f"metadata changed: {role}")
        for name, expected in detail["arrays"].items():
            path = directory / f"{name}.npy"
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
                raise ValueError(f"array contract changed: {role}/{name}")
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"array size changed: {role}/{name}")
            if full_hash and sha256(path) != expected["sha256"]:
                raise ValueError(f"array hash changed: {role}/{name}")
    return manifest


def build_cache(output: Path, *, smoke_scenes: int = 0) -> dict[str, Any]:
    output = guard(output)
    if output.exists():
        return verify_cache(output)
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"unfinished cache exists: {staging}")
    splits = g246_data.load_splits(role="fit+validation")
    if splits.receipt_sha256 != source.REGISTERED_RECEIPT_SHA256:
        raise ValueError("public campaign receipt changed")
    if len(splits.fit) != 603 or len(splits.validation) != 45:
        raise ValueError("public campaign must be Fit603 + Validation45")
    normalization = source._load_registered_normalization(
        source.DEFAULT_NORMALIZATION,
        expected_sha256=source.REGISTERED_NORMALIZATION_SHA256,
        expected_fit_view_sha256=source.REGISTERED_FIT_VIEW_SHA256,
        expected_scene_ids=[entry.scene_id for entry in splits.fit],
    )
    staging.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "building",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_test_opened": False,
        "normalization_scope": "registered_Fit603_only_public_development",
        "normalization_sha256": source.REGISTERED_NORMALIZATION_SHA256,
        "source_binding_sha256": source.REGISTERED_FINE52_BINDING_SHA256,
        "split_receipt_sha256": splits.receipt_sha256,
        "fit_view_sha256": splits.fit_view_sha256,
        "validation_view_sha256": splits.validation_view_sha256,
        "smoke_only": bool(smoke_scenes),
        "augment": False,
        "query_only": True,
        "roles": {},
        "code_sha256": {
            name: sha256(WORKSPACE / "code" / name)
            for name in ("build_g246_8h_cache.py", "build_g246_half_q_cache.py",
                         "g246_data.py", "g246_r2_data.py", "g246_r2_multisource.py")
        },
    }
    for role, entries in (("fit", splits.fit), ("validation", splits.validation)):
        predictor = source.Fine52PredictorOnlySource(
            entries, normalization,
            texture_manifest=source.DEFAULT_TEXTURE_MANIFEST,
            weather_manifest=source.DEFAULT_WEATHER_MANIFEST,
            view_role=role,
        )
        ordered = predictor.entries[:smoke_scenes] if smoke_scenes else predictor.entries
        count = len(ordered)
        directory = staging / role
        directory.mkdir()
        contracts = {
            "fine": ((count, 52, 160, 160), np.float32),
            "coarse": ((count, 1, 40, 40), np.float32),
            "support": ((count, 1, 160, 160), np.bool_),
            "context": ((count, 15), np.float32),
            "target": ((count, 1, 160, 160), np.float32),
            "formal": ((count, 1, 160, 160), np.bool_),
            "valid": ((count, 1, 160, 160), np.bool_),
        }
        arrays = {
            name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+",
                                            dtype=dtype, shape=shape)
            for name, (shape, dtype) in contracts.items()
        }
        scenes = []
        for row, (entry, sample) in enumerate(zip(ordered, itertools.islice(predictor, count))):
            if sample["scene_id"] != entry.scene_id or sample["target_arrays_opened"]:
                raise ValueError("predictor order or access contract changed")
            fine = np.asarray(sample["fine"])[0, 0]
            coarse = np.asarray(sample["coarse_k"])[0, 0]
            support = np.asarray(sample["support"])[0, 0].astype(bool)
            context = np.asarray(sample["context"])[0, 0, list(CONTEXT_KEEP)]
            # The fallback is used exclusively outside valid supervision. It
            # cannot change the training/evaluation target on scored pixels.
            labels = r2._query_supervision(entry, float(np.median(fine[0])))
            formal = labels["valid"] & labels["eligible"]
            values = {"fine": fine, "coarse": coarse, "support": support,
                      "context": context, "target": labels["target_k"],
                      "formal": formal, "valid": labels["valid"]}
            for name, value in values.items():
                if value.shape != arrays[name].shape[1:]:
                    raise ValueError(f"shape mismatch {role}/{row}/{name}: {value.shape}")
                if name != "coarse" and np.issubdtype(value.dtype, np.floating) \
                        and not np.isfinite(value).all():
                    raise ValueError(f"non-finite {role}/{row}/{name}")
                arrays[name][row] = value
            if np.isinf(coarse).any() or not formal.any():
                raise ValueError(f"invalid observed coarse or formal mask: {entry.scene_id}")
            scenes.append({
                "index": row, "scene_id": entry.scene_id, "city": entry.city,
                "region": entry.region, "year": entry.year, "datetime": entry.datetime,
                "item_id": entry.item_id, "view_role": role,
                "source_sha256": entry.sha256,
                "formal_pixel_count": int(formal.sum()),
                "valid_pixel_count": int(labels["valid"].sum()),
                "support_pixel_count": int(support.sum()),
            })
            if row == 0 or (row + 1) % 25 == 0 or row + 1 == count:
                print(f"{role} {row + 1}/{count}; elapsed={time.monotonic()-started:.1f}s", flush=True)
        for value in arrays.values():
            value.flush()
        del arrays, value
        metadata = {
            "schema_version": SCHEMA, "role": role, "scenes": scenes,
            "fine_channel_names": list(multi.MULTISOURCE_FINE_CHANNEL_NAMES),
            "context_channel_names": [multi.MULTISOURCE_CONTEXT_NAMES[i] for i in CONTEXT_KEEP],
            "removed_stored_context_indices": [5, 6, 7, 8],
            "model_inputs": ["fine", "coarse", "support", "context"],
            "label_arrays": ["target", "formal", "valid"],
            "formal_definition": "valid AND eligible; valid also requires finite target",
            "coarse_missing_value": "NaN; downstream preserve O/U semantics",
            "invalid_target_fill": "median of predictor-only interpolated base; masked from loss",
            "identity_metadata_not_model_inputs": True,
            "locked_test_opened": False,
        }
        write_json(directory / "metadata.json", metadata)
        manifest["roles"][role] = {
            "scene_count": count, "city_count": len({e.city for e in ordered}),
            "metadata_sha256": sha256(directory / "metadata.json"),
            "predictor_provenance": dict(predictor.provenance_record()),
            "arrays": {
                name: {"shape": list(shape), "dtype": np.dtype(dtype).name,
                       "bytes": (directory / f"{name}.npy").stat().st_size,
                       "sha256": sha256(directory / f"{name}.npy")}
                for name, (shape, dtype) in contracts.items()
            },
        }
        del predictor
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["array_bytes"] = sum(v["bytes"] for role in manifest["roles"].values()
                                  for v in role["arrays"].values())
    write_json(staging / "manifest.json", manifest)
    os.rename(staging, output)
    verify_cache(output, full_hash=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-scenes", type=int, default=0,
                        help="debug only: first N scenes per role; not a complete training cache")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.smoke_scenes < 0:
        parser.error("--smoke-scenes must be nonnegative")
    result = verify_cache(args.output) if args.verify else build_cache(
        args.output, smoke_scenes=args.smoke_scenes)
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                      "array_bytes": result["array_bytes"],
                      "roles": {role: value["scene_count"] for role, value in result["roles"].items()}},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
