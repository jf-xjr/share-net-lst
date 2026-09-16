#!/usr/bin/env python3
"""Reuse existing causal POWER tokens in exactly the public Fine52 scene order.

No acquisition, target arrays, or locked scenes are opened. Every token is
replayed from the existing 48 completed hourly bins and the original Fit603
normalization is independently reproduced. This is retrospective forcing;
hourly rainfall and soil wetness are not present in these assets.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

import g246_data
import build_g246_8h_cache as base
import build_g246_r2_power_hourly_causal_sidecars as hourly
import build_g246_r2_power_hourly_sequence_sidecars as sequence
from ordered_residual_data import SequenceAdapter

SCHEMA = "g246-8h-hourly-cache-v1"
WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WORKSPACE / "artifacts/g246_8h/hourly_v1"
DEFAULT_SEQUENCE = WORKSPACE / "artifacts/g246_r2_power_hourly_sequence_sidecars_v2"
TOKEN_NAMES = tuple(sequence.TOKEN_NAMES)


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular local JSON: {path}")
    return json.loads(path.read_text())


def _member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("source member path escapes its artifact")
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular source member: {path}")
    return path


def _replay_source(root: Path, entry: Any) -> tuple[Any, dict[str, Any]]:
    candidates = (Path(sequence.DEFAULT_V1_SOURCE), root / "v1_acquisition_cache")
    # The v2 name uses the same registered scene stem as its v1 provenance.
    matches = []
    for source in candidates:
        directory = source / "sidecars" / entry.view_role
        if directory.is_dir():
            for path in directory.glob(f"{entry.city}_*.json"):
                payload = _read(path)
                if payload.get("scene_binding", {}).get("scene_id") == entry.scene_id:
                    matches.append((source, path, payload))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one existing v1 provenance: {entry.scene_id}")
    source, path, payload = matches[0]
    hourly.validate_sidecar(payload)
    audit_path = _member(source, f"private_provenance/audit/{path.name}")
    audit_hash = base.sha256(audit_path)
    if audit_hash != payload["private_provenance_digest"]["audit_sha256"]:
        raise ValueError("hourly audit hash mismatch")
    audit = _read(audit_path)
    raw_path = _member(source, audit["raw_cache"]["path"])
    if base.sha256(raw_path) != audit["raw_cache"]["sha256"]:
        raise ValueError("hourly raw source hash mismatch")
    raw = _read(raw_path)
    query = hourly.HourlySceneQuery(**{
        key: audit["query"][key] for key in hourly.HourlySceneQuery.__dataclass_fields__
    })
    expected = {"scene_id": entry.scene_id, "role": entry.view_role,
                "city": entry.city, "region": entry.region, "item_id": entry.item_id,
                "source_scene_sha256": entry.sha256}
    if any(getattr(query, key) != value for key, value in expected.items()) \
            or query.t0 != hourly.parse_utc(entry.datetime):
        raise ValueError("hourly query differs from registered base scene")
    if query.query_sha256 != audit["query_sha256"] \
            or query.query_sha256 != payload["private_provenance_digest"]["query_sha256"] \
            or raw["query"] != audit["query"]:
        raise ValueError("hourly query binding mismatch")
    response_hash = hourly.daily.canonical_sha256(raw["response"])
    if response_hash != raw["response_canonical_sha256"] \
            or response_hash != payload["private_provenance_digest"]["response_canonical_sha256"]:
        raise ValueError("raw response content mismatch")
    for document in (audit, raw):
        if document.get("target_arrays_opened") is not False \
                or document.get("validation_target_opened") is not False \
                or document.get("locked_test_opened") is not False:
            raise ValueError("hourly source has an unsafe access declaration")
    # Reapply the original causal selector to the hash-bound, cached response.
    _, quality, lag, timing = hourly.derive_causal_predictor(raw["response"], query)
    if lag != payload["sanitized_complete_lag_tensor"] \
            or quality != payload["predictor_quality"] \
            or timing != audit["interval_and_derivation"]:
        raise ValueError("hourly source replay changed")
    end = hourly.parse_utc(timing["last_selected_interval_end_utc"])
    start = hourly.parse_utc(timing["first_selected_interval_start_utc"])
    if end != query.t0.replace(minute=0, second=0, microsecond=0) \
            or start != end - timedelta(hours=48) or end > query.t0 \
            or timing["partial_or_future_interval_used"] is not False \
            or timing["expected_lookback_interval_count"] != 48 \
            or not all(value == 48 for value in timing["valid_interval_count_by_parameter"].values()):
        raise ValueError("48-hour causal interval contract failed")
    row = sequence.derive_sequence_from_v1(payload, query)
    audit_record = {
        "source_v1_sidecar_path": str(path.resolve()), "source_v1_sidecar_sha256": base.sha256(path),
        "audit_path": str(audit_path.resolve()), "audit_sha256": audit_hash,
        "raw_sha256": audit["raw_cache"]["sha256"], "query_sha256": query.query_sha256,
        "first_interval_start_utc": start.isoformat(), "last_interval_end_utc": end.isoformat(),
        "query_utc": query.t0.isoformat(), "query_gap_hours": (query.t0-end).total_seconds()/3600,
        "forcing_ready": bool(np.all(row.valid)), "time_and_source_replay_passed": True,
    }
    return row, audit_record


class HourlyCache:
    """Read-only [N,48,10] mmap; metadata never enters the returned tensor."""
    def __init__(self, root: Path = DEFAULT_OUTPUT, role: str = "fit", *,
                 expected_scene_ids: Sequence[str] | None = None, verify_hash: bool = True):
        self.root = base.guard(Path(root))
        self.manifest = _read(self.root / "manifest.json")
        if role not in {"fit", "validation"} or self.manifest.get("schema_version") != SCHEMA \
                or self.manifest.get("status") != "complete" \
                or self.manifest.get("locked_test_opened") is not False \
                or self.manifest.get("target_arrays_opened") is not False \
                or self.manifest.get("token_names") != list(TOKEN_NAMES):
            raise ValueError("invalid hourly cache contract")
        record = self.manifest["roles"][role]
        directory = self.root / role
        if base.sha256(directory / "metadata.json") != record["metadata_sha256"]:
            raise ValueError("hourly metadata changed")
        self.metadata = _read(directory / "metadata.json")
        self.scene_ids = [scene["scene_id"] for scene in self.metadata["scenes"]]
        if expected_scene_ids is not None and self.scene_ids != list(expected_scene_ids):
            raise ValueError("hourly/base scene order mismatch")
        self.array = np.load(directory / "tokens.npy", mmap_mode="r", allow_pickle=False)
        if self.array.shape != (len(self.scene_ids), 48, 10) or self.array.dtype != np.float32 \
                or len(self.scene_ids) != {"fit": 603, "validation": 45}[role] \
                or record["forcing_ready_count"] != len(self.scene_ids):
            raise ValueError("hourly tensor geometry or ready count differs")
        if verify_hash and base.sha256(directory / "tokens.npy") != record["tokens"]["sha256"]:
            raise ValueError("hourly tokens changed")

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: Any) -> np.ndarray:
        return np.array(self.array[index], dtype=np.float32, copy=True)


def build_cache(output: Path = DEFAULT_OUTPUT, *, base_root: Path = base.DEFAULT_OUTPUT,
                sequence_root: Path = DEFAULT_SEQUENCE) -> dict[str, Any]:
    output, base_root, sequence_root = map(base.guard, (output, base_root, sequence_root))
    if output.exists():
        for role in ("fit", "validation"):
            HourlyCache(output, role)
        return _read(output / "manifest.json")
    started = time.monotonic()
    base_manifest = _read(base_root / "manifest.json")
    if base_manifest.get("status") != "complete" or base_manifest.get("smoke_only") \
            or base_manifest.get("locked_test_opened") is not False:
        raise ValueError("base cache is incomplete or unsafe")
    splits = g246_data.load_splits(role="fit+validation")
    if splits.receipt_sha256 != base_manifest["split_receipt_sha256"]:
        raise ValueError("base cache belongs to another split")
    entries = {entry.scene_id: entry for entry in (*splits.fit, *splits.validation)}
    adapter = SequenceAdapter(sequence_root / "manifest.json", list(entries.values()))
    transform = _read(sequence_root / "normalization.json")
    source_rows, audit_rows, tokens_by_id = [], {}, {}
    for index, entry in enumerate(entries.values()):
        row, audit = _replay_source(sequence_root, entry)
        source_rows.append(row)
        tokens, _, ready = adapter.load(entry)
        replay, replay_ready = sequence.model_tokens(row, transform)
        if not ready or not replay_ready or not np.array_equal(tokens, replay):
            raise ValueError(f"v2 token replay failed: {entry.scene_id}")
        tokens_by_id[entry.scene_id], audit_rows[entry.scene_id] = tokens, audit
        if (index+1) % 100 == 0:
            print(f"replayed {index+1}/648 source chains in {time.monotonic()-started:.1f}s", flush=True)
    # Match the original acquisition order as well as its equal hierarchy;
    # changing floating summation order alone can move a moment by 1e-14.
    normalization_rows = sorted(source_rows, key=lambda row: (
        row.query.role, row.query.region, row.query.city,
        row.query.t0.date().isoformat(), row.query.item_id))
    replay_transform = sequence.fit_transform(normalization_rows)
    if replay_transform != transform or transform.get("fit_scene_count") != 603:
        raise ValueError("existing token normalization is not the reproduced Fit603-only transform")
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA, "status": "complete", "token_names": list(TOKEN_NAMES),
        "lookback_hours": 48, "time_order": "oldest completed interval to latest completed interval",
        "normalization": {"path": str(sequence_root / "normalization.json"),
                          "sha256": base.sha256(sequence_root / "normalization.json"),
                          "scope": "Fit603 only; no Validation values in fitted moments",
                          "reproduced_from_raw_fit_rows": True, "source_document": transform},
        "source_sequence_manifest": {"path": str(sequence_root / "manifest.json"),
                                     "sha256": base.sha256(sequence_root / "manifest.json")},
        "base_cache_manifest": {"path": str(base_root / "manifest.json"),
                                "sha256": base.sha256(base_root / "manifest.json")},
        "split_receipt_sha256": splits.receipt_sha256,
        "raw_parameters": list(sequence.RAW_ORDER), "raw_units": {key: hourly.EXPECTED_PROVIDER_UNITS[key] for key in sequence.RAW_ORDER},
        "hourly_rain_or_soil_wetness_present": False, "retrospective_batch_only": True,
        "near_real_time_claim": False, "query_causality": "all 48 interval ends <= query; half-open hourly bins",
        "target_arrays_opened": False, "locked_test_opened": False, "network_requests": 0,
        "identity_metadata_not_model_inputs": True, "roles": {},
        "code_sha256": {name: base.sha256(WORKSPACE / "code" / name) for name in (
            "build_g246_8h_hourly_cache.py", "build_g246_r2_power_hourly_sequence_sidecars.py",
            "build_g246_r2_power_hourly_causal_sidecars.py", "ordered_residual_data.py")},
    }
    for role, count in (("fit", 603), ("validation", 45)):
        original_path = base_root / role / "metadata.json"
        if base.sha256(original_path) != base_manifest["roles"][role]["metadata_sha256"]:
            raise ValueError("base metadata hash changed")
        original = _read(original_path)
        scenes = original["scenes"]
        if len(scenes) != count or len({s["scene_id"] for s in scenes}) != count:
            raise ValueError("base scene inventory differs")
        for index, scene in enumerate(scenes):
            entry = entries[scene["scene_id"]]
            if scene["index"] != index or entry.view_role != role or scene["source_sha256"] != entry.sha256:
                raise ValueError("base scene binding mismatch")
        directory = staging / role
        directory.mkdir()
        np.save(directory / "tokens.npy", np.stack([tokens_by_id[s["scene_id"]] for s in scenes]), allow_pickle=False)
        metadata = {"schema_version": SCHEMA, "role": role, "scenes": scenes,
                    "token_names": list(TOKEN_NAMES), "source_audits": [audit_rows[s["scene_id"]] for s in scenes],
                    "base_metadata_sha256": base.sha256(original_path),
                    "identity_metadata_not_model_inputs": True, "locked_test_opened": False,
                    "target_arrays_opened": False}
        base.write_json(directory / "metadata.json", metadata)
        manifest["roles"][role] = {"scene_count": count, "forcing_ready_count": count,
            "metadata_sha256": base.sha256(directory / "metadata.json"),
            "tokens": {"shape": [count, 48, 10], "dtype": "float32", "bytes": (directory / "tokens.npy").stat().st_size,
                       "sha256": base.sha256(directory / "tokens.npy")}}
    manifest["elapsed_seconds"] = time.monotonic() - started
    base.write_json(staging / "manifest.json", manifest)
    os.replace(staging, output)
    for role in ("fit", "validation"):
        HourlyCache(output, role)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-cache", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE)
    args = parser.parse_args()
    result = build_cache(args.output, base_root=args.base_cache, sequence_root=args.sequence_root)
    print(json.dumps({"status": result["status"], "roles": result["roles"], "output": str(args.output)}, indent=2))
