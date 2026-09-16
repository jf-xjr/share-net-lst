"""Read-only binding of historical predictors to their frozen public sources.

This module opens manifests and encoded fields only. It does not import source
acquisition code, open remote resources, or read query labels or scoring masks.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "g246-8h-historical-cache-v1"
FAMILIES = ("historical_r6a", "historical_innovation_r6a", "historical_innovation_emissivity_r6a",
            "historical_attended_innovation_emissivity_r6a",
            "historical_multiscale_innovation_emissivity_r6a")
SIX_FAMILY = "historical_innovation_emissivity_r6a_six"
SIX_FAMILIES = (SIX_FAMILY, "historical_multiscale_innovation_emissivity_r6a_six")
CHANNELS = [
    "filled_historical_meanT_minus300_div20", "filled_historical_T_parent_anomaly_div5",
    "historical_clear_coverage", "filled_historical_mean_ST_QA_div3",
    "historical_joint_meanE_minus098_div001", "historical_joint_emissivity_coverage",
    "historical_doy_sin", "historical_doy_cos", "positive_age_days_div3652p5",
]
READY = {"acquisition_complete", "missing_no_eligible_candidate", "confirmed_non_tiff_missing"}


def _path(value, relative_root=ROOT):
    path = Path(value)
    path = (path if path.is_absolute() else relative_root / path).resolve()
    if "locked" in str(path).lower():
        raise ValueError("historical contracts accept public paths only")
    return path


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def _json(path, expected_sha):
    if _sha(path) != expected_sha:
        raise ValueError(f"historical bound JSON hash differs: {path.name}")
    return json.loads(path.read_text())


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("historical dates require explicit timezones")
    return result


def _identity(item_id):
    match = re.match(r"^(L[A-Z]\d{2})_L\d[A-Z0-9]{2}_(\d{6})_(\d{8})(?:_|$)", item_id)
    if not match:
        raise ValueError("invalid Landsat item identifier")
    return match.groups()


def _guard(condition, message):
    if not condition:
        raise ValueError(message)


def check_cache(root, expected_sha256, base_manifest_sha256, expected_scene_ids,
                base_metadata_sha256, expected_records):
    root = _path(root)
    manifest = _json(root / "manifest.json", expected_sha256)
    _guard(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete",
           "historical feature cache is incomplete or has a different schema")
    _guard(all(manifest.get(k) is False for k in (
        "target_arrays_opened", "target_masks_opened", "current_query_qa_arrays_opened", "locked_test_opened")),
        "historical cache lacks its predictor-only declaration")
    _guard(manifest.get("base_manifest_sha256") == base_manifest_sha256
           and set(manifest.get("roles", {})) == {"fit", "validation"},
           "historical base cache or public roles differ")
    contract = manifest.get("contract", {})
    _guard(contract.get("schema") == "g246-8h-historical-feature-encoding-v1"
           and contract.get("channels") == CHANNELS and contract.get("source_years") == [2018, 2019, 2020]
           and contract.get("query_target_or_mask_arguments") is False,
           "historical nine-channel encoding contract differs")
    audit = _json(_path(manifest["pilot_admissibility_receipt"]),
                  manifest["pilot_admissibility_receipt_sha256"])
    _guard(audit.get("pass") is True and audit.get("all_36_bitwise_reencoded") is True
           and audit.get("locked_test_opened") is False, "historical pilot input audit did not pass")
    # Preserve the audited pure encoder, while allowing a separately audited new network.
    hashes = manifest.get("source_code_sha256", {})
    _guard(hashes.get("g246_8h_historical_features.py") ==
           audit.get("source_code_sha256", {}).get("g246_8h_historical_features.py"),
           "historical pure encoder changed after its input audit")
    for name, digest in hashes.items():
        _guard(Path(name).name == name, "historical source snapshot must use simple filenames")
        _guard(_sha(root / "source" / name) == digest, "historical source snapshot changed")
    raw = _json(_path(manifest["raw_manifest_path"]), manifest["raw_manifest_sha256"])
    _guard(raw.get("ready_for_feature_build") is True and raw.get("status") == "source_plan_complete",
           "historical raw acquisition still has unresolved slots")
    _guard(all(raw.get(k) is False for k in (
        "current_query_target_arrays_opened", "current_query_qa_arrays_opened",
        "current_query_target_masks_opened", "locked_test_opened", "new_supervision_created")),
        "historical raw source violates the query-information boundary")
    raw_rows = {(r["city"], r["historical_year"]): r for r in raw["records"]}
    _guard(len(raw_rows) == len(raw["records"]) == 648 and all(r["status"] in READY for r in raw_rows.values()),
           "historical source must retain exactly 648 resolved city-year slots")
    inventory = _json(_path(raw["inventory_path"]), raw["inventory_sha256"])
    alignment = _json(_path(inventory["temporal_alignment_path"]), inventory["temporal_alignment_sha256"])
    _guard(alignment.get("locked_test_opened") is False and alignment.get("target_arrays_read") is False,
           "historical canonical-grid binding used target arrays")
    city_grid = {r["city"]: r for r in inventory["cities"]}
    public = {r["scene_id"]: r for r in inventory["public_query_scenes"]}
    selected = {(r["city"], r["historical_year"]): r for r in inventory["records"]}
    _guard(len(city_grid) == 216 and len(public) == 648 and len(selected) == 648,
           "historical inventory no longer covers the full public query population")
    aliases = {_identity(r["item_id"]) for r in public.values()}
    _guard(all(2021 <= _time(r["datetime"]).year <= 2025 for r in public.values()),
           "historical inventory query years differ from the public contract")
    for key, raw_row in raw_rows.items():
        city, year = key
        grid = city_grid[city]
        item = selected[key]["selected_item"]
        _guard(raw_row["query_scene_id"] == grid["query_scene_id"]
               and raw_row["query_view_role"] == grid["role"]
               and raw_row["query_scene_sha256"] == grid["query_scene_sha256"]
               and raw_row["canonical_crs"] == grid["canonical_crs"]
               and raw_row["canonical_transform30"] == grid["canonical_transform30"]
               and raw_row["canonical_shape"] == [640, 640]
               and raw_row["resampling"] == "nearest",
               "historical raw record differs from its frozen city/query grid")
        _guard(all(raw_row.get(k) is False for k in (
            "current_query_target_arrays_opened", "current_query_qa_arrays_opened",
            "current_query_target_masks_opened", "locked_test_opened")),
            "historical raw record used query target or quality fields")
        if item is None:
            _guard(raw_row["status"] == "missing_no_eligible_candidate"
                   and raw_row["item_id"] is None and raw_row["raw_file"] is None,
                   "historical no-candidate status differs from the frozen inventory")
            continue
        item_digest = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode()).hexdigest()
        acquired = _time(item["properties"]["datetime"])
        _guard(raw_row["item_id"] == item["id"] and raw_row["item_metadata_sha256"] == item_digest
               and raw_row["acquired_utc"] == item["properties"]["datetime"]
               and acquired.year == year and year in (2018, 2019, 2020)
               and 5 <= acquired.month <= 9 and _identity(item["id"]) not in aliases,
               "historical source identity or pre2021 acquisition boundary differs")
        if raw_row["status"] == "confirmed_non_tiff_missing":
            _guard(raw_row.get("failures") and all(
                f.get("source_confirmation", {}).get("confirmed") is True for f in raw_row["failures"]),
                "historical non-TIFF source lacks a confirmation receipt")
        else:
            _guard(raw_row["status"] == "acquisition_complete" and not raw_row.get("failures"),
                   "historical selected source is unresolved")

    detail = manifest["roles"]["validation"]
    _guard(detail.get("scene_count") == 45 and detail.get("base_metadata_sha256") == base_metadata_sha256,
           "historical validation scene count or base metadata binding differs")
    metadata = _json(root / "validation/metadata.json", detail["metadata_sha256"])
    records = metadata["scenes"]
    _guard(metadata.get("role") == "validation" and metadata.get("schema") == SCHEMA
           and [r["scene_id"] for r in records] == list(expected_scene_ids)
           and [r["scene_id"] for r in expected_records] == list(expected_scene_ids),
           "historical validation metadata order or role differs")
    path = root / "validation/features.npy"
    _guard(_sha(path) == detail["features_sha256"], "historical validation feature hash differs")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    _guard(array.shape == (45, 3, 9, 160, 160) and array.dtype == np.float32
           and list(array.shape) == detail["features_shape"] and str(array.dtype) == detail["features_dtype"]
           and path.stat().st_size == detail["features_bytes"] and np.isfinite(array).all(),
           "historical validation array shape/type/bytes or finite check failed")
    missing_source_slots = empty_thermal_slots = 0
    for i, (row, base) in enumerate(zip(records, expected_records)):
        sid, city = row["scene_id"], base["city"]
        query, grid = public[sid], city_grid[city]
        _guard(query["role"] == "validation" and query["datetime"] == base["datetime"]
               and query["source_sha256"] == base["source_sha256"] and sid in grid["all_query_scene_ids"]
               and row["query_datetime"] == base["datetime"]
               and row["query_source_sha256"] == base["source_sha256"]
               and row["canonical_crs"] == grid["canonical_crs"]
               and row["canonical_transform30"] == grid["canonical_transform30"],
               "historical query datetime, source or canonical grid binding differs")
        bound_grid_rows = alignment["roles"]["validation"]["cities"][city]
        bound = next((r for r in bound_grid_rows if r["scene_id"] == sid), None)
        _guard(bound is not None and bound["declared_source_sha256"] == base["source_sha256"]
               and bound["grid_signature_sha256"] == grid["grid_signature_sha256"],
               "historical same-city grid proof does not cover this query")
        _guard(len(row["sources"]) == 3, "historical query lost a date slot")
        for slot, (source, year) in enumerate(zip(row["sources"], (2018, 2019, 2020))):
            value = array[i, slot]
            r = raw_rows[(city, year)]
            selected_item = selected[(city, year)]["selected_item"]
            _guard(source.get("year", source.get("historical_year")) == year
                   and r["query_scene_id"] == grid["query_scene_id"]
                   and r["canonical_crs"] == grid["canonical_crs"]
                   and r["canonical_transform30"] == grid["canonical_transform30"],
                   "historical date-slot order or representative query grid differs")
            _guard(source["item_id"] == r["item_id"] and source["raw_sha256"] == r["raw_sha256"],
                   "historical feature source does not match its raw receipt")
            if r["status"] != "acquisition_complete":
                _guard(source["source_available"] is False and source["all_thermal_missing"] is True
                       and not np.count_nonzero(value), "missing historical source must yield nine exact zero channels")
                if r["status"] == "missing_no_eligible_candidate":
                    _guard(selected_item is None and r["raw_file"] is None and r["item_id"] is None,
                           "historical no-candidate fallback has a selected source")
                else:
                    _guard(selected_item is not None and r["failures"]
                           and all(f.get("source_confirmation", {}).get("confirmed") is True for f in r["failures"]),
                           "historical non-TIFF missing source lacks repeated confirmation")
                missing_source_slots += 1
                continue
            _guard(source["source_available"] is True and selected_item is not None
                   and selected_item["id"] == r["item_id"]
                   and source["historical_datetime"] == r["acquired_utc"]
                   and selected_item["properties"]["datetime"] == r["acquired_utc"],
                   "historical source timestamp or selected identity differs")
            acquired, query_time = _time(r["acquired_utc"]), _time(base["datetime"])
            _guard(acquired.year == year and 5 <= acquired.month <= 9 and acquired < query_time
                   and _identity(r["item_id"]) not in aliases,
                   "historical observation is not a distinct pre2021 acquisition")
            if source["all_thermal_missing"] is True:
                _guard(not np.count_nonzero(value), "all-cloud historical date must yield nine zero channels")
                empty_thermal_slots += 1
                continue
            _guard(source["all_thermal_missing"] is False and np.any(value[2] > 0),
                   "historical nonempty date lacks actual clear coverage")
            for coverage in (value[2], value[5]):
                _guard(np.all((0 <= coverage) & (coverage <= 1))
                       and np.all(coverage * 16 == np.rint(coverage * 16)),
                       "historical coverage must count actual delivered30 cells")
            _guard(np.all((-3.00001 <= value[0]) & (value[0] <= 4.00001))
                   and np.all((0 < value[3]) & (value[3] <= 1.00001)),
                   "historical physical temperature or uncertainty encoding is outside its contract")
            phase = 2 * np.pi * acquired.timetuple().tm_yday / 365.25
            expected_time = (np.sin(phase), np.cos(phase),
                             (query_time - acquired).total_seconds() / 86400 / 3652.5)
            _guard(all(np.all(value[6 + j] == np.float32(x)) for j, x in enumerate(expected_time)),
                   "historical time fields reused a different query's timestamp")
    return {"root": str(root), "manifest_sha256": expected_sha256,
            "validation_features_sha256": detail["features_sha256"],
            "validation_metadata_sha256": detail["metadata_sha256"],
            "base_metadata_sha256": base_metadata_sha256,
            "raw_manifest_sha256": manifest["raw_manifest_sha256"],
            "inventory_sha256": raw["inventory_sha256"],
            "temporal_alignment_sha256": inventory["temporal_alignment_sha256"],
            "pilot_admissibility_receipt_sha256": manifest["pilot_admissibility_receipt_sha256"],
            "scene_count": 45, "historical_date_slots": 135, "shape": list(array.shape),
            "dtype": str(array.dtype), "finite": True, "source_identity_time_grid_binding_pass": True,
            "missing_source_date_slots": missing_source_slots, "all_cloud_date_slots": empty_thermal_slots,
            "time_contract": "2026 retrospective replay; historical public availability unknown",
            "query_target_arrays_opened": False, "query_masks_opened": False, "locked_test_opened": False}
