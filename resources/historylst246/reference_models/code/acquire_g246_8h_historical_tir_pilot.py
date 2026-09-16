#!/usr/bin/env python3
"""Collect the frozen Fit12 historical thermal pilot under 2026 replay.

Only the six explicitly named assets from the 36 selected 2018--2020 products
are opened. Query archives are used solely for their metadata and file hash.
This collector publishes raw observations, never labels or training features.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit

from affine import Affine
import numpy as np
import rasterio

import g246_data
import build_landsat30_texture_sidecars as transport
from acquire_g246_8h_emissivity_pilot import read_spec, select_pilot
from g246_r2_optical_cache import _transport_env

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "artifacts/g246_8h_20260905/historical_tir_inventory/manifest.json"
OUTPUT = ROOT / "artifacts/g246_8h/historical_tir_pilot_v1"
ASSETS = ("lwir11", "qa", "qa_pixel", "qa_radsat", "emis", "emsd")
SCHEMA = "g246-8h-historical-tir-raw-pilot-v1"
PRODUCT = re.compile(r"^(L[A-Z]\d{2})_L\d[A-Z0-9]{2}_(\d{6})_(\d{8})"
                     r"(?:_(\d{8}))?_\d{2}_[A-Z]\d(?:_|$)")
EXPECTED = {
    "lwir11": ("uint16", 0.00341802, 149.0, "ST_B10.TIF"),
    "qa": ("int16", 0.01, 0.0, "ST_QA.TIF"),
    "qa_pixel": ("uint16", 1.0, 0.0, "QA_PIXEL.TIF"),
    "qa_radsat": ("uint16", 1.0, 0.0, "QA_RADSAT.TIF"),
    "emis": ("int16", 0.0001, 0.0, "ST_EMIS.TIF"),
    "emsd": ("int16", 0.0001, 0.0, "ST_EMSD.TIF"),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def unsigned(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "landsateuwest.blob.core.windows.net":
        raise ValueError("historical asset must use the frozen public Landsat HTTPS source")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def acquisition_identity(identifier):
    """Ignore processing date, level, collection, and tier reprocessing aliases."""
    matched = PRODUCT.match(identifier)
    if matched is None:
        raise ValueError("unrecognized Landsat acquisition identifier")
    return matched.group(1), matched.group(2), matched.group(3)


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source timestamps require explicit timezone")
    return parsed.astimezone(timezone.utc)


def prepare(inventory_path, expected_hash):
    """Offline validation; does not open any remote resource or query array."""
    inventory_path = Path(inventory_path)
    if sha(inventory_path) != expected_hash:
        raise ValueError("frozen inventory SHA256 differs")
    inventory = json.loads(inventory_path.read_text())
    if (inventory.get("schema") != "g246-8h-historical-tir-inventory-v1"
            or inventory.get("status") != "complete" or inventory.get("selected_count") != 36
            or inventory.get("historical_years") != [2018, 2019, 2020]
            or inventory.get("remote_raster_opened") is not False
            or inventory.get("target_arrays_opened") is not False
            or inventory.get("target_masks_opened") is not False
            or inventory.get("locked_test_opened") is not False):
        raise ValueError("inventory scope or closure declaration differs")
    splits = g246_data.load_splits(role="fit+validation")
    public = (*splits.fit, *splits.validation)
    if len(splits.fit) != 603 or len(splits.validation) != 45:
        raise ValueError("public query contract requires Fit603 plus Validation45")
    if any(parse_time(e.datetime).year not in range(2021, 2026) for e in public):
        raise ValueError("a public current query is outside the registered 2021--2025 scope")
    if splits.fit_view_sha256 != inventory["fit_view_sha256"]:
        raise ValueError("Fit descriptor changed since inventory")
    queries = {acquisition_identity(e.item_id) for e in public}
    specs = {e.scene_id: read_spec(e) for e in select_pilot(splits.fit)}
    if set(specs) != set(inventory["scene_ids"]):
        raise ValueError("inventory no longer matches the frozen hash-ranked Fit12")
    rows = inventory["records"]
    expected_pairs = {(sid, year) for sid in specs for year in (2018, 2019, 2020)}
    if len(rows) != 36 or {(r["query_scene_id"], r["historical_year"]) for r in rows} != expected_pairs:
        raise ValueError("inventory is not exactly the 36 fixed city-year slots")
    jobs = []
    for row in rows:
        item = row["selected_item"]
        if item is None or item["id"] != row["ranked_item_ids"][0]:
            raise ValueError("selected product differs from frozen catalogue ranking")
        spec = specs[row["query_scene_id"]]
        p = item["properties"]
        acquired = parse_time(p["datetime"])
        if (acquired.year != row["historical_year"] or acquired.year not in (2018, 2019, 2020)
                or not 5 <= acquired.month <= 9 or acquired >= parse_time(spec.entry.datetime)
                or row["query_datetime"] != spec.entry.datetime):
            raise ValueError("historical observation time violates frozen query separation")
        if (p.get("platform") != "landsat-8" or p.get("landsat:correction") != "L2SP"
                or p.get("landsat:collection_category") != "T1"):
            raise ValueError("historical product must be Landsat8 L2SP Tier1")
        identity = acquisition_identity(item["id"])
        if identity in queries or identity[2] != acquired.strftime("%Y%m%d"):
            raise ValueError("historical product overlaps a query acquisition or alias")
        if not set(ASSETS).issubset(item["assets"]):
            raise ValueError("frozen historical product lacks a required control asset")
        product_names = set()
        for key in ASSETS:
            asset = item["assets"][key]
            name = Path(urlsplit(unsigned(asset["href"])).path).name
            dtype, scale, offset, suffix = EXPECTED[key]
            band = asset["raster:bands"][0]
            if (not name.endswith(suffix) or acquisition_identity(name) != identity
                    or band["data_type"] != dtype or band.get("scale", 1.) != scale
                    or band.get("offset", 0.) != offset):
                raise ValueError("asset identity/type/scale/offset differs from registered product")
            product_names.add(name[:-len(suffix)].rstrip("_"))
        if len(product_names) != 1:
            raise ValueError("assets mix different reprocessed product versions")
        jobs.append((spec, row, product_names.pop()))
    selection = {
        "schema": SCHEMA, "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": expected_hash, "fit_view_sha256": splits.fit_view_sha256,
        "query_scene_ids": sorted(specs), "historical_years": [2018, 2019, 2020],
        "selected_products": [{"query_scene_id": s.entry.scene_id, "historical_year": r["historical_year"],
                               "item_id": r["selected_item"]["id"], "product_id": pid,
                               "item_metadata_sha256": json_sha(r["selected_item"])} for s, r, pid in jobs],
        "assets": list(ASSETS), "selection_rule": "exact frozen inventory selections; no substitutions",
        "resampling": "common nearest reprojection onto metadata-only canonical640 query grid",
        "time_contract": "2026 present-day historical replay of pre2021 observed state",
        "all_public_queries_2021_2025_verified": True,
        "public_query_acquisition_aliases_excluded": True,
        "current_query_target_arrays_opened": False, "current_query_qa_arrays_opened": False,
        "current_query_target_masks_opened": False, "locked_test_opened": False,
        "historical_thermal_observations_opened": True,
        "historical_thermal_observations_opened_meaning": "acquisition run intent; per-asset attempts record actual completed reads",
        "historical_qa_only": True, "new_supervision_created": False,
        "training_features_created": False, "input_admissibility": "separate downstream feature audit required",
        "fallback_policy": "failure retained in its original slot; no date/product replacement; feature fallback not defined by this collector",
        "signed_urls_persisted": False, "source_code_sha256": sha(Path(__file__)),
    }
    return selection, jobs


def acquire_one(spec, row, product_id, output, selection, tokens):
    item = row["selected_item"]
    stem = hashlib.sha256((spec.entry.scene_id + "|" + item["id"]).encode()).hexdigest()[:24]
    receipt_path, raw_path = output / (stem + ".json"), output / (stem + ".npz")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["inventory_sha256"] != selection["inventory_sha256"]
                or receipt["item_metadata_sha256"] != json_sha(item)
                or receipt["query_scene_sha256"] != spec.entry.sha256
                or receipt["raw_sha256"] != sha(raw_path)):
            raise ValueError("existing historical acquisition receipt differs")
        return receipt  # Completed and failed fixed-slot records both remain immutable.
    if raw_path.exists():
        raise ValueError("orphan raw artifact requires explicit inspection before reuse")
    grid = {"crs": spec.metadata["canonical_grid_crs"],
            "transform": Affine(*spec.metadata["transform30"]), "shape": (640, 640)}
    arrays, assets, failures = {}, {}, []
    first_grid = None
    started = utc_now()
    for key in ASSETS:
        asset = item["assets"][key]
        band = asset["raster:bands"][0]
        attempts = []
        for attempt in range(1, 4):
            observed = utc_now()
            try:
                href = transport.builder.signed_href(item, key, tokens.get())
                with rasterio.Env(**_transport_env()):
                    with rasterio.open(href) as ds:
                        signature = transport.builder.assert_delivered_grid(
                            ds, {"fine_delivery_m": 30}, reference=first_grid, label=key)
                        p = item["properties"]
                        if (ds.count != 1 or str(ds.crs) != f"EPSG:{p['proj:epsg']}"
                                or [ds.height, ds.width] != p["proj:shape"]
                                or not ds.transform.almost_equals(Affine(*p["proj:transform"]), precision=1e-8)
                                or ds.dtypes[0] != band["data_type"]):
                            raise ValueError("delivered historical grid/type disagrees with frozen STAC")
                        header = {"scales": list(ds.scales), "offsets": list(ds.offsets),
                                  "units": list(ds.units), "driver": ds.driver, "band_count": ds.count}
                        fill = int(band.get("nodata", 0))
                        array = transport.builder.reproject_asset_to_canonical(ds, grid, fill_value=fill)
                if array.shape != (640, 640) or array.dtype != np.dtype(band["data_type"]):
                    raise ValueError("canonical raw dtype or shape differs")
                first_grid = signature if first_grid is None else first_grid
                arrays[key] = array
                attempts.append({"attempt": attempt, "retrieved_utc": observed,
                                 "completed_utc": utc_now(), "status": "read_complete"})
                assets[key] = {"unsigned_url": unsigned(asset["href"]), "catalogue_type": asset.get("type"),
                               "raster_bands": asset["raster:bands"], "delivered_grid": signature,
                               "source_dtype": band["data_type"], "scale": band.get("scale", 1.),
                               "offset": band.get("offset", 0.), "scale_applied_to_saved_dn": False,
                               "delivered_header": header, "canonical_fill_dn": fill,
                               "canonical_raw_dtype": str(array.dtype), "canonical_raw_shape": list(array.shape),
                               "canonical_raw_dn_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                               "canonical_raw_bytes": array.nbytes,
                               "complete_remote_object_sha256": None,
                               "complete_remote_object_sha256_reason": "windowed canonical extraction, not a complete COG download",
                               "catalogue_file_checksum": asset.get("file:checksum"), "attempts": attempts}
                break
            except Exception as exc:
                # Signed URLs can occur in exceptions; retain only safe type/classification.
                attempts.append({"attempt": attempt, "retrieved_utc": observed,
                                 "completed_utc": utc_now(), "status": "failed", "error_type": type(exc).__name__})
                if attempt < 3 and isinstance(exc, rasterio.errors.RasterioIOError):
                    try:
                        tokens.refresh()
                    except Exception as refresh_error:
                        attempts.append({"status": "token_refresh_failed", "error_type": type(refresh_error).__name__})
                    continue
                failures.append({"asset": key, "unsigned_url": unsigned(asset["href"]), "attempts": attempts})
                break
    temporary = raw_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(raw_path)
    p = item["properties"]
    result = {"schema": SCHEMA, "status": "acquisition_complete" if not failures else "acquisition_partial",
              "inventory_sha256": selection["inventory_sha256"], "item_metadata_sha256": json_sha(item),
              "query_scene_id": spec.entry.scene_id, "query_datetime": spec.entry.datetime,
              "historical_year": row["historical_year"],
              "city": spec.entry.city, "region": spec.entry.region, "query_view_role": "fit",
              "query_scene_sha256": spec.entry.sha256, "item_id": item["id"], "product_id": product_id,
              "acquisition_identity": list(acquisition_identity(item["id"])),
              "acquired_utc": p["datetime"], "processing_date_from_product_id": PRODUCT.match(product_id).group(4),
              "processing_date_precision": "day", "processing_datetime_utc": None,
              "stac_created_utc": p.get("created"), "stac_updated_utc": p.get("updated"),
              "first_public_availability_utc": None, "availability_reason": "not established by catalogue created or processing day",
              "retrieval_started_utc": started, "retrieval_finished_utc": utc_now(),
              "time_contract": selection["time_contract"], "raw_file": raw_path.name,
              "raw_sha256": sha(raw_path), "raw_bytes": raw_path.stat().st_size,
              "raw_stores": "canonical640 nearest-sampled unscaled historical DN arrays",
              "canonical_crs": str(grid["crs"]), "canonical_transform30": list(grid["transform"]),
              "canonical_shape": [640, 640], "resampling": "nearest", "assets": assets, "failures": failures,
              "current_query_target_arrays_opened": False, "current_query_qa_arrays_opened": False,
              "current_query_target_masks_opened": False, "locked_test_opened": False,
              "historical_thermal_observations_opened": "lwir11" in arrays,
              "training_features_created": False, "new_supervision_created": False, "signed_urls_persisted": False}
    atomic_json(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    selection, jobs = prepare(args.inventory, args.inventory_sha256)
    if args.validate_only:
        print(json.dumps({"status": "offline_validation_complete", "fixed_products": len(jobs),
                          "inventory_sha256": args.inventory_sha256, "remote_raster_opened": False,
                          "historical_thermal_observations_opened": False,
                          "current_query_target_arrays_opened": False, "locked_test_opened": False}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("frozen collector selection or code changed")
    else:
        atomic_json(selection_path, selection)
    tokens = transport.TokenManager()
    started = utc_now()
    clock_start = time.monotonic()
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=min(4, max(1, args.workers))) as pool:
        futures = {pool.submit(acquire_one, s, r, pid, args.output, selection, tokens): (s, r)
                   for s, r, pid in jobs}
        for future in as_completed(futures):
            s, r = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"query_scene_id": s.entry.scene_id, "item_id": r["selected_item"]["id"],
                                 "historical_year": r["historical_year"], "error_type": type(exc).__name__})
            complete = sum(record["status"] == "acquisition_complete" for record in records)
            manifest = {**selection, "status": "acquisition_complete" if complete == 36 and not failures else "acquisition_partial",
                        "source_products_complete": complete, "source_products_expected": 36,
                        "source_assets_complete": sum(len(record["assets"]) for record in records),
                        "source_assets_expected": 216, "records": sorted(records, key=lambda a: (a["query_scene_id"], a["acquired_utc"])),
                        "failures": failures, "retrieval_started_utc": started, "retrieval_observed_utc": utc_now(),
                        "elapsed_seconds": time.monotonic() - clock_start,
                        "raw_bytes": sum(record["raw_bytes"] for record in records),
                        "historical_thermal_observations_opened": any(record["historical_thermal_observations_opened"] for record in records)}
            atomic_json(args.output / "manifest.json", manifest)
            event = {"products_complete": complete, "products_finished": len(records),
                     "exception_count": len(failures), "assets_complete": manifest["source_assets_complete"],
                     "observed_utc": utc_now(), "elapsed_seconds": manifest["elapsed_seconds"]}
            with (args.output / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
    if manifest["status"] != "acquisition_complete":
        raise SystemExit("Frozen historical pilot has missing assets; all original slots and sanitized failures retained.")


if __name__ == "__main__":
    main()
