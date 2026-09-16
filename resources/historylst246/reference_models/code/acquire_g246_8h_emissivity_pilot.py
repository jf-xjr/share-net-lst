#!/usr/bin/env python3
"""Acquire only EMIS/EMSD for a frozen twelve-scene Fit pilot.

This is an ancillary-input feasibility audit, not a trained predictor or a
declaration of inference admissibility. No query thermal band or target mask
is requested. The original G246 data and evaluation are unchanged.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from affine import Affine
import numpy as np
import rasterio

import g246_data
import build_landsat30_texture_sidecars as transport
from g246_r2_optical_cache import _transport_env
from train_g246_8h import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ('emis', 'emsd')


def select_pilot(entries):
    cities = {}
    for e in entries:
        cities.setdefault((e.region, e.city), []).append(e)
    selected = []
    for region in sorted({e.region for e in entries}):
        keys = sorted((k for k in cities if k[0] == region),
                      key=lambda k: hashlib.sha256(('emis-pilot-20260905:' + k[1]).encode()).hexdigest())
        for key in keys[:4]:
            dates = sorted(cities[key], key=lambda e: (e.datetime, e.scene_id))
            selected.append(dates[len(dates)//2])
    assert len(selected) == 12 and len({e.city for e in selected}) == 12
    return selected


def read_spec(entry, allow_validation=False):
    if entry.view_role not in (('fit', 'validation') if allow_validation else ('fit',)):
        raise ValueError('this pilot allows Fit scenes only')
    path = g246_data.reject_forbidden_path(entry.file)
    if sha(path) != entry.sha256:
        raise ValueError('source scene content hash differs')
    with np.load(path, allow_pickle=False) as a:
        metadata = json.loads(str(a['metadata'].item()))
    if metadata['item_id'] != entry.item_id or metadata['datetime'] != entry.datetime:
        raise ValueError('source identity differs')
    if not (Affine(*metadata['transform30']) * Affine.scale(4)).almost_equals(
            Affine(*metadata['transform120']), precision=1e-8):
        raise ValueError('canonical 30m/120m grid differs')
    return SimpleNamespace(entry=entry, metadata=metadata)


def acquire(spec, root, tokens):
    entry = spec.entry
    stem = hashlib.sha256(entry.scene_id.encode()).hexdigest()[:16]
    path, receipt_path = root / (stem + '.npz'), root / (stem + '.json')
    if path.exists() and receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old['scene_id'] != entry.scene_id or old['source_scene_sha256'] != entry.sha256 \
                or old['sha256'] != sha(path):
            raise ValueError('existing pilot artifact differs')
        return old
    item = transport.fetch_exact_item(spec)
    if any(k not in item['assets'] for k in ASSETS):
        raise ValueError('required emissivity ancillary asset missing')
    grid = {'crs': spec.metadata['canonical_grid_crs'],
            'transform': Affine(*spec.metadata['transform30']), 'shape': (640, 640)}
    arrays, assets = {}, {}
    for key in ASSETS:
        for attempt in range(3):
            try:
                href = transport.builder.signed_href(item, key, tokens.get())
                with rasterio.Env(**_transport_env()):
                    with rasterio.open(href) as ds:
                        signature = transport.builder.grid_signature(ds)
                        array = transport.builder.reproject_asset_to_canonical(ds, grid, fill_value=0)
                if array.shape != (640, 640) or array.dtype not in (np.dtype('uint16'), np.dtype('int16')):
                    raise ValueError('unexpected ancillary shape/dtype')
                arrays[key] = array
                asset = item['assets'][key]
                assets[key] = {'href_without_query': asset['href'].split('?')[0],
                               'raster_bands': asset.get('raster:bands'),
                               'stored_dtype': str(array.dtype),
                               'delivered_grid': signature,
                               'nonzero_fraction': float((array != 0).mean()),
                               'minimum_dn': int(array.min()), 'maximum_dn': int(array.max())}
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 2:
                    raise
                tokens.refresh()
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix('.tmp').replace(path)
    row = {'scene_id': entry.scene_id, 'city': entry.city, 'region': entry.region,
           'view_role': entry.view_role, 'datetime': entry.datetime, 'item_id': entry.item_id,
           'source_scene_sha256': entry.sha256, 'file': path.name,
           'sha256': sha(path), 'bytes': path.stat().st_size, 'assets': assets,
           'canonical_crs': grid['crs'], 'transform30': list(grid['transform']),
           'canonical_shape': [640, 640], 'resampling': 'nearest',
           'target_arrays_opened': False, 'target_masks_opened': False,
           'locked_test_opened': False, 'sas_persisted': False}
    atomic_json(receipt_path, row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/g246_8h/emissivity_pilot_v1')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = g246_data.load_splits(role='fit+validation')
    entries = select_pilot(splits.fit)
    selection = {'schema': 'g246-8h-emissivity-fit-pilot-v1',
                 'selection': 'four hash-ranked Fit cities per region, median chronological date',
                 'scene_ids': [e.scene_id for e in entries], 'assets': list(ASSETS),
                 'training_started': False, 'inference_admissibility': 'pending independent review',
                 'target_arrays_opened': False, 'target_masks_opened': False, 'locked_test_opened': False}
    selection_path = args.output/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('frozen pilot selection differs')
    atomic_json(selection_path, selection)
    specs = [read_spec(e) for e in entries]
    tokens = transport.TokenManager()
    started = time.time()
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=min(8, max(1, args.workers))) as pool:
        futures = {pool.submit(acquire, s, args.output, tokens): s.entry for s in specs}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                # Raster errors may embed signed URLs; persist only the class.
                failures.append({'scene_id': entry.scene_id, 'error_type': type(exc).__name__})
            progress = {**selection, 'status': 'complete' if len(rows)==12 else 'partial',
                        'rows': sorted(rows, key=lambda r:r['scene_id']), 'failures': failures,
                        'elapsed_seconds': time.time()-started}
            atomic_json(args.output/'manifest.json', progress)
            print(json.dumps({'completed': len(rows), 'failed': len(failures),
                              'elapsed_seconds': time.time()-started}), flush=True)
    if failures:
        raise SystemExit('Ancillary pilot incomplete; sanitized failures saved in manifest.')


if __name__ == '__main__':
    main()
