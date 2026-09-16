#!/usr/bin/env python3
"""Bind fixed earlier thermal observations to every public query; no labels read."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from g246_8h_historical_features import CONTRACT, encode, parse_time
from train_g246_8h import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/g246_8h/cache_v1'
DEFAULT_ROOT = ROOT / 'artifacts/g246_8h/historical_v1'
SCHEMA = 'g246-8h-historical-cache-v1'
YEARS = (2018, 2019, 2020)
EXPECTED = {'lwir11': ('uint16', .00341802, 149.), 'qa': ('int16', .01, 0.),
            'qa_pixel': ('uint16', 1., 0.), 'qa_radsat': ('uint16', 1., 0.),
            'emis': ('int16', .0001, 0.), 'emsd': ('int16', .0001, 0.)}
FORBIDDEN = ('current_query_target_arrays_opened', 'current_query_target_masks_opened',
             'current_query_qa_arrays_opened', 'locked_test_opened')


class HistoricalCache:
    def __init__(self, root, role, expected_scene_ids):
        root = Path(root)
        m = json.loads((root / 'manifest.json').read_text())
        if m.get('schema') != SCHEMA or m.get('status') != 'complete' or \
                m.get('target_arrays_opened') is not False or m.get('locked_test_opened') is not False:
            raise ValueError('historical cache incomplete or outside public predictor scope')
        rows = json.loads((root / role / 'metadata.json').read_text())['scenes']
        if [r['scene_id'] for r in rows] != list(expected_scene_ids):
            raise ValueError('historical scene order differs from base cache')
        self.array = np.load(root / role / 'features.npy', mmap_mode='r', allow_pickle=False)
        if self.array.shape != (len(rows), 3, 9, 160, 160) or self.array.dtype != np.float32:
            raise ValueError('historical feature geometry or encoding differs')


def validate_source(record, city, raw_root):
    if any(record.get(k) is not False for k in FORBIDDEN):
        raise ValueError('source receipt crosses query target boundary')
    if record['query_view_role'] != city['role'] or record['query_scene_id'] != city['query_scene_id'] \
            or record['query_scene_sha256'] != city['query_scene_sha256']:
        raise ValueError('raw canonical query identity differs from fixed inventory')
    for k in ('canonical_crs', 'canonical_transform30', 'canonical_shape'):
        if record[k] != city[k]:
            raise ValueError('raw historical canonical grid differs')
    if record['status'] == 'missing_no_eligible_candidate':
        if (record['city'], record['historical_year']) != ('dazhou_cn', 2020) or record['item_id'] is not None:
            raise ValueError('unexpected no-candidate historical slot')
        return None
    if record['status'] == 'confirmed_non_tiff_missing':
        failures = record.get('failures', [])
        if not failures or not all(f.get('source_confirmation', {}).get('confirmed') is True for f in failures):
            raise ValueError('missing historical raster lacks repeated source confirmation')
        return None
    if record['status'] != 'acquisition_complete':
        raise ValueError('unresolved historical source cannot become a zero fallback')
    if parse_time(record['acquired_utc']).year != record['historical_year']:
        raise ValueError('historical observation differs from its frozen year slot')
    path = (raw_root / record['raw_file']).resolve()
    if not path.is_relative_to(raw_root.resolve()) or sha(path) != record['raw_sha256']:
        raise ValueError('historical raw path or hash differs')
    reference = record['assets']['lwir11']['delivered_grid']
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(EXPECTED):
            raise ValueError('historical source must contain exactly six registered arrays')
        raw = {k: stored[k] for k in EXPECTED}
    for key, (dtype, scale, offset) in EXPECTED.items():
        asset = record['assets'][key]
        if asset['source_dtype'] != dtype or asset['scale'] != scale or asset['offset'] != offset \
                or asset.get('scale_applied_to_saved_dn') is not False:
            raise ValueError('historical radiometric type or scale differs')
        if any(asset['delivered_grid'][k] != reference[k] for k in ('crs', 'shape', 'transform')):
            raise ValueError('historical assets are not on a common delivered grid')
        if raw[key].shape != (640, 640) or raw[key].dtype != np.dtype(dtype) or \
                hashlib.sha256(raw[key].tobytes()).hexdigest() != asset['canonical_raw_dn_sha256']:
            raise ValueError('historical raw DN geometry or hash differs')
        if f"/{record['product_id']}/{record['product_id']}_" not in asset['unsigned_url']:
            raise ValueError('historical assets mix full processing-product identities')
    return raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--raw-root', type=Path, default=ROOT / 'artifacts/g246_8h/historical_tir_full_v1')
    p.add_argument('--pilot-receipt', type=Path, default=ROOT / 'artifacts/g246_8h_20260905/historical_pilot_feature_admissibility_audit.json')
    a = p.parse_args()
    out, raw_root = a.output.resolve(), a.raw_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()).get('status') == 'complete':
        raise FileExistsError('completed historical input cache is immutable')
    start = time.time()
    raw_manifest_path = raw_root / 'manifest.json'
    raw_manifest = json.loads(raw_manifest_path.read_text())
    if raw_manifest.get('schema') != 'g246-8h-historical-tir-full-source-v1' or \
            raw_manifest.get('status') != 'source_plan_complete' or \
            raw_manifest.get('ready_for_feature_build') is not True or raw_manifest['unresolved_failure_slots'] != 0 \
            or len(raw_manifest['records']) != 648 or any(raw_manifest.get(k) is not False for k in FORBIDDEN):
        raise ValueError('full historical acquisition must resolve all 648 city-year slots')
    inventory_path = Path(raw_manifest['inventory_path'])
    if sha(inventory_path) != raw_manifest['inventory_sha256']:
        raise ValueError('historical selection inventory hash differs')
    inv = json.loads(inventory_path.read_text())
    alignment_path = Path(inv['temporal_alignment_path'])
    if sha(alignment_path) != inv['temporal_alignment_sha256']:
        raise ValueError('query temporal grid audit binding differs')
    alignment = json.loads(alignment_path.read_text())
    if alignment.get('locked_test_opened') is not False or not all(
            alignment['roles'][role]['all_three_date_grids_identical'] for role in ('fit', 'validation')):
        raise ValueError('historical reuse requires identical three-date city grids')
    receipt = json.loads(a.pilot_receipt.read_text())
    if receipt.get('pass') is not True:
        raise ValueError('historical pilot admissibility audit must pass')
    cities = {r['city']: r for r in inv['cities']}
    queries = {r['scene_id']: r for r in inv['public_query_scenes']}
    sources = {(r['city'], r['historical_year']): r for r in raw_manifest['records']}
    if len(cities) != 216 or len(queries) != 648 or set(sources) != {(c, y) for c in cities for y in YEARS}:
        raise ValueError('fixed historical city/year universe differs')
    selected = {(r['city'], r['historical_year']): r['selected_item'] for r in inv['records']}
    for key, source in sources.items():
        item = selected[key]
        if source['item_id'] != (item['id'] if item is not None else None):
            raise ValueError('historical source substitutes a frozen selected product')
        if source['status'] == 'acquisition_complete' and source['acquired_utc'] != item['properties']['datetime']:
            raise ValueError('historical observation time differs from frozen catalogue metadata')
    source_dir = out / 'source'
    source_dir.mkdir(exist_ok=True)
    code_hashes = {}
    for path in (Path(__file__), ROOT / 'code/g246_8h_historical_features.py',
                 ROOT / 'code/build_g246_8h_emissivity_cache.py'):
        shutil.copy2(path, source_dir / path.name)
        code_hashes[path.name] = sha(path)
    for path in (a.pilot_receipt, alignment_path):
        shutil.copy2(path, source_dir / path.name)
    pilot_root = ROOT / 'artifacts/g246_8h/historical_tir_pilot_features_v1'
    pilot_manifest = json.loads((pilot_root / 'manifest.json').read_text())
    if sha(pilot_root / 'manifest.json') != '8f8b75097e323f0c85a724814214cc38dfa659a3f5d339a5e5987424b3665b84' \
            or sha(pilot_root / 'features.npy') != pilot_manifest['features_sha256']:
        raise ValueError('audited real pilot feature binding differs')
    pilot_meta = json.loads((pilot_root / 'metadata.json').read_text())['scenes']
    pilot_features = np.load(pilot_root / 'features.npy', mmap_mode='r', allow_pickle=False)
    pilot_ids = {r['scene_id']: i for i, r in enumerate(pilot_meta)}
    pilot_matches = []
    manifest = {'schema': SCHEMA, 'status': 'building', 'base_manifest_sha256': sha(BASE / 'manifest.json'),
        'raw_manifest_path': str(raw_manifest_path), 'raw_manifest_sha256': sha(raw_manifest_path),
        'inventory_path': str(inventory_path), 'inventory_sha256': sha(inventory_path),
        'temporal_alignment_path': str(source_dir / alignment_path.name), 'temporal_alignment_sha256': sha(alignment_path),
        'pilot_admissibility_receipt': str(source_dir / a.pilot_receipt.name),
        'pilot_admissibility_receipt_sha256': sha(a.pilot_receipt),
        'pilot_feature_manifest_sha256': sha(pilot_root / 'manifest.json'),
        'contract': CONTRACT, 'source_code_sha256': code_hashes, 'roles': {},
        'target_arrays_opened': False, 'target_masks_opened': False, 'locked_test_opened': False,
        'current_query_qa_arrays_opened': False, 'new_supervision_created': False,
        'training_supervision': 'unchanged Fit603; historical observations are new predictor inputs',
        'query_scene_count': 648, 'historical_city_year_slots': 648,
        'missing_source_policy': 'whole historical date nine fields zero; retain every query and scoring pixel',
        'source_products_complete': raw_manifest['source_products_complete'],
        'no_candidate_slots': raw_manifest['no_candidate_slots'],
        'confirmed_non_tiff_slots': raw_manifest['confirmed_non_tiff_slots']}
    atomic_json(out / 'manifest.json', manifest)
    for role, expected_n in (('fit', 603), ('validation', 45)):
        base_records = json.loads((BASE / role / 'metadata.json').read_text())['scenes']
        if len(base_records) != expected_n:
            raise ValueError('query role count differs')
        directory = out / role
        directory.mkdir(exist_ok=True)
        array = np.lib.format.open_memmap(directory / 'features.npy', mode='w+', dtype='float32',
                                         shape=(expected_n, 3, 9, 160, 160))
        rows, cached_city, encoded = [], None, None
        for index, base_row in enumerate(base_records):
            city = cities[base_row['city']]
            query = queries[base_row['scene_id']]
            if query['datetime'] != base_row['datetime'] or query['source_sha256'] != base_row['source_sha256'] \
                    or query['role'] != role or city['role'] != role:
                raise ValueError('current query metadata differs from frozen source inventory')
            audited = next(r for r in alignment['roles'][role]['cities'][city['city']]
                           if r['scene_id'] == base_row['scene_id'])
            if audited['declared_source_sha256'] != base_row['source_sha256'] or \
                    audited['grid_signature_sha256'] != city['grid_signature_sha256']:
                raise ValueError('query source or canonical grid differs from temporal alignment audit')
            if cached_city != city['city']:
                encoded = []
                for year in YEARS:
                    source = sources[(city['city'], year)]
                    raw = validate_source(source, city, raw_root)
                    if raw is None:
                        value = np.zeros((9, 160, 160), np.float32)
                        stats = {'all_thermal_missing': True, 'historical_clear_fraction30': 0.,
                                 'historical_nonempty_fraction120': 0., 'historical_clear_count30': 0,
                                 'historical_datetime': source.get('acquired_utc')}
                    else:
                        value, stats = encode(raw, source['acquired_utc'], city['query_datetime'])
                    encoded.append((value, stats))
                cached_city = city['city']
            source_rows = []
            for slot, year in enumerate(YEARS):
                source = sources[(city['city'], year)]
                value, stats = encoded[slot]
                array[index, slot] = value
                if not stats['all_thermal_missing']:
                    age = (parse_time(base_row['datetime']) - parse_time(source['acquired_utc'])).total_seconds() / 86400 / 3652.5
                    if age <= 0:
                        raise ValueError('historical source is not earlier than current query')
                    array[index, slot, 8] = age
                source_rows.append({**stats, 'query_datetime': base_row['datetime'], 'year': year,
                    'historical_year': year, 'source_available': source['status'] == 'acquisition_complete',
                    'item_id': source['item_id'], 'product_id': source.get('product_id'),
                    'raw_sha256': source['raw_sha256'], 'raw_file': source['raw_file'],
                    'status': source['status'], 'missing_reason': None if source['status'] == 'acquisition_complete' else source['status']})
            if base_row['scene_id'] in pilot_ids:
                if not np.array_equal(array[index], pilot_features[pilot_ids[base_row['scene_id']]]):
                    raise ValueError('full historical cache differs from audited pilot query features')
                pilot_matches.append(base_row['scene_id'])
            rows.append({'scene_id': base_row['scene_id'], 'city': city['city'], 'region': city['region'],
                'query_datetime': base_row['datetime'], 'query_source_sha256': base_row['source_sha256'],
                'canonical_crs': city['canonical_crs'], 'canonical_transform30': city['canonical_transform30'],
                'canonical_shape': city['canonical_shape'], 'grid_signature_sha256': city['grid_signature_sha256'],
                'sources': source_rows})
            if (index + 1) % 60 == 0:
                print(json.dumps({'event': 'encode', 'role': role, 'query_count': index + 1,
                                  'elapsed_seconds': time.time() - start}), flush=True)
        array.flush()
        del array
        atomic_json(directory / 'metadata.json', {'schema': SCHEMA, 'role': role, 'scenes': rows})
        manifest['roles'][role] = {'scene_count': len(rows), 'base_metadata_sha256': sha(BASE / role / 'metadata.json'),
            'metadata_sha256': sha(directory / 'metadata.json'), 'features_sha256': sha(directory / 'features.npy'),
            'features_shape': [expected_n, 3, 9, 160, 160], 'features_dtype': 'float32',
            'features_bytes': (directory / 'features.npy').stat().st_size,
            'all_missing_source_dates': sum(s['all_thermal_missing'] for r in rows for s in r['sources'])}
        atomic_json(out / 'manifest.json', manifest)
    if set(pilot_matches) != set(pilot_ids):
        raise ValueError('full historical feature cache must reproduce all twelve pilot query scenes')
    if sha(raw_manifest_path) != manifest['raw_manifest_sha256']:
        raise ValueError('historical raw manifest changed while encoding')
    manifest.update(status='complete', pilot_twelve_queries_bitwise_identical=True,
                    pilot_query_scene_ids=sorted(pilot_matches), elapsed_seconds=time.time() - start)
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps({'event': 'complete', 'manifest_sha256': sha(out / 'manifest.json'),
                      'roles': manifest['roles'], 'elapsed_seconds': time.time() - start}), flush=True)


if __name__ == '__main__':
    main()
