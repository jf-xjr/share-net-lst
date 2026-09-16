#!/usr/bin/env python3
"""Add query-season matched historical slots to the frozen three-source cache."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from build_g246_8h_historical_cache import validate_source
from g246_8h_historical_features import CONTRACT, encode, parse_time
from train_g246_8h import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/g246_8h/cache_v1'
V1 = ROOT / 'artifacts/g246_8h/historical_v1'
SCHEMA = 'g246-8h-historical-six-cache-v1'
YEARS = (2018, 2019, 2020)


class HistoricalSixCache:
    def __init__(self, root, role, expected_scene_ids):
        root = Path(root)
        m = json.loads((root / 'manifest.json').read_text())
        if m.get('schema') != SCHEMA or m.get('status') != 'complete' or \
                m.get('target_arrays_opened') is not False or m.get('locked_test_opened') is not False:
            raise ValueError('six-source historical cache incomplete or outside public input scope')
        rows = json.loads((root / role / 'metadata.json').read_text())['scenes']
        if [r['scene_id'] for r in rows] != list(expected_scene_ids):
            raise ValueError('six-source historical query order differs')
        self.array = np.load(root / role / 'features.npy', mmap_mode='r', allow_pickle=False)
        if self.array.shape != (len(rows), 6, 9, 160, 160) or self.array.dtype != np.float32:
            raise ValueError('six-source historical tensor shape or dtype differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=ROOT / 'artifacts/g246_8h/historical_tir_seasonal_v2')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/g246_8h/historical_six_v2')
    a = p.parse_args()
    raw_root, out = a.raw_root.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()).get('status') == 'complete':
        raise FileExistsError('completed six-source historical cache is immutable')
    started = time.time()
    raw_path = raw_root / 'manifest.json'
    raw = json.loads(raw_path.read_text())
    if raw.get('schema') != 'g246-8h-historical-seasonal-source-v2' or \
            raw.get('status') != 'source_plan_complete' or raw.get('ready_for_feature_build') is not True:
        raise ValueError('supplementary historical source plan must be fully resolved')
    if any(raw.get(k) is not False for k in ('current_query_target_arrays_opened',
            'current_query_target_masks_opened', 'current_query_qa_arrays_opened', 'locked_test_opened')):
        raise ValueError('supplementary source plan crossed the query-information boundary')
    if sha(raw_root / 'selection.json') != raw['selection_sha256']:
        raise ValueError('supplementary historical selection hash differs')
    old = json.loads((V1 / 'manifest.json').read_text())
    audit_path = ROOT / 'artifacts/g246_8h_20260905/historical_full_cache_audit.json'
    audit = json.loads(audit_path.read_text())
    if old.get('status') != 'complete' or old.get('contract') != CONTRACT or \
            audit.get('pass') is not True or audit['manifest_sha256'] != sha(V1 / 'manifest.json'):
        raise ValueError('frozen original historical cache must have a matching complete audit')
    if raw['v1_source_manifest_sha256'] != old['raw_manifest_sha256']:
        raise ValueError('supplementary selection refers to a different original source universe')
    if sha(Path(raw['v1_source_manifest_path'])) != raw['v1_source_manifest_sha256'] or \
            sha(Path(raw['inventory_manifest_path'])) != raw['inventory_manifest_sha256']:
        raise ValueError('supplementary source inventory binding differs')
    query_plan = {r['scene_id']: r for r in raw['query_records']}
    records = {r['source_key']: r for r in raw['records']}
    if len(query_plan) != len(raw['query_records']):
        raise ValueError('supplementary query plan has duplicate scenes')
    if len(query_plan) != 648 or len(records) != len(raw['records']) or len(records) != 1186:
        raise ValueError('frozen supplementary query/source counts differ')
    if any(r['status'] not in ('acquisition_complete', 'confirmed_non_tiff_missing') for r in records.values()):
        raise ValueError('unresolved supplementary source cannot become a zero feature')
    source = out / 'source'
    source.mkdir(exist_ok=True)
    code_hashes = {}
    for path in (Path(__file__), ROOT / 'code/build_g246_8h_historical_cache.py',
                 ROOT / 'code/g246_8h_historical_features.py', ROOT / 'code/build_g246_8h_emissivity_cache.py'):
        shutil.copy2(path, source / path.name)
        code_hashes[path.name] = sha(path)
    shutil.copy2(audit_path, source / audit_path.name)
    manifest = {'schema': SCHEMA, 'status': 'building', 'source_count': 6,
        'base_manifest_sha256': sha(BASE / 'manifest.json'),
        'base_historical_manifest_path': str(V1 / 'manifest.json'),
        'base_historical_manifest_sha256': sha(V1 / 'manifest.json'),
        'base_historical_audit_path': str(source / audit_path.name),
        'base_historical_audit_sha256': sha(audit_path),
        'raw_manifest_path': str(raw_path), 'raw_manifest_sha256': sha(raw_path),
        'selection_path': str(raw_root / 'selection.json'), 'selection_sha256': raw['selection_sha256'],
        'contract': {'encoding': CONTRACT, 'source_count': 6,
            'slots': ['v1_2018', 'v1_2019', 'v1_2020', 'query_season_2018', 'query_season_2019', 'query_season_2020'],
            'seasonal_rule': 'query-specific closest circular month/day within frozen May-Sep candidates; catalogue cloud<=40; ties cloud/datetime/id',
            'duplicate_policy': 'additional selected item already in corresponding V1 slot: all nine additional fields zero; never double-weight',
            'missing_policy': 'no candidate or confirmed unavailable raster: all nine fields zero; retain every query',
            'time_contract': '2026 present-day historical replay; observed pre2021, historical public availability unproven'},
        'source_code_sha256': code_hashes, 'roles': {},
        'target_arrays_opened': False, 'target_masks_opened': False, 'current_query_qa_arrays_opened': False,
        'locked_test_opened': False, 'new_supervision_created': False,
        'training_supervision': 'unchanged Fit603', 'query_scene_count': 648,
        'source_availability_semantics': 'physical raster source availability; slot_supplied and slot_active separately describe deduplication and thermal validity'}
    atomic_json(out / 'manifest.json', manifest)
    counts = {'duplicate_v1': 0, 'no_candidate': 0, 'new_source': 0}
    used_source_keys = set()
    for role, expected_n in (('fit', 603), ('validation', 45)):
        base_meta_path = BASE / role / 'metadata.json'
        base_rows = json.loads(base_meta_path.read_text())['scenes']
        old_detail = old['roles'][role]
        if sha(V1 / role / 'features.npy') != old_detail['features_sha256'] or \
                sha(V1 / role / 'metadata.json') != old_detail['metadata_sha256']:
            raise ValueError('original three-source cache bytes changed')
        old_rows = json.loads((V1 / role / 'metadata.json').read_text())['scenes']
        if len(base_rows) != expected_n or [r['scene_id'] for r in old_rows] != [r['scene_id'] for r in base_rows]:
            raise ValueError('original historical/base query order differs')
        old_array = np.load(V1 / role / 'features.npy', mmap_mode='r', allow_pickle=False)
        directory = out / role
        directory.mkdir(exist_ok=True)
        array = np.lib.format.open_memmap(directory / 'features.npy', mode='w+', dtype='float32',
                                         shape=(expected_n, 6, 9, 160, 160))
        rows, cached_city, encoded = [], None, {}
        for i, (base, original) in enumerate(zip(base_rows, old_rows)):
            q = query_plan[base['scene_id']]
            if q['city'] != base['city'] or q['region'] != base['region'] or q['role'] != role or \
                    q['query_datetime'] != base['datetime'] or q['query_source_sha256'] != base['source_sha256']:
                raise ValueError('supplementary query identity differs from base')
            if any(q[k] != original[k] for k in ('canonical_crs', 'canonical_transform30', 'canonical_shape', 'grid_signature_sha256')):
                raise ValueError('supplementary query grid differs from audited original history')
            if [s['historical_year'] for s in q['slots']] != list(YEARS):
                raise ValueError('supplementary year slot order differs')
            if cached_city != q['city']:
                cached_city, encoded = q['city'], {}
            array[i] = 0
            array[i, :3] = old_array[i]
            assert np.array_equal(array[i, :3], old_array[i])
            source_rows = []
            for j, old_row in enumerate(original['sources']):
                source_rows.append({**copy.deepcopy(old_row), 'slot_index': j, 'slot_kind': 'v1',
                    'slot_supplied': old_row['source_available'],
                    'slot_active': bool(np.any(array[i, j, 2] > 0)),
                    'slot_all_zero': bool(not np.any(array[i, j])), 'source_key': None})
            for j, slot in enumerate(q['slots']):
                kind = slot['status']
                if kind not in counts:
                    raise ValueError('unknown supplementary historical slot status')
                counts[kind] += 1
                old_row = original['sources'][j]
                if slot['v1_item_id'] != old_row['item_id']:
                    raise ValueError('supplementary deduplication compares a different V1 item')
                entry = {'slot_index': j + 3, 'slot_kind': kind, 'historical_year': YEARS[j], 'year': YEARS[j],
                    'item_id': slot['selected_item_id'], 'historical_datetime': slot['selected_datetime'],
                    'query_datetime': base['datetime'], 'seasonal_distance_days': slot['seasonal_distance_days'],
                    'source_key': slot['source_key'], 'v1_item_id': slot['v1_item_id']}
                if kind == 'duplicate_v1':
                    if slot['source_key'] is not None or slot['selected_item_id'] != old_row['item_id'] \
                            or slot['selected_datetime'] != old_row['historical_datetime']:
                        raise ValueError('duplicate slot is not the existing source')
                    entry.update(source_available=old_row['source_available'], slot_supplied=False,
                        all_thermal_missing=old_row['all_thermal_missing'], raw_sha256=old_row['raw_sha256'],
                        raw_file=old_row['raw_file'], missing_reason='duplicate_v1_zero_additional_slot')
                elif kind == 'no_candidate':
                    if slot['selected_item_id'] is not None or slot['source_key'] is not None:
                        raise ValueError('no-candidate slot contains a source')
                    entry.update(source_available=False, slot_supplied=False, all_thermal_missing=True,
                                 raw_sha256=None, raw_file=None, missing_reason='no_candidate')
                else:
                    key = slot['source_key']
                    r = records[key]
                    used_source_keys.add(key)
                    if r['city'] != q['city'] or r['historical_year'] != YEARS[j] or \
                            r['item_id'] != slot['selected_item_id'] or r['acquired_utc'] != slot['selected_datetime'] \
                            or base['scene_id'] not in r['query_scene_ids']:
                        raise ValueError('supplementary historical source is not its frozen query selection')
                    if key not in encoded:
                        anchor = query_plan[r['query_scene_id']]
                        city = {**anchor, 'query_scene_id': anchor['scene_id'],
                                'query_scene_sha256': anchor['query_source_sha256']}
                        raw_arrays = validate_source(r, city, raw_root)
                        if raw_arrays is None:
                            value = np.zeros((9, 160, 160), np.float32)
                            stats = {'all_thermal_missing': True, 'historical_datetime': r['acquired_utc'],
                                     'historical_clear_fraction30': 0., 'historical_nonempty_fraction120': 0.,
                                     'historical_clear_count30': 0}
                        else:
                            value, stats = encode(raw_arrays, r['acquired_utc'], base['datetime'])
                        encoded[key] = (value, stats)
                    value, stats = encoded[key]
                    array[i, j + 3] = value
                    if not stats['all_thermal_missing']:
                        age = (parse_time(base['datetime']) - parse_time(r['acquired_utc'])).total_seconds() / 86400 / 3652.5
                        if age <= 0:
                            raise ValueError('supplementary source is not earlier than query')
                        array[i, j + 3, 8] = age
                    available = r['status'] == 'acquisition_complete'
                    entry.update(**stats)
                    entry.update(query_datetime=base['datetime'], source_available=available, slot_supplied=available,
                        raw_sha256=r['raw_sha256'], raw_file=r['raw_file'], product_id=r.get('product_id'),
                        source_status=r['status'], missing_reason=None if available else r['status'])
                entry['slot_active'] = bool(np.any(array[i, j + 3, 2] > 0))
                entry['slot_all_zero'] = bool(not np.any(array[i, j + 3]))
                source_rows.append(entry)
            rows.append({k: original[k] for k in ('scene_id', 'city', 'region', 'query_datetime',
                'query_source_sha256', 'canonical_crs', 'canonical_transform30', 'canonical_shape', 'grid_signature_sha256')})
            rows[-1]['sources'] = source_rows
            if (i + 1) % 60 == 0:
                print(json.dumps({'event': 'six_source_encode', 'role': role, 'query_count': i + 1,
                                  'elapsed_seconds': time.time() - started}), flush=True)
        array.flush()
        del array
        atomic_json(directory / 'metadata.json', {'schema': SCHEMA, 'role': role, 'scenes': rows})
        manifest['roles'][role] = {'scene_count': expected_n, 'base_metadata_sha256': sha(base_meta_path),
            'base_historical_metadata_sha256': old_detail['metadata_sha256'],
            'base_historical_features_sha256': old_detail['features_sha256'],
            'features_shape': [expected_n, 6, 9, 160, 160], 'features_dtype': 'float32',
            'features_bytes': (directory / 'features.npy').stat().st_size,
            'features_sha256': sha(directory / 'features.npy'), 'metadata_sha256': sha(directory / 'metadata.json'),
            'additional_slot_active_count': sum(s['slot_active'] for r in rows for s in r['sources'][3:])}
        atomic_json(out / 'manifest.json', manifest)
    if counts != {'duplicate_v1': 388, 'no_candidate': 6, 'new_source': 1550} or used_source_keys != set(records):
        raise ValueError('supplementary reference/source counts changed from the frozen policy')
    if sha(raw_path) != manifest['raw_manifest_sha256']:
        raise ValueError('supplementary source manifest changed during feature encoding')
    manifest.update(status='complete', supplementary_slot_counts=counts,
                    base_three_slots_bitwise_identical=True, elapsed_seconds=time.time() - started)
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps({'event': 'six_source_complete', 'manifest_sha256': sha(out / 'manifest.json'),
                      'roles': manifest['roles'], 'elapsed_seconds': time.time() - started}), flush=True)


if __name__ == '__main__':
    main()
