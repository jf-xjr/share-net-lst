#!/usr/bin/env python3
"""Append one fully excluded, pre-query recent observation to audited six slots."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from acquire_g246_8h_historical_tir_pilot import acquisition_identity
from build_g246_8h_historical_cache import validate_source
from g246_8h_recent_historical_features import CONTRACT as RECENT_CONTRACT, encode, parse_time
from train_g246_8h import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/g246_8h/cache_v1'
SIX = ROOT / 'artifacts/g246_8h/historical_six_v2'
SCHEMA = 'g246-8h-historical-seven-cache-v1'


def read(path):
    return json.loads(Path(path).read_text())


def alias_digest(identifier):
    value = '|'.join(('g246-acquisition-alias-v1', *acquisition_identity(identifier)))
    return hashlib.sha256(value.encode('ascii')).hexdigest()


class HistoricalSevenCache:
    def __init__(self, root, role, expected_scene_ids):
        root = Path(root)
        manifest = read(root / 'manifest.json')
        if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete' or \
                manifest.get('target_arrays_opened') is not False or manifest.get('locked_test_opened') is not False:
            raise ValueError('seven-source historical cache is incomplete or outside the input contract')
        rows = read(root / role / 'metadata.json')['scenes']
        if [r['scene_id'] for r in rows] != list(expected_scene_ids):
            raise ValueError('seven-source query scene order differs')
        self.array = np.load(root / role / 'features.npy', mmap_mode='r', allow_pickle=False)
        if self.array.shape != (len(rows), 7, 9, 160, 160) or self.array.dtype != np.float32:
            raise ValueError('seven-source historical feature shape or dtype differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'artifacts/g246_8h/historical_tir_recent_raw_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/g246_8h/historical_seven_v3')
    args = parser.parse_args()
    raw_root, out = args.raw_root.resolve(), args.output.resolve()
    if (out / 'manifest.json').exists() and read(out / 'manifest.json').get('status') == 'complete':
        raise FileExistsError('completed historical input cache is immutable')
    started = time.monotonic()
    raw_path = raw_root / 'manifest.json'
    raw = read(raw_path)
    if raw.get('schema') != 'g246-8h-historical-recent-source-v2' or \
            raw.get('status') != 'source_plan_complete' or raw.get('ready_for_feature_build') is not True:
        raise ValueError('recent source plan must be complete before feature construction')
    if any(raw.get(k) is not False for k in ('current_query_target_arrays_opened',
            'current_query_target_masks_opened', 'current_query_qa_arrays_opened', 'locked_test_opened')):
        raise ValueError('recent source plan crossed the query-information boundary')
    bindings = {}
    for name in ('inventory_manifest', 'selection', 'campaign_denylist', 'campaign_denylist_audit'):
        path = Path(raw[name + '_path'])
        if sha(path) != raw[name + '_sha256']:
            raise ValueError(f'recent source {name} binding differs')
        bindings[name] = path
    deny = read(bindings['campaign_denylist'])
    deny_audit = read(bindings['campaign_denylist_audit'])
    if deny.get('status') != 'complete' or deny.get('source_scene_count') != 738 or \
            deny.get('acquisition_count') != 633 or deny.get('namespace') != 'g246-acquisition-alias-v1' or \
            deny_audit.get('pass') is not True or deny_audit.get('status') != 'complete' or \
            deny_audit['denylist_sha256'] != raw['campaign_denylist_sha256']:
        raise ValueError('full campaign acquisition exclusion is not established')
    denied = set(deny['acquisition_digests'])
    if len(denied) != 633:
        raise ValueError('full acquisition exclusion count differs')
    previous = read(SIX / 'manifest.json')
    previous_audit_path = ROOT / 'artifacts/g246_8h_20260905/historical_six_full_cache_audit.json'
    previous_audit = read(previous_audit_path)
    if previous.get('status') != 'complete' or previous_audit.get('pass') is not True or \
            previous_audit.get('status') != 'complete' or previous_audit['manifest_sha256'] != sha(SIX / 'manifest.json'):
        raise ValueError('original six-source cache needs its exact complete independent audit')
    if previous['base_manifest_sha256'] != sha(BASE / 'manifest.json'):
        raise ValueError('six-source base query universe differs')
    queries = {r['scene_id']: r for r in raw['query_records']}
    sources = {r['source_key']: r for r in raw['records']}
    if len(queries) != 648 or len(queries) != len(raw['query_records']) or len(sources) != len(raw['records']):
        raise ValueError('recent query universe or source uniqueness differs')
    if any(r['status'] not in ('acquisition_complete', 'confirmed_non_tiff_missing') for r in sources.values()):
        raise ValueError('unresolved recent source cannot become a missing modality')
    out.mkdir(parents=True, exist_ok=True)
    source_dir = out / 'source'
    source_dir.mkdir(exist_ok=True)
    code_hashes = {}
    for path in (Path(__file__), ROOT / 'code/build_g246_8h_historical_cache.py',
                 ROOT / 'code/g246_8h_historical_features.py', ROOT / 'code/g246_8h_recent_historical_features.py',
                 ROOT / 'code/acquire_g246_8h_historical_tir_pilot.py'):
        shutil.copy2(path, source_dir / path.name)
        code_hashes[path.name] = sha(path)
    for path in (previous_audit_path, bindings['campaign_denylist'], bindings['campaign_denylist_audit']):
        shutil.copy2(path, source_dir / path.name)
    manifest = {'schema': SCHEMA, 'status': 'building', 'source_count': 7,
        'base_manifest_sha256': sha(BASE / 'manifest.json'),
        'base_historical_manifest_path': str(SIX / 'manifest.json'),
        'base_historical_manifest_sha256': sha(SIX / 'manifest.json'),
        'base_historical_audit_path': str(source_dir / previous_audit_path.name),
        'base_historical_audit_sha256': sha(previous_audit_path),
        'raw_manifest_path': str(raw_path), 'raw_manifest_sha256': sha(raw_path),
        **{key: raw[key] for key in raw if key in {
            name + suffix for name in ('inventory_manifest', 'selection', 'campaign_denylist', 'campaign_denylist_audit')
            for suffix in ('_path', '_sha256')}},
        'contract': {'encoding': {'base_six': previous['contract']['encoding'], 'recent': RECENT_CONTRACT}, 'source_count': 7,
            'slots': previous['contract']['slots'] + ['recent_pre_query_8_to_64_days'],
            'recent_rule': 'full 633 acquisition aliases excluded before ranking; UTC age 8<=days<=64; L8/L9 L2SP T1 whole AOI; catalogue cloud<=40; age/cloud/datetime/item id',
            'time_contract': '2026 present-day historical replay; product availability at query time unproven',
            'recent_interpretation': 'historical spatial template, not hours-scale thermal initial state',
            'missing_policy': 'no candidate or confirmed unavailable raster or historical QA empty: entire recent nine fields zero; retain every query'},
        'roles': {}, 'source_code_sha256': code_hashes,
        'target_arrays_opened': False, 'target_masks_opened': False, 'current_query_qa_arrays_opened': False,
        'locked_test_opened': False, 'new_supervision_created': False,
        'training_supervision': 'unchanged Fit603', 'query_scene_count': 648}
    atomic_json(out / 'manifest.json', manifest)
    counts = {'new_source': 0, 'no_candidate': 0}
    used = set()
    for role, expected_n in (('fit', 603), ('validation', 45)):
        base_path = BASE / role / 'metadata.json'
        base_rows = read(base_path)['scenes']
        old_detail = previous['roles'][role]
        for key, filename in (('features_sha256', 'features.npy'), ('metadata_sha256', 'metadata.json')):
            if sha(SIX / role / filename) != old_detail[key]:
                raise ValueError('audited original six-source bytes changed')
        old_rows = read(SIX / role / 'metadata.json')['scenes']
        if len(base_rows) != expected_n or [r['scene_id'] for r in old_rows] != [r['scene_id'] for r in base_rows]:
            raise ValueError('six-source and query scene membership differ')
        old_array = np.load(SIX / role / 'features.npy', mmap_mode='r')
        directory = out / role
        directory.mkdir(exist_ok=True)
        array = np.lib.format.open_memmap(directory / 'features.npy', mode='w+', dtype='float32',
                                         shape=(expected_n, 7, 9, 160, 160))
        rows = []
        for index, (base, original) in enumerate(zip(base_rows, old_rows)):
            q = queries[base['scene_id']]
            if q['city'] != base['city'] or q['region'] != base['region'] or q['role'] != role or \
                    q['query_datetime'] != base['datetime'] or q['query_source_sha256'] != base['source_sha256']:
                raise ValueError('recent query identity differs')
            if any(q[k] != original[k] for k in ('canonical_crs', 'canonical_transform30', 'canonical_shape', 'grid_signature_sha256')):
                raise ValueError('recent query grid differs')
            if len(q['slots']) != 1:
                raise ValueError('the recent contract supplies exactly one additional slot')
            slot = q['slots'][0]
            kind = slot['status']
            if kind not in counts:
                raise ValueError('unknown recent slot status')
            counts[kind] += 1
            array[index] = 0
            array[index, :6] = old_array[index]
            assert np.array_equal(array[index, :6], old_array[index])
            entry = {'slot_index': 6, 'slot_kind': kind, 'item_id': slot['selected_item_id'],
                'historical_datetime': slot['selected_datetime'], 'query_datetime': base['datetime'],
                'source_key': slot['source_key'], 'age_days': slot.get('age_days')}
            if kind == 'no_candidate':
                if slot['selected_item_id'] is not None or slot['source_key'] is not None:
                    raise ValueError('no-candidate recent slot contains a source')
                entry.update(source_available=False, slot_supplied=False, all_thermal_missing=True,
                             raw_sha256=None, raw_file=None, missing_reason='no_candidate')
            else:
                key = slot['source_key']
                r = sources[key]
                if r['city'] != q['city'] or r['item_id'] != slot['selected_item_id'] or \
                        r['acquired_utc'] != slot['selected_datetime'] or base['scene_id'] not in r['query_scene_ids']:
                    raise ValueError('recent raw source is not the frozen selected query observation')
                age = (parse_time(base['datetime']) - parse_time(r['acquired_utc'])).total_seconds() / 86400
                if not 8 <= age <= 64 or abs(age - slot['age_days']) > 1e-9 or alias_digest(r['item_id']) in denied:
                    raise ValueError('recent source violates exact chronology or full campaign exclusion')
                anchor = queries[r['query_scene_id']]
                anchor_city = {**anchor, 'query_scene_id': anchor['scene_id'],
                               'query_scene_sha256': anchor['query_source_sha256']}
                raw_arrays = validate_source(r, anchor_city, raw_root)
                if raw_arrays is None:
                    stats = {'all_thermal_missing': True, 'historical_datetime': r['acquired_utc'],
                             'historical_clear_fraction30': 0., 'historical_nonempty_fraction120': 0.,
                             'historical_clear_count30': 0}
                else:
                    value, stats = encode(raw_arrays, r['acquired_utc'], base['datetime'])
                    array[index, 6] = value
                available = r['status'] == 'acquisition_complete'
                entry.update(**stats)
                entry.update(historical_year=r['historical_year'], year=r['historical_year'],
                    source_available=available, slot_supplied=available,
                    raw_sha256=r['raw_sha256'], raw_file=r['raw_file'], product_id=r.get('product_id'),
                    source_status=r['status'], missing_reason=None if available else r['status'],
                    campaign_acquisition_exclusion_pass=True, acquisition_digest=alias_digest(r['item_id']))
                used.add(key)
            entry['slot_active'] = bool(np.any(array[index, 6, 2] > 0))
            entry['slot_all_zero'] = bool(not np.any(array[index, 6]))
            rows.append(copy.deepcopy(original))
            rows[-1]['sources'].append(entry)
            if (index + 1) % 60 == 0:
                print(json.dumps({'event': 'seven_source_encode', 'role': role, 'query_count': index + 1,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        array.flush()
        del array
        atomic_json(directory / 'metadata.json', {'schema': SCHEMA, 'role': role, 'scenes': rows})
        manifest['roles'][role] = {'scene_count': expected_n, 'base_metadata_sha256': sha(base_path),
            'base_historical_metadata_sha256': old_detail['metadata_sha256'],
            'base_historical_features_sha256': old_detail['features_sha256'],
            'features_shape': [expected_n, 7, 9, 160, 160], 'features_dtype': 'float32',
            'features_bytes': (directory / 'features.npy').stat().st_size,
            'features_sha256': sha(directory / 'features.npy'), 'metadata_sha256': sha(directory / 'metadata.json'),
            'additional_slot_active_count': sum(r['sources'][6]['slot_active'] for r in rows)}
        atomic_json(out / 'manifest.json', manifest)
    if sum(counts.values()) != 648 or used != set(sources) or sha(raw_path) != manifest['raw_manifest_sha256']:
        raise ValueError('recent source universe or raw manifest changed during encoding')
    manifest.update(status='complete', supplementary_slot_counts=counts,
                    base_six_slots_bitwise_identical=True, elapsed_seconds=time.monotonic() - started)
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps({'event': 'seven_source_complete', 'manifest_sha256': sha(out / 'manifest.json'),
                      'roles': manifest['roles'], 'elapsed_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
