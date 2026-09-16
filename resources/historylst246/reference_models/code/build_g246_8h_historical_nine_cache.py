#!/usr/bin/env python3
"""Preserve the audited first seven slots and append recent ranks two and three."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from acquire_g246_8h_historical_tir_pilot import acquisition_identity
from build_g246_8h_historical_cache import validate_source
from build_g246_8h_historical_seven_cache import BASE, ROOT, alias_digest, read
from g246_8h_recent_historical_features import CONTRACT as RECENT_CONTRACT, encode, parse_time
from train_g246_8h import atomic_json, sha

SEVEN = ROOT / 'artifacts/g246_8h/historical_seven_v3'
SCHEMA = 'g246-8h-historical-nine-cache-v1'


def overpass(identifier, acquired):
    platform, pathrow, _ = acquisition_identity(identifier)
    return platform, pathrow[:3], parse_time(acquired).date().isoformat()


class HistoricalNineCache:
    def __init__(self, root, role, expected_scene_ids):
        root = Path(root)
        manifest = read(root / 'manifest.json')
        if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete' or \
                manifest.get('target_arrays_opened') is not False or manifest.get('locked_test_opened') is not False:
            raise ValueError('nine-source cache is incomplete or outside the input contract')
        rows = read(root / role / 'metadata.json')['scenes']
        if [r['scene_id'] for r in rows] != list(expected_scene_ids):
            raise ValueError('nine-source query scene order differs')
        self.array = np.load(root / role / 'features.npy', mmap_mode='r', allow_pickle=False)
        if self.array.shape != (len(rows), 9, 9, 160, 160) or self.array.dtype != np.float32:
            raise ValueError('nine-source historical feature shape or dtype differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'artifacts/g246_8h/historical_tir_recent_pair_raw_v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/g246_8h/historical_nine_v4')
    args = parser.parse_args()
    raw_root, out = args.raw_root.resolve(), args.output.resolve()
    if (out / 'manifest.json').exists() and read(out / 'manifest.json').get('status') == 'complete':
        raise FileExistsError('completed nine-source cache is immutable')
    started = time.monotonic()
    raw_path = raw_root / 'manifest.json'
    raw = read(raw_path)
    if raw.get('schema') != 'g246-8h-historical-recent-pair-source-v1' or \
            raw.get('status') != 'source_plan_complete' or raw.get('ready_for_feature_build') is not True:
        raise ValueError('additional recent pair must be complete before feature construction')
    if any(raw.get(k) is not False for k in ('current_query_target_arrays_opened',
            'current_query_target_masks_opened', 'current_query_qa_arrays_opened', 'locked_test_opened')):
        raise ValueError('recent pair crossed the query-information boundary')
    binding_names = ('selection', 'campaign_denylist', 'campaign_denylist_audit',
                     'original_recent_raw_manifest', 'original_recent_raw_selection',
                     'top3_plan', 'top3_count_receipt', 'full_responses_manifest')
    bindings = {}
    for name in binding_names:
        path = Path(raw[name + '_path'])
        if sha(path) != raw[name + '_sha256']:
            raise ValueError(f'recent pair {name} binding differs')
        bindings[name] = path
    deny = read(bindings['campaign_denylist'])
    deny_audit = read(bindings['campaign_denylist_audit'])
    denied = set(deny['acquisition_digests'])
    if deny.get('status') != 'complete' or deny.get('source_scene_count') != 738 or \
            deny.get('acquisition_count') != 633 or len(denied) != 633 or \
            deny.get('namespace') != 'g246-acquisition-alias-v1' or \
            deny_audit.get('pass') is not True or deny_audit.get('status') != 'complete' or \
            deny_audit['denylist_sha256'] != raw['campaign_denylist_sha256']:
        raise ValueError('full campaign source exclusion is not established')
    previous = read(SEVEN / 'manifest.json')
    audit_path = ROOT / 'artifacts/g246_8h_20260905/historical_seven_full_cache_audit.json'
    audit = read(audit_path)
    if previous.get('status') != 'complete' or audit.get('pass') is not True or \
            audit.get('status') != 'complete' or audit['manifest_sha256'] != sha(SEVEN / 'manifest.json'):
        raise ValueError('original seven-source cache needs its exact complete independent audit')
    if previous['base_manifest_sha256'] != sha(BASE / 'manifest.json') or \
            previous['raw_manifest_sha256'] != raw['original_recent_raw_manifest_sha256'] or \
            previous['selection_sha256'] != raw['original_recent_raw_selection_sha256']:
        raise ValueError('the first seven observations or query universe differ')
    old_raw = read(bindings['original_recent_raw_manifest'])
    old_queries = {r['scene_id']: r for r in old_raw['query_records']}
    queries = {r['scene_id']: r for r in raw['query_records']}
    sources = {r['source_key']: r for r in raw['records']}
    if len(queries) != 648 or len(queries) != len(raw['query_records']) or \
            set(queries) != set(old_queries) or len(sources) != len(raw['records']):
        raise ValueError('recent-pair query universe or source uniqueness differs')
    if any(r['status'] not in ('acquisition_complete', 'confirmed_non_tiff_missing') for r in sources.values()):
        raise ValueError('unresolved recent source cannot become a missing modality')
    out.mkdir(parents=True, exist_ok=True)
    code_dir = out / 'source'
    code_dir.mkdir(exist_ok=True)
    code_hashes = {}
    for path in (Path(__file__), ROOT / 'code/build_g246_8h_historical_seven_cache.py',
                 ROOT / 'code/build_g246_8h_historical_cache.py',
                 ROOT / 'code/g246_8h_historical_features.py', ROOT / 'code/g246_8h_recent_historical_features.py',
                 ROOT / 'code/acquire_g246_8h_historical_tir_pilot.py'):
        shutil.copy2(path, code_dir / path.name)
        code_hashes[path.name] = sha(path)
    shutil.copy2(audit_path, code_dir / audit_path.name)
    manifest = {'schema': SCHEMA, 'status': 'building', 'source_count': 9,
        'base_manifest_sha256': sha(BASE / 'manifest.json'),
        'base_historical_manifest_path': str(SEVEN / 'manifest.json'),
        'base_historical_manifest_sha256': sha(SEVEN / 'manifest.json'),
        'base_historical_audit_path': str(code_dir / audit_path.name),
        'base_historical_audit_sha256': sha(audit_path),
        'raw_manifest_path': str(raw_path), 'raw_manifest_sha256': sha(raw_path),
        **{name + suffix: raw[name + suffix] for name in binding_names for suffix in ('_path', '_sha256')},
        'contract': {**copy.deepcopy(previous['contract']), 'source_count': 9,
            'slots': previous['contract']['slots'] + ['recent_distinct_overpass_rank2', 'recent_distinct_overpass_rank3'],
            'recent_encoding': RECENT_CONTRACT,
            'recent_rule': 'original rank1 unchanged; full633 aliases excluded before UTC age8..64/cloud<=40/L8or9/L2SP/T1/wholeAOI/six-asset screening; age/cloud/datetime/id ranking; select three distinct platform/path/UTC-date overpasses',
            'missing_policy': 'no candidate, confirmed unavailable raster, or historical QA empty: entire additional nine fields zero; preserve all queries and all original seven slots'},
        'roles': {}, 'source_code_sha256': code_hashes,
        'target_arrays_opened': False, 'target_masks_opened': False, 'current_query_qa_arrays_opened': False,
        'locked_test_opened': False, 'new_supervision_created': False,
        'training_supervision': 'unchanged Fit603', 'query_scene_count': 648}
    atomic_json(out / 'manifest.json', manifest)
    used = set()
    counts = {'new_source': 0, 'no_candidate': 0}
    for role, expected_n in (('fit', 603), ('validation', 45)):
        base_path = BASE / role / 'metadata.json'
        base_rows = read(base_path)['scenes']
        old_detail = previous['roles'][role]
        for key, filename in (('features_sha256', 'features.npy'), ('metadata_sha256', 'metadata.json')):
            if sha(SEVEN / role / filename) != old_detail[key]:
                raise ValueError('audited original seven-source bytes changed')
        old_rows = read(SEVEN / role / 'metadata.json')['scenes']
        if len(base_rows) != expected_n or [r['scene_id'] for r in old_rows] != [r['scene_id'] for r in base_rows]:
            raise ValueError('seven-source and query membership differ')
        old_array = np.load(SEVEN / role / 'features.npy', mmap_mode='r', allow_pickle=False)
        directory = out / role
        directory.mkdir(exist_ok=True)
        array = np.lib.format.open_memmap(directory / 'features.npy', mode='w+', dtype='float32',
                                         shape=(expected_n, 9, 9, 160, 160))
        rows = []
        for index, (base, original) in enumerate(zip(base_rows, old_rows)):
            q = queries[base['scene_id']]
            old_q = old_queries[base['scene_id']]
            for key in ('scene_id', 'city', 'region', 'role', 'query_datetime', 'query_source_sha256',
                        'canonical_crs', 'canonical_transform30', 'canonical_shape', 'grid_signature_sha256'):
                if q[key] != old_q[key]:
                    raise ValueError(f'recent-pair query identity/grid differs: {key}')
            if q['query_datetime'] != base['datetime'] or q['query_source_sha256'] != base['source_sha256']:
                raise ValueError('recent-pair query does not match base')
            slots = q['slots']
            if len(slots) != 2 or [s['rank'] for s in slots] != [2, 3]:
                raise ValueError('exact ordered recent ranks two and three are required')
            first = original['sources'][6]
            seen = set()
            if first['item_id'] is not None:
                seen.add(overpass(first['item_id'], first['historical_datetime']))
            elif any(s['status'] != 'no_candidate' for s in slots):
                raise ValueError('additional source exists despite no first-ranked candidate')
            array[index] = 0
            array[index, :7] = old_array[index]
            assert np.array_equal(array[index, :7], old_array[index])
            row = copy.deepcopy(original)
            for offset, slot in enumerate(slots, start=7):
                kind = slot['status']
                if kind not in counts:
                    raise ValueError('unknown additional slot status')
                counts[kind] += 1
                entry = {'slot_index': offset, 'slot_kind': kind, 'rank': slot['rank'],
                    'item_id': slot['selected_item_id'], 'historical_datetime': slot['selected_datetime'],
                    'query_datetime': base['datetime'], 'source_key': slot['source_key'], 'age_days': slot.get('age_days')}
                if kind == 'no_candidate':
                    if slot['selected_item_id'] is not None or slot['source_key'] is not None:
                        raise ValueError('no-candidate slot contains a source')
                    entry.update(source_available=False, slot_supplied=False, all_thermal_missing=True,
                                 raw_sha256=None, raw_file=None, missing_reason='no_candidate')
                else:
                    key = slot['source_key']
                    r = sources[key]
                    if r['city'] != q['city'] or r['item_id'] != slot['selected_item_id'] or \
                            r['acquired_utc'] != slot['selected_datetime'] or base['scene_id'] not in r['query_scene_ids']:
                        raise ValueError('raw source is not the frozen selected query observation')
                    age = (parse_time(base['datetime']) - parse_time(r['acquired_utc'])).total_seconds() / 86400
                    pass_id = overpass(r['item_id'], r['acquired_utc'])
                    if not 8 <= age <= 64 or abs(age - slot['age_days']) > 1e-9 or \
                            alias_digest(r['item_id']) in denied or pass_id in seen or \
                            slot['overpass_key'] != '|'.join(pass_id):
                        raise ValueError('source violates chronology, campaign exclusion, or distinct overpass')
                    seen.add(pass_id)
                    anchor = queries[r['query_scene_id']]
                    anchor_city = {**anchor, 'query_scene_id': anchor['scene_id'],
                                   'query_scene_sha256': anchor['query_source_sha256']}
                    raw_arrays = validate_source(r, anchor_city, raw_root)
                    if raw_arrays is None:
                        stats = {'all_thermal_missing': True, 'historical_datetime': r['acquired_utc'],
                                 'historical_clear_fraction30': 0., 'historical_nonempty_fraction120': 0.,
                                 'historical_clear_count30': 0}
                    else:
                        array[index, offset], stats = encode(raw_arrays, r['acquired_utc'], base['datetime'])
                    available = r['status'] == 'acquisition_complete'
                    entry.update(**stats)
                    entry.update(historical_year=r['historical_year'], year=r['historical_year'],
                        source_available=available, slot_supplied=available, raw_sha256=r['raw_sha256'],
                        raw_file=r['raw_file'], product_id=r.get('product_id'), source_status=r['status'],
                        missing_reason=None if available else r['status'], overpass_identity=list(pass_id),
                        campaign_acquisition_exclusion_pass=True, acquisition_digest=alias_digest(r['item_id']))
                    used.add(key)
                entry['slot_active'] = bool(np.any(array[index, offset, 2] > 0))
                entry['slot_all_zero'] = bool(not np.any(array[index, offset]))
                row['sources'].append(entry)
            rows.append(row)
            if (index + 1) % 60 == 0:
                print(json.dumps({'event': 'nine_source_encode', 'role': role, 'query_count': index + 1,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        array.flush()
        del array
        atomic_json(directory / 'metadata.json', {'schema': SCHEMA, 'role': role, 'scenes': rows})
        manifest['roles'][role] = {'scene_count': expected_n, 'base_metadata_sha256': sha(base_path),
            'base_historical_metadata_sha256': old_detail['metadata_sha256'],
            'base_historical_features_sha256': old_detail['features_sha256'],
            'features_shape': [expected_n, 9, 9, 160, 160], 'features_dtype': 'float32',
            'features_bytes': (directory / 'features.npy').stat().st_size,
            'features_sha256': sha(directory / 'features.npy'), 'metadata_sha256': sha(directory / 'metadata.json'),
            'additional_slot_active_counts': [sum(r['sources'][i]['slot_active'] for r in rows) for i in (7, 8)]}
        atomic_json(out / 'manifest.json', manifest)
    if sum(counts.values()) != 1296 or used != set(sources) or sha(raw_path) != manifest['raw_manifest_sha256']:
        raise ValueError('recent-pair universe or raw manifest changed during encoding')
    manifest.update(status='complete', supplementary_slot_counts=counts,
                    base_seven_slots_bitwise_identical=True, elapsed_seconds=time.monotonic() - started)
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps({'event': 'nine_source_complete', 'manifest_sha256': sha(out / 'manifest.json'),
                      'roles': manifest['roles'], 'elapsed_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
