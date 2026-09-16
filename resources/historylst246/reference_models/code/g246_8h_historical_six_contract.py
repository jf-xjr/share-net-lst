"""Strict, offline contract for the distinct six-source historical cache.

The original three-source contract remains unchanged. Selection is recomputed
from frozen catalogue metadata, never from query labels or historical pixels.
This module reads only local manifests and encoded predictor arrays.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

from g246_8h_historical_contract import (
    _path, _sha, _json, _time, _identity, _guard, check_cache as check_three_cache,
)

SCHEMA = 'g246-8h-historical-six-cache-v1'
RAW_SCHEMA = 'g246-8h-historical-seasonal-source-v2'
YEARS = (2018, 2019, 2020)
SLOTS = ['v1_2018', 'v1_2019', 'v1_2020',
         'query_season_2018', 'query_season_2019', 'query_season_2020']
COUNTS = {'duplicate_v1': 388, 'no_candidate': 6, 'new_source': 1550}
GRID_KEYS = ('canonical_crs', 'canonical_transform30', 'canonical_shape', 'grid_signature_sha256')
FORBIDDEN = ('current_query_target_arrays_opened', 'current_query_target_masks_opened',
             'current_query_qa_arrays_opened', 'locked_test_opened', 'new_supervision_created')


def distance(a, b):
    """Frozen non-leap month/day distance, independently recomputed."""
    a, b = _time(a), _time(b)
    gap = abs((date(2023, a.month, a.day) - date(2023, b.month, b.day)).days)
    return min(gap, 365 - gap)


def check_sources(raw_path, expected_sha256, original_manifest):
    """Validate every source reference without opening any raster or label."""
    raw_path = _path(raw_path)
    root = raw_path.parent
    raw = _json(raw_path, expected_sha256)
    _guard(raw.get('schema') == RAW_SCHEMA and raw.get('status') == 'source_plan_complete'
           and raw.get('ready_for_feature_build') is True,
           'seasonal historical source acquisition is unresolved')
    _guard(all(raw.get(k) is False for k in FORBIDDEN), 'seasonal source opened query label/QA fields')
    selection = _json(root / 'selection.json', raw['selection_sha256'])
    _guard(selection.get('schema') == RAW_SCHEMA and selection.get('status') == 'selection_frozen'
           and all(selection.get(k) is False for k in FORBIDDEN), 'seasonal selection is not frozen predictor-only metadata')
    _guard(selection['query_records'] == raw['query_records']
           and selection['source_plans'] == raw['source_plans'], 'seasonal source plan changed after selection')
    _guard(raw['v1_source_manifest_sha256'] == original_manifest['raw_manifest_sha256'],
           'seasonal deduplication refers to a different V1 source universe')
    _json(_path(raw['v1_source_manifest_path']), raw['v1_source_manifest_sha256'])
    inventory = _json(_path(raw['inventory_manifest_path']), raw['inventory_manifest_sha256'])
    _guard(raw['inventory_manifest_sha256'] == original_manifest['inventory_sha256'],
           'seasonal selection uses a different eligible catalogue pool')
    for key in ('inventory_manifest_sha256', 'v1_source_manifest_sha256', 'temporal_alignment_sha256'):
        _guard(selection[key] == raw[key], 'seasonal selection provenance binding differs')
    alignment = _json(_path(raw['temporal_alignment_path']), raw['temporal_alignment_sha256'])
    _guard(raw['temporal_alignment_sha256'] == inventory['temporal_alignment_sha256']
           and alignment.get('target_arrays_read') is False and alignment.get('locked_test_opened') is False,
           'seasonal query grid proof differs')
    city_grids = {r['city']: r for r in inventory['cities']}
    public = {r['scene_id']: r for r in inventory['public_query_scenes']}
    pools = {(r['city'], r['historical_year']): r for r in inventory['records']}
    queries = {r['scene_id']: r for r in raw['query_records']}
    plans = {r['source_key']: r for r in selection['source_plans']}
    records = {r['source_key']: r for r in raw['records']}
    _guard(len(city_grids) == 216 and len(public) == len(queries) == len(raw['query_records']) == 648
           and set(public) == set(queries) and len(pools) == 648
           and len(plans) == len(selection['source_plans']) == len(records) == len(raw['records']) == 1186
           and set(plans) == set(records), 'seasonal source/query population differs')
    aliases = {_identity(r['item_id']) for r in public.values()}
    refs, counts = defaultdict(list), Counter()
    for sid, q in queries.items():
        base, city = public[sid], city_grids[q['city']]
        _guard(q['city'] == base['city'] and q['region'] == base['region'] and q['role'] == base['role']
               and q['query_datetime'] == base['datetime'] and q['query_source_sha256'] == base['source_sha256']
               and 2021 <= _time(q['query_datetime']).year <= 2025
               and all(q[k] == city[k] for k in GRID_KEYS) and q['canonical_shape'] == [640, 640]
               and sid in city['all_query_scene_ids'], 'seasonal query identity, timestamp or grid differs')
        binding = next((x for x in alignment['roles'][q['role']]['cities'][q['city']] if x['scene_id'] == sid), None)
        _guard(binding is not None and binding['declared_source_sha256'] == q['query_source_sha256']
               and binding['grid_signature_sha256'] == q['grid_signature_sha256'], 'seasonal query has no metadata-only grid proof')
        _guard(len(q['slots']) == 3 and [s['historical_year'] for s in q['slots']] == list(YEARS),
               'seasonal query year slots differ')
        for year, slot in zip(YEARS, q['slots']):
            pool = pools[(q['city'], year)]
            original_item = pool['selected_item']
            original_id = original_item['id'] if original_item else None
            candidates = [r for r in pool['ranked_candidates'] if r['eo:cloud_cover'] <= 40]
            candidates.sort(key=lambda r: (distance(q['query_datetime'], r['datetime']),
                r['eo:cloud_cover'], r['datetime'], r['item_id']))
            picked = candidates[0] if candidates else None
            duplicate = bool(picked and original_id and _identity(picked['item_id']) == _identity(original_id))
            kind = 'no_candidate' if picked is None else 'duplicate_v1' if duplicate else 'new_source'
            key = f"{q['city']}|{year}|{picked['item_id']}" if kind == 'new_source' else None
            _guard(slot['status'] == kind and slot['source_key'] == key and slot['v1_item_id'] == original_id
                   and slot['selected_item_id'] == (picked['item_id'] if picked else None)
                   and slot['selected_datetime'] == (picked['datetime'] if picked else None)
                   and slot['selected_catalogue_cloud_percent'] == (picked['eo:cloud_cover'] if picked else None)
                   and slot['seasonal_distance_days'] == (distance(q['query_datetime'], picked['datetime']) if picked else None),
                   'seasonal source was reranked or did not follow frozen closest-DOY rule')
            counts[kind] += 1
            if picked:
                acquired = _time(picked['datetime'])
                _guard(acquired.year == year and 5 <= acquired.month <= 9 and acquired < _time(q['query_datetime'])
                       and _identity(picked['item_id']) not in aliases, 'seasonal observation crosses pre2021 acquisition boundary')
            if key:
                refs[key].append(sid)
    _guard(dict(counts) == COUNTS and set(refs) == set(records), 'seasonal deduplication/reference counts differ')
    metadata = _json(_path(raw['metadata_manifest_path']), raw['metadata_manifest_sha256'])
    _guard(metadata.get('status') == 'metadata_complete' and not metadata.get('errors')
           and metadata['selection_sha256'] == raw['selection_sha256'], 'exact selected STAC metadata unresolved')
    item_entries = {r['item_id']: r for r in metadata['items']}
    _guard(len(item_entries) == len(metadata['items']), 'duplicate item receipt identities')
    for key, r in records.items():
        p, city = plans[key], city_grids[r['city']]
        _guard(r.get('schema') == RAW_SCHEMA and r.get('status') in ('acquisition_complete', 'confirmed_non_tiff_missing')
               and all(r.get(k) is False for k in FORBIDDEN)
               and r['selection_sha256'] == raw['selection_sha256'], 'seasonal raw source is unresolved or used query labels')
        _guard(all(r[k] == p[k] for k in ('city', 'region', 'role', 'historical_year', 'item_id', 'acquired_utc',
               'catalogue_cloud_percent', 'query_scene_ids', 'grid_query_scene_id', 'grid_query_source_sha256', *GRID_KEYS))
               and sorted(r['query_scene_ids']) == sorted(refs[key])
               and len(r['query_scene_ids']) == len(set(r['query_scene_ids'])), 'seasonal source key or actual-query membership differs')
        _guard(r['query_scene_id'] == city['query_scene_id'] and r['query_scene_sha256'] == city['query_scene_sha256']
               and r['query_view_role'] == city['role'] and r['query_datetime'] == public[city['query_scene_id']]['datetime']
               and all(r[k] == city[k] for k in GRID_KEYS) and r['resampling'] == 'nearest',
               'seasonal raw source representative query/grid differs')
        item_entry = item_entries[r['item_id']]
        _guard(item_entry['receipt_file'] == r['item_receipt_file']
               and item_entry['receipt_sha256'] == r['item_receipt_sha256'], 'seasonal STAC receipt binding differs')
        receipt = _json(_path(r['item_receipt_file'], root), r['item_receipt_sha256'])
        item = receipt['selected_item']
        item_sha = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        _guard(receipt['selection_sha256'] == raw['selection_sha256'] and receipt['status'] == 'metadata_complete'
               and item_sha == r['item_metadata_sha256'] == receipt['item_metadata_sha256'] == item_entry['item_metadata_sha256']
               and item['id'] == r['item_id'] and item['properties']['datetime'] == r['acquired_utc']
               and item['properties']['eo:cloud_cover'] == r['catalogue_cloud_percent'] <= 40
               and item['properties'].get('platform') == 'landsat-8'
               and item['properties'].get('landsat:correction') == 'L2SP'
               and item['properties'].get('landsat:collection_category') == 'T1', 'seasonal source does not match its exact frozen STAC product')
        if r['status'] == 'acquisition_complete':
            _guard(not r.get('failures') and r.get('raw_file') and r.get('raw_sha256')
                   and r['raw_bytes'] > 0, 'completed seasonal source lacks its raw byte receipt')
        else:
            _guard(r.get('failures') and all(f.get('source_confirmation', {}).get('confirmed') is True
                   for f in r['failures']), 'seasonal missing source lacks repeated non-TIFF confirmation')
    return {'raw': raw, 'selection': selection, 'inventory': inventory, 'queries': queries, 'records': records,
            'raw_root': root, 'counts': dict(counts)}


def prepare(root, expected_sha256, base_manifest_sha256, scene_ids, base_metadata_sha256, base_rows):
    root = _path(root)
    manifest = _json(root / 'manifest.json', expected_sha256)
    _guard(manifest.get('schema') == SCHEMA and manifest.get('status') == 'complete'
           and manifest.get('source_count') == 6 and manifest.get('query_scene_count') == 648
           and manifest.get('base_manifest_sha256') == base_manifest_sha256
           and set(manifest.get('roles', {})) == {'fit', 'validation'}, 'six-source cache schema, status or population differs')
    _guard(all(manifest.get(k) is False for k in ('target_arrays_opened', 'target_masks_opened',
           'current_query_qa_arrays_opened', 'locked_test_opened', 'new_supervision_created')),
           'six-source cache lacks predictor-only declarations')
    old_path = _path(manifest['base_historical_manifest_path'])
    old = _json(old_path, manifest['base_historical_manifest_sha256'])
    old_report = check_three_cache(old_path.parent, manifest['base_historical_manifest_sha256'], base_manifest_sha256,
                                   scene_ids, base_metadata_sha256, base_rows)
    audit = _json(_path(manifest['base_historical_audit_path']), manifest['base_historical_audit_sha256'])
    _guard(audit.get('pass') is True and audit.get('status') == 'complete'
           and audit['manifest_sha256'] == manifest['base_historical_manifest_sha256']
           and audit.get('all648_query_scenes_present') is True
           and audit.get('query_target_arrays_opened') is False and audit.get('locked_test_opened') is False,
           'original three-source full audit binding differs')
    contract = manifest.get('contract', {})
    _guard(contract.get('encoding') == old['contract'] and contract.get('source_count') == 6
           and contract.get('slots') == SLOTS and manifest.get('supplementary_slot_counts') == COUNTS
           and manifest.get('base_three_slots_bitwise_identical') is True, 'six-source encoding or slot semantics differ')
    hashes = manifest['source_code_sha256']
    _guard(hashes.get('g246_8h_historical_features.py') == old['source_code_sha256']['g246_8h_historical_features.py'],
           'six-source cache changed the audited pure physical encoding')
    for name, digest in hashes.items():
        _guard(Path(name).name == name and _sha(root / 'source' / name) == digest, 'six-source source snapshot differs')
    sources = check_sources(manifest['raw_manifest_path'], manifest['raw_manifest_sha256'], old)
    _guard(manifest['selection_sha256'] == sources['raw']['selection_sha256']
           and _sha(_path(manifest['selection_path'])) == manifest['selection_sha256'], 'six-source selection hash differs')
    return {'root': root, 'manifest': manifest, 'old_root': old_path.parent, 'old': old,
            'old_report': old_report, 'sources': sources}


def check_role(state, role, base_rows, base_metadata_sha256):
    root, manifest, old = state['root'], state['manifest'], state['old']
    count = {'fit': 603, 'validation': 45}[role]
    detail, previous = manifest['roles'][role], old['roles'][role]
    _guard(len(base_rows) == count and detail['scene_count'] == count
           and detail['base_metadata_sha256'] == base_metadata_sha256
           and detail['base_historical_metadata_sha256'] == previous['metadata_sha256']
           and detail['base_historical_features_sha256'] == previous['features_sha256'], 'six-source role/base bindings differ')
    metadata = _json(root / role / 'metadata.json', detail['metadata_sha256'])
    original = _json(state['old_root'] / role / 'metadata.json', previous['metadata_sha256'])['scenes']
    rows = metadata['scenes']
    _guard(metadata.get('schema') == SCHEMA and metadata.get('role') == role
           and [r['scene_id'] for r in rows] == [r['scene_id'] for r in original] == [r['scene_id'] for r in base_rows],
           'six-source cache scene order or role differs')
    path = root / role / 'features.npy'
    old_path = state['old_root'] / role / 'features.npy'
    _guard(_sha(path) == detail['features_sha256'] and _sha(old_path) == previous['features_sha256'],
           'six-source or frozen original feature bytes differ')
    array, old_array = (np.load(p, mmap_mode='r', allow_pickle=False) for p in (path, old_path))
    _guard(array.shape == (count, 6, 9, 160, 160) and array.dtype == np.float32
           and list(array.shape) == detail['features_shape'] and detail['features_dtype'] == 'float32'
           and path.stat().st_size == detail['features_bytes'], 'six-source array shape/type/bytes differ')
    active = source_missing = thermal_missing = 0
    for i, (row, base, old_row) in enumerate(zip(rows, base_rows, original)):
        q = state['sources']['queries'][base['scene_id']]
        _guard(np.isfinite(array[i]).all() and np.array_equal(array[i, :3], old_array[i]),
               'six-source first three fields changed or nonfinite input exists')
        _guard(all(row[k] == old_row[k] for k in ('scene_id', 'city', 'region', 'query_datetime', 'query_source_sha256', *GRID_KEYS))
               and row['query_datetime'] == base['datetime'] and row['query_source_sha256'] == base['source_sha256']
               and row['city'] == base['city'] and row['region'] == base['region']
               and all(row[k] == q[k] for k in GRID_KEYS)
               and len(row['sources']) == 6, 'six-source query metadata or date count differs')
        for j, source in enumerate(row['sources']):
            value = array[i, j]
            supplied_active = bool(np.any(value[2] > 0))
            zero = not bool(np.any(value))
            _guard(source['slot_index'] == j and source['slot_active'] is supplied_active
                   and source['slot_all_zero'] is zero and source['query_datetime'] == base['datetime'],
                   'historical slot activity, index or actual-query time differs')
            if j < 3:
                baseline = old_row['sources'][j]
                _guard(all(source[k] == v for k, v in baseline.items()) and source['slot_kind'] == 'v1'
                       and source['source_key'] is None and source['slot_supplied'] == baseline['source_available'],
                       'original three-source metadata changed')
                continue
            slot, baseline = q['slots'][j - 3], old_row['sources'][j - 3]
            kind = slot['status']
            _guard(source['slot_kind'] == kind and source['historical_year'] == source['year'] == YEARS[j - 3]
                   and source['item_id'] == slot['selected_item_id'] and source['historical_datetime'] == slot['selected_datetime']
                   and source['source_key'] == slot['source_key'] and source['v1_item_id'] == slot['v1_item_id'] == baseline['item_id']
                   and source['seasonal_distance_days'] == slot['seasonal_distance_days'], 'additional slot differs from frozen query-specific selection')
            active += supplied_active
            if kind == 'duplicate_v1':
                _guard(zero and source['slot_supplied'] is False
                       and source['source_available'] == baseline['source_available']
                       and source['all_thermal_missing'] == baseline['all_thermal_missing']
                       and source['raw_sha256'] == baseline['raw_sha256'] and source['raw_file'] == baseline['raw_file']
                       and _identity(source['item_id']) == _identity(baseline['item_id'])
                       and source['historical_datetime'] == baseline['historical_datetime']
                       and source['missing_reason'] == 'duplicate_v1_zero_additional_slot',
                       'duplicate date must be zero without falsely marking its physical source missing')
                continue
            if kind == 'no_candidate':
                _guard(zero and source['slot_supplied'] is False and source['source_available'] is False
                       and source['all_thermal_missing'] is True and source['raw_sha256'] is None
                       and source['raw_file'] is None and source['historical_datetime'] is None
                       and source['missing_reason'] == 'no_candidate', 'no-candidate date must have nine zero fields')
                source_missing += 1
                continue
            r = state['sources']['records'][source['source_key']]
            available = r['status'] == 'acquisition_complete'
            _guard(source['source_available'] is available and source['slot_supplied'] is available
                   and source['raw_sha256'] == r['raw_sha256'] and source['raw_file'] == r['raw_file']
                   and source['source_status'] == r['status'] and source.get('product_id') == r.get('product_id')
                   and source['missing_reason'] == (None if available else r['status']), 'additional feature/raw availability binding differs')
            if not available:
                _guard(zero and source['all_thermal_missing'] is True, 'confirmed unavailable date must be entirely zero')
                source_missing += 1
                continue
            if source['all_thermal_missing'] is True:
                _guard(zero, 'all-cloud historical date must have nine zero fields')
                thermal_missing += 1
                continue
            _guard(source['all_thermal_missing'] is False and supplied_active, 'historical date has no actual clear coverage')
            for coverage in (value[2], value[5]):
                _guard(np.all((0 <= coverage) & (coverage <= 1))
                       and np.all(coverage * 16 == np.rint(coverage * 16)), 'historical coverage must count delivered30 cells')
            _guard(np.all((-3.00001 <= value[0]) & (value[0] <= 4.00001))
                   and np.all((0 < value[3]) & (value[3] <= 1.00001)), 'historical temperature/uncertainty outside physical contract')
            acquired, query = _time(r['acquired_utc']), _time(base['datetime'])
            phase = 2 * np.pi * acquired.timetuple().tm_yday / 365.25
            time_fields = (np.sin(phase), np.cos(phase), (query - acquired).total_seconds() / 86400 / 3652.5)
            _guard(all(np.all(value[6 + j] == np.float32(x)) for j, x in enumerate(time_fields)),
                   'supplemental age or DOY fields reused another query time')
    _guard(active == detail['additional_slot_active_count'], 'additional active slot count differs')
    return {'scene_count': count, 'shape': list(array.shape), 'dtype': 'float32', 'finite': True,
            'features_sha256': detail['features_sha256'], 'metadata_sha256': detail['metadata_sha256'],
            'base_three_slots_bitwise_identical': True, 'additional_active_slots': active,
            'additional_source_missing_slots': source_missing, 'additional_all_cloud_slots': thermal_missing,
            'actual_query_age_binding_pass': True}


def check_cache(root, expected_sha256, base_manifest_sha256, expected_scene_ids,
                base_metadata_sha256, expected_records):
    state = prepare(root, expected_sha256, base_manifest_sha256, expected_scene_ids, base_metadata_sha256, expected_records)
    role = check_role(state, 'validation', expected_records, base_metadata_sha256)
    return {'root': str(state['root']), 'manifest_sha256': expected_sha256,
            'validation_features_sha256': role['features_sha256'],
            'validation_metadata_sha256': role['metadata_sha256'],
            'raw_manifest_sha256': state['manifest']['raw_manifest_sha256'],
            'selection_sha256': state['manifest']['selection_sha256'],
            'base_historical_manifest_sha256': state['manifest']['base_historical_manifest_sha256'],
            'base_historical_audit_sha256': state['manifest']['base_historical_audit_sha256'],
            **role, 'historical_date_slots': 270, 'supplementary_slot_counts': state['sources']['counts'],
            'all648_seasonal_selections_recomputed': True, 'source_identity_time_grid_binding_pass': True,
            'time_contract': '2026 retrospective replay; historical public availability unknown',
            'query_target_arrays_opened': False, 'query_masks_opened': False, 'locked_test_opened': False}
