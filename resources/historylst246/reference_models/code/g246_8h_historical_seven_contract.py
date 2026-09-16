"""Explicit seven-slot predictor contract, without relaxing three/six contracts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from g246_8h_historical_contract import _path, _json, _sha, _time, _guard
from g246_8h_historical_six_contract import prepare as prepare_six, check_role as check_six_role, GRID_KEYS
from g246_8h_recent_source_contract import check_raw, acquisition_digest

SCHEMA = 'g246-8h-historical-seven-cache-v1'
FAMILY = 'historical_recent_innovation_emissivity_r6a_seven'
FAMILIES = (FAMILY, 'historical_recent_refinement_emissivity_r6a_seven')


def prepare(root, expected_sha256, base_manifest_sha256, scene_ids, base_metadata_sha256, base_rows):
    root = _path(root)
    m = _json(root / 'manifest.json', expected_sha256)
    _guard(m.get('schema') == SCHEMA and m.get('status') == 'complete' and m.get('source_count') == 7
           and m.get('query_scene_count') == 648 and m.get('base_manifest_sha256') == base_manifest_sha256
           and set(m.get('roles', {})) == {'fit', 'validation'}, 'seven-source cache is incomplete or has a different population/schema')
    _guard(all(m.get(k) is False for k in ('target_arrays_opened', 'target_masks_opened',
           'current_query_qa_arrays_opened', 'locked_test_opened', 'new_supervision_created')),
           'seven-source cache lacks its predictor-only declarations')
    old_path = _path(m['base_historical_manifest_path'])
    old_state = prepare_six(old_path.parent, m['base_historical_manifest_sha256'], base_manifest_sha256,
                            scene_ids, base_metadata_sha256, base_rows)
    old_report = check_six_role(old_state, 'validation', base_rows, base_metadata_sha256)
    old = old_state['manifest']
    audit = _json(_path(m['base_historical_audit_path']), m['base_historical_audit_sha256'])
    _guard(audit.get('pass') is True and audit.get('status') == 'complete'
           and audit['manifest_sha256'] == m['base_historical_manifest_sha256']
           and audit.get('all648_query_scenes_present') is True
           and audit.get('all1944_v1_slots_bitwise_unchanged') is True,
           'seven-source cache does not preserve its audited six-source input')
    contract = m.get('contract', {})
    encoding = contract.get('encoding', {})
    recent = encoding.get('recent', {})
    original = old['contract']['encoding']
    _guard(encoding.get('base_six') == original and contract.get('source_count') == 7
           and contract.get('slots') == old['contract']['slots'] + ['recent_pre_query_8_to_64_days']
           and m.get('base_six_slots_bitwise_identical') is True
           and m.get('supplementary_slot_counts') == {'new_source': 627, 'no_candidate': 21}, 'seven-slot structure or base physics changed')
    _guard(recent.get('schema') == 'g246-8h-recent-historical-feature-encoding-v1'
           and recent.get('query_years') == [2021, 2022, 2023, 2024, 2025]
           and recent.get('source_age_days_inclusive') == [8, 64] and 'source_years' not in recent
           and all(recent.get(k) == v for k, v in original.items() if k not in ('schema', 'source_years', 'interpretation')),
           'recent temporal contract or frozen spatial/radiometric operations differ')
    hashes = m['source_code_sha256']
    _guard(hashes.get('g246_8h_historical_features.py') == old['source_code_sha256']['g246_8h_historical_features.py']
           and 'g246_8h_recent_historical_features.py' in hashes, 'seven-source encoder provenance does not separate old and recent contracts')
    for name, digest in hashes.items():
        _guard(Path(name).name == name and _sha(root / 'source' / name) == digest, 'seven-source implementation snapshot changed')
    sources = check_raw(m['raw_manifest_path'], m['raw_manifest_sha256'])
    for name in ('inventory_manifest', 'selection', 'campaign_denylist', 'campaign_denylist_audit'):
        _guard(m[name + '_sha256'] == sources['raw'][name + '_sha256']
               and _sha(_path(m[name + '_path'])) == m[name + '_sha256'], 'seven-source raw/selection/exclusion binding differs')
    return {'root': root, 'manifest': m, 'old_root': old_path.parent, 'old': old,
            'old_state': old_state, 'old_report': old_report, 'sources': sources}


def check_role(state, role, base_rows, base_metadata_sha256):
    root, m, old = state['root'], state['manifest'], state['old']
    count = {'fit': 603, 'validation': 45}[role]
    detail, previous = m['roles'][role], old['roles'][role]
    _guard(len(base_rows) == count and detail['scene_count'] == count
           and detail['base_metadata_sha256'] == base_metadata_sha256
           and detail['base_historical_features_sha256'] == previous['features_sha256']
           and detail['base_historical_metadata_sha256'] == previous['metadata_sha256'], 'seven-source role/base binding differs')
    metadata = _json(root / role / 'metadata.json', detail['metadata_sha256'])
    original = _json(state['old_root'] / role / 'metadata.json', previous['metadata_sha256'])['scenes']
    rows = metadata['scenes']
    _guard(metadata.get('schema') == SCHEMA and metadata.get('role') == role
           and [r['scene_id'] for r in rows] == [r['scene_id'] for r in original] == [r['scene_id'] for r in base_rows],
           'seven-source metadata role or scene order differs')
    path, old_path = root / role / 'features.npy', state['old_root'] / role / 'features.npy'
    _guard(_sha(path) == detail['features_sha256'] and _sha(old_path) == previous['features_sha256'],
           'seven-source or original six feature bytes changed')
    array, previous_array = (np.load(p, mmap_mode='r', allow_pickle=False) for p in (path, old_path))
    _guard(array.shape == (count, 7, 9, 160, 160) and array.dtype == np.float32
           and list(array.shape) == detail['features_shape'] and detail['features_dtype'] == 'float32'
           and path.stat().st_size == detail['features_bytes'], 'seven-source array shape/type/bytes differs')
    active = missing = cloud = 0
    for i, (row, base, old_row) in enumerate(zip(rows, base_rows, original)):
        q = state['sources']['queries'][base['scene_id']]
        _guard(np.isfinite(array[i]).all() and np.array_equal(array[i, :6], previous_array[i]),
               'seven-source base six slots changed or recent fields are nonfinite')
        _guard(all(row[k] == old_row[k] for k in old_row if k != 'sources')
               and row['query_datetime'] == base['datetime'] and row['query_source_sha256'] == base['source_sha256']
               and all(row[k] == q[k] for k in GRID_KEYS) and len(row['sources']) == 7
               and row['sources'][:6] == old_row['sources'], 'seven-source query or base six metadata differs')
        source, slot, value = row['sources'][6], q['slots'][0], array[i, 6]
        nonempty, zero = bool(np.any(value[2] > 0)), not bool(np.any(value))
        _guard(source['slot_index'] == 6 and source['slot_kind'] == slot['status']
               and source['source_key'] == slot['source_key'] and source['item_id'] == slot['selected_item_id']
               and source['historical_datetime'] == slot['selected_datetime'] and source['query_datetime'] == base['datetime']
               and source['age_days'] == slot['age_days'] and source['slot_active'] is nonempty and source['slot_all_zero'] is zero,
               'recent feature slot differs from its actual query/source selection')
        active += nonempty
        if slot['status'] == 'no_candidate':
            _guard(zero and source['source_available'] is False and source['slot_supplied'] is False
                   and source['all_thermal_missing'] is True and source['raw_file'] is None
                   and source['raw_sha256'] is None and source['missing_reason'] == 'no_candidate',
                   'no-candidate recent date must have exactly nine zero fields')
            missing += 1
            continue
        r = state['sources']['raw_records'][slot['source_key']]
        available = r['status'] == 'acquisition_complete'
        _guard(source['source_available'] is available and source['slot_supplied'] is available
               and source['raw_sha256'] == r['raw_sha256'] and source['raw_file'] == r['raw_file']
               and source.get('product_id') == r.get('product_id') and source['source_status'] == r['status']
               and source['year'] == source['historical_year'] == r['historical_year']
               and source['campaign_acquisition_exclusion_pass'] is True
               and source['acquisition_digest'] == acquisition_digest(r['item_id'])
               and source['acquisition_digest'] not in state['sources']['denied'], 'recent feature/raw identity, availability or campaign exclusion differs')
        if not available:
            _guard(zero and source['all_thermal_missing'] is True and source['missing_reason'] == r['status'],
                   'confirmed unavailable recent date is not entirely zero')
            missing += 1
            continue
        _guard(source['missing_reason'] is None, 'available recent source was marked missing')
        if source['all_thermal_missing'] is True:
            _guard(zero, 'all-cloud recent source must have nine zero fields')
            cloud += 1
            continue
        _guard(source['all_thermal_missing'] is False and nonempty, 'recent date has no clear coverage')
        for coverage in (value[2], value[5]):
            _guard(np.all((0 <= coverage) & (coverage <= 1)) and np.all(coverage * 16 == np.rint(coverage * 16)),
                   'recent coverage must count delivered30 cells')
        _guard(np.all((-3.00001 <= value[0]) & (value[0] <= 4.00001))
               and np.all((0 < value[3]) & (value[3] <= 1.00001)), 'recent temperature/uncertainty outside physical contract')
        acquired, query = _time(r['acquired_utc']), _time(base['datetime'])
        age = (query - acquired).total_seconds() / 86400
        phase = 2 * np.pi * acquired.timetuple().tm_yday / 365.25
        _guard(8 <= age <= 64 and age == slot['age_days']
               and all(np.all(value[6 + j] == np.float32(x)) for j, x in enumerate((np.sin(phase), np.cos(phase), age / 3652.5))),
               'recent DOY/age does not use its real actual-query-relative time')
    _guard(active == detail['additional_slot_active_count'], 'recent active date count differs')
    return {'scene_count': count, 'shape': list(array.shape), 'dtype': 'float32', 'finite': True,
            'features_sha256': detail['features_sha256'], 'metadata_sha256': detail['metadata_sha256'],
            'base_six_slots_bitwise_identical': True, 'additional_active_slots': active,
            'additional_source_missing_slots': missing, 'additional_all_cloud_slots': cloud,
            'actual_query_age_and_full_campaign_exclusion_pass': True}


def check_cache(root, expected_sha256, base_manifest_sha256, expected_scene_ids, base_metadata_sha256, expected_records):
    state = prepare(root, expected_sha256, base_manifest_sha256, expected_scene_ids, base_metadata_sha256, expected_records)
    role = check_role(state, 'validation', expected_records, base_metadata_sha256)
    return {'root': str(state['root']), 'manifest_sha256': expected_sha256,
            'validation_features_sha256': role['features_sha256'], 'validation_metadata_sha256': role['metadata_sha256'],
            'raw_manifest_sha256': state['manifest']['raw_manifest_sha256'],
            'campaign_denylist_sha256': state['manifest']['campaign_denylist_sha256'],
            'selection_sha256': state['manifest']['selection_sha256'],
            'base_historical_manifest_sha256': state['manifest']['base_historical_manifest_sha256'],
            'base_historical_audit_sha256': state['manifest']['base_historical_audit_sha256'],
            **role, 'historical_date_slots': 315, 'all648_recent_selections_recomputed': True,
            'time_contract': '2026 retrospective replay; historical public availability unknown',
            'query_target_arrays_opened': False, 'query_masks_opened': False, 'locked_test_opened': False}
