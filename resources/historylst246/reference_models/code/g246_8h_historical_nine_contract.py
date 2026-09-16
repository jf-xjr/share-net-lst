"""Explicit nine-slot input contract; earlier three/six/seven contracts stay strict."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from g246_8h_historical_contract import _path, _json, _sha, _time, _guard
from g246_8h_historical_seven_contract import prepare as prepare_seven, check_role as check_seven_role
from g246_8h_recent_pair_source_contract import overpass_key
from g246_8h_recent_source_contract import GRID_KEYS, acquisition_digest

SCHEMA = 'g246-8h-historical-nine-cache-v1'
FAMILIES = ('historical_recent_innovation_emissivity_r6a_nine',
            'historical_recent_refinement_emissivity_r6a_nine')
BINDINGS = ('selection', 'campaign_denylist', 'campaign_denylist_audit',
            'original_recent_raw_manifest', 'original_recent_raw_selection',
            'top3_plan', 'top3_count_receipt', 'full_responses_manifest')


def prepare(root, expected_sha256, base_manifest_sha256, scene_ids, base_metadata_sha256, base_rows):
    root = _path(root)
    m = _json(root / 'manifest.json', expected_sha256)
    _guard(m.get('schema') == SCHEMA and m.get('status') == 'complete' and m.get('source_count') == 9
           and m.get('query_scene_count') == 648 and m.get('base_manifest_sha256') == base_manifest_sha256
           and set(m.get('roles', {})) == {'fit', 'validation'}, 'nine-source cache is incomplete or has a different population/schema')
    _guard(all(m.get(k) is False for k in ('target_arrays_opened', 'target_masks_opened',
           'current_query_qa_arrays_opened', 'locked_test_opened', 'new_supervision_created')),
           'nine-source cache lacks its predictor-only declarations')
    old_path = _path(m['base_historical_manifest_path'])
    old_state = prepare_seven(old_path.parent, m['base_historical_manifest_sha256'], base_manifest_sha256,
                              scene_ids, base_metadata_sha256, base_rows)
    old_report = check_seven_role(old_state, 'validation', base_rows, base_metadata_sha256)
    old = old_state['manifest']
    audit = _json(_path(m['base_historical_audit_path']), m['base_historical_audit_sha256'])
    _guard(audit.get('pass') is True and audit.get('status') == 'complete'
           and audit['manifest_sha256'] == m['base_historical_manifest_sha256']
           and audit.get('all648_query_scenes_present') is True
           and audit.get('all3888_base_six_slots_bitwise_unchanged') is True,
           'nine-source cache does not preserve its audited seven-source input')
    contract = m.get('contract', {})
    _guard(contract.get('source_count') == 9 and contract.get('encoding') == old['contract']['encoding']
           and contract.get('recent_encoding') == old['contract']['encoding']['recent']
           and contract.get('slots') == old['contract']['slots'] + ['recent_distinct_overpass_rank2', 'recent_distinct_overpass_rank3']
           and m.get('base_seven_slots_bitwise_identical') is True
           and m.get('supplementary_slot_counts') == {'new_source': 1023, 'no_candidate': 273},
           'nine-slot structure or frozen physics changed')
    hashes = m['source_code_sha256']
    for name, digest in hashes.items():
        _guard(Path(name).name == name and _sha(root / 'source' / name) == digest, 'nine-source implementation snapshot changed')
    for name in ('g246_8h_historical_features.py', 'g246_8h_recent_historical_features.py'):
        _guard(hashes[name] == old['source_code_sha256'][name], 'nine-source pure encoder differs from the frozen seven-source encoder')
    from g246_8h_recent_pair_source_contract import check_raw
    sources = check_raw(m['raw_manifest_path'], m['raw_manifest_sha256'])
    for name in BINDINGS:
        _guard(m[name + '_sha256'] == sources['raw'][name + '_sha256']
               and _sha(_path(m[name + '_path'])) == m[name + '_sha256'], 'nine-source raw/plan/exclusion binding differs')
    _guard(m['original_recent_raw_manifest_sha256'] == old['raw_manifest_sha256']
           and m['original_recent_raw_selection_sha256'] == old['selection_sha256'],
           'additional recent pair changes the first recent source universe')
    return {'root': root, 'manifest': m, 'old_root': old_path.parent, 'old': old,
            'old_state': old_state, 'old_report': old_report, 'sources': sources}


def check_role(state, role, base_rows, base_metadata_sha256):
    root, m, old = state['root'], state['manifest'], state['old']
    count = {'fit': 603, 'validation': 45}[role]
    detail, previous = m['roles'][role], old['roles'][role]
    _guard(len(base_rows) == count and detail['scene_count'] == count
           and detail['base_metadata_sha256'] == base_metadata_sha256
           and detail['base_historical_features_sha256'] == previous['features_sha256']
           and detail['base_historical_metadata_sha256'] == previous['metadata_sha256'], 'nine-source role/base binding differs')
    metadata = _json(root / role / 'metadata.json', detail['metadata_sha256'])
    original = _json(state['old_root'] / role / 'metadata.json', previous['metadata_sha256'])['scenes']
    rows = metadata['scenes']
    _guard(metadata.get('schema') == SCHEMA and metadata.get('role') == role
           and [r['scene_id'] for r in rows] == [r['scene_id'] for r in original] == [r['scene_id'] for r in base_rows],
           'nine-source metadata role or scene order differs')
    path, old_path = root / role / 'features.npy', state['old_root'] / role / 'features.npy'
    _guard(_sha(path) == detail['features_sha256'] and _sha(old_path) == previous['features_sha256'],
           'nine-source or original seven feature bytes changed')
    array, previous_array = (np.load(p, mmap_mode='r', allow_pickle=False) for p in (path, old_path))
    _guard(array.shape == (count, 9, 9, 160, 160) and array.dtype == np.float32
           and list(array.shape) == detail['features_shape'] and detail['features_dtype'] == 'float32'
           and path.stat().st_size == detail['features_bytes'], 'nine-source array shape/type/bytes differs')
    active, missing, cloud = [0, 0], [0, 0], [0, 0]
    for i, (row, base, old_row) in enumerate(zip(rows, base_rows, original)):
        q = state['sources']['queries'][base['scene_id']]
        _guard(np.isfinite(array[i]).all() and np.array_equal(array[i, :7], previous_array[i]),
               'nine-source base seven slots changed or recent fields are nonfinite')
        _guard(all(row[k] == old_row[k] for k in old_row if k != 'sources')
               and row['query_datetime'] == base['datetime'] and row['query_source_sha256'] == base['source_sha256']
               and all(row[k] == q[k] for k in GRID_KEYS) and len(row['sources']) == 9
               and row['sources'][:7] == old_row['sources'], 'nine-source query or base seven metadata differs')
        first = row['sources'][6]
        seen = set() if first['item_id'] is None else {overpass_key(first['item_id'], first['historical_datetime'])}
        for index, slot in enumerate(q['slots']):
            source, value = row['sources'][7 + index], array[i, 7 + index]
            nonempty, zero = bool(np.any(value[2] > 0)), not bool(np.any(value))
            _guard(source['slot_index'] == 7 + index and source['rank'] == slot['rank'] == 2 + index
                   and source['slot_kind'] == slot['status'] and source['source_key'] == slot['source_key']
                   and source['item_id'] == slot['selected_item_id'] and source['historical_datetime'] == slot['selected_datetime']
                   and source['query_datetime'] == base['datetime'] and source['age_days'] == slot['age_days']
                   and source['slot_active'] is nonempty and source['slot_all_zero'] is zero,
                   'additional recent slot differs from its actual query/ranked source')
            active[index] += nonempty
            if slot['status'] == 'no_candidate':
                _guard(zero and source['source_available'] is False and source['slot_supplied'] is False
                       and source['all_thermal_missing'] is True and source['raw_file'] is None
                       and source['raw_sha256'] is None and source['missing_reason'] == 'no_candidate',
                       'no-candidate additional date must have exactly nine zero fields')
                missing[index] += 1
                continue
            r = state['sources']['raw_records'][slot['source_key']]
            available = r['status'] == 'acquisition_complete'
            identity = overpass_key(r['item_id'], r['acquired_utc'])
            _guard(identity not in seen and '|'.join(source['overpass_identity']) == identity
                   and source['source_available'] is available and source['slot_supplied'] is available
                   and source['raw_sha256'] == r['raw_sha256'] and source['raw_file'] == r['raw_file']
                   and source.get('product_id') == r.get('product_id') and source['source_status'] == r['status']
                   and source['year'] == source['historical_year'] == r['historical_year']
                   and source['campaign_acquisition_exclusion_pass'] is True
                   and source['acquisition_digest'] == acquisition_digest(r['item_id'])
                   and source['acquisition_digest'] not in state['sources']['denied'], 'additional feature/raw identity, availability or overpass exclusion differs')
            seen.add(identity)
            if not available:
                _guard(zero and source['all_thermal_missing'] is True and source['missing_reason'] == r['status'],
                       'confirmed unavailable additional date is not entirely zero')
                missing[index] += 1
                continue
            _guard(source['missing_reason'] is None, 'available additional source was marked missing')
            if source['all_thermal_missing'] is True:
                _guard(zero, 'all-cloud additional source must have nine zero fields')
                cloud[index] += 1
                continue
            _guard(source['all_thermal_missing'] is False and nonempty, 'additional recent date has no clear coverage')
            for coverage in (value[2], value[5]):
                _guard(np.all((0 <= coverage) & (coverage <= 1)) and np.all(coverage * 16 == np.rint(coverage * 16)),
                       'additional coverage must count delivered30 cells')
            _guard(np.all((-3.00001 <= value[0]) & (value[0] <= 4.00001))
                   and np.all((0 < value[3]) & (value[3] <= 1.00001)), 'additional temperature/uncertainty outside physical contract')
            acquired, query = _time(r['acquired_utc']), _time(base['datetime'])
            age = (query - acquired).total_seconds() / 86400
            phase = 2 * np.pi * acquired.timetuple().tm_yday / 365.25
            _guard(8 <= age <= 64 and age == slot['age_days']
                   and all(np.all(value[6 + j] == np.float32(x)) for j, x in enumerate((np.sin(phase), np.cos(phase), age / 3652.5))),
                   'additional DOY/age does not use actual-query-relative time')
    _guard(active == detail['additional_slot_active_counts'], 'additional active date counts differ')
    return {'scene_count': count, 'shape': list(array.shape), 'dtype': 'float32', 'finite': True,
            'features_sha256': detail['features_sha256'], 'metadata_sha256': detail['metadata_sha256'],
            'base_seven_slots_bitwise_identical': True, 'additional_active_slots': active,
            'additional_source_missing_slots': missing, 'additional_all_cloud_slots': cloud,
            'actual_query_age_distinct_overpass_and_full_campaign_exclusion_pass': True}


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
            **role, 'historical_date_slots': 405, 'all648_top3_selections_recomputed': True,
            'time_contract': '2026 retrospective replay; historical public availability unknown',
            'query_target_arrays_opened': False, 'query_masks_opened': False, 'locked_test_opened': False}
