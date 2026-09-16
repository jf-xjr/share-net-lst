"""Independent metadata admission of the second/third distinct recent overpasses."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from g246_8h_recent_source_contract import (
    AUDIT_SHA, DENY_SHA, FORBIDDEN, GRID_KEYS, acquisition_digest, check_selection,
    independent_rank, json_sha, _guard, _identity, _json, _path, _sha, _time,
)

PLAN_SHA = '346dfd26701400243b5f0ad2a2cd8179691c4417e909e51054a1d40ea6fd45b3'
RAW_SCHEMA = 'g246-8h-historical-recent-pair-source-v1'


def overpass_key(identifier, acquired):
    platform, pathrow, _ = _identity(identifier)
    return platform + '|' + pathrow[:3] + '|' + _time(acquired).date().isoformat()


def check_top3(path, expected_sha256=PLAN_SHA):
    plan = _json(_path(path), expected_sha256)
    _guard(expected_sha256 == PLAN_SHA and plan.get('schema') == 'g246-recent-top3-metadata-only-plan-v1'
           and plan.get('variant') == 'distinct_overpass_top3'
           and plan.get('status') == 'feasibility_plan_only_not_acquired'
           and plan.get('first_slot_complete_metadata_equal_to_recent_v2_all648') is True
           and plan.get('network_calls') == 0
           and all(plan.get(k) is False for k in (*FORBIDDEN, 'locked_descriptor_opened', 'model_metrics_used',
                    'scientific_arrays_opened', 'raster_downloaded', 'training_features_created',
                    'existing_selection_or_manifest_modified')), 'top3 plan is not the frozen metadata-only distinct-overpass revision')
    state = check_selection(plan['inventory_manifest_path'], plan['inventory_manifest_sha256'])
    inventory = state['inventory']
    _guard(plan['campaign_denylist_sha256'] == DENY_SHA and plan['campaign_denylist_audit_sha256'] == AUDIT_SHA
           and plan['full_responses_manifest_sha256'] == inventory['full_responses_manifest_sha256'],
           'top3 plan changes the established catalogue or full633 exclusion')
    original = _json(_path(inventory['original_inventory_manifest_path']), inventory['original_inventory_manifest_sha256'])
    originals = {q['scene_id']: q for q in original['records']}
    old_raw = _json(_path(plan['current_recent_raw_manifest_path']), plan['current_recent_raw_manifest_sha256'])
    _guard(old_raw['schema'] == 'g246-8h-historical-recent-source-v2' and old_raw['ready_for_feature_build'] is True,
           'top3 plan uses a different first-slot source universe')
    old_sources = {r['source_key']: r for r in old_raw['records']}
    queries = {q['scene_id']: q for q in plan['query_records']}
    additional = {p['source_key']: p for p in plan['additional_source_plans']}
    _guard(len(queries) == len(plan['query_records']) == 648 and set(queries) == set(state['queries'])
           and len(additional) == len(plan['additional_source_plans']) == 1023, 'top3 population or source uniqueness differs')
    refs, counts, metadata = defaultdict(list), Counter(), {}
    for sid, q in queries.items():
        old_q, old_record = state['queries'][sid], state['records'][sid]
        _guard(all(q[k] == old_q[k] for k in ('scene_id', 'city', 'region', 'role', 'query_datetime',
                   'query_source_sha256', *GRID_KEYS))
               and _path(q['full_response_file']) == _path(old_record['full_response_file'],
                   _path(plan['full_responses_manifest_path']).parent)
               and q['full_response_sha256'] == old_record['full_response_sha256'], 'top3 query/grid/full-response binding differs')
        response = _json(_path(q['full_response_file']), q['full_response_sha256'])
        ranked, _ = independent_rank(originals[sid], response, state['denied'])
        distinct, seen = [], set()
        for value in ranked:
            key = overpass_key(value[3], value[2])
            if key not in seen:
                seen.add(key)
                distinct.append(value)
        _guard(len(q['slots']) == 3 and [s['rank'] for s in q['slots']] == [1, 2, 3], 'top3 ranks are not exact ordered1/2/3')
        _guard((distinct[0][4] if distinct else None) == old_record['selected_item'], 'top3 first complete catalogue metadata changed')
        seen_slots = set()
        for index, slot in enumerate(q['slots']):
            candidate = distinct[index] if index < len(distinct) else None
            if candidate is None:
                _guard(slot['planned_status'] == 'no_candidate'
                       and all(slot[k] is None for k in ('source_key', 'item_id', 'acquired_utc', 'age_days',
                           'catalogue_cloud_percent', 'acquisition_digest', 'overpass_key', 'item_metadata_sha256', 'existing_raw_status')),
                       'unavailable top3 candidate contains invented metadata')
                counts[f'rank{index+1}_no_candidate'] += 1
                continue
            age, cloud, acquired, item_id, item = candidate
            key = q['city'] + '|' + item_id
            old = old_sources.get(key)
            expected_status = 'reuse_existing_identity' if old else 'new_raw_required'
            _guard(slot['source_key'] == key and slot['item_id'] == item_id and slot['acquired_utc'] == acquired
                   and slot['age_days'] == age and slot['catalogue_cloud_percent'] == cloud
                   and slot['acquisition_digest'] == acquisition_digest(item_id)
                   and slot['acquisition_digest'] not in state['denied']
                   and slot['item_metadata_sha256'] == json_sha(item)
                   and slot['overpass_key'] == overpass_key(item_id, acquired)
                   and slot['overpass_key'] not in seen_slots and slot['planned_status'] == expected_status
                   and slot['existing_raw_status'] == (old['status'] if old else None), 'top3 independently ranked distinct source differs')
            seen_slots.add(slot['overpass_key'])
            counts[f'rank{index+1}_available'] += 1
            if index == 0:
                _guard(old is not None and old['item_metadata_sha256'] == json_sha(item), 'first recent source complete metadata changed')
            else:
                _guard(key in additional and old is None, 'additional source improperly reuses first-source identity')
                expected = {**slot, 'city': q['city'], 'region': q['region'], 'role': q['role']}
                _guard(all(additional[key][k] == v for k, v in expected.items()), 'additional source plan differs from exact ranked item')
                refs[key].append({'scene_id': sid, 'rank': index + 1})
                metadata[key] = item
    _guard(dict(counts) == {'rank1_available': 627, 'rank2_available': 559, 'rank3_available': 464,
                           'rank2_no_candidate': 89, 'rank3_no_candidate': 184, 'rank1_no_candidate': 21}
           and set(refs) == set(additional), 'top3 exact reference counts changed')
    for key, rows in refs.items():
        _guard(additional[key]['references'] == rows, 'top3 source reference membership/order changed')
    return {'plan': plan, 'queries': queries, 'plans': additional, 'metadata': metadata, 'original_state': state,
        'old_raw': old_raw, 'report': {'pass': True, 'status': 'complete', 'top3_plan_sha256': expected_sha256,
            'inventory_manifest_sha256': plan['inventory_manifest_sha256'],
            'original_recent_raw_manifest_sha256': plan['current_recent_raw_manifest_sha256'],
            'campaign_denylist_sha256': DENY_SHA, 'full_response_manifest_sha256': plan['full_responses_manifest_sha256'],
            'all648_first_slot_complete_metadata_unchanged': True, 'all648_rankings_recomputed_from_full_responses': True,
            'all633_aliases_excluded_before_ranking': True, 'all_selected_overpasses_distinct_within_query': True,
            'overpass_key': 'platform|WRSpath3|actualUTCdate', 'all_actual_query_age_8to64_days': True,
            'all_query_identity_and_canonical_grid_bindings_pass': True, 'additional_unique_sources': 1023,
            'counts': dict(counts), 'current_query_target_arrays_opened': False,
            'current_query_target_masks_opened': False, 'current_query_qa_arrays_opened': False,
            'scientific_arrays_opened': False, 'locked_descriptor_opened': False, 'locked_test_opened': False,
            'network_calls': 0, 'model_metrics_used': False, 'new_supervision_created': False,
            'scope': 'source acquisition metadata boundary only; no raw feature or future encoder approval',
            'time_contract': '2026 historical replay; historical public availability unknown'}}


def check_raw(path, expected_sha256):
    """Bind completed additional raw sources to the independently ranked plan."""
    path = _path(path)
    raw = _json(path, expected_sha256)
    _guard(raw.get('schema') == RAW_SCHEMA and raw.get('status') == 'source_plan_complete'
           and raw.get('ready_for_feature_build') is True
           and all(raw.get(k) is False for k in (*FORBIDDEN, 'locked_descriptor_opened',
               'training_features_created', 'original_recent_or_seven_inputs_modified', 'model_metrics_used_for_selection')),
           'additional recent raw source is incomplete or crossed the query boundary')
    state = check_top3(raw['top3_plan_path'], raw['top3_plan_sha256'])
    selection = _json(_path(raw['selection_path']), raw['selection_sha256'])
    _guard(selection['schema'] == RAW_SCHEMA and raw['query_records'] == selection['query_records']
           and raw['source_plans'] == selection['source_plans']
           and raw['selected_by_rank'] == {'2': 559, '3': 464}
           and raw['no_candidate_by_rank'] == {'2': 89, '3': 184}
           and raw.get('first_recent_and_existing_seven_inputs_unchanged') is True,
           'additional source selection/query population changed')
    for name in ('campaign_denylist', 'campaign_denylist_audit', 'original_recent_raw_manifest',
                 'original_recent_raw_selection', 'top3_plan', 'top3_count_receipt', 'full_responses_manifest'):
        _guard(raw[name + '_sha256'] == selection[name + '_sha256']
               and _sha(_path(raw[name + '_path'])) == raw[name + '_sha256'], 'additional raw source provenance binding differs')
    _guard(raw['campaign_denylist_sha256'] == DENY_SHA and raw['campaign_denylist_audit_sha256'] == AUDIT_SHA
           and raw['original_recent_raw_manifest_sha256'] == state['plan']['current_recent_raw_manifest_sha256']
           and raw['original_recent_raw_selection_sha256'] == state['old_raw']['selection_sha256']
           and raw['full_responses_manifest_sha256'] == state['plan']['full_responses_manifest_sha256'],
           'additional source changes the audited original source universe')
    count_receipt = _json(_path(raw['top3_count_receipt_path']), raw['top3_count_receipt_sha256'])
    _guard(count_receipt.get('pass') is True and count_receipt['first_slot_complete_metadata_matches_recent_v2_queries'] == 648,
           'top3 metadata count receipt failed')
    queries = {q['scene_id']: q for q in raw['query_records']}
    plans = {p['source_key']: p for p in raw['source_plans']}
    records = {r['source_key']: r for r in raw['records']}
    _guard(len(queries) == len(raw['query_records']) == 648 and set(queries) == set(state['queries'])
           and len(plans) == len(raw['source_plans']) == len(records) == len(raw['records']) == 1023
           and set(plans) == set(records) == set(state['plans']), 'additional raw source identity population differs')
    refs = defaultdict(list)
    for sid, q in queries.items():
        original = state['queries'][sid]
        _guard(all(q[k] == original[k] for k in ('scene_id', 'city', 'region', 'role', 'query_datetime',
                   'query_source_sha256', *GRID_KEYS))
               and q['first_recent_item_id'] == original['slots'][0]['item_id']
               and q['first_recent_overpass_key'] == original['slots'][0]['overpass_key']
               and len(q['slots']) == 2, 'additional raw query or first-slot identity changed')
        for index, slot in enumerate(q['slots']):
            source = original['slots'][index + 1]
            _guard(slot['rank'] == source['rank'] == index + 2
                   and slot['status'] == ('no_candidate' if source['item_id'] is None else 'new_source')
                   and slot['source_key'] == source['source_key'] and slot['selected_item_id'] == source['item_id']
                   and slot['selected_datetime'] == source['acquired_utc'] and slot['age_days'] == source['age_days']
                   and slot['selected_catalogue_cloud_percent'] == source['catalogue_cloud_percent']
                   and slot['overpass_key'] == source['overpass_key']
                   and slot['historical_year'] == (_time(source['acquired_utc']).year if source['item_id'] else None),
                   'additional raw slot no longer matches the independent fixed rank')
            if source['source_key']:
                refs[source['source_key']].append(sid)
    for key, record in records.items():
        plan, prior = plans[key], state['plans'][key]
        anchor = queries[plan['query_scene_id']]
        item = state['metadata'][key]
        _guard(plan['source_key'] == prior['source_key'] and plan['query_scene_ids'] == refs[key]
               and plan['query_scene_id'] in refs[key]
               and all(plan[k] == prior[k] for k in ('city', 'region', 'role', 'item_id', 'acquired_utc',
                           'acquisition_digest', 'overpass_key', 'rank', 'item_metadata_sha256'))
               and plan['query_scene_sha256'] == anchor['query_source_sha256']
               and all(plan[k] == anchor[k] for k in GRID_KEYS)
               and plan['full_response_sha256'] == state['queries'][plan['query_scene_id']]['full_response_sha256']
               and _path(plan['full_response_file']) == _path(state['queries'][plan['query_scene_id']]['full_response_file']),
               'additional raw source plan/query anchor differs')
        _guard(record.get('schema') == RAW_SCHEMA and record.get('status') in ('acquisition_complete', 'confirmed_non_tiff_missing')
               and all(record.get(k) is False for k in FORBIDDEN)
               and record['selection_sha256'] == raw['selection_sha256']
               and record['top3_plan_sha256'] == PLAN_SHA
               and record['original_recent_raw_manifest_sha256'] == raw['original_recent_raw_manifest_sha256']
               and record['campaign_denylist_sha256'] == DENY_SHA, 'additional raw source unresolved or misbound')
        _guard(record['item_id'] == plan['item_id'] == item['id']
               and record['item_metadata_sha256'] == plan['item_metadata_sha256'] == json_sha(item)
               and record['acquired_utc'] == plan['acquired_utc'] == item['properties']['datetime']
               and record['historical_year'] == plan['historical_year'] == _time(record['acquired_utc']).year
               and record['query_scene_ids'] == refs[key] and record['query_scene_id'] == plan['query_scene_id']
               and record['query_scene_sha256'] == plan['query_scene_sha256']
               and record['query_view_role'] == anchor['role'] and record['query_datetime'] == anchor['query_datetime']
               and record['city'] == plan['city'] and all(record[k] == plan[k] for k in GRID_KEYS)
               and record['resampling'] == 'nearest'
               and record['full_response_sha256'] == plan['full_response_sha256']
               and record['rank_at_anchor_query'] == plan['rank']
               and record['overpass_key'] == overpass_key(record['item_id'], record['acquired_utc']) == plan['overpass_key'],
               'additional raw source identity/time/grid/overpass differs')
        digest = acquisition_digest(record['item_id'])
        age = (_time(anchor['query_datetime']) - _time(record['acquired_utc'])).total_seconds() / 86400
        _guard(digest == record['acquisition_digest'] == plan['acquisition_digest']
               and digest not in state['original_state']['denied']
               and 8 <= age <= 64 and age == record['age_days_at_anchor_query'], 'additional source exclusion or actual anchor age differs')
        if record['status'] == 'acquisition_complete':
            _guard(not record.get('failures') and record.get('raw_file') and record.get('raw_sha256')
                   and len(record['assets']) == 6, 'complete additional source lacks six raw assets')
        else:
            _guard(record.get('failures') and all(f.get('source_confirmation', {}).get('confirmed') is True
                   for f in record['failures']), 'missing additional source lacks repeated nonTIFF confirmation')
    state.update(raw=raw, raw_root=path.parent, raw_records=records, queries=queries,
                 denied=state['original_state']['denied'], raw_selection=selection)
    return state
