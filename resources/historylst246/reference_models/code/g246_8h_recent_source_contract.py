"""Offline, independent selection and source binding for recent history.

Only frozen catalogue JSON, public query metadata and anonymous acquisition
hashes are read. No locked descriptor, query archive or science array is used.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from shapely.geometry import box, shape

from build_g246_8h_acquisition_denylist import acquisition_digest
from g246_8h_historical_contract import _path, _json, _sha, _time, _guard, _identity

DENY_SHA = '1dc9d39a44ebcf34864d72a72b7c61c8ae488fb98c58ea6c185964c633f6a90a'
AUDIT_SHA = 'd4fc37a1adcde53d586bc2d5643a1fc6f613b938250169d8d56b710ebcb44e3d'
RAW_SCHEMA = 'g246-8h-historical-recent-source-v2'
GRID_KEYS = ('canonical_crs', 'canonical_transform30', 'canonical_shape', 'grid_signature_sha256')
FORBIDDEN = ('current_query_target_arrays_opened', 'current_query_target_masks_opened',
             'current_query_qa_arrays_opened', 'locked_test_opened', 'new_supervision_created')
ASSETS = {'lwir11': ('uint16', .00341802, 149., '_ST_B10.TIF'),
          'qa': ('int16', .01, 0., '_ST_QA.TIF'), 'qa_pixel': ('uint16', 1., 0., '_QA_PIXEL.TIF'),
          'qa_radsat': ('uint16', 1., 0., '_QA_RADSAT.TIF'),
          'emis': ('int16', .0001, 0., '_ST_EMIS.TIF'), 'emsd': ('int16', .0001, 0., '_ST_EMSD.TIF')}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def independent_rank(query, response, forbidden):
    _guard(not any(x.get('rel') == 'next' for x in response.get('links', [])), 'recent catalogue response was silently truncated')
    ranked, excluded_count, identities = [], 0, set()
    for item in response.get('features', []):
        p, item_id = item['properties'], item['id']
        if item_id.startswith(('LC08_L2SP_', 'LC09_L2SP_')) and acquisition_digest(item_id) in forbidden:
            excluded_count += 1
            continue
        acquired = _time(p['datetime'])
        age = (_time(query['datetime']) - acquired).total_seconds() / 86400
        if not 8 <= age <= 64:
            continue
        if p.get('platform') not in ('landsat-8', 'landsat-9') or p.get('landsat:correction') != 'L2SP' \
                or p.get('landsat:collection_category') != 'T1' or float(p['eo:cloud_cover']) > 40:
            continue
        if not set(ASSETS) <= set(item['assets']) or not shape(item['geometry']).covers(box(*query['bbox_wgs84'])):
            continue
        identity = _identity(item_id)
        products = set()
        for key, (dtype, scale, offset, suffix) in ASSETS.items():
            asset = item['assets'][key]
            band = asset['raster:bands'][0]
            name = Path(urlsplit(asset['href']).path).name
            _guard(_identity(name) == identity and name.endswith(suffix)
                   and band['data_type'] == dtype and band.get('scale', 1.) == scale
                   and band.get('offset', 0.) == offset, 'recent selected asset has a different acquisition/type/scale')
            products.add(name[:-len(suffix)])
        _guard(len(products) == 1 and identity not in identities, 'recent source has mixed or ambiguous processing products')
        identities.add(identity)
        ranked.append((age, float(p['eo:cloud_cover']), p['datetime'], item_id, item))
    ranked.sort(key=lambda value: value[:4])
    return ranked, excluded_count


def check_selection(inventory_path, expected_sha256, raw_selection_path=None, raw_selection_sha256=None):
    inventory_path = _path(inventory_path)
    inv = _json(inventory_path, expected_sha256)
    _guard(inv.get('schema') == 'g246-recent-historical-template-metadata-v2'
           and inv.get('status') == 'metadata_inventory_complete' and inv.get('revision') == 2
           and all(inv.get(k) is False for k in FORBIDDEN), 'recent inventory is incomplete or used query data')
    _guard(inv['campaign_denylist_sha256'] == DENY_SHA and inv['campaign_denylist_audit_sha256'] == AUDIT_SHA,
           'recent inventory is not bound to the audited full campaign exclusion')
    deny = _json(_path(inv['campaign_denylist_path']), DENY_SHA)
    audit = _json(_path(inv['campaign_denylist_audit_path']), AUDIT_SHA)
    forbidden = set(deny['acquisition_digests'])
    _guard(deny.get('status') == 'complete' and len(forbidden) == deny['acquisition_count'] == 633
           and deny['source_scene_count'] == 738 and deny['namespace'] == 'g246-acquisition-alias-v1'
           and audit.get('pass') is True and audit.get('status') == 'complete'
           and audit['denylist_sha256'] == DENY_SHA and audit['all738_source_scene_hash_identity_bindings_verified'] is True,
           'anonymous campaign acquisition exclusion proof failed')
    selection = _json(_path(inv['selection_path']), inv['selection_sha256'])
    _guard(selection['schema'] == 'g246-recent-historical-template-selection-v2'
           and selection['query_records'] == inv['query_records'] and selection['source_plans'] == inv['source_plans'],
           'recent source selection differs from frozen inventory')
    original = _json(_path(inv['original_inventory_manifest_path']), inv['original_inventory_manifest_sha256'])
    _guard(original['status'] == 'metadata_inventory_complete' and all(original.get(k) is False for k in FORBIDDEN),
           'recent original metadata inventory is incomplete')
    base = _json(_path(original['source_inventory_path']), original['source_inventory_sha256'])
    public = {q['scene_id']: q for q in base['public_query_scenes']}
    cities = {c['city']: c for c in base['cities']}
    original_rows = {q['scene_id']: q for q in original['records']}
    records = {q['scene_id']: q for q in inv['records']}
    queries = {q['scene_id']: q for q in inv['query_records']}
    plans = {p['source_key']: p for p in inv['source_plans']}
    _guard(len(public) == len(original_rows) == len(records) == len(queries) == len(inv['records']) == len(inv['query_records']) == 648
           and set(public) == set(original_rows) == set(records) == set(queries)
           and len(plans) == len(inv['source_plans']) == inv['source_count'] == 627, 'recent public population/source count differs')
    response_path = _path(inv['full_responses_manifest_path'])
    responses = _json(response_path, inv['full_responses_manifest_sha256'])
    _guard(responses.get('ready_for_denylist_revision') is True
           and responses.get('all_original_rankings_and_selected_metadata_reproduced') is True
           and responses['original_manifest_sha256'] == inv['original_inventory_manifest_sha256']
           and not responses.get('errors') and all(responses.get(k) is False for k in FORBIDDEN),
           'recent full catalogue response provenance differs')
    response_rows = {r['scene_id']: r for r in responses['records']}
    _guard(len(response_rows) == len(responses['records']) == 648 and set(response_rows) == set(queries),
           'recent full responses do not cover every public query')
    refs, counts, blocked, changed = defaultdict(list), Counter(), 0, 0
    for sid, q in queries.items():
        old, row, scene, city = original_rows[sid], records[sid], public[sid], cities[q['city']]
        _guard(q['city'] == scene['city'] and q['region'] == scene['region'] and q['role'] == scene['role']
               and q['query_datetime'] == scene['datetime'] == old['datetime']
               and q['query_source_sha256'] == scene['source_sha256'] == old['source_sha256']
               and all(q[k] == city[k] == old[k] for k in GRID_KEYS)
               and all(row[k] == v for k, v in q.items()), 'recent query mapping, actual time or canonical grid differs')
        receipt = response_rows[sid]
        _guard(receipt.get('pass') is True and all(receipt['checks'].values())
               and row['full_response_file'] == receipt['response_file']
               and row['full_response_sha256'] == receipt['response_sha256'], 'recent response receipt binding differs')
        response = _json(_path(receipt['response_file'], response_path.parent), receipt['response_sha256'])
        ranked, n_blocked = independent_rank(old, response, forbidden)
        blocked += n_blocked
        compact = [{'age_days': r[0], 'eo:cloud_cover': r[1], 'datetime': r[2], 'item_id': r[3]} for r in ranked]
        item = ranked[0][4] if ranked else None
        _guard(compact == row['ranked_candidates'] and item == row['selected_item'] and len(ranked) == row['eligible_count'],
               'recent full633 exclusion was not applied before frozen ranking')
        _guard(len(q['slots']) == 1, 'recent query must have exactly one added date slot')
        slot = q['slots'][0]
        kind = 'new_source' if item else 'no_candidate'
        counts[kind] += 1
        expected_id = item['id'] if item else None
        _guard(slot['status'] == kind and slot['selected_item_id'] == expected_id
               and slot['source_key'] == (q['city'] + '|' + expected_id if item else None)
               and slot['selected_datetime'] == (item['properties']['datetime'] if item else None)
               and slot['age_days'] == (ranked[0][0] if item else None)
               and slot['selected_catalogue_cloud_percent'] == (ranked[0][1] if item else None),
               'recent one-slot identity/time metadata differs from selected catalogue source')
        previous_id = old['selected_item']['id'] if old['selected_item'] else None
        changed += previous_id != expected_id
        if item:
            key, acquired = slot['source_key'], _time(item['properties']['datetime'])
            plan = plans[key]
            _guard(acquisition_digest(expected_id) not in forbidden and slot['historical_year'] == acquired.year
                   and plan['item_id'] == expected_id and plan['item_metadata_sha256'] == json_sha(item)
                   and plan['acquired_utc'] == item['properties']['datetime']
                   and plan['historical_year'] == acquired.year and all(plan[k] == q[k] for k in GRID_KEYS)
                   and plan['city'] == q['city'] and plan['region'] == q['region'] and plan['role'] == q['role']
                   and plan['acquisition_digest'] == acquisition_digest(expected_id), 'recent raw source plan identity or exclusion differs')
            refs[key].append(sid)
    _guard(dict(counts) == {'new_source': 627, 'no_candidate': 21} and set(refs) == set(plans)
           and blocked == inv['campaign_alias_candidate_references_excluded'] == 86
           and changed == inv['different_selected_query_count_from_public_only_v1'] == 0, 'recent frozen selection counts differ')
    for key, plan in plans.items():
        _guard(sorted(plan['query_scene_ids']) == sorted(refs[key]) and len(plan['query_scene_ids']) == len(set(plan['query_scene_ids']))
               and plan['query_scene_id'] in refs[key] and plan['query_scene_sha256'] == queries[plan['query_scene_id']]['query_source_sha256'],
               'recent source actual-query membership/anchor differs')
    frozen_raw = None
    if raw_selection_path is not None:
        frozen_raw = _json(_path(raw_selection_path), raw_selection_sha256)
        _guard(frozen_raw['schema'] == RAW_SCHEMA and frozen_raw['inventory_manifest_sha256'] == expected_sha256
               and frozen_raw['inventory_selection_sha256'] == inv['selection_sha256']
               and frozen_raw['query_records'] == inv['query_records'] and frozen_raw['source_plans'] == inv['source_plans']
               and frozen_raw['campaign_denylist_sha256'] == DENY_SHA and frozen_raw['campaign_denylist_audit_sha256'] == AUDIT_SHA
               and all(frozen_raw.get(k) is False for k in FORBIDDEN), 'recent frozen raw selection differs')
    return {'inventory': inv, 'queries': queries, 'records': records, 'plans': plans,
            'denied': forbidden, 'raw_selection': frozen_raw,
            'report': {'pass': True, 'inventory_manifest_sha256': expected_sha256,
                'raw_selection_sha256': raw_selection_sha256, 'campaign_denylist_sha256': DENY_SHA,
                'campaign_denylist_audit_sha256': AUDIT_SHA, 'queries': 648, 'source_plans': 627,
                'no_candidate_queries': 21, 'all648_full_response_rankings_independently_recomputed': True,
                'all633_acquisition_hashes_excluded_before_ranking': True,
                'campaign_alias_candidate_references_excluded': blocked,
                'changed_from_metadata_v1': changed, 'all_public_query_identity_time_grid_bindings_pass': True,
                'current_query_target_arrays_opened': False, 'current_query_target_masks_opened': False,
                'current_query_qa_arrays_opened': False, 'locked_test_opened': False,
                'remote_raster_opened': False, 'network_calls': 0}}


def check_raw(raw_path, expected_sha256):
    """Bind each completed raw source to its already audited recent selection."""
    raw_path = _path(raw_path)
    raw = _json(raw_path, expected_sha256)
    _guard(raw.get('schema') == RAW_SCHEMA and raw.get('status') == 'source_plan_complete'
           and raw.get('ready_for_feature_build') is True
           and all(raw.get(k) is False for k in FORBIDDEN), 'recent raw source is incomplete or outside query-information boundary')
    state = check_selection(raw['inventory_manifest_path'], raw['inventory_manifest_sha256'],
                            raw['selection_path'], raw['selection_sha256'])
    frozen = state['raw_selection']
    _guard(raw['query_records'] == frozen['query_records'] and raw['source_plans'] == frozen['source_plans']
           and raw['campaign_denylist_sha256'] == DENY_SHA and raw['campaign_denylist_audit_sha256'] == AUDIT_SHA,
           'recent raw manifest changed its frozen query/source selection')
    records = {r['source_key']: r for r in raw['records']}
    _guard(len(records) == len(raw['records']) == 627 and set(records) == set(state['plans']),
           'recent raw source universe differs from all frozen selections')
    for key, record in records.items():
        plan = state['plans'][key]
        anchor = state['queries'][plan['query_scene_id']]
        item = state['records'][plan['query_scene_id']]['selected_item']
        _guard(record.get('schema') == RAW_SCHEMA and record.get('status') in ('acquisition_complete', 'confirmed_non_tiff_missing')
               and all(record.get(k) is False for k in FORBIDDEN)
               and record['selection_sha256'] == raw['selection_sha256']
               and record['campaign_denylist_sha256'] == DENY_SHA, 'recent source unresolved or mismatched exclusion contract')
        _guard(record['item_id'] == plan['item_id'] and record['item_metadata_sha256'] == plan['item_metadata_sha256'] == json_sha(item)
               and record['acquired_utc'] == plan['acquired_utc'] == item['properties']['datetime']
               and record['historical_year'] == plan['historical_year']
               and record['query_scene_ids'] == plan['query_scene_ids']
               and record['query_scene_id'] == plan['query_scene_id']
               and record['query_scene_sha256'] == plan['query_scene_sha256']
               and record['query_view_role'] == anchor['role'] and record['query_datetime'] == anchor['query_datetime']
               and record['city'] == plan['city'] and all(record[k] == plan[k] == anchor[k] for k in GRID_KEYS)
               and record['resampling'] == 'nearest', 'recent raw identity, anchor time, source hash or grid differs')
        digest = acquisition_digest(record['item_id'])
        _guard(digest == record['acquisition_digest'] == plan['acquisition_digest'] and digest not in state['denied'],
               'recent raw source aliases a registered campaign acquisition')
        age = (_time(anchor['query_datetime']) - _time(record['acquired_utc'])).total_seconds() / 86400
        _guard(8 <= age <= 64 and age == record['age_days_at_anchor_query'], 'recent raw anchor does not have its real positive 8--64 day age')
        if record['status'] == 'acquisition_complete':
            _guard(not record.get('failures') and record.get('raw_file') and record.get('raw_sha256')
                   and len(record['assets']) == 6, 'recent complete source lacks all six raw assets')
        else:
            _guard(record.get('failures') and all(f.get('source_confirmation', {}).get('confirmed') is True
                   for f in record['failures']), 'recent missing source lacks repeated non-TIFF confirmation')
    state.update(raw=raw, raw_root=raw_path.parent, raw_records=records)
    return state
