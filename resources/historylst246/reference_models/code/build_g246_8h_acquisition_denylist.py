#!/usr/bin/env python3
"""Publish anonymous acquisition exclusions from frozen SOURCE JSON only.

No locked descriptor, source NPZ, scientific array, or remote asset is opened.
The global tier's source-manifest ancestry supplies 588 scalar item identities;
all 150 base50 identities are already in the explicitly public Fit descriptor.
Only fields needed for identity/hash/time/grid binding are used. No target/QA
statistics present elsewhere in source JSON are used for selection or outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / 'data/v2/source_global_G246_v1/manifest.json'
SPLIT = ROOT / 'data/v2/g246_metric_campaign_v1/split_receipt.json'
DEFAULT_OUTPUT = ROOT / 'artifacts/g246_8h/acquisition_exclusion_v1'
NAMESPACE = 'g246-acquisition-alias-v1'
PATTERN = re.compile(r'^(LC0[89])_L2SP_(\d{6})_(\d{8})(?:_\d{8})?_02_T1(?:_|$)')


def acquisition_digest(item):
    """Same platform/path-row/day across STAC and processing-product aliases."""
    match = PATTERN.match(str(item))
    if match is None:
        raise ValueError('requires a Landsat8/9 L2SP T1 acquisition identifier')
    return hashlib.sha256(('|'.join((NAMESPACE, *match.groups()))).encode('ascii')).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    path = Path(path)
    if path.suffix != '.json' or any(x in str(path).lower() for x in ('locked_test', 'locked-test', 'sealed')):
        raise ValueError('only whitelisted public/source JSON metadata may be read')
    data = path.read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('frozen source/descriptor JSON binding differs')
    return json.loads(data)


def atomic(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    out = a.output_root.resolve()
    if out.exists():
        raise FileExistsError('acquisition exclusion output must be a new immutable directory')
    receipt = read(SPLIT)
    source_binding = receipt['source_manifest']
    if Path(source_binding['path']) != GLOBAL:
        raise ValueError('split source is not the registered global source tier')
    global_source = read(GLOBAL, source_binding['sha256'])
    assert global_source['build_complete'] and global_source['source_only'] and global_source['split'] == 'source'
    assert global_source['scene_count'] == 738 and global_source['city_count'] == 246
    assert global_source['sealed_test_opened'] is False and global_source['sealed_test_unlocked'] is False
    public, fit_keys, bindings = {}, set(), []
    # Only these two literal descriptors are used; no locked descriptor field is
    # inspected, resolved, hashed, or opened.
    for role in ('fit', 'validation'):
        descriptor = receipt['split_files'][role + '.json']
        path = SPLIT.parent / (role + '.json')
        view = read(path, descriptor['sha256'])
        assert view['role'] == role and view['locked'] is False and view['arrays_allowed'] is True
        assert view['source_manifest']['sha256'] == source_binding['sha256']
        for row in view['scenes']:
            key = (row['city'], row['year'])
            assert key not in public
            public[key] = row
            if role == 'fit':
                fit_keys.add(key)
        bindings.append({'role': role, 'path': str(path), 'sha256': descriptor['sha256'], 'scene_count': len(view['scenes'])})
    assert len(public) == 648 and len(fit_keys) == 603
    aliases, grid_rows, used = set(), {}, set()
    lineage = []
    for role, binding in global_source['input_manifests'].items():
        source_path = Path(binding['path'])
        upstream = read(source_path, binding['sha256'])
        upstream_rows = {(x['city'], x['year']): x for x in upstream['scenes']}
        assert len(upstream_rows) == len(upstream['scenes'])
        direct = fallback = 0
        for row in global_source['scenes']:
            if row['input_role'] != role:
                continue
            key = (row['city'], row['year'])
            assert key not in used
            used.add(key)
            source = upstream_rows[key]
            assert source['sha256'] == row['sha256'] and row['input_manifest_sha256'] == binding['sha256']
            if source.get('item_id'):
                item_id, timestamp = source['item_id'], source['datetime']
                direct += 1
            else:
                assert role == 'base50' and key in fit_keys
                item_id, timestamp = public[key]['item_id'], public[key]['datetime']
                fallback += 1
            if key in public:
                assert public[key]['sha256'] == row['sha256']
                assert (item_id, timestamp) == (public[key]['item_id'], public[key]['datetime'])
            match = PATTERN.match(item_id)
            assert match is not None
            observed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            assert observed.tzinfo is not None and observed.strftime('%Y%m%d') == match.group(3)
            assert 2021 <= observed.year <= 2025 and observed.year == row['year']
            aliases.add(acquisition_digest(item_id))
            grid_rows[key] = {k: source[k] for k in ('canonical_grid_crs', 'transform120', 'shape120')}
        lineage.append({'input_role': role, 'path': str(source_path), 'sha256': binding['sha256'],
                        'direct_source_identity_count': direct, 'public_fit_identity_fallback_count': fallback})
    assert len(used) == len(global_source['scenes']) == len(grid_rows) == 738
    public_aliases = {acquisition_digest(r['item_id']) for r in public.values()}
    assert public_aliases <= aliases and len(aliases) == 633 and len(public_aliases) == 543
    # Source metadata can also describe all canonical AOI boundaries. A padded
    # metadata-only spatial check is additional evidence, not a substitute for
    # rejecting every frozen acquisition identity.
    from rasterio.warp import transform_bounds
    bounds = {}
    for (city, year), grid in grid_rows.items():
        t, shape = grid['transform120'], grid['shape120']
        assert len(t) == 9 and shape == [160, 160] and t[1] == t[3] == 0
        corners = (t[2], t[5] + t[4] * shape[0], t[2] + t[0] * shape[1], t[5])
        b = transform_bounds(grid['canonical_grid_crs'], 'EPSG:4326', *corners, densify_pts=32)
        b = (b[0] - .001, b[1] - .001, b[2] + .001, b[3] + .001)
        if city in bounds:
            assert bounds[city] == b
        bounds[city] = b
    public_cities = {r['city'] for r in public.values()}
    other_cities = set(bounds) - public_cities
    def intersects(a, b):
        return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]
    intersections = sum(intersects(bounds[a], bounds[b]) for a in public_cities for b in other_cities)
    spatial = {'source_grid_metadata_rows': 738, 'all246_city_three_date_grids_identical': True,
        'public_city_count': 216, 'remaining_source_city_count': 30,
        'cross_group_padded_geographic_bbox_pairs': 6480, 'intersections': intersections,
        'densify_points_per_edge': 32, 'bbox_padding_degrees': .001,
        'interpretation': 'metadata AOI footprint check; does not certify independence of scene-wide retrieval or shared ASTER artifacts'}
    out.mkdir(parents=True)
    deny = {'schema': 'g246-all-source-acquisition-denylist-v1', 'status': 'complete',
        'namespace': NAMESPACE,
        'digest_rule': "SHA256(ASCII('g246-acquisition-alias-v1|' + platform + '|' + WRS_pathrow_6digits + '|' + acquisition_YYYYMMDD)); processing date ignored",
        'acquisition_digests': sorted(aliases), 'acquisition_count': len(aliases), 'source_scene_count': 738,
        'source_manifest_sha256': source_binding['sha256'], 'split_receipt_sha256': sha(SPLIT),
        'source_manifests': lineage, 'public_views': bindings, 'builder_sha256': sha(Path(__file__)),
        'scope': 'all registered G246 source acquisitions irrespective of split; no split membership exported',
        'locked_descriptor_opened': False, 'npz_opened': False, 'scientific_arrays_opened': False,
        'raw_item_ids_or_city_memberships_exported': False}
    atomic(out / 'denylist.json', deny)
    audit = {'schema': 'g246-recent-history-input-isolation-audit-v1', 'pass': True, 'status': 'complete',
        'type': 'metadata provenance and acquisition exclusion only; not pixel feature or network audit',
        'denylist_path': str(out / 'denylist.json'), 'denylist_sha256': sha(out / 'denylist.json'),
        'source_manifest_sha256': source_binding['sha256'], 'split_receipt_sha256': sha(SPLIT),
        'all738_source_scene_hash_identity_bindings_verified': True,
        'all648_public_identity_datetime_hashes_match': True, 'source_acquisition_count': 633,
        'public_acquisition_count': 543, 'additional_excluded_acquisition_count': 90,
        'source_manifests': lineage, 'public_views': bindings, 'spatial_metadata_check': spatial,
        'verdict': 'conditionally admissible in a new revision after all633 exclusion before frozen ranking',
        'metadata_inventory_v1_is_raster_clearance': False,
        'required_restrictions': [
            'Exclude candidate acquisition_digest from all633 before age/cloud/datetime/id ranking; freeze new revision and receipt hash.',
            'Only actual public query AOI; exact 8 to 64 day positive UTC age; Landsat8/9 L2SP T1; catalogue cloud<=40; full AOI and six asset contracts.',
            'No query target, query QA, scoring masks or target caches as features, selection inputs, fallback or download paths.',
            'Read only historical assets from independent deployment source; verify asset full product identity, common delivered grid and canonical query grid.',
            'Keep every query and formal scoring pixel; historical-only QA, declared missing fallback, no coverage-based scoring.',
            '2026 retrospective replay; product availability at historical query time unproven. Days-old input is a template, not a short-time dynamical initial state.',
            'Shared ASTER/product and registration artifacts remain; compare against the fixed emissivity and older-history controls.',
            'No locked descriptor or scientific data access is authorized by this receipt; no scoring claim is made.'
        ],
        'conservative_fallback': 'retain the already audited pre2021 three/six-source input if any provenance, full633 membership, timestamp or geometry condition cannot be established',
        'locked_descriptor_opened': False, 'npz_opened': False, 'scientific_arrays_opened': False,
        'query_qa_statistics_used': False, 'target_statistics_used': False, 'network_calls': 0,
        'builder_sha256': sha(Path(__file__))}
    atomic(out / 'admissibility_audit.json', audit)
    print(json.dumps({'pass': True, 'status': 'complete', 'source_scenes': 738, 'acquisitions': 633,
        'denylist_sha256': sha(out / 'denylist.json'), 'audit_sha256': sha(out / 'admissibility_audit.json'),
        'spatial_padded_bbox_intersections': intersections}))


if __name__ == '__main__':
    main()
