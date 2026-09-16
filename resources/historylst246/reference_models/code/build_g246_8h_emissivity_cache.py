#!/usr/bin/env python3
"""Target-free public emissivity cache with joint-mask QA-path removal."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

import g246_data
from acquire_g246_8h_emissivity_pilot import acquire, read_spec, transport
from train_g246_8h import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/'artifacts/g246_8h/emissivity_v1'
BASE = ROOT/'artifacts/g246_8h/cache_v1'
SCHEMA = 'g246-8h-joint-masked-emissivity-cache-v1'
CHANNELS = ['mean_emissivity_minus_098_div_001', 'std_emissivity_div_001',
            'log1p_mean_emsd_div_001', 'joint_ancillary_coverage']


def encode(e_dn, s_dn):
    """Joint mask precedes all averaging; only mask-invariant values survive."""
    assert e_dn.shape == s_dn.shape == (640, 640)
    mask = (e_dn >= 6000) & (e_dn <= 10000) & (s_dn > 0) & (s_dn <= 10000)
    e, s = e_dn.astype(np.float64)*1e-4, s_dn.astype(np.float64)*1e-4
    block = lambda a: a.reshape(160,4,160,4)
    counts = block(mask).sum((1,3))
    mean = lambda a: block(np.where(mask,a,0)).sum((1,3))/counts.clip(1)
    em, sm = mean(e), mean(s)
    es = np.sqrt(np.maximum(0, mean(e*e)-em*em))
    missing = counts == 0
    if missing.all():
        em[:] = .98; es[:] = 0; sm[:] = 0
    elif missing.any():
        nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
        for value in (em, es, sm):
            value[missing] = value[tuple(nearest[:,missing])]
    output = np.stack(((em-.98)/.01, es/.01, np.log1p(sm/.01), counts/16)).astype(np.float32)
    assert np.isfinite(output).all()
    return output, {'joint_source_coverage30': float(mask.mean()),
                    'nonempty_fraction120': float((~missing).mean()),
                    'rejected_positive_out_of_physical_range30': int(
                        (((e_dn>0)&((e_dn<6000)|(e_dn>10000))) | (s_dn>10000)).sum())}


class EmissivityCache:
    def __init__(self, root, role, expected_scene_ids):
        self.root = Path(root)
        m = json.loads((self.root/'manifest.json').read_text())
        if m.get('schema') != SCHEMA or m.get('status') != 'complete' \
                or m.get('locked_test_opened') is not False or m.get('target_arrays_opened') is not False:
            raise ValueError('emissivity cache not complete or unsafe')
        rows = json.loads((self.root/role/'metadata.json').read_text())['scenes']
        if [r['scene_id'] for r in rows] != list(expected_scene_ids):
            raise ValueError('emissivity scene order differs')
        self.array = np.load(self.root/role/'features.npy', mmap_mode='r')
        if self.array.shape != (len(rows),4,160,160) or self.array.dtype != np.float32:
            raise ValueError('emissivity feature array differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--prefetch-only', action='store_true')
    p.add_argument('--source-failure-receipt', type=Path)
    p.add_argument('--admissibility-receipt', type=Path, default=ROOT/'artifacts/g246_8h_20260905/emissivity_input_admissibility_audit.json')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'complete':
        raise FileExistsError('completed input cache is immutable')
    raw = out/'raw'; raw.mkdir(exist_ok=True)
    splits = g246_data.load_splits(role='fit+validation')
    entries = list(splits.fit)+list(splits.validation)
    assert len(splits.fit)==603 and len(splits.validation)==45
    unavailable = set()
    if a.source_failure_receipt:
        failure_audit = json.loads(a.source_failure_receipt.read_text())
        if failure_audit.get('pass') is not True or any(
                failure_audit.get(k) is not False for k in (
                    'target_arrays_opened', 'target_masks_opened', 'locked_test_opened')):
            raise ValueError('source-unavailability evidence audit must pass without labels')
        unavailable = set(failure_audit['scene_ids'])
        if not unavailable or not unavailable.issubset({e.scene_id for e in entries}):
            raise ValueError('source failure inventory differs from fixed scene universe')
        if failure_audit.get('schema') != 'g246-emissivity-source-failure-audit-v1' \
                or {r['scene_id'] for r in failure_audit['records']} != unavailable:
            raise ValueError('source failure audit records differ from declared inventory')
        for record in failure_audit['records']:
            attempts=record['attempts']
            if record.get('classification') != 'catalog_TIFF_URL_contains_stable_HTML_object' \
                    or record.get('identical_bytes_and_hash_on_repeat') is not True \
                    or len(attempts)<2 or len({r['object_sha256'] for r in attempts})!=1 \
                    or any(r.get('sha256_covers_entire_object') is not True for r in attempts):
                raise ValueError('fallback requires independently repeated non-raster source evidence')
    selection = {'schema': SCHEMA, 'scene_ids': [e.scene_id for e in entries],
                 'base_manifest_sha256': sha(BASE/'manifest.json'),
                 'assets': ['emis','emsd'], 'locked_test_opened': False,
                 'target_arrays_opened': False, 'target_masks_opened': False,
                 'training_supervision': 'unchanged Fit603',
                 'channel_names': CHANNELS, 'normalization': 'fixed physical scales; no fitted statistics'}
    selection_path = out/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('frozen public scope differs')
    atomic_json(selection_path, selection)
    pilot = ROOT/'artifacts/g246_8h/emissivity_pilot_v1'
    for e in entries:
        stem = hashlib.sha256(e.scene_id.encode()).hexdigest()[:16]
        for suffix in ('.npz','.json'):
            src, dst = pilot/(stem+suffix), raw/(stem+suffix)
            if src.exists() and not dst.exists(): os.link(src,dst)
    specs = [read_spec(e,allow_validation=True) for e in entries]
    tokens = transport.TokenManager(); rows=[]; failures=[]; started=time.time()
    with ThreadPoolExecutor(max_workers=min(16,max(1,a.workers))) as pool:
        futures={pool.submit(acquire,s,raw,tokens):s.entry for s in specs
                 if s.entry.scene_id not in unavailable}
        for f in as_completed(futures):
            try: rows.append(f.result())
            except Exception as exc: failures.append({'scene_id':futures[f].scene_id,'error_type':type(exc).__name__})
            if (len(rows)+len(failures))%12==0 or len(rows)+len(failures)==648-len(unavailable):
                progress={**selection,'status':'prefetch','completed':len(rows),'failures':failures,
                          'confirmed_unavailable_scene_ids':sorted(unavailable),
                          'elapsed_seconds':time.time()-started}
                atomic_json(manifest_path,progress)
                print(json.dumps({'event':'prefetch','completed':len(rows),'failed':len(failures),
                                  'elapsed_seconds':time.time()-started}),flush=True)
    if failures: raise SystemExit('Public ancillary prefetch incomplete; sanitized failures saved.')
    if len(rows)+len(unavailable) != len(entries):
        raise ValueError('source plus unavailable inventory must preserve every scene')
    if a.prefetch_only: return
    # Production-chain audit is required before publishing features for training.
    admissibility=json.loads(a.admissibility_receipt.read_text())
    if admissibility.get('pass') is not True:
        raise ValueError('independent emissivity input audit did not pass')
    source=out/'source';source.mkdir(exist_ok=True)
    source_hashes={}
    source_paths=[Path(__file__), ROOT/'code/acquire_g246_8h_emissivity_pilot.py',
                  ROOT/'code/acquire_effective120_v2.py', a.admissibility_receipt]
    if a.source_failure_receipt:
        source_paths.append(a.source_failure_receipt)
    for path in source_paths:
        destination=source/path.name
        shutil.copy2(path,destination)
        source_hashes[path.name]=sha(destination)
    by_id={r['scene_id']:r for r in rows}
    manifest={**selection,'status':'building',
              'admissibility_receipt':str((source/a.admissibility_receipt.name).resolve()),
              'admissibility_source_receipt':str(a.admissibility_receipt.resolve()),
              'admissibility_receipt_sha256':sha(a.admissibility_receipt),'roles':{},
              'source_code_sha256':source_hashes,
              'source_completed':len(rows),'feature_scene_count':len(entries),
              'confirmed_unavailable_scene_ids':sorted(unavailable),
              'source_failure_policy':'confirmed non-raster ancillary source: entire scene four channels zero; no scene or scoring pixel dropped',
              'joint_mask':'6000<=EMIS_DN<=10000 AND 0<EMSD_DN<=10000; int16 nodata<=0 excluded',
              'resampling':'same-grid joint nearest, THEN joint mask, THEN 4x4 aggregation',
              'gap_fill':'shared nearest 120m cell selected solely by joint availability',
              'original_qa_water_fill_or_separate_nodata_masks_not_features':True}
    if a.source_failure_receipt:
        manifest['source_failure_receipt']=str((source/a.source_failure_receipt.name).resolve())
        manifest['source_failure_receipt_sha256']=sha(a.source_failure_receipt)
    for role in ('fit','validation'):
        metadata=json.loads((BASE/role/'metadata.json').read_text())
        directory=out/role;directory.mkdir(exist_ok=True)
        feature=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype='float32',
                                         shape=(len(metadata['scenes']),4,160,160))
        meta_rows=[]
        for i,base_row in enumerate(metadata['scenes']):
            if base_row['scene_id'] in unavailable:
                feature[i] = 0
                meta_rows.append({'scene_id':base_row['scene_id'],'raw_sha256':None,
                                  'source_available':False,'fallback':'entire_scene_four_channels_zero',
                                  'source_failure_receipt_sha256':manifest['source_failure_receipt_sha256'],
                                  'joint_source_coverage30':0.,'nonempty_fraction120':0.})
                continue
            record=by_id[base_row['scene_id']]
            eg,sg=[record['assets'][k]['delivered_grid'] for k in ('emis','emsd')]
            if any(eg[k]!=sg[k] for k in ('crs','shape','transform')):
                raise ValueError('ancillary grids differ; joint-nearest proof not applicable')
            for key in ('emis','emsd'):
                band=record['assets'][key]['raster_bands'][0]
                if band['scale']!=.0001 or band['nodata']>0 \
                        or band['data_type']!='int16' or band.get('offset',0)!=0:
                    raise ValueError('ancillary encoding differs from joint-mask proof')
            with np.load(raw/record['file'],allow_pickle=False) as src:
                if any(src[k].dtype!=np.int16 for k in ('emis','emsd')):
                    raise ValueError('ancillary raw dtype differs from source contract')
                feature[i],stats=encode(src['emis'],src['emsd'])
            meta_rows.append({'scene_id':base_row['scene_id'],'raw_sha256':record['sha256'],
                              'source_available':True,
                              'source_scene_sha256':record['source_scene_sha256'],**stats})
        feature.flush();del feature
        atomic_json(directory/'metadata.json',{'schema':SCHEMA,'role':role,'scenes':meta_rows})
        manifest['roles'][role]={'scene_count':len(meta_rows),
            'base_metadata_sha256':sha(BASE/role/'metadata.json'),
            'metadata_sha256':sha(directory/'metadata.json'),
            'features_sha256':sha(directory/'features.npy'),'features_dtype':'float32',
            'features_shape':[len(meta_rows),4,160,160], 'features_bytes':(directory/'features.npy').stat().st_size}
    manifest['status']='complete';manifest['elapsed_seconds']=time.time()-started
    atomic_json(manifest_path,manifest)
    print(json.dumps({'event':'complete','roles':manifest['roles']}),flush=True)


if __name__=='__main__':main()
