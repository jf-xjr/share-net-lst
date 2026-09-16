"""Explicitly scheduled deployment check: fixed Fit scene0, seed905, one view.

Prepare-only until root authorizes running after the real bundle is built.
No fitting, validation, Test, labels, alternative scenes or tolerance search.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time

ROOT=Path(__file__).resolve().parents[4]
PACKAGE=ROOT/'resources/historylst246'
RECEIPT=ROOT/'research/sub04_20260913/compact_history_actions/policy/fit_predictions_v1/predictions_complete.json'
RECEIPT_SHA='02eed578577983cc70652e6e0f3230a9dc464b058fd5416b921abc02ab756bde'
CHECKPOINT_SHA='21432eb1bcf43b787e3fd8df9d198d3b21944bd47b9d53e825b73ecfc5f536ee'
FIELDS=('fine','coarse','support','context','emissivity','history')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--confirm-scheduled-fit-check',action='store_true',required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if sha(RECEIPT)!=RECEIPT_SHA:raise ValueError('Original fixed Fit prediction receipt changed')
    receipt=json.loads(RECEIPT.read_text());manifest=json.loads((PACKAGE/'manifest.json').read_text())
    if (receipt['split']!='fit' or len(receipt['scene_ids'])!=603 or receipt['batch']!=2
        or receipt['fp32'] is not True or receipt['tf32'] is not False
        or receipt['repair_dtype']!='float64' or receipt['labels_opened'] is not False
        or receipt['validation_opened'] is not False or receipt['test_opened'] is not False):
        raise ValueError('Original fixed Fit-only prediction protocol changed')
    if sha(PACKAGE/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('Original input contract changed')
    if receipt['model_bindings']['compact_full9']['checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('Cache uses different compact905')
    paths={k:(PACKAGE/manifest['roles']['fit']['fields'][k]['path']).resolve() for k in FIELDS}
    reference=receipt['entries']['compact_full9'];reference_path=Path(reference['path']).resolve()
    for k,path in paths.items():
        binding=receipt['input_bindings'][k]
        if (path!=Path(binding['path']).resolve() or sha(path)!=binding['sha256']
                or binding['sha256']!=manifest['roles']['fit']['fields'][k]['sha256']):
            raise ValueError('Original Fit input binding changed: '+k)
    if sha(reference_path)!=reference['sha256']:raise ValueError('Original sealed compact prediction changed')
    allowed=set(paths.values())|{reference_path}
    opened=[]
    def audit(event,event_args):
        if event=='open' and isinstance(event_args[0],(str,bytes)):
            path=Path(event_args[0]).resolve()
            if path.suffix in ('.npy','.npz'):
                if path not in allowed:raise RuntimeError('Only six Fit inputs and the sealed Fit prediction may be opened')
                opened.append(str(path))
    sys.addaudithook(audit)
    bundle=args.bundle.resolve()
    spec=importlib.util.spec_from_file_location('actual_portable_fit_check',bundle/'predict.py')
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    p.setup();bundle_manifest=p.verified_manifest();model,entry=p.load_model(bundle_manifest,20260905,args.device)
    if entry['sha256']!=CHECKPOINT_SHA:raise ValueError('Portable905 differs from predeclared original905')
    import numpy as np
    sample={k:np.array(np.load(path,mmap_mode='r',allow_pickle=False)[0:1],copy=True) for k,path in paths.items()}
    started=time.perf_counter();prediction=p.predict(model,sample,args.device);seconds=time.perf_counter()-started
    expected=np.array(np.load(reference_path,mmap_mode='r',allow_pickle=False)[0:1],copy=True)
    mask=sample['support'];difference=prediction[mask]-expected[mask]
    supported=bool(np.isfinite(prediction[mask]).all() and np.isfinite(expected[mask]).all() and np.isfinite(difference).all())
    unsupported=bool(np.isnan(prediction[~mask]).all() and np.isnan(expected[~mask]).all())
    maximum=float(np.max(np.abs(difference))) if supported and difference.size else None
    passed=bool(supported and unsupported and maximum is not None and maximum<=1e-4)
    result=dict(status='complete_fixed_fit_scene_portable_equivalence_check',passed=passed,scene_index=0,
        scene_id=receipt['scene_ids'][0],seed=20260905,checkpoint_sha256=entry['sha256'],
        bundle_manifest_sha256=sha(bundle/'manifest.json'),source_sha256=sha(__file__),
        prediction_receipt_sha256=RECEIPT_SHA,reference_prediction_sha256=reference['sha256'],
        original_input_manifest_sha256=receipt['manifest_sha256'],device=args.device,
        tolerance_k=1e-4,supported_maximum_difference_k=maximum,supported_finite=supported,
        unsupported_nan=unsupported,batch=1,reference_batch=2,image_forwards=1,network_calls=1,
        teacher_used_at_inference=False,ensemble_used=False,labels_opened=False,validation_opened=False,
        test_opened=False,opened_array_paths=sorted(set(opened)),elapsed_seconds=seconds,timing_is_benchmark=False,
        note='Single fixed deployment equality check; FP32 batch1 versus cached batch2 can differ by rounding.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,allow_nan=False))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
