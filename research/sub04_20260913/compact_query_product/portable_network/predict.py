"""Portable single-view query-product compact LST inference from six already encoded NPZ fields.

Only bundle-local source/weight hashes are verified. No experiment workspace,
training parents, teacher, observations, dataset manifest or logs are required.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
SEEDS=(20260905,20260912,20260913)
PARAMETERS=5977045


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def local_file(name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Bundle manifest paths must be relative')
    path=(ROOT/relative).resolve()
    if ROOT not in path.parents or not path.is_file():raise ValueError('Missing or external bundle file: '+str(name))
    return path


def verified_manifest():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if (manifest['protocol']!='portable_query_product_three_seed_inference_v1'
            or manifest['status']!='complete_actual_three_weights_after_six_model_evaluation'
            or manifest['parameters']!=PARAMETERS or manifest['inference_views']!=1
            or manifest['teacher_used_at_inference'] is not False
            or manifest['ensemble_used'] is not False
            or [e['seed'] for e in manifest['checkpoints']]!=list(SEEDS)):
        raise ValueError('Actual single-model three-seed bundle required')
    for name,digest in manifest['files_sha256'].items():
        if sha(local_file(name))!=digest:raise ValueError('Bundle source or artifact hash differs: '+name)
    for entry in manifest['checkpoints']:
        if (entry['sha256']!=manifest['files_sha256'][entry['path']]
                or entry['original_checkpoint_sha256']!=entry['sha256']):
            raise ValueError('Selected checkpoint provenance differs')
    return manifest


def model_class():
    sys.path.insert(0,str(ROOT/'research/sub04_20260913'))
    from compact_query_product.model import QueryProductCompactHistoryNAF
    return QueryProductCompactHistoryNAF


def strict_state(model,state):
    expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('Compact state keys differ')
    for name,value in state.items():
        if not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype:
            raise ValueError('Compact state schema differs: '+name)
        if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite selected tensor')
    model.load_state_dict(state,strict=True)


def load_model(manifest,seed,device):
    if seed not in SEEDS:raise ValueError('Unknown original training seed')
    entry,=[e for e in manifest['checkpoints'] if e['seed']==seed]
    saved=torch.load(local_file(entry['path']),map_location='cpu',weights_only=True)
    cfg=saved['config']
    if (cfg['protocol']!='compact_query_product_fp32_half_teacher_half_GT_1500_8_v1'
            or cfg['architecture']!='compact' or cfg['original_seed']!=seed
            or cfg['updates']!=1500 or cfg['teacher_used_at_inference'] is not False
            or cfg['inference_views']!=1):raise ValueError('Original selected compact checkpoint configuration differs')
    model=model_class()(history_dropout=0.,emissivity_dropout=0.)
    strict_state(model,saved['state_dict'])
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise ValueError('Parameter count differs')
    return model.float().to(device).eval(),entry


def ops():
    spec=importlib.util.spec_from_file_location('bundle_original_repair_and_input_contract',ROOT/'portable_ops.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def predict(model,sample,device):
    operation=ops();sample=operation.validate_inputs(sample)
    with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        batch={k:torch.from_numpy(v).to(device) for k,v in sample.items()}
        output=model(**batch)
        if output.dtype!=torch.float32 or tuple(output.shape)!=sample['support'].shape:raise ValueError('One original FP32 output required')
        output=operation.repair(output.cpu().numpy(),sample['coarse'],sample['support'])
    mask=sample['support']
    if output.dtype!=np.float64 or not np.isfinite(output[mask]).all() or not np.isnan(output[~mask]).all():
        raise ValueError('Invalid original repaired observation support')
    return output


def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=20260905)
    parser.add_argument('--inputs-npz',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args()
    if args.output.suffix!='.npy':parser.error('--output must end in .npy')
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError('Use a new prediction output')
    setup();manifest=verified_manifest();manifest_sha=sha(ROOT/'manifest.json')
    with np.load(args.inputs_npz,allow_pickle=False) as archive:sample={k:np.array(archive[k],copy=True) for k in archive.files}
    sample=ops().validate_inputs(sample);model,entry=load_model(manifest,args.seed,args.device)
    started=time.perf_counter();prediction=predict(model,sample,args.device);elapsed=time.perf_counter()-started
    if sha(ROOT/'manifest.json')!=manifest_sha:raise ValueError('Bundle manifest changed during inference')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:np.save(stream,prediction,allow_pickle=False)
    receipt=dict(status='portable_single_model_prediction_complete',seed=args.seed,
        checkpoint_sha256=entry['sha256'],parameters=PARAMETERS,device=args.device,
        network_calls=1,image_forwards=1,inference_views=1,teacher_used_at_inference=False,
        ensemble_used=False,labels_opened=False,tf32=False,autocast=False,repair_dtype='float64',
        input_npz_sha256=sha(args.inputs_npz),output_sha256=sha(args.output),output_shape=list(prediction.shape),
        bundle_manifest_sha256=manifest_sha,elapsed_seconds=elapsed,timing_is_benchmark=False)
    with args.output.with_suffix('.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
