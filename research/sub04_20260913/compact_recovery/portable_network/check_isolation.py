"""Code-only CPU synthetic check; does not create a release manifest or weights."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,tempfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('portable_bundle_builder',HERE/'builder.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

CHILD=r'''
from pathlib import Path
import importlib.util,json,sys
root=Path(__file__).resolve().parent;forbidden=Path(sys.argv[1]).resolve()
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=Path(args[0]).resolve()
        if path==forbidden or forbidden in path.parents:raise RuntimeError('Original project access forbidden: '+str(path))
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('isolated_portable_predictor',root/'predict.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import numpy as np
import torch
p.setup();torch.manual_seed(190)
model=p.model_class()(history_dropout=0.,emissivity_dropout=0.).float().eval()
assert sum(v.numel() for v in model.parameters())==5911525
p.strict_state(model,model.state_dict())
bad=dict(model.state_dict());bad.pop(next(iter(bad)))
try:p.strict_state(model,bad)
except ValueError:pass
else:raise AssertionError('Missing state key accepted')
try:p.verified_manifest()
except FileNotFoundError:pass
else:raise AssertionError('Code-only staging was accepted as release')
checks=[]
for size in (16,32):
    rng=np.random.default_rng(size)
    sample=dict(fine=rng.normal(0,.05,(1,52,size,size)).astype('float32'),
        coarse=np.full((1,1,size//4,size//4),300,dtype='float32'),
        support=(rng.random((1,1,size,size))>.3),context=np.zeros((1,15),dtype='float32'),
        emissivity=np.zeros((1,4,size,size),dtype='float32'),
        history=np.zeros((1,9,9,size,size),dtype='float32'))
    sample['fine'][:,:1]=300
    sample['support'][:,:,:4,:4]=False;sample['coarse'][:,:,0,0]=np.nan
    sample['history'][:,:,2]=.5;sample['history'][:,:,5]=.5
    output=p.predict(model,sample,'cpu');mask=sample['support']
    blocks=output.reshape(1,1,size//4,4,size//4,4)
    masks=mask.reshape(1,1,size//4,4,size//4,4);count=masks.sum((3,5))
    means=np.where(masks,blocks,0).sum((3,5))/np.maximum(count,1)
    selected=(count>0)&np.isfinite(sample['coarse'])
    error=float(np.max(np.abs(means[selected]-sample['coarse'][selected])))
    assert error<1e-10
    checks.append(dict(size=size,supported_mean_error_k=error,supported_finite=bool(np.isfinite(output[mask]).all()),unsupported_nan=bool(np.isnan(output[~mask]).all())))
for name,mod in tuple(sys.modules.items()):
    if name.startswith(('compact_recovery','multihead_fusion','naf_history','historylst')):
        if getattr(mod,'__file__',None):assert root in Path(mod.__file__).resolve().parents,(name,mod.__file__)
print(json.dumps(dict(status='pass_cpu_synthetic_isolated_import',parameters=5911525,checks=checks,
    original_project_reads=0,real_observations_opened=False,selected_weights_packaged=False,
    release_manifest_created=False,GPU_used=False,torch=torch.__version__,numpy=np.__version__)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(prefix='compact_portable_cpu_') as temporary:
        stage=Path(temporary)/'code';provenance=builder.stage_code(stage)
        (stage/'isolated_check.py').write_text(CHILD)
        result=subprocess.run([sys.executable,'-I','-B',str(stage/'isolated_check.py'),str(builder.ROOT)],cwd=stage,
            capture_output=True,text=True,timeout=120,check=False)
        if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
        receipt=json.loads(result.stdout.strip().splitlines()[-1]);receipt['source_provenance']=provenance
        receipt['prepared_source_sha256']={p.name:builder.sha(p) for p in (HERE/'builder.py',HERE/'predict.py',Path(__file__))}
        receipt['code_and_metadata_bytes']=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file() and p.name!='isolated_check.py')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_provenance','prepared_source_sha256')}))


if __name__=='__main__':main()
