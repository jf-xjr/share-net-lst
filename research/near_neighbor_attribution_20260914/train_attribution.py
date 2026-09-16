"""Fresh paired training using the established portable task optimizer/loader."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
from pathlib import Path
import argparse, importlib.util, json, sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'resources/historylst246'
sys.path[:0]=[str(HERE),str(PACKAGE)]
spec=importlib.util.spec_from_file_location('attribution_portable_runner',PACKAGE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from attribution_models import construct


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train'])
    parser.add_argument('--variant',choices=['learned','coverage'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=dict(architecture=args.variant,seed=args.seed,updates=12000,batch_size=4,
             lr=.001,warmup=300,weight_decay=.0001,validation_interval=1000,
             evaluation_batch=2,history_dropout=0.,emissivity_dropout=0.,
             selection='minimum FP32 Val45 macro RMSE; 26 raw/EMA opportunities',
             teacher=False,training_crop=128,
             initialization='fresh shared random tensors within each paired seed')
    args.device='cuda';args.root=PACKAGE
    args.output=HERE/('smoke_crop128' if args.mode=='smoke' else 'attribution')/f'{args.variant}_{args.seed}'
    args.output.mkdir(parents=True,exist_ok=True)
    args.config=args.output/'config.json'
    if args.config.exists():
        if json.loads(args.config.read_text())!=cfg:raise ValueError('Existing config differs')
    else:r.dump(args.config,cfg)
    r.HistoryUTAE=lambda:construct(args.variant)
    original_inference=r.inference
    r.inference=lambda model,data,device,batch_size,amp=False:original_inference(model,data,device,batch_size,amp=False)
    original_augment=r.augment
    def crop_augment(batch,code):
        batch=original_augment(batch,code)
        # A deterministic, parent-aligned training crop shared by paired arms;
        # validation and prediction always retain the complete 160 x 160 scene.
        rng=r.np.random.default_rng(r.torch.initial_seed())
        top,left=(int(x)*8 for x in rng.integers(0,5,size=2))
        return {k:(v if k=='context' else
                   v[...,top//4:top//4+32,left//4:left//4+32] if k=='coarse' else
                   v[...,top:top+128,left:left+128]).contiguous()
                for k,v in batch.items()}
    r.augment=crop_augment
    sources=[Path(__file__),HERE/'attribution_models.py',PACKAGE/'run.py',
             ROOT/'research/sub04_20260913/compact_query_product/model.py',
             ROOT/'research/sub04_20260913/compact_recovery/model.py',
             ROOT/'research/sub04_20260913/multihead_fusion/model.py',
             ROOT/'research/sub04_20260911/naf_history/model.py']
    source_hash={str(p):r.digest(p) for p in sources}
    receipt=args.output/'implementation.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=source_hash:raise ValueError('Training sources changed')
    else:r.dump(receipt,source_hash)
    r.setup('cuda')
    r.torch.backends.cuda.matmul.allow_tf32=False
    r.torch.backends.cudnn.allow_tf32=False
    r.train(args)
    if any(r.digest(p)!=h for p,h in source_hash.items()):raise ValueError('Source changed during run')


if __name__=='__main__':main()
