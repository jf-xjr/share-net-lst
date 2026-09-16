"""Finite strong-comparator experiment; model selection uses Val45 only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import argparse, copy, importlib.util, json, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT/'resources/historylst246'
sys.path[:0] = [str(HERE), str(PACKAGE)]
spec = importlib.util.spec_from_file_location('jstars_portable', PACKAGE/'run.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
from mocolsk_model import CommonMoCoLSK


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke','train','predict','score','check'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--phase', choices=['native','calibrated'], default='native')
    a = p.parse_args()
    a.root = PACKAGE; a.device = 'cuda'; a.split = 'test'
    out = HERE/'mocolsk'/a.phase
    a.output = out if a.mode=='train' else out/a.mode
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = dict(architecture='MoCoLSK published x4: width32 blocks4 stages4 guidance155',
               seed=20260916, updates=10000 if a.phase=='native' else 1500,
               batch_size=4, lr=.0001, warmup=1 if a.phase=='native' else 50,
               weight_decay=.00001 if a.phase=='native' else .0001,
               validation_interval=1000 if a.phase=='native' else 500,
               evaluation_batch=1, history_dropout=0., emissivity_dropout=0.,
               training_crop=128, teacher=False,
               loss='scene-normalized masked L1' if a.phase=='native' else 'scene-normalized masked RMSE',
               phase=a.phase, selection='minimum full-scene FP32 Val45 macro RMSE among raw and EMA')
    a.config = a.output/'config.json'
    if a.config.exists():
        assert json.loads(a.config.read_text()) == cfg
    else: r.dump(a.config, cfg)
    def construct():
        net = CommonMoCoLSK()
        if a.phase=='calibrated':
            parent = HERE/'mocolsk/native/best.pt'
            assert (parent.parent/'complete.json').exists()
            net.load_state_dict(r.torch.load(parent, map_location='cpu', weights_only=False)['state_dict'])
        return net
    r.HistoryUTAE = construct
    original_inference = r.inference
    r.inference = lambda model,data,device,batch_size,amp=False: original_inference(model,data,device,batch_size,amp=False)
    original_augment = r.augment
    def crop(batch, code):
        batch = original_augment(batch, code)
        rng = r.np.random.default_rng(r.torch.initial_seed())
        top,left = (int(x)*8 for x in rng.integers(0,5,size=2))
        return {k:(v if k=='context' else
                   v[...,top//4:top//4+32,left//4:left//4+32] if k=='coarse' else
                   v[...,top:top+128,left:left+128]).contiguous() for k,v in batch.items()}
    r.augment = crop
    if a.phase=='native':
        def loss(prediction,batch):
            error = (prediction.float()-batch['target'].float()).abs()
            mask = batch['formal'].bool()
            return (r.torch.where(mask,error,0.).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)).mean()
        r.loss_fn = loss
    sources = [Path(__file__), HERE/'mocolsk_model.py', ROOT/'code/train_mocolsk_v2.py',
               PACKAGE/'run.py', PACKAGE/'historylst/model.py']
    sources += sorted((ROOT/'baselines/PGDM/lib/moco').glob('*.py'))
    receipt = dict(source_sha256={str(x):r.digest(x) for x in sources},
                   published_architecture='https://doi.org/10.1109/TGRS.2025.3547945',
                   common_input_fields=155,
                   gradient_repair='Functional dynamic convolutions preserve B=1 forward values and allow MCWG training; per-sample generation preserves B>1 independence',
                   initializer='random' if a.phase=='native' else str(HERE/'mocolsk/native/best.pt'))
    if a.phase=='calibrated': receipt['initialization_sha256'] = r.digest(HERE/'mocolsk/native/best.pt')
    rp = a.output/'implementation.json'
    if rp.exists(): assert json.loads(rp.read_text()) == receipt
    else: r.dump(rp,receipt)
    r.setup('cuda')
    r.torch.backends.cuda.matmul.allow_tf32 = False
    r.torch.backends.cudnn.allow_tf32 = False
    if a.mode=='check':
        from mocolsk_model import auxiliary
        from thst_adapter import auxiliary as reference_auxiliary
        data = r.Dataset(PACKAGE,'fit',labels=False)
        b = r.batch(data,[0], 'cuda')
        assert r.torch.equal(auxiliary(b),reference_auxiliary(b))
        original = CommonMoCoLSK(repair_gradient=False).cuda().eval()
        patched = CommonMoCoLSK().cuda().eval()
        patched.load_state_dict(original.state_dict())
        with r.torch.no_grad():
            op=r.forward(original,b); pp=r.forward(patched,b)
        maximum_fp32_difference = float((op-pp).abs().max())
        assert r.torch.allclose(op,pp,atol=.0001,rtol=0.), maximum_fp32_difference
        # Removing a detached Parameter can change cuDNN's FP32 kernel choice.
        # Compare the modified operation itself in CPU float64 as well.
        block0=copy.deepcopy(original.backbone.bridges[0].dlsk).cpu().double()
        block1=copy.deepcopy(patched.backbone.bridges[0].dlsk).cpu().double()
        xx=r.torch.randn(1,32,16,16,dtype=r.torch.float64)
        gg=r.torch.randn_like(xx)
        with r.torch.no_grad():bb0=block0(xx,gg);bb1=block1(xx,gg)
        maximum_fp64_difference=float((bb0-bb1).abs().max())
        assert r.torch.allclose(bb0,bb1,atol=1e-12,rtol=1e-12)
        xx2=r.torch.randn_like(xx);gg2=r.torch.randn_like(gg)
        with r.torch.no_grad():
            joint=block1(r.torch.cat((xx,xx2)),r.torch.cat((gg,gg2)))
            separate=r.torch.cat((block1(xx,gg),block1(xx2,gg2)))
        batch_operation_difference=float((joint-separate).abs().max())
        assert r.torch.allclose(joint,separate,atol=1e-12,rtol=1e-12)
        patched.zero_grad(); r.forward(patched,b).square().mean().backward()
        grads={name:float(t.grad.abs().sum()) for name,t in patched.named_parameters() if 'dynamic_mlp' in name and t.grad is not None}
        assert grads and sum(grads.values())>0
        both={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in b.items()}
        with r.torch.no_grad():
            paired=r.forward(patched,both)
        assert r.torch.allclose(pp,paired[:1],atol=.0002,rtol=0.)
        batch_output_difference=float((paired[:1]-paired[1:]).abs().max())
        assert r.torch.allclose(paired[:1],paired[1:],atol=.0002,rtol=0.),batch_output_difference
        r.dump(a.output/'equivalence.json',dict(native_single_forward_maximum_difference_K=maximum_fp32_difference,
               repaired_operation_fp64_maximum_difference=maximum_fp64_difference,
               batch_operation_fp64_maximum_difference=batch_operation_difference,
               repeated_batch_fp32_maximum_difference_K=batch_output_difference,
               common_input_bitwise_equal=True, batch_independent=True,
               dynamic_mlp_nonzero_gradient_tensors=len(grads), parameters=sum(x.numel() for x in patched.parameters())))
        print(json.dumps(json.loads((a.output/'equivalence.json').read_text())),flush=True)
    elif a.mode in ('smoke','train'):r.train(a)
    elif a.mode=='predict':
        a.checkpoint=out/'best.pt';r.predict(a)
    elif a.mode=='score':
        a.output=out/'predict';r.evaluate(a)
    assert all(r.digest(path)==digest for path,digest in receipt['source_sha256'].items())

if __name__=='__main__':main()
