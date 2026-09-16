"""Actual parent weights plus small synthetic inputs; no dataset or GPU access."""
from pathlib import Path
import hashlib,json,sys
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;NEW=HERE.parent
sys.path.insert(0,str(NEW));sys.dont_write_bytecode=True
from compact_query_product.model import QueryProductCompactHistoryNAF,load_compact_parent_state,ADDED_KEYS
from compact_recovery.model import CompactFourHeadHistoryNAF

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);torch.manual_seed(20260953)
    checkpoint=NEW/'compact_recovery/runs/compact_20260905/best.pt'
    expected='21432eb1bcf43b787e3fd8df9d198d3b21944bd47b9d53e825b73ecfc5f536ee'
    assert sha(checkpoint)==expected
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    parent=CompactFourHeadHistoryNAF(history_dropout=0.,emissivity_dropout=0.)
    parent.load_state_dict(saved['state_dict'],strict=True)
    model=QueryProductCompactHistoryNAF(history_dropout=0.,emissivity_dropout=0.)
    loading=load_compact_parent_state(model,saved['state_dict'])
    assert sum(p.numel() for p in model.parameters())==5977045
    assert sum(p.numel() for n,p in model.named_parameters() if n in ADDED_KEYS)==65520
    size=32;fine=torch.randn(1,52,size,size)*.05;fine[:,0]=300+torch.randn(1,size,size)
    support=torch.ones(1,1,size,size,dtype=torch.bool);support[:,:,:2,:2]=False
    coarse=F.avg_pool2d(fine[:,:1]*support,4)/F.avg_pool2d(support.float(),4)
    history=torch.randn(1,9,9,size,size)*.1;history[:,:,2]=torch.rand(1,9,size,size);history[:,:,5]=.75
    history[:,:,2,:,:4]=0.;emissivity=torch.randn(1,4,size,size)*.03;context=torch.randn(1,15)*.05
    args=(fine,coarse,support,context,emissivity,history)
    identities={}
    for training in (False,True):
        parent.train(training);model.train(training)
        for empty in (False,True):
            inputs=(*args[:-1],torch.zeros_like(history) if empty else history)
            with torch.inference_mode():a=parent(*inputs);b=model(*inputs)
            assert torch.isfinite(a[support]).all() and torch.isfinite(b[support]).all()
            assert torch.equal(a[support],b[support])
            assert torch.equal(torch.isnan(a),torch.isnan(b))
            identities[f'training_{training}_empty_{empty}']=dict(support_output_exact=True,max_difference_k=float((a[support]-b[support]).abs().max()))
    model.train();model.zero_grad(set_to_none=True);out=model(*args)
    probe=torch.randn_like(out);loss=(out[support]*probe[support]).mean();loss.backward()
    gradients={}
    for name,p in model.named_parameters():
        if name in ADDED_KEYS:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name
            gradients[name]=float(p.grad.norm())
    own=model.state_dict();model.load_state_dict(own,strict=True)
    invalid=dict(own);invalid.pop(ADDED_KEYS[0])
    try:model.load_state_dict(invalid,strict=True)
    except ValueError:pass
    else:raise AssertionError('Missing query tensor accepted')
    invalid=dict(own);invalid[ADDED_KEYS[0]]=invalid[ADDED_KEYS[0]].double()
    try:model.load_state_dict(invalid,strict=True)
    except ValueError:pass
    else:raise AssertionError('Wrong query dtype accepted')
    assert not torch.cuda.is_initialized()
    result=dict(status='actual_compact905_parent_small_CPU_identity_gradients_PASS',parent_checkpoint=dict(path=str(checkpoint),sha256=expected),
        input_size=32,synthetic_inputs=True,observations_opened=False,GPU_used=False,loading=loading,identities=identities,first_backward_query_gradient_norms=gradients,
        selected_missing_key_and_dtype_rejected=True,source_sha256={str(HERE/n):sha(HERE/n) for n in ('model.py','check_model.py')})
    with (HERE/'CPU_checks.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))

if __name__=='__main__':main()
