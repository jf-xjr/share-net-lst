"""CPU-only actual-checkpoint initialization equivalence and gradient checks."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse, json, time
import torch
from model import (FourHeadLocalHistoryFusion,FourHeadHistoryNAF,HistoryNAFReconstructor,
    LocalHistoryFusion,EXPANDED_KEYS,expand_single_head_state_dict,initialize_from_single_checkpoint,sha)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FREEZE=ROOT/'research/sub04_20260911/final_delivery_late_20260912/matched/continuation_selection_freeze.json'


def comparison(actual,expected):
    assert actual.dtype==expected.dtype==torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference=(actual-expected).abs()
    upward=(torch.nextafter(expected,torch.full_like(expected,float('inf')))-expected).abs()
    downward=(expected-torch.nextafter(expected,torch.full_like(expected,float('-inf')))).abs()
    spacing=torch.maximum(upward,downward)
    ratio=torch.where(difference==0,0.,difference/spacing)
    return dict(bitwise_equal=torch.equal(actual,expected),max_absolute_error=float(difference.max()),
        max_local_ulp_error=float(ratio.max()),within_one_ulp=bool((difference<=spacing).all()))


def local_check():
    torch.manual_seed(20260913)
    original=LocalHistoryFusion(8,16)
    with torch.no_grad():
        original.score[-1].weight.normal_(0,.2);original.score[-1].bias.fill_(.17)
    candidate=FourHeadLocalHistoryFusion(8,16)
    candidate.load_state_dict({k:(v.repeat(4,1,1,1) if k=='score.2.weight' else v.repeat(4) if k=='score.2.bias' else v.clone())
        for k,v in original.state_dict().items()},strict=True)
    current=torch.randn(2,8,8,8);history=torch.randn(2,9,16,8,8)
    metadata=torch.randn(2,9,4,8,8)
    metadata[:,:,0]=torch.rand(2,9,8,8)
    metadata[:,2,0]=0.;metadata[:,:,0,2:4,3:5]=0.
    with torch.no_grad(): a,sa=original(current,history,metadata)
    b,sb=candidate(current,history,metadata)
    values=dict(output=comparison(b.detach(),a),summary=comparison(sb.detach(),sa))
    assert values['output']['within_one_ulp'] and values['summary']['within_one_ulp']
    assert torch.equal(b[:,:,2:4,3:5],current[:,:,2:4,3:5]) and torch.equal(sb[:,:,2:4,3:5],torch.zeros_like(sb[:,:,2:4,3:5]))
    (b.mul(torch.randn_like(b)).mean()+sb.square().mean()).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in candidate.parameters())
    gradient=candidate.score[-1].weight.grad.reshape(4,-1)
    assert (gradient.abs().sum(1)>0).all()
    assert any(not torch.equal(gradient[0],gradient[i]) for i in range(1,4))
    values.update(all_gradients_finite=True,each_head_nonzero_gradient=True,
        copied_heads_can_receive_different_gradients=True,empty_pool_injection_exactly_zero=True,
        original_source_slots=9,history_channels=16,groups=4)
    return values


def synthetic_inputs():
    torch.manual_seed(20260914)
    n,h,w=1,16,16
    fine=torch.randn(n,52,h,w)*.1;fine[:,:1]=300+torch.randn(n,1,h,w)*2
    support=(torch.rand(n,1,h,w)>.2).float()
    coarse=300+torch.randn(n,1,h//4,w//4);coarse[:,:,0,0]=float('nan');support[:,:,:4,:4]=0
    history=torch.randn(n,9,9,h,w)*.2
    history[:,:,2]=torch.rand(n,9,h,w);history[:,0,2]=0.;history[:,:,2,6:8,9:11]=0.
    history[:,:,5]=torch.rand(n,9,h,w);history[:,:,8]=torch.rand(n,9,h,w)*.2
    return dict(fine=fine,coarse=coarse,support=support,context=torch.randn(n,15)*.1,
        emissivity=torch.rand(n,4,h,w),history=history)


def run(output):
    started=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if output.exists():raise FileExistsError('New CPU check output only')
    if torch.cuda.is_initialized():raise RuntimeError('This check must not initialize CUDA')
    local=local_check()
    frozen=json.loads(FREEZE.read_text())
    entry,=[e for e in frozen['checkpoints'] if (e['architecture'],e['seed'])==('naf_history',20260905)]
    source=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
    original=HistoryNAFReconstructor(history_dropout=.25,emissivity_dropout=.25).float().eval()
    original.load_state_dict(source['state_dict'],strict=True)
    model,conversion,state=initialize_from_single_checkpoint(entry['checkpoint'],entry['checkpoint_sha256'])
    model.float().eval()
    assert set(state)==set(source['state_dict'])
    assert all(torch.equal(state[k],source['state_dict'][k]) for k in state if k not in EXPANDED_KEYS)
    # Reject missing or already-expanded/mis-shaped source tensors instead of a
    # permissive strict=False load that would hide unrelated incompatibility.
    bad=dict(source['state_dict']);bad.pop('current.weight')
    try:expand_single_head_state_dict(bad,model)
    except ValueError:pass
    else:raise AssertionError('A missing non-fusion parameter was accepted')
    bad=dict(source['state_dict']);bad['fusion.0.score.2.bias']=torch.zeros(4)
    try:expand_single_head_state_dict(bad,model)
    except ValueError:pass
    else:raise AssertionError('A non-single-head source was accepted')
    bad=dict(source['state_dict']);bad['fusion.0.score.2.weight']=bad['fusion.0.score.2.weight'].double()
    try:expand_single_head_state_dict(bad,model)
    except ValueError:pass
    else:raise AssertionError('A silent score dtype conversion was accepted')
    inputs=synthetic_inputs();captured={name:[] for name in ('original','four_head')};hooks=[]
    for name,network in [('original',original),('four_head',model)]:
        for layer in network.fusion:
            hooks.append(layer.register_forward_hook(lambda module,args,result,name=name:
                captured[name].append(tuple(x.detach().clone() for x in result))))
    with torch.no_grad():before=original(**inputs);after=model(**inputs)
    for hook in hooks:hook.remove()
    final=comparison(after,before)
    scales=[dict(level=i,output=comparison(b[0],a[0]),summary=comparison(b[1],a[1]))
        for i,(a,b) in enumerate(zip(captured['original'],captured['four_head']))]
    assert len(scales)==4 and final['within_one_ulp']
    assert all(row[key]['within_one_ulp'] for row in scales for key in ('output','summary'))
    model.zero_grad(set_to_none=True)
    prediction=model(**inputs)
    target=inputs['fine'][:,:1]+torch.randn_like(prediction)*.1
    loss=(prediction-target).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradients={f'fusion.{i}':dict(per_head_norm=layer.score[-1].weight.grad.reshape(4,-1).norm(dim=1).tolist(),
        copied_heads_can_receive_different_gradients=any(not torch.equal(layer.score[-1].weight.grad[0],layer.score[-1].weight.grad[j]) for j in range(1,4)))
        for i,layer in enumerate(model.fusion)}
    assert all(all(v>0 for v in row['per_head_norm']) for row in gradients.values())
    assert all(row['copied_heads_can_receive_different_gradients'] for row in gradients.values())
    original_count=sum(x.numel() for x in original.parameters());count=sum(x.numel() for x in model.parameters())
    assert count-original_count==396
    assert torch.cuda.is_initialized() is False
    output.mkdir(parents=True,exist_ok=False)
    init=output/'init_from_d4_naf905.pt'
    config=dict(source['config'],architecture='naf_history_multihead4',source_attention_heads=4,
        initialization_protocol='only_score_last_conv_1_to_4_exact_replication')
    torch.save(dict(state_dict=state,config=config,conversion=conversion,initialization_only=True,trained=False),init)
    report=dict(status='CPU_actual_checkpoint_initialization_checks_pass',device='cpu',gpu_used=False,
        actual_observations_opened=False,test_opened=False,optimizer_updates=0,
        source_checkpoint=conversion['source_checkpoint'],source_checkpoint_sha256=conversion['source_checkpoint_sha256'],
        local_synthetic_check=local,actual_checkpoint_small_synthetic_input=dict(shape=[1,52,16,16],
            source_count=9,final_output=final,fusion_scales=scales,all_parameter_gradients_finite=True,
            head_gradients=gradients),original_parameters=original_count,four_head_parameters=count,
        added_parameters=count-original_count,conversion=conversion,
        initialization_checkpoint=dict(path=str(init.resolve()),sha256=sha(init)),
        source_sha256={str(q.resolve()):sha(q) for q in (Path(__file__),HERE/'model.py',FREEZE,ROOT/'research/sub04_20260911/naf_history/model.py')},
        source_model_unmodified=True,scientific_goal_complete=False,trained=False,seconds=time.perf_counter()-started)
    with (output/'results.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],initialization_output=final,
        original_parameters=original_count,four_head_parameters=count,added_parameters=count-original_count,
        gpu_used=False,optimizer_updates=0,seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output.resolve())
