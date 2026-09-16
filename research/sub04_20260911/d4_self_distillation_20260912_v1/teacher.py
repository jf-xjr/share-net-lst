"""One immutable Fit603 eight-view teacher; single-forward supervised KD."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
ALPHA_SHA='7e2c2fa8292054a21a90c7b35902850969f9ac4e55a2fde325800cf6f21c4309'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def teacher_receipt(path,package):
    receipt=json.loads(Path(path).read_text());manifest=json.loads((package/'manifest.json').read_text())
    require(receipt['status']=='complete_Fit603_D4_teacher_before_labels' and receipt['split']=='fit'
        and receipt['architecture']=='naf_history' and receipt['seed']==20260905 and receipt['alpha']==.25
        and receipt['checkpoint']['sha256']==ALPHA_SHA and sha(receipt['checkpoint']['path'])==ALPHA_SHA,
        'Actual fixed alpha25 NAF905 Fit teacher required')
    require(receipt['views']==list(range(8)) and receipt['aggregation']=='registered_inverse_then_float64_equal_mean_then_original_support_repair'
        and receipt['device']=='cuda' and receipt['fp32'] is True and receipt['tf32'] is False
        and receipt['batch']==2 and receipt['repair_dtype']=='float64','Exact fixed eight-view FP32 teacher protocol required')
    require(all(receipt[k] is False for k in ('labels_opened','fit_labels_opened','validation_arrays_opened','test_opened')),
        'Teacher generation must be label-free and Fit-input-only')
    require(receipt['manifest_sha256']==sha(package/'manifest.json') and receipt['scene_ids']==[r['scene_id'] for r in manifest['roles']['fit']['scenes']]
        and len(receipt['scene_ids'])==603,'Complete ordered original Fit603 required')
    for key in ('fine','coarse','support','context','emissivity','history'):
        field=manifest['roles']['fit']['fields'][key];bound=receipt['input_bindings'][key]
        require(Path(bound['path']).resolve()==(package/field['path']).resolve() and bound['sha256']==field['sha256'],'Teacher input identity differs')
    counts=receipt['counters']
    require(counts['model_forward_calls']==2416 and counts['image_forwards']==4824 and counts['historical_source_encodings']==43416,
        'All eight actual teacher views over Fit603 required')
    item=receipt['teacher'];require(item['shape']==[603,1,160,160] and item['dtype']=='float64' and sha(item['path'])==item['sha256'],'Teacher cache changed/incomplete')
    for p,h in receipt['source_sha256'].items():require(sha(p)==h,'Teacher producer changed: '+p)
    return receipt

def mixed_loss(prediction,batch,runner):
    """Mask before squaring: off-support NaNs never enter the loss/gradient."""
    ground_truth=runner.loss_fn(prediction,batch)
    mask=batch['support'].bool()
    error=torch.where(mask,prediction.float(),0.)-torch.where(mask,batch['teacher'].float(),0.)
    mse=error.square().sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)
    distilled=(mse+1e-6).sqrt().mean()
    return .5*ground_truth+.5*distilled,ground_truth,distilled
