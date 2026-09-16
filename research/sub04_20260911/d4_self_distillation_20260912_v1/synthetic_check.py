"""CPU-only masked KD, synchronized D4, and exact trainer retry tests."""
import copy
from pathlib import Path
import sys,importlib.util
import numpy as np
import torch
from teacher import mixed_loss
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'resources/historylst246'
sys.path.insert(0,str(PACKAGE))
spec=importlib.util.spec_from_file_location('kd_original_runner',PACKAGE/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
spec=importlib.util.spec_from_file_location('kd_teacher_augmentation',PACKAGE/'reference_models/code/g246_8h_augment.py');aug=importlib.util.module_from_spec(spec);spec.loader.exec_module(aug)
def check():
    torch.set_num_threads(1);torch.manual_seed(66)
    pred=torch.randn(2,1,8,8,requires_grad=True)
    support=torch.rand(2,1,8,8)>.25
    batch=dict(target=torch.randn(2,1,8,8),teacher=torch.randn(2,1,8,8),support=support,
        formal=support&(torch.rand(2,1,8,8)>.2),fine=torch.randn(2,52,8,8),coarse=torch.randn(2,1,2,2),
        history=torch.randn(2,9,9,8,8),context=torch.randn(2,15),emissivity=torch.randn(2,4,8,8))
    batch['teacher'][~support]=float('nan')
    loss,gt,kd=mixed_loss(pred,batch,runner);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    manual=[]
    for i in range(2):manual.append(((pred.detach()[i][support[i]]-batch['teacher'][i][support[i]]).square().mean()+1e-6).sqrt())
    assert torch.allclose(kd,torch.stack(manual).mean()) and torch.equal(loss,.5*gt+.5*kd)
    assert torch.equal(pred.grad[~support],torch.zeros_like(pred.grad[~support]))
    rng_state=torch.get_rng_state().clone()
    errors=[]
    for code in range(8):
        changed=runner.augment(batch,code)
        expected=aug.transform_batch({k:batch[k] for k in runner.INPUTS},code)
        for key in runner.INPUTS:assert torch.allclose(changed[key],expected[key],atol=1e-6,rtol=0,equal_nan=True)
        teacher=runner.augment({'teacher':batch['teacher'],'context':batch['context']},code)['teacher']
        assert torch.allclose(changed['teacher'],teacher,atol=0,rtol=0,equal_nan=True)
        p=runner.augment({'prediction':pred.detach(),'context':batch['context']},code)['prediction']
        augmented_loss,_,_=mixed_loss(p,changed,runner);errors.append(float(abs(augmented_loss-loss.detach())))
    assert max(errors)<1e-6 and torch.equal(rng_state,torch.get_rng_state())
    results=dict(loss_formula_exact=True,teacher_off_support_NaNs_safe=True,finite_gradients=True,
        teacher_and_labels_follow_original_D4=True,student_D4_matches_registered_teacher_vectors=True,
        maximum_D4_loss_difference=max(errors),original_mask_rng_unchanged=True)
    # Exercise exactly the actual trainer retry block on a synthetic BN network.
    # AST extraction prevents a separately rewritten helper from hiding a defect.
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parent / 'train.py').read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    retry = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'attempt')
    definition = ast.Module(body=[retry], type_ignores=[])
    code = compile(ast.fix_missing_locations(definition), '<actual_d4_kd_retry_loop>', 'exec')
    class TinyBN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.linear = torch.nn.Linear(3, 1)
        def forward(self, value):
            return self.linear(self.bn(value)).reshape(-1,1,1,1)
    model = TinyBN().train()
    reference = copy.deepcopy(model)
    inputs = torch.randn(4, 3)
    batch = {'x': inputs, 'history': torch.zeros(4,9,9,1,1), 'support':torch.ones(4,1,1,1,dtype=torch.bool),'teacher':torch.zeros(4,1,1,1)}
    with torch.no_grad():
        reference(inputs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
    scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=1024.)
    calls = [0]
    class Runner:
        @staticmethod
        def forward(net, batch):
            return net(batch['x'])
        @staticmethod
        def loss_fn(prediction, batch):
            calls[0] += 1
            loss = prediction.square().mean()
            if calls[0] == 1:
                loss.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
            return loss
    from types import SimpleNamespace
    counters = dict(amp_attempts=0, training_forward_calls=0, training_backward_calls=0,
                    attempted_forward_fine_pixels=0, attempted_history_frame_pixels=0, amp_backoffs=0)
    env = dict(mixed_loss=mixed_loss,torch=torch, model=model, optimizer=optimizer, scaler=scaler, runner=Runner,
        args=SimpleNamespace(device='cpu'), cfg={'seed': 20260921}, step=1, b=batch,
        pixels=4, counters=counters, check_deadline=lambda: None,
        before_buffers={key: value.detach().clone() for key, value in model.named_buffers()})
    exec(code, env)
    assert counters['amp_attempts'] == 2 and counters['amp_backoffs'] == 1
    assert scaler.get_scale() == 512.
    for key, value in model.named_buffers():
        assert torch.equal(value, dict(reference.named_buffers())[key]), 'BN retry updated buffers twice'
    assert model.bn.num_batches_tracked.item() == 1
    return dict(status='synthetic_cpu_checks_pass', models=results,
                actual_trainer_retry_loop_executed_on_synthetic_BN=True,
                nonfinite_gradient_retried_once=True, bn_running_buffers_equal_one_successful_forward=True,
                updates_change_weights=False, actual_data_opened=False, pretrained_weights_opened=False,
                gpu_used=False, validation_opened=False, test_opened=False, real_training_started=False)
