"""Seed913 uses the separately authorized final-reserve deadline; numerical code unchanged."""
from pathlib import Path
import argparse,ast,copy,importlib.util,json,time
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sealed_QP_confirmation_runner',HERE/'continue_confirm.py')
confirmation=importlib.util.module_from_spec(spec);spec.loader.exec_module(confirmation)
base=confirmation.base;ALLOCATION=HERE/'final_evaluation_reserve_allocation.json';PRIOR=HERE/'runs/compact_20260912/complete.json'


def prepare():
    base.require(base.sha(HERE/'continue_confirm.py')=='73d6b36a864f53131c604b05341a8166d03fe7cd359f1a7f865f65d7d7605d39'
        and base.sha(HERE/'train.py')=='d0c6fc43803cc9fce23c708dbb1d29970c993d7ba2f07884bd341053fd8ec7b4',
        'Original numerical and confirmation sources must remain unchanged')
    a=base.read(ALLOCATION);old=base.read(confirmation.AUTH)
    base.require(a['status']=='explicit_root_reallocation_of_final_evaluation_reserve' and a['budget_added_seconds']==0
        and a['previous_confirmation_allocation']==base.binding(confirmation.AUTH)
        and a['hard_deadline_unix']==old['hard_deadline_unix']==1789302396
        and a['effective_training_stop_unix']==1789301796 and a['maximum_job_seconds']==700
        and a['original_pilot_gate_pass'] is False and a['final_numeric_gates_unchanged'] is True
        and a['no_Test_weight_or_method_selection'] is True,'Actual unchanged-hard-budget tail allocation required')
    tree=ast.parse((HERE/'continue_confirm.py').read_text())
    function,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authority']
    replacement=copy.deepcopy(function);replacement.name='_authority_with_tail_clock';count=0
    # Preserve all previous allocation, SAM, parent and completion checks. Only
    # replace the final current-time comparison against the old training cutoff.
    for statement in replacement.body:
        if (isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call)
            and statement.value.args and isinstance(statement.value.args[0],ast.Compare)
            and ast.dump(statement.value.args[0].left)==ast.dump(ast.parse('time.time()',mode='eval').body)):
            statement.value.args[0]=ast.parse("time.time()<_tail_allocation['effective_training_stop_unix']",mode='eval').body
            count+=1
    base.require(count==1,'Exactly one old current-time deadline guard may change')
    confirmation._tail_allocation=a
    exec(compile(ast.fix_missing_locations(ast.Module(body=[replacement],type_ignores=[])),str(HERE/'continue_confirm.py'),'exec'),confirmation.__dict__)
    return a


def run(args):
    a=prepare();base.require(args.original_seed==20260913,'This supplemental runner is only for913')
    def authority(current):
        value=confirmation._authority_with_tail_clock(current)
        # Above original authority already verifies actual912 complete1500/8.
        prior=base.read(PRIOR)
        base.require(prior['status']=='complete' and prior['original_seed']==20260912
            and prior['updates']==1500 and prior['validation_weight_candidates']==8
            and not (PRIOR.parent/'interrupted.json').exists(),'Actual successful912 precedes913')
        return dict(value,training_stop_unix=a['effective_training_stop_unix'])
    confirmation.authority=authority
    old_seal=base.source_seal
    base.source_seal=lambda:dict(old_seal(),**{str(p.resolve()):base.sha(p) for p in (Path(__file__),ALLOCATION,PRIOR)})
    old_dump=base.r.dump
    def dump(path,value):
        if Path(path).name in ('run.json','complete.json','interrupted.json'):
            value=dict(value,tail_allocation=base.binding(ALLOCATION),tail_runner=base.binding(__file__),
                effective_training_stop_unix=a['effective_training_stop_unix'],preceding_confirmation=base.binding(PRIOR),
                final_hard_deadline_unix=a['hard_deadline_unix'],user_budget_added_seconds=0)
        return old_dump(path,value)
    base.r.dump=dump
    confirmation.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-seed',type=int,choices=(20260913,),default=20260913)
    parser.add_argument('--architecture',choices=('compact',),default='compact');parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:prepare();print(json.dumps(dict(status='CPU_one_tail_clock_substitution_and_bound_sources_PASS',GPU_used=False,Test_opened=False)));raise SystemExit(0)
    if args.output is None:parser.error('Actual new913 output required')
    args.output=args.output.resolve();run(args)
