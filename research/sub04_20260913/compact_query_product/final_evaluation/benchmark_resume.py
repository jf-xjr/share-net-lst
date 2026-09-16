"""One engineering retry: equal source digests merge without duplicate kwargs.

The sealed evaluator and all timing/data/weight/metric operations remain intact.
Use a new output directory; the failed original invocation is not overwritten.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'run.py'
ORIGINAL_SHA='92ab5584855d9f0f6fa99e3cf69a2458f5b266f8ea924ec03c7cecdf23e68663'

def merge_sources(*tables):
    merged={}
    for table in tables:
        for path,digest in table.items():
            if path in merged and merged[path]!=digest:
                raise ValueError('Conflicting source digest: '+path)
            merged[path]=digest
    return merged

def prepare():
    spec=importlib.util.spec_from_file_location('unchanged_QP_final_benchmark_entry',ORIGINAL)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    base.reader.require(base.reader.sha(ORIGINAL)==ORIGINAL_SHA,'Original sealed evaluator changed')
    tree=ast.parse(ORIGINAL.read_text())
    original,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='benchmark']
    transformed=copy.deepcopy(original);changes=[]
    expected=ast.parse("dict(core.v1.source_seal(),**reader.source_seal(),**{str(Path(core.__file__).resolve()):reader.sha(core.__file__)})",mode='eval').body
    for node in ast.walk(transformed):
        if isinstance(node,ast.FunctionDef) and node.name=='seal':
            base.reader.require(len(node.body)==1 and isinstance(node.body[0],ast.Return)
                and ast.dump(node.body[0].value)==ast.dump(expected),'Only original duplicate-key seal expression may change')
            node.body[0].value=ast.parse("_resume_merge_sources(core.v1.source_seal(),reader.source_seal(),{str(Path(core.__file__).resolve()):reader.sha(core.__file__)},_resume_source_binding)",mode='eval').body
            changes.append('nested seal return: equal-digest map merge plus wrapper source')
    base.reader.require(len(changes)==1,'Exactly one nested source-seal expression must change')
    base._resume_merge_sources=merge_sources
    base._resume_source_binding={str(Path(__file__).resolve()):base.reader.sha(__file__)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[transformed],type_ignores=[])),str(ORIGINAL),'exec'),base.__dict__)
    return base,dict(original_source_sha256=ORIGINAL_SHA,wrapper_source_sha256=base.reader.sha(__file__),
        changed_expressions=changes,original_benchmark_AST_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        adapted_benchmark_AST_sha256=hashlib.sha256(ast.dump(transformed).encode()).hexdigest(),
        data_numerical_timing_protocol_changed=False,Test_repeated=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();base,adaptation=prepare()
    if args.check:
        assert merge_sources({'a':'same'},{'a':'same','b':'new'})=={'a':'same','b':'new'}
        try:merge_sources({'a':'one'},{'a':'two'})
        except ValueError:pass
        else:raise AssertionError('Conflicting source digest must reject')
        assert not base.reader.torch.cuda.is_initialized()
        print(json.dumps(dict(status='CPU_one_expression_same_digest_merge_PASS',adaptation=adaptation,GPU_used=False,observations_opened=False)));return
    if args.freeze is None or args.output is None:parser.error('Actual same --freeze and new --output required')
    args.freeze=args.freeze.resolve();args.output=args.output.resolve()
    base.reader.require(not args.output.exists(),'New benchmark output required; retain original failure')
    print(json.dumps(dict(status='one_source_merge_engineering_retry',adaptation=adaptation,output=str(args.output))),flush=True)
    base.benchmark(args)

if __name__=='__main__':main()
