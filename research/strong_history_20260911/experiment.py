"""Train frozen model candidates through the original U-TAE experiment loop."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PACKAGE=ROOT/'resources/historylst246'
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('original_historylst_runner',PACKAGE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def factory(name):
    if name=='baseline':
        from historylst.model import HistoryUTAE
        return HistoryUTAE
    if name=='current_query':
        from candidates.current_query import HistoryCrossAttention
        return HistoryCrossAttention
    if name=='wide':
        from candidates.network_review import HistoryWideUTAE
        return HistoryWideUTAE
    raise ValueError(name)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train','predict']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=['fit','validation','test'],default='validation')
    a=p.parse_args();a.root=PACKAGE;a.config=a.config.resolve();a.output=a.output.resolve()
    cfg=json.loads(a.config.read_text());name=cfg.get('architecture','baseline');cls=factory(name);runner.HistoryUTAE=cls
    model_file=Path(sys.modules[cls.__module__].__file__)
    receipt={'architecture':name,'config':cfg,'sources':{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),model_file,PACKAGE/'run.py']},'test_used_to_select_configuration':False}
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'implementation.json'
    if path.exists():
        if not a.resume:raise FileExistsError(path)
        assert json.loads(path.read_text())==receipt
    else:path.write_text(json.dumps(receipt,indent=2)+'\n')
    runner.setup(a.device)
    if a.mode=='predict':runner.predict(a)
    else:runner.train(a)
