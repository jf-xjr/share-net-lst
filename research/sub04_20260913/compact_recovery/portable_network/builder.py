"""Build a small portable bundle only after real six-weight Test/cost/readout.

The staging helper copies code for CPU isolation checks without a manifest,
weights, zip or a claim of a finished release. Only build() makes a real bundle.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,importlib.util,json,platform,shutil,sys,zipfile

HERE=Path(__file__).resolve().parent;RECOVERY=HERE.parent;NEW=RECOVERY.parent;ROOT=NEW.parents[1]
MODEL_FILES=[
 'research/sub04_20260913/compact_recovery/model.py',
 'research/sub04_20260913/multihead_fusion/model.py',
 'research/sub04_20260911/naf_history/model.py',
 'resources/historylst246/historylst/model.py',
 'resources/historylst246/historylst/__init__.py',
 'resources/historylst246/historylst/third_party/__init__.py',
]
SUPPORT_FILES=[
 'resources/historylst246/PREPROCESSING.md',
 'resources/historylst246/LICENSE',
 'resources/historylst246/provenance/optical_normalization.json',
 'resources/historylst246/provenance/weather_normalization.json',
]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def extracted_function(path,name):
    source=path.read_text();tree=ast.parse(source)
    node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    text=ast.get_source_segment(source,node)
    if ast.dump(ast.parse(text).body[0])!=ast.dump(node):raise ValueError('Extracted function AST differs')
    return text,dict(original_source=str(path.relative_to(ROOT)),original_source_sha256=sha(path),
        function=name,function_text_sha256=hashlib.sha256(text.encode()).hexdigest(),AST_identical=True)


def source_paths():
    paths=[ROOT/p for p in MODEL_FILES+SUPPORT_FILES]
    for directory in (ROOT/'research/sub04_20260911/naf_history/upstream',ROOT/'resources/historylst246/historylst/third_party/utae'):
        paths.extend(p for p in directory.iterdir() if p.is_file() and (p.suffix in ('.py','.json') or p.name=='LICENSE'))
    return sorted(set(paths))


def stage_code(output):
    """Code-only preparation: deliberately no release manifest or selected weight."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New staging directory only')
    output.mkdir(parents=True);copied={}
    for source in source_paths():
        relative=source.relative_to(ROOT);destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        if sha(source)!=sha(destination):raise ValueError('Source copy changed bytes')
        copied[str(relative)]=sha(source)
    shutil.copyfile(HERE/'predict.py',output/'predict.py')
    repair,repair_receipt=extracted_function(ROOT/'resources/historylst246/historylst/metrics.py','repair')
    validate,validate_receipt=extracted_function(RECOVERY/'predict_single.py','validate_inputs')
    header='"""Verbatim original repair and input validator; see manifest provenance."""\nimport numpy as np\nINPUTS=(\'fine\',\'coarse\',\'support\',\'context\',\'emissivity\',\'history\')\n\n'
    (output/'portable_ops.py').write_text(header+repair+'\n\n'+validate+'\n')
    texture_source=ROOT/'resources/historylst246/provenance/texture_normalization.json'
    texture=json.loads(texture_source.read_text())
    texture_keys=('value_arrays','missing_rule','weighting','fit_view_sha256','scope')
    compact_texture={k:texture[k] for k in texture_keys}
    compact_texture['channels']={k:{stat:v[stat] for stat in ('mean','std')} for k,v in texture['channels'].items()}
    compact_texture['note']='Exact fitted values copied; per-scene training statistics omitted. No recomputation.'
    target=output/'resources/historylst246/provenance/texture_normalization_inference.json'
    target.write_text(json.dumps(compact_texture,indent=2,allow_nan=False)+'\n')
    input_source=ROOT/'resources/historylst246/manifest.json';original=json.loads(input_source.read_text())
    input_keys=('contract','input_keys','fine_channel_names','context_channel_names','history_channels','emissivity_channels','query_support','limitations')
    contract={k:original[k] for k in input_keys}
    (output/'input_contract.json').write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
    return dict(copied_unmodified=copied,extracted_functions=[repair_receipt,validate_receipt],
        metadata_projections=[dict(source=str(texture_source.relative_to(ROOT)),sha256=sha(texture_source),
            copied_keys=list(texture_keys)+['channels.*.mean','channels.*.std'],output=str(target.relative_to(output))),
            dict(source=str(input_source.relative_to(ROOT)),sha256=sha(input_source),copied_keys=list(input_keys),output='input_contract.json')])


def runtime_versions():
    import torch,numpy
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
        installed_distribution_metadata=dict(torch=importlib.metadata.version('torch'),numpy=importlib.metadata.version('numpy')))


def readme(manifest):
    versions=manifest['recorded_environment'];metrics=manifest['evaluation']['mean_metrics']
    return f'''# Compact historical LST network: three independently selected seeds

This bundle contains the original compact model code and exactly three selected
weights. Each prediction uses one model, one view and all nine history packets.
The teacher is used only during training. Parameter count: 5,911,525.

Unzip and run from this directory:

```bash
python predict.py --seed 20260905 --inputs-npz encoded_six_fields.npz --output new.npy
```

CPU is the default. Add `--device cuda` for a working PyTorch CUDA installation.
Seed 20260905 is the predeclared example, not a best-seed choice made on Test.
Other available seeds are 20260912 and 20260913. `manifest.json` verifies only
files within this bundle. No original workspace, parent checkpoints, experiment
logs, teacher or observation dataset is needed to run an already encoded NPZ.
Checkpoint metadata retain original provenance strings; inference never opens
the paths mentioned in that metadata.

Runtime dependencies are PyTorch and NumPy. Recorded working environment:
Python {versions['python']}, PyTorch {versions['torch']}, NumPy {versions['numpy']}.
`requirements.txt` names the runtime dependencies without claiming those are the
latest versions or prescribing a CUDA wheel for another machine.

The NPZ must contain exactly these original preprocessed fields, including the
leading batch dimension of one:

| Field | Shape at the evaluated grid | dtype |
|---|---|---|
| fine |1×52×160×160|float32|
| coarse |1×1×40×40|float32; NaN for missing parents|
| support |1×1×160×160|bool|
| context |1×15|float32|
| emissivity |1×4×160×160|float32|
| history |1×9×9×160×160|float32|

These are encoded model inputs, not raw TIFF files. `fine[:,:1]` is the
coarse-derived temperature in kelvin, not the fine target. History includes the
original normalized temperature/anomaly/coverage/QA/emissivity/date channels.
`input_contract.json` specifies the exact channel order and encoding. The original
preprocessing specification and fitted normalization values are retained under
`resources/historylst246/`. The texture JSON is an explicitly traced projection
of the original fitted means and standard deviations; per-scene fitting records
are omitted. This small inference bundle does
not download observations or implement a new raw-image preprocessing pipeline.
All fields except missing coarse observations must be finite. The original
query prediction support must be retained; it is not the target scoring mask.
Other spatial dimensions divisible by 8 and at least 16 are accepted by the
interface, but the reported evaluation uses 160×160.

Output is `[1,1,H,W]` kelvin in float64 after the verbatim original supported
4×4 mean repair; unsupported pixels are NaN. Forward computation is FP32,
TF32 and autocast are off. The adjacent JSON records the exact weight and inputs.

Actual consumed-Test30 macro RMSE: network {metrics['naf_history']['rmse']:.9f} K,
matched U-TAE {metrics['baseline']['rmse']:.9f} K. Goal 1 passed:
{manifest['evaluation']['goal1_pass']}; Goal 2 passed: False. Full two-goal outcome
and limitations are in `evaluation.json`; this is not a fresh independent Test.

The model files are copied unchanged with their original relative layout.
Their historical development comments do not replace the actual selected-weight
metadata. NAFNet upstream sources/license/commit are preserved in
`research/sub04_20260911/naf_history/upstream/`; U-TAE's dependency sources,
MIT license, pinned commit and task modifications are preserved in
`resources/historylst246/historylst/third_party/utae/`. U-TAE is imported by the
unchanged support-projection module; it is not a second inference model.
Upstream reference NAFNet source uses BasicSR but is included for provenance
only, and the portable inference path does not import BasicSR.
'''


def build(args):
    if args.output.exists() or args.zip.exists() or args.zip.with_suffix(args.zip.suffix+'.sha256').exists():raise FileExistsError('New final bundle directory and zip required')
    delivery=module(RECOVERY/'build_delivery.py','portable_verified_delivery_source')
    actual=delivery.build(args)
    if 'test' not in actual or 'cost' not in actual or actual['goal1_status'] not in ('passed','failed'):
        raise ValueError('Real six-weight freeze, completed Test and actual common cost are required')
    given=json.loads(args.delivery.read_text())
    for key in ('freeze','selected_family','checkpoints','test','cost','goal1_status','goal1_pass','goal2','goal2_status','goal2_pass','both_goals_pass'):
        if given[key]!=actual[key]:raise ValueError('Existing final readout does not match verified actual evidence: '+key)
    if actual['selected_family']!='compact':raise ValueError('This fixed portable bundle supports compact only')
    entries=[e for e in actual['checkpoints'] if e['architecture']=='naf_history']
    if [e['seed'] for e in entries]!=[20260905,20260912,20260913]:raise ValueError('Three actual selected compact seeds required')
    import torch
    torch.set_num_threads(1)
    for entry in entries:
        model=delivery.reader.loader.load_model(entry,'cpu')
        if sum(p.numel() for p in model.parameters())!=5911525:raise ValueError('Wrong compact parameter count')
        saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
        if 'optimizer' in saved or 'scaler' in saved or 'ema' in saved:raise ValueError('Selected best only, no optimizer or alternate weights')
        del model,saved
    provenance=stage_code(args.output);weights=[]
    for entry in entries:
        destination=args.output/'weights'/f"compact_{entry['seed']}.pt";destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(entry['checkpoint'],destination)
        if sha(destination)!=entry['checkpoint_sha256']:raise ValueError('Selected weight copy changed bytes')
        saved=torch.load(destination,map_location='cpu',weights_only=True)
        weights.append(dict(seed=entry['seed'],path=str(destination.relative_to(args.output)),sha256=sha(destination),
            original_checkpoint_sha256=entry['checkpoint_sha256'],selected_step=saved['step'],selected_weights=saved['weights']))
        del saved
    evaluation=dict(goal1_status=actual['goal1_status'],goal1_pass=actual['goal1_pass'],goal2_status=actual['goal2_status'],goal2_pass=False,
        both_goals_pass=False,mean_metrics=actual['test']['mean_metrics'],paired_effects=actual['test']['paired_effects'],
        parameters=5911525,user_absolute_threshold_k=actual['test']['user_absolute_threshold_k'],
        common_cost=actual['cost']['summaries'],cost_protocol=actual['cost']['protocol'],
        consumed_Test=True,independent_holdout=False,goal2_failed_fit_summary=actual['goal2']['summary'])
    manifest=dict(protocol='portable_compact_three_seed_inference_v1',status='complete_actual_three_weights_after_six_model_evaluation',
        parameters=5911525,inference_views=1,teacher_used_at_inference=False,ensemble_used=False,
        example_seed=20260905,example_seed_selected_before_Test=True,checkpoints=weights,
        recorded_environment=runtime_versions(),provenance=provenance,evaluation=evaluation,
        original_evidence_sha256=dict(freeze=sha(args.freeze),Test_results=sha(args.test_results),
            common_cost=sha(args.cost),delivery_readout=sha(args.delivery),builder=sha(__file__),
            portable_predictor=sha(HERE/'predict.py'),delivery_verifier=sha(RECOVERY/'build_delivery.py')))
    (args.output/'evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False)+'\n')
    (args.output/'requirements.txt').write_text('numpy\ntorch\n')
    (args.output/'README.md').write_text(readme(manifest))
    manifest['files_sha256']={str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    args.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():archive.write(path,arcname=str(Path(args.output.name)/path.relative_to(args.output)))
    with args.zip.with_suffix(args.zip.suffix+'.sha256').open('x') as stream:stream.write(sha(args.zip)+'  '+args.zip.name+'\n')
    print(json.dumps(dict(status='actual_portable_three_seed_bundle_complete',output=str(args.output),zip=str(args.zip),zip_sha256=sha(args.zip),bytes=args.zip.stat().st_size,
        manifest_sha256=sha(args.output/'manifest.json'),goal1_pass=actual['goal1_pass'],goal2_pass=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','test-results','cost','delivery','output','zip'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('freeze','test_results','cost','delivery','output','zip'):setattr(args,name,getattr(args,name).resolve())
    build(args)
