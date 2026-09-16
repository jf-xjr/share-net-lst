# Reproduce the main SHaRe-Net result

This guide takes you from the downloaded release to the paper's **0.426132 K
mean test RMSE**. It assumes basic remote-sensing knowledge and familiarity with
running terminal commands. You can use an AI assistant to help with the setup;
the commands, expected files and score definitions below provide the reference.

The task reconstructs a 120 m temperature grid from supplied 480 m temperatures,
current ancillary predictors and nine slots of past observations. Each scene is
160 × 160 study cells. This is a controlled, retrospective Landsat-product
reconstruction experiment; the processed arrays already contain the inputs,
reference temperatures and fixed partitions.

## 1. Choose what to reproduce

| Aim | Required assets | What this release supports |
|---|---|---|
| Recompute the final model's main test score | Source tree, test-data part and selected-checkpoints part | Follow Sections 2–6; CPU or CUDA inference and scoring have portable entry points |
| Inspect the method and existing experiments | Source tree only | Read the manuscript, [configuration](configuration.md), [training explanation](training.md) and preserved result records |
| Train the final SHaRe-Net system again from random initialization | All three data splits, five training stages, two generated teacher fields and new run records | The settings and historical implementations are provided; additional runner adaptation and intermediate-artifact generation are needed, as detailed in Section 8 |

Recomputing the main score uses three already trained networks independently.
It requires neither teacher fields nor the fit and validation arrays. Training
from scratch is a separate, substantially longer experiment.

## 2. Obtain the source and prepare Python

The GitHub repository and the Google Drive dataset folder are currently private.
Obtain repository access from its owner and request dataset access through the
[Drive folder](https://drive.google.com/drive/folders/1xzp6lgvS1q4fS1Dcqmya80vaH8-_1--s).
The two permissions are separate. Sign in with the accounts that received access.

With Git and GitHub CLI installed, clone the repository:

```bash
gh auth login
gh repo clone jf-xjr/share-net-lst
cd share-net-lst
```

Alternatively, extract the source ZIP attached to the
[manuscript release](https://github.com/jf-xjr/share-net-lst/releases/tag/v1.0.0-manuscript)
and enter its `share-net-lst` directory. All subsequent commands run from that
directory, which contains `README.md`, `release-assets.json` and `scripts/`.

The shell examples below use Bash on Linux, macOS or WSL. Use Python 3.10 or
later, with NumPy ≥1.24 and PyTorch ≥2.3 as specified in
[requirements.txt](../requirements.txt). Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sys, numpy, torch; print(sys.version); print('NumPy', numpy.__version__); print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On native Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and
run the per-seed commands individually instead of the Bash loops. CPU inference
works without CUDA. For NVIDIA GPU inference, use a PyTorch installation
compatible with your driver; follow the
[official installation selector](https://pytorch.org/get-started/locally/) if
the check above reports `False`. Use `--device cuda` only when it reports `True`.

The main-score route downloads about 0.95 GB and restores about 1.85 GB of data
and checkpoints. Reserve at least 4 GB for these artifacts and predictions,
in addition to the source tree and Python environment. The scripts process one
scene at a time and use memory-mapped arrays. Runtime and peak RAM/VRAM depend
on hardware; this guide does not establish a minimum GPU memory requirement.

## 3. Download and restore the two required parts

Create a download directory:

```bash
mkdir -p downloads
```

Download
[historylst246-test.tar.gz.part001](https://drive.google.com/file/d/1WvYbTuWGz5vLK3NbANMOOo0ZyY78TmFK/view)
from Drive into `downloads/`, retaining the exact filename. This part is
433,481,225 bytes. Then download the checkpoint part from GitHub:

```bash
gh release download v1.0.0-manuscript --repo jf-xjr/share-net-lst \
  --dir downloads --pattern 'selected-checkpoints.tar.gz.part*'
```

You can also download that attachment through the release page. Its filename
is `selected-checkpoints.tar.gz.part001` and its size is 518,527,934 bytes.
Although only the three final compact checkpoints are needed for this route,
the distributed checkpoint part also contains the selected comparison models.

Restore both groups:

```bash
python scripts/restore_assets.py --assets downloads \
  --groups historylst246-test selected-checkpoints
```

This command checks the size and SHA-256 of each part, checks the SHA-256 of
every file inside it, and restores original relative paths. Successful output
includes `historylst246-test: 9 files verified and restored` and
`selected-checkpoints: 17 files verified and restored`. Identical files are
accepted if you repeat the command; different existing files are not overwritten.
An optional `--verify-only` checks the archives and their contents without
extracting or checking an already restored directory.

The restored files include:

```text
resources/historylst246/data/test/inputs/{fine,coarse,support,context,emissivity,history}.npy
resources/historylst246/data/test/labels/{target,formal,valid}.npy
research/sub04_20260913/compact_query_product/runs/compact_20260905/best.pt
research/sub04_20260913/compact_query_product/runs/compact_20260912/best.pt
research/sub04_20260913/compact_query_product/runs/compact_20260913/best.pt
```

Braces above abbreviate separate filenames. The checked identities are recorded
in [release-assets.json](../release-assets.json). Do not normalize or reorder
these arrays again. For the additional fit and validation parts, see the
[complete download table](release.md#download-and-restore).

## 4. Validate the selected model weights

Run the lightweight checkpoint checks before inference:

```bash
for seed in 20260905 20260912 20260913; do
  python scripts/predict_compact.py --check --seed "$seed" || break
done
```

Each check should report `parameters: 5977045`, `strict_loading: "passed"` and
`predictions_generated: false`. It verifies the selected checkpoint hash and
loads all model tensors. It does not open data, execute a forward pass or measure
accuracy. Here `--seed` chooses one retained trained model; it does not train a
new model or choose the inference noise.

## 5. Predict and score all three seeds

Start with CPU for the most portable route. Change `cpu` to `cuda` in the loop
if CUDA is available. Keep the same device for all three models when making a
comparison.

```bash
for seed in 20260905 20260912 20260913; do
  python scripts/predict_compact.py --seed "$seed" --device cpu --split test \
    --output "outputs/reproduction/compact_${seed}_test.npy" || break
  python resources/historylst246/score.py --split test \
    --predictions "outputs/reproduction/compact_${seed}_test.npy" \
    --output "outputs/reproduction/compact_${seed}_test_scores.json" || break
done
```

Inference reports progress every 10 scenes, ending at `90/90 scenes`. Each model
processes one complete scene per forward pass, using FP32 with TF32 disabled,
then applies the existing coarse-mean correction in FP64. The result is a
float64 temperature array in kelvin with shape `[90, 1, 160, 160]`, ordered by
the test scene IDs in the manifest. Values outside supplied observation support
are NaN by design.

For each seed, the output directory contains a prediction `.npy`, an inference
record with the same stem and `.json` extension, and a `_scores.json` file. The
inference record includes `scene_count: 90`, `labels_opened: false`, `views: 1`,
checkpoint and prediction hashes, and ordered scene IDs. The scorer opens the
reference labels separately. Its output includes scene, city, region and macro
metrics, hotspot metrics and `consistency_max_abs_k`, the largest remaining
coarse-temperature discrepancy in kelvin. That discrepancy should be near
floating-point precision after the common correction.

Use the scorer as shown, without `--project`: the prediction command already
applies the correction. The default CPU thread count is four; `--threads 2`,
for example, changes inference CPU parallelism without changing the protocol.
Both commands refuse to overwrite completed output files. For a second run,
choose a new output directory such as `outputs/reproduction_cuda/`.

## 6. Compare with the paper

The paper's result is the **arithmetic mean of the three model scores**. For
each model, the scorer first computes RMSE over each scene's supplied urban
evaluation mask, averages its three scenes within each city, averages cities
within each region, and then averages the three regions. This is not a pooled
pixel RMSE. Read `macro.rmse` from each score file and average those values:

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean

scores = []
for seed in (20260905, 20260912, 20260913):
    path = Path(f"outputs/reproduction/compact_{seed}_test_scores.json")
    result = json.loads(path.read_text())
    assert result["evaluation"]["split"] == "test"
    assert len(result["scenes"]) == 90 and len(result["cities"]) == 30
    rmse = result["macro"]["rmse"]
    scores.append(rmse)
    print(f"{seed}: {rmse:.9f} K")

reference_path = Path("research/jstars_20260914/experiments/revision_evidence/scores.json")
reference = json.loads(reference_path.read_text())["final_0.426"]["macro"]["rmse"]
reproduced = mean(scores)
print(f"Mean of three model scores: {reproduced:.9f} K")
print(f"Recorded paper result:     {reference:.9f} K")
print(f"Difference:                {reproduced - reference:+.9f} K")
PY
```

The recorded value is **0.42613199570471244 K**, displayed as **0.426132 K** in
the repository and 0.426 K at the paper's three-decimal precision. Averaging the
three prediction arrays before scoring creates an ensemble experiment and does
not reproduce this statistic. A single seed need not equal the three-seed mean.

For diagnosis, the original per-model results are:

| Initial seed | Recorded test macro RMSE (K) |
|---|---:|
| 20260905 | 0.425812393 |
| 20260912 | 0.427238413 |
| 20260913 | 0.425345181 |

Exact prediction hashes can vary across CPU/GPU and PyTorch numerical kernels.
Keep the printed environment versions and inference records when reporting a
difference; the released data and checkpoint hashes should match exactly.
If the RMSE difference changes the paper's displayed precision, inspect the
split, model identity, aggregation and device before attributing it to hardware.

The test cohort contains 30 cities and 90 scenes and was previously used in
follow-up development. Repeating this evaluation verifies the recorded
retrospective result. The existing [evaluation history](evaluation_history.md)
explains its interpretation and the limits of independent generalization claims.

## 7. Common problems

| Symptom | Check or next step |
|---|---|
| GitHub returns 404, or Drive requests access | Confirm that the signed-in account has access to that service; repository permission does not grant Drive permission |
| `python` is not found or NumPy/PyTorch cannot be imported | Activate `.venv`; use `python -m pip` from that same environment. Before creating it, use the available `python3` executable |
| A part is missing or its checksum differs | Check its exact filename, size and download completion. Re-download the indicated part; an HTML access page is not the archive |
| `Refusing to overwrite different content` during restoration | Use a clean source copy or move the conflicting local file aside after inspecting it. Keep the manifest hashes unchanged |
| `--check` cannot find `best.pt` | Restore `selected-checkpoints` into the repository root. For an intentionally relocated selected checkpoint, pass `--checkpoint /absolute/path/best.pt` with its matching `--seed` |
| Inference cannot find arrays | Restore `historylst246-test`. If using a relocated dataset, `--data` must point to the directory containing `manifest.json`; give the scorer that same directory with `--root` |
| CUDA is unavailable or runs out of memory | Use `--device cpu` and a new output path. The portable entry point already processes one scene at a time |
| `FileExistsError` for a prediction, score or `.partial` file | Use a new output prefix. An interrupted `.partial` file is not a completed prediction and has no automatic resume path |
| Prediction shape or nonfinite scoring error | Preserve all 90 scenes in manifest order and produce kelvin predictions at all `formal` pixels. NaNs outside `support` are expected |
| A plausible score does not match 0.426132 K | Check that all three final compact seeds were scored on `test` and their `macro.rmse` values were averaged. The independent U-TAE in the dataset directory is a different model |

## 8. Training the final system from scratch

The current release includes the complete processed dataset, final selected
weights, historical training code, configuration records and artifact hashes.
It does **not** provide a single portable command that recreates all five stages
of the final SHaRe-Net system. Teacher arrays, intermediate training checkpoints
and optimizer states are excluded from the download, and several original
runners enforce expired job deadlines and paths tied to the original workspace.
Some also verify original checkpoint, source and run-record hashes.

The scientific training sequence is recoverable from these records:

| Order | Operation and dependency | Existing implementation / record |
|---|---|---|
| 1 | Train each initial network for 18,000 reference-only updates using seeds 20260905, 20260912 and 20260913 | [Initial trainer](../research/sub04_20260911/train.py); each lineage's `reference` stage |
| 2 | Build the first fixed teacher from the initial 20260905 model, then refine all three initial networks for 1,000 updates using it | [Weight interpolation](../research/sub04_20260911/weight_interpolation_20260912.py), [first teacher producer](../research/sub04_20260911/d4_fit_teacher_cache_20260912.py), [first refinement](../research/sub04_20260911/d4_self_distillation_20260912_v1/train.py); `first_teacher` stages |
| 3 | Build the second fixed teacher from all three networks after stage 2, then use it for 3,000 updates with four attention heads | [Second teacher producer](../research/sub04_20260913/strong_teacher/run.py), [four-head trainer](../research/sub04_20260913/multihead_fusion/train.py), [confirmation adapter](../research/sub04_20260913/final_confirmation/confirm_train.py); `four_heads` stages |
| 4 | Transfer selected blocks into the smaller model and train for 6,000 updates with the second teacher and decreasing teacher weight | [Compact transfer](../research/sub04_20260913/compact_recovery/model.py), [trainer](../research/sub04_20260913/compact_recovery/train.py); `compact_recovery` stages |
| 5 | Add zero-initialized query projections and refine for 1,500 updates with the second teacher | [Final transfer/model](../research/sub04_20260913/compact_query_product/model.py), [trainer](../research/sub04_20260913/compact_query_product/train.py); `query_refinement` stages |

[training_lineage.json](training_lineage.json) contains three `lineages`, each
with an `initialization_seed` and five `stages`. Every stage has its
`configuration`, `run_record`, `completed_updates`, `checkpoint` and hashes.
The top-level `teachers` entries identify the two cached fields and their
producers' receipts. Use each stage's recorded seed and configuration: later
stages need not reuse the initial random seed. The
[configuration table](configuration.md#final-system-training--section-iii-e)
provides the shared learning rates, teacher weights and validation budgets, and
[training.md](training.md) explains block transfers and optimizer/EMA resets.

For the first teacher, interpolate the validation-selected initial 20260905
weights (0.75) with that run's final EMA weights (0.25), retaining selected
buffers, then average eight inverse-transformed spatial views. The second
teacher averages eight views from each of the three stage-2 models. Both are
fixed float64 fields for all 603 fit scenes, projected to the supplied coarse
means. They are generated once and shared across the three training sequences.
They are not needed when evaluating the final models.

A new complete training run therefore needs the following preparation:

1. Restore all 27 arrays from the five data parts. Keep the 603/45/90 scene
   partition and the supplied `support` and `formal` masks. Model selection uses
   validation; test labels are used only by the final scorer.
2. Create new output directories and portable runner/configuration copies that
   resolve paths in your checkout and use current job limits. Preserve the
   originals as records of the reported runs. Retain data-isolation checks and
   record the new source/configuration/artifact hashes for the new experiment.
3. Retain stage-1 selected and final EMA weights to generate the first teacher;
   retain all stage-2 selected weights to generate the second. The distributed
   final weights cannot substitute for those earlier models. Regenerate new
   teacher fields and their metadata, or obtain the original intermediate
   artifacts identified in the lineage record if exact artifact replay is needed.
4. Port and check initialization transfers, scene sampling, spatial transforms,
   teacher generation, validation selection and each stage's precision settings
   before starting long jobs. Spatial transforms also rotate/reflect the
   solar-azimuth sine/cosine components at `context[6:8]`.
5. Complete all five stages for each seed, then evaluate the selected final
   models once with the same scoring protocol. Record your new training results
   separately; matching the released checkpoint hashes is not an expectation
   for a new run from random initialization.

Later historical training runners explicitly use CUDA; changing only their
command-line device is insufficient to obtain a portable CPU training workflow.
The detailed files above support adaptation, but completing and validating that
adaptation remains additional reproduction work. The standalone
[HistoryLST246 README](../resources/historylst246/README.md) also provides an
independent U-TAE training workflow. That is useful for testing the data and
training pipeline, but it is a separate baseline from the final SHaRe-Net system
and from the teacher-trained U-TAE comparison in the paper.

## 9. A useful brief for an AI assistant

For the main-score route, give your assistant this repository and ask:

> Follow docs/reproduction.md through Section 6. Use the portable extraction,
> compact prediction and scoring entry points, and keep the released model and
> data identities. First check the environment, permissions, downloaded assets
> and all three checkpoint hashes. Run each final seed separately, retain its
> prediction and score records, and average the three macro.rmse values. Report
> the environment, mean RMSE and difference from the recorded paper result.

For full retraining, first ask it to inventory the missing artifacts and the
historical runner dependencies in Section 8. A valid inventory should identify
the two teachers and their earlier checkpoint dependencies, expired execution
limits, source/receipt checks, and five-stage selection and transfer rules.
Passing the portable inference checks alone does not establish that full
retraining has been reproduced.
