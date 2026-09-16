# HistoryLST246: a portable historical LST reconstruction task

The directory contains the full fixed **246-city / 738-query** task: 201 cities
(603 queries) for fitting, 15 cities (45 queries) for validation, and 30 cities
(90 queries) for evaluation. Each city is a 19.2 km square crop with three sampled
dates. The 30 evaluation cities have already been used in follow-up research;
they are not a fresh confirmatory holdout. This is a local release candidate,
not yet a publicly deposited dataset.

The question supported by the task is how methods use a common pool of past
thermal observations to reconstruct fine spatial temperatures given current
coarse temperature and ancillary predictors. It supports independent training,
inference and scoring without the original network, checkpoint or private cache.
The supplied observations are a retrospective Landsat-product 480-to-120 m
controlled experiment. Current coarse temperatures are aggregated from the
fine reference product, and current support depends on its thermal DN/QA.

## Quick start

Python 3.10, NumPy and PyTorch suffice for the learned baseline. Loading, historical
rules and external scoring need NumPy only. Copy this entire directory to any
location; the files use paths relative to the manifest and need no credentials.
There are 10.49 GB of uncompressed arrays. Install PyTorch for your hardware and
install the remaining requirements in a virtual environment.

```bash
python -I run.py predict --checkpoint models/utae_best.pt --output runs/test --device cpu --split test
python -I score.py --predictions runs/test/predictions.npy --output runs/test/scores.json
```

To train the independent learner from scratch:

```bash
python -I run.py smoke --config configs/utae.json --output runs/smoke --device cuda
python -I run.py train --config configs/utae.json --output runs/utae --device cuda
python -I run.py predict --checkpoint runs/utae/best.pt --output runs/retrained_test --device cpu --split test
```

The fixed independent U-TAE-LST uses the official U-TAE encoder, temporal attention
and decoder with spatial observation masks, two input stems, and a regression
head. It has 2,985,697 parameters and trains from scratch. Its configuration uses
18,000 updates, batch four, and 50 raw/EMA validation candidates. `--resume` resumes
the latest 750-update checkpoint. This is not a compute- or pretraining-matched
architecture ablation of the original model. Its upstream source, license and
changes are recorded in `historylst/third_party/utae/UPSTREAM.json`.

Two previously fixed, unfitted historical mosaic rules provide NumPy examples:

```bash
python -I run_rules.py --mode all9_coarse --output runs/all9.npy
python -I score.py --predictions runs/all9.npy --output runs/all9_scores.json
python -I run_rules.py --mode recent_coarse --output runs/recent.npy
```

These use locally recent valid temperatures after overlap centering and common
support projection, with coverage at least 0.75. They are simple references,
not reproductions of SED or WGAST and not the stronger trained EMIS template.

The optional original reference models include the stronger Fit-fitted EMIS
template and the matched current/history controls. Install
`requirements-reference.txt` to use their preserved implementation:

```bash
python -I run_reference.py --model template --output runs/template.npy
python -I run_reference.py --model full_history --output runs/full.npy
python -I run_reference.py --model current_only --output runs/current.npy
python -I score.py --predictions runs/template.npy --output runs/template_scores.json
```

`without_explicit` and `released_history` are also available. The matched controls
continue the same EMIS ancestor for 6,000 updates; `released_history` is the original
selected chain. Keep these lineages distinct when attributing gains. The reference
checkpoints are optional for external training and for U-TAE-LST; they supply
comparison outputs rather than inputs to new methods. Only state and architecture
metadata are packed, with original checkpoint hashes in their manifest.

The existing past-only all-pair ubESTARFM adaptation is provided separately with
the unchanged official numerical core and its MIT license. Install
`requirements-fusion.txt`, then run:

```bash
python -I run_fusion.py --output runs/fusion.npy
python -I score.py --predictions runs/fusion.npy --output runs/fusion_scores.json
```

The adaptation restores missing observations, builds historical coarse means,
combines all available reference pairs, uses coarse interpolation as fallback,
and applies the shared support correction. It consumes the same thermal source
pool but does not use optical/emissivity/context features. Its settings and source
hashes are preserved under `fusion_reference/`; it is not the original publication's
bracketing-date real-sensor experiment. No parameter or reference rule was retuned
for this release.

## Connecting another implementation

```python
from historylst.data import Dataset
from historylst.metrics import repair
data = Dataset('/path/to/historylst246', 'fit', labels=True)
batch = data.batch([0, 1, 2, 3])
observed = data.observed_history(0)  # NaN where the source has no observation

test = Dataset('/path/to/historylst246', 'test', labels=False)
# your_model receives only test.arrays, in test.records order.
# Save predictions as float32/float64 Kelvin [90, 1, 160, 160].
# Optional common projection: repair(predictions, coarse, support).
```

An external method can select or aggregate past sources from the same nine-slot
pool. Preserve the ordered query IDs, source availability, training cities and
scoring support. Report which channels it uses and any added inputs. Prediction
must not read `data/test/labels/`. `score.py` accepts any implementation's NPY;
it does not require an author checkpoint or a model-specific receipt. It reports
raw scores by default; `--project` explicitly requests projection, which is
recorded in the result. Predictions must cover every formal pixel; omitting hard
pixels or averaging only finite predictions is rejected.
The same output includes top-decile surface-hotspot IoU and reference-hotspot
MAE, with stable tie handling and the same city/region aggregation as RMSE.
Compare an external method with a supplied reference using paired whole-city
resampling within each region:

```bash
python -I compare.py --first runs/test/scores.json --second results/method_scores/full_history.json --output runs/test/comparison.json
```

The intervals describe variation across the sampled cities and retain all dates
of each resampled city; they do not measure training-seed variation. Both score
files must use the same task, formal support and aggregation.

## Arrays and metadata

Each split stores separate `inputs/` and `labels/` arrays. All arrays are ordered
by `manifest.json`, which records their shape, dtype, SHA-256, each query's identity,
city, region, UTC, grid and historical source identities/statuses.

| Input | Per-query shape | Interpretation |
|---|---|---|
| `fine` | 52 × 160 × 160 | Current predictors; first channel is interpolated coarse K |
| `coarse` | 1 × 40 × 40 | Current coarse K; NaN for an unobserved parent |
| `support` | 1 × 160 × 160 | Supplied prediction support |
| `context` | 15 | Date, platform, geometry and retrospective daily weather |
| `emissivity` | 4 × 160 × 160 | Current emissivity with its own availability |
| `history` | 9 × 9 × 160 × 160 | Past thermal, quality, availability, emissivity and time |

Labels comprise `target` (K), `valid` and `formal`. The latter is the frozen scoring
domain, not a model input. The original protocol computes each scene's metric,
then averages scenes within city, cities within region, and the three regions.
Thus macro RMSE is a mean of scene RMSEs, not the square root of pooled pixel MSE.
Dates within a city and shared scene acquisitions must not be counted as
independent cities in uncertainty estimates.

Historical channel 0 is nearest-filled normalized temperature; channel 2 records
actual thermal coverage. `observed_history` decodes valid temperature, counts,
QA and independently valid emissivity. Missing filled cells are discarded, not
promoted to observations. All thermal-empty dates are all-zero slots. Native
30 m QA layouts cannot be recovered from the supplied 120 m aggregates.

See `PREPROCESSING.md` for source selection, encoding and evaluation definitions,
`provenance/` for normalization and source metadata, and `DATA_SOURCES.md` for
provider attribution and licenses. This task does not supply historical optical
indices, real MODIS observations, or Sentinel-2 references required by the original
WGAST contract. It cannot by itself assess all-sky recovery, sensor-independent
temperature accuracy, urban–rural heat-island intensity, or health exposure.

## Completed reference results

All scores below use the same consumed 30-city/90-query evaluation cohort.

| Implementation | RMSE (K) | Hotspot IoU |
|---|---:|---:|
| Matched current only | 0.749086 | 0.655394 |
| Fitted EMIS template | 0.617840 | 0.713574 |
| History with two terms fixed to zero | 0.528565 | 0.760679 |
| Matched full history | 0.526310 | 0.762044 |
| Independent U-TAE-LST | 0.477453 | 0.786715 |
| Past-only pooled ubESTARFM | 0.921699 | 0.667710 |

For this tested task, U-TAE-LST is the recommended learned reference. Relative
to matched full history, its RMSE is lower by 0.048857 K (pointwise city 95%
interval for the reduction: 0.039558–0.058823 K), with 29 of 30 cities improving.
It has 2.986 rather than 4.252 million parameters. The same-host CPU FP32 median
forward time is 0.466 rather than 0.562 s per crop (batch one, two CPU threads).
It trains from scratch for 18,000 updates; the original comparison continues a
pretrained ancestor for 6,000. These are tested-system adoption results, not a
matched-initialization architecture effect or a ranking against all LST methods.

The same U-TAE weights give 0.747871 K with all history zero and 0.507703 K with
seven visible slots. All three access predictions were sealed before scoring.
The original selected release gives 0.513115 K under a different training lineage
and is retained separately in `results/method_scores/released_history.json`.

`results/training/` preserves the complete 50-candidate validation trajectory,
configuration and runtime. `results/acceptance/` records relocation checks: the
independent implementation completed Fit-only training, label-free inference and
generic scoring while a Python file-open audit prohibited original-project reads.
The archive includes per-scene/city/region results and paired-comparison code.
`RELEASE_FILES.json` lists all distributed files and their hashes.
