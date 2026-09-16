# SHaRe-Net: Scale-wise Historical Reweighting Network

Code, fixed processed data and selected models accompanying **Accurate and Efficient Urban Land Surface Temperature Reconstruction with Multiscale Historical Fusion** (IEEE JSTARS submission draft).

**SHaRe-Net** names the network's scale-wise historical source reweighting. Internal model classes and original run identifiers are retained for traceability.

Target repository: **`jf-xjr/share-net-lst`**, private. Release: **`v1.0.0-manuscript`**.

The final network reconstructs 480 m observations on a 120 m grid with 5,977,045 parameters. Its recorded mean test RMSE is 0.426132 K across three trained models. The original 246-city / 738-scene partition, trained weights and experiment results are preserved.

## Contents

| Location | Contents |
|---|---|
| [Manuscript PDF](research/jstars_20260914/manuscript/main.pdf) | Revised nine-page article; LaTeX is in the same directory |
| [Configuration](docs/configuration.md) | Five training stages, all 74 channels, history selection, QA and baseline adaptations |
| [Release guide](docs/release.md) | Download, checksums, extraction and checkpoint inventory |
| [Data manifest](resources/historylst246/manifest.json) | Fixed fit/validation/test splits, scene metadata and original array hashes |
| [Release asset manifest](release-assets.json) | Compressed parts and all restored file hashes |
| `research/` and `resources/` | Preserved implementations, configurations, provenance and experiment records |
| `scripts/` | Portable extraction and compact-model inference entry points |

## Quick start

Use Python 3.10 or later and install PyTorch appropriate for your CPU/GPU, then:

```bash
python -m pip install -r requirements.txt
```

Download and restore at least `historylst246-test` and `selected-checkpoints` as described in the [release guide](docs/release.md). Then, from the repository root:

```bash
python scripts/predict_compact.py --check --seed 20260905
python scripts/predict_compact.py --seed 20260905 --device cuda --split test \
  --output outputs/compact_20260905_test.npy
python resources/historylst246/score.py --split test \
  --predictions outputs/compact_20260905_test.npy \
  --output outputs/compact_20260905_test_scores.json
```

Use `--device cpu` on a machine without CUDA. The other retained seeds are `20260912` and `20260913`. Inference uses one model and one view; the article averages model scores rather than ensembling these predictions. `--check` loads and validates model weights without generating predictions.

The complete processed dataset has 27 arrays totaling 10.49 GB uncompressed, distributed in lossless Release parts of at most 1 GiB. All original `.npy` files and selected checkpoint bytes are preserved. Large binary files are excluded from Git history. See [release scope](docs/release.md#scope-and-attribution) for source attribution, licenses and historical runner assumptions.

The study evaluates retrospective, controlled Landsat-product reconstruction. Existing evaluation-history and interpretation records are retained in [docs/evaluation_history.md](docs/evaluation_history.md). Author and funding fields in the manuscript still require the authors' information.
