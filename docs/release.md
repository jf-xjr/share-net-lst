# Data and checkpoint downloads

Repository: `jf-xjr/share-net-lst` (private). Version: `v1.0.0-manuscript`.
**SHaRe-Net** stands for **Scale-wise Historical Reweighting Network**.

The source tree preserves the manuscript implementations and fixed partitions.
Processed dataset arrays are available in the [Google Drive data folder](https://drive.google.com/drive/folders/1xzp6lgvS1q4fS1Dcqmya80vaH8-_1--s).
The folder is **private**: sign in with an account granted access, or request
access from the folder owner through Google Drive. Dataset parts are not hosted on GitHub.
Selected checkpoint weights are attached to the GitHub Release.

`release-assets.json` records each group's storage location and the exact
byte sizes and SHA-256 checksums of every compressed part and restored file.
The original dataset contains 27 arrays (10,490,274,936 bytes), preserving
603 training, 45 validation and 90 test scenes. Lossless compression produces
five data parts totaling 3,626,248,102 bytes, each at most 1 GiB.

## Download and restore

Download the required data parts below into `downloads/`, retaining the exact
filenames. The test and validation sets each have one part; training has
three parts, all of which are required to restore the training set.

| Google Drive part | Bytes |
|---|---:|
| [historylst246-test.tar.gz.part001](https://drive.google.com/file/d/1WvYbTuWGz5vLK3NbANMOOo0ZyY78TmFK/view) | 433,481,225 |
| [historylst246-validation.tar.gz.part001](https://drive.google.com/file/d/172egi4HM7t8M-Ms7r09Uirdr7UajlprV/view) | 219,698,623 |
| [historylst246-fit.tar.gz.part001](https://drive.google.com/file/d/1t1VxDkQTktiUEGhl64op9TpaAvKR_EUj/view) | 1,073,741,824 |
| [historylst246-fit.tar.gz.part002](https://drive.google.com/file/d/1kFsu_gQI5n16gAGP6beqiA2jFpWU26UP/view) | 1,073,741,824 |
| [historylst246-fit.tar.gz.part003](https://drive.google.com/file/d/1ibQGPemF8UroLPS_qvEqBrnYhEYBSkHj/view) | 825,584,606 |

All five uploaded parts were verified on 2026-09-16 against their local byte
sizes and MD5 checksums. The [SHA-256 manifest](https://drive.google.com/file/d/1dr8svKE9-6Q2uggUEqtAokAI34gQvnFL/view)
is in the same folder. The repository [release-assets.json](../release-assets.json)
contains each part's SHA-256, MD5 and Drive URL; extraction checks SHA-256
for both downloaded parts and original array files.

For model weights in this private GitHub repository, authenticate GitHub CLI
with an account that has access:

```bash
gh release download v1.0.0-manuscript --repo jf-xjr/share-net-lst \
  --dir downloads --pattern 'selected-checkpoints.tar.gz.part*'
python scripts/restore_assets.py --assets downloads --groups selected-checkpoints
```

Restore the test data downloaded from Google Drive:

```bash
python scripts/restore_assets.py --assets downloads --groups historylst246-test
```

Or restore all three data splits:

```bash
python scripts/restore_assets.py --assets downloads \
  --groups historylst246-fit historylst246-validation historylst246-test
```

The restore script uses Python's standard library. It checks both compressed
parts and extracted file contents, restores original relative paths, skips
identical existing files, and refuses to overwrite different content.
Use `--verify-only` to check archives without extracting them. No extra joined
archive copy is needed. Reserve space for downloads plus the 10.49 GB
uncompressed dataset and selected checkpoints.

## Checkpoints

`selected-checkpoints` includes the three final compact models, three full
parents, three teacher-trained U-TAE comparators, calibrated MoCoLSK and
THSTNet, and the portable task's independent/reference checkpoints: 17 files
in one 518,527,934-byte archive part. All files retain their original paths
and bytes. The first eleven selected model identities were checked against
the original evaluation receipts. Independent/reference model manifests
remain under `resources/historylst246/`.

Teacher caches, optimizer state, intermediate training checkpoints, raw
Landsat scenes and development prediction caches are not part of this
release. The 27 processed arrays form the complete study dataset. Existing
training lineage records describe additional training inputs and provenance.

## Scope and attribution

`scripts/predict_compact.py` is a portable entry point to the unchanged final
model. It loads a hash-verified selected checkpoint, uses one full-scene FP32
forward pass with TF32 disabled, and applies the existing FP64 coarse
projection. It opens inputs only; the existing scorer opens labels separately.
Packaging generated no new predictions or training results.

Original development runners and JSON receipts are retained for provenance;
some retain historical absolute paths and job-deadline guards. The portable
inference and extraction commands above do not require those historical paths.
For training settings, see [configuration.md](configuration.md) and
[training.md](training.md).

Data attribution and provider terms are preserved in
[DATA_SOURCES.md](../resources/historylst246/DATA_SOURCES.md). Original upstream
license files remain alongside their implementations. The existing MIT license
under `resources/historylst246/` covers that portable task's software; it does
not replace source-data or third-party terms. No new blanket license has been
assigned to the assembled research repository.
