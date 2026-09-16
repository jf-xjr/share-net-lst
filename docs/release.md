# Data and checkpoint downloads

Repository: `jf-xjr/share-net-lst` (private). Version: `v1.0.0-manuscript`.
**SHaRe-Net** stands for **Scale-wise Historical Reweighting Network**.

The source tree preserves the manuscript implementations and fixed partitions.
Processed dataset arrays will be distributed through **Google Drive**. The
Google Drive link is **pending**; dataset parts are not hosted on GitHub.
Selected checkpoint weights are attached to the GitHub Release.

`release-assets.json` records each group's storage location and the exact
byte sizes and SHA-256 checksums of every compressed part and restored file.
The original dataset contains 27 arrays (10,490,274,936 bytes), preserving
603 training, 45 validation and 90 test scenes. Lossless compression produces
five data parts totaling 3,626,248,102 bytes, each at most 1 GiB.

## Download and restore

After the Google Drive link is supplied, download the required data parts into
`downloads/`. The test and validation sets each have one part; training has
three parts, all of which are required to restore the training set.

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
