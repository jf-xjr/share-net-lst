# Data and checkpoint release

Repository: `jf-xjr/share-net-lst` (private). Version: `v1.0.0-manuscript`.
The source tree preserves the manuscript implementations and fixed partitions.
Large files are stored in accompanying Release assets rather than Git history.

`release-assets.json` lists every part and restored file, including exact byte
sizes and SHA-256 checksums. The three data groups contain all 27 original
arrays (10,490,274,936 bytes). They preserve the 603 training, 45 validation and
90 test scenes. Parts are at most 1 GiB; splitting is lossless and does not
change the `.npy` arrays or the scientific data partition.

## Download and restore

For this private repository, authenticate GitHub CLI with an account that has
access. No token belongs in a source file or in this repository.

```bash
gh release download v1.0.0-manuscript --repo jf-xjr/share-net-lst \
  --dir downloads --pattern 'historylst246-test.tar.gz.part*' \
  --pattern 'selected-checkpoints.tar.gz.part*'
python scripts/restore_assets.py --assets downloads \
  --groups historylst246-test selected-checkpoints
```

For the full dataset:

```bash
gh release download v1.0.0-manuscript --repo jf-xjr/share-net-lst \
  --dir downloads --pattern '*.tar.gz.part*' --skip-existing
python scripts/restore_assets.py --assets downloads
```

The restore script uses Python's standard library. It checks both compressed
parts and extracted file contents, restores original relative paths, skips
identical existing files, and refuses to overwrite different content.
Use `--verify-only` to check the compressed archives without extracting them.
No extra joined-archive copy is needed. Reserve space for the downloaded
parts plus the 10.49 GB uncompressed data and selected checkpoints.

## Checkpoints

`selected-checkpoints` includes the three final compact models, three full
parents, three teacher-trained U-TAE comparators, calibrated MoCoLSK and
THSTNet, and the portable task's independent/reference checkpoints.
All files retain their original paths and bytes. The first eleven selected
model identities are checked against their original evaluation receipts.
Independent/reference model manifests remain under `resources/historylst246/`.

Teacher caches, optimizer state, intermediate training checkpoints, raw
Landsat scenes and development prediction caches are not part of this release.
The released 27 arrays are the complete processed study dataset. The existing
training lineage describes additional training inputs and their provenance.

## Scope and attribution

The new `scripts/predict_compact.py` is a portable entry point to the unchanged
final model. It loads a hash-verified selected checkpoint, uses one full-scene
FP32 forward pass with TF32 disabled, and applies the existing FP64 coarse
projection. It opens input arrays only; the existing scorer opens labels
separately. It is not a new experiment or a revised model.

Original development runners and JSON receipts are retained for provenance;
some retain historical absolute paths and job-deadline guards. The portable
inference and data commands above do not require those historical paths.
For training settings, see [configuration.md](configuration.md) and
[training.md](training.md).

Data attribution and provider terms are preserved in
[DATA_SOURCES.md](../resources/historylst246/DATA_SOURCES.md). Original upstream
license files remain alongside their implementations. The existing MIT license
under `resources/historylst246/` covers that portable task's software; it does
not replace source-data or third-party terms. No new blanket license has been
assigned to the assembled research repository.
