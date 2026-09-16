"""Seal the exact official DINO architecture inside a weights-only deployment.

The official source archive is embedded as bytes; pretrained weight files are
never copied or needed by this helper. Extraction accepts only files whose
hashes match the captured manifest and the pinned official archive.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = Path("artifacts/g246_8h_20260905/external/dinov2")
REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
ARCHIVE_SHA256 = "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_files(bundle: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    if bundle.get("schema") != "g246-official-dino-source-bundle-v1":
        raise ValueError("unsupported DINO source bundle")
    archive, metadata = bundle["archive_bytes"], bundle["asset_manifest_bytes"]
    if (not isinstance(archive, bytes) or not isinstance(metadata, bytes)
            or _sha(archive) != ARCHIVE_SHA256
            or bundle.get("archive_sha256") != ARCHIVE_SHA256
            or _sha(metadata) != bundle.get("asset_manifest_sha256")):
        raise ValueError("DINO architecture archive or manifest binding changed")
    manifest = json.loads(metadata)
    if manifest.get("status") != "complete" or manifest.get("git_revision") != REVISION:
        raise ValueError("DINO architecture revision differs from the registered official source")
    files = {}
    prefix = f"dinov2-{REVISION}/"
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for relative, digest in manifest["source_file_hashes"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "source":
                raise ValueError("unsafe source path in DINO manifest")
            content = source.read(prefix + str(path.relative_to("source")))
            if _sha(content) != digest:
                raise ValueError(f"DINO source file differs from the official archive: {relative}")
            files[str(path)] = content
    if "source/dinov2/models/vision_transformer.py" not in files or "source/LICENSE" not in files:
        raise ValueError("DINO architecture or license is missing")
    return files, manifest


def capture_dino_source() -> dict[str, Any]:
    """Return a torch.save-compatible bundle from the pinned local official assets."""
    root = ROOT / ASSET_PATH
    archive = (root / "official_source.zip").read_bytes()
    metadata = (root / "asset_manifest.json").read_bytes()
    bundle = {"schema": "g246-official-dino-source-bundle-v1", "official_revision": REVISION,
              "archive_sha256": _sha(archive), "asset_manifest_sha256": _sha(metadata),
              "archive_bytes": archive, "asset_manifest_bytes": metadata}
    files, _ = _validated_files(bundle)
    for relative, content in files.items():
        if (root / relative).read_bytes() != content:
            raise ValueError(f"local DINO architecture differs from the captured official archive: {relative}")
    return bundle


def source_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """A JSON-compatible immutable binding for training configuration files."""
    files, manifest = _validated_files(bundle)
    hashes = {name: _sha(value) for name, value in sorted(files.items())}
    return {"schema": bundle["schema"], "official_revision": REVISION,
            "archive_sha256": ARCHIVE_SHA256, "asset_manifest_sha256": bundle["asset_manifest_sha256"],
            "source_tree_sha256": _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()),
            "source_file_count": len(files), "source_file_sha256": hashes,
            "source_license": manifest.get("license"),
            "pretrained_weights_required_for_inference": False}


def materialize_dino_source(bundle: dict[str, Any], destination_root: Path) -> dict[str, Any]:
    """Preserve source under the ancestor-search layout used by DinoR6Net."""
    files, _ = _validated_files(bundle)
    destination_root = Path(destination_root).resolve()
    if "locked" in str(destination_root).lower():
        raise ValueError("DINO deployment source destination must be public")
    root = destination_root / ASSET_PATH
    for relative, content in {**files, "asset_manifest.json": bundle["asset_manifest_bytes"]}.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"existing deployment source differs: {path}")
            continue
        temporary = path.with_name(path.name + f".partial-{os.getpid()}")
        temporary.write_bytes(content)
        temporary.replace(path)
    return {**source_summary(bundle), "source_root": str(root / "source")}


def _uses_dino(spec: dict[str, Any]) -> bool:
    return spec.get("family") == "dino_r6a" or any(
        _uses_dino(entry["model_spec"]) for entry in spec.get("members", []))


def prepare_dino_source(payload: dict[str, Any], destination_root: Path | None = None) -> dict[str, Any] | None:
    """Verify, extract and activate sealed code before a cold model is built."""
    if not _uses_dino(payload["model_spec"]):
        return None
    dependencies = payload.setdefault("source_dependencies", {})
    if "dinov2" not in dependencies:
        dependencies["dinov2"] = capture_dino_source()
    bundle = dependencies["dinov2"]
    summary = source_summary(bundle)
    declared = payload.get("config", {}).get("source_dependencies", {})
    declared = declared.get("dinov2", declared)
    if declared:
        for key in ("archive_sha256", "asset_manifest_sha256", "source_tree_sha256"):
            if declared.get(key) != summary[key]:
                raise ValueError(f"DINO configuration source binding differs: {key}")
    if destination_root is None:
        destination_root = Path(tempfile.gettempdir()) / "g246_dino_deployment_sources" / summary["source_tree_sha256"]
    materialized = materialize_dino_source(bundle, destination_root)
    selected = Path(materialized["source_root"])
    existing = sys.modules.get("dinov2.models.vision_transformer")
    if existing is not None:
        existing_root = Path(existing.__file__).resolve().parents[2]
        files, _ = _validated_files(bundle)
        for relative, content in files.items():
            if (existing_root / Path(relative).relative_to("source")).read_bytes() != content:
                raise ValueError("an incompatible DINO source package is already imported")
        selected = existing_root
    from g246_8h_dino import set_official_source
    set_official_source(selected)
    return {**materialized, "runtime_source_root": str(selected), "embedded_in_checkpoint": True}


__all__ = ["capture_dino_source", "source_summary", "materialize_dino_source", "prepare_dino_source"]
