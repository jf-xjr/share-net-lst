"""Strict ORQ data attachment, supervision helpers, and coarse hiding policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

import g246_data
import g246_r2_data as r2
from ordered_residual_q import parent_support_center, parent_support_mean


TEMPO_SCHEMA = "uhi-cdc-g246-tempo-sidecars-v1"
SEQUENCE_SCHEMA = "uhi-cdc-g246-r2-power-hourly-sequence-manifest-v2"
SEQUENCE_SIDECAR_SCHEMA = "uhi-cdc-g246-r2-power-hourly-sequence-sidecar-v2"
TOKEN_NAMES = (
    "tair_fit_z", "rh_fit_z", "wind_fit_z", "shortwave_fit_z",
    "solar_cos_zenith", "solar_azimuth_east", "solar_azimuth_north",
    "sunrise_exposure_0_1", "solar_phase_0_1", "delta_t_over_48",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: str | Path, label: str) -> Path:
    candidate = g246_data.reject_forbidden_path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular local file")
    return candidate.resolve()


def _member(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label} has an unsafe path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} has an unsafe path")
    candidate = root.joinpath(*pure.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its manifest root") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must name a regular file")
    return resolved


def _d4_solar(value: np.ndarray, code: int) -> np.ndarray:
    """Apply the same D4 frame transform as the raster augmentation."""

    result = np.array(value, dtype=np.float32, copy=True)
    east, north = np.array(result[..., 1], copy=True), np.array(result[..., 2], copy=True)
    rotation = int(code) & 3
    if rotation == 1:
        east, north = -north, east
    elif rotation == 2:
        east, north = -east, -north
    elif rotation == 3:
        east, north = north, -east
    if int(code) & 4:
        east = -east
    result[..., 1], result[..., 2] = east, north
    return result


def _d4_forcing(value: np.ndarray, code: int) -> np.ndarray:
    """Rotate only the Solar3 columns embedded in a 48x10 forcing tensor."""

    result = np.array(value, dtype=np.float32, copy=True)
    result[..., 4:7] = _d4_solar(result[..., 4:7], code)
    return result


class TempoAdapter:
    """Read the registered city-level 2023Q4 TEMPO morphology safely."""

    def __init__(self, manifest_path: str | Path, entries: Sequence[g246_data.G246Scene]) -> None:
        self.path = _regular(manifest_path, "TEMPO manifest")
        raw = self.path.read_bytes()
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("TEMPO manifest is malformed") from exc
        scope = manifest.get("public_scope", {})
        if (
            manifest.get("schema_version") != TEMPO_SCHEMA
            or manifest.get("full_public_scope") is not True
            or manifest.get("locked_test_opened") is not False
            or scope.get("fit_scenes") != 603 or scope.get("validation_scenes") != 45
        ):
            raise ValueError("TEMPO manifest is not the complete public v1 scope")
        records = manifest.get("records")
        if not isinstance(records, list) or len(records) != 216:
            raise ValueError("TEMPO manifest record count differs")
        self.rows: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in records:
            if not isinstance(row, Mapping):
                raise ValueError("TEMPO record is malformed")
            key = (str(row.get("view_role")), str(row.get("city")))
            if key in self.rows or key[0] not in {"fit", "validation"} or not key[1]:
                raise ValueError("TEMPO records have duplicate/unsafe identity")
            if not isinstance(row.get("file"), str) or len(str(row.get("sha256"))) != 64:
                raise ValueError("TEMPO record lacks a file/hash")
            self.rows[key] = row
        self.entries = {entry.scene_id: entry for entry in entries}
        if len(self.entries) != len(entries):
            raise ValueError("ORQ TEMPO adapter received duplicate scenes")
        for entry in entries:
            if entry.view_role not in {"fit", "validation"}:
                raise ValueError("ORQ accepts public fit/validation only")
            row = self.rows.get((entry.view_role, entry.city))
            if row is None or row.get("region") != entry.region:
                raise ValueError(f"TEMPO manifest omits {entry.scene_id}")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def load(self, entry: g246_data.G246Scene) -> tuple[np.ndarray, np.ndarray]:
        if self.entries.get(entry.scene_id) != entry:
            raise ValueError("scene is outside the TEMPO adapter scope")
        key = (entry.view_role, entry.city)
        if key not in self._cache:
            row = self.rows[key]
            path = _member(self.path.parent, row["file"], "TEMPO sidecar")
            if _sha256(path) != row["sha256"]:
                raise ValueError("TEMPO sidecar hash differs")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"tempo_raw6", "tempo_q9", "tempo_valid2", "metadata"}:
                    raise ValueError("TEMPO sidecar array contract differs")
                q = np.asarray(archive["tempo_q9"], np.float32)
                valid = np.asarray(archive["tempo_valid2"], bool)
            if q.shape != (9, 160, 160) or valid.shape != (2, 160, 160) or not np.all(np.isfinite(q)):
                raise ValueError("TEMPO sidecar geometry/value differs")
            mask = valid[1:2]
            value = q[3:6] * mask.astype(np.float32)
            self._cache[key] = (value, mask.astype(np.float32))
        value, mask = self._cache[key]
        return np.array(value, copy=True), np.array(mask, copy=True)


class SequenceAdapter:
    """Load only ORQ token values; validity remains a non-learned route gate."""

    def __init__(self, manifest_path: str | Path, entries: Sequence[g246_data.G246Scene]) -> None:
        self.path = _regular(manifest_path, "ORQ sequence manifest")
        raw = self.path.read_bytes()
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("ORQ sequence manifest is malformed") from exc
        content = dict(manifest)
        declared = content.pop("manifest_content_sha256", None)
        canonical = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (
            manifest.get("schema") != SEQUENCE_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("scene_count") != 648 or manifest.get("token_names") != list(TOKEN_NAMES)
            or manifest.get("lookback_hours") != 48 or manifest.get("locked_test_opened") is not False
            or manifest.get("target_arrays_opened") is not False or declared != canonical
        ):
            raise ValueError("ORQ sequence manifest is incomplete or unsafe")
        normalization = manifest.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("ORQ sequence normalization binding is missing")
        normal_path = _member(self.path.parent, normalization.get("path"), "ORQ normalization")
        if _sha256(normal_path) != normalization.get("sha256"):
            raise ValueError("ORQ sequence normalization hash differs")
        normal = json.loads(normal_path.read_text())
        if normal.get("schema") != "uhi-cdc-g246-r2-power-hourly-sequence-normalization-v2" or normal.get("target_arrays_opened") is not False:
            raise ValueError("ORQ sequence normalization is unsafe")
        records = manifest.get("records")
        if not isinstance(records, list) or len(records) != 648:
            raise ValueError("ORQ sequence records differ")
        self.rows: dict[str, Mapping[str, Any]] = {}
        for row in records:
            scene_id = str(row.get("scene_id", ""))
            if not scene_id or scene_id in self.rows or row.get("role") not in {"fit", "validation"}:
                raise ValueError("ORQ sequence record is malformed")
            self.rows[scene_id] = row
        self.entries = {entry.scene_id: entry for entry in entries}
        for scene_id, entry in self.entries.items():
            row = self.rows.get(scene_id)
            if row is None or any(row.get(key) != value for key, value in {
                "role": entry.view_role, "region": entry.region, "city": entry.city,
                "item_id": entry.item_id, "source_scene_sha256": entry.sha256,
            }.items()):
                raise ValueError(f"ORQ sequence row differs from public scene: {scene_id}")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}

    def load(self, entry: g246_data.G246Scene) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.entries.get(entry.scene_id) != entry:
            raise ValueError("scene is outside ORQ sequence scope")
        if entry.scene_id not in self._cache:
            row = self.rows[entry.scene_id]
            path = _member(self.path.parent, row.get("path"), "ORQ sequence sidecar")
            if _sha256(path) != row.get("sha256"):
                raise ValueError("ORQ sequence sidecar hash differs")
            payload = json.loads(path.read_text())
            scene = payload.get("scene")
            token = payload.get("tokens")
            quality = payload.get("quality")
            if (
                payload.get("schema") != SEQUENCE_SIDECAR_SCHEMA
                or not isinstance(scene, Mapping) or scene.get("scene_id") != entry.scene_id
                or not isinstance(token, Mapping) or token.get("names") != list(TOKEN_NAMES)
                or not isinstance(quality, Mapping) or payload.get("coordinates_or_absolute_time_exposed_to_model") is not False
                or payload.get("target_arrays_opened") is not False or payload.get("locked_test_opened") is not False
            ):
                raise ValueError("ORQ sequence sidecar contract differs")
            tokens = np.asarray(token.get("values"), np.float32)
            valid = np.asarray(quality.get("forcing_valid"), bool)
            solar = np.asarray(payload.get("solar_at_overpass"), np.float32)
            ready = bool(token.get("forcing_ready"))
            if tokens.shape != (48, 10) or valid.shape != (48, 4) or solar.shape != (3,) or not np.all(np.isfinite(tokens)) or not np.all(np.isfinite(solar)):
                raise ValueError("ORQ sequence tensor geometry differs")
            if ready != bool(np.all(valid)) or (not ready and np.any(tokens != 0)):
                raise ValueError("ORQ sequence missingness policy differs")
            self._cache[entry.scene_id] = (tokens, solar, ready)
        token, solar, ready = self._cache[entry.scene_id]
        return np.array(token, copy=True), np.array(solar, copy=True), bool(ready)


class OrderedResidualDataset:
    """Attach ORQ-only TEMPO and hourly inputs to a strict Core22 dataset."""

    def __init__(self, base_dataset: r2.R2TemporalDataset, *, tempo_manifest: str | Path, sequence_manifest: str | Path) -> None:
        if (
            base_dataset.fine_channels != r2.FINE_CHANNELS or base_dataset.context_dim != r2.CONTEXT_DIM
            or base_dataset.normalization.optical_source != r2.OPTICAL_SOURCE
        ):
            raise ValueError("ORQ requires the strict Core22 base dataset")
        self.base_dataset = base_dataset
        self.entries = tuple(base_dataset.entries)
        self.entry_by_id = {entry.scene_id: entry for entry in self.entries}
        self.tempo = TempoAdapter(tempo_manifest, self.entries)
        self.sequence = SequenceAdapter(sequence_manifest, self.entries)
        self.evaluation_size = base_dataset.evaluation_size
        self.provenance = {
            "schema": "uhi-cdc-g246-ordered-residual-data-v1",
            "tempo_manifest_sha256": self.tempo.manifest_sha256,
            "sequence_manifest_sha256": self.sequence.manifest_sha256,
            "locked_test_opened": False, "target_arrays_opened_by_adapter": False,
        }

    def provenance_record(self) -> dict[str, Any]:
        return dict(self.provenance)

    def _attach(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        fine = batch.get("fine")
        if not isinstance(fine, Tensor) or fine.ndim != 5 or fine.shape[1:3] != (1, 22):
            raise ValueError("ORQ base batch must be query-only Core22")
        output = dict(batch)
        height, width = fine.shape[-2:]
        tempo_values, tempo_masks, forcings, solars, ready = [], [], [], [], []
        for scene_id, origin, code in zip(batch["scene_id"], batch["crop_origin"], batch["d4_code"]):
            entry = self.entry_by_id.get(scene_id)
            if entry is None:
                raise ValueError("ORQ base batch contains an unknown scene")
            tempo, tempo_mask = self.tempo.load(entry)
            forcing, solar, forcing_ready = self.sequence.load(entry)
            row, col = map(int, origin)
            tempo_values.append(r2._spatial_transform(tempo[..., row:row + height, col:col + width], int(code)))  # noqa: SLF001
            tempo_masks.append(r2._spatial_transform(tempo_mask[..., row:row + height, col:col + width], int(code)))  # noqa: SLF001
            forcings.append(_d4_forcing(forcing, int(code)))
            solars.append(_d4_solar(solar, int(code)))
            ready.append(forcing_ready)
        forcing_array = np.stack(forcings).astype(np.float32)
        output.update({
            "fine22": fine[:, 0], "tempo": torch.from_numpy(np.stack(tempo_values)).float(),
            "tempo_valid": torch.from_numpy(np.stack(tempo_masks)).float(),
            "solar3": torch.from_numpy(np.stack(solars)).float(),
            "forcing": torch.from_numpy(forcing_array).float(),
            "forcing_ready": torch.as_tensor(ready, dtype=torch.bool),
        })
        return output

    def batch(self, batch_index: int, batch_size: int = 32, *, full: bool | None = None) -> dict[str, Any]:
        return self._attach(self.base_dataset.batch(batch_index, batch_size, full=full, query_only=True))

    def batch_slice(self, batch_index: int, start: int, stop: int, *, full: bool | None = None) -> dict[str, Any]:
        return self._attach(self.base_dataset.batch_slice(batch_index, start, stop, full=full, query_only=True))

    def evaluation_batch(self, start: int, batch_size: int) -> dict[str, Any]:
        return self._attach(self.base_dataset.evaluation_batch(start, batch_size, query_only=True))

    def panel_batch(self, batch_index: int, batch_size: int = 4) -> dict[str, Any]:
        """Return three aligned, unaugmented full scenes per sampled city.

        This is deliberately a separate small batch: the primary sampler keeps
        its normal patch/full cadence and panel supervision never changes its
        scene weighting or crop decisions.
        """

        if batch_index < 0 or batch_size <= 0:
            raise ValueError("panel batch index/size is invalid")
        schedule = self.base_dataset._schedule(batch_index, batch_size)  # noqa: SLF001
        per_slot: list[list[dict[str, Any]]] = [[], [], []]
        for position, (group, _query_slot) in enumerate(schedule):
            for slot in range(3):
                per_slot[slot].append(self.base_dataset._sample(  # noqa: SLF001
                    group, slot, full=True,
                    token=("orq-panel", batch_index, position, slot),
                    do_augment=False, sample_loss_weight=1.0,
                    query_only=True, include_supervision=True,
                ))
        attached = [self._attach(self.base_dataset._collate(samples)) for samples in per_slot]  # noqa: SLF001
        tensor_keys = (
            "fine22", "tempo", "tempo_valid", "solar3", "forcing", "forcing_ready",
            "coarse_k", "support", "target_k", "valid", "eligible",
        )
        result: dict[str, Any] = {
            key: torch.stack([batch[key] for batch in attached], dim=1)
            for key in tensor_keys
        }
        result["city"] = attached[0]["city"]
        result["region"] = attached[0]["region"]
        result["scene_ids"] = [batch["scene_id"] for batch in attached]
        return result


def hide_observed_blocks(coarse_k: Tensor, support: Tensor, *, seed: int) -> tuple[Tensor, Tensor]:
    """Hide a small contiguous observed coarse block per sample for U training."""

    if coarse_k.ndim != 4 or support.ndim != 4 or coarse_k.shape[0] != support.shape[0]:
        raise ValueError("coarse hiding tensors must be NCHW with equal batches")
    _mean, count = parent_support_mean(torch.zeros_like(support), support)
    observed = torch.isfinite(coarse_k) & (count > 0)
    hidden = torch.zeros_like(observed)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for batch in range(coarse_k.shape[0]):
        candidates = torch.nonzero(observed[batch, 0].detach().cpu(), as_tuple=False)
        if candidates.numel() == 0:
            continue
        chosen = None
        for _ in range(8):
            anchor = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
            h = int(torch.randint(1, 4, (1,), generator=generator))
            w = int(torch.randint(1, 4, (1,), generator=generator))
            row, col = int(anchor[0]), int(anchor[1])
            row, col = min(row, observed.shape[-2] - h), min(col, observed.shape[-1] - w)
            area = observed[batch, 0, row:row + h, col:col + w]
            if bool(area.all()):
                chosen = (row, col, h, w)
                break
        if chosen is None:
            anchor = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
            chosen = (int(anchor[0]), int(anchor[1]), 1, 1)
        row, col, h, w = chosen
        hidden[batch, :, row:row + h, col:col + w] = True
    return torch.where(hidden, torch.full_like(coarse_k, torch.nan), coarse_k), hidden


def hierarchical_scene_rmse_loss(
    prediction: Tensor, target: Tensor, valid: Tensor, eligible: Tensor,
    cities: Sequence[str], regions: Sequence[str],
) -> Tensor:
    """Scene RMSE, then equal city and macro-region means."""

    if prediction.shape != target.shape or prediction.ndim != 4 or valid.shape != prediction.shape or eligible.shape != prediction.shape:
        raise ValueError("hierarchical RMSE tensors must share [B,1,H,W]")
    if len(cities) != prediction.shape[0] or len(regions) != prediction.shape[0]:
        raise ValueError("city/region metadata differs from batch")
    values: dict[str, dict[str, list[Tensor]]] = {}
    for index, (city, region) in enumerate(zip(cities, regions)):
        mask = valid[index].bool() & eligible[index].bool()
        count = mask.sum()
        if int(count.detach().cpu()) == 0:
            raise ValueError("each ORQ sample needs valid and eligible supervision")
        rmse = torch.sqrt((((prediction[index].float() - target[index].float()) ** 2)[mask]).mean() + 1.0e-8)
        values.setdefault(str(region), {}).setdefault(str(city), []).append(rmse)
    regional = [torch.stack([torch.stack(scene).mean() for scene in city.values()]).mean() for city in values.values()]
    return torch.stack(regional).mean()


def u_mean_huber_loss(
    u_mean: Tensor, target: Tensor, support: Tensor, supervision: Tensor, unknown_parent: Tensor,
) -> Tensor:
    """Supervise U means only where every physical support pixel is labelled."""

    target_mean, count = parent_support_mean(target.float(), support)
    supervision_mean, _ = parent_support_mean(supervision.float(), support)
    complete = (count > 0) & torch.isclose(supervision_mean, torch.ones_like(supervision_mean), atol=1.0e-6)
    mask = unknown_parent.bool() & complete
    if not bool(mask.any()):
        return u_mean.sum() * 0.0
    return F.smooth_l1_loss(u_mean[mask], target_mean[mask], beta=1.0)


def panel_delta_huber_loss(
    prediction: Tensor, target: Tensor, support: Tensor, supervision: Tensor,
) -> Tensor:
    """Q_S panel Huber loss over legally common, fully supervised support."""

    if prediction.ndim != 5 or target.shape != prediction.shape or support.shape != prediction.shape or supervision.shape != prediction.shape or prediction.shape[1] != 3:
        raise ValueError("panel tensors must have shape [B,3,1,H,W]")
    terms: list[Tensor] = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        common = support[:, first].bool() & support[:, second].bool()
        supervised = supervision[:, first].bool() & supervision[:, second].bool()
        _none, count = parent_support_mean(torch.zeros_like(common, dtype=prediction.dtype), common)
        label_mean, _ = parent_support_mean(supervised.float(), common)
        usable_parent = (count >= 2) & torch.isclose(label_mean, torch.ones_like(label_mean), atol=1.0e-6)
        usable = common & usable_parent.repeat_interleave(4, -2).repeat_interleave(4, -1)
        if bool(usable.any()):
            residual = parent_support_center(prediction[:, second] - prediction[:, first], usable) - parent_support_center(target[:, second] - target[:, first], usable)
            terms.append(F.smooth_l1_loss(residual[usable], torch.zeros_like(residual[usable]), beta=1.0))
    return prediction.sum() * 0.0 if not terms else torch.stack(terms).mean()
