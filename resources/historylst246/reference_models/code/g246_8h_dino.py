"""Eight-block DINOv2 initialization fused into a full G246 thermal backbone.

This is transfer of an external self-supervised visual prior, not an additional
physical observation.  Inference consumes only the existing Fine52, coarse,
physical support and Context15.  The retained DINO encoder, including its unused
mask token, is counted in the deployment budget.  Construction never loads
pretrained weights or accesses the network; official local architecture code is
a normal Python dependency.  ``load_pretrained`` is an explicit training step.
"""

from __future__ import annotations

from functools import partial
import hashlib
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import _Block, _norm, support_project


KEPT_BLOCKS = (0, 1, 3, 4, 6, 7, 9, 11)
OFFICIAL_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
OFFICIAL_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
ASSET_RELATIVE_PATH = Path("artifacts/g246_8h_20260905/external/dinov2")
_SOURCE_OVERRIDE: Path | None = None


def set_official_source(path: Path) -> None:
    """Select an architecture location already verified by the deployment helper."""
    global _SOURCE_OVERRIDE
    candidate = Path(path).resolve()
    if not (candidate / "dinov2/models/vision_transformer.py").is_file():
        raise FileNotFoundError("the selected DINO architecture is incomplete")
    _SOURCE_OVERRIDE = candidate


def _official_source() -> Path:
    """Find the cached architecture from normal code or a captured source tree."""
    if _SOURCE_OVERRIDE is not None:
        return _SOURCE_OVERRIDE
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ASSET_RELATIVE_PATH / "source"
        if (candidate / "dinov2/models/vision_transformer.py").is_file():
            return candidate
    raise FileNotFoundError("the captured official DINOv2 Python architecture source is required")


def _encoder() -> tuple[nn.Module, Path]:
    source = _official_source()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from dinov2.models import vision_transformer
    from dinov2.layers.block import Block
    from dinov2.layers.attention import Attention
    if not Path(vision_transformer.__file__).resolve().is_relative_to(source):
        raise RuntimeError("a different DINOv2 package is already imported")
    # Regular official Block + Attention selects Torch SDPA explicitly; its
    # parameters and tensor computation match the official xFormers fallback.
    encoder = vision_transformer.DinoVisionTransformer(
        img_size=518, patch_size=14, in_chans=3, embed_dim=384,
        depth=len(KEPT_BLOCKS), num_heads=6, mlp_ratio=4,
        qkv_bias=True, ffn_bias=True, proj_bias=True, drop_path_rate=0,
        init_values=1.0, block_fn=partial(Block, attn_class=Attention),
        ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
        interpolate_antialias=False, interpolate_offset=0.1,
    )
    # No masked-token task is performed, but the retained 384 parameters remain
    # registered and counted.  Every parameter used by inference is trainable.
    encoder.mask_token.requires_grad_(False)
    return encoder, source


class DinoR6Net(nn.Module):
    """Fine52-conditioned DINO features with two initially zero injection heads."""

    def __init__(self, backbone: nn.Module, *, width: int = 32) -> None:
        super().__init__()
        if isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError("fusion width must be a positive multiple of eight")
        self.width = int(width)
        self.backbone = backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        self.encoder, source = _encoder()
        self.architecture_source_path = str(source)
        self.rgb_calibration = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.rgb_calibration.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
            self.rgb_calibration.bias.zero_()
        self.token_projection = nn.Sequential(
            nn.Conv2d(384, width, 1, bias=False), _norm(width), nn.SiLU(),
        )
        # Fine52, physical support, and coarse level/availability/base defect.
        self.fine_stem = nn.Sequential(
            nn.Conv2d(56, width, 3, padding=1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        self.local_fusion = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        self.parent = nn.Sequential(
            nn.Conv2d(width, 2 * width, 4, stride=4, bias=False),
            _norm(2 * width), nn.SiLU(), _Block(2 * width), _Block(2 * width),
        )
        self.context = nn.Linear(15, 4 * width)
        nn.init.normal_(self.context.weight, std=0.01)
        nn.init.zeros_(self.context.bias)
        self.merge = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width), _Block(width),
        )
        self.fine_injection = nn.Conv2d(width, 51, 1)
        self.dense_proposal = nn.Conv2d(width, 1, 3, padding=1)
        for head in (self.fine_injection, self.dense_proposal):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.encoder_parameter_count != 14_955_648:
            raise ValueError("DINOv2 eight-block parameter contract differs")
        if self.fusion_parameter_count > 300_000:
            raise ValueError("DINO fusion must add at most 300,000 parameters")
        if self.parameter_count >= 20_000_000:
            raise ValueError("all deployed DINO, backbone and fusion parameters must total below twenty million")

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def encoder_parameter_count(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    @property
    def fusion_parameter_count(self) -> int:
        return self.parameter_count - self.encoder_parameter_count \
            - sum(p.numel() for p in self.backbone.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-dino-r6-network-v1", "class_name": "DinoR6Net",
            "width": self.width, "parameter_count": self.parameter_count,
            "encoder_parameter_count": self.encoder_parameter_count,
            "fusion_parameter_count": self.fusion_parameter_count,
            "official_architecture_revision": OFFICIAL_REVISION,
            "architecture_dependency": str(ASSET_RELATIVE_PATH / "source"),
            "retained_official_blocks": list(KEPT_BLOCKS),
            "deployed_blocks": list(range(len(KEPT_BLOCKS))),
            "same_as_full_official_model": False,
            "encoder_initialization": "external self-supervised DINOv2 ViT-S/14 visual prior",
            "inputs": "existing Fine52/coarse/support/Context15; no new physical observation",
            "rgb_fine_indices": [4, 3, 2],
            "rgb_preprocessing": "global-z plus learned affine, optical-valid mask, bilinear antialiased resize224",
            "rgb_domain_matches_pretraining_exactly": False,
            "patch_token_grid": [16, 16, 384],
            "attention": "official Torch SDPA; no xFormers requirement",
            "frozen_unused_parameters": {"encoder.mask_token": 384},
            "all_used_encoder_parameters_trainable": True,
            "frozen_parameters_counted": True, "backbone_trainable": True,
            "u0": "exact inherited backbone; two zero CNN injection heads",
            "target_or_scoring_mask_inputs": False,
            "external_pretrained_weights_required_at_cold_build": False,
            "residual_amplitude_cap": None,
        }

    def load_pretrained(self, path: str | Path | None = None) -> dict[str, object]:
        """Validate the official full checkpoint and strictly remap eight blocks."""
        source = Path(path).resolve(strict=True) if path is not None else \
            Path(self.architecture_source_path).parent / "dinov2_vits14_pretrain.pth"
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_sha256 = digest.hexdigest()
        if source_sha256 != OFFICIAL_WEIGHTS_SHA256:
            raise ValueError("DINO weights do not match the captured official checkpoint SHA256")
        state = torch.load(source, map_location="cpu", weights_only=True)
        destination = self.encoder.state_dict()
        expected_shapes = {k: tuple(v.shape) for k, v in destination.items()
                           if not k.startswith("blocks.")}
        template = {k.removeprefix("blocks.0."): tuple(v.shape)
                    for k, v in destination.items() if k.startswith("blocks.0.")}
        for index in range(12):
            expected_shapes.update({f"blocks.{index}.{k}": shape for k, shape in template.items()})
        if not isinstance(state, dict) or set(state) != set(expected_shapes):
            raise ValueError("official DINO checkpoint key set differs from the full twelve-block model")
        for key, shape in expected_shapes.items():
            if not isinstance(state[key], Tensor) or tuple(state[key].shape) != shape:
                raise ValueError(f"official DINO checkpoint tensor shape differs: {key}")
        remapped = {k: state[k] for k in destination if not k.startswith("blocks.")}
        mapping = []
        for deployed, original in enumerate(KEPT_BLOCKS):
            for suffix in template:
                remapped[f"blocks.{deployed}.{suffix}"] = state[f"blocks.{original}.{suffix}"]
            mapping.append({"official_block": original, "deployed_block": deployed})
        self.encoder.load_state_dict(remapped, strict=True)
        return {
            "source_path": str(source), "source_sha256": source_sha256,
            "official_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "official_repository": "https://github.com/facebookresearch/dinov2",
            "official_revision": OFFICIAL_REVISION,
            "full_checkpoint_key_and_shape_validation": True,
            "strict_remapped_encoder_load": True,
            "retained_block_mapping": mapping,
            "removed_official_blocks": [i for i in range(12) if i not in KEPT_BLOCKS],
            "loaded_encoder_parameter_count_including_mask_token": self.encoder_parameter_count,
            "network_access": False, "target_data_used": False,
        }

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor, context: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must be [B,52,H,W]")
        b, _, h, w = fine.shape
        if support.shape != (b, 1, h, w) or context.shape != (b, 15):
            raise ValueError("support or Context15 geometry differs from the query")
        if coarse.shape != (b, 1, h // 4, w // 4) or h % 4 or w % 4:
            raise ValueError("coarse must be the exact x4 parent grid")
        rgb = self.rgb_calibration(fine[:, [4, 3, 2]].float())
        rgb = rgb * fine[:, 18:19].float().clamp(0, 1)
        rgb224 = F.interpolate(rgb, size=(224, 224), mode="bilinear",
                               align_corners=False, antialias=True)
        tokens = self.encoder.forward_features(rgb224)["x_norm_patchtokens"]
        token_grid = tokens.transpose(1, 2).reshape(b, 384, 16, 16)
        semantics = F.interpolate(self.token_projection(token_grid), size=(h, w),
                                  mode="bilinear", align_corners=False)

        mask = support.float()
        base = fine[:, :1].float()
        normalized = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        parent_base = F.avg_pool2d(base * mask, 4, 4) / fraction.clamp_min(1 / 16)
        valid = torch.isfinite(coarse) & (fraction > 0)
        safe_coarse = torch.where(valid, coarse.float(), parent_base)
        coarse_maps = F.interpolate(torch.cat(
            ((safe_coarse - 300) / 20, valid.float(), (safe_coarse - parent_base) / 20), dim=1),
            size=(h, w), mode="nearest")
        content = self.fine_stem(torch.cat((normalized, mask, coarse_maps), dim=1))
        value = self.local_fusion(torch.cat((content, semantics), dim=1))
        parent = self.parent(value)
        scale, shift = self.context(context.float()).tanh().chunk(2, dim=1)
        parent = parent * (1 + 0.25 * scale[:, :, None, None]) + 0.25 * shift[:, :, None, None]
        value = self.merge(torch.cat((value, F.interpolate(
            parent, size=(h, w), mode="bilinear", align_corners=False)), dim=1))
        injection = self.fine_injection(value).float() * mask
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        prediction = self.backbone(augmented, coarse, support, context).float()
        proposal = self.dense_proposal(value).float() * mask
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse)
        return prediction + support_project(proposal, zero_coarse.float(), support)


__all__ = ["DinoR6Net", "KEPT_BLOCKS"]
