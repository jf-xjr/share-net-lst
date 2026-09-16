"""One optical r6a backbone with the registered emissivity guidance branch.

The optical input is an explicit forward argument. No query tensors are kept
on the module, and this class neither constructs a second backbone nor uses a
teacher. The external trainer owns model_spec, data binding, and checkpoint
selection. The physical gain of 0.5 comes from the existing Fit12 screen.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_emissivity_network import EmissivityNet
from g246_8h_network import support_project


class OpticalEmissivityNet(EmissivityNet):
    """Forward(fine, coarse, support, context, detail, emissivity).

    ``backbone`` is the complete five-input optical model, including its one
    r6a. At initialization this model returns the optical prediction plus the
    support-projected physical prior. Missing emissivity falls back to that
    same current optical model, including after training the guidance heads.
    """

    def __init__(self, backbone: nn.Module, *, width: int = 32,
                 modality_dropout: float = 0.25) -> None:
        super().__init__(backbone, width=width, modality_dropout=modality_dropout)

    @property
    def model_config(self) -> dict[str, object]:
        config = dict(super().model_config)
        optical = getattr(self.backbone, "net", self.backbone)
        config.update({
            "schema_version": "g246-8h-optical-emissivity-network-v1",
            "class_name": "OpticalEmissivityNet",
            "backbone_family": "optical_r6a",
            "optical_config": getattr(optical, "detail_config", None),
            "forward_arguments": ["fine", "coarse", "support", "context",
                                  "detail", "emissivity"],
            "detail_channels": 128,
            "backbone_instances": 1,
            "u0": "complete optical backbone + 0.5 * actual-support projected physical prior",
            "missing_modality": "missing emissivity exactly preserves current optical backbone",
            "query_tensors_stored_on_module": False,
        })
        return config

    def load_optical_deploy(self, checkpoint: Mapping[str, Any]) -> dict[str, object]:
        """Strictly initialize only the backbone from a complete optical deploy.

        Call this on a freshly constructed combination to retain the registered
        0.5 gain and zero CNN heads. A combination checkpoint is restored by
        ordinary ``load_state_dict(..., strict=True)`` on a fresh combination.
        """
        if checkpoint.get("schema") != "g246-8h-deploy-v1":
            raise ValueError("requires a complete g246-8h optical deploy checkpoint")
        spec = checkpoint.get("model_spec")
        if not isinstance(spec, Mapping) or spec.get("family") != "optical_r6a":
            raise ValueError("backbone initialization requires optical_r6a family")
        if checkpoint.get("locked_test_opened") is not False:
            raise ValueError("optical initialization must explicitly keep locked test closed")
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("optical deploy is missing its complete state_dict")
        count = sum(p.numel() for p in self.backbone.parameters())
        if checkpoint.get("parameter_count") != count:
            raise ValueError("optical checkpoint parameter count differs from the supplied backbone")
        self.backbone.load_state_dict(state, strict=True)
        return {"source_schema": checkpoint["schema"], "source_family": spec["family"],
                "strict_backbone_load": True, "backbone_parameter_count": count,
                "state_key": "state_dict", "teacher_dependency": False,
                "locked_test_opened": False}

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, detail: Tensor, emissivity: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must have shape [B,52,H,W]")
        b, _, h, w = fine.shape
        if detail.shape != (b, 128, h, w):
            raise ValueError("detail must have shape [B,128,H,W]")
        if emissivity.shape != (b, 4, h, w):
            raise ValueError("emissivity must have shape [B,4,H,W]")
        if support.shape != (b, 1, h, w) or context.shape != (b, 15):
            raise ValueError("support or Context15 geometry differs from the query")
        if coarse.shape != (b, 1, h // 4, w // 4) or h % 4 or w % 4:
            raise ValueError("coarse must be the query's exact x4 parent grid")

        scene_available = (emissivity[:, 3:4] > 0).flatten(1).any(dim=1)
        scene_available = scene_available.float()[:, None, None, None]
        if self.training and self.modality_dropout:
            keep = torch.rand((b, 1, 1, 1), device=fine.device) >= self.modality_dropout
            scene_available = scene_available * keep.float()
        physical = torch.where(scene_available > 0, emissivity.float(),
                               torch.zeros_like(emissivity, dtype=torch.float32))
        mask = support.float()
        base = fine[:, :1].float()
        scaled_fine = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        parent_base = F.avg_pool2d(base * mask, 4, 4) / fraction.clamp_min(1 / 16)
        valid = torch.isfinite(coarse) & (fraction > 0)
        safe_coarse = torch.where(valid, coarse.float(), parent_base)
        coarse_maps = F.interpolate(torch.cat(
            ((safe_coarse - 300) / 20, valid.float(), (safe_coarse - parent_base) / 20), dim=1),
            size=(h, w), mode="nearest")
        value = self.stem(torch.cat((physical, scaled_fine, mask, coarse_maps), dim=1))
        parent = self.parent(value)
        scale, shift = self.context(context.float()).tanh().chunk(2, dim=1)
        parent = parent * (1 + 0.25 * scale[:, :, None, None]) + 0.25 * shift[:, :, None, None]
        value = self.merge(torch.cat((value, F.interpolate(
            parent, size=(h, w), mode="bilinear", align_corners=False)), dim=1))

        gate = scene_available * mask
        injection = self.fine_injection(value).float() * gate
        augmented = torch.cat((base, fine[:, 1:].float() + injection), dim=1)
        # This is the only difference from the registered single-modality
        # computation: explicitly preserve the optical model's fifth argument.
        prediction = self.backbone(augmented, coarse, support, context, detail).float()
        zero_coarse = torch.where(torch.isfinite(coarse), torch.zeros_like(coarse), coarse)
        prior = support_project(-0.8 * physical[:, :1] * mask, zero_coarse.float(), support)
        learned = support_project(self.dense_proposal(value).float() * gate,
                                  zero_coarse.float(), support)
        return prediction + self.physical_gain.float() * prior + learned


__all__ = ["OpticalEmissivityNet"]
