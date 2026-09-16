"""Offline ImageNet initialization with a complete Fine52 thermal decoder.

Construction performs no file or network access.  The training harness can
explicitly call ``load_pretrained`` once; a deployment checkpoint contains all
encoder weights and can be restored on a machine without the source checkpoint.
The RGB inputs are G246 global-z channels, not the original ImageNet input
distribution.  A learned RGB affine calibration and a separate full Fine52
path permit adaptation without discarding the other physical predictors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet18

from g246_8h_network import _Block, _Context, _Down, _blocks, _norm, support_project


DEFAULT_PRETRAINED_PATH = Path(
    "/home/jf_xjr/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)


class _Decode(nn.Module):
    def __init__(self, side: int, rgb: int, incoming: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(side + rgb + incoming, side, 1, bias=False),
            _norm(side), nn.SiLU(), _Block(side), _Block(side),
        )

    def forward(self, side: Tensor, rgb: Tensor | None, value: Tensor) -> Tensor:
        value = F.interpolate(value, size=side.shape[-2:], mode="bilinear",
                              align_corners=False)
        fields = (side, value) if rgb is None else (side, rgb, value)
        return self.project(torch.cat(fields, dim=1))


class G246EightHourResNet18(nn.Module):
    """Trainable ResNet18 with fixed BN statistics and an unrestricted decoder."""

    def __init__(self, width: int = 48) -> None:
        super().__init__()
        if isinstance(width, bool) or int(width) != width or width < 16:
            raise ValueError("width must be an integer of at least sixteen")
        self.width = int(width)
        self.widths = tuple(max(8, int(c * width / 48 / 8 + 0.5) * 8)
                            for c in (48, 64, 96, 128, 192, 256))
        c0, c1, c2, c3, c4, c5 = self.widths
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.rgb_calibration = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.rgb_calibration.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
            self.rgb_calibration.bias.zero_()
        self.fine_stem = nn.Sequential(
            nn.Conv2d(108, c0, 3, padding=1, bias=False),
            _norm(c0), nn.SiLU(), _Block(c0),
        )
        self.side_down = nn.ModuleList(
            nn.Sequential(_Down(a, b), _Block(b))
            for a, b in zip(self.widths, self.widths[1:])
        )
        self.parent_injection = nn.Sequential(
            nn.Conv2d(c2 + 56, c2, 1, bias=False),
            _norm(c2), nn.SiLU(), _Block(c2),
        )
        self.contexts = nn.ModuleList(_Context(c) for c in (c2, c3, c4, c5))
        self.deep_fusion = nn.Sequential(
            nn.Conv2d(c5 + 512, c5, 1, bias=False),
            _norm(c5), nn.SiLU(), *_blocks(c5, 2),
        )
        self.decoders = nn.ModuleList((
            _Decode(c4, 256, c5), _Decode(c3, 128, c4),
            _Decode(c2, 64, c3), _Decode(c1, 64, c2), _Decode(c0, 0, c1),
        ))
        self.head = nn.Conv2d(c0, 1, 3, padding=1)
        nn.init.normal_(self.head.weight, std=0.002)
        nn.init.zeros_(self.head.bias)
        self.train(True)
        if self.parameter_count >= 18_000_000:
            raise ValueError("ResNet18 and all decoder parameters must total below eighteen million")

    def train(self, mode: bool = True) -> G246EightHourResNet18:
        """Preserve pretrained BN running moments; affine parameters still learn."""
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-resnet18-v1", "family": "resnet18",
            "width": self.width, "widths": list(self.widths),
            "parameter_count": self.parameter_count,
            "encoder_parameter_count": sum(p.numel() for p in self.encoder.parameters()),
            "inputs": "Fine52/coarse_kelvin/physical_support/Context15",
            "rgb_fine_indices": [4, 3, 2],
            "rgb_normalization": "G246 global-z plus learned affine; domain differs from ImageNet",
            "batchnorm": "fixed running moments; trainable affine and convolution weights",
            "scale_path": "160-80-40-20-10-5-10-20-40-80-160",
            "all_parameters_trainable": True,
            "residual_amplitude_cap": None, "teacher_dependency": False,
            "geographic_inputs": False, "constructor_external_weights": False,
            "output": "float32 complete Kelvin field with actual-support projection",
        }

    def load_pretrained(self, path: str | Path = DEFAULT_PRETRAINED_PATH) -> dict[str, object]:
        """Strictly validate and copy the local torchvision ImageNet checkpoint.

        Loading the full source model first validates the classifier too.  Only
        its explicitly discarded classifier is absent from the final encoder.
        The second strict load proves every deployed encoder tensor was copied.
        """
        source = Path(path).resolve(strict=True)
        state = torch.load(source, map_location="cpu", weights_only=True)
        complete = resnet18(weights=None)
        complete.load_state_dict(state, strict=True)
        source_parameters = sum(p.numel() for p in complete.parameters())
        complete.fc = nn.Identity()
        self.encoder.load_state_dict(complete.state_dict(), strict=True)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.train(self.training)
        return {
            "source_path": str(source), "source_sha256": digest.hexdigest(),
            "architecture": "torchvision ResNet18 ImageNet1K",
            "strict_source_load": True, "strict_encoder_load": True,
            "source_parameter_count": source_parameters,
            "loaded_encoder_parameter_count": sum(p.numel() for p in self.encoder.parameters()),
            "discarded_source_keys": ["fc.weight", "fc.bias"],
            "network_access": False,
        }

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must have shape [B,52,H,W]")
        batch, _, height, width = fine.shape
        if height % 32 or width % 32:
            raise ValueError("fine height and width must be divisible by thirty-two")
        if support.shape != (batch, 1, height, width):
            raise ValueError("support must have shape [B,1,H,W]")
        if coarse.shape != (batch, 1, height // 4, width // 4):
            raise ValueError("coarse must be the exact x4 grid")
        if context.shape != (batch, 15):
            raise ValueError("context must have fifteen nongeographic features")

        mask = support.float()
        base = fine[:, :1].float()
        scaled = torch.cat(((base - 300) / 20, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        denominator = fraction.clamp_min(1 / 16)
        parent = F.avg_pool2d(scaled, 4, 4) / denominator
        centred = (scaled - F.interpolate(parent, scale_factor=4, mode="nearest")) * mask
        valid = torch.isfinite(coarse) & (fraction > 0)
        base_parent = 300 + 20 * parent[:, :1]
        safe_coarse = torch.where(valid, coarse.float(), base_parent)
        coarse_z = (safe_coarse - 300) / 20
        discrepancy = (safe_coarse - base_parent) / 20
        parent_scalars = torch.cat((coarse_z, valid.float(), fraction, discrepancy), dim=1)
        fine_scalars = F.interpolate(
            torch.cat((coarse_z, valid.float(), discrepancy), dim=1),
            scale_factor=4, mode="nearest",
        )
        side = self.fine_stem(torch.cat((scaled, centred, mask, fine_scalars), dim=1)) * mask
        sides = [side]
        for index, down in enumerate(self.side_down, start=1):
            side = down(side)
            if index == 2:
                side = self.parent_injection(torch.cat((side, parent, parent_scalars), dim=1))
            if index >= 2:
                side = self.contexts[index - 2](side, context)
            sides.append(side)

        # Fine52 stores global-z blue/green/red at 2/3/4, respectively.
        rgb = self.rgb_calibration(scaled[:, [4, 3, 2]]) * mask
        rgb = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(rgb)))
        rgb_skips = [rgb]
        rgb = self.encoder.maxpool(rgb)
        for stage in (self.encoder.layer1, self.encoder.layer2,
                      self.encoder.layer3, self.encoder.layer4):
            rgb = stage(rgb)
            rgb_skips.append(rgb)
        value = self.deep_fusion(torch.cat((sides[-1], rgb_skips[-1]), dim=1))
        for decoder, side, rgb_skip in zip(
            self.decoders, reversed(sides[:-1]), [*reversed(rgb_skips[:-1]), None],
        ):
            value = decoder(side, rgb_skip, value)
        raw = base + self.head(value).float()
        return support_project(raw, coarse.float(), mask)


def build_model(model_spec: dict[str, object]) -> G246EightHourResNet18:
    if model_spec.get("family") != "resnet18" \
            and model_spec.get("schema_version") != "g246-8h-resnet18-v1":
        raise ValueError("unsupported G246 eight-hour ResNet18 schema")
    model = G246EightHourResNet18(width=int(model_spec.get("width", 48)))
    if "parameter_count" in model_spec and model_spec["parameter_count"] != model.parameter_count:
        raise ValueError("model specification parameter count differs")
    return model


__all__ = ["G246EightHourResNet18", "build_model", "DEFAULT_PRETRAINED_PATH"]
