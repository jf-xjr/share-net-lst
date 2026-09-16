"""A compact full-field G246 network for the bounded eight-hour experiment.

Inputs are registered query-only Fine52, coarse Kelvin observations, physical
support and Context15 (the four geographic columns have already been removed).
The model has no teacher, target, scoring mask or scene-identity dependency.
Fine52 channel zero is the interpolated Kelvin base; remaining channels use
their existing train-fitted normalization.  Shapes are B,C,H,W throughout.
"""

from __future__ import annotations

from math import gcd
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def support_project(raw: Tensor, coarse: Tensor, support: Tensor) -> Tensor:
    """Project on actual supported O means; retain U and remove Z coordinates.

    Arithmetic follows ``raw.dtype``.  Forward uses float32; an evaluator can
    call this same operation with float64 fields for a final precision repair.
    There are no tensor-to-Python synchronizations or scoring-mask inputs.
    """
    mask = support.to(dtype=raw.dtype)
    safe_raw = torch.where(mask > 0, raw, torch.zeros_like(raw))
    count_fraction = F.avg_pool2d(mask, 4, 4)
    parent_mean = F.avg_pool2d(safe_raw, 4, 4) / count_fraction.clamp_min(1.0 / 16.0)
    observed = torch.isfinite(coarse) & (count_fraction > 0)
    safe_coarse = torch.where(observed, coarse.to(raw.dtype), parent_mean)
    correction = safe_coarse - parent_mean
    correction = F.interpolate(correction, scale_factor=4, mode="nearest")
    return torch.where(mask > 0, safe_raw + correction, torch.zeros_like(raw))


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(gcd(channels, 8), channels)


class _Block(nn.Module):
    """Depthwise spatial mixing with an open, moderate residual branch."""

    def __init__(self, channels: int, expansion: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2,
                                   groups=channels, bias=False)
        self.norm = _norm(channels)
        self.expand = nn.Conv2d(channels, expansion * channels, 1)
        self.contract = nn.Conv2d(expansion * channels, channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        update = self.depthwise(value)
        update = F.silu(self.norm(update))
        update = self.contract(F.silu(self.expand(update)))
        return value + 0.5 * update


def _blocks(channels: int, depth: int) -> nn.Sequential:
    return nn.Sequential(*(_Block(channels) for _ in range(depth)))


class _Down(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__(nn.Conv2d(incoming, outgoing, 2, stride=2, bias=False),
                         _norm(outgoing), nn.SiLU())


class _Fuse(nn.Module):
    def __init__(self, skip_channels: int, incoming: int, depth: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(skip_channels + incoming, skip_channels, 1, bias=False),
            _norm(skip_channels), nn.SiLU(),
        )
        self.blocks = _blocks(skip_channels, depth)

    def forward(self, skip: Tensor, value: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.blocks(self.project(torch.cat((skip, value), dim=1)))


class _Context(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(15, 2 * channels)
        nn.init.normal_(self.affine.weight, std=0.01)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value: Tensor, context: Tensor) -> Tensor:
        scale, shift = self.affine(context).tanh().chunk(2, dim=1)
        return value * (1.0 + 0.25 * scale[:, :, None, None]) \
            + 0.25 * shift[:, :, None, None]


class G2468HNetwork(nn.Module):
    """Full-coordinate guided ResUNet with support-aware parent interactions."""

    def __init__(
        self, widths: Sequence[int] = (48, 80, 128, 192, 256), depth: int = 2,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in widths)
        if len(widths) != 5 or any(value < 8 or value % 8 for value in widths):
            raise ValueError("five widths, each a positive multiple of eight, required")
        if depth < 1:
            raise ValueError("depth must be positive")
        self.widths = widths
        self.depth = int(depth)
        c0, c1, c2, c3, c4 = widths
        # Current fine features plus their actual-support-centred contrasts.
        # The four scalar channels are support, coarse level, coarse validity
        # and observed coarse-minus-interpolated-base parent discrepancy.
        self.stem = nn.Sequential(nn.Conv2d(108, c0, 3, padding=1, bias=False),
                                  _norm(c0), nn.SiLU())
        self.encoders = nn.ModuleList(_blocks(c, self.depth) for c in widths)
        self.downs = nn.ModuleList(_Down(a, b) for a, b in zip(widths, widths[1:]))
        self.parent_injection = nn.Sequential(
            nn.Conv2d(c2 + c0 + 52 + 4, c2, 1, bias=False),
            _norm(c2), nn.SiLU(), _Block(c2),
        )
        self.contexts = nn.ModuleList(_Context(c) for c in (c2, c3, c4))
        self.decoders = nn.ModuleList(
            _Fuse(widths[index], widths[index + 1], self.depth)
            for index in range(3, -1, -1)
        )
        self.head = nn.Conv2d(c0, 1, 3, padding=1)
        # Small nonzero weights allow all encoder/decoder paths to receive
        # gradients on the first update without imposing a residual amplitude cap.
        nn.init.normal_(self.head.weight, std=0.002)
        nn.init.zeros_(self.head.bias)
        if self.parameter_count >= 20_000_000:
            raise ValueError("the network must contain fewer than twenty million parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-network-v1",
            "class_name": type(self).__name__,
            "widths": list(self.widths), "depth": self.depth,
            "parameter_count": self.parameter_count,
            "inputs": "Fine52/coarse_kelvin/physical_support/Context15",
            "fine_channel_zero": "interpolated_kelvin_base",
            "output": "float32_complete_kelvin_field_actual_support_projection",
            "teacher_dependency": False, "geographic_inputs": False,
            "residual_amplitude_cap": None,
            "scale_path": "160-80-40-20-10-20-40-80-160",
        }

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context15: Tensor) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 52:
            raise ValueError("fine must have shape [B,52,H,W]")
        batch, _, height, width = fine.shape
        if height % 16 or width % 16:
            raise ValueError("fine height and width must be divisible by sixteen")
        if support.shape != (batch, 1, height, width):
            raise ValueError("support must have shape [B,1,H,W]")
        if coarse.shape != (batch, 1, height // 4, width // 4):
            raise ValueError("coarse must be an exact x4 grid")
        if context15.shape != (batch, 15):
            raise ValueError("context15 must contain exactly fifteen nongeographic features")

        mask = support.to(dtype=torch.float32)
        base = fine[:, :1].float()
        scaled = torch.cat(((base - 300.0) / 20.0, fine[:, 1:].float()), dim=1)
        scaled = scaled * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        denominator = fraction.clamp_min(1.0 / 16.0)
        parent_features = F.avg_pool2d(scaled, 4, 4) / denominator
        centred = (scaled - F.interpolate(parent_features, scale_factor=4,
                                          mode="nearest")) * mask
        coarse_valid = torch.isfinite(coarse) & (fraction > 0)
        base_parent = 300.0 + 20.0 * parent_features[:, :1]
        coarse_safe = torch.where(coarse_valid, coarse.float(), base_parent)
        coarse_z = (coarse_safe - 300.0) / 20.0
        discrepancy = (coarse_safe - base_parent) / 20.0
        scalar_parent = torch.cat((coarse_z, coarse_valid.float(), fraction,
                                   discrepancy), dim=1)
        scalar_fine = F.interpolate(
            torch.cat((coarse_z, coarse_valid.float(), discrepancy), dim=1),
            scale_factor=4, mode="nearest",
        )
        value = self.stem(torch.cat((scaled, centred, mask, scalar_fine), dim=1))
        fine_stem = value * mask
        skips: list[Tensor] = []
        value = fine_stem
        for index, encoder in enumerate(self.encoders):
            if index:
                value = self.downs[index - 1](value)
            if index == 2:
                pooled_stem = F.avg_pool2d(fine_stem, 4, 4) / denominator
                value = self.parent_injection(torch.cat(
                    (value, pooled_stem, parent_features, scalar_parent), dim=1))
            if index >= 2:
                value = self.contexts[index - 2](value, context15)
            value = encoder(value)
            skips.append(value)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            value = decoder(skip, value)
        raw = base + self.head(value).float()
        return support_project(raw, coarse.float(), mask)


class G246EightHourNet(G2468HNetwork):
    """Trainer-facing width-scaled alias; width 48 is the default 3.05M model."""

    def __init__(self, width: int = 48) -> None:
        if isinstance(width, bool) or int(width) != width or width < 16:
            raise ValueError("width must be an integer of at least sixteen")
        channels = tuple(max(8, int(value * width / 48.0 / 8.0 + 0.5) * 8)
                         for value in (48, 80, 128, 192, 256))
        super().__init__(widths=channels, depth=2)
        self.width = int(width)


def build_model(model_spec: dict[str, object]) -> G2468HNetwork:
    """Reconstruct the deployment architecture from a saved model_config."""
    if model_spec.get("family") == "multiscale":
        return G246EightHourNet(width=int(model_spec["width"]))
    if model_spec.get("schema_version") != "g246-8h-network-v1":
        raise ValueError("unsupported G246 eight-hour model schema")
    model = G2468HNetwork(widths=model_spec["widths"], depth=int(model_spec["depth"]))
    if model_spec.get("parameter_count") != model.parameter_count:
        raise ValueError("model_spec parameter count differs from the architecture")
    return model


__all__ = ["G2468HNetwork", "G246EightHourNet", "build_model", "support_project"]
