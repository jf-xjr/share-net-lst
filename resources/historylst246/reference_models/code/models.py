"""Reusable same-resolution PyTorch baselines for thermal residual prediction.

All models accept tensors in ``NCHW`` layout and return one signed residual
channel at the input spatial resolution.  The caller is responsible for adding
that residual to any coarse thermal reference.
"""

from __future__ import annotations

from math import gcd
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_width(width: int) -> int:
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    return width


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = gcd(channels, min(max_groups, channels))
    return nn.GroupNorm(groups, channels)


def _pad_to_multiple(x: Tensor, multiple: int) -> tuple[Tensor, tuple[int, int]]:
    """Zero-pad bottom/right so encoder pooling is safe, returning crop sizes."""
    height, width = x.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (height, width)


def _crop(x: Tensor, size: tuple[int, int]) -> Tensor:
    return x[..., : size[0], : size[1]]


def _cubic_weight(distance: float, coefficient: float = -0.75) -> float:
    value = abs(float(distance))
    if value <= 1.0:
        return (coefficient + 2.0) * value**3 - (coefficient + 3.0) * value**2 + 1.0
    if value < 2.0:
        return (
            coefficient * value**3
            - 5.0 * coefficient * value**2
            + 8.0 * coefficient * value
            - 4.0 * coefficient
        )
    return 0.0


def deterministic_bicubic_upsample2x(x: Tensor) -> Tensor:
    """Exact half-pixel 2x Keys-cubic interpolation via grouped convolutions."""
    if x.ndim != 4:
        raise ValueError("bicubic upsampling expects an NCHW tensor")
    channels = x.shape[1]
    even = [_cubic_weight(value) for value in (1.75, 0.75, 0.25, 1.25)]
    odd = [_cubic_weight(value) for value in (1.25, 0.25, 0.75, 1.75)]
    horizontal_even = x.new_tensor(even).view(1, 1, 1, 4).expand(channels, 1, 1, 4)
    horizontal_odd = x.new_tensor(odd).view(1, 1, 1, 4).expand(channels, 1, 1, 4)
    left = F.conv2d(
        F.pad(x, (2, 1, 0, 0), mode="replicate"), horizontal_even, groups=channels
    )
    right = F.conv2d(
        F.pad(x, (1, 2, 0, 0), mode="replicate"), horizontal_odd, groups=channels
    )
    horizontal = torch.stack((left, right), dim=-1).reshape(
        x.shape[0], channels, x.shape[2], x.shape[3] * 2
    )
    vertical_even = x.new_tensor(even).view(1, 1, 4, 1).expand(channels, 1, 4, 1)
    vertical_odd = x.new_tensor(odd).view(1, 1, 4, 1).expand(channels, 1, 4, 1)
    top = F.conv2d(
        F.pad(horizontal, (0, 0, 2, 1), mode="replicate"), vertical_even, groups=channels
    )
    bottom = F.conv2d(
        F.pad(horizontal, (0, 0, 1, 2), mode="replicate"), vertical_odd, groups=channels
    )
    return torch.stack((top, bottom), dim=3).reshape(
        x.shape[0], channels, x.shape[2] * 2, x.shape[3] * 2
    )


def _init_kaiming(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    """Two-convolution residual block with batch-size-stable normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(x)
        return self.block(torch.cat((skip, x), dim=1))


class ResidualUNet(nn.Module):
    """Four-downsample residual U-Net for competent same-grid regression."""

    default_width = 32

    def __init__(self, in_channels: int = 7, width: int = default_width) -> None:
        super().__init__()
        width = _check_width(width)
        channels = (width, 2 * width, 4 * width, 8 * width, 16 * width)
        self.stem = ResidualBlock(in_channels, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.down4 = DownBlock(channels[3], channels[4])
        self.up4 = UpBlock(channels[4], channels[3], channels[3])
        self.up3 = UpBlock(channels[3], channels[2], channels[2])
        self.up2 = UpBlock(channels[2], channels[1], channels[1])
        self.up1 = UpBlock(channels[1], channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.apply(_init_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        x, original_size = _pad_to_multiple(x, 16)
        e1 = self.stem(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        bottleneck = self.down4(e4)
        x = self.up4(bottleneck, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return _crop(self.head(x), original_size)


class EDSRResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_scale: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.residual_scale = float(residual_scale)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.activation(self.conv1(x)))
        return x + self.residual_scale * residual


class EDSRResidualCNN(nn.Module):
    """EDSR-like residual CNN without upsampling or batch normalization."""

    default_width = 64

    def __init__(
        self,
        in_channels: int = 7,
        width: int = default_width,
        num_blocks: int = 16,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        width = _check_width(width)
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*(EDSRResidualBlock(width, residual_scale) for _ in range(num_blocks)))
        self.body_tail = nn.Conv2d(width, width, 3, padding=1)
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        self.apply(_init_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        features = self.head(x)
        body = self.body_tail(self.blocks(features))
        return self.output(features + body)


class ConvBNReLU(nn.Sequential):
    """PyTorch equivalent of the official Keras convolution_block."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ParallelDilatedStage(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int, dilations: Sequence[int]) -> None:
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("DST-UNet stages require exactly four dilation branches")
        self.dilations = tuple(int(value) for value in dilations)
        self.branches = nn.ModuleList(
            ConvBNReLU(in_channels, branch_channels, dilation=value) for value in self.dilations
        )

    @property
    def out_channels(self) -> int:
        return sum(branch[0].out_channels for branch in self.branches)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class DSTUNet(nn.Module):
    """Five-level PyTorch adaptation of the official parallel-dilation DST-UNet.

    ``width=16`` reproduces the official per-branch channel schedule. Width is
    the number of channels in each finest-level branch, not the concatenated
    stage width.
    """

    default_width = 16
    encoder_dilations = (
        (1, 12, 24, 36),
        (1, 6, 12, 18),
        (1, 3, 6, 9),
        (1, 2, 4, 6),
        (1, 2, 3, 4),
    )
    decoder_dilations = (
        (1, 2, 3, 4),
        (1, 2, 4, 6),
        (1, 3, 6, 9),
        (1, 6, 12, 18),
        (1, 12, 18, 36),
    )

    def __init__(self, in_channels: int = 7, width: int = default_width) -> None:
        super().__init__()
        w = _check_width(width)
        branch_channels = (w, 2 * w, 4 * w, 8 * w, 8 * w)
        merge_channels = tuple(4 * value for value in branch_channels)

        encoders: list[nn.Module] = []
        stage_in = in_channels
        for branch_width, dilations in zip(branch_channels, self.encoder_dilations):
            encoders.append(ParallelDilatedStage(stage_in, branch_width, dilations))
            stage_in = 4 * branch_width
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            ConvBNReLU(merge_channels[-1], 64 * w),
            ConvBNReLU(64 * w, 64 * w),
        )

        decoder_pre_channels = (32 * w, 32 * w, 16 * w, 8 * w, 4 * w)
        decoder_branch_channels = (8 * w, 8 * w, 4 * w, 2 * w, w)
        decoder_skip_channels = tuple(reversed(merge_channels))
        decoder_input_channels = (64 * w, 32 * w, 32 * w, 16 * w, 8 * w)
        self.decoder_pre = nn.ModuleList(
            ConvBNReLU(in_ch, out_ch)
            for in_ch, out_ch in zip(decoder_input_channels, decoder_pre_channels)
        )
        self.decoders = nn.ModuleList(
            ParallelDilatedStage(pre_ch + skip_ch, branch_ch, dilations)
            for pre_ch, skip_ch, branch_ch, dilations in zip(
                decoder_pre_channels,
                decoder_skip_channels,
                decoder_branch_channels,
                self.decoder_dilations,
            )
        )

        self.head = nn.Sequential(
            nn.Conv2d(4 * w, 2 * w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(2 * w, 1, 3, padding=1)
        self.apply(_init_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        x, original_size = _pad_to_multiple(x, 32)
        skips: list[Tensor] = []
        for stage in self.encoders:
            x = stage(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for pre, stage, skip in zip(self.decoder_pre, self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = pre(x)
            x = stage(torch.cat((skip, x), dim=1))
        return _crop(self.output(self.head(x)), original_size)


class ASPPBranch(nn.Sequential):
    """One regression-safe atrous branch with batch-size-stable normalization."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class AtrousSpatialPyramid(nn.Module):
    """ASPP at the four rates reported by the urban-thermal comparator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Sequence[int] = (1, 6, 12, 18),
    ) -> None:
        super().__init__()
        if tuple(dilations) != (1, 6, 12, 18):
            raise ValueError("DeepLabV3+ comparator requires ASPP rates (1, 6, 12, 18)")
        self.dilations = tuple(int(value) for value in dilations)
        self.branches = nn.ModuleList(
            ASPPBranch(in_channels, out_channels, dilation) for dilation in self.dilations
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(self.dilations) + 1) * out_channels, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))


class DeepLabV3PlusResidual(nn.Module):
    """Matched-input adaptation of the 2026 urban-thermal DeepLabV3+ model.

    The cited model uses a ResNet-50 encoder, ASPP rates 1/6/12/18, low-level
    decoder fusion, and progressive convolution/bicubic upscaling.  This
    adaptation keeps those structural elements but replaces its bounded absolute
    LST output with the signed residual contract shared by every comparator.

    ImageNet weights are deliberately not downloaded implicitly.  A separately
    frozen workspace-local state dictionary can be supplied by the training
    harness for the pretraining ablation; the default comparator starts from the
    standard torchvision initialization.
    """

    default_width = 128
    aspp_dilations = (1, 6, 12, 18)

    def __init__(
        self,
        in_channels: int = 7,
        width: int = default_width,
        *,
        backbone_state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        width = _check_width(width)
        try:
            from torchvision.models import resnet50
        except ImportError as exc:  # pragma: no cover - environment diagnostic.
            raise RuntimeError("DeepLabV3+ requires torchvision.models.resnet50") from exc

        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        if backbone_state_dict is not None:
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            permitted_missing = {"fc.weight", "fc.bias"}
            if set(missing).difference(permitted_missing) or unexpected:
                raise ValueError(
                    "incompatible ResNet-50 state dictionary: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if backbone_state_dict is None:
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        else:
            # Preserve RGB pretraining and initialize any additional physical
            # channels with the pretrained filter mean. This rule is fixed and
            # avoids an undocumented random first-layer advantage/disadvantage.
            with torch.no_grad():
                rgb = original_conv.weight
                copied = min(in_channels, rgb.shape[1])
                backbone.conv1.weight[:, :copied].copy_(rgb[:, :copied])
                if in_channels > copied:
                    mean_filter = rgb.mean(dim=1, keepdim=True)
                    backbone.conv1.weight[:, copied:].copy_(
                        mean_filter.expand(-1, in_channels - copied, -1, -1)
                    )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = AtrousSpatialPyramid(2048, width, self.aspp_dilations)
        low_width = max(16, width // 4)
        self.low_project = nn.Sequential(
            nn.Conv2d(256, low_width, 1, bias=False),
            _group_norm(low_width),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width + low_width, width, 3, padding=1, bias=False),
            _group_norm(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            _group_norm(width),
            nn.ReLU(inplace=True),
        )
        progressive_width = max(16, width // 2)
        self.progressive_up1 = nn.Sequential(
            nn.Conv2d(width, progressive_width, 3, padding=1, bias=False),
            _group_norm(progressive_width),
            nn.ReLU(inplace=True),
        )
        self.progressive_up2 = nn.Sequential(
            nn.Conv2d(progressive_width, progressive_width, 3, padding=1, bias=False),
            _group_norm(progressive_width),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(progressive_width, 1, 3, padding=1)
        for module in (
            self.aspp,
            self.low_project,
            self.fuse,
            self.progressive_up1,
            self.progressive_up2,
            self.output,
        ):
            module.apply(_init_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        x, original_size = _pad_to_multiple(x, 16)
        padded_size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat((x, self.low_project(low)), dim=1))
        half_size = (max(1, padded_size[0] // 2), max(1, padded_size[1] // 2))
        if (x.shape[-2] * 2, x.shape[-1] * 2) != half_size:
            raise RuntimeError("DeepLab progressive bicubic stage requires an exact 2x scale")
        x = deterministic_bicubic_upsample2x(x)
        x = self.progressive_up1(x)
        if (x.shape[-2] * 2, x.shape[-1] * 2) != padded_size:
            raise RuntimeError("DeepLab final bicubic stage requires an exact 2x scale")
        x = deterministic_bicubic_upsample2x(x)
        x = self.progressive_up2(x)
        return _crop(self.output(x), original_size)


_MODEL_BUILDERS: dict[str, tuple[type[nn.Module], int]] = {
    "unet": (ResidualUNet, ResidualUNet.default_width),
    "edsr": (EDSRResidualCNN, EDSRResidualCNN.default_width),
    "dstunet": (DSTUNet, DSTUNet.default_width),
    "deeplabv3plus": (DeepLabV3PlusResidual, DeepLabV3PlusResidual.default_width),
}


def build_model(name: str, in_channels: int = 7, width: int = 0, **kwargs: object) -> nn.Module:
    """Build a named baseline.

    ``width=0`` selects the class-specific canonical default. Positive widths
    give an explicit capacity-curve point shared by the training CLI.
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _MODEL_BUILDERS:
        choices = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(f"unknown model {name!r}; expected one of: {choices}")
    model_class, default_width = _MODEL_BUILDERS[key]
    selected_width = default_width if width == 0 else _check_width(width)
    return model_class(in_channels=in_channels, width=selected_width, **kwargs)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# Short aliases keep direct imports convenient without hiding descriptive names.
UNet = ResidualUNet
EDSR = EDSRResidualCNN


__all__ = [
    "ResidualUNet",
    "UNet",
    "EDSRResidualCNN",
    "EDSR",
    "DSTUNet",
    "DeepLabV3PlusResidual",
    "deterministic_bicubic_upsample2x",
    "build_model",
    "trainable_parameter_count",
]
