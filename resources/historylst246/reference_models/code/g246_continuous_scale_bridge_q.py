"""Continuous Scale-Bridge Q model for G246.

The model is one r6a ``SceneCalibratedContinuousQ`` backbone, not an ensemble.
It adds only the missing 160->80->40 content route and the matching
40->80->160 band decoder.  A zero bridge preserves the inherited parent path,
and zero QM/QH output convolutions make a migrated model exactly the r6a
function at initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_calibrated_continuous_q import SceneCalibratedContinuousQ
from g246_ipmr_q import (
    IPMRQ,
    _Downsample,
    _Merge,
    _Project,
    _ResidualDW,
)
from g246_q_bands import lift_p80_coefficients, orthogonal_q_bands
from ocnir import support_project


__all__ = ["ContinuousScaleBridgeQ", "ContinuousScaleBridgeQComponents"]


class ContinuousScaleBridgeQComponents(NamedTuple):
    prediction_k: Tensor
    base_k: Tensor
    q_k: Tensor
    q_preclosure_k: Tensor
    inherited_q_k: Tensor
    correction_q_middle_k: Tensor
    correction_q_high_k: Tensor
    fine160: Tensor
    encoded80: Tensor
    encoded40: Tensor
    decoded40: Tensor
    decoded80: Tensor
    decoded160: Tensor


class ContinuousScaleBridgeQ(SceneCalibratedContinuousQ):
    """r6a plus one zero-start shared V-cycle scale bridge."""

    schema_version = "g246-continuous-scale-bridge-q-v1"
    scale_path = "r6a:F160->E80->E40->D40->D80->D160"
    band_contract = "q=q_r6a+QM(delta80)+QH(delta160); final Q40 repair"
    new_prefixes = (
        "down80.", "encode80.", "down40.", "encode40.", "bridge40.",
        "up80.", "decode80.", "middle_head.", "high_basis.",
        "high_gate.", "high_value.", "decode160.", "high_head.",
    )

    def __init__(self, *, activation_checkpointing: bool = False) -> None:
        super().__init__(
            fine_channels=52,
            context_dim=19,
            width=48,
            activation_checkpointing=activation_checkpointing,
            no_geo_core=True,
        )
        self.down80 = _Downsample(64, 96)
        self.encode80 = nn.Sequential(_ResidualDW(96), _ResidualDW(96))
        self.down40 = _Downsample(96, 144)
        self.encode40 = nn.Sequential(_ResidualDW(144), _ResidualDW(144))
        self.bridge40 = nn.Conv2d(144, 144, 1)
        self.up80 = _Project(144, 96)
        self.decode80 = _Merge(96, 96, 96)
        self.middle_head = nn.Sequential(
            _ResidualDW(96), nn.Conv2d(96, 1, 3, padding=1)
        )
        self.high_basis = nn.Conv2d(64, 32, 1, bias=False)
        self.high_gate = nn.Conv2d(96, 32, 1, bias=False)
        self.high_value = nn.Conv2d(96, 32, 1, bias=False)
        self.decode160 = nn.Sequential(
            _Project(64, 64),
            _ResidualDW(64, dilation=1),
            _ResidualDW(64, dilation=2),
        )
        self.high_head = nn.Sequential(
            _ResidualDW(64, dilation=1),
            _ResidualDW(64, dilation=2),
            nn.Conv2d(64, 1, 3, padding=1),
        )
        for name, module in self.named_children():
            if any(name == prefix[:-1] for prefix in self.new_prefixes):
                module.apply(IPMRQ._initialize)
        self.reset_bridge_identity()

    def reset_bridge_identity(self) -> None:
        nn.init.zeros_(self.bridge40.weight)
        nn.init.zeros_(self.bridge40.bias)
        nn.init.zeros_(self.middle_head[-1].weight)
        nn.init.zeros_(self.middle_head[-1].bias)
        nn.init.zeros_(self.high_head[-1].weight)
        nn.init.zeros_(self.high_head[-1].bias)

    @property
    def added_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(self.new_prefixes)
        )

    def load_r6a_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Strictly migrate one exact r6a state while retaining new init."""

        if not isinstance(state, Mapping):
            raise TypeError("r6a state must be a mapping")
        reference = SceneCalibratedContinuousQ(
            fine_channels=52, context_dim=19, width=48,
            activation_checkpointing=self.activation_checkpointing,
            no_geo_core=True,
        ).state_dict()
        if set(state) != set(reference):
            missing = sorted(set(reference) - set(state))
            unexpected = sorted(set(state) - set(reference))
            raise ValueError(
                f"r6a state keys differ; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )
        for key, expected in reference.items():
            value = state[key]
            if not isinstance(value, Tensor) or value.shape != expected.shape \
                    or value.dtype != expected.dtype:
                raise ValueError(f"r6a state tensor contract differs at {key!r}")
        migrated = self.state_dict()
        migrated.update({key: value.detach().clone() for key, value in state.items()})
        self.load_state_dict(migrated, strict=True)

    def _condition_high(self, fine160: Tensor, decoded80: Tensor) -> Tensor:
        basis = self.high_basis(fine160)
        gate = torch.tanh(F.interpolate(
            self.high_gate(decoded80), size=fine160.shape[-2:],
            mode="bilinear", align_corners=False,
        ))
        value = F.interpolate(
            self.high_value(decoded80), size=fine160.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        return self.decode160(torch.cat((value, basis * gate), dim=1))

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> ContinuousScaleBridgeQComponents:
        support_bool, available, query = self._validate_inputs(
            fine, coarse_k, support, context, temporal_available, query_index
        )
        effective_context = self._effective_context(context)
        batch, times, channels, height, width = fine.shape
        fine_all, tokens, descriptors = self.date_encoder(
            fine.reshape(batch * times, channels, height, width),
            coarse_k.reshape(batch * times, 1, height // 4, width // 4),
            support_bool.to(dtype=fine.dtype).reshape(
                batch * times, 1, height, width
            ),
            effective_context.reshape(batch * times, self.encoder_context_dim),
            available.reshape(batch * times),
        )
        fine_all = fine_all.reshape(batch, times, 64, height, width)
        tokens = tokens.reshape(batch, times, 144, height // 4, width // 4)
        descriptors = descriptors.reshape(batch, times, 144)
        fine160 = self._gather_query(fine_all, query)
        query_descriptor = self._gather_query(descriptors, query)
        available_float = available.to(dtype=descriptors.dtype)
        set_mean = (descriptors * available_float[..., None]).sum(1)
        set_mean = set_mean / available_float.sum(1, keepdim=True).clamp_min(1.0)
        scene = self.scene_fusion(torch.cat((
            query_descriptor, set_mean, query_descriptor - set_mean,
        ), dim=1))

        encoded80 = self.encode80(self.down80(fine160))
        encoded40 = self.encode40(self.down40(encoded80))
        native40 = self.temporal_fusion(tokens, available, query)
        decoded40 = self.context_unet(native40 + self.bridge40(encoded40))
        raw_q, _weights = self.continuous_head(fine160, decoded40, scene)

        decoded80 = self.decode80(encoded80, self.up80(F.interpolate(
            decoded40, size=encoded80.shape[-2:], mode="bilinear",
            align_corners=False,
        )))
        middle_coefficients = self.middle_head(decoded80)
        decoded160 = self._condition_high(fine160, decoded80)
        high_proposal = self.high_head(decoded160)

        query_support = self._gather_query(support_bool, query)
        query_coarse = self._gather_query(coarse_k, query)
        coarse_valid = torch.isfinite(query_coarse)
        inherited_q = support_project(
            raw_q, torch.zeros_like(query_coarse), query_support, coarse_valid
        ) * query_support
        correction_middle = orthogonal_q_bands(
            lift_p80_coefficients(middle_coefficients, query_support).float(),
            query_support, coarse_valid,
        ).q_middle
        correction_high = orthogonal_q_bands(
            high_proposal.float(), query_support, coarse_valid,
        ).q_high
        correction_preclosure = correction_middle + correction_high
        correction_q = support_project(
            correction_preclosure,
            torch.zeros_like(query_coarse, dtype=correction_preclosure.dtype),
            query_support, coarse_valid,
        ) * query_support
        # Do not numerically re-project the already delivered inherited Q.
        # P(P(q)) is mathematically P(q), but a second float32 block reduction
        # can move a high-amplitude inherited field by multiple ULPs.  Repair
        # only the new correction, whose required coarse observation is zero.
        # This makes zero correction an exact identity while preserving closure.
        q_preclosure = inherited_q.float() + correction_preclosure
        q = inherited_q.float() + correction_q
        query_fine = self._gather_query(fine, query)
        base = support_project(query_fine[:, :1], query_coarse, query_support).float()
        return ContinuousScaleBridgeQComponents(
            prediction_k=base + q,
            base_k=base,
            q_k=q,
            q_preclosure_k=q_preclosure,
            inherited_q_k=inherited_q.float(),
            correction_q_middle_k=correction_middle,
            correction_q_high_k=correction_high,
            fine160=fine160,
            encoded80=encoded80,
            encoded40=encoded40,
            decoded40=decoded40,
            decoded80=decoded80,
            decoded160=decoded160,
        )

    def forward(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> Tensor:
        return self.forward_components(
            fine, coarse_k, support, context, temporal_available, query_index
        ).prediction_k
