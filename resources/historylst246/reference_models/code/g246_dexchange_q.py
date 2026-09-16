"""Direct deep/fine state exchange candidate for G246.

``DExchangeContinuousScaleBridgeQ`` is a single r6a backbone.  It retains the
zero-start 160 -> 80 -> 40 -> 80 -> 160 U1 scale bridge and adds exactly four
direct state exchanges around the sealed 40 -> 20 -> 10 -> 20 -> 40 context
path::

    E80  --support-weighted pool--> E20
    F160 --support-weighted pool--> E10
    D20  -------------------------> D80
    E10  -------------------------> D160

All four exchanges are bias-free, zero-initialised 1x1 projections.  Together
with U1's identity initialisation, strict migration from r6a therefore retains
the registered r6a function at update zero.  The final delivered residual is
still repaired only by the established support-aware Q projection.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_continuous_scale_bridge_q import ContinuousScaleBridgeQ
from g246_q_bands import lift_p80_coefficients, orthogonal_q_bands
from ocnir import support_project


__all__ = [
    "DExchangeContinuousScaleBridgeQ",
    "DExchangeQComponents",
    "ParentContextStates",
    "replay_parent_context",
    "support_weighted_aligned_pool",
]


class ParentContextStates(NamedTuple):
    """The five otherwise-sealed states in the registered parent U-Net."""

    e40: Tensor
    e20: Tensor
    e10: Tensor
    d20: Tensor
    d40: Tensor


class DExchangeQComponents(NamedTuple):
    """Delivered fields, band diagnostics, and explicit multiscale states."""

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
    parent_states: ParentContextStates
    decoded80: Tensor
    decoded160: Tensor


def support_weighted_aligned_pool(
    values: Tensor,
    support: Tensor,
    factor: int,
) -> Tensor:
    """Pool aligned cells with fractional physical-support mass in FP32.

    ``support`` is defined on the same lattice as ``values`` and may be
    fractional (for example S80 is the 2x2 average of binary S160).  Integer,
    non-overlapping blocks preserve lattice alignment.  Empty blocks return
    exactly zero.  The result is cast back to the input feature dtype only
    after the weighted mean has been formed in float32.
    """

    if not isinstance(values, Tensor) or not isinstance(support, Tensor):
        raise TypeError("values and support must be torch tensors")
    if values.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("values/support must have shapes [B,C,H,W]/[B,1,H,W]")
    if values.shape[0] != support.shape[0] or values.shape[-2:] != support.shape[-2:]:
        raise ValueError("values and support must share batch and spatial geometry")
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    height, width = values.shape[-2:]
    if height % factor or width % factor:
        raise ValueError("pool geometry must be divisible by factor")
    if values.device != support.device:
        raise ValueError("values and support must be on the same device")
    if not values.is_floating_point() or not support.is_floating_point():
        raise TypeError("values and support must have floating dtypes")

    support32 = support.float()
    finite = torch.isfinite(support32).all()
    in_range = ((support32 >= 0.0) & (support32 <= 1.0)).all()
    if support32.device.type == "cpu":
        if not bool(finite.item()):
            raise ValueError("support must be finite")
        if not bool(in_range.item()):
            raise ValueError("support must lie in [0,1]")
    else:
        # Device-side assertions preserve the public fail-closed contract
        # without a per-forward CUDA -> CPU synchronisation.
        torch._assert_async(finite, "support must be finite")
        torch._assert_async(in_range, "support must lie in [0,1]")
    return _support_weighted_aligned_pool_unchecked(values, support32, factor)


def _support_weighted_aligned_pool_unchecked(
    values: Tensor,
    support32: Tensor,
    factor: int,
) -> Tensor:
    """Hot-path implementation after geometry/value contracts are known."""

    values32 = values.float()
    # Mask before multiplication so a physically unsupported non-finite value
    # cannot leak through 0 * NaN.  Model features are validated finite, but
    # this keeps the helper's support contract fail-closed in isolation.
    safe_values = torch.where(support32 > 0.0, values32, torch.zeros_like(values32))
    numerator = F.avg_pool2d(
        safe_values * support32, kernel_size=factor, stride=factor
    )
    denominator = F.avg_pool2d(
        support32, kernel_size=factor, stride=factor
    )
    pooled = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    pooled = torch.where(denominator > 0.0, pooled, torch.zeros_like(pooled))
    return pooled.to(dtype=values.dtype)


def replay_parent_context(
    context_unet: nn.Module,
    inputs: Tensor,
    *,
    e80_to_e20: Tensor | None = None,
    f160_to_e10: Tensor | None = None,
) -> ParentContextStates:
    """Replay the registered parent U-Net while exposing its sealed states.

    With both exchanges omitted this is deliberately a line-for-line replay
    of ``_ParentContextUNet.forward``.  The optional tensors are added before
    the corresponding encoder blocks, so learned fine-state messages are
    processed as native E20/E10 state rather than appended at an output head.
    The helper owns no parameters and does not alter ``context_unet``.
    """

    e40 = context_unet.high(inputs)
    e20_input = context_unet.down_middle(e40)
    if e80_to_e20 is not None:
        if e80_to_e20.shape != e20_input.shape:
            raise ValueError("E80->E20 exchange shape differs from native E20 input")
        e20_input = e20_input + e80_to_e20
    e20 = context_unet.middle(e20_input)

    e10_input = context_unet.down_low(e20)
    if f160_to_e10 is not None:
        if f160_to_e10.shape != e10_input.shape:
            raise ValueError("F160->E10 exchange shape differs from native E10 input")
        e10_input = e10_input + f160_to_e10
    e10 = context_unet.low(e10_input)

    d20 = F.interpolate(
        e10, size=e20.shape[-2:], mode="bilinear", align_corners=False
    )
    d20 = context_unet.up_middle(d20)
    d20 = context_unet.decode_middle(
        context_unet.merge_middle(torch.cat((d20, e20), dim=1))
    )
    d40 = F.interpolate(
        d20, size=e40.shape[-2:], mode="bilinear", align_corners=False
    )
    d40 = context_unet.up_high(d40)
    d40 = context_unet.decode_high(
        context_unet.merge_high(torch.cat((d40, e40), dim=1))
    )
    return ParentContextStates(e40=e40, e20=e20, e10=e10, d20=d20, d40=d40)


class DExchangeContinuousScaleBridgeQ(ContinuousScaleBridgeQ):
    """U1/r6a with four zero-start direct deep/fine state exchanges."""

    schema_version = "g246-dexchange-continuous-scale-bridge-q-v1"
    scale_path = (
        "r6a:F160->E80->E40->E20->E10->D20->D40->D80->D160;"
        "direct:E80->E20,F160->E10,D20->D80,E10->D160"
    )
    band_contract = (
        "q=q_r6a+QM(delta80)+QH(delta160); final correction-only Q40 repair"
    )
    exchange_prefixes = (
        "exchange_e80_e20.",
        "exchange_f160_e10.",
        "exchange_d20_d80.",
        "exchange_e10_d160.",
    )

    def __init__(self, *, activation_checkpointing: bool = False) -> None:
        # Do not override U1's ``new_prefixes``: its constructor must initialise
        # only its own registered modules, without dynamic-dispatch side effects.
        super().__init__(activation_checkpointing=activation_checkpointing)
        self.exchange_e80_e20 = nn.Conv2d(96, 208, 1, bias=False)
        self.exchange_f160_e10 = nn.Conv2d(64, 288, 1, bias=False)
        self.exchange_d20_d80 = nn.Conv2d(208, 96, 1, bias=False)
        self.exchange_e10_d160 = nn.Conv2d(288, 64, 1, bias=False)
        self.reset_exchange_identity()

    def reset_exchange_identity(self) -> None:
        """Zero only the four candidate projections; leave inherited U1 alone."""

        for name in self.exchange_prefixes:
            module = getattr(self, name[:-1])
            nn.init.zeros_(module.weight)

    @property
    def exchange_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(self.exchange_prefixes)
        )

    @property
    def added_parameter_count(self) -> int:
        prefixes = self.new_prefixes + self.exchange_prefixes
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(prefixes)
        )

    def _condition_high_with_exchange(
        self,
        fine160: Tensor,
        decoded80: Tensor,
        encoded10: Tensor,
    ) -> Tensor:
        basis = self.high_basis(fine160)
        gate = torch.tanh(F.interpolate(
            self.high_gate(decoded80), size=fine160.shape[-2:],
            mode="bilinear", align_corners=False,
        ))
        value = F.interpolate(
            self.high_value(decoded80), size=fine160.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        conditioned = torch.cat((value, basis * gate), dim=1)
        deep_exchange = F.interpolate(
            self.exchange_e10_d160(encoded10),
            size=fine160.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # The deep message enters the existing D160 feature *before* decode160,
        # rather than bypassing it through a late prediction-head concat.
        return self.decode160(conditioned + deep_exchange)

    def forward_components(
        self,
        fine: Tensor,
        coarse_k: Tensor,
        support: Tensor,
        context: Tensor,
        temporal_available: Tensor,
        query_index: Tensor,
    ) -> DExchangeQComponents:
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

        # Only the selected query's physical support defines both aligned
        # pooling operators.  S80 remains fractional support mass.
        query_support = self._gather_query(support_bool, query)
        support160 = query_support.float()
        support80 = F.avg_pool2d(support160, kernel_size=2, stride=2)
        # _validate_inputs has already established binary finite S160; the
        # aligned average mathematically guarantees fractional S80 in [0,1].
        # Use the unchecked core here to avoid device assertions in training.
        pooled80_to20 = _support_weighted_aligned_pool_unchecked(
            encoded80, support80, factor=4
        )
        pooled160_to10 = _support_weighted_aligned_pool_unchecked(
            fine160, support160, factor=16
        )
        parent_states = replay_parent_context(
            self.context_unet,
            native40 + self.bridge40(encoded40),
            e80_to_e20=self.exchange_e80_e20(pooled80_to20),
            f160_to_e10=self.exchange_f160_e10(pooled160_to10),
        )
        raw_q, _weights = self.continuous_head(
            fine160, parent_states.d40, scene
        )

        decoded40_up = self.up80(F.interpolate(
            parent_states.d40,
            size=encoded80.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ))
        direct20_up = F.interpolate(
            self.exchange_d20_d80(parent_states.d20),
            size=encoded80.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded80 = self.decode80(encoded80, decoded40_up + direct20_up)
        middle_coefficients = self.middle_head(decoded80)
        decoded160 = self._condition_high_with_exchange(
            fine160, decoded80, parent_states.e10
        )
        high_proposal = self.high_head(decoded160)

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
            query_support,
            coarse_valid,
        ) * query_support
        # Diagnostics retain the two proposed correction bands separately;
        # the final q_k below is the delivered inherited-plus-correction field.
        q_preclosure = inherited_q.float() + correction_preclosure
        q = inherited_q.float() + correction_q
        query_fine = self._gather_query(fine, query)
        base = support_project(
            query_fine[:, :1], query_coarse, query_support
        ).float()
        return DExchangeQComponents(
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
            parent_states=parent_states,
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
