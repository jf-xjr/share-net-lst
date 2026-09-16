"""Query-relative three-date network for the bounded G246 experiment.

Every date supplies only registered predictors, its own coarse observation and
physical support.  Auxiliary fine targets and evaluation masks are absent from
the forward interface.  Auxiliary-date ordering is immaterial.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from g246_8h_network import G246EightHourNet, _Block, _norm, support_project


class _DateEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(56, width, 3, padding=1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )
        self.down_half = nn.Sequential(
            nn.Conv2d(width, 2 * width, 2, stride=2, bias=False),
            _norm(2 * width), nn.SiLU(), _Block(2 * width),
        )
        self.parent = nn.Sequential(
            nn.Conv2d(2 * width, 3 * width, 2, stride=2, bias=False),
            _norm(3 * width), nn.SiLU(), _Block(3 * width),
        )
        self.context = nn.Linear(15, 6 * width)
        self.decode = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, bias=False),
            _norm(width), nn.SiLU(), _Block(width),
        )

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor) -> Tensor:
        mask = support.float()
        base = fine[:, :1].float()
        scaled = torch.cat(((base - 300.0) / 20.0, fine[:, 1:].float()), dim=1) * mask
        fraction = F.avg_pool2d(mask, 4, 4)
        parent_base = F.avg_pool2d(base * mask, 4, 4) / fraction.clamp_min(1.0 / 16.0)
        valid = torch.isfinite(coarse) & (fraction > 0)
        safe = torch.where(valid, coarse.float(), parent_base)
        coarse_features = F.interpolate(torch.cat(
            ((safe - 300.0) / 20.0, valid.float(), (safe - parent_base) / 20.0),
            dim=1), scale_factor=4, mode="nearest")
        fine_embedding = self.stem(torch.cat((scaled, mask, coarse_features), dim=1)) * mask
        parent = self.parent(self.down_half(fine_embedding))
        scale, shift = self.context(context).tanh().chunk(2, dim=1)
        parent = parent * (1.0 + 0.25 * scale[:, :, None, None]) \
            + 0.25 * shift[:, :, None, None]
        parent = F.interpolate(parent, size=fine.shape[-2:], mode="bilinear",
                               align_corners=False)
        return self.decode(torch.cat((fine_embedding, parent), dim=1))


class G246EightHourTemporalNet(nn.Module):
    """An unfrozen full-field backbone plus learned query-specific date fusion.

    The temporal path changes the complete query feature field and supplies an
    independent dense Kelvin field.  Neither action is constrained to a fixed
    output dictionary, scalar mixture or small-amplitude residual span.  The
    zero heads choose a recoverable query-only initial function; they do not
    freeze that function or restrict the final coordinate dimension.
    """

    def __init__(self, width: int = 48, temporal_width: int = 32) -> None:
        super().__init__()
        if temporal_width < 8 or temporal_width % 8:
            raise ValueError("temporal_width must be a multiple of eight")
        self.width = int(width)
        self.temporal_width = int(temporal_width)
        self.query_network = G246EightHourNet(width=width)
        self.date_encoder = _DateEncoder(temporal_width)
        self.context_difference = nn.Sequential(nn.Linear(15, 8), nn.Tanh())
        # Query, auxiliary and their difference; relative physical state;
        # observed coarse difference, joint coarse availability, local support.
        incoming = 3 * temporal_width + 8 + 3
        self.interaction = nn.Sequential(
            nn.Conv2d(incoming, temporal_width, 1, bias=False),
            _norm(temporal_width), nn.SiLU(), _Block(temporal_width),
        )
        self.gate = nn.Conv2d(temporal_width, 1, 1)
        self.fusion = nn.Sequential(_Block(temporal_width), _Block(temporal_width))
        self.feature_head = nn.Conv2d(temporal_width, 52, 1)
        self.field_head = nn.Conv2d(temporal_width, 1, 3, padding=1)
        for head in (self.feature_head, self.field_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.parameter_count >= 8_000_000:
            raise ValueError("the three-date model must remain below eight million parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "schema_version": "g246-8h-temporal-v1",
            "family": "temporal", "width": self.width,
            "temporal_width": self.temporal_width,
            "parameter_count": self.parameter_count,
            "query_parameter_count": self.query_network.parameter_count,
            "query_model_spec": self.query_network.model_config,
            "dates": 3, "auxiliary_target_inputs": False,
            "geographic_inputs": False,
            "input_contract": "three registered Fine52/Context15/coarse/support dates",
            "temporal_contract": "query-relative auxiliary permutation-invariant fusion",
            "initialization": "exact inherited query-only function; zero temporal heads",
            "all_parameters_trainable": True,
            "residual_amplitude_cap": None,
            "projection": "query actual-support affine field plus projected dense tangent",
        }

    def initialize_query_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load a single-date deployment state; retain fresh temporal parameters."""
        self.query_network.load_state_dict(state_dict, strict=True)

    def forward(self, fine: Tensor, coarse: Tensor, support: Tensor,
                context: Tensor, query_index: Tensor) -> Tensor:
        if fine.ndim != 5 or fine.shape[1:3] != (3, 52):
            raise ValueError("fine must have shape [B,3,52,H,W]")
        batch, dates, _, height, width = fine.shape
        if support.shape != (batch, dates, 1, height, width):
            raise ValueError("support must have shape [B,3,1,H,W]")
        if coarse.shape != (batch, dates, 1, height // 4, width // 4):
            raise ValueError("coarse must have shape [B,3,1,H/4,W/4]")
        if context.shape != (batch, dates, 15) or query_index.shape != (batch,):
            raise ValueError("context must be [B,3,15] and query_index [B]")
        row = torch.arange(batch, device=fine.device)
        query = query_index.to(device=fine.device, dtype=torch.long)
        query_fine, query_coarse = fine[row, query], coarse[row, query]
        query_support, query_context = support[row, query], context[row, query]

        encoded = self.date_encoder(
            fine.reshape(batch * dates, 52, height, width),
            coarse.reshape(batch * dates, 1, height // 4, width // 4),
            support.reshape(batch * dates, 1, height, width),
            context.reshape(batch * dates, 15),
        ).reshape(batch, dates, self.temporal_width, height, width)
        query_encoded = encoded[row, query][:, None].expand_as(encoded)
        relative_context = self.context_difference(context - query_context[:, None])
        relative_context = relative_context[..., None, None].expand(-1, -1, -1, height, width)
        coarse_joint = torch.isfinite(coarse) & torch.isfinite(query_coarse[:, None])
        # Replace each missing operand before subtraction to avoid NaN branches
        # entering autodiff, even though coarse is an observed, nonlearned input.
        coarse_delta = (torch.nan_to_num(coarse.float())
                        - torch.nan_to_num(query_coarse[:, None].float())) / 20.0
        coarse_delta = torch.where(coarse_joint, coarse_delta, torch.zeros_like(coarse_delta))
        coarse_pair = F.interpolate(torch.cat((coarse_delta, coarse_joint.float()), dim=2)
            .reshape(batch * dates, 2, height // 4, width // 4),
            size=(height, width), mode="nearest").reshape(batch, dates, 2, height, width)
        pair = torch.cat((query_encoded, encoded, encoded - query_encoded,
                          relative_context, coarse_pair, support.float()), dim=2)
        interaction = self.interaction(pair.reshape(batch * dates, -1, height, width))
        gate = self.gate(interaction).sigmoid().reshape(batch, dates, 1, height, width)
        interaction = interaction.reshape(batch, dates, self.temporal_width, height, width)
        is_auxiliary = torch.arange(dates, device=fine.device)[None, :] != query[:, None]
        available = support.float() * is_auxiliary[:, :, None, None, None]
        count = available.sum(dim=1)
        mixed = (interaction * gate * available).sum(dim=1) / count.clamp_min(1.0)
        fused = self.fusion(mixed)
        available_pixel = (count > 0).to(dtype=fused.dtype)
        feature_delta = self.feature_head(fused).float() * available_pixel * query_support.float()
        # The head works in the normalized feature units of the date encoder.
        feature_delta = torch.cat((20.0 * feature_delta[:, :1], feature_delta[:, 1:]), dim=1)
        query_prediction = self.query_network(query_fine + feature_delta,
                                               query_coarse, query_support, query_context)
        field_delta = self.field_head(fused).float() * available_pixel
        zero_coarse = torch.where(torch.isfinite(query_coarse),
                                  torch.zeros_like(query_coarse),
                                  torch.full_like(query_coarse, float("nan")))
        tangent = support_project(field_delta, zero_coarse.float(), query_support)
        # Both terms have the correct query-support semantics.  Avoid a second
        # float32 full-field repair here so zero heads reproduce the inherited
        # query prediction bit-for-bit; final persistence uses float64 repair.
        return query_prediction.float() + tangent.float()


def build_model(model_spec: dict[str, object]) -> G246EightHourTemporalNet:
    if model_spec.get("schema_version") not in (None, "g246-8h-temporal-v1") \
            or model_spec.get("family") != "temporal":
        raise ValueError("unsupported three-date model specification")
    model = G246EightHourTemporalNet(width=int(model_spec.get("width", 48)),
                                   temporal_width=int(model_spec.get("temporal_width", 32)))
    if "parameter_count" in model_spec and model.parameter_count != model_spec["parameter_count"]:
        raise ValueError("temporal model parameter count differs")
    return model


__all__ = ["G246EightHourTemporalNet", "build_model"]
