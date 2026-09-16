"""Remove only the full-resolution thermal bypass; retain all encoder fusion."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/sub04_20260913'))
from compact_query_product.model import QueryProductCompactHistoryNAF


def zero_returned_summary(module, inputs, output):
    # Fusion has already injected pooled features AND thermal moments into
    # current. Only the summary sent separately to the decoder/readout is cut.
    current_with_history, summary = output
    return current_with_history, torch.zeros_like(summary)


class WithoutThermalBypass(QueryProductCompactHistoryNAF):
    def __init__(self, history_dropout=0., emissivity_dropout=0.):
        super().__init__(history_dropout=history_dropout,
                         emissivity_dropout=emissivity_dropout)
        self.fusion[0].register_forward_hook(zero_returned_summary)
        self.detail_skip.requires_grad_(False)
        self.thermal_gain.requires_grad_(False)


def construct():
    return WithoutThermalBypass()
