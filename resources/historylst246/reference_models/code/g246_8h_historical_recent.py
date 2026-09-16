"""Seven registered historical observations: frozen six plus one recent source.

Only the number and metadata of input slots change. The existing shared layers,
physical correction, and actual-support projection are reused without any new
learned parameters. Recent-source admissibility belongs to the bound input cache.
"""
from g246_8h_historical_multisource import HistoricalMultiSourceNet


class HistoricalRecentInnovationNet(HistoricalMultiSourceNet):
    def __init__(self, backbone, *, width=32, source_count=7, initial_gain=.28,
                 initial_replacement_gain=.22, modality_dropout=.25):
        if isinstance(source_count, bool) or source_count != 7:
            raise ValueError('the recent historical family requires exactly seven registered slots')
        # Build the unchanged shared layers using the parent's existing legal
        # contract. This separate subclass owns its explicit seven-slot forward.
        super().__init__(backbone, width=width, source_count=6, initial_gain=initial_gain,
                         initial_replacement_gain=initial_replacement_gain,
                         modality_dropout=modality_dropout)
        self.source_count = 7

    @property
    def model_config(self):
        return {**super().model_config,
            'schema_version': 'g246-8h-historical-recent-innovation-network-v1',
            'class_name': 'HistoricalRecentInnovationNet',
            'source_count': 7, 'history_shape': ['B', 7, 9, 'H', 'W'],
            'history_date_slots': ['v1_2018', 'v1_2019', 'v1_2020',
                                   'seasonal_2018', 'seasonal_2019', 'seasonal_2020',
                                   'recent_pre_query_8_to_64_days'],
            'recent_selection': 'minimum positive UTC age in [8,64] days after full campaign acquisition alias exclusion; then cloud/datetime/item id',
            'time_contract': '2026 present-day historical replay; acquisition precedes query; historical product availability at query time unproven',
            'recent_interpretation': 'historical thermal spatial template, not an hours-scale dynamical initial state',
            'six_source_reference': 'unchanged shared parameters and forward algebra; zero seventh slot is the six-source function',
            'warmstart': 'strict complete six-source historical wrapper state; no extra learned parameters',
            'parameter_count': self.parameter_count,
            'additional_parameters_vs_six': 0}


__all__ = ['HistoricalRecentInnovationNet']
