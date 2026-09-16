"""Nine registered slots: the frozen first seven plus two recent overpasses.

Shared layers and the physical correction are unchanged. The bound cache owns
source ranking, full-campaign exclusion, and distinct-overpass admissibility.
"""
from g246_8h_historical_multisource import HistoricalMultiSourceNet


class HistoricalRecentMultiInnovationNet(HistoricalMultiSourceNet):
    def __init__(self, backbone, *, width=32, source_count=9, initial_gain=.28,
                 initial_replacement_gain=.22, modality_dropout=.25):
        if isinstance(source_count, bool) or source_count != 9:
            raise ValueError('the multi-recent family requires exactly nine registered slots')
        super().__init__(backbone, width=width, source_count=6, initial_gain=initial_gain,
                         initial_replacement_gain=initial_replacement_gain,
                         modality_dropout=modality_dropout)
        self.source_count = 9

    @property
    def model_config(self):
        return {**super().model_config,
            'schema_version': 'g246-8h-historical-recent-multi-innovation-network-v1',
            'class_name': 'HistoricalRecentMultiInnovationNet',
            'source_count': 9, 'history_shape': ['B', 9, 9, 'H', 'W'],
            'history_date_slots': ['v1_2018', 'v1_2019', 'v1_2020',
                                   'seasonal_2018', 'seasonal_2019', 'seasonal_2020',
                                   'recent_rank1', 'recent_rank2', 'recent_rank3'],
            'recent_selection': 'same age/cloud/datetime/item ranking in UTC age [8,64] days after full campaign acquisition alias exclusion; select first three distinct platform/path/UTC-date overpasses',
            'first_seven_contract': 'all first seven cached observations and their order remain unchanged',
            'time_contract': '2026 present-day historical replay; acquisition precedes query; historical product availability at query time unproven',
            'recent_interpretation': 'historical thermal spatial templates, not hours-scale dynamical initial states',
            'warmstart': 'strict complete six- or seven-source shared wrapper state; no extra learned parameters',
            'parameter_count': self.parameter_count,
            'additional_parameters_vs_six_or_seven': 0}


__all__ = ['HistoricalRecentMultiInnovationNet']
