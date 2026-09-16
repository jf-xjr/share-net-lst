"""Pure nine-field encoding for a source acquired 8--64 days before a query.

The spatial/radiometric operations intentionally match the frozen pre-2021
encoder. This separate contract accepts actual recent timestamps; it never
rewrites a source date. Acquisition exclusion and grid binding are checked by
the caller, before these six integer rasters are supplied.
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from build_g246_8h_emissivity_cache import encode as encode_emissivity
from g246_8h_historical_features import CONTRACT as PRE2021_CONTRACT, parse_time

CONTRACT = {**{k: v for k, v in PRE2021_CONTRACT.items() if k != 'source_years'},
    'schema': 'g246-8h-recent-historical-feature-encoding-v1',
    'query_years': [2021, 2022, 2023, 2024, 2025],
    'source_age_days_inclusive': [8, 64],
    'source_timestamp': 'actual acquisition datetime, never rewritten to a pre2021 date',
    'caller_contract': 'full 633 G246 acquisition aliases excluded before source selection; exact original query AOI and source radiometric identity audited independently',
    'interpretation': 'recent historical spatial template; not an hours-scale dynamical initial state'}


def encode(raw, historical_datetime, query_datetime):
    historical, query = parse_time(historical_datetime), parse_time(query_datetime)
    age_days = (query - historical).total_seconds() / 86400
    if query.year not in (2021, 2022, 2023, 2024, 2025) or not 8 <= age_days <= 64:
        raise ValueError('recent historical acquisition must be exactly 8--64 days before the query')
    keys = ('lwir11', 'qa', 'qa_pixel', 'qa_radsat', 'emis', 'emsd')
    if any(raw[k].shape != (640, 640) or raw[k].dtype.kind not in 'iu' for k in keys):
        raise ValueError('requires six640x640 recent historical integer rasters')
    st = raw['lwir11'].astype(np.float64) * .00341802 + 149.
    qa = raw['qa'].astype(np.float64) * .01
    contaminated = (raw['qa_pixel'].astype(np.uint32) & 63) != 0
    contaminated = binary_dilation(contaminated, structure=np.ones((3, 3), bool), iterations=1)
    clear = (~contaminated) & (raw['qa_radsat'] == 0) & (raw['lwir11'] > 0) \
        & (st >= 240) & (st <= 380) & (raw['qa'] > 0) & (raw['qa'] <= 300)
    block = lambda a: a.reshape(160, 4, 160, 4)
    count = block(clear).sum((1, 3))
    stats = {'historical_clear_fraction30': float(clear.mean()),
        'historical_nonempty_fraction120': float((count > 0).mean()),
        'historical_clear_count30': int(clear.sum()),
        'historical_datetime': historical_datetime, 'query_datetime': query_datetime}
    if not count.any():
        return np.zeros((9, 160, 160), np.float32), {**stats, 'all_thermal_missing': True}
    mean = lambda x: block(np.where(clear, x, 0)).sum((1, 3)) / count.clip(1)
    temp, uncertainty = mean(st), mean(qa)
    missing = count == 0
    if missing.any():
        donor = distance_transform_edt(missing, return_distances=False, return_indices=True)
        for field in (temp, uncertainty):
            field[missing] = field[tuple(donor[:, missing])]
    parent = temp.reshape(40, 4, 40, 4).mean((1, 3)).repeat(4, 0).repeat(4, 1)
    e, _ = encode_emissivity(raw['emis'], raw['emsd'])
    doy = historical.timetuple().tm_yday
    phase = 2 * np.pi * doy / 365.25
    age = age_days / 3652.5
    constant = lambda x: np.full((160, 160), x, np.float64)
    result = np.stack(((temp - 300) / 20, (temp - parent) / 5, count / 16, uncertainty / 3,
                       e[0], e[3], constant(np.sin(phase)), constant(np.cos(phase)), constant(age))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('recent historical feature encoding produced non-finite values')
    return result, {**stats, 'all_thermal_missing': False}
