"""Fixed physical encoding of independent pre-2021 thermal observations.

No query array or mask is an argument. Historical radiometric and cloud quality
controls describe the historical observation only. These are nominal 120 m
aggregates of a 30 m delivered, native approximately 100 m thermal product.
"""
from datetime import datetime

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from build_g246_8h_emissivity_cache import encode as encode_emissivity

CHANNELS = [
    'filled_historical_meanT_minus300_div20',
    'filled_historical_T_parent_anomaly_div5',
    'historical_clear_coverage',
    'filled_historical_mean_ST_QA_div3',
    'historical_joint_meanE_minus098_div001',
    'historical_joint_emissivity_coverage',
    'historical_doy_sin', 'historical_doy_cos', 'positive_age_days_div3652p5',
]
CONTRACT = {
    'schema':'g246-8h-historical-feature-encoding-v1',
    'source_years':[2018,2019,2020], 'channels':CHANNELS,
    'temperature_scale':0.00341802,'temperature_offset_k':149.,
    'st_qa_scale_k':.01,
    'historical_clear_rule':'ST_DN>0,240<=T<=380,0<ST_QA_DN<=300,QA_RADSAT=0; reject QA_PIXEL bits0-5 plus one-pixel 3x3 dilation',
    'thermal_aggregation':'historical-clear count-weighted 4x4 mean; one or more clear cells allowed, count/16 exposed',
    'thermal_gap_fill':'shared nearest nonempty historical120 cell for T and ST_QA; no query arrays',
    'historical_parent_anomaly':'filled T minus unweighted4x4 filled-T parent mean; a predictor, not query closure',
    'historical_emissivity':'same independently audited joint E/EMSD mask and nearest fill as emissivity_v1',
    'all_thermal_missing_date':'all nine channels zero; no source-date metadata-only cue',
    'time_contract':'present-day2026 historical replay; historical public availability unproven',
    'interpretation':'wide-season static historical-state proxy, not a short-time initial condition',
    'query_target_or_mask_arguments':False,
}


def parse_time(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('source and query times require explicit timezone')
    return result


def encode(raw, historical_datetime, query_datetime):
    """Encode six historical integer rasters into nine scalar120 fields."""
    historical, query = parse_time(historical_datetime), parse_time(query_datetime)
    if historical.year not in (2018,2019,2020) or query.year<2021 or historical>=query:
        raise ValueError('historical acquisition boundary violated')
    keys=('lwir11','qa','qa_pixel','qa_radsat','emis','emsd')
    if any(raw[k].shape!=(640,640) or raw[k].dtype.kind not in 'iu' for k in keys):
        raise ValueError('requires six640x640 historical integer rasters')
    st=raw['lwir11'].astype(np.float64)*.00341802+149.
    qa=raw['qa'].astype(np.float64)*.01
    contaminated=(raw['qa_pixel'].astype(np.uint32)&63)!=0
    contaminated=binary_dilation(contaminated,structure=np.ones((3,3),bool),iterations=1)
    clear=(~contaminated)&(raw['qa_radsat']==0)&(raw['lwir11']>0) \
        &(st>=240)&(st<=380)&(raw['qa']>0)&(raw['qa']<=300)
    block=lambda a:a.reshape(160,4,160,4)
    count=block(clear).sum((1,3))
    stats={'historical_clear_fraction30':float(clear.mean()),
           'historical_nonempty_fraction120':float((count>0).mean()),
           'historical_clear_count30':int(clear.sum()),
           'historical_datetime':historical_datetime,'query_datetime':query_datetime}
    if not count.any():
        return np.zeros((9,160,160),np.float32),{**stats,'all_thermal_missing':True}
    mean=lambda x:block(np.where(clear,x,0)).sum((1,3))/count.clip(1)
    temp,uncertainty=mean(st),mean(qa)
    missing=count==0
    if missing.any():
        donor=distance_transform_edt(missing,return_distances=False,return_indices=True)
        for field in (temp,uncertainty):
            field[missing]=field[tuple(donor[:,missing])]
    parent=temp.reshape(40,4,40,4).mean((1,3)).repeat(4,0).repeat(4,1)
    e,_=encode_emissivity(raw['emis'],raw['emsd'])
    doy=historical.timetuple().tm_yday
    phase=2*np.pi*doy/365.25
    age=(query-historical).total_seconds()/86400/3652.5
    constant=lambda x:np.full((160,160),x,np.float64)
    result=np.stack(((temp-300)/20,(temp-parent)/5,count/16,uncertainty/3,
                     e[0],e[3],constant(np.sin(phase)),constant(np.cos(phase)),constant(age))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('historical feature encoding produced non-finite values')
    return result,{**stats,'all_thermal_missing':False}


def physical_template(history):
    """Coverage-weighted historical anomalies before query-support projection."""
    value=np.asarray(history,np.float64)
    if value.ndim!=4 or value.shape[:2]!=(3,9):
        raise ValueError('history must be three dates by nine scalar fields')
    coverage=value[:,2]
    weights=coverage.sum(0)
    return np.divide((value[:,1]*5*coverage).sum(0),weights,
                     out=np.zeros_like(weights),where=weights>0)
