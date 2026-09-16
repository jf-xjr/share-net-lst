# SHaRe-Net manuscript configuration

Repository **`share-net-lst`**, source release **`v1.0.0-manuscript`**. This is the configuration entry point for Sections II, III-E and IV-A of *Accurate and Efficient Urban Land Surface Temperature Reconstruction with Multiscale Historical Fusion*. The accompanying archive is `share-net-lst-v1.0.0-manuscript.zip`; open this file under `share-net-lst/docs/` after extraction. [Release metadata](../release.json) identifies the prepared source package. All links below resolve within this repository.

## Final-system training — Section III-E

The table describes the five executed stages for each of the three final student lineages. Learning rates are the rates reached at the end of linear warm-up, before cosine decay. Batch size counts complete 160 × 160 scenes per optimizer update.

| Stage | Optimizer | Learning rate | Effective batch | Updates | Teacher weight λ | Warm-up updates | Cosine floor / peak |
|---|---|---|---:|---:|---|---:|---:|
| Reference training | AdamW | 1e-3 | 4 | 18,000 | 0 | 300 | 0.05 |
| First-teacher refinement | AdamW | 1e-4 | 4 | 1,000 | 0.5 | 100 | 0.05 |
| Four-head refinement | AdamW | 1e-4 | 4 | 3,000 | 0.9 | 100 | 0.05 |
| Compact recovery | AdamW | 1e-4 | 4 | 6,000 | 0.9 − 0.8u/6000 at update u | 100 | 0.05 |
| Query refinement | AdamW | 5e-5 inherited; 5e-4 new query projections | 4 (2 × 2 microbatches) | 1,500 | 0.5 | 50 | 0.10 |

All stages use weight decay 1e-4, gradient-norm clipping at 1, and EMA with maximum decay 0.995. Each stage initializes a fresh optimizer and EMA from its selected parent. The recovery schedule uses u = 1,…,6000; its nominal endpoints are 0.9 and 0.1. The loss is `(1−λ) × scene_RMSE(prediction, reference; urban mask) + λ × scene_RMSE(prediction, teacher; observation support)`, with 1e-6 inside each scene's square root. Initial and first-teacher stages use history/emissivity dropout 0.25; subsequent stages use zero dropout. Final refinement uses FP32 and disables TF32.

The first fixed teacher averages eight spatial views of one reference-trained model; the second averages eight views of each of three refined models. All three student lineages share these teachers. [training.md](training.md) describes the initialization maps and teacher construction. [training_lineage.json](training_lineage.json) contains every stage's executed configuration, completed update count, selected checkpoint hash and bound run record. The table's settings are common across the three lineages; seeds and selected checkpoints remain in those existing records.

## Current input: 74 channels — Section II

The network concatenates **52 `fine` + 4 `emissivity` + 1 `support` + 2 coarse fields + 15 `context` = 74** channels, in that order. Context scalars are broadcast over the 160 × 160 grid; coarse temperature and coarse validity use nearest-neighbor expansion from 40 × 40. The stored `fine[0]` is kelvin and becomes `(B−300)/20` on input. Missing coarse temperature is zero after normalization, accompanied by the separate finite-value indicator. The input assembly is explicit in [the final model's inherited encoder](../research/sub04_20260911/naf_history/model.py) and [the U-TAE adapter](../resources/historylst246/historylst/model.py).

The complete ordered channel dictionary is below (zero-based indices). Names match [manifest.json](../resources/historylst246/manifest.json); the two coarse fields and support are assembled at inference.

| Index | Channel | Packet location | Definition |
|---:|---|---|---|
| 0 | `base_lst_k` | `fine[0]` | (B−300)/20 at network input; stored B is kelvin |
| 1 | `interpolation_weight` | `fine[1]` | Coarse interpolation validity weight |
| 2 | `blue_global_z` | `fine[2]` | Band mean, training z-score |
| 3 | `green_global_z` | `fine[3]` | Band mean, training z-score |
| 4 | `red_global_z` | `fine[4]` | Band mean, training z-score |
| 5 | `nir_global_z` | `fine[5]` | Band mean, training z-score |
| 6 | `swir1_global_z` | `fine[6]` | Band mean, training z-score |
| 7 | `swir2_global_z` | `fine[7]` | Band mean, training z-score |
| 8 | `blue_scene_iqr` | `fine[8]` | Band mean, scene median/IQR anomaly |
| 9 | `green_scene_iqr` | `fine[9]` | Band mean, scene median/IQR anomaly |
| 10 | `red_scene_iqr` | `fine[10]` | Band mean, scene median/IQR anomaly |
| 11 | `nir_scene_iqr` | `fine[11]` | Band mean, scene median/IQR anomaly |
| 12 | `swir1_scene_iqr` | `fine[12]` | Band mean, scene median/IQR anomaly |
| 13 | `swir2_scene_iqr` | `fine[13]` | Band mean, scene median/IQR anomaly |
| 14 | `ndvi` | `fine[14]` | Spectral index; formula below |
| 15 | `ndbi` | `fine[15]` | Spectral index; formula below |
| 16 | `mndwi` | `fine[16]` | Spectral index; formula below |
| 17 | `bsi` | `fine[17]` | Spectral index; formula below |
| 18 | `optical_valid` | `fine[18]` | Indicator: optical coverage >0 |
| 19 | `built_fraction` | `fine[19]` | Land-cover fraction in [0,1] |
| 20 | `water_fraction` | `fine[20]` | Land-cover fraction in [0,1] |
| 21 | `lulc_coverage` | `fine[21]` | Land-cover fraction in [0,1] |
| 22 | `texture_blue_std_fit_z` | `fine[22]` | Within-cell optical statistic, training z-score |
| 23 | `texture_blue_q25_fit_z` | `fine[23]` | Within-cell optical statistic, training z-score |
| 24 | `texture_blue_q75_fit_z` | `fine[24]` | Within-cell optical statistic, training z-score |
| 25 | `texture_green_std_fit_z` | `fine[25]` | Within-cell optical statistic, training z-score |
| 26 | `texture_green_q25_fit_z` | `fine[26]` | Within-cell optical statistic, training z-score |
| 27 | `texture_green_q75_fit_z` | `fine[27]` | Within-cell optical statistic, training z-score |
| 28 | `texture_red_std_fit_z` | `fine[28]` | Within-cell optical statistic, training z-score |
| 29 | `texture_red_q25_fit_z` | `fine[29]` | Within-cell optical statistic, training z-score |
| 30 | `texture_red_q75_fit_z` | `fine[30]` | Within-cell optical statistic, training z-score |
| 31 | `texture_nir08_std_fit_z` | `fine[31]` | Within-cell optical statistic, training z-score |
| 32 | `texture_nir08_q25_fit_z` | `fine[32]` | Within-cell optical statistic, training z-score |
| 33 | `texture_nir08_q75_fit_z` | `fine[33]` | Within-cell optical statistic, training z-score |
| 34 | `texture_swir16_std_fit_z` | `fine[34]` | Within-cell optical statistic, training z-score |
| 35 | `texture_swir16_q25_fit_z` | `fine[35]` | Within-cell optical statistic, training z-score |
| 36 | `texture_swir16_q75_fit_z` | `fine[36]` | Within-cell optical statistic, training z-score |
| 37 | `texture_swir22_std_fit_z` | `fine[37]` | Within-cell optical statistic, training z-score |
| 38 | `texture_swir22_q25_fit_z` | `fine[38]` | Within-cell optical statistic, training z-score |
| 39 | `texture_swir22_q75_fit_z` | `fine[39]` | Within-cell optical statistic, training z-score |
| 40 | `texture_ndvi_mean_fit_z` | `fine[40]` | Within-cell optical statistic, training z-score |
| 41 | `texture_ndvi_std_fit_z` | `fine[41]` | Within-cell optical statistic, training z-score |
| 42 | `texture_ndvi_q25_fit_z` | `fine[42]` | Within-cell optical statistic, training z-score |
| 43 | `texture_ndvi_q75_fit_z` | `fine[43]` | Within-cell optical statistic, training z-score |
| 44 | `texture_ndbi_mean_fit_z` | `fine[44]` | Within-cell optical statistic, training z-score |
| 45 | `texture_ndbi_std_fit_z` | `fine[45]` | Within-cell optical statistic, training z-score |
| 46 | `texture_ndbi_q25_fit_z` | `fine[46]` | Within-cell optical statistic, training z-score |
| 47 | `texture_ndbi_q75_fit_z` | `fine[47]` | Within-cell optical statistic, training z-score |
| 48 | `texture_mndwi_mean_fit_z` | `fine[48]` | Within-cell optical statistic, training z-score |
| 49 | `texture_mndwi_std_fit_z` | `fine[49]` | Within-cell optical statistic, training z-score |
| 50 | `texture_mndwi_q25_fit_z` | `fine[50]` | Within-cell optical statistic, training z-score |
| 51 | `texture_mndwi_q75_fit_z` | `fine[51]` | Within-cell optical statistic, training z-score |
| 52 | `emissivity_mean_scaled` | `emissivity[0]` | (filled_mean_emissivity-.98)/.01 |
| 53 | `emissivity_std_scaled` | `emissivity[1]` | std_emissivity/.01 |
| 54 | `emissivity_uncertainty_log` | `emissivity[2]` | log1p(mean_EMSD/.01) |
| 55 | `emissivity_coverage` | `emissivity[3]` | availability_fraction |
| 56 | `observation_support` | `support[0]` | Supplied binary current thermal support |
| 57 | `coarse_temperature_scaled` | `coarse[0]` | Nearest-expanded (C−300)/20; zero where missing |
| 58 | `coarse_validity` | `isfinite(coarse[0])` | Nearest-expanded finite-coarse indicator |
| 59 | `power_tmax_z` | `context[0]` | Legacy POWER daily maximum temperature z-score |
| 60 | `doy_sin` | `context[1]` | Current DOY phase; formula below |
| 61 | `doy_cos` | `context[2]` | Current DOY phase; formula below |
| 62 | `platform_l8` | `context[3]` | Platform indicator |
| 63 | `platform_l9` | `context[4]` | Platform indicator |
| 64 | `cos_solar_zenith` | `context[5]` | Acquisition-time solar geometry |
| 65 | `solar_azimuth_sin` | `context[6]` | Acquisition-time solar geometry |
| 66 | `solar_azimuth_cos` | `context[7]` | Acquisition-time solar geometry |
| 67 | `power_daily_t2m_max_fit_z` | `context[8]` | POWER daily weather, training z-score |
| 68 | `power_daily_t2m_min_fit_z` | `context[9]` | POWER daily weather, training z-score |
| 69 | `power_daily_rh2m_fit_z` | `context[10]` | POWER daily weather, training z-score |
| 70 | `power_daily_ws2m_fit_z` | `context[11]` | POWER daily weather, training z-score |
| 71 | `power_daily_allsky_sfc_sw_dwn_fit_z` | `context[12]` | POWER daily weather, training z-score |
| 72 | `power_daily_prectotcorr_fit_z` | `context[13]` | POWER daily weather, training z-score |
| 73 | `power_daily_gwettop_fit_z` | `context[14]` | POWER daily weather, training z-score |

`global_z` and `fit_z` mean `(value − training_mean) / training_std`. Numeric statistics are already provided in [optical_normalization.json](../resources/historylst246/provenance/optical_normalization.json), [texture_normalization.json](../resources/historylst246/provenance/texture_normalization.json) and [weather_normalization.json](../resources/historylst246/provenance/weather_normalization.json). These records use training-only, region/city/scene-balanced statistics. The texture JSON's `channels[name].mean` and `.std` give each feature's parameters; its additional per-city records need not be read to recover the configuration.

The six optical bands are B2–B7, in blue, green, red, NIR, SWIR1, SWIR2 order. Reflectance is `2.75e-5 × DN − 0.2`, averaged over accepted 30 m children. `scene_iqr` uses each band's scene median and `max(q75−q25, 1e-6)`, clipped to [−8, 8]. NDVI, NDBI, MNDWI and BSI use `(NIR−red)/(NIR+red)`, `(SWIR1−NIR)/(SWIR1+NIR)`, `(green−SWIR1)/(green+SWIR1)` and `(SWIR1+red−NIR−blue)/(SWIR1+red+NIR+blue)`. Ratios use zero when the denominator magnitude is at most 1e-6 and are clipped to [−1, 1]. Optical means, normalized bands and indices are zero where optical coverage is absent.

Texture features summarize the accepted 30 m children within each 120 m cell: standard deviation and 25th/75th percentiles for each reflectance band, followed by mean, standard deviation and 25th/75th percentiles for NDVI, NDBI and MNDWI. Their normalized values are zero for empty cells. Built/water fractions and land-cover coverage are clipped to [0, 1]. The interpolation-weight channel is the bilinearly expanded validity of the coarse thermal field. The base temperature divides interpolated supported temperature by this weight; weights below 0.5 trigger filling from the containing valid coarse value, otherwise the median finite interpolant. [g246_r2_data.py](../resources/historylst246/reference_models/code/g246_r2_data.py) defines these transforms and the acquisition-time solar geometry.

The four current emissivity channels are `(filled_mean_E−0.98)/0.01`, `std_E/0.01`, `log1p(mean_EMSD/0.01)` and accepted fraction. Emissivity and its uncertainty are scaled by 1e-4 from their digital numbers; the separate mask and filling are defined below. Current seasonal phase is `2π(DOY−1)/365.2425`. The two platform indicators identify Landsat 8/9. POWER daily weather fields are maximum/minimum 2 m air temperature, relative humidity, wind speed, downward shortwave radiation, corrected precipitation and top-layer soil wetness. Maximum temperature is retained twice, once in the legacy context and once in the separately normalized weather extension. The retained 15-channel context excludes the four latitude/longitude sine/cosine fields present in the older 19-channel builder.

## Nine historical slots and channels — Section II

| Zero-based slots | Candidate pool | Ranking / retained record |
|---|---|---|
| 0, 1, 2 | Landsat 8, May–September of 2018, 2019, 2020 respectively; catalog cloud ≤60% | One per year; ascending cloud percentage, acquisition timestamp, item identity |
| 3, 4, 5 | Same yearly/seasonal pools; catalog cloud ≤40% | Ascending circular month/day distance to the target date, then cloud, timestamp, identity |
| 6, 7, 8 | Landsat 8/9 Tier-1 L2SP, 8–64 days before target; catalog cloud ≤40%, full crop coverage, required thermal/QA/emissivity assets | First three distinct acquisitions in ascending age, cloud, timestamp, identity order |

Seasonal distance maps both month/day pairs to the common non-leap year 2023 and takes `min(abs(day_difference), 365−abs(day_difference))`. If a seasonal winner duplicates that year's first archival record, its slot stays zero; the selector does not substitute a lower-ranked acquisition. Recent candidates exclude target acquisition identities and equivalent aliases before ranking and are deduplicated by overpass identity. Unavailable records keep their fixed slots. Per-scene selected identities and timestamps are in the dataset manifest.

Selection is preserved in the [six-slot contract](../resources/historylst246/reference_models/code/g246_8h_historical_six_contract.py), [recent-source ranking](../resources/historylst246/reference_models/code/g246_8h_recent_source_contract.py) and [three-recent-slot contract](../resources/historylst246/reference_models/code/g246_8h_recent_pair_source_contract.py). The source-selection window is defined using catalog timestamps; the encoded recorded ages span approximately 7.9–64.1 days after timestamp conversion, as documented in the existing [experiment conditions](experiment_conditions.md).

Each slot contains the following nine channels. The history is flattened in slot-major, then channel-major order only for MoCoLSK/THSTNet guidance.

| Within-slot index | Definition |
|---:|---|
| 0 | `(filled_T_K−300)/20` |
| 1 | `(filled_T_K − filled_parent_mean_K)/5`; the parent is the unweighted mean of the filled 4 × 4 study cells |
| 2 | Accepted historical thermal children / 16 |
| 3 | Nearest-filled mean ST_QA in kelvin / 3 |
| 4 | `(filled_mean_emissivity−0.98)/0.01` |
| 5 | Accepted emissivity children / 16 |
| 6, 7 | `sin(2π DOY/365.25)`, `cos(2π DOY/365.25)` of the historical acquisition |
| 8 | `(target_time−history_time)` in days / 3652.5 |

Before encoding, thermal coverage masks channels 0, 1 and 3; emissivity coverage masks channel 4. A date with no valid thermal observation anywhere has nine zero channels, including its metadata. Temperature and quality fill from the nearest nonempty study cell; coverage remains the original accepted fraction. [Historical preprocessing](../resources/historylst246/reference_models/code/g246_8h_historical_features.py) defines the exact formulas, with the same channel meanings for [recent records](../resources/historylst246/reference_models/code/g246_8h_recent_historical_features.py).

## Quality and support thresholds — Section II

| Field | Executed acceptance rule |
|---|---|
| Current reference, 30 m | Reject QA_PIXEL bits 0–5 and one pixel of dilation using SciPy's default cross neighborhood; require QA_RADSAT = 0, nonzero thermal DN and all six optical DNs, 240 ≤ T ≤ 380 K, ST_QA_DN ≤300 (scale 0.01 K). The current implementation imposes no separate positive-ST_QA test. |
| Current aerosol | Reject fill bit 0; require valid bit 1 or interpolated bit 5; reject high aerosol level `(QA >> 6) & 3 == 3`. |
| Current optical | Use stored QA-reason bits 0,1,2,3,4,5,9 (cloud/dilation, saturation, aerosol and zero reflectance) and nonzero B2–B7. Thermal DN, thermal range and ST_QA reason bits are excluded. A 120 m optical cell is nonempty when at least one child is accepted. |
| Historical thermal, 30 m | Require ST_DN >0, 240 ≤ T ≤380 K, 0 < ST_QA_DN ≤300, QA_RADSAT =0; reject QA_PIXEL bits 0–5 with one 3 × 3 dilation. Historical thermal screening has no aerosol criterion. |
| Current/historical emissivity | Joint mask `6000 ≤ EMIS_DN ≤10000` and `0 < EMSD_DN ≤10000`; both scale by 1e-4. At least one accepted child gives an observed study cell. Continuous fills use the nearest valid donor and retain zero coverage for unobserved cells. |
| Current 30 →120 m thermal aggregation | At least 12 of 16 accepted children; average accepted values. |
| Current 120 →480 m coarse aggregation | At least 12 of 16 supported children; average supported values. |
| Historical 30 →120 m thermal aggregation | At least 1 accepted child; average accepted values and retain count /16 as coverage. |

The current QA rules come from [acquire_effective120_v2.py](../resources/historylst246/reference_models/code/acquire_effective120_v2.py); optical QA is applied by [build_landsat30_texture_sidecars.py](../resources/historylst246/reference_models/code/build_landsat30_texture_sidecars.py). Emissivity rules are in [build_g246_8h_emissivity_cache.py](../resources/historylst246/reference_models/code/build_g246_8h_emissivity_cache.py). These source files and the [processing provenance](../resources/historylst246/provenance/processing_provenance.json) retain the executed distinctions between the current and historical masks.

Observation `support` and the urban scoring `formal` mask are separate supplied arrays. Only `support` enters reconstruction and coarse projection. The unchanged dataset manifest identifies both arrays and their checksums. Conditional evaluation intersects `formal` with its input-defined subset and requires at least 32 scored urban pixels per scene; its definitions remain in [experiment_conditions.md](experiment_conditions.md).

## Baseline input adaptations — Section IV-A

**Historical U-TAE.** Separate 3 × 3 stems map the 74 current fields and each nine-channel historical observation to 32 channels. The current predictor-only token is concatenated with nine historical tokens. Its time position is zero; historical positions are negative recorded ages in days. Local thermal coverage masks temporal attention and decoder skip aggregation, while the current token stays visible. A signed regression head adds a residual to the current coarse interpolant, followed by the common projection. See [historylst/model.py](../resources/historylst246/historylst/model.py). The three scored students share the final fusion system's two fixed teachers, as recorded in [training.md](training.md).

**MoCoLSK.** The thermal branch receives one normalized 40 × 40 coarse-temperature channel. Missing coarse cells use the 4 × 4 average of the supplied interpolant. The guidance branch receives 155 channels: the 74 current fields plus all 81 historical fields, with thermal/emissivity masking applied before flattening. This keeps every historical slot as separate channels. The published four-stage, width-32 large-kernel network uses scale factor four and four residual blocks per stage. Its prediction is converted back to kelvin and projected to the same coarse means. See [mocolsk_model.py](../research/jstars_20260914/experiments/mocolsk_model.py). The original-loss and scene-RMSE-calibrated versions and dynamic-weight numerical checks are recorded in [implementation.md](implementation.md#mocolsk-adaptation-and-numerical-checks).

**THSTNet.** Each reconstruction uses one historical fine/coarse pair and the current coarse field. The same 155-channel tensor used by MoCoLSK is projected into the patch embeddings of both THSTNet stages, so even the single-reference version has access to all historical auxiliary fields. Reference pairs are prepared by [prepare_thst_inputs.py](../research/near_neighbor_attribution_20260914/prepare_thst_inputs.py). Training samples an available reference at random. For single-reference inference, [matched_cache](../research/near_neighbor_attribution_20260914/thst_reference_variants.py) maximizes Pearson correlation between historical and current coarse temperatures over their observed overlap. A coarse cell counts as overlapping when it contains at least one fine cell supported in both observations; at least 64 overlapping coarse cells and a centered-vector norm product >1e-12 are required for a valid correlation. Undefined correlations rank below valid correlations, ties favor the smaller recorded age, and any remaining tie retains slot order.

For multiple-reference inference, [predict](../research/near_neighbor_attribution_20260914/train_thst.py) reconstructs from every available reference and weights the outputs at each pixel by that reference's thermal coverage divided by summed coverage. Where summed local coverage is zero, it averages the available reconstructions equally. A scene with no available reference uses the prepared slot-0 fallback. Both versions use four overlapping 128 × 128 windows to cover the 160 × 160 scene, average window overlaps and apply the common coarse projection. [thst_adapter.py](../research/near_neighbor_attribution_20260914/thst_adapter.py) implements the auxiliary embedding. The multiple-reference, task-calibrated version is the reported accuracy comparator; both reference strategies appear in the inference-cost comparison.

## Existing evidence and article interpretation

The 11.6% RMSE improvement compares the final teacher-trained compact system with historical U-TAE. The weighting control compares paired reference-only models and yields 0.0042 K overall, with 0.0152 K and 0.0096 K gains in the two reported thermal-mismatch subsets. Compact recovery/refinement reduces the full parent's parameter count by 35.8% while preserving its accuracy. The direct-bypass removal improves the reference-only control; encoder thermal moments remain active and the final 0.426 K system retains its original bypass. Existing scores, city partitions and model files are unchanged in this documentation release.
