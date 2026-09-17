# Implementation and experiment record

Start with [configuration.md](configuration.md) for the executed training values, complete 74-channel dictionary, slot selection, quality thresholds and baseline input adaptations. This repository documents the inputs, model lineage, comparator adaptations, and evaluation settings of the recorded experiments. Executable sources and JSON records retain the exact configurations.

## Data package and split

The fixed data package contains 603 training, 45 validation, and 90 test scenes. Each scene has current spatial predictors, a coarse temperature map, observation support, a context vector, emissivity predictors, and nine historical observations. Reference temperature and the urban scoring mask are stored separately. Scene identities, acquisition times, affine transforms, coordinate reference systems, and array checksums are retained in the manifest and per-scene records. The supplied arrays define the experiment exactly.

## Reference temperature and quality screening

Temperature conversion follows the Landsat Collection 2 convention, $T=0.00341802\,\mathrm{DN}+149$ K [reference: usgs2026]. Query quality screening rejects QA_PIXEL bits 0--5, with one delivered-pixel dilation, radiometric saturation, invalid temperature or reflectance, and the registered aerosol states. Accepted temperatures lie within 240--380 K and have reported temperature uncertainty no greater than 3 K. At least 12 accepted delivered pixels are required for each 120 m target, and at least 12 supported targets are required for each 480 m observation. Coarse values average supported 120 m children. Urban evaluation uses the distributed `formal` mask, which applies the original urban-eligibility rules to the accepted support.

## Current optical predictors

Current reflectance is constructed from exact-query optical sidecars with their own optical quality screening. Six surface-reflectance bands use $0.0000275\,\mathrm{DN}-0.2$. Each band has a training-normalized value and a scene median/interquartile-range anomaly clipped to $[-8,8]$. The four spectral indices are NDVI, NDBI, MNDWI, and BSI. Optical texture statistics summarize the delivered pixels within the study-grid cell. Built and water fractions use the 2021 nine-class annual land-cover layer [reference: lulc2026]. The input manifest enumerates the complete packet.

## Emissivity and weather

Current emissivity and emissivity uncertainty use their own joint digital-number mask: $6000\le E_{\mathrm{DN}}\le10000$ and $0<\mathrm{EMSD}_{\mathrm{DN}}\le10000$, both scaled by $10^{-4}$. Their means and dispersion are computed within each delivered $4\times4$ block. Missing continuous values use the nearest valid donor, with zero retained in the availability channel. NASA POWER supplies UTC daily meteorology [reference: power2026]. Maximum temperature appears twice because the retained input representation contains a legacy feature and a separately normalized weather extension.

## Historical observation selection

The first three historical slots select one May--September Landsat 8 record from each of 2018, 2019, and 2020, ordered by catalog cloud cover, acquisition time, and identity, with cloud cover at most 60%. The next three slots use the same years and season, a 40% cloud limit, and circular calendar distance to the target date as the first ranking criterion. Duplicate records become empty slots. The final three slots select distinct Landsat 8/9 acquisitions 8--64 days before the target, with cloud cover at most 40%, full crop coverage, and all required assets; they are ranked by age, cloud cover, time, and identity. Target identities and equivalent aliases are excluded.

## Historical preprocessing and missing values

Historical rasters are reprojected to the fixed delivered grid by nearest neighbor. Historical thermal support uses valid digital numbers, 240--380 K temperature, positive ST_QA no greater than 3 K, no radiometric saturation, and rejection of QA_PIXEL bits 0--5 with one-pixel dilation. A historical study cell can retain a single accepted delivered pixel, with its fractional count supplied explicitly. Filled temperatures provide a complete numerical field; actual coverage masks temperature, anomaly, and quality before network encoding. Emissivity uses its separate availability. A wholly thermal-empty date has nine zero channels.

## Model lineage and compact architecture

The final three models inherit independently initialized historical networks with seeds 20260905, 20260912, and 20260913. The training configurations give the retained training sequence. The initial historical architecture has 9,310,105 parameters and one source-weight head. Four-head conversion adds 396 parameters. Compact recovery removes alternating bottleneck blocks, retaining indices 0, 2, and 4, and retains indices 0, 1, and 3 in the third encoder stage. The compact parent has 5,911,525 parameters. The final query projections add 65,520 parameters.

## Optimization and loss

All stages use AdamW with weight decay $10^{-4}$, balanced scene sampling, the eight rotations/reflections of the square, and EMA with maximum decay 0.995. Learning rates follow warm-up and cosine decay. In the final stage, the cosine floor is 0.1 and two microbatches of size two form one update of size four. This stage uses full single-precision forward, backward, and optimizer operations with TF32 disabled. Reference loss is computed on the urban scoring mask, and teacher loss uses the supplied observation support. Both losses are scene-normalized RMSE with $10^{-6}$ inside the square root.

## Initial teacher construction

The first teacher is a fixed eight-view prediction of the seed-20260905 initial historical model after parameter interpolation. Its parameters are $0.75\theta_{\mathrm{selected}}+0.25\theta_{\mathrm{last\ EMA}}$, with buffers retained from the selected checkpoint. The selected and last checkpoints are at updates 9000 and 18000. The interpolation belongs to the inherited model-development record; the three candidate interpolation coefficients and endpoint checksums remain in that record. All three historical students and all three U-TAE students receive this same fixed teacher. Each student starts from its own selected initial checkpoint.

## Shared refinement teacher

For four-head refinement and subsequent compact stages, the fixed teacher averages the three historical models after eight-view refinement, each evaluated under eight spatial transformations. These 24 predictions are inverse-transformed, averaged in double precision, and projected to the supplied coarse means. Teacher generation uses only the 603 training inputs, and its predictions are fixed before student refinement. The three final model lineages therefore share teacher targets while retaining their own initialization lineage. One model and one view produce each delivered test prediction.

## Selected checkpoints and stage costs

The final selected checkpoints occur at update 1000 with raw weights for seeds 20260905 and 20260912, and update 1500 with raw weights for seed 20260913. Their validation RMSE values are 0.410356, 0.412125, and 0.411257 K, respectively. Test RMSE values are 0.425812, 0.427238, and 0.425345 K. Final refinement takes approximately 584, 608, and 565 seconds on the RTX 5060 Ti. The preceding compact-recovery runs take approximately 1427, 1466, and 1380 seconds. These times exclude initial-model training and teacher generation; the corresponding stage receipts retain their separate costs.

## Common comparator inputs

The common-input neural comparison contains 155 channels: 52 current spatial fields, four emissivity fields, observation support, coarse temperature and validity, 15 context values, and 81 historical fields. Missing thermal and emissivity values are masked by their own coverage. THSTNet and MoCoLSK receive bitwise-identical auxiliary tensors. The current fine-grid target temperature and urban scoring mask are excluded from these tensors. Each method's output undergoes the same support-weighted coarse projection.

## Local linear forests

The local linear forests use `grf` version 2.0.2 with 2000 trees, sampling fraction 0.5, minimum node size five, and honest splitting. The candidate-variable count is $\min(\lceil\sqrt p+20\rceil,p)$ for $p$ predictors. Historical kernel selection uses five spatial folds made from $8\times8$ coarse-cell tiles; positive final LASSO coefficients determine the thermal correction variables. The forest chooses its linear-correction penalty through the package's fixed out-of-bag path. All predictor aggregation uses the same supported children as the current coarse observation. The complete-input variant retains every available historical field and uses kernel selection only to define the local linear correction variables.

## Historical U-TAE

U-TAE uses encoder and decoder widths of 32, 64, 128, and 256, eight temporal-attention heads, and a 256-dimensional temporal representation. The current observation becomes a predictor-only token beside the nine historical tokens. Historical visibility masks act in temporal attention and decoder skip aggregation. The final regression head estimates a signed residual added to the coarse interpolant. Its last 3000-update teacher refinement shares the same 24-prediction training teacher used by the historical model.

## THSTNet

THSTNet uses randomly chosen available references during training. The original-loss stages use Adam at $10^{-4}$, no weight decay, and a validation-driven learning-rate reduction. Calibration starts from the selected second-stage checkpoint and uses AdamW, scene RMSE, and EMA for 1500 updates. Multiple-reference inference coverage-weights the per-reference reconstructions; the single-reference variant chooses a reference matched to the current coarse thermal field. The task-calibrated multiple-reference model is selected on validation. Both reference strategies appear in the inference-cost comparison.

## MoCoLSK adaptation and numerical checks

MoCoLSK retains the published dynamic large-kernel blocks and coarse/fine projection structure. Missing coarse cells are filled from the coarse average of the supplied interpolant. A functional implementation of the generated convolution weights restores gradients that were detached by the imported parameter wrapper, and weight generation is independent for each batch sample. The modified operation matches the original single-sample computation exactly in CPU double precision. Complete-network single-precision differences are at most $3.06\times10^{-5}$ K, and all 48 dynamic-MLP parameter tensors receive nonzero gradients in the implementation check. Bitwise auxiliary equality and batch independence are also verified before training.

## MoCoLSK training and selection

The MoCoLSK experiment uses seed 20260916, batch size four, aligned $128\times128$ crops, square rotations/reflections, AdamW at $10^{-4}$, and EMA decay 0.995. The native phase uses scene-normalized masked L1, weight decay $10^{-5}$, and 22 full-scene raw/EMA validation candidates over 10,000 updates. Calibration uses scene-normalized RMSE, weight decay $10^{-4}$, and eight candidates over 1500 updates. The phase and checkpoint are frozen from validation scores before generating either test prediction packet. All test predictions are sealed before test scoring.

## Paired source-weight experiment

The learned-versus-coverage experiment isolates source scoring within the final 5,977,045-parameter architecture. Fixed weights normalize visible thermal coverage; historical feature extraction, feature injection, thermal moments, and the decoder remain active. Source-scoring and query-projection parameters, totaling 97,408, are frozen in the coverage group. Each of the two seed pairs shares initial tensors, sampled scenes, rotations/reflections, aligned crops, and a 12,000-update schedule. Both groups use only reference supervision, AdamW at $10^{-3}$, 300 warm-up updates, and 26 validation candidates.

## Historical-input removal

For history availability, the delivered final networks are frozen. Archival-only evaluation zeroes all nine channels of slots 6--8; recent-only evaluation zeroes slots 0--5. This produces 540 additional scene predictions across three seeds and two removal conditions. Every historical timestamp is checked to precede the target; the retained recent observations have actual ages of approximately 7.9--64.1 days after timestamp conversion. Complete-input predictions reproduce the original predictions exactly on the audited scene for each seed. All 540 predictions are sealed before labels are opened, and each passes the common coarse-mean check.

## Availability scores and paired intervals

The availability score JSON reports full availability scores. Confidence intervals resample cities with replacement within each region, using 10,000 draws and a fixed random seed of 20260914. For a comparison, each city's difference is formed before resampling, and the three regional means receive equal weight. The intervals represent paired variation across the sampled cities. Both removal conditions increase RMSE in all 30 cities. The experiment measures dependence of the trained system on its historical inputs; six archival and three recent slots differ in age, count, and coverage.

## Conditional evaluation

Conditional evaluation intersects each input-defined stratum with the unchanged urban scoring support. A scene contributes to a stratum when at least 32 scored pixels remain. History-count strata use the number of dates with positive thermal coverage at each location. Historical parent temperature is decoded as $20h_0+300-5h_1$, where $h_0$ and $h_1$ are the stored normalized temperature and anomaly. Current-versus-historical range strata require at least two visible historical dates and a valid current coarse observation. The below/above groups use a 1 K margin outside the historical range. Nearest-temperature-gap groups use thresholds of 2 and 5 K.

## Hotspots and metric aggregation

Hotspot sets contain exactly $\lceil n/10\rceil$ of the $n$ scored pixels in a scene. Stable descending sorting resolves temperature ties. IoU compares the predicted and reference sets, while hotspot MAE uses the reference set. Scene metrics are averaged within city and then equally across regions. The conditional-score JSON supplies the scene and city counts behind conditional comparisons, since different strata contain different geographic compositions.

## Evaluation cohort and reference provenance

Parameter fitting uses the training partition; checkpoints are selected by validation RMSE, and model candidates are compared using validation performance and parameter budgets. Fixed models are evaluated on the 30 test cities excluded from training. See [evaluation_history.md](evaluation_history.md) for the evaluation sequence and retained development records.

## Complete training lineage

[Training stages and teachers](training.md) explain the executed sequence. [training_lineage.json](training_lineage.json) binds every selected parent, exact configuration, checkpoint and teacher field across the three final lineages.
