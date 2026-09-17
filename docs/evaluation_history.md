# Evaluation history

Network parameters are fitted on the Fit partition. Checkpoints are selected by Validation RMSE, and model candidates are compared using Validation performance and parameter budgets. Fixed models are evaluated on the 30-city, 90-scene Test partition, whose cities are excluded from training. Training normalization and both teacher caches use Fit data.

Earlier Test results were available before later development stages, and Test scores were also used to check the project accuracy targets. The retained `research/sub04_20260911/final_delivery_20260912/network_result.json` reports comparisons dated 2026-09-12. Subsequent records cover the four-head, compact-recovery and query-refinement stages. This chronology is distinct from the Fit and Validation criteria used for parameter fitting, checkpoint selection and candidate comparison; original development records are retained unchanged.

Final weights are frozen before test prediction. Inference reads test inputs without labels or gradient updates, and scoring reads labels after predictions are complete. Bootstrap intervals describe paired variation across the sampled cities, with resampling stratified by region. Repeating the released evaluation checks the recorded result on the same evaluation cohort.

The three final students share fixed teacher fields. Their score variation describes student training conditional on those teachers. Historical-input removal holds the trained networks fixed; natural local sparsity and removal of whole groups answer different questions and have separate records.
