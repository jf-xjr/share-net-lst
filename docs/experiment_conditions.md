# Experimental conditions

This record contains the operational settings needed to reproduce the recorded experiments. Data channels, selection rules and comparator adapters are described in [implementation.md](implementation.md); stage configurations and teachers are described in [training.md](training.md).

## Common inputs and outputs

Current inputs contain 52 spatial channels, four emissivity channels, a 15-element context vector, observation support, and two coarse-grid channels. The historical tensor has nine dates and nine channels per date. `resources/historylst246/PREPROCESSING.md`, the manifest and normalization JSON files define every channel, scaling constant, acquisition and fill rule. The source package includes the actual loader and all comparator input adapters.

The reference and coarse observation are generated from the same Landsat Collection 2 product. Thermal aggregation requires 12 accepted children out of 16 at both the 30-to-120 m and 120-to-480 m stages. Urban scoring uses `formal`; the observation layout uses `support`. Only the latter is a model input. Predictions receive the same float64 support-weighted coarse-mean projection. The scoring routines check the observation constraint to 1e-8 K.

## Timing

Measurements use an NVIDIA RTX 5060 Ti, complete 160 by 160 scenes, and batch size one. The network uses FP32 with TF32 disabled; output projection uses FP64. The interval starts before transfer of prepared CPU arrays and ends after the projected output returns to the CPU. Product download and preprocessing are outside this interval.

Each of three validation scenes spanning historical availability receives three warm-up calls and ten timed repetitions. CUDA is synchronized before and after each timed interval. The reported latency is the median of the three scene medians. Memory is the maximum peak allocated CUDA memory across those cases. `utae_inference_benchmark.json`, the MoCoLSK benchmark and the original/task-calibrated benchmark records retain individual timings and source hashes.

## Paired component experiments

The learned-weight controls, coverage-only variants and thermal-bypass variants use two paired seeds, 20260914 and 20260915. Each is trained for 12,000 updates from random initialization with reference-only supervision, batch four, aligned 128 by 128 crops, AdamW, learning rate 1e-3, warm-up 300 and weight decay 1e-4. History and emissivity dropout are zero. Validation compares the same 26 raw/EMA candidates in FP32. Initial tensors, scene sequence, spatial transforms and crop selection are shared within each pair.

Coverage-only variants replace learned historical scores with normalized coverage. Bypass variants retain learned source scores and the historical feature and thermal-moment injections at every encoder scale. They zero only the separately returned full-resolution summary before it enters `detail_skip` and the anomaly readout. The implementation check verifies identical injected encoder features, active gradients for all retained history-fusion levels, and zero bypass gradients. The inactive 1,348 parameters remain in the state dictionary to preserve identical initialization of every retained tensor.

## Historical availability and conditional scores

Local valid-date counts use positive thermal coverage, grouped as 0, 1–2, 3–5 and 6–9. Mean coverage averages all nine slots, including empty slots; the low-coverage subset uses a threshold of 0.25. A scene contributes to a subset only when it contains at least 32 formal urban pixels in that subset. These subset scores retain equal regional weighting. Pixels with no local historical coverage can still receive spatial context from neighboring pixels.

Historical-range subsets require two visible dates and use a 1 K margin around historical parent temperatures. Nearest-parent-temperature gaps use thresholds of 2 and 5 K. Hotspots are the warmest 10% of scored urban pixels in each scene. Bootstrap intervals resample cities within each region 10,000 times, preserving the paired comparisons and regional weights.

## Spatial examples

The three scenes are those closest to each region's median reference-temperature standard deviation: Ganzhou, Zurich and Indio. Both model predictions use the 20260905 lineage. Within each scene, a 40 by 40 window is selected on an eight-pixel stride by maximum formal urban coverage, then minimum distance to the center, then coordinates. The selected top/left coordinates are (48, 24), (48, 48) and (56, 32). Selection uses neither prediction errors nor model differences. Temperature scales use each scene's reference 1st–99th percentiles; error maps use −2 to 2 K.

## Thermal-bypass result

Both paired jobs completed all 12,000 updates and 26 validation candidates. Validation selected seed 20260914 at its recorded minimum and seed 20260915 at update 8,000 EMA. The 180 new test predictions were saved before scoring. Mean RMSE is 0.455039060 K without the bypass versus 0.457766028 K with it. Per-run values are 0.452960975 versus 0.456153593 K and 0.457117146 versus 0.459378464 K. Removing the bypass improves 20 of 30 cities; the paired improvement is 0.002726968 K, with a 95% interval of [0.000105740, 0.005501056] K. Hotspot IoU rises from 0.792240083 to 0.794252442.

The result concerns paired, randomly initialized, reference-only training. The delivered 0.426 K models retain their original thermal bypass and teacher-guided training. `thermal_bypass/analysis/` contains selections, predictions' identities, per-scene/city scores and paired statistics.
