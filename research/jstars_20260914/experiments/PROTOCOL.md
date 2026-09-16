# Limited supplements for the JSTARS manuscript

Fixed before new test predictions, 2026-09-14. Reuse Fit603, Val45, Test90 and existing final checkpoints. No new data acquisition or model search.

## E1: recent guided-downscaling comparator

Run the published x4 MoCoLSK architecture (four stages, 32 base channels, four residual blocks per group). Expand its guidance stem to the same 155 explicit fields used by the THSTNet comparator. Preserve the official dynamic-fusion forward operation; use the existing functional-convolution repair to restore gradients severed by the author's construction of detached dynamic parameters. Verify single-sample forward equivalence, nonzero dynamic-MLP gradients, batch independence, and common-input identity before training.

Train from a fixed fresh seed 20260916 for 10,000 updates with batch four, AdamW (learning rate 1e-4, weight decay 1e-5), masked L1, parent-aligned 128-pixel crops, D4 augmentation, region/city-balanced sampling and EMA. Preserve the paper's architecture, loss family and optimization scale; use the project's fixed crop and sampling protocol. Validate raw and EMA at initialization and every 1000 updates using full 160-pixel scenes and FP32. Then initialize from the selected checkpoint and perform 1500 updates of the same RMSE/AdamW/EMA task calibration used for THSTNet, validating every 500 updates. Select the final native/calibrated version by Val45 only. Report all versions, convergence and total training cost. Test90 is an existing research test set; it does not drive this experiment's configuration or selection.

## E2: history availability with frozen final networks

Use all three final network checkpoints. Evaluate all nine histories (existing reference), archive-only (slots 0–5), and recent-only (slots 6–8), zeroing every channel of removed slots. Keep current fields, support, weights and coarse-mean projection fixed. Generate complete input-only Test90 predictions before scoring; use paired city comparisons and the established regional macro metrics. This measures operation with reduced archival inputs without retraining. Analyze the supplied slot dates before execution to confirm semantics.

These two experiments directly test contemporary competitive performance and a practical input requirement. Additional physical mechanisms, cross-climate claims and external sensor expansion are outside this finite experiment set.
