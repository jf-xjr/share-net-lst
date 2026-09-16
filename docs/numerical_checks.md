# Numerical validation and timing

All scored predictions pass the common FP64 support-weighted coarse-mean check with maximum error below 1e-8 K. This checks the observation equation after projection.

The MoCoLSK dynamic-weight repair was checked before training: the modified operation matches the original B=1 operation in CPU float64; its batched output matches separate sample evaluation. Complete-network FP32 equivalence is within approximately 3.1e-5 K, and all 48 dynamic-MLP tensors receive gradients. The exact auxiliary packet matches THSTNet bitwise.

The first timing attempts applied a 1e-6 K repeat-output check. That bound was tighter than one FP32 temperature increment near 300 K (about 3.05e-5 K). Timing was repeated with the previously used 2e-4 K full-network FP32 bound and explicit recording of every observed difference. Maximum measured differences were 3.2697405e-5 K for MoCoLSK and 3.0517578e-5 K for U-TAE. Training and Test predictions were not repeated or changed. The failed initial check and timing-recovery source are retained in the workspace archive.

The completed timings use three warm-up passes and ten timed repetitions for each of the same three validation scenes, with five, eight, and nine available histories. The score is the median of scene medians. All GPU training was complete. CPU prepared inputs, GPU transfer, FP32 forward computation with TF32 disabled, FP64 output projection, and CPU output are included. The original THSTNet and compact-model timing records use this same hardware, precision, scene selection, and measurement boundary.
