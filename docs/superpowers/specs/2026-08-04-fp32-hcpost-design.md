# FP32 through hc_post Design

**Date**: 2026-08-04  
**Status**: Approved

## Problem

HC state grows exponentially (post≈2 × FFN≈6.4/layer). After ~27 steps,
the FP32 accumulation buffer exceeds 65504. Converting to FP16 for
`hc_post_kernel` produces Inf → NaN. The reference model uses BF16
(range ±3.4e38) and never overflows.

## Fix

Keep the entire hc_post boundary in FP32. No FP16 conversion.

### Changes

1. **hc_post_kernel** (`inference_attn.cu`): Change `const __half* x` → `const float* x`.
   Remove `__half2float(x[i])` → just `x[i]`.

2. **Pipeline** (`inference_pipeline.cu`):
   - Remove `clamp_f32_to_fp16` + `float_to_half_kernel` before FFN hc_post.
     Pass `d_ffn_accum` (FP32) directly.
   - Convert `d_attn_out` (FP16) → FP32 before attention hc_post.
     Add `half_to_float_kernel` call.
   - Change `d_attn_out` buffer from `__half*` to `float*` (2× size).

3. **hc_post declarations** (`inference_load.cuh`): Update function signature.

### What stays the same
- HC state (d_hidden, d_hidden_next) — already FP32
- hc_pre_kernel, hc_head_kernel — already FP32
- All GEMM inputs — FP16 (clamped from FP32 dequant)
- Only hc_post input changes: FP16 → FP32
