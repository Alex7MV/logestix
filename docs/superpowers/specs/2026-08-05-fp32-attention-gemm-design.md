# FP32 Activations for Attention GEMM Design

**Date**: 2026-08-05  
**Status**: Approved

## Problem

Step-35 NaN from FP16 activation overflow in attention GEMM.
FP16 max = 65504; HC state grows exponentially, making attention
inputs exceed this range.

## Fix

Convert FP16 activations → FP32 before each attention GEMM.
Use FP32×FP32→FP16 cuBLAS GEMM (FP32 accumulator, no overflow).

### Changes in f8_gemm_1d

```
Current: dequant F8→FP32, clamp→FP16, FP16×FP16→FP16
New:     dequant F8→FP32, convert d_x FP16→FP32, FP32×FP32→FP16
```

### Buffer layout

d_scratch: [FP32 weights (n elements)] [FP32 activation (k elements)]
Output: d_out (__half*, FP16 — safe, GEMM accumulates in FP32)

### O-projection

Same: dequant→FP32, convert d_o FP16→FP32, FP32×FP32→FP16 GEMM.

### No changes needed

- mla_attn_kernel, csa_sparse_attn_kernel: read FP16 KV/Q (unchanged)
- rope_apply_kernel, per_head_rms_kernel: operate on FP16 (unchanged)
- d_attn_out_h → d_attn_out: half_to_float conversion (unchanged)
- Routed expert FFN: already pure FP32 (unchanged)
- hc_post: already FP32 (unchanged)
