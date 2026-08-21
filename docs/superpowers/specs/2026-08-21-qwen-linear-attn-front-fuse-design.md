# Qwen Linear-Attention (GatedDeltaNet) Front-End Fuse — Design

## Goal

Reduce the linear-attention layer's **11 launches to 5** by fusing the
input-side split launches into **2 fused kernels**, matching the split path
within the established numeric tolerance (`FP32` accumulate, `1e-2` vs split).

## Current split path (`RunQwenLinearAttention`, `qwen_inference.cu:1163`)

11 kernels per layer call:

| # | Kernel | Launch shape |
|---|---|---|
| 1 | `LaunchNvfp4Gemm(lw.qkv)` | qkv GEMM, `[seq,H] -> [seq, S]`, S=KD\*2+VD |
| 2 | `LaunchNvfp4Gemm(lw.z)` | z GEMM, `[seq,H] -> [seq, VD]` |
| 3 | `qwen_conv1d_kernel` | causal depthwise conv + silu, `[seq,S]` |
| 4 | `LaunchNvfp4Gemm(lw.a)` | a GEMM `[seq,H] -> [seq,nv]` |
| 5 | `LaunchNvfp4Gemm(lw.b)` | b GEMM `[seq,H] -> [seq,nv]` |
| 6 | `qwen_prep_gating_kernel` | a,b -> g/beta |
| 7 | `qwen_l2norm_rows_kernel` (q) | l2norm q slice, q_scale=1/sqrt(kd) |
| 8 | `qwen_l2norm_rows_kernel` (k) | l2norm k slice |
| 9 | `qwen_gated_delta_kernel` | recurrence (`seq \* nv` blocks) |
| 10 | `rmsnorm_gated_kernel` | `d_core,d_z -> d_gated` |
| 11 | `LaunchNvfp4Gemm(lw.out)` | out_proj `[seq,VD] -> [seq,H]` |

Kernels 9-11 (recurrence, RMSNormGated, out_proj) are **kept as-is** — the
recurrence is a different compute class; RMSNormGated and out_proj already run
separately (YAGNI to fuse them here). Task 1 only fuses the **front-end**:
kernels 1,3,7,8 -> **F1** (qkv GEMM + conv1d + l2norm q/k), and kernels
2,4,5,6 -> **F2** (z + a + b + prep_gating).

## Channel order (unchanged)

`qkv` weight `rows=H, cols=S`. Output column `c in [0,S)`:
- `[0, KD)`             -> q (head `kh`: cols `[kh*kd, (kh+1)*kd)`)
- `[KD, 2*KD)`          -> k (head `kh`: cols `[KD+kh*kd, ...)`)
- `[2*KD, 2*KD+VD)`     -> v (v-head `vh`: cols `[2*KD+vh*vd, ...)`)

`qsplit = d_conv + 0`, `ksplit = d_conv + KD`, `vsplit = d_conv + 2*KD`
(row stride `S`) — matches split sites at `qwen_inference.cu:1233-1235`.

## F1: fused qkv + conv1d + l2norm (grid `(seq,1)`, blockDim `pow2(kd)`)

One block per token row `t`. Threads = `pow2(kd)` (power-of-2 >= kd, mirrors
`qwen_l2norm_rows_kernel`; `kd <= 256` guarantee, as split `qwen_inference.cu:1221`).

Block phases (barrier-guarded, no RMW race — stage-6 invariant):

1. **Raw qkv row** (grid-stride over cols): thread computes
   `val[c] = NVFP4_dot(x_norm_row, qkv, col c)` over `H`, writes the raw
   GEMM output to the persistent history `d_qkv[(kv_base+t)*S + c]`
   (kv_base=0 for transient). `__syncthreads()`.
2. **Conv1d + silu** (grid-stride over cols): thread reads history rows
   `d_qkv[(kv_base+t-k)*S + c]` for `k in [0,kernel)`, `acc +=
   w[c*kernel + (kernel-1-k)] * raw`, writes `d_conv[t*S+c] = silu(acc)`
   (same math as `qwen_conv1d_kernel`, `qwen_inference.cu:203`). `__syncthreads()`.
3. **l2norm(q) and l2norm(k)** (sequential per head `h in [0,nk)`):
   block-wide power-of-2 reduction over `kd` on the q slice
   `d_conv[t*S + h*kd]` with eps `1e-6` and `q_scale=1/sqrt(kd)`, then the k
   slice `d_conv[t*S + KD + h*kd]` with scale 1. Replicates
   `qwen_l2norm_rows_kernel` (`qwen_inference.cu:589`) exactly, including
   the `rsqrtf(sum+1e-6)*scale` reduce order.

Parallelism is modest per token (correct — dims are small; the win is launch
count), and occupancy is plenty (blockDim up to 256).

## F2: fused z + a + b + prep_gating (grid `(seq,1)`, blockDim `pow2(kd)`)

One block per token `t`:
1. grid-stride compute of `d_z[t*VD + c]` (z GEMM, H->VD),
   `a_buf[t*nv + h]` (a GEMM, H->nv), `b_buf[t*nv + h]` (b GEMM, H->nv).
   `__syncthreads()`.
2. prep_gating: thread for each `h in [0,nv)` computes
   `g[t*nv+h] = -expf(A_log[h]) * softplus(a + dt_bias[h])`,
   `beta[t*nv+h] = sigmoid(b)` (replicates `qwen_prep_gating_kernel`,
   `qwen_inference.cu:572`).

## Public entry & wiring (mirrors fused-decode-attn precedent)

`RunQwenLinearAttention` (split) is **kept intact** as the numeric reference
for tests. A new public entry replicates its call signature and runs the
fused front-end internally:

```cpp
cudaError_t RunQwenFusedLinearAttnKernel(
    cudaStream_t stream, const QwenLayerWeights &lw, const QwenTopology &top,
    int seq, const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta, __half *d_conv,
    __half *d_core, __half *d_gated, int kv_base = 0,
    float *d_gd_state = nullptr);
```

Body: F1 -> F2 -> (unchanged) `qwen_gated_delta_kernel` ->
`rmsnorm_gated_kernel` -> `LaunchNvfp4Gemm(lw.out)`. Same scratch layout and
`d_gd_state` seed/writeback semantics as the split path.

## Generator wiring

Both `is_linear` call sites in `RunQwenGeneration`
(`qwen_inference.cu:1583` and `:1674`) switch from `RunQwenLinearAttention`
to `RunQwenFusedLinearAttnKernel`. Argument order/buffers identical.

## Testing (TDD)

New `tests/test_qwen_fused_linear_attn.cu` (registered in `test_qwen_cu`
OBJECT + `foreach(qt ...)` in `tests/CMakeLists.txt`, next to the other Qwen
CUDA tests):

- **F1 unit parity**: build tiny layer (H=16, nk=2, kd=8, nv=2, vd=8, kernel=3),
  run split (qkv GEMM -> conv -> l2norm q/k writing the same buffers) vs
  fused `RunQwenFusedLinearAttnKernel` front, assert `d_conv` (q/k slices,
  post-norm) within `1e-2` for seq=1 and seq=2.
- **Full-layer parity**: run split `RunQwenLinearAttention` vs fused
  `RunQwenFusedLinearAttnKernel` end-to-end, compare `d_out` within `1e-2`
  for seq=1 (decoder) and seq=2 (prefill). `cudaDeviceSynchronize` before
  readback.

## Files

- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare
  `RunQwenFusedLinearAttnKernel`.
- Modify: `libs/lsxcommon/src/qwen_inference.cu` — `fused_linear_front_kernel`
  (F1), `fused_linear_zab_kernel` (F2), and the public entry
  `RunQwenFusedLinearAttnKernel` (external, `lsxcommon::qwen`, after the
  second anonymous-ns close ~`:837`); rewire the two generator sites.
- Create: `tests/test_qwen_fused_linear_attn.cu`.
- Modify: `tests/CMakeLists.txt` — register the test.

## Success criteria

- Front-end 8 launches -> 2 (layer 11 -> 5).
- `test_qwen_fused_linear_attn` green (F1 parity + full-layer parity, seq=1/2).
- `test_qwen_model` green (generator now runs the fused entry); Qwen ctest
  suite green; full build clean.
