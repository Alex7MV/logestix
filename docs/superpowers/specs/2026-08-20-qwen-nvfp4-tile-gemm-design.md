# Qwen NVFP4 Dense GEMM Tile — Design Spec (Direction A step 1)

**Date:** 2026-08-20
**Status:** Approved
**Scope:** `libs/lsxcommon` — `qwen_nvfp4_gemm_kernel` + `LaunchNvfp4Gemm` in `libs/lsxcommon/src/qwen_inference.cu`. Direction **A** of `docs/superpowers/specs/2026-08-20-blackwell-inference-optimization-strategy-design.md`.
**Numeric contract:** FP32 accumulate; the `k` summation ORDER is free (tile regroup), as long as the result stays within the existing tolerance envelope of `test_qwen_model` and the new parity test vs the legacy per-element kernel. NVFP4 decode stays bit-identical via the shared `nvfp4_codec.h`.

## 1. Problem

`qwen_nvfp4_gemm_kernel` (`qwen_inference.cu:345-368`) and `LaunchNvfp4Gemm` (`:397-409`) implement the dense NVFP4 GEMM as a **naive one-thread-per-output-element** loop:

```cpp
for (int k = 0; k < K; ++k) {
    const int64_t e = n*K + k;
    const uint8_t byte = packed[e>>1];
    const uint8_t nib = (e&1)?(byte>>4):(byte&0xF);
    const float scl = E4M3ScaleToFloat(scales[e>>4]) * global_scale;
    acc += __half2float(xrow[k]) * kE2M1Table[nib] * scl;
}
```

Problems on Blackwell:
- **No reuse of `x`**: each output element re-reads the whole K row of `x` from L2/HBM (redundant HBM traffic `~N × K` per output, i.e. `N` copies of `x`).
- **`scales[e>>4]` re-decoded per element**: each scale is read and `E4M3ScaleToFloat`-converted 16 times (once per element in the block of 16) with no caching.
- **No tile/TMA**: no shared-memory staging, no vectorized loads, one output per thread → low compute/memory ratio, poor occupancy for the real dense sizes. `LaunchNvfp4Gemm` is called with heterogeneous shapes across the pipeline: Q/K/V/O with `N = num_heads*head_dim` / `num_kv_heads*head_dim`, gate/up with `N = intermediate_size`, down with `K = intermediate_size`, and head logits with `N = vocab_size`. For Qwen3.5-9B, `hidden_size=5120`, `num_attention_heads=24`, `intermediate_size` typically ~14k. A single tile kernel must cover small and large `K`/`N`.
- No `__half2`/`float2` vectorization.

## 2. Goal / non-goal

**Goal:** Replace the per-element kernel with a **tile-based GEMM** that:
- stages the `x` tile and the decoded weight tile in shared memory (reused across the K-loop),
- decodes each `scales` block once per 16 elements (cached in the dequant snapshot),
- accumulates in FP32 (free `k`-order),
- keeps NVFP4 compressed in HBM (no materialized fp16 copy),
- stays within the numeric contract (tolerance envelope of `test_qwen_model`, parity vs legacy kernel).

**Non-goal:** no change to the NVFP4 codec, no cuBLAS route (that would dequantize to HBM), no change to other kernels, no numeric relaxation beyond the FP32/k-free contract.

## 3. Kernel design

Grid/block layout (output tile `[N_tile × M_tile]` over `y`):

```
block (bx, by):
  n0 = bx * BN                     # output-column tile origin
  m0 = by * BM                     # token (row) tile origin
  half acc[BN][BM] in registers (fp32) -- or [BM][BN], whichever bank-friendly
  shared half sX[BM][BK]           # X tile: x rows [m0, m0+BM), k in [kb*BK, (kb+1)*BK)
  shared half sW[BN][BK]          # NVFP4 dequant weight tile for rows [n0, n0+BN)

for (kb = 0; kb < K; kb += BK):
  // load + decode the weight tile into shared, decoding each scale once
  dequant_weight_tile(sW, packed, scales, global_scale, K, N, n0, kb);
  // load X tile
  copy_X_tile(sX, x, m0, kb);
  __syncthreads();
  // fp32 fma over BK
  for (i in BM) for (n in BN) for (k in BK)
      acc[n][m] += sX[m*BK+k] * sW[n*BK+k];
  __syncthreads();
// store y[m][n] = __float2half(acc[n][m])
```

Key properties:
- `BM×BN` tile in registers; **one `x` load + one weight decode per `K`-block** instead of `N` per element; scales decoded once per 16.
- Bank friendliness: iterate `k` innermost (contiguous `sW[n*BK+k]`); choose `BN,BM` so that `BN*BK`... (fill after measuring real geometry; default `BK=32, BN=16, BM=16` tentative — final in plan).
- **Small-K fallback**: when `K < BK` (synthetic test `K=64`), do a single tail pass — no regress; the parity test covers it.
- Occupancy: ≤48 warps/SM (per strategy), block dims ensure. Use cooperative-groups block-sync where helpers are reused; no shared per-block beyond the two tiles + trivial.

## 3. Numeric contract

- FP32 accum in registers; `k` order free (any tile regroup). Result must stay within the tolerance of:
  - the **new parity test** vs the legacy per-element kernel (same dequant → bit-identical decode, so diff comes only from FP32 add order — bounded by few-ULP relative, e.g. `1e-3` relative like the existing dense test), and
  - `test_qwen_model` (synthetic full-pipeline, currently only checks `ok`; we keep passing).
- The NVFP4 decode (`E4M3ScaleToFloat`, `E2M1Table`) is UNCHANGED (from `nvfp4_codec.h`), so bit-exact decode is preserved; only the accumulation grouping changes.

## 4. Testing / acceptance

- **New** `tests/test_qwen_nvfp4_tile_gemm.cu`: synthetic random `x`, generated packed/scales/global_scale; run the legacy per-element kernel and the new tile kernel on identical inputs; `CHECK(abs(a-b) < tol)` for each output with `tol=1e-3*|ref|+1e-6`. Covers small-K (`K=64`) and a realistic large-K (`K=5120, N=5120`, subset rows). Runs in CI (no checkpoint needed).
- **Existing**: `test_qwen_nvfp4_dense.cu` (its own kernels unaffected), `test_qwen_model` (synthetic pipeline, must stay green), `test_qwen_selfcheck` (SKIP w/o checkpoint; if present → bit-exact `"Paris"`).
- **Occupancy/throughput guard**: block dims chosen so warps/SM ≤48; document expected overlap math.

## 5. Files

- Modify: `libs/lsxcommon/src/qwen_inference.cu` — new tile kernel + `LaunchNvfp4Gemm` grid/tile + small-K fallback. Keep legacy `qwen_nvfp4_gemm_kernel` intact **temporarily** for the parity test (it becomes the reference; the production launch switches to the tile variant). `nvfp4_codec.h` already included.
- Create: `tests/test_qwen_nvfp4_tile_gemm.cu`.
- Modify: `tests/CMakeLists.txt` — add the new test (arrow POST_BUILD + `add_test`).
- No CMake source list change in `libs/`.

## 6. Open decisions (resolve in plan / during impl)

- Exact `BK/BN/BM` (start 32/16/16; tune against K=5120 for occupancy ≤48; may split `LaunchNvfp4Gemm` small-K vs large-K dispatch).
- Whether to keep the legacy kernel in the object for the parity test — plan: keep (static), the tile launch is the production path; parity test references it.
- Optional `__half2` vectorization if bank-conflict-free.

## Self-review

Coverage: contract (§3), numerical terms (§3), test list (§4), files (§5), decisions (§6). No placeholders (open decisions are flagged as plan-scoped, not TBD-in-spec). Inconsistency check none found. Note conflict: the AGENTS.md policy is "always prefer tile programming" — this aligns. Scope: single kernel + one test — single plan.