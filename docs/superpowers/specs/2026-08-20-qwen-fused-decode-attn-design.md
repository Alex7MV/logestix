# Qwen Fused Decoder Attention Layer — Design Spec (Direction A step 3)

**Date:** 2026-08-20
**Status:** Approved
**Scope:** `libs/lsxcommon` — Qwen-3.5 decoder full-attention layer for the **incremental-decode (seq=1)** path. Direction **A** of the Blackwell optimization strategy.
**Numeric contract:** FP32 accumulate; the existing per-kernel FP order may be regrouped inside the fused kernel, as long as the output stays within a tolerance of the split (existing) path on identical inputs. NVFP4 decode unchanged (`nvfp4_codec.h`).

## 1. Problem (what it saves)

In `RunQwenGeneration` the decoder step (seq=1) runs, per transformer layer, a **sequence of ~10-12 separate kernel launches** with HBM read/write round-trips through scratch buffers:

`RunQwenFullAttention` (`qwen_inference.cu:740-821`) alone issues 8 launches:
1. `LaunchNvfp4Gemm` q_proj (`:764`)
2. `LaunchNvfp4Gemm` k_proj (`:766`)
3. `LaunchNvfp4Gemm` v_proj (`:767`)
4. `qwen_split_query_kernel` (`:772`)
5. `qwen_head_norm_kernel` (k) (`:782`)
6. `RunQwenRopeCosSin` + `RunQwenPartialRope` (`:789-792`, 2 launches)
7. `full_attn_softmax_kernel` (`:804`)
8. `qwen_out_gate_kernel` (`:813`)
9. `LaunchNvfp4Gemm` o_proj (`:819`)

So the attention chain alone is **9 launches**; the caller's surrounding `input-norm`, `qwen_add` residual, `post-norm`, and MLP (3 GEMMs + swiglu) add ~8 more — ≈17 launches/successive-token. At seq=1 each intermediate (`d_qc`, `d_q`, `d_k`, `d_v`, `d_attn`) is a write-then-read round-trip through a scratch buffer, and RoPE needs `d_pos`/`d_cos`/`d_sin` arrays.

## 2. Goal / non-goal

**Goal:** A fused decoder attention layer for the **seq=1** case that replaces the 9-launch split chain with **3 kernels on one stream** (`fused_qkv` → `fused_attn` → `fused_o_proj`), eliminating the 8 intermediate scratch round-trips while keeping each stage a simple, correct, individually-testable kernel. The caller's input-norm/residual-add and post-norm/MLP are NOT part of this kernel (they stay separate in this step).

> **Revision (operator decision):** a strict single-`__global__` launch would require device-wide/cooperative grid sync to combine the block-wide attention (needs all NH heads' q) with the cross-block o_proj accumulation — disproportionate risk. The accepted architecture is **3 fused stages on one `cudaStream_t`**, replacing 9 launches → 3. This still removes the ~8 intermediate scratch buffers and their HBM round-trips; it does not collapse to a literal single kernel.

**Non-goal:** no change to prefill (seq>1) — it keeps the split kernels (its block-parallelism is different). No change to the NVFP4 codec or the GEMM numerics. No change to the linear-attention (GatedDeltaNet) layers in this step (separate). No change to MLP fusion (separate follow-up). No change to the persistent KV cache semantics (the fused kernel appends to the same `kv_base` rows).

## 3. Kernel design

**Launch** (seq=1): grid = `NH` heads (one block per (token=0, head=h)); block = `HD` threads. Weights staged cooperatively in shared once per block; heads read them.

**Stage decomposition (operator-approved):** the fused layer runs **3 kernels on one `cudaStream_t`** — a strict single-launch would need cooperative grid sync between the block-wide attention (which wants a full persistent-KV pass) and the cross-block o_proj reduction:

1. `fused_qkv_kernel` (grid=NH, block=HD): input-norm → per-head inline NVFP4 q/k/v + gate → per-head RMSNorm(q/k) → inline RoPE → append k/v to persistent KV row `kv_base`.
2. `fused_attn_softmax_kernel` (grid=NH, block=HD): per-head block-wide online-softmax over attended KV rows [0..kv_base], value-accumulate to `out_h`, apply gate `× sigmoid(gate)` → write `out_h[NH*HD]`.
3. `fused_o_proj_kernel` (grid=NH, block=HD): each head accumulates its o_proj contribution into `d_out[0..H)` via cross-block `atomicAdd` (float accumulator).

The pseudocode below shows the per-stanza math; the three kernels are individually testable against the corresponding split launches.

```
host (RunQwenGeneration decoder step, full layer):
  launch compute_bounded_decoder_full_attn_kernel
      <<<NH, HD, smem, stream>>>
      (d_act, lw.* (q,k,v,o weights + q/k head-norms + rope theta),
       kv_k/kv_v base, d_out, kv_base)

kernel body (per block = head h):
  // shared: staged input token x0[0..H) once; staged q_proj cols (partial?),
  // o_proj needs full H->H. For HD-threads block we cannot hold 5120-wide
  // rows; instead each block computes ONLY its head's output columns of q/k/v
  // and reads shared weight tiles cooperatively in the K loop.
  x0 = d_act[0..H)            // read H=5120 once (in shared or re-read)
  // q/k/v projections for this head only:
  q_vec[HD], k_vec[HD], v_vec[HD] = dequant(proj(h, x0))
  q_vec = rmsnorm(q_vec) * headnorm
  k_vec = rmsnorm(k_vec)      // per-head norm
  rope(q_vec), rope(k_vec)    // inline cos/sin for position kv_base (t=0)
  // GQA: kvh = h/(NH/NKV); append k_vec[v] to kv_k[v][kv_base], v_vec similarly
  // online softmax over s in [0, kv_base+0] (causal full attended = kv_base)
  scores[s] = dot(q_vec, kv_k[kvh][s])*scale ; max/sum block-reduce (HD)
  // value accumulate (sink? name GP): out_h[d] = Σ exp(score)*inv * kv_v[v][s][d]
  out_h = out_h * sigmoid(gate)   // gate from q gate
  // o_proj for this head's contribution: d_out[0..H) += o_proj_cols(h, out_h)
  // (o_proj weight [H, NH*HD]; each block owns cols h*HD..(h+1)*HD of o)
  for n in 0..H: atomically or partial-accumulate d_out[n] += Σ_{d} o[n][h*HD+d] * out_h[d]
  __threadfence / cooperative block reduce
```

**Dataflow / memory:**
- The fused kernel reads each weight tensor **once per block cooperatively** in the K-loop (NVFP4 decode to fp16/registers), avoiding re-launch + re-D2H. `x0` (the residual token, H=5120) is read once into shared.
- q/k/v projection writes go straight to **registers** for the head (no `d_qc`/`d_q` scratch) → removes 4 scratch buffers for this path.
- RoPE computed inline from `kv_base` (position = kv_base) — no `d_pos`/`d_cos`/`d_sin` arrays.
- The o_proj contribution is accumulated into `d_out` via a cross-block atomics/`red.release` (or an accum-and-second-pass if we prefer). Since o_proj weight `[H, NH*HD]` is per-head-columns, a single block computes exactly its `HD` source columns → no extra pass needed: block loads o[n][h*HD..h*HD+HD) for all n, computes partial, and atomicAdds into `d_out[n]`.

**Online softmax & block-reduce** reuse the already-validated pattern from the parallel-attention step: `scores[s]` in a dynamic `slogits` shared, `m`/`sum` via a static `sreduce[HD]` (HD ≤ blockDim ≤ 512), value pass stride-d. This is bit-compatible with the step-2 kernel for the same scores.

**Numeric:** FP32 accumulate; re-grouping of the projection sum and softmax is allowed; NVFP4 decode identical. Tolerance contract: fused output must match the split path on the same `d_act`/weights within `1e-2` relative (half-roundtrip + fp32 regroup) — verified by a 1-token parity test.

## 4. Testing / acceptance

- **New** `tests/test_qwen_fused_decode_attn.cu`: 
  - Build a small synthetic `QwenLayerWeights` (tiny NH/HD/H, e.g. per `qwentest::TestCfg` geometry but forced to 1 token) with random NVFP4 q/k/v/o weights + norms.
  - Compute **reference A** by calling the existing split path (`RunQwenFullAttention` with seq=1) → `ref_out`.
  - Compute **reference B** with the new `RunQwenFusedDecodeAttnKernel` (public numeric entry, a la `RunQwenFullAttnKernel`) → `fused_out`.
  - `CHECK(abs(fused - ref) < 1e-2 * max(...) + 1e-6)` for each `H` element.
  - Also a **kv-persistence** check: after a fused call, then another fused call with `kv_base+1`, output matches an equivalent two-step split path (verifies KV write-back).
- **Existing**: `test_qwen_model` (synthetic full pipeline) — the decoder must still use the fused path when seq==1 and stay `ok`. Keep the split kernels intact (used for seq>1 and as the reference).
- Real `test_qwen_selfcheck` SKIP w/o checkpoint; if present, bit-exact `"Paris"` — note: with FP32/tolerance fusion the selfcheck parity is *not* byte-exact; confirm against the strategy §5 gate (tolerance, not bit-exact, for Qwen dense ops).

## 5. Files

- Modify `libs/lsxcommon/src/qwen_inference.cu` — add `compute_bounded_decoder_full_attn_kernel` + `RunQwenFusedDecodeAttnKernel`; wire it into `RunQwenGeneration`'s decoder branch (seq==1 && full layer) with a compile-time/`seq==1` guard; keep the split path for prefill (seq>1).
- Modify `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare `RunQwenFusedDecodeAttnKernel`.
- Create `tests/test_qwen_fused_decode_attn.cu`; add to `test_qwen_cu` OBJECT lib + foreach list (`tests/CMakeLists.txt`).
- No codec/CMake-else change. `QwenNvfp4Weight`/`QwenLayerWeights` reused as-is.

## 6. Open decisions (resolve in plan)

- **o_proj accumulation**: resolved — cross-block **`atomicAdd` on a `float*` scratch buffer** (FP32 accuracy; converted to half at the end), `NH×H` atomicAdds total, acceptable at 9B (NH=24, H=5120). A `__threadfence()` + final half-conversion kernel (or a fused tail) completes it.
- **RoPE**: inline trig from the `kv_base` position (the attended position is `kv_base` for token t=0) using a shared cos/sin table computed once per block. **Pair-ownership**: because each thread owns one `d`, RoPE's rotate-half pairing `(j, j+rot/2)` is done via a shared staging round-trip (threads write q/k pairs to `sq_pair[]`/`sk_pair[]`, sync, then pair-owning threads compute the rotation and write back).
- **q/k/v projection**: per-head inline dequant dot-product loop (per-head output N = HD is small; do not reuse the tile kernel whose block-per-output-columns geometry is wasteful at N=HD). The full-layer weight tile is loaded cooperatively into shared once per block (all heads reuse for the same layer), then each block consumes only its head's columns.
- **Persistent-KV append**: the fused kernel writes `k_vec`/`v_vec` into `kv_k[kvh][kv_base]` / `kv_v[kvh][kv_base]` (exactly the rows the split path uses for `kv_base`), and reads attended rows `[0..kv_base]` for the causal softmax.

## Self-review

Coverage: §1 quantifies the 8+ launches & round trips; §2 the goal (one launch per full-attn decoder layer); §3 design (per-head block, inline projections/rope/softmax/O via cross-block accum); §4 test (split-vs-fused parity + KV persistence); §5 files; §6 open (decision to resolve in plan). Consistent with previous step patterns (online softmax + sreduce, public numeric entry). Scope: single fused attention kernel + harness — a single plan. Numeric contract: FP32/tolerance (per user). No placeholders.
