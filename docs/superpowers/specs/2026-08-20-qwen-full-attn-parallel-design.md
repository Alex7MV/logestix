# Qwen Full-Attention Parallel Softmax — Design Spec (Direction A step 2)

**Date:** 2026-08-20
**Status:** Approved
**Scope:** `libs/lsxcommon` — production `full_attn_softmax_kernel` + its launch in `RunQwenFullAttention` (`libs/lsxcommon/src/qwen_inference.cu`). Direction **A** of the Blackwell optimization strategy.
**Numeric contract:** FP32 accumulate; the online-softmax grouping (max-n-subtract + exp sum + inverse) is preserved but the reduction/exp passes are restructured to be block-wide. Result stays within the tolerance envelope of a new parity test vs the existing thread-0 kernel and the synthetic `test_qwen_model`.

## 1. Problem

The production `full_attn_softmax_kernel` (`qwen_inference.cu:229-268`) computes causal GQA attention, but with two severe Blackwell-under-utilization defects:

1. **Thread-0-only scores:** `if (threadIdx.x == 0)` computes *all* `ctx+1` dot-products serially (244-252). At long context (`kv_base+seq` into the tens of thousands) this is a single-serial-stream bottleneck — the block's other 511 threads idle.
2. **Double-exp + scalar reductions:** the `m`/`sum` (max and softmax-denominator) are accumulated in full scalar scans (`255-258`) re-reading all scores, then the value accumulation (`261-267`) recomputes `__expf(slogits[j]-m)` *again per output element* — so every exp is computed up to `HD` redundant times and `m`/`sum` are serial.
3. Causal `v`-row re-read for each output `d`: each of `HD` threads re-walks all `ctx` V rows (though V rows are read once per d, cache-poor).

The score buffer `smem` (`kv_base+seq+1` floats) may exceed `blockDim.x`; the current kernel hides this because only thread 0 writes it.

## 2. Goal / non-goal

**Goal:** Replace with a block-wide online-softmax attention kernel:
- **Scores in parallel** across all `threads` (chunk-per-thread: each thread owns `ceil(ctx+1 / threads)` scores, computes its dots sharing the q-row).
- **Block-wide softmax reduction**: per-chunk local `m_i`/`sum_i`, then a block reduction (shared tree over `threads` partials) to the global max and denominator — removing the serial re-scan of all scores.
- **Softmax stats once, exp in the value pass only**: global `m`/`inv` computed once block-wide; each `d` thread computes `exp(s-m)*inv` per score during the per-d value accumulation (each exp appears once per output dim, exactly as today); no separate serial max/sum scans.
- Keeps the GQA `kvh = h/(heads/kv_heads)`, causal `ctx = kv_base+t`, `kv_base` incremental-decoding offset, and FP32 accumulate.

**Non-goal:** no change to QKV projections, RoPE, RMSNorm, output gate, or the `d_attn [seq, NH, HD]` layout. No cuBLAS path. No change to the current `__half` in/out.

## 3. Kernel design

Same grid: `blockIdx.x = t`, `blockIdx.y = h`. `threads = min(HD, 512)`. Shared: `slogits[ctx+1]` (may exceed blockDim.x — chunked).

```
score pass (all threads cooperate):
  ctx  = kv_base + t                       // global rows attended (inclusive)
  chunk_hi = (ctx+1) / threads
  my j range = [my*chunk, min((my+1)*chunk, ctx+1))     // covers ctx+1 scores
  for j in my range: slogits[j] = dot(q_row, k_row[j]) * scale      // parallel
__syncthreads()

global softmax constant (block-reduce m, then sum):
  local m_i = -inf; for j in my range: m_i = max(m_i, slogits[j])
  block-reduce m (shared tree over threads partials)  -> global m
  sum_i = 0;  for j in my range: sum_i += expf(slogits[j] - m)     // one exp
  block-reduce sum -> inv = 1.0/(sum + 1e-10)
__syncthreads()

value pass (parallel by output dim d, NOT by score chunk):
  for d = my_d; d < HD; d += threads:          // my_d = threadIdx.x
      acc = 0
      for j = 0..ctx:                          // all scores, shared slogits
          acc += expf(slogits[j] - m) * inv * half2float(v_row[j][d])
      out[t,h,d] = float2half(acc)
```

The `out` element per `(t,h,d)` is owned by exactly one thread (its `d`), summing over ALL `ctx+1` scores with the shared global `exp(s-m)*inv` — the second exp appears per `d` (necessarily, the V product differs per d), but the expensive softmax stats (m, sum) are computed once block-wide rather than re-scanned serially. This is the same FP32 per-element formula as today but with a block-wide (non-thread-0) softmax; each exp is computed `HD` times (once per d) exactly like today — the serialization saved is the score computation and the m/sum scans.

Critical numeric property: the **global max m** is computed before any exp, so `exp(s-m)` never overflows; FP32 accumulate; the float result if stored to half — same as today. Bit-exactness not intended; tolerance vs a fresh host-reference (which uses the current serial order? no — the new kernel's order is not the serial order; the harness compares to an exact FP64 dot-product+softmax over the same q/k/v, k-order and softmax grouped differently) — so a small relative tolerance (e.g. `1e-3`) vs FP64-based host is the contract. `test_qwen_model` (synthetic) passes under its tolerance.

## 4. Testing / acceptance

- **New** `tests/test_qwen_full_attn_parallel.cu`: synthetic small (seq=4, heads=2, HD=8, kv_base=0) and a long-context (kv_base=2000, seq=64) case. Host reference `attn_ref_fp64` (exact FP64 per (t,h,d) with causal mask) — the SAME reference used by the existing suite. `CHECK(abs(got - ref) < 1e-3*max(...)+1e-6)` per element.
- A direct **launch harness**: expose `RunQwenFullAttnKernel(cudaStream, q, k, v, out, seq, heads, head_dim, kv_heads, scale, kv_base, ctx_len)` that launches the production kernel so the test can drive it in squash without the whole high-level layer. (Mirror the `RunQwenNvfp4Gemm` external-numeric-entry pattern from step 1 — define in header, external linkage out of anonymous namespace.)
- **Existing**: `test_qwen_model` (synthetic full `RunQwenFullAttention` path) stays green; `test_qwen_full_attn.cu` keeps its own kernels and must also pass (it tests its own path, not production; do not add ours as a case name collision).
- Real-model `test_qwen_selfcheck` SKIP without checkpoint.

## 5. Files

- Modify `libs/lsxcommon/src/qwen_inference.cu` — replace `full_attn_softmax_kernel` body (keep grid/threads/smem contract from `RunQwenFullAttention`); add `RunQwenFullAttnKernel` public entry (analogous to `RunQwenNvfp4Gemm`).
- Modify `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare `RunQwenFullAttnKernel`.
- Create `tests/test_qwen_full_attn_parallel.cu`; add to `test_qwen_cu` OBJECT lib + the foreach executable list in `tests/CMakeLists.txt`.
- No CMake change elsewhere; no dequant changes.

## 6. Open decisions (resolve in plan)

- Exact chunking / slice: `ctx+1` may be ≫ threads; chunked with the shared exponent as a volume. The `smem` shared buffer size is `ctx+1` floats (`RunQwenFullAttention` keeps it); the kernel must not exceed that. `HD` may be 32-ish real, so `threads = min(HD,512)` in the real 9B (`head_dim=…` from topo).
- Whether to keep `smem` cap or grow for the full `ctx+1` — keep it (the caller already allocates `ctx+1+`), the kernel just requires `smem >= ctx+1`.
- Two exp passes vs one: choose one (one-on-accum) unless the chunk-reduction needs the second pass to be correctly weighted (online reshape) — if V accumulation uses the global inv, a full second pass is required; this is fine (one real exp, reuse stored scores in slogits).

## Self-review

Coverage: §1 problem, §2 goal (parallel online softmax, FP32) — matches the user's "parallel, FP32, tolerance" choice. §3 design preserves grid/contract. §4 test harness (parallel + fp64 ref + a parallel `RunQwenFullAttnKernel` numeric entry — matching the step-1 pattern). §5 files, §6 decisions (chunking / smem / one-vs-two pass). No AI-placeholders: chunk formula given; the tests list is concrete. Scope: one kernel, one harness — single plan. Checked: `test_qwen_full_attn` is a *different* kernel's own test; we do **not** reuse its name — ours is `test_qwen_full_attn_parallel`. Good.