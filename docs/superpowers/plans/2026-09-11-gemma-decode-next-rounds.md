# Gemma-4-26B-A4B-it Decode Optimization Roadmap

Date: 2026-09-11
Status: committed (not yet scheduled)
Baseline: 98.1 t/s (2048-tok gen, RTX 5090, EPYC 9V74, preset linux-release-znver4)
Journey: 55.9 (initial after IMA fix) → 74.5 → 77.9 → 98.1 t/s

## Remaining profile (nsys, 512-tok, prof8-9)

| Kernel | % GPU | ms/token | GB/s (estimate) | Comment |
|---|---|---|---|---|
| `lsx_nvfp4_gemv_kernel<2>` | 26.1% | 2.55 | ~400-420 | attn q/k/v/o + dense gate/up/down, ~185 launches/token |
| `RmsNormTreeKernel` (W1) | 13.5% | 1.32 | — | 108k launches, 6.2µs each |
| `gemma_gemv_pair_batch_kernel<2>` | 13.4% | 1.31 | ~410 | routed experts gate/up (fusion already present) |
| `gemma_logits_rows_nvfp4_kernel` | 10.3% | 0.99 | ~610 | LM head warp-per-row ILP2 |
| `gemma_split_kv_partials_kernel` | 9.4% | 0.91 | — | chunk 128; avg 30µs, tail up to 45µs |
| `gemma_gemv_down_batch_kernel<1>` | 7.4% | 0.72 | ~410 | routed down |
| `lsx_nvfp4_gemv_kernel<4>` | 3.8% | 0.37 | ~470 | large shapes (dense) |
| `gemma_router_softmax_topk_kernel` | 2.9% | 0.29 | — | 9.6µs, block/token |
| Misc (rope, embed, f16↔bf16, axpy, scale, combine) | ~13% | ~1.3 | — | small stuff at 0.5-4µs × many |

Total streaming-GEMV family ≈ 51% of GPU time, all at ~23% of peak bandwidth
(1792 GB/s). Wall target of the "hardware" ceiling ≈ 140-160 t/s.

## Directions (by decreasing payoff)

### R1. Shared NVFP4 GEMV: raise 400 → 900+ GB/s  (main lever, ~+25-30 t/s)

The `lsx_nvfp4_gemv_kernel` family (shared by qwen/glm/gemma — `lsx_inference_nvfp4.cu`)
holds ~26% of the time at ~23% of bandwidth. Reasons:
- KP=2: 2 warps per row, lanes read uint4 with a stride of 32 — on the gemma geometry
  (K=2816 → 176 uint4/row) that is 5-6 rounds of dependent loads with ILP=1.
- Neither prefetch nor double-buffering of the next weight portion.

Techniques (in order of bit-exactness-contract safety):
1. **ILP within a lane** — accumulate 2-4 uint4 chunks into independent
   accumulators (as done in the gemma head/router): the merge order is fixed
   → bit-exactness is preserved by construction. Verified by the parity tests
   `test_lsx_nvfp4_dispatch.cu` (144k assertions) and gemma/qwen host-reference.
2. **uint4×2 (32B) per transaction** — 2 adjacent uint4s, paired decode.
3. **cp.async/TMA prefetch** of the next chunk — only if (1)-(2) do not give
   2×; the most invasive option.
Estimate: GEMV family 13.9µs → 6-7µs, ~+25-30 t/s.
Risk: touches 3 models — run the full fast + gemma/qwen/glm kernel suites.

### R2. RmsNormTreeKernel: fuse into consumers  (~+10-15 t/s)

13.5% — 6.2µs per launch for <1µs of useful work. 108k launches/512 tokens
= 6-7 norms per layer × 30 layers. Options:
- **Fuse scale+norm** (the router pre-scale is already a separate kernel `gemma_scale_vec_kernel`
  — fold it into the norm call).
- **Fuse norm+bf16_to_half** on the inputs of the NVFP4-GEMV branches (currently a separate
  cast kernel before each chain).
- **Fuse residual+norm** (axpy → norm → one kernel).
Each pair removes −1 launch and −2-4µs of latency gap; target count <40k.

### R3. LM head → TC mxf4nvf4 / segmented TC path  (~+8-10 t/s)

10.3%: warp-per-row 0.95ms — bottlenecked by scalar nibble decode. The head is the only
V×H matrix (262144×2816, 368MB NVFP4), this is the TC class. Options:
- The `RunLsxNvfp4Gemm` dispatcher with a TC path (already present for seq==1) — but the head needs
  argmax, not full logits: a fused TC-GEMM + block-argmax is needed, or a TC GEMM into
  d_logits + a separate reduce.
- Contract: does the argmax winner change with summation order? NO — floating-point
  summation changes logits bits, the winner can change on a near-tie. A
  tolerance-envelope contract is needed (like split-KV attention) or an explicit
  bit-exactness-vs-accuracy decision.

### R4. Split-KV: remove the 45µs tail  (~+3-5 t/s)

Chunk 128 gave avg 30µs, but max 45µs (with a window band → 8 chunks). Tail causes:
warp divergence on boundary chunks + two cudaMallocAsync/launch.
- `d_part/d_st` pools in GenWorkspace (remove mallocasync from the hot path).
- Chunk 64 at a small band (early tokens): grid 16×30 blocks.
- Balancing: chunk by band, not fixed.

### R5. Router phase-2: block 256 → warp-scan over 32 threads  (~+1-2 t/s)

9.6µs still: block-reduce with 8 barriers. Option: 1 warp, ballot-argmax
with lower-index tie-break via `__ffs(__ballot(...))` — 0 barriers.

### R6. Converter packing [gate|up] for dense-MLP + attn qkv  (~with R1, +5-8 t/s)

Like GLM-Flash pair-GEMV: pack gate/up into one NVFP4 pack → one launch
`RunLsxNvfp4GemmGateUpPair` instead of two. For dense (I=2112, K=2816) and qkv.
Format change: a new layout in the logestix-gemma index, converter + loader +
dispatcher. Do it AFTER R1 (shared kernel first).

### R7. Expert batch: more K in flight  (infrastructural)

`gemma_gemv_pair_batch/down_batch` already batch K=8 experts. The ILP depth
within a batch is limited by the shared reduction. Re-evaluate after R1.

## Discipline

- Every item: TDD, bit-exactness/determinism contract explicit in the test.
- R1 touches the shared layer — run fast 41/41 + all kernel suites of the 3 models
  + the real-model selfcheck of each.
- Re-profile after each item (nsys, 512 tok, the same prompt).
- Record the speed in AGENTS.md (gemma_moe_model entry) after reaching a
  stable number on a full run (2048 tok).
