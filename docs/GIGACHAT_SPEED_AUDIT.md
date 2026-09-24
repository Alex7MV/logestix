# GigaChat3.5-432B-A28B-Reasoning — Decode Speed Audit

Date: 2026-09-24
Build: `linux-release-znver4-aocc` (perf reference), CUDA 13.2, RTX 5090
(32.6 GB, PCIe 5.0 x16)
Model: `/mnt/llm_models/models/GigaChat3.5-432B-A28B-Reasoning`
(`logestix-gigachat-nvfp4`, 40 layers, `first_k_dense=3` ⇒ 37 MoE layers,
256 experts/top-8 ⇒ 9472 experts ≈ 219 GiB NVFP4 host store; MLA
full-attention at {3,7,…,39}, GatedDeltaNet elsewhere, ZeroCenteredGatedNorm
everywhere).

Tooling: `nsys profile -t cuda` + `nsys export sqlite` (`ncu` unavailable on
this box). nsys overhead is negligible for wall rate: profiled 256-token run
179.9 ms/token vs canonical plain 2048-token run 176.8 ms/token decode (<2%).

## Reproduced baseline (fixed prompt "The capital of France is")

| run | max_gen | t/s | total ms | TTFT | decode ms/tok | power | resident |
|---|---|---|---|---|---|---|---|
| canonical #1 | 2048 | **5.5** | 369856 | 7926 | 176.7 | 156 W | 26788 MiB |
| canonical #2 | 2048 | **5.5** | 370231 | 8068 | 176.8 | 156 W | 26788 MiB |
| nsys | 256 | 4.8 | 53768 | 7889 | 179.9 | 238 W | 26765 MiB |

Two canonical runs agree to 0.1%. The committed `perf_baselines.json` entry is
4.84 t/s (2026-09-21, mode `perf`), so the current tree is ~13% faster than the
record and the record is stale. Startup pins the 219 GiB expert store in
14 × 16 GiB chunks: 24–75 s (page-cache dependent) with peak host RSS 443 GiB.

## Attribution (nsys, decode window = 255 steps after prefill)

| term | ms/token | share | note |
|---|---|---|---|
| GPU kernels | 112.5 | 63% | 181 norms + 37 routers dominate |
| H2D expert copies | 65.8 | 37% | 3.74 GB/token @ **56.8 GB/s** |
| D2D copies | 0.05 | — | residual/rot |
| D2H copies | 0.06 | — | 513 KB logits/token |
| **sum** | **178.3** | | vs measured wall **179.9** |

`wall ≈ kernels + H2D` ⇒ **compute and copy do not overlap at all**. GPU
kernel-busy union is 28.69 s of 45.88 s (37.5% idle), and the idle is exactly
the copy time. H2D runs at 56.8 GB/s — the PCIe Gen5 ceiling, no bandwidth
headroom.

Host side, per token: **37 `cudaStreamSynchronize`** (one per MoE layer), **905
H2D `cudaMemcpyAsync`** (6 per missing expert: gate/up/down packs + scales),
and **1033 `cudaMemcpyAsync` total**; host-side `memcpyAsync` time is
174 ms/token. Prefill is the same story: 44 prompt tokens × 179 ms = TTFT
7926 ms (kernel 4.94 s + memcpy 2.88 s).

## Kernel breakdown (share of decode GPU kernel time)

| kernel | ms/tok | ms/call | calls/tok | share |
|---|---|---|---|---|
| `zero_centered_gated_norm_kernel` (dim 7168) | 56.6 | **0.352** | 161 | 50.4% |
| `router_topk_kernel` | 37.3 | **1.008** | 37 | 33.2% |
| `lsx_nvfp4_gemv_expert_batch_kernel<4>` | 5.7 | 0.077 | 74 | 5.1% |
| `lsx_nvfp4_gemv_expert_batch_kernel<2>` | 2.8 | 0.076 | 37 | 2.5% |
| `lsx_nvfp4_gemv_warprow_kernel` | 2.5 | 0.055 | 46 | 2.3% |
| `gdn_recurrent_kernel` | 1.6 | 0.052 | 30 | 1.4% |
| `GemvKernel<bf16,bf16,2>` | 1.5 | 0.048 | 31 | 1.3% |
| `lsx_nvfp4_gemv_kernel<8/4/2/1>` | 2.3 | — | — | 2.0% |
| all attention/rope/softmax/casts/fusions | <2.0 | — | — | ~2% |

The two top kernels are **83% of all kernel time and 52% of the whole decode
wall**, and neither is a matmul. All NVFP4 GEMVs together are ~13 ms/token.

## Root causes (ranked)

**P0a — `zero_centered_gated_norm_kernel` is a serial single-block kernel.**
`gigachat_moe_norm_kernels.cu:20-59` launches grid `(rows=1,1,1)`, 256 threads
(`:113`). Thread 0 alone does the 7168-element sum-of-squares *and* the
7168-element normalize+store (`:32-38`); only 16 threads do the 7168-long
`w_up` dot product (`:41-48`); the final `w_down` step uses all 256 threads but
only 16 FMAs each. The three serial-ish phases are latency-bound global-load
chains on one SM (~8 warps of the 48/SM budget; `NVIDIA_DOCS_NVFP4_MEMORY.md`
§B1). Measured 352 µs/call × 161 calls/token = **56.6 ms/token**. The same
kernel at dim 1536/512 costs 72 µs / 24.5 µs — the cost scales with the serial
length, not the useful work.

**P0b — `router_topk_kernel` is a serial single-block kernel.**
`gigachat_moe_kernels.cu:33-88` launches grid `(1,1,1)`, 256 threads (`:127`).
Each of the 256 threads computes one expert's 7168-long dot product as a
dependent BF16 load+FMA chain (`:44-52`), then thread 0 runs the O(top_k·E)
selection (`:55-87`). Measured 1008 µs/call × 37 calls/token = **37.3
ms/token** for a [256×7168] matvec + top-8. Latency-bound on the same single
SM.

**P1 — expert H2D is serialized on the compute stream with zero overlap.**
`gigachat_moe_expert_cache.cpp:273-300` issues 6 `cudaMemcpyAsync(...,
cudaMemcpyHostToDevice, stream)` per missing expert **on the compute stream**
(there is no dedicated copy stream). The per-layer router readback + full
stream drain (`gigachat_moe_inference_pipeline.cu:482-485`) precedes the
fetch, so every copy is fully exposed. Cost 65.8 ms/token, 3.74 GB/token at the
PCIe ceiling; measured miss rate **151/296 experts per token (49% hit)** across
a device LRU auto-sized to free VRAM minus a 4 GiB reserve
(`gigachat_moe_expert_cache.h:145`). The host also issues 1033 memcpyAsync
calls/token (174 ms/token host-side), so after P0a/P0b are fixed the issue
**rate** and copy size become the next limit.

**P2 — prefill is token-by-token.** `gigachat_moe_model.cpp:199-210` calls
`pipeline_.Step` once per prompt token, so TTFT = n_prompt × full-forward
(179 ms/token). A 44-token frame costs 7.9 s. The pipeline has no batched GEMM
path (`RunLsxNvfp4Gemm` is always `M=1`), and prefill re-fetches the same
per-layer expert sets token-by-token (2.9 s of prefill is expert H2D).

**P3 — fine-grained launches/casts.** 38250 `f32_to_f16_kernel` + 38250
`f16_to_f32_kernel` (150 each/token) and 117 `add_f32_kernel`/token are not
fused into the norms/GEMVs; ~2 ms/token total, plus **1413 kernel
launches/token** at 4.28 ms/token host. Secondary.

## Recommended order

1. **P0a** — rewrite the gated norm to a fixed-geometry parallel kernel
   (multi-warp block reduction for the RMS term, warp-per-row split for the
   `w_up` dot). ~56.6 → ~2-5 ms/token expected.
2. **P0b** — parallelize the router matvec (split hidden across warps with a
   fixed reduction tree; keep the exact top-k tie rule, lower index wins).
   ~37.3 → ~2-4 ms/token expected.
3. **P1** — dedicated copy stream + `cudaMemcpyAsync` batching (one expert =
   3+3 copies) and device-side expert-id resolution to drop the 37 host syncs;
   explore overlap of the routed fetch with the independent shared-expert path.
4. **P2** — chunked/parallel prefill (batched MLA/GDN/GEMV) to cut TTFT; the
   per-layer expert union across the chunk also cuts prefill H2D.
5. **P3** — fuse the f16↔f32 casts into the consuming kernels.

P0a + P0b alone remove ~94 ms/token of a 179.9 ms/token wall ⇒ a projected
**~85 ms/token ≈ 11-12 t/s** (≈2.2×), after which P1's 65.8 ms/token expert
floor dominates and must be overlapped or reduced.

## Post-P0 result (2026-09-24)

| step | 2048 t/s | decode ms/tok | TTFT | norm ms/tok | router ms/tok | kernel ms/tok | H2D ms/tok |
|---|---|---|---|---|---|---|---|
| baseline | 5.5 | 179.9 | 7926 | 56.6 | 37.3 | 112.5 | 65.8 |
| P0 | **10.6** | **94.4** | **4179** | 7.9 | 1.9 | 27.3 | 65.5 |

- **P0a** (`gigachat_moe_norm_kernels.cu`): parallel RMS warp/block tree +
  warp-per-row `w_up` dot, full warps. Microbench 243.6 → **23.8 µs/call**
  (dim=7168); production 56.6 → 7.9 ms/token.
- **P0b** (`gigachat_moe_kernels.cu`): multi-SM `router_score_kernel` (one block
  per expert) + block-parallel `router_select_kernel` (fixed masked-argmax
  tree, ties → lower index). Microbench 1005 → **26.3 µs/call**; production
  37.3 → 1.9 ms/token. The former single-SM kernel was issue-bound (9.2 M
  instructions at ~4/cycle/SM).
- E2E 5.5 → **10.6 t/s** (×1.93), reproduced twice within 0.03%; TTFT
  7.93 → 4.18 s (prefill runs the same kernels). Power 156 → 193 W.
- Correctness: `test_gigachat_moe_selfcheck` Paris, `logits_hash=9b46473108d82b99`
  identical across 3 cycles; `test_gigachat_moe_attn`,
  `test_gigachat_moe_parity` and `test_gigachat_moe_devkernels` (with the two
  new `[budget]` cases) all green.
- New wall: kernel 27.3 ms/token (29%) + H2D 65.5 ms/token (69%), still
  additive. **The expert H2D is now the dominant term** and P1 is the next
  lever.

## Numeric-class note

The default GigaChat path is `kAuto` and is C-exact (R3) by the numeric
contract (`docs/numeric-contract.md` §1). P0a/P0b change a **reduction order**
(currently thread-0 sequential). Per §3 a reduction reorder is C-exact only if
the golden token sequence is preserved, run-to-run `final_logits` stay
bit-identical, and the selfcheck stays green; otherwise it is C-fast. P0a's
post-feedforward norm output is added to the F32 residual, so its bits reach
the logits directly. Both P0 changes cleared `test_gigachat_moe_selfcheck`
(Paris, ×3, identical `logits_hash`) plus the parity tests, so the default path
stays C-exact and golden-token-correct.

## Results log (branch `perf/gigachat-decode`)

| step | max_gen | t/s | notes |
|---|---|---|---|
| audit | 2048 | 5.5 | baseline; committed entry 4.84 |
| P0a+P0b | 2048 | **10.6** | norm + router parallel; bit-exact selfcheck, reproduced ×2 |
