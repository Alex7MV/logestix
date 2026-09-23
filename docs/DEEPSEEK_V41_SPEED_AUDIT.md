# DeepSeek-V4.1-Flash — Decode Speed Audit

Date: 2026-09-23
Build: `linux-release-znver4-aocc` (perf reference), CUDA 13.2, RTX 5090
Model: `/mnt/llm_models/models/DeepSeek-V4.1-Flash` (`logestix-v41-nvfp4`, 40
layers, 384 experts/top-6, 285 GiB host expert store, 698-slot device LRU)

Tooling: `nsys profile -t cuda` + sqlite query (`ncu` unavailable on this box).
nsys overhead was calibrated to be negligible: profiled vs plain runs agree
within 1% (512: 11.2 vs 11.3 t/s; 1024: 8.8 vs 8.9 t/s).

## Reproduced baseline (fixed-length, prompt "The capital of France is")

| max_gen | t/s | ms/token | TTFT | power | cache hit |
|---|---|---|---|---|---|
| 2048 (canonical) | **6.8** | 147 | 530 ms | 153 W | 53.9% |
| 1024 | 8.9 | 115.6 | 526 ms | 165 W | 57.0% |
| 512 | 11.3 | 88.7 | 526 ms | 178 W | 58.1% |
| 64 | 13.8 | 72.3 | 533 ms | 235 W | 58.6% |

Committed baseline was 6.64 t/s (2026-09-21); reproduces. Decode speed degrades
roughly linearly with context (~+35 ms/token per +512 tokens). Startup pinning
of the 288 GiB expert store costs 113 s cold / 46 s warm — separate from decode.

## Attribution (nsys, decode window)

| len | wall/tok | kernel/tok | expert H2D/tok | host CUDA-API/tok | cudaFree/tok |
|---|---|---|---|---|---|
| 64 | 72.3 ms | 36.2 | 37.2 ms (129 GiB) | 69.2 ms | 53.1 ms (217/tok) |
| 512 | 89.6 ms | 55.4 | 35.4 ms (967 GiB) | 86.7 ms | 71.8 ms (204/tok) |
| 1024 | 113.6 ms | 78.8 | 36.1 ms (1972 GiB) | 110.7 ms | 95.8 ms (203/tok) |

- **`wall ≈ kernel + H2D` at every length** (512: 46.5 s vs 45.9 s span) ⇒ copy
  and compute effectively **do not overlap** (GPU busy union 44.22/45.87 s, only
  3.6% idle). Expert H2D runs at **57.3 GB/s** — the PCIe Gen5 ceiling; no
  bandwidth headroom.
- The host thread is occupied almost the whole window by CUDA API calls,
  dominated by `cudaFree`.

## Kernel breakdown (share of GPU kernel time)

| kernel | 64 tok | 512 tok | 1024 tok |
|---|---|---|---|
| `indexer_topk_kernel` | 0.7% | 19.0% | **35.7%** |
| `sparse_attn_half_kernel` | 10.9% | 26.6% | 26.2% |
| `hc_mixes_kernel` | 24.2% | 15.0% | 10.5% |
| `GemvKernel<half,half,0>` | 23.7% | 14.7% | 10.3% |
| `lsx_nvfp4_gemv_kernel<4>` | 11.5% | 7.2% | 5.0% |
| `router_topk_kernel` | 6.7% | 4.1% | 2.9% |

## Root causes (ranked)

**P0a — `cudaMalloc`/`cudaFree` per layer (host serialization).**
`SharedExpertForward` (`deepseek_moe_v41_inference.cu:598-621`) does 4+4
allocations per layer, `LaunchRouterTopK` (`:665/675`) 1+1, and the ratio-0
`Scratch::Alloc/Free` (`deepseek_moe_v41_inference_attn.cu:157/174`) 1+1 ⇒
**~204 alloc + 204 free per token**. `cudaFree` alone occupies 72–96 ms/token of
host time. The routed-expert path already avoids this via a preallocated scratch
(`deepseek_moe_v41_inference.h:142-145` explicitly documents that cudaFree
serializes the device and kills copy/compute overlap); the shared-expert and
router paths were not converted.

**P0b — single-thread top-k selection.** `indexer_topk_kernel`
(`deepseek_moe_v41_inference_attn.cu:43-71`) returns all but thread 0 and does
an O(topk×n_pos) serial scan-in-select (topk=512, n_pos≈seq); 8 calls/token
(index-source layers). It grows linearly: 5.4 s@512 → 28.8 s@1024 (35.7%).
Also single-thread: `select_candidate_blocks_kernel` (`:76-114`) and
`router_topk_kernel` (`deepseek_moe_v41_inference.cu:377`).

**P1a — `sparse_attn` occupancy.** `deepseek_moe_v41_inference.cu:459-476`
launches grid `(1, heads=64)`, 32 threads ⇒ only 64 warps total (~0.8% of the
48 warps/SM budget; `docs/NVIDIA_DOCS_NVFP4_MEMORY.md` §B1). 26% of kernel time,
~513 µs/layer; the KV latent is re-read per head.

**P1b — zero copy/compute overlap.** `deepseek_moe_v41_inference_pipeline.cu:570-577`
does a D2H router readback + `cudaStreamSynchronize(stream)` on every layer (40
syncs/token); expert copies are issued only after the sync. 103 misses/token ×
19.9 MB = 2.05 GB/token is the exposed 35.8 ms/token floor.

**P2 — `hc_mixes` constant 8.3 ms/token**, 2 launches/layer, one block, Sinkhorn
(20 iters) on thread 0 (`deepseek_moe_v41_inference.cu:71-153`).

**P2 — launch count / fine-grained kernels.** ~3864 kernel launches per token;
`cast_half_to_f32` 526/token and `cast_f32_to_half` 366/token are not fused into
SwiGLU.

**P2 — expert byte volume.** per-16 E4M3 scales (1/16 of elements). The source
is E8M0/32; storing per-32 UE8M0 (`kind::mxf4`, SF UE8M0 vec32,
`docs/NVIDIA_DOCS_NVFP4_MEMORY.md` §B4) would cut ~5.5% of the 2.05 GB/token.

## Recommended order

1. **P0a** — preallocate shared-expert + router scratch (numeric-neutral).
2. **P0b** — parallelize `indexer_topk`, `router_topk`, `select_candidate_blocks`
   (must preserve exact selection: ties → lower index).
3. **P1a** — split-KV / multi-warp sparse attention with fixed reduction order.
4. **P1b** — decouple expert selection from the host sync (device-side ids or
   speculative next-layer prefetch) to overlap the 35 ms/token H2D.
5. **P2** — fuse casts, merge `hc_mixes`, evaluate per-32 scales.

## Numeric-class note

Per `docs/numeric-contract.md` any reduction-order change (split-KV, parallel
top-k composition) is **C-fast** unless it preserves the golden token and
run-to-run determinism. Each change must keep
`test_deepseek_moe_v41_selfcheck` (Paris, bit-exact) green and must be
re-measured twice at `--max-gen 2048`.
