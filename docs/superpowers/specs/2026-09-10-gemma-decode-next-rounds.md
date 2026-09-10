# Gemma-4-26B-A4B decode acceleration: status and next rounds

Date: 2026-09-10
Status: round 1 landed (see `2026-09-10-gemma-nvfp4-decode.md` plan + commits
`8e5a401..d42fdef` rebased as `aa55b20..d42fdef`); rounds 2-4 queued below.

## Where we are (end of round 1)

- Model: `/mnt/llm_models/models/gemma-4-26B-A4B-it-nvfp4` (converted with the
  NVFP4 pipeline; legacy BF16 conversion at `/mnt/llm_models/models/
  gemma-4-26B-A4B-it` still loads — back-compat verified).
- Decode: **45.0 t/s** (was 30.9 at session start, 38.7 after the MoE fast-path
  fix). Baseline reference: Qwen3.6-35B-A3B = 103 t/s on the same GPU.
- Pinned VRAM: 4.5 GB → 1.0 GB.
- All NVFP4 dense weights route through the neutral dispatcher
  (`RunGemmaNvfp4Gemm` in `gemma_moe_inference.cu`); GemmaBaseModel opens the
  per-call `LsxFastMathScope(cfg_.fast_math, /*seq1_gemv=*/true)`.
- Verified: all gemma suites + `test_qwen_nvfp4_gemv` green; selfcheck Paris
  bit-exact ×3 cycles on BOTH conversions.

## Why 45 t/s is the current ceiling

Decode is weight-bandwidth-bound: ~4.3 GB of NVFP4 weights read per token
(dense MLP 30×3×2112×2816, attention q/k/v/o incl. 5 full-attention layers
with o_proj [2816, 8192], tied lm_head 262144×2816, MoE top-8) at ~200 GB/s
effective streaming bandwidth ≈ 21.5 ms/token. Qwen3.6 reads roughly half the
bytes (hidden 2048, no dense MLP, vocab 248k) — its 103 t/s is the same
bandwidth on half the traffic, not a better kernel.

Measured negative results (do NOT retry without new data):
- **TC mxf4nvf4 vs GEMV at decode shapes: parity** (45.1 vs 44.9 t/s).
  M=1 underfills TC tiles; DRAM-bound either way. A/B harness: `--fast-math`
  + `LSX_GEMV_MAX_MIB=0` (bypasses the GEMV size gate in the dispatcher).
- Router-handshake removal and full expert residency: already in (round 1).

## Round 2 — overlap the weight streams (highest expected gain, ~1.5-1.8×)

Every kernel today streams weights SERIALLY on one stream: while the q GEMV
runs, the o/MLP/MoE weight bytes idle. The DRAM controller is not saturated
by a single GEMV at M=1 (grid fills SMs but per-SM load pipelines stall on
latency).

- Dual-stream staging like `glm_moe_flash_model` (dual-stream expert staging
  with event plumbing): issue the NEXT projection's weight-touching work so
  two GEMVs are always in flight. Concretely: split each layer's weight
  reads across a compute stream and a prefetch stream with `cudaEvent`
  fencing; or restructure `ForwardLayer` so q and k GEMVs run on stream A
  while v (tied: free) runs on stream B.
- TMA bulk-tensor staging of weight slabs (`LsxProbeTmaBulkCopy` already
  probes capability; `lsx_tc_gemm.cu` has the mbarrier staging skeleton) to
  convert the GEMV's latency-bound loads into bulk copies.
- Gate: `LSX_GEMMA_DUAL_STREAM=1` env first; default after A/B wins.
- Expected: 45 → 60-70 t/s if achieved bandwidth rises to ~300 GB/s.

## Round 3 — cut the remaining bytes (~1.15-1.25× on top)

- LM head NVFP4 GEMV is already the biggest single consumer (368 MB/token).
  Its rows kernel accumulates in scalar FP32; a VPB (values-per-block) x2
  ILP variant like the tile kernel's VPT=2 may lift its bandwidth share.
- KV cache is BF16; the sliding-window layers (25/30) re-read up to 1024
  rows × 2048 cols per token. FP8 KV for the K/V rows (decode-only tolerance
  class) would halve that traffic; needs a parity study on the real
  checkpoint before committing (selfcheck is bit-exact today — KV quant
  breaks that contract, gate behind `LSX_GEMMA_KV_FP8=1` and relax the
  selfcheck to tolerance for that mode explicitly).

## Round 4 — scheduler-level (multi-request only)

- Batch decode across requests (M>1 amortizes weight reads across the
  batch): out of scope for the single-stream CLI but the natural next step
  if fabric serves concurrent sessions. Requires seq>1 GEMM-class kernels
  (TC path becomes relevant again at M>=8) and a session scheduler in
  `lsxfabric`.

## Conventions for the next session

- Perf measurement: `apps/lsxfabric/lsxfabric --localmodel
  /mnt/llm_models/models/gemma-4-26B-A4B-it-nvfp4 --testprompt --max-gen 64`
  (3 runs, report t/s; 28-token generations end on EOS — compare with
  `--prompt` fixed and same token counts).
- Rollback envs (all verified): `LSX_GEMMA_MLP_BF16=1` crashes on the NVFP4
  conversion (legacy weights absent — by design, rollback = use the legacy
  conversion directory), `LSX_GEMMA_GEMV=0` / `LSX_GEMMA_MOE_BATCH=0` restore
  legacy MoE routes, `LSX_GEMMA_EXPERT_CACHE_SLOTS=N<3840` disables
  no-handshake routing.
- Debug: `LSX_TEST_GEMMA_DEBUG_SYNC=1` (per-stage sync fault pinpoint);
  the hash-instrumentation from the d_gemv buffer hunt was removed from
  `gemma_moe_inference.cu` — re-add per-stage FNV hashes of `ws.d_x` if a
  nondeterminism hunt repeats (pattern: hash after embed, attn, mlp, moe).
- Model-independence guard: `test_model_independence_guard` requires the
  per-call `LsxFastMathScope` marker in `gemma_base_model.cpp` (one-line
  comment) — keep it when touching the policy.
