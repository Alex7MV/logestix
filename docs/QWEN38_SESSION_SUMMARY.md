# Qwen3.8-Flash-Next Session Summary (for next session)

## Objective
Ingest Qwen3.8-Flash-Next per Model Ingestion Protocol, TDD.
Branch feat/model-Qwen3.8-Flash-Next: MERGED to main (f042db6, ff) and
deleted (local + remote). Ingestion COMPLETE.

## STATUS: DONE - merged to main 7aa7008..f042db6
Commits: 05958a3 (accuracy), 250ad09 (decode 15x), f8b04a0 (copy-stream
overlap), f042db6 (O(1) LRU bookkeeping). SelfCheck Paris green, suite 5/5.

## Decode tok/s: 0.17 -> 2.5 tok/s (6.0 -> 0.40 s/token) — 15x
nsys profiling drove everything. History:
1. hc dequant cache (BuildExpHcDequantCache): NVFP4 hc down/up/inject
   dequantized ONCE at load into per-layer fp16 device caches
   (ExpHcDequantCache in qwen_moe_exp_hc_cache.h; built in
   LoadQwen4ExpDenseWeights per layer + mixer; ~1.27GB fp16 total).
   RunExpHcGatedResidualCached consumes them. Legacy dequant path kept
   as fallback (cache null). fp16 path bit-identical (same dequant
   kernel; lossless upcast in kernel via WToF<__half>).
2. ple_inject_kernel WAS single-thread (thread 0 does all work):
   82% GPU time, 4.46 s/call! Parallelized 1024 threads, phase barriers,
   thread-0 norm reductions + thread-0 gate (bit-exact op order kept).
   Now 28 ms/call (160x). One block (1024 thr) is enough.
3. hc_gated_residual_f32_kernel: double->fp32 accumulation (GeForce
   FP64 = 1/64), warp-per-row GEMV + ordered shfl reduce, block 1024.
   Parity test tolerance Approx(2e-3) — unaffected. 8.7ms -> 1.47ms.
4. Multi-block hc fast path (RunExpHcGatedResidualCached): separate
   launches hc_norm_kernel / hc_down_gemv_kernel / hc_up_gemv_kernel /
   hc_epilogue_kernel — single-block was 1 SM bandwidth-bound (9GB/s
   eff). All 4 hc stages now 2.7ms/token total (was 142ms).
5. ExpExpertStore: single pinned arena (64.8GB RAM) for all 48 slabs.
   Was 48 x cudaMallocHost ~660ms each = 31.8s of load time. Load now
   47s total (disk-bound ~185ms/layer).
   CRITICAL GOTCHA: RunQwenMoeRoutedFFN consumes ExpExpertStore through
   reinterpret_cast<QwenMoeExpertStore*> — new fields MUST go after the
   identical-prefix members (h_slabs..gemv_ready), or layout mismatch
   produces 'expert out of range: eid 71' garbage errors.

## Current profile (0.40 s/token, 64-token --testprompt = 25.4s wall)
- GPU busy ~12s (47% util). qwen_nvfp4_gemm_tile 66% GPU (routed+shared
  experts, 117ms/tok); PLE 15.8% (28ms/tok); hc fast path ~7%.
- DECODE host-side is now CLEAN: memcpyAsync 7.75ms total on decode
  (the 8.9s in cuda_api_sum is LOAD-phase dense uploads through the
  staging ring — profile-time, not decode). cudaMalloc/Free pairs
  removed from hot loop (PLE conv scratch, mixer scratch persistent).
  O(1) LRU bookkeeping (h_prev/h_next recency chain + h_free stack in
  expert_device_cache.h; legacy array paths kept for unit tests).
- REMAINING WALL GAP (structural): per-layer router D2H readback +
  cudaEventSynchronize 48x/token — GPU idles while host enqueues each
  layer (host-driven LRU streaming limit). Next structural levers:
  speculative expert prefetch (route is unknown until router runs, but
  PREVIOUS token's routes could warm the cache), CUDA graphs per layer
  (dynamic slot pointers complicate capture), or full residency
  (24576 x 2.7MB = 65GB — needs GPU with >72GB VRAM).

## Round-3 fixes (commit 05958a3, accuracy phase) — see git log
1. block_inject_weight never dequantized (EnsureInjectScratch added).
2. Arena d_mixed_f region sized H not hc.
3. d_moe_acc zeroed per layer.
4. PLE norm-THEN-weight, conv tap w{K-1}->current, F.silu restored.
5. Mixer hc_norm override (d_hc_norm_override param).
6. Deterministic warp-per-row head GEMV (cuBLAS split-K nondeterminism).
7. PLE/QSA state memset per generation.

## Build/test commands (AOCC toolchain)
- Build: cmake --build build/linux-release-znver4-aocc --target lsxfabric -j 160
- Tests: ctest --test-dir build/linux-release-znver4-aocc -R "qwen_moe_exp"
- lsxfabric: --localmodel /mnt/llm_models/models/Qwen3.8-Flash-Next --testprompt
  (use --localmodel; vault ROOT path, NOT .../logestix)
- Profile: nsys profile -o /tmp/opencode/p --trace=cuda --sample=none \
    ./build/.../apps/lsxfabric/lsxfabric --localmodel ... --testprompt
  nsys stats --report cuda_gpu_kern_sum p.nsys-rep | sed -n '6,20p'
  nsys stats --report cuda_api_sum p.nsys-rep (host-side costs)

## Gotchas (cumulative)
- dequant_shared_nvfp4(NULL, 0, ...) returns cudaErrorInvalidValue — guard
  empty-weight dequant paths.
- Arena region sizing: fp32 scratch written by a kernel MUST be sized for
  the kernel's actual write (H floats, not hc).
- GDN norm: HF RMSNormGated applies weight DIRECTLY (w*normed); hc/PLE
  norms: (1+w) AFTER RMS.
- cuBLAS GEMM_DEFAULT nondeterministic (split-K atomics) — custom
  ordered-reduce GEMV where bit-exact parity required.
- LRU expert values deterministic; only layout varies — parity safe.
- cudaStreamSynchronize BEFORE any D2H read of stream-ordered results.
- GeForce FP64 = 1/64 FP32 — never accumulate in double on device.
- Single-block kernels are 1-SM bandwidth-bound (~9GB/s eff) — split
  phases into separate launches for multi-SM.
- reinterpret_cast pairing QwenMoeExpertStore<->ExpExpertStore: keep
  member order identical; append new fields at the END only.
- cudaMallocHost of GB-scale buffers page-locks at ~0.5ms/MB — batch
  into one arena (48 slabs cost 31.8s; arena costs 0.7s once).

## Key artifacts
- /tmp/opencode/parity/hf_parity2.py: FIXED parity harness (per-layer dumps).
- /tmp/opencode/qwen38_profile{,2..7}.nsys-rep — decode profiles.
- /tmp/opencode/qwen38-session-summary.md — this file.
- HF reference: /tmp/opencode/hfcheck/transformers/models/qwen4_exp/modeling_qwen4_exp.py
