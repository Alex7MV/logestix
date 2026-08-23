# Blackwell Audit Phase 3 — Tensor-Core NVFP4 Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native tensor-core NVFP4 GEMM path (W4A4) behind a runtime flag, targeting tens of tok/s for Qwen3.5-9B on RTX 5060 Ti.

**Architecture:** New `.cu` object library compiled EXCLUSIVELY for `compute_120a/sm_120a` + `compute_121a/sm_121a` (the `kind::mxf4nvf4` PTX qualifier does NOT assemble on plain sm_120 — verified against PTX ISA 9.3). Kernel: warp-level `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`; weights consumed in the existing `[N,K]` nibble-packed layout (= B col-major K-major — zero repack); activations quantized on the fly to NVFP4 (per-16 E4M3, reusing the DeepSeek `act_quant` pattern). Default scalar path untouched (bit-exact contract preserved).

**Tech Stack:** CUDA 13.3 (PTX ISA 8.7+ features; toolkit ships 9.3), inline PTX asm, Catch2.

**Verified instruction template (PTX ISA 9.3, verbatim-derived):**
```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
    .f32.e2m1.e2m1.f32.ue4m3
  {d0,d1,d2,d3},        // 4 x f32 acc (std m16n8 layout)
  {a0,a1,a2,a3},        // 4 x b32 = 32 x e2m1 per lane; lane quad t covers cols 8t..8t+7 (+32)
  {b0,b1},              // 2 x b32 = 16 x e2m1 per lane; col = lane>>2
  {c0,c1,c2,c3},
  sfA, {0, 0}, sfB, {0, 0});  // 4X => byte-id/thread-id must be 0
```

---

### Task 1: MMA probe — single-instruction correctness

**Files:**
- Create: `tests/test_blackwell_tc_mma.cu`
- Modify: `tests/CMakeLists.txt` (new OBJECT lib `test_blackwell_tc_cu` with `-arch=compute_120a -code=sm_120a,compute_121a,code=sm_121a`)

- [ ] **Step 1:** Probe kernel: one warp executes ONE mma on synthetic operands built per the fragment formulas above (host fills a[32]/b[16]/sf values per-lane via known lane mapping), D written to global. Test compares vs exact host reference of `(A*SFA)*(B*SFB)` summed over k=0..63 within rel tol 1e-3. This empirically pins register packing before any GEMM work.
- [ ] **Step 2:** Build fails or mismatches -> adjust packing until green (iterate here; this task absorbs the format risk).
- [ ] **Step 3:** Commit `test: sm_120a nvfp4 block-scaled mma probe`.

### Task 2: Activation quantization kernel

**Files:**
- Create: `libs/lsxcommon/src/qwen_tc_act_quant.cu` (+ declaration in a new `libs/lsxcommon/src/qwen_tc_gemm.h`, host-visible)
- Object lib `lsxcommon_qwen_tc_cu` in `libs/lsxcommon/CMakeLists.txt` (arch-a targets only; linked PUBLIC into lsxcommon like siblings)

- [ ] **Step 1:** `QuantizeActsNvfp4(stream, const __half* x, int seq, int K, uint8_t* d_packed, uint8_t* d_scales)`: per-16-element absmax -> E4M3 (reuse encode math from `nvfp4_codec.h`), nibble-pack. Grid covers seq*K/16 blocks.
- [ ] **Step 2:** Unit test vs host reference (round-trip decode == table values, scales match absmax formula) in `tests/test_blackwell_tc_mma.cu`.
- [ ] **Step 3:** Commit `feat: nvfp4 activation quantization kernel (w4a4 fast path)`.

### Task 3: Tensor-core tile GEMM

**Files:**
- Create: `libs/lsxcommon/src/qwen_tc_gemm.cu`: `RunQwenTcNvfp4Gemm(stream, x_fp16, w_packed, w_scales, w_global_scale, y, seq, K, N)`
- Test: extend `tests/test_blackwell_tc_mma.cu`

- [ ] **Step 1:** Kernel: block computes 64x64 output tile; 8 warps; K loop in 64-wide steps; A fragments loaded from smem per mma fragment layout (`ldmatrix` optional first pass — plain loads acceptable v1); B fragments straight from the existing packed weight buffer (already K-major col-major). SFA from activation scales, SFB from weight scales, global_scale folded into epilogue (multiply D by gsA*gsB once per tile — NOTE: repo stores ONE global_scale per weight; activation side has none -> fold as scalar multiply).
- [ ] **Step 2:** Tolerance test vs FP32-dequant host reference (rel tol ~2e-2, consistent with double-NVFP4 rounding), shapes incl. K%64!=0 and N%8!=0 tails.
- [ ] **Step 3:** Micro-benchmark same four shapes as `test_blackwell_micro_gemm`; record us/call in registry F-01.
- [ ] **Step 4:** Commit `feat: sm_120a tensor-core nvfp4 gemm kernel`.

### Task 4: Runtime flag plumbing

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model.h` (`ModelInitConfig` + `bool fast_math = false`)
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (`RunQwenNvfp4Gemm` dispatches to TC kernel when enabled), `qwen_v5_model.cpp` / factory plumbing
- Modify: `apps/lsxfabric/src/*.cpp`, `apps/taskmosaic/src/*.cpp` (`--fast-math` CLI -> config)

- [ ] **Step 1:** Thread the flag; default OFF preserves bit-exact behavior byte-for-byte.
- [ ] **Step 2:** Tests: default-path suites all green unchanged; fast-path run-to-run determinism (3 runs memcmp); SelfCheck "Paris" top-1 passes with fast_math=true; perf harness with flag reports tok/s.
- [ ] **Step 3:** Commit `feat: --fast-math runtime switch to tensor-core gemm path`.

### Task 5: Wrap-up

- [ ] Registry F-01 update with final numbers (tok/s before/after on qwen harness).
- [ ] Full suite green; merge per finishing-a-development-branch.

## Risks

- Fragment-packing mistakes -> absorbed by Task 1 probe (single instruction, empirical).
- sm_120a code cannot load on non-Blackwell-consumer GPUs — irrelevant here (Blackwell-only project policy).
- W4A4 numerics differ from W4A16 scalar path by design; validated by tolerance + "Paris" + determinism, never by bit-exactness.
