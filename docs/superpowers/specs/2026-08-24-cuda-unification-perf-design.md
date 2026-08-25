# CUDA Unification + Inference Acceleration — Design

Date: 2026-08-24
Branch: `feat/blackwell-phase4-cuda-unification`
Model: Qwen3.5-9B fast path. Baseline: 18.3 tok/s / ttft 214 ms @ RTX 5060 Ti (findings, round 6).

## Goals

1. **Phase A — unification**: eliminate duplicated CUDA kernels across all model
   object libraries via a shared header-template kernel library. Bit-exact
   contracts preserved per flavor.
2. **Phase B — acceleration**: gate+up concat at load, swiglu-in-epilogue,
   TMA bulk-copy staging, Release profiling.

## Targets

| GPU | Target | Validation |
|---|---|---|
| RTX 5090 (this box) | **280–320 tok/s** Qwen3.5-9B decode | direct e2e harness, >=3 runs |
| RTX 5060 Ti | **70–80 tok/s** | report effective weight-streaming GB/s + extrapolation; measured by operator on another machine |

Roofline math: 5090 ~1.79 TB/s -> 355 tok/s ceiling for 5.04 GB weights;
target = 79–90% of peak. 5060 Ti 448 GB/s -> 89 tok/s ceiling; target = 85%.

## Phase A — shared kernel library

### Files (header-only, NVCC-safe, in `libs/lsxcommon/src/`)

Pattern: existing `glm_expert_kernels.cuh` header-include precedent. Each model
`.cu` includes what it needs; no new link-time library; no CMake object-lib
surgery.

- `lsx_kernels_math.cuh` — block/warp reductions (`sum`/`max`), half<->float
  converts, clamp f32->f16.
- `lsx_kernels_norm.cuh` — RMSNorm `template<typename T, ExpFlavor F,
  bool HasOffset>`.
- `lsx_kernels_act.cuh` — SwiGLU `template<typename T, ExpFlavor F, bool
  Clamp>`; residual add/axpy; embed-gather; two-phase argmax (canonical form
  from `sampling.cu`).
- `lsx_kernels_quant.cuh` — NVFP4 elementwise dequant
  (`template<typename OutT>` over canonical `nvfp4_codec.h`); F8_E4M3/E8M0
  decoders; E2M1 encode (threshold-chain).
- `lsx_kernels_router.cuh` — softmax->top-k->renorm, thread-0 deterministic
  selection, `template<bool SigmoidBias, typename ScaleT>`.

### ExpFlavor policy

enum: `Fast` (`__expf`) / `Precise` (`expf`) / `Glibc` (glibc-replica from
`glm_device_math.h`). Bit-exact models (GLM/Kimi/DeepSeek) instantiate their
flavor explicitly; contracts preserved bit-for-bit.

### fmad-anomaly resolution (operator-approved inclusion)

Shared math kernels are written with explicit intrinsics
(`__fmaf_rn`/`__fmul_rn`/`__fadd_rn` or provably non-contractible forms) so
results are independent of the `-fmad` flag. Object-lib flags stay as-is
(`lsxcommon_qwen_cu`: FMA on; `lsxcommon_qwen_moe_cu`, `lsxcommon_glm_cu`,
`lsxcommon_kimi_cu`: `-fmad=false`) — zero numeric drift vs current baselines.
Policy documented in a CMakeLists comment.

### Migration clusters (each: red/green test first, then migrate, then delete)

Order: quant -> math/act -> norm -> router.

1. **Quant**: 5 hand-rolled NVFP4 dequant kernels (`shared_experts.cu:94,404`,
   `gemma_inference.cu:28`, `kimi_inference.cu:636`, `glm_inference_load.cu:50`,
   `benchmark_gpu.cu:34,45`) -> one dtype-templated kernel. F8/E8M0 verbatim
   triplicates (`safetensors_gpu.cu:63,82`, `deepseek_inference_attn.cu:30,49`,
   `shared_experts.cu:33,54`) -> single source. E2M1 encode twins
   (`qwen_tc_gemm.cu:38`, `safetensors_gpu.cu:96`).
2. **Math**: residual adds (x7), half/float converts (x4), clamps (x3),
   embed gathers (x4), argmax (x3 -> two-phase), block reductions (sum/max
   twins).
3. **Act**: SwiGLU variants (x7) parameterized by dtype/clamp/exp-flavor;
   resolves dense(`__expf`) vs MoE(`expf`) divergence via explicit flavors.
4. **Norm**: RMSNorm family (x6).
5. **Router**: qwen_moe/gemma near-identical top-k kernels; kimi sigmoid-bias
   variant via template param; GLM bit-exact variants keep `Glibc` flavor.
6. **Shim removal**: `benchmark_gpu.cu` extern-linkage dequant shims die with
   the dead externs in deepseek pipeline/moe.

Test rule: per cluster, new-vs-old equivalence first — memcmp for preserved
flavors, rel-tol 2e-2 where the cluster is tolerance-based anyway.

## Phase B — acceleration (on the unified base)

### B1. Gate+up concat at load

Loader merges `gate_proj`+`up_proj` into one `[2I, K]` packed buffer at upload
(dense MLP and MoE expert stacks). Row convention: `[0,I)=gate`, `[I,2I)=up`
(HF parity). Fast-path MLP: single GEMM -> `[seq, 2I]` -> swiglu -> down.
Removes one GEMM and redundant act-quant passes per layer.

### B2. SwiGLU-in-epilogue (decode seq==1)

In `TcGemmDecodeKernel` epilogue: warp holds gate columns `D[0..I)` and reads
paired up-columns `D[I..2I)` from the sibling col-group accumulator -> computes
`silu(g)*u` in FP32 before the global store. The `[1, 2I]` intermediate is never
materialized; standalone swiglu launch disappears from the decode path.
Tile path (prefill) stays split — fusion only when both halves live in one block.

### B3. TMA bulk-copy weight staging

Replace the hand-rolled `cp.async.cg` ring with `cp.async.bulk.tensor` +
`cuTensorMap` (sm_120a; 3D map over `[K/2 bytes x N-tile x buf]`), mbarrier
completion, pipeline depth 4-6 chunks. Goal: effective weight streaming toward
380 GB/s-equivalent roofline share. Runtime fallback to the cp.async ring on
`cudaErrorNotSupported`. Unit test of a single bulk-copy round trip BEFORE any
GEMM integration (absorbs descriptor/mbarrier geometry risk).

### B4. Release profiling

Build `linux-release-znver4`; nsys/ncu profile of the full per-token launch
graph; tune grid/block and register pressure from data; final numbers into
findings. Debug-host launch overhead removed from measurements.

## Verification protocol (TDD throughout)

- Phase A: equivalence tests per cluster before migration (see above); full
  suite green after each cluster; default scalar path must remain byte-exact
  with zero expectation changes.
- Phase B: rel-tol 2e-2 vs FP32-dequant reference; 3x memcmp run-to-run
  determinism on the fast path; SelfCheck "Paris" top-1 with fast_math=true;
  `compute-sanitizer --tool memcheck` clean on every new/changed kernel;
  micro-benchmarks + e2e tok/s (>=3 runs, +-15% single-run noise lesson).
- Findings doc updated after each optimization round.

## Out of scope (next cycles)

- Porting the fast path to other models (GLM/Kimi/DeepSeek/Gemma); their
  bit-exact GEMM paths untouched.
- tcgen05 / 5th-gen MMA rewrite (current `mma.sync.m16n8k64` retained).
- Converters, host loaders, network layer, GUI, multi-GPU, batched decode.
- Changing any `-fmad` flags (anomaly resolved flag-independently instead).

## Risks

- TMA descriptor/mbarrier geometry -> absorbed by the B3 single-copy probe.
- Epilogue fusion changes fast-path rounding -> acceptable under the tolerance
  contract; guarded by determinism + selfcheck.
- Dedup regressions -> caught by bit-exact per-cluster equivalence tests.
