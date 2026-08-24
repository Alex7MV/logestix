# Blackwell CUDA Audit — Findings Registry

Companion to `2026-08-23-blackwell-cuda-audit-design.md`. Every finding gets a
severity (P0 blocker / P1 major / P2 minor), file:line evidence, remediation
status, and before/after measurements when available.

| ID | Sev | Dimension | Evidence | Summary | Status |
|---|---|---|---|---|---|
| F-01 | P0 | A | glm_inference.cu:84; qwen_inference.cu:388-475 | Scalar GEMM kernels on CUDA cores; no tensor-core MMA usage anywhere. MEASURED (micro-bench `test_blackwell_micro_gemm`): NVFP4 tile GEMM ~17-19 GB/s effective vs 448 GB/s roofline (~25x gap) on sm_120; ptxas: 32 regs / 2112 B smem / no spills -> latency-bound single dependent-FMA chain, NOT resource-bound. Root cause of missing "tens of tok/s" | open -> Phase 3 (native mxf4nvf4 tensor-core MMA) |
| F-02 | P2 | B/C | qwen_inference.cu:1996 | Final-logits publish uses blocking null-stream cudaMemcpy inside timed region | fixed (d4c26e7..): stream-ordered cudaMemcpyAsync; qwen 22/22 green; ttft 6931->6457ms |
| F-03 | P1 | C | glm_inference.cu:1649-1653 | Per-token H2D roundtrip of sampled id + full cudaStreamSynchronize | fixed: device-side embed feed from d_argmax->best + event-scoped fence; glm 19/19 green (реальные веса GLM отсутствуют на стенде — прогон без real-model тестов) |
| F-04 | P1 | C | deepseek_inference_pipeline.cu:772 | cudaDeviceSynchronize inside pipeline | fixed: stream-scoped fence (pipeline owns a single stream); deepseek 8/8 green; ttft 37187->26562ms (-29%), total 81292->59059ms (-27%), tok/s 0.2->0.3 |
| F-05 | P2 | A | various launch sites | Degenerate launch configs (<<<1,...>>>, flat 256-thread blocks); no occupancy tuning vs 48 warps/SM | partially fixed: tile GEMM reworked (shared E2M1 table, VPT=2 dual accumulator chains, register prefetch of next chunk) — kernel 517->459 us (~10%), bit-exact (memcmp test). Remaining configs are latency-bound small kernels; scalar path exhausted -> F-01/Phase 3 |
| F-06 | TBD | E | libs/lsxcommon/CMakeLists.txt | -fmad=false applied to whole lsxcommon_glm_cu object library | open |

## Baseline measurements (Phase 0)

Machine: RTX 5060 Ti 16GB, Windows 11, Debug build (`windows-debug` preset),
prompt "The capital of France is", `LSX_TEST_PERF_GEN_TOKENS=16` default.
GLM-5.2 SKIP: checkpoint not present on this machine (vault has
DeepSeek-V4-Flash + Qwen3.5-9B only).

| Model | tok/s | ttft_ms | total_ms | gen_tokens | resident_mb |
|---|---|---|---|---|---|
| qwen3.5-9b | 0.6 | 6931.4 | 26458.4 | 16 | 8755 |
| deepseek-v4-flash | 0.2 | 37187.0 | 81292.1 | 17 | 11694 |
| glm-5.2 | n/a | n/a | n/a | n/a | n/a |

Notable pre-fix observations:
- DeepSeek ttft (37s) dwarfs decode — prefill path dominates; F-04 sync sits at
  the end of that pipeline.
- Qwen ttft ~7s for a 6-token prompt also indicates prefill-side overhead, not
  just first-token latency.

## Post-fix measurements (Phase 1 wrap-up)

Same machine/build/prompt. Debug build — treat deltas as directional, not
absolute.

| Model | tok/s | ttft_ms | total_ms | Delta vs baseline |
|---|---|---|---|---|
| qwen3.5-9b | 0.6 | 6457.2 | 25997.0 | ttft -6.8% |
| deepseek-v4-flash | 0.3 | 26561.9 | 59059.0 | ttft -28.6%, total -27.4% |

Full suite after fixes: `ctest -R "glm|deepseek|qwen|blackwell"` -> 49/49
passed (7 SKIP by design: GLM real-model tests without weights,
test_qwen_kv_cache).

## New findings discovered during execution

| ID | Sev | Dimension | Evidence | Summary | Status |
|---|---|---|---|---|---|
| F-07 | P1 | E | tests/test_schema_views.cpp:20 | Dead `#include <unistd.h>` broke MSVC build of the whole test tree (committed from Linux; symbol unused) | fixed (d4c26e7) |

### Compiler flags audit (2026-08-23)

- `-G` (device debug, kills ptxas optimizations): ABSENT from all configs
  (CMakeCache + vcxproj verified). Debug builds keep device code at ptxas -O3;
  only host code is unoptimized.
- `--expt-relaxed-constexpr` ADDED to lsxcommon_cu / lsxcommon_glm_cu /
  lsxcommon_qwen_cu / test_qwen_cu: without it, calling a host-side constexpr
  function from device code outside constant-evaluation context is formal UB
  (CUDA 13.3 docs); belt-and-braces alongside --extended-lambda.
- `LSX_NVCC_RES_USAGE=ON` CMake option added: passes nvcc `-res-usage`
  (registers/smem per kernel). Diagnostic-only, zero codegen effect.
  Profiled: qwen_nvfp4_gemm_tile_kernel<16,16,32> = 32 regs / 2112 B smem /
  1 barrier; no spills in lsxcommon_qwen_cu.
- Bit-exactness re-verified after flag changes: test_qwen_nvfp4_tile_gemm
  green (incl. memcmp bit-exact case); perf unchanged (24647 ms total).

### Phase 2 kernel experiments (2026-08-23, all bit-exact unless noted)

| Experiment | Result | Verdict |
|---|---|---|
| seq==1 tile specialization <1,256,32> | total 25997->27760 ms (+7%) — fewer blocks (N/256 vs N/16) starve SMs | reverted |
| __half2-vectorized qwen_add_kernel | A/B identical (26610/25787 vs 25794/25798) — add not a bottleneck | reverted (YAGNI) |
| shared-mem E2M1 table (vs __constant__ 16-way serialization) | ~5% (517->484 us) | kept |
| VPT=2 dual accumulator chains (ILP x2) | neutral (484 vs 484 us) — FMA latency not the limiter | kept (harmless, helps larger seq) |
| register prefetch of next chunk bytes | ~5-10% (484->459 us qkv; mlp_down 1216->1069 us) | kept |

Conclusion: scalar NVFP4 path plateaus at ~20 GB/s effective; dequant
arithmetic (~30 instr/element, 2x the MAC work) is the floor. Reaching tens of
tok/s requires Phase 3: native `mma.sync.kind::mxf4nvf4.block_scale` tensor
cores (hardware dequant+MAC fusion).

### Phase 3 Task 1 status (2026-08-23, branch feat/blackwell-phase3-tensorcore)

**SOLVED.** `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` fully verified on RTX 5060 Ti
(`--generate-code=arch=compute_120a,code=sm_120a`; plain sm_120 does NOT
assemble this). Empirical semantics (test_blackwell_tc_mma, all stages green):

- e2m1 operands: standard table {0,.5,1,1.5,2,3,4,6}; nibble pairs low-first;
  A reg r holds elements 8r..8r+7; rows alternate every 8 elements
  (g / g+8), cols tig*8+(i&7) then +32.
- ue4m3 scale decode: value = 2^(E-7) — BIT-IDENTICAL to the repo's existing
  `E4M3ScaleToFloat` ((e4+120)<<23). Zero exponent byte (E=0) -> 0.
- SFA metadata: lane tig==0 carries row g's chunk scales in bytes 0..3,
  lane tig==1 carries row g+8's; lanes tig>=2 are IGNORED by hardware.
- SFB metadata: byte c = chunk c for column lane>>2 (all lanes read).
- Selectors {byte-id,thread-id} = {0,0} as documented.

Earlier "anomalies" (half-exponent response, 1/16 factor) were a calibration-
harness bug: hardcoded 0x18181818 scales instead of kScale1.

Next steps (per docs/superpowers/plans/2026-08-23-blackwell-audit-phase3.md):
Task 2 activation quantization kernel; Task 3 TC tile GEMM; Task 4 --fast-math
plumbing.

### Phase 3 Tasks 1-3 status (2026-08-24, same branch)

- Task 2 DONE: `QuantizeActsNvfp4` (qwen_tc_gemm.cu) — per-16 pow2-E4M3 scale
  + E2M1 nibble pack, repo-codec-compatible. Round-trip test green (err <=
  1/3 block absmax, the NVFP4-pow2 theoretical bound).
- Task 3 CORE DONE: `RunQwenTcNvfp4Gemm` — 64x64 tile, 8 warps 4x2, four
  m16n8k64 mxf4nvf4 atoms per warp per k-chunk; weights consumed in the
  EXISTING [N,K] packed layout (no repack); activations quantized on the fly
  into stream-ordered cudaMallocAsync scratch; epilogue x global_scale.
  Verified vs exact host reference at seq=3/K=544/N=36 incl. K%64 and N%8
  tails: max_abs_err 0.234 @ ref 1085 (0.02%).
  MEASURED (Debug, seq=1): attn_qkv_o 466->291 us (1.60x), mlp_gate_up
  1142->784 us (1.46x), mlp_down 1351->713 us (1.90x).
- Hardware ue4m3 nuance discovered: e4==0 with M!=0 decodes as SUBNORMAL on
  sm_120a while the repo decoder says 0. Production pow2 scales never hit
  this; test data now uses repo-valid bytes only.

Remaining: fragment smem-staging + vectorized loads (headroom to several more x),
Task 4 --fast-math runtime plumbing, end-to-end tok/s measurement.
