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

### Phase 3 Task 4 + decode-path breakthrough (2026-08-24)

- `--fast-math` plumbed: ModelInitConfig::fast_math -> SetQwenTcFastMath
  toggle -> RunQwenNvfp4Gemm dispatch (TC for supported shapes, scalar
  fallback otherwise). Default OFF = bit-exact.
- **Dominant bottleneck found & fixed (F-05 extreme case)**: ncu (GPU perf
  counters enabled) showed fused_linear_qkv_kernel at 29.6 ms/invocation and
  fused_linear_conv_zab_kernel at 16.6 ms with grids (1,1,1)x128 and
  (2,1,1)x128 — the whole GatedDeltaNet QKV projection on a single block.
  Replaced on the seq==1 path with warp-per-output kernels
  (warpdot_nvfp4_kernel, ~1024 blocks, coalesced nibble reads, shfl reduce)
  plus split conv/norm and gating launches.
- END-TO-END (RTX 5060 Ti, Debug, 16 tokens): scalar 0.6 -> 4.2 tok/s
  (x6.6); fast-math 5.4 tok/s; ttft 6226 -> 729 ms. qwen suite 26/26 green.
- Phase breakdown WITH instrumentation overhead (fast-math): MLP 2730 ms,
  linear_attn 831 ms, full_attn 437 ms, norm 49 ms over 20 steps.
- Hardware nuance: sm_120a ue4m3 e4==0 & M!=0 decodes as subnormal; repo
  decoder says 0. Production pow2 scales never hit this.

Next: TC GEMM optimization (smem staging/vectorized loads), MLP gate+up fuse,
attention tuning -> target tens of tok/s.

### Phase 3 optimization round 1 (2026-08-24)

- TC GEMM v2: shared-memory staging of raw packed K64 slabs (A 64x32B,
  B 64x32B + scale slabs); fragments built from smem; global traffic purely
  streaming. Micro: attn_qkv_o 466->144 us, mlp_gate_up 1142->267 us
  (3.8x), mlp_down 1351->313 us (3.6x) vs scalar.
- Event-pair phase instrumentation (LSX_PERF_PHASES=1, no pipeline syncs,
  capped at first 8 steps). Decode-step GPU budget @ fast-math:
  attn ~23 ms/step (32 layers), mlp+norm ~66 ms/step.
- END-TO-END progression (Qwen3.5-9B @ RTX 5060 Ti, Debug, 16 tokens):
  audit start 0.6 tok/s / ttft 6226 ms -> now **8.3 tok/s** / ttft 474 ms /
  total 1925 ms. Full suite 51/51 green.

Remaining levers (ranked): act-quant launch overhead inside MLP (~226 launches
per run), gate+up GEMM fusion (halve MLP atom count), attention warpdot
tuning, prefill path (still scalar tile kernel at seq>1).

### Optimization round 2 (2026-08-24)

- uint4-vectorized slab staging in TcGemmKernel (K%64 fast path, scalar
  fallback for tails): attn_qkv_o 144->112 us (4.35x vs scalar),
  mlp_gate_up 267->236 us (5.0x), mlp_down 313->243 us (5.0x).
- End-to-end unchanged at ~8.2 tok/s: the micro-level GEMM gain is absorbed
  elsewhere — measured mlp_norm phase (~2.06 ms/layer) exceeds the sum of
  its TC GEMMs (~0.75 ms/layer @ I=12288, H=4096, 25 MB weights each).
  Gap = act-quant launches + swiglu + unaccounted per-layer overhead;
  ncu GPU perf counters now ENABLED on this machine for precise attribution.
- MLP decode is weight-bandwidth bound: 75 MB weights/layer/token; achieved
  effective BW in TC path still far below the 448 GB/s roofline -> further
  staging/pipeline work has large headroom.

Session totals (Qwen3.5-9B @ RTX 5060 Ti, Debug, 16 tokens):
audit start 0.6 tok/s / ttft 6226 ms -> **8.2 tok/s** / ttft 491 ms.
Suite state: qwen+blackwell 26/26 green.

### Optimization round 3 + main sync (2026-08-24)

- TC path extended to accumulate-in GEMMs (mlp down-proj acc=true was silently
  running the SCALAR tile kernel): **14.8 tok/s end-to-end**, ttft 262 ms
  (prefill GEMMs now also route through tensor cores).
- Merged origin/main (gemma4 + kimi models). Synced qwen warp-dot kernels to
  the renamed codec (kE2M1Table -> E2M1Value). Full rebuild green.
- Pre-existing Windows/MSVC defects inherited from main (NOT caused by this
  branch; filed here for the mainline owners):

| ID | Sev | Evidence | Summary |
|---|---|---|---|
| F-08 | P2 | test_schema_views (3/78 asserts fail) | Shared-memory schema views fail on MSVC |
| F-09 | P2 | test_gemma_prompt_framing | Path concat missing separator: `Temp\logestix_gpf_C:\Users\...` |
| F-10 | P1 | test_kimi_attention SegFault | New kimi kernel crashes at runtime on sm_120 |

Full suite: 95/98 (3 failed = F-08..F-10; 7 SKIP by design).

### Cumulative session result

**Qwen3.5-9B decode @ RTX 5060 Ti: 0.6 -> 14.8 tok/s (x24.7), ttft 6226 ->
262 ms.** Default scalar path bit-exact and untouched; fast-math behind
ModelInitConfig::fast_math / LSX_TEST_FAST_MATH=1.

### Optimization round 4 (2026-08-24) — WIP state at session end

- attn qkv/z/a/b routed through the TC GEMM on the fast path (warpdot kept as
  fallback): attn phase 182 -> 154 ms / 8 steps; e2e 15.4 tok/s.
- cp.async double-buffered decode-specialized kernel (TcGemmDecodeKernel)
  written but DISABLED (#if 0 in qwen_tc_gemm.cu): compute-sanitizer reports
  `Invalid __shared__ read` (constant addresses ~0x1f81, warps 7+, only at
  K=4096/9728, not at K=64). All apparent indices are bounds-checked; needs
  SASS-level investigation next session. v2 tile kernel handles all shapes
  correctly meanwhile.
- Target math for 70-80 tok/s: 5.04 GB weights / 448 GB/s = 11.2 ms/token
  ceiling (~89 tok/s); requires ~380 GB/s effective weight streaming
  (currently ~100 GB/s in TC path, ~21% roofline).

Next session queue: (1) debug decode kernel OOB via SASS/-lineinfo,
(2) gate+up weight concatenation at load -> single GEMM + single act-quant,
(3) swiglu fused into TC epilogue, (4) prefill TC path tuning (ttft 262 ms).

### Optimization round 5 (2026-08-24) — decode kernel fixed & enabled

ROOT CAUSE of the OOB found (lineinfo + sanitizer): the decode kernel used
`wn = warp & 1` (2 col-groups x 32 = 64 cols) while kDecCols was later raised
to 128 with `wn = warp >> 1` (4 col-groups); warps 4-7 indexed
sW[buf][64..151][...] past the slab. Fixed by kDecCols=128 matching the
8-warp/4-col-group geometry.

- Decode kernel re-enabled with DYNAMIC shared memory (sW|sWs|sXp|sXsc sized
  at launch) — removes the K/2<=2048 static limit; any K%64==0 supported.
- Micro: attn_qkv_o 84 us (5.8x), mlp_gate_up 136 us (9.4x), mlp_down
  172 us (7.5x) vs scalar. Sanitizer: 0 errors.
- END-TO-END: **17.0-17.4 tok/s** (ttft ~228 ms), qwen+blackwell 26/26 green.
- A/B noise lesson: single-run deltas on this box are +-15%; use >=3 runs.

Cumulative: **0.6 -> ~17 tok/s (x28)**.

Path to 70-80 (next session, ranked): multi-stage cp.async pipeline depth
(currently 1 chunk in flight; target 3-4), gate+up concat at load, swiglu-in-
epilogue fusion, prefill TC tuning, Release-build measurement (Debug host
code adds per-launch overhead).

### Optimization round 6 (2026-08-24)

- cp.async ring deepened to kDecBufs=4 (wait_group<2> before compute).
- Fragment build rewritten: layout identity means each 32-bit fragment
  register = ONE aligned smem u32 load (packed nibble pairs land exactly in
  hardware bit positions) -- zero shift/mask work. Dead byte-gather code
  removed.
- Micro: attn_qkv_o 72 us (6.5x), mlp_gate_up 106 us (9.4x), mlp_down
  171 us (6.6x) vs scalar.
- END-TO-END: **18.3 tok/s** / ttft 214 ms / total 876 ms. 26/26 green.

### Cumulative

**Qwen3.5-9B decode @ RTX 5060 Ti: 0.6 -> 18.3 tok/s (x30.5), ttft 6226 ->
214 ms** (fast-math path; bit-exact scalar default untouched).

Path to 70-80 tok/s remains bandwidth-bound engineering: effective weight
streaming must rise from ~150 GB/s to ~380 GB/s (85% roofline). Required
(next sessions): TMA bulk copies or wider multi-chunk staging, swiglu-in-
epilogue fusion, gate+up concatenation at load time, Release-build profiling
of the full per-token launch graph.

### Phase A — CUDA kernel unification (2026-08-24, branch feat/blackwell-phase4)

Shared header-template library (all byte-equal per cluster equivalence tests,
`tests/test_lsx_kernels.cu`):

- `lsx_kernels_quant.cuh`: Nvfp4DequantKernel<OutT>, F8E4m3ToFloat,
  E8m0ToFloat/E8m0ClampedToFloat, QuantizeE2m1TieUp/Nearest. Removed 5
  elementwise dequant copies, 3 F8/E8M0 decoder twins, benchmark legacy
  kernel + dead deepseek externs.
- `lsx_kernels_common.cuh`: CastIn/CastOut, ExpFlavor (Fast/Precise/Glibc).
  Shared math uses explicit *_rn intrinsics -> results independent of the
  historical -fmad object-lib flags (anomaly resolved flag-independently).
- `lsx_kernels_math.cuh`: AddInPlace/AddOut/EmbedGather/F32toT — replaced
  ~12 per-model copies across qwen/qwen_moe/glm/kimi.
- `lsx_kernels_act.cuh`: SwigluKernel (division form) + SwigluSigKernel
  (sigmoid form) x exp flavor x clamp — replaced 7 copies. Kept
  family-local: qwen dense silu_device form, deepseek asymmetric silu_mul.
- `lsx_kernels_norm.cuh`: RmsNormTreeKernel<T,NormWeight> — replaced qwen
  offset + gemma direct/plain. GLM/Kimi sequential-scan bit-exact kernels
  and DeepSeek F32 stay family-local.

Documented deviations: router top-k kernels NOT unified (qwen_moe vs gemma
differ in probs rounding path/dtype/output format — 2 sites only); glm
load-time vectorized dequant kept (perf-specialized).

Full suite 104/104 green, zero expectation changes.

### Optimization round 7 (2026-08-24, feat/blackwell-phase4)

- Phase A dedup: shared `lsx_kernels_{common,quant,math,act,norm}.cuh`;
  ~25 kernel copies removed, byte-equal per cluster (see Phase A summary).
- B1 gate+up fusion moved to the CONVERTER: joint [2I,K] quantization with a
  single global scale (`mlp.gate_up_proj.weight`) — loader derives gate/up
  as non-owning views; pre-fusion artifacts load unchanged. Dense fast path:
  one GEMM + one act quant per MLP.
- Decode-kernel K-cap removed via dynamic shared memory for activations:
  mlp_down (K=12288) left the tile-kernel fallback (~183 us) for the decode
  kernel (~55 us). This was the single largest e2e win of the round.
- END-TO-END @ RTX 5090 (Release, qwen3.5-9b): fast-math **51.1 tok/s** /
  ttft 83 ms (42.3 before this round); scalar default 14.4 tok/s untouched.
  NOTE: LSX_TEST_FAST_MATH is presence-checked — setting it to "0" ENABLES
  fast-math; unset the variable for scalar A/B.

### TMA integration status (blocked, next session)

Single-block probe and a minimal multi-block repro (exact decode geometry,
box {32B x 128 rows}, coords (kb/2, col_base)) both PASS on HW + sanitizer.
Integrating the same staging INTO TcGemmDecodeKernel faults (HW: illegal
address 700; sanitizer: out-of-range shared / NONCGA_CTA_ID emulation
asserts with .shared::cluster form). Differences vs repro: dynamic-smem
activation section coexistence, larger K grid of chunks in flight. Needs a
dedicated session: ncu/SASS walk of the faulting launch, try
cudaFuncAttributeMaxDynamicSharedMemorySize interplay, and mbarrier phase
accounting per buffer. cp.async ring retained in production meanwhile.

Path to 280-320 tok/s @5090 (next sessions, ranked): (1) resolve TMA
staging in decode kernel -> effective streaming toward roofline,
(2) swiglu-in-epilogue (paired-col variant), (3) linear-attn stage fusion
(conv_norm + gated_delta = 20% of GPU time), (4) act-quant caching across
GEMM calls.

### Round 7 addendum — TMA RESOLVED (2026-08-25)

Root cause of the round-7 integration fault: the cp.async.bulk.tensor
shared-memory DESTINATION must be 128-byte aligned (16 is insufficient).
Isolated empirically by walking a working multi-block repro toward the
kernel in steps (dyn+static coexistence -> dst-in-dynamic at 16B = fault ->
128B-rounded = pass). TMA staging now live in TcGemmDecodeKernel: static
128B-aligned ring + per-buffer mbarrier parity waits (.shared::cta form),
host-side tensormap cache keyed by (ptr,K,N). e2e @5090: 51.1 -> 53.1 tok/s.

### Round 8 profile post-TMA (2026-08-25)

TcGemmDecodeKernel = 78.7% GPU time (4980 launches, avg 45.7us / med 32.9).
Effective weight streaming ~245 GB/s = 13.7% of 5090 HBM peak -> the decode
path is LATENCY/LAUNCH-bound, not bandwidth-bound: ~310 kernel launches per
token. Reaching 280-320 tok/s @5090 requires structural work, not staging
tuning: persistent megakernel or CUDA Graphs for the per-token launch graph,
wider output tiles (kDecCols=128 -> 256), fusing linear-attn stages
(conv_norm + gated_delta ~= 10% GPU), act-quant reuse across GEMM calls.
Extrapolated 5060 Ti at current efficiency: ~14 tok/s-equivalent BW share;
the 70-80 target rides on the same structural fixes.

### Round 8 CORRECTNESS INVESTIGATION (2026-08-25) — fast-math was broken

Systematic debugging (triggered by a new acc-contract test) exposed that the
fast-math path produced GARBAGE TOKENS end-to-end on the real model despite
green perf harnesses (they never checked output text). Findings:

1. TcGemmDecodeKernel fails host-reference validation at ALL shapes
   (test "[tc][gemm][acc]" now locks this; LSX_DEC_ENABLE=1 debug-only).
   Production routes seq==1 to the verified tile kernel.
2. TILE kernel acc=true multiplied the SEEDED RESIDUAL by wgs
   ((prev+sum)*wgs instead of prev+sum*wgs) — fixed at the D-seed site.
3. With both fixes, fast-math e2e @5090: **32.9 tok/s, output " Paris"**
   (correct!). Previous 42-53 tok/s figures were garbage-generation rates.
4. Perf-harness lesson: ALWAYS decode and print gen_tokens in throughput
   tests (now permanent in test_blackwell_perf_qwen).
5. LSX_TEST_FAST_MATH presence-check trap documented ("=0" enables).

Remaining gap to targets is now measured against a CORRECT baseline:
32.9 -> 70-80 (5060 Ti) / 280-320 (5090). Next levers (ranked): wider decode
tiles / more blocks (N=4096 -> only 32 CTAs on 170 SMs), linear-attn stage
fusion (~10% GPU), act-quant reuse across GEMM calls, CUDA Graphs.

### Round 8 final state (2026-08-25)

Shipped fixes: decode-kernel A-fragment register map corrected (K-halves were
fed into row-g+8 slots), epilogue row-0 guard (g==0) added, tile acc=true
residual scaling fixed, production dispatch routes seq==1 to the verified
tile kernel (decode kernel behind LSX_DEC_ENABLE for debugging).
Fast-math e2e @5090 real model: **32.9 tok/s with correct " Paris" output**
(vs 14.4 scalar; earlier 42-53 tok/s figures were garbage-generation).

OPEN ISSUE (next session, top priority): synthetic host-model validation of
TC kernels at seq==1 with K>=256 deviates beyond tolerance IDENTICALLY for
both kernels (e.g. got=198 vs want=23 at K=256/N=128), while K=64 passes and
the REAL model validates end-to-end. Independent Python NVFP4 model confirms
the C++ host reference is correct -> the deviation is in the kernels' shared
input path (act-quant scratch or weight-upload conventions under synthetic
generators). Requires a dedicated session with compute-sanitizer + per-stage
dumps; do NOT raise tolerances to mask it.

Structural perf queue (unchanged): wider tiles / more CTAs (N=4096 -> 32
blocks on 170 SMs), linear-attn fusion (~10% GPU), act-quant reuse, CUDA
Graphs. MoE models (35B-A3B, gemma-26B) suggested by operator as additional
measurement targets once fast-path correctness work concludes.

### Round 8 RESOLUTION (2026-08-25, later) — decode kernel FIXED

Three root causes found and fixed in TcGemmDecodeKernel (each isolated via
per-chunk d-trajectory dumps vs an independent Python NVFP4 model):
1. A-fragment register map: K-halves fed into row-g+8 slots ({lo,hi,0,0}
   instead of {row-g-lo, row8-lo, row-g-hi, row8-hi}).
2. Epilogue raced four lanes per output cell (m16n8 D layout: only g==0
   lanes hold row 0); added the missing guard.
3. A-fragment k-offset lost the per-chunk base (sXp[koff>>1] instead of
   sXp[(kb+koff)>>1]) -> every chunk consumed chunk-0 activations.
Plus the earlier tile acc=true residual-scaling fix.

Post-fix validation: host-ref sweep {1,64..4096} all PASS with errors
<=0.37 (tolerance ~12+), identical to the tile kernel; act quant proven
byte-exact at seq1/K256 ("[tc][actquant][k256]").

ROUTING DECISION (measured): despite decode winning isolated micro-benchmarks,
the TILE kernel is faster end-to-end (32.9 vs 31.1 tok/s @5090, correct
" Paris" both) -- confirming the Phase-3 A/B conclusion. Production default
= tile; LSX_DEC_ENABLE=1 opts into the now-correct decode kernel.

Fast-math honest baseline: **32.9 tok/s** (x2.3 vs scalar 14.4). Perf queue
unchanged: wider tiles/more CTAs, linear-attn fusion, act-quant reuse,
CUDA Graphs; MoE models as additional targets.

### Optimization round 9 (2026-08-25) — tile double-buffering

TcGemmKernel k-loop pipelined: slabs for chunk i+1 stream via cp.async
(double-buffered smem, wait<1>/wait<0> tail) while chunk i computes.
Also fixed a PRE-EXISTING tail-K OOB in the scalar staging path (reads ran
past the row end when K%64 != 0; sanitizer now reports 0 errors on
[tc][gemm], was 43).

Micro @5090: mlp_down 109 -> 99 us, gate_up 44.6 us. e2e: 33.0 -> **34.6
tok/s** (+5%), ttft 126 -> 121 ms. Correctness unchanged (" Paris").

Cumulative honest fast-math: 14.4 (scalar) -> 34.6 tok/s (x2.4).

### Optimization round 10 (2026-08-25) — split-K

TcGemmKernel gained deterministic split-K: when seq==1 underfills the SM
array (grid.x < 96), K is spread over blockIdx.z slices writing disjoint
fp32 partials (no atomics); TcSplitKFinalize sums slices and applies wgs
(+acc). Tail-K chunk-bound bug fixed en route ((k_hi+63)/64).

Micro @5090: mlp_down 99 -> 42 us, attn_qkv_o 45 -> 23 us. Sanitizer: 0.

Quality note: splitting the attention projections (qkv/o, K=4096) shifted
the greedy top-1 away from " Paris" on the canonical prompt (still fluent);
split is therefore restricted to K>=8192 (mlp_down class) where quality is
bit-stable. e2e @5090: **33.3 tok/s** with correct output; the aggressive
variant reached 37.8 and is worth revisiting once fast-path numerics are
quality-audited.

### Optimization round 11 (2026-08-25) — linear-attn fusion: closed as low-value

Fast-math profile after r10 (LSX_TEST_FAST_MATH=1): TcGemmKernel 86.6%,
fused_linear_conv_norm_kernel 3.3% (480x32us, single block),
qwen_gated_delta_kernel 2.8% (bandwidth-bound: 16MB state r/w per call).

Implemented column/head-parallel split of conv+l2norm (bit-exact tree
order): correct, all tests green, but e2e UNCHANGED (33.2 vs 33.3 tok/s,
x3 runs each). The decode segment is launch-bound, not compute-bound;
splitting one launch into two eats the parallelism gain. REVERTED.

Conclusions:
- remaining non-GEMM kernels total <14% and are individually small;
  kernel-by-kernel micro-fusion here is exhausted
- the path to targets is GEMM-wide tiles (more work per CTA at seq==1)
  and/or CUDA Graphs for launch overhead
- NOTE: one anomalous run produced fluent-but-wrong text ("A. True B.
  False...") right after rebuild; subsequent x6 runs stable " Paris".
  Treat single-run text anomalies with suspicion; re-run before diagnosing.

### ROOT CAUSE (round 12, 2026-08-25): W4 weight streaming is latency-bound

New [tc][gemm][perf] l2cold variant (256 MB L2-flush between iterations):
attn_qkv_o 23 -> 171 us, gate_up 47 -> 174 us, down 42 -> 169 us (~x4,
size-INDEPENDENT). Micro-benchmarks were L2-resident (weights <= 19 MB vs
96 MB L2) and hid the real decode behavior: in e2e each layer's ~100 MB
working set evicts weights -> DRAM reads at ~110-250 GB/s effective.

Cause: B-staging reads 32 B per column with a K/2-byte stride (row-major
[N, K/2] packed layout); every cp.async lands on a different DRAM row.

Fix direction: transposed device layout [K/2][N] built once at load time
(nibble-preserving byte shuffle kernel); TcGemmKernel stages contiguous
runs along N. Expected: x3-7 GEMM improvement -> e2e toward ~80 tok/s.

### Optimization round 12 (2026-08-25) — real cp.async pipeline: +14% e2e

Investigated why e2e GEMM (avg 81us) ran ~4x slower than micro-benchmarks
of the same shapes. Built an L2-flush variant of [tc][gemm][perf]: cold runs
are 4x slower and size-independent -> suspected strided DRAM reads.

Tested the fix (transposed weight layout [K/2][N], byte-transpose is
nibble-safe because packed bytes already hold (k,k+1) pairs; lazy cache +
InvalidateTransposedCache hook): NO GAIN, hot even regressed. Root cause was
NOT bandwidth: TcGemmKernel spends ~4us/64-chunk in MMA and the rest in the
synchronous staging chain (plain stores + syncthreads per chunk). Reverted
the transpose machinery entirely.

Kept: true cp.async staging (cp.async.cg 16B with zero-fill predicate) for
A/B slabs in vec_ok path -- next chunk streams while current computes.
Micro @5090: qkv_o 49->39.7us, gate_up 47->39.5us, down 42->35.9us.
e2e @5090: **33.3 -> 38.1 tok/s (+14%), ttft 125 -> 110ms**, " Paris." OK,
104/104 tests green, sanitizer clean. Cumulative fast-math: 14.4 -> 38.1
tok/s (x2.65).

Lessons:
- micro-benchmarks without L2 pressure mislead; always A/B both regimes
- verify a bottleneck hypothesis by implementing its fix BEFORE building
  out infrastructure around it (transpose cost one session; cp.async won)

### Optimization round 13 (2026-08-25) — 4-deep slab pipeline: +28% e2e

Post-r12 profile: TcGemmKernel still 84.6% of GPU (avg 69us vs 36-40us
L2-hot micro). Cold-GEMM time is a per-chunk latency chain (64 chunks x
~2.5us), not bandwidth. Deepened the cp.async slab ring from 2 to 4 buffers
(wait_group 3/2/1/0 by position, numerically identical accumulation order).

Micro @5090 (hot): ~36us flat across shapes. e2e @5090: **38.1 -> 48.8
tok/s (+28%), ttft 110 -> 87ms**, " Paris." stable x4 runs. 105/105 tests,
sanitizer clean.

Cumulative fast-math: scalar 14.4 -> **48.8 tok/s (x3.4)**.
Remaining tail: conv_norm 3.8%, ActQuant 3.4%, gated_delta 2.6%,
RmsNormTree 2.7%; GEMM still ~80% and now latency-vs-bandwidth balanced --
next levers: wider split-K revisit, ActQuant fusion into staging.

### Optimization round 14 (2026-08-25) — split-K widened: 53.9 tok/s

kPipe=8 experiment: no change (cold latency is CTA-count bound, ~64 CTAs <
170 SMs, per-SM MSHR limits) -> reverted to 4. The real parallelism lever is
split-K: widened to K>=2048, grid.x<192 (covers attention qkv/o). The
r10 top-1 shift did NOT reproduce in the current numeric context (pipe-4
changed the rounding trajectory): " Paris." stable x4.

e2e @5090: **48.8 -> 53.9 tok/s (+10%), ttft 87 -> 79ms**. 105/105 tests,
sanitizer clean. Cumulative: scalar 14.4 -> **53.9 tok/s (x3.7)**.

### Optimization round 15 (2026-08-25) — fused qkv|z|a|b input GEMM: 59.7 tok/s

The four linear-attn decode projections share one input row but ran as 4
GEMMs (4 ActQuant passes, 4 launches). Added BuildFusedGemmWeights: cached
N-concatenation of the packed matrices + a device per-segment global-scale
vector; TcGemmKernel epilogue and split-K finalize select wgs by output
column (4 ranges). Numerically identical per column.

kTileN=32 experiment (2x CTAs): helped qkv_o micro (36->24us) but broke
split-K conditions for MLP shapes -> e2e regression, reverted.

e2e @5090: **53.9 -> 59.7 tok/s (+11%), ttft 79 -> 72ms**, " Paris." x2.
105/105 tests, sanitizer clean. Cumulative: scalar 14.4 -> **59.7 tok/s
(x4.1)**.

### Optimization round 16 (2026-08-25) — decode tail cleanup: 62.8 tok/s

- RmsNormTreeKernel: x cached in dynamic smem on pass 1, pass 2 reads smem
  (values identical -> accumulation order unchanged); launch sites size the
  extra smem (qwen/gemma/tests). Single-block rows stop re-reading global.
- conv/l2norm parallel split from r11 restored: with the GEMM pipeline now
  dense it converts to e2e (r11 was launch-bound when GEMM had gaps).

e2e @5090: **59.7 -> 62.8 tok/s (+5%), ttft 72 -> 69ms**, " Paris." x3.
105/105 tests, sanitizer clean. Cumulative: scalar 14.4 -> **62.8 tok/s
(x4.4)**.

### Optimization round 17 (2026-08-25) — gated-delta state coalescing: 64.2

qwen_gated_delta_kernel state layout transposed [col][kd] -> [kd][vflat]:
warp lanes now touch consecutive floats when streaming the 8 MB recurrent
state (was strided by kd per thread, ~600 GB/s effective). Internal
workspace only; numerically identical per thread.

e2e @5090: **62.8 -> 64.2 tok/s (+2%), ttft 69 -> 68ms**, " Paris." x3.
Cumulative: scalar 14.4 -> **64.2 tok/s (x4.5)**.

### Host-side pipeline audit + MoE handshake fixes (2026-08-25, branch feat/moe-router-async-handshake)

Second workstation (RTX 5060 Ti, Windows) audited the full inference
strategy for GPU utilization while the first machine continued TC GEMM
optimization. Full findings delivered to the operator; the non-conflicting
subset was implemented here (no overlap with qwen_tc_gemm.cu / fast-math
dispatch):

| ID | Sev | Fix | Commit |
|---|---|---|---|
| H-01 | P0 | DeepSeek route_token split into async begin/finish (event-scoped fence replaces full cudaStreamSynchronize); shared-expert FFN enqueued BETWEEN the phases so the GPU executes it during the host's router roundtrip — was a full drain x43 layers/token. Fused API kept as bit-exact wrapper | 254c931 |
| H-02 | P0 | Qwen3.5-MoE per-layer slot readback: acc-init kernel moved ahead of the wait; blocking cudaEventSynchronize -> busy-poll cudaEventQuery (wake latency x40 layers/token) | 254c931 |
| H-03 | P2 | test_schema_views kept mmaps alive across fs::remove_all -> Windows file lock (F-08 closed); gemma path-separator (F-09) already fixed on main | cb9674b |
| H-04 | P3 | deepseek/glm perf harnesses now decode+print gen text (round-8 garbage-generation lesson ported) | 05a860a |
| H-05 | P1 | DeepSeek persistent CUDA context via ModelBase::EnsureContext(true); RunInferenceModel accepts external stream/cuBLAS pair | 9bb436f |
| H-06 | P1 | Qwen3.5-MoE cache-miss H2D moved to a dedicated copy stream: residency resolved up front, per-expert event fences, ev_front guard prevents overwriting slots still read by enqueued compute; driver owns stream+events per generation | 2a4d049 |
| H-07 | P2 | LSX_QWEN_MOE_EXPERT_CACHE_SLOTS env override for VRAM sweeps (parity with DeepSeek knob) | 89df85d |

Numerics untouched everywhere (ordering of independent work only).
Verification on this box: 15/15 synthetic suites green (incl. two new TDD
tests: router split-phase bitwise equivalence score/bias/hash; copy-stream
path bit-identical to single-stream incl. LRU eviction) + real-model
selfchecks PASS with the new pipeline: qwen3.5-35b-a3b (3 cycles, " Paris",
177 s) and gemma-4-26b-a4b-it (87 s).

Deferred to real-HW sessions (blocked locally by weights availability /
Windows model-test policy): DeepSeek expert-upload copy stream (mmap eviction
barrier must cover both streams), GLM two-stream production wiring (existing
plumbing at glm_inference_load.cu:974-1089 is unreachable from
RunGlmRealGeneration), LSX_QWEN_MOE_EXPERT_CACHE_SLOTS sweep above the 64-slot
default, e2e tok/s deltas for the touched pipelines.

Still-open structural levers from this audit (unchanged by design, bit-exactness
contract): batched prefill (both models still token-at-a-time), single-block
kernels in hot loops, CUDA Graphs for the ~310-launch decode graph, request
batching in the orchestrator.


### MoE speed campaign (2026-08-26) — 8.3 -> 32.1 tok/s on Qwen3.5-35B-A3B

- TC fast-math toggle was never set for the MoE model (ModelBase-derived):
  every GEMM ran the scalar tile kernel. x2.55 from one line.
- router top-k dot loop vectorized (uint4, 8 accumulator chains): +10%.
- routed gate+up fused into ONE GEMM per expert: slot staging reordered to
  [w13=w1|w3][w2]; per-expert {gs1|gs3} device pairs; kernel epilogue
  half-split at N/2 when seg_bounds==nullptr. Scalar fallback keeps the
  split pair.
- shared expert gate+up fused via cached BuildFusedGemmWeights N-concat;
  BuildFusedGemmWeights validation relaxed to allow null pointers for
  zero-row segments (was rejecting the 2-segment form outright).
- e2e @5090: 8.3 -> **32.1 tok/s (x3.9)**, ttft 478 -> 135ms. " Paris."
  stable everywhere; 107/107 tests.

Next levers (profiled): ActQuant fusion for w2 inputs, batched expert
execution (8 sequential slot chains), CUDA Graphs (~45 launches/layer),
gemma TC route evaluation.

### MoE campaign addendum (2026-08-26): 37.4 tok/s

Two-stage router at decode: block-per-expert sequential dots (bit-identical)
+ separate HF-parity select kernel. Router GPU time 131 -> ~30 ms.
e2e @5090: 32.2 -> **37.4 tok/s (+16%)**, campaign cumulative
8.3 -> **37.4 (x4.5)**, ttft 478 -> 122 ms.

Remaining profile (GPU ~250ms / host-gaps ~170ms): TcGemmKernel ~99ms,
router dots ~30ms, ActQuant+swiglu_quant ~45ms, host launch gaps now the
single biggest lever (CUDA Graphs / batched experts).
