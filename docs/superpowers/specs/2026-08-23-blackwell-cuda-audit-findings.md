# Blackwell CUDA Audit — Findings Registry

Companion to `2026-08-23-blackwell-cuda-audit-design.md`. Every finding gets a
severity (P0 blocker / P1 major / P2 minor), file:line evidence, remediation
status, and before/after measurements when available.

| ID | Sev | Dimension | Evidence | Summary | Status |
|---|---|---|---|---|---|
| F-01 | P0 | A | glm_inference.cu:84; qwen_inference.cu:388-475 | Scalar GEMM kernels on CUDA cores; no tensor-core MMA usage anywhere. MEASURED (micro-bench `test_blackwell_micro_gemm`): NVFP4 tile GEMM ~17-19 GB/s effective vs 448 GB/s roofline (~25x gap) on sm_120; ptxas: 32 regs / 2112 B smem / no spills -> latency-bound single dependent-FMA chain, NOT resource-bound. Root cause of missing "tens of tok/s" | open -> Phase 3 (native mxf4nvf4 tensor-core MMA) |
| F-02 | P2 | B/C | qwen_inference.cu:1996 | Final-logits publish uses blocking null-stream cudaMemcpy inside timed region | fixed (d4c26e7..): stream-ordered cudaMemcpyAsync; qwen 22/22 green; ttft 6931->6457ms |
| F-03 | P1 | C | glm_inference.cu:1649-1653 | Per-token H2D roundtrip of sampled id + full cudaStreamSynchronize | fixed: device-side embed feed from d_argmax->best + event-scoped fence; glm 19/19 green (real GLM weights are absent on the stand — run without real-model tests) |
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

### MoE campaign status (2026-08-26, end of session)

Qwen3.5-35B-A3B @5090: **37.4 tok/s** (x4.5 over the 8.3 baseline), ttft
122 ms. gemma-4-26B-A4B-it baseline: 2.5 tok/s (own dequant+cuBLAS path,
TC route not wired yet).

Next levers by profile: host launch gaps ~170 ms (CUDA Graphs / batched
expert execution), TcGemmKernel ~99 ms, gemma TC route. Input quant cache
landed (launch hygiene) but was e2e-neutral.

### gemma TC expert route experiment (2026-08-26): rejected

Per-expert TC W4A4 substitution (bf16->half token, quant once, three
PreQuant GEMMs, half geglu, convert back) measured 2.7 tok/s vs 3.0
baseline AND shifted greedy output away from "**Paris**." — the per-expert
granularity adds quant/convert launches that outweigh tensor-core gains at
N=704, and W4A4 numerics move gemma's top-1. Reverted.

If revisited: needs a grouped/batched expert GEMM (all top-K experts in one
launch over gathered rows) plus a quality audit; not a drop-in swap.

### Session addendum (2026-08-26): clean-HEAD measurement discipline

After heavy file checkouts the incremental ninja build silently served a
stale binary: "72 tok/s garbage" readings were an artifact. Full rebuild on
clean HEAD: **38.7 tok/s**, " Paris..." correct, 107/107 tests.
RULE: after any git checkout of sources, touch + full rebuild before
measuring; never trust perf numbers across binary/staleness boundaries.

Batched expert execution prototype was lost in the same checkout churn;
re-implement on this clean base next session (design is documented above:
grouped TC GEMM via grid.z groups + strides, gather kernel, flat-indexed
batched swiglu-quant, serial-order weighted sum).


## Round 18 — residency default + profile truthing (2026-08-26)

### Batched grouped-GEMM: PARKED
Working-tree prototype reverted to HEAD `3babf56` after hypothesis-churn
(grid.z plumbing produced zeroed groups 1-2; sanitizer showed OOB reads in
stage_slabs A-side; a masked build error fed stale .o into three test runs).
Root process lesson recorded: never measure behind a failed library build;
verify ninja actually relinked (`Linking` line visible) before trusting a
number. Design notes preserved in history; revisit only after the decode
GEMV lever below.

### Silent regression found & fixed
`e010bfe` accidentally rewrote `nvfp4_scale.h` from a pre-416d793 snapshot,
resurrecting whole-byte E4M3 zero-decode (broke pinned regression test).
Restored in `3e1e1d1`. Lesson: any commit touching a file should diff it
against expected content when unrelated hunks appear.

### Full expert residency by default — `17348a1`
Default expert cache now targets ALL routed experts (VRAM-bounded, 6 GiB
headroom, shrink-retry onto LRU path). Measured 42.1 -> **65.6 tok/s**
(release, TC fast-math, 128 tokens, RTX 5090, 35B-A3B). SelfCheck x3 bit-
stable; suite 107/107.

### Reproducible perf invocation (was mis-measured twice)
    LSX_TEST_FAST_MATH=1 LSX_TEST_PERF_GEN_TOKENS=128 \
      ./build/linux-release-znver4/tests/test_blackwell_perf_moe "[qwen35moe]"
Without LSX_TEST_FAST_MATH=1 the harness silently runs the SCALAR path
(~10 tok/s) — do not compare those numbers against TC baselines.

### Profile truthing (nsys r2)
CUPTI tracing inflates small-kernel durations ~4-5x here (gemm_tile med
68.7 us traced vs ~15 ms/step real wall for >1100 launches — impossible
otherwise). Trust relative shares, not absolute us, from nsys on this box.
Post-residency GPU is nearly saturated with qwen_nvfp4_gemm_tile_kernel
(~88% of traced kernel time, >1100 launches/step incl. split-K inflation).

### Next lever (priority 1): decode-path NVFP4 GEMV
At seq==1 the routed/shared GEMMs read ~1.73 MB/expert but sustain only
~30 GB/s effective through the 16x16x32 tile kernel (M=1 wastes the tile
M-dimension entirely). Bandwidth floor says ~420 us/token of weight
traffic -> 700+ tok/s realistic ceiling vs current 65. Options: dedicated
weight-streaming GEMV kernel for seq==1, or cuBLASLt NVFP4 path. Router
dots/select (~7% traced) second; head GEMM third.


## Round 19 — decode GEMV for seq==1 (2026-08-27)

### Shipped (opt-in): weight-streaming NVFP4 GEMV
`qwen_nvfp4_gemv_kernel` (`qwen_inference.cu`): one warp owns a K-slice of an
output row; packed E2M1 nibbles stream straight from global memory via
coalesced `uint4` loads (32 elements/lane load) -- no shared-memory slab, no
MMA, no cp.async. Per-row partials merge through a shfl tree + ordered smem
sum (fixed partitioning -> run-to-run bit-stable). Host entries:
`RunQwenNvfp4GemmSeq1GemmV` (direct, tests) and opt-in
`SetQwenSeq1Gemv(true)` dispatch inside `RunQwenNvfp4Gemm` under fast-math
(K%32==0, K>=256, <=12 MiB packed+scales, no fused segments).

### TDD net & the scale-mapping trap
Suite `test_qwen_nvfp4_gemv`: host-ref parity, run-to-run bit-determinism,
dispatch-reached (bitwise-equal vs direct entry), and an adversarial
chunk-scale-block pin. LESSON: the first kernel passed parity while mapping
nibbles onto the wrong half of the 32-element chunk's two scale blocks --
the pseudo-random scale pattern was octet-quantized so ~87% of adjacent
block pairs decoded identically and masked the bug. The adversarial test
alternates scale bytes 2^-6 / 2^7 so any mixup explodes. Every dequant
kernel test needs a block-alternating scale pattern.

### Micro (release, RTX 5060 Ti 36 SM / 448 GB/s theory)
GEMV vs TC mxf4nvf4 per-kernel: 4096x4096 63 vs 116 us; 4096x9728 128 vs
160; 9728x4096 178 vs 155; head-class 1536x4096 29 vs 50; MoE expert class
2048x512 / 512x2048 14/13 vs 50/59 us (3.6-4.7x). Effective DRAM ~120-167
GB/s (~35% theory) in this seq==1 latency-bound shape region.

### E2E (128 tok, FAST_MATH=1, qwen3.5-35B-A3B, this box)
TC baseline 20.7 tok/s. Default opt-in OFF == baseline bits exactly.
GEMV experts probe (`LSX_GEMV_EXPERTS=1`, routed w1/w3/w2 split path via the
public entry) 21.9 tok/s (+6%) with coherent text. Flat default result
root-caused: the routed-expert chain calls `tc::RunQwenTcGemmPreQuant`
DIRECTLY and bypasses the public entry entirely -- the entry-level lever
only reaches aux/shared-down shapes that are not the bottleneck on this
box (workload is launch-latency heavy: >1100 small launches/step).
First-step logits FP16-act (W?A16) vs TC W4A4: mean|d|~0.44 across vocab --
a different but equally valid tolerance-class basin (both selfcheck-pass;
text style differs: TC phrase-loop vs GEMV clean sentence-loop).

### Decision / next levers
Kernel ships tested but OFF by default (LSX_DEC_ENABLE precedent); the
routed chain needs decode-driver integration (single-stream fused expert
loop reading d_x directly, or prequant-GEMV form) rather than entry-level
routing. Launch-overhead reduction (>1100 tiny launches/step) is the first
order term before any bandwidth ceiling can matter at seq==1.


### Round 19b — batched routed-expert GEMV chain (2026-08-27)
`RunQwenMoeRoutedGemmVBatch` (qwen_moe_inference.cu): the per-slot {w1,w3,w2}
loop (K x 5 kernels/layer) becomes 4 launches per layer -- w13 GEMV batch
(rows = K*2I, gs1|gs3 split at I), elementwise swiglu, down GEMV batch to
per-slot partials, ascending-order coeff reduce into d_acc (identical fp32
add order to the legacy coeff_add chain -> deterministic). Device slot
table {dev_slot,gs1,g3,g2,coeff} built host-side after residency resolve
and uploaded per step; scratch via cudaMallocAsync (~57 KiB). Gate:
`LSX_GEMV_EXPERTS=1`, zero cache misses this step (misses keep the legacy
fencing contract), I%32==0 && H%32==0.

TDD: test_qwen_nvfp4_gemv [batch] -- ground truth is the PINNED single-row
GEMV entry invoked per stage (gate/up slab slices + down on host-silu'd
h), all stage buffers bit-compared, acc within 1e-6 rel. Two earlier test
drafts taught a lesson: hand-written host references drift (an unnoticed
scale-index divergence in a "correct-looking" ref cost hours); when a
pinned production kernel exists for the same math, USE IT as the oracle.

E2E (release, 128 tok, FAST_MATH=1): baseline 19.8 tok/s / ttft 714 ms ->
batched 21.0 tok/s (+6%) / ttft 581 ms (-19%). Text coherent. SelfCheck x3
passes under the flag. Default (flag off) bit-unchanged vs TC baseline.
Launch count cut: ~840 fewer launches/step, yet only +6% e2e on this box --
confirms launch overhead shares time with per-layer host syncs and small
attention kernels; deeper gains need the single-stream decode driver pass.


### Round 19c — no-handshake routed chain (latent, flag-gated)
`RunQwenMoeGemvGatherTable` + `QwenMoeExpertStore::{h,d}_gemv_meta`
({dev_slot,gs1..gs3} per flat expert): EnsureQwenMoeExpertResident mirrors
residency+scales at insert time; once EVERY expert is mirrored, the decode
driver skips the per-layer router readback/busy-poll entirely and composes
the batch GEMV slot table on device (`LSX_GEMV_EXPERTS=1`). Correct by
construction: readiness requires a complete mirror; under full residency
evictions are impossible so the mirror is authoritative forever.

STATUS on this Windows box: LATENT ONLY. The 6 GiB VRAM headroom forces
partial residency (resident_mb oscillates 1427-1548 across processes), so
some experts stay unstaged and readiness never trips; e2e effect here is
unmeasurable (short-run tok/s variance +-15% from WDDM/clock effects
swamps the lever). Validate on the full-residency Linux/5090 config next
session before any default-flip decision.

TEST lesson: catch2 fixture allocated cudaMalloc(&p, sizeof(vector)) --
sizeof the std::vector OBJECT (24 B), not its payload; route/meta reads on
device returned garbage while coeff checks coincidentally passed. Always
size cudaMalloc from .size()*sizeof(T[0]) of the payload.


## Round 20 — replay audit + head-GEMV (2026-08-27)

### Replay stage audit (LSX_AUDIT_STAGES=1, stderr)
Replays the final decode step 24x with sequential timestamp events
(idempotent, fixed kv/scratch); elapsed between stamps = GPU-timeline time
per stage including submission queue gaps. Permanent env-gated diagnostic.

GPU-busy breakdown per decode step (release, RTX 5060 Ti, FAST_MATH):
embed ~0% | full-attn 6.3% | linear-attn (GatedDeltaNet) 17.1% |
MoE FFN (routed+shared+router) 72.9% | head 3.7%.

### The audit's real headline: wall >> GPU-busy
Real steps measure ~50-170 ms/token against ~2 ms of audited GPU work on
this WDDM box -- >90% of wall time is submission/fence latency, not GPU.
Consequences: (a) kernel micro-wins (incl. head-GEMV vs TC) barely move
Windows e2e; (b) the priority ladder flips to CUDA Graphs / no-handshake
everywhere / kernel-count reduction before ANY further bandwidth tuning;
(c) Linux native stacks sit much closer to GPU-busy, which is where all
earlier tok/s targets were defined.

Head-GEMV verdict from [MICRO][lm_head] (K=2048,N=248320, 272.8 MB):
tc=1693 us vs gemv=1625 us (168 GB/s effective, both DRAM-limited) --
parity, no 4x win available here because the flat-row GEMV saturates at
~37% of theory while TC MMA is already DRAM-bound at M=1. Shipped as
opt-in `LSX_GEMV_HEAD=1` inside RunQwenNvfp4Gemm (K>=1024 && N>=16384)
with a large-N parity pin ({256,65536} and {2048,20000}).

### Test suite footprint
test_qwen_nvfp4_gemv now 32 cases / 141856 assertions (Debug+Release green);
qwen selfcheck x3 green; default dispatch bit-unchanged when flags off.


### Round 20b — persistent decode scratch (launch-overhead pass, phase A)
Replaced the hot-loop cudaMallocAsync/FreeAsync pairs with arena slices:
(1) per-layer activation quantize (was 2 alloc+2 free x 40 layers/step),
(2) batched-GEMV scratch {table,ygu,hbuf,ypart} in both the handshake and
no-handshake paths (was 4+4 driver calls per layer). RoutedFFN signature
grows trailing optional scratch params (header synced). Also fixed a
round-19b refactor slip: the handshake branch passed a DEVICE table into
the H2D wrapper -- it now targets MoeGemmVBatchCore directly with the
same host-table memcpy.

Interleaved 64-token A/B (release): default 12.9-13.7 vs gemv-experts
14.2-16.0 tok/s (+10..17%) -- batched chain benefits consistently from
the submission reduction. Absolute tok/s drifts +-20% between sessions on
this WDDM box; only interleaved same-session comparisons are meaningful.
Selfcheck x3 green under the flag.


### Round 20c — FIXED NEXT STEP: CUDA Graph around the per-layer FFN section
Preconditions (reuse no-handshake readiness): `QwenMoeGemvNohandshake(store)`
true AND `LSX_GRAPH_FFN=1`. Both features stay latent on boxes where VRAM
forces partial residency (this Windows card); engage properly on the
Linux/5090 full-residency config.

Capture protocol:
1. Warm-up step must complete through the LEGACY/batched flow so cache
   fills and all LRU state settles; engagement is evaluated per layer on
   the SECOND clean pass.
2. For layer l capture once at first engaged pass: nodes = gather-table ->
   w13 GEMV batch -> swiglu -> down GEMV batch -> coeff-reduce ->
   SharedCombine (incl its fused shared weights built before capture).
   Pointers are session-stable (arena slices round-20b; weights static).
3. Replay = one cudaGraphLaunch per layer; NO dynamic scalars inside
   (routed FFN section carries none -- kv_base lives in attention only).
4. Invalidation: if any miss occurs later (impossible when
   cache.slots >= total_experts), fall back to chain launch and mark
   graphs dirty; rebuild after next clean full pass.

Measurement protocol (WDDM box):
- WALL tok/s: only interleaved same-session comparisons, >= 3 alternations;
  absolute values drift +-20% between sessions.
- GPU budget: LSX_AUDIT_STAGES replay averages (stable across sessions);
  report avg_step_ms per config plus share table.

Acceptance for graphs: audited GPU-step unchanged (numerics bit-identical)
AND wall tok/s improves >= 10% interleaved on the full-residency box, else
park like batched experiments.


### Round 20d — measurement session results (audit v2: per-round fenced)
Fixed audit methodology: accumulate event pairs immediately after a
per-round stream fence (previously reused events were re-recorded before
readout, scrambling cross-round spans -- head segment showed impossible
3.6 TB/s). Head now validates against [MICRO][lm_head]: 1.73 ms == 1693 us.

GPU-timeline per decode step (includes ALL submission gaps):
  default TC    : 40.4 ms/step   share% ff 68.5 | lin-attn 20.6 | full-attn 6.5 | head 4.3
  gemv-batched  : 42.9 ms/step   share% ffn 74.3 | lin-attn 15.8 | full-attn 5.8 | head 4.0
Head 272.8 MB @ ~430 GB/s effective (1.7 ms) is near the DRAM roof and
identical both paths. Estimated pure-kernel compute <= ~8 ms/step; the
remaining ~32+ ms are submission/fence stalls (WDDM).

WALL interleaved x3 (128 tok): default avg 20.5 tok/s (ttft 508-576ms)
vs gemv-batched avg 20.8 (+1.5%, within noise). Conclusion unchanged and
now NUMERICALLY pinned: kernel-level wins cannot move this box until
submission overhead dies. Phase-B graphs targeting ~35 ms/step of stalls
is the single highest-leverage item; on native-Linux stacks expect the
stall fraction materially smaller -- measure there first.

Priorities ladder (post-measurement):
  P0 (any stack): verify no-handshake/graphs on Linux full-residency.
  P1 (this stack): CUDA Graph per-layer FFN (round-20c protocol).
  P2: fuse linear-attn epilogues (lin-attn is 20% of GPU budget, biggest
      single non-FFN slice; one fused decode kernel could halve it).


### Round 20e — Phase B implemented: per-layer FFN CUDA graphs (latent)
`LSX_GRAPH_FFN=1` + no-handshake readiness capture, once per layer:
gather-table -> RunQwenMoeRoutedGemmVBatchCore (4 kernels) -> SharedCombine,
then a single cudaGraphLaunch per engaged pass; failed captures / launches
fall back to the plain chain permanently for that layer (capture executes
no kernels, so fallback is numerically identical by construction).
Bit-transparency of the captured section pinned by
test_qwen_nvfp4_gemv "[graph] Batched GEMV core is CUDA-graph transparent"
(chain vs capture+replay outputs memcmp-equal).

MoeGemmVBatchCore renamed/exported as public
`RunQwenMoeRoutedGemmVBatchCore` (device-table variant) for the driver,
the tests and future reuse.

Status on this Windows box: latent (partial residency => readiness never
trips); e2e ran the fallback chain with batched ids unchanged and
selfcheck x3 green. Actual graph win measurement deferred to the Linux
full-residency session together with no-handshake (round-20c acceptance):
expect collapse of per-layer submission stalls (~32 ms/step WDDM-bounded
here), plus P2 candidate: fused linear-attn decode epilogue.


## Round 21 — gemma routed GEMV (experimental) + branch finalization

Shipped: `RunRoutedGemvBatchSeg` (gemma_inference.cu tail) + shared
`nvfp4_gemv_math.cuh` dot core (single source of truth, also consumed by
the qwen batched chain). Gemma slot layout keeps three separated slabs
(w1|w2|w3 / s1|s2|s3), so the pair stage takes independent gate/up base
offsets; activation is geGLU (gelu_tanh), reduce lands in FP32 and the
driver casts to the BF16 pipeline. Opt-in `LSX_GEMMA_GEMV=1`, independent
of LSX_GEMMA_MOE_BATCH. Tests [gemma]: host-ref parity incl adversarial
block-alternating scales + determinism (suite now 36 cases).

STATUS: kernel chain engages ([GEMMA-GEMV] trace confirmed; geometry
H=2816/MI=704 both %32==0, offsets/scales verified via trace dump) but
e2e generation DIVERGES from step 2 (junk token then early stop). Step-1
top-1 matches baseline; unit parity passes. Working hypothesis: the
W?A16/F32-hybrid numeric class vs gemma's BF16 pipeline (norms +
softcaps tightly coupled) shifts logits beyond this model's basin
stability -- unlike qwen35moe where the same class change stayed benign.
NEXT investigation (Linux session): BF16-output variant of pair/down
stages (or tolerance study against the dequant+cuBLAS reference path)
before any default consideration.

### Branch finalization state (feat/decode-gemv)
Landed: seq==1 GEMV core + qwen entry dispatch (opt-in), batched routed
chains for qwen_moe (handshake + no-handshake) with persistent scratch,
per-layer FFN CUDA graphs under no-handshake readiness, head-GEMV opt-in,
replay stage auditor, gemma GEMV experiment. All suites green locally;
default production routing bit-unchanged everywhere. Deferred to Linux/
5090: no-handshake + graphs validation (round-20c acceptance >= 10% wall),
gemma numeric-class resolution, linear-attn fusion (P2).

Windows regression gate (final, 2026-08-27, windows-release): PASS --
micro_gemm, moe_kernels, moe_model, moe_selfcheck(x3), nvfp4_dense,
nvfp4_gemv(36 cases), nvfp4_tile_gemm. Default routing bit-unchanged
(all experimental paths env/opt-in gated). Branch ready for merge;
acceptance metrics for no-handshake/graphs/gemma numerics to be collected
on the Linux full-residency stack per round 20c/21 protocols.


## Round 22 — Blackwell utilization audit v2 (2026-08-27)

Scope: fresh pass over every .cu TU against the Blackwell feature matrix
(TMA / mxf4nvf4 MMA / cp.async pipelines / stream-ordered memory / pinned
staging / L2 persistence / clusters / CUDA Graphs / sync discipline),
cross-checked with nvidia-cuda-docs MCP. Regression gate re-run first:

`ctest --preset windows-debug -R "qwen|gemma|blackwell"` -> **40/40 PASS**
(6 SKIP by design: kv_cache + perf harnesses + selfchecks without weights).
Docs reference updated in the same session per protocol (see B4 addendum
below).

### Coverage matrix (feature x model family)

| Feature | qwen dense | qwen35moe | gemma | glm | deepseek | kimi |
|---|---|---|---|---|---|---|
| TC NVFP4 mxf4nvf4 W4A4 | yes (fast-math) | yes | rejected r21 (measured) | no (bit-exact contract) | **no (scalar)** | n/a (BF16 weights) |
| Tensor cores via cuBLAS | fallback path | shared/routed fallback | yes (main GEMMs) | no | no | yes (cublasGemmEx BF16) |
| TMA bulk-tensor staging | behind LSX_DEC_ENABLE only | same tile kernel | no | no | no | no |
| cp.async slab ring | yes (TC GEMM) | yes | no | no | no | no |
| Stream-ordered allocs | yes | yes (+arena r20b) | partial (prefill tail) | load-time only | load-time + cache | load-time |
| Async handshakes (event/busy-poll) | yes | yes + no-handshake latent | host mirror for layer_scalar | event-scoped tok_ready | H-01 split route_token | **FULL drain per layer/token** |
| CUDA Graphs | latent (FFN section) | latent (FFN section) | no | no | no | no |

### New findings

| ID | Sev | Evidence | Summary | Status |
|---|---|---|---|---|
| F-11 | P1 | kimi_inference.cu:947,1164 | Kimi MoE router ids readback does a FULL cudaStreamSynchronize per layer per token (pre-H-01 pattern) and stages experts after the sync; GPU drains completely K layers x steps. Fix = DeepSeek-style split-phase begin/finish + busy-poll event; expert prefetch enqueued between phases | open (needs kimi tests: Windows policy blocks running them here -> Linux session) |
| F-12 | P1 | deepseek_inference_moe.cu / deepseek_inference_pipeline.cu (no qwen_tc_gemm.h include) | DeepSeek-V4-Flash runs ALL GEMMs through custom scalar NVFP4 kernels -- zero tensor-core usage in the heaviest model of the vault. TC wiring needs a numeric-class decision like qwen's fast_math toggle (W4A4 changes accumulation order vs scalar default) | phase-1 landed (Round 24): routed TC W4A4 under fast_math + streaming default (0.6->4.0 tok/s); e2e staging-bound; shared/attention caching pending |
| F-13 | P2 | NVIDIA docs B4 (nvidia-cuda-docs MCP) | tcgen05.mma tensor-memory forms are SM_100a/f+103a/f+110a/f ONLY per PTX ISA (re-verified via MCP): consumer sm_120a/121a cannot use them. Repo's warp-level mma.sync mxf4nvf4 route is confirmed correct; TMA (copies) IS available on sm_120 as probed | documented (B4 addendum) |
| F-14 | P3 | grep: no cudaStreamSetAttribute/cudaAccessPolicyWindow anywhere | L2 persistence hints unexplored: candidate micro-lever (persisting window over router scores / KV rows) for the Linux stack where wall==GPU-busy roughly holds; meaningless under this box's WDDM submission stalls | parked |
| F-15 | P3 | qwen_inference.cu:2633, qwen_moe_inference.cu:1374 | Hot-loop fallback still has blocking cudaMemcpy when pinned h_step staging is unavailable; both production paths allocate h_step so this is dead-in-practice defensive code | wontfix-until-simplified (note only) |

### Audit conclusions

1. The submission-bound verdict (Round 20d) stands and shapes everything:
   kernel-level feature gaps above do NOT move this Windows box; they are
   real on native-Linux/full-residency stacks where the earlier tok/s
   targets apply.
2. Biggest genuine architecture-utilization gaps ranked: F-12 (DeepSeek
   scalar-only GEMMs), F-11 (Kimi full-drain handshake), then the standing
   P0/P1 ladder from Round 20d/20e (no-handshake + FFN graphs validation,
   linear-attn fusion P2).
3. Consumer-Blackwell feature usage is otherwise near-complete where it is
   wired: warp-level NVFP4 tensor-core MMA, cp.async double/quad rings, TMA
   (opt-in), stream-ordered pools, pinned staging, zero-sync decode loops,
   launch-overhead work (fused GEMMs, arenas).

### Handoff queue — Linux / full-residency session (Round 22 -> next)

Execution order (each item TDD-first per protocol; Windows box cannot run
glm/deepseek/kimi model tests):

1. F-12: DeepSeek TC fast-math wiring. Design first (numeric-class decision
   mirroring qwen `fast_math` toggle), then tests before kernels; keep
   scalar default bit-exact.
2. Standing P0 from Round 20d/20e: validate no-handshake routed chain +
   per-layer FFN CUDA graphs on full residency (LSX_GEMV_EXPERTS=1,
   LSX_GRAPH_FFN=1); round-20c acceptance >= 10% interleaved wall.
3. F-11: Kimi router split-phase handshake (H-01 pattern) + TDD
   bitwise-equivalence test for the split API.
4. P2 carry-over: linear-attn decode fusion (conv_norm + gated_delta tail)
   with bit-exact cluster test; measure via LSX_AUDIT_STAGES on a stack
   where wall ~ GPU-busy.
5. F-14 exploration (optional): L2 persistence window over router/KV.
F-13/F-15 require no code.

## Round 23 — round-20c acceptance on Linux/5090 full residency (2026-09-01)

Platform: Ubuntu 26.04, AOCC 5.2.0 release build (`linux-release-znver4-aocc`,
`-march=znver4`, LTO), RTX 5090 32 GB (170 SM), EPYC 9V74. Qwen3.5-35B-A3B
full residency (all 40x256 experts device-resident, resident_mb=1548).
Harness: `test_blackwell_perf_moe "[qwen35moe]"`, prompt "The capital of
France is", LSX_TEST_PERF_GEN_TOKENS=128, interleaved same-session, 3
alternations per config (within-config spread < 1%).

### WALL tok/s (128 tok, interleaved)

| Config | tok/s | ttft_ms | vs A |
|---|---|---|---|
| A: TC default (FAST_MATH only) | 63.3-63.6 | 116-124 | — |
| E: A + GEMV_HEAD | 62.4-62.6 | 116 | -1.6% |
| C: GEMV_EXPERTS no-handshake | 52.7-52.8 | 164 | -17% |
| D: C + GRAPH_FFN | 52.7 | 164 | =C (bit-identical output) |
| B: D + GEMV_HEAD | 47.8-47.9 | 164 | -25% |

### LSX_AUDIT_STAGES GPU-step breakdown (24 replay rounds)

| Config | avg_step_ms | share% ffn | lin | full | head | top FFN layer |
|---|---|---|---|---|---|---|
| A (TC) | 10.94 | 64.8 | 21.4 | 10.1 | 3.6 | 0.180 ms |
| C (GEMV) | 20.60 | 81.7 | 11.0 | 5.4 | 1.9 | 0.424 ms |
| D (C+graphs) | 20.61 | 81.7 | 11.0 | 5.4 | 1.9 | 0.424 ms |

### Verdicts

1. **Round-20c acceptance: FAIL — park.** The no-handshake routed GEMV chain
   regresses 17% wall on the full-residency Linux/5090 stack. Root cause
   flips the Windows-box picture: Linux native submission is cheap, so the
   WDDM-motivated launch-count reduction (K x 5 -> 4 launches) buys nothing,
   while the GEMV chain's raw kernel efficiency loses to TC W4A4 MMA 2.35x
   per FFN layer (0.424 vs 0.180 ms). Round-19's "e2e flat" conclusion
   degrades to a regression on this stack.
2. **FFN CUDA graphs: bit-transparent, GPU-time and wall neutral** (C vs D:
   same ids, same 20.6 ms/step, same 52.7 tok/s). The round-20e premise
   ("collapse of ~32 ms/step submission stalls") does not apply on Linux —
   those stalls were WDDM-only. Graphs remain latent and harmless; keep the
   code, do not enable by default.
3. **Head-GEMV (LSX_GEMV_HEAD)**: near-neutral on the TC path (-1.6%),
   harmful inside the GEMV-experts config (-9% on top of D). Stays opt-in;
   not recommended.
4. **Production routing for 5090/Linux confirmed: default TC
   (fast_math only) = 63.5 tok/s.** "All levers on" (B) is the WORST config
   (47.9) — flag-stacking by name, not by measurement, is an anti-pattern.
5. Numerics: A vs C/D/E texts diverge after the shared " Paris." prefix
   (W4A4 vs W?A16 activation class) — expected; opt-in levers only, default
   routing bit-unchanged. First-gen "Paris" selfcheck contract intact in all
   configs.
6. Registry rows updated: F-11 remains open (next queue item); standing P0
   from Round 20d/20e CLOSED (verdict above); P2 linear-attn fusion remains
   the next kernel-level lever (linear-attn = 21.4% of TC GPU step).

## Round 24 — F-12 DeepSeek TC W4A4 phase 1 (2026-09-01)

Branch `feat/nohandshake-graphs-5090`. Design:
`2026-09-01-deepseek-tc-fastmath-design.md`; plan:
`2026-08... 2026-09-01-deepseek-tc-fastmath.md`. Baseline this session
(AOCC release, RTX 5090): 0.6 tok/s / ttft 20.6 s with the default partial
LRU expert cache.

### Shipped

1. **Streaming default (slots=0)** — `c79a297`. The routed working set
   (kTopK x layers x ~13.4 MB ≈ 3.5 GB/token) exceeds any VRAM; a partial
   LRU cache thrashed every token. Default now streams experts from pinned
   host memory. **0.6 -> 4.0 tok/s (x6.7), ttft 20.6 -> 1.8 s.**
2. **E8M0→UE4M3 scale repack** — `840cc12`. Repo pow2-E4M3 byte layout:
   exponent in bits [6:3] (`E4M3ScaleToFloat` = `((b>>3)+120)<<23`); mapping
   `E = e8 - 120`, byte = `E << 3`. Kernel + load-time range validator
   (`ValidateE8m0Range`, e8 > 135 -> load error naming expert). NOTE for the
   docs reference: the ue4m3 MMA operand reads the full byte
   (sign<<7|exp<<3|mant) — the repo's pow2 codec shifts the exponent into
   bits [6:3]; both agree on decode `2^(e-7)`.
3. **fast_math toggle** — `1411181`. Process-wide
   `SetDeepseekFastMath(bool)` from `ModelInitConfig::fast_math`; default
   OFF bit-exact.
4. **TC W4A4 routed-expert body** — `04aba62`. `run_routed_expert_ffn`
   dispatches to `qwen::tc::RunQwenTcNvfp4Gemm` (W1/W3: K=4096→N=2048;
   W2: K=2048→N=4096, routing weight folded into the TC epilogue scale;
   FP16 silu_mul in-place on the gate slab). UE4M3 slabs repacked on device
   into the paired TC stage region per expert (stage path). Load-time
   validation `fb0ef88`.
5. **Correctness evidence:** stage test (TC GEMM vs act-quant-aware host
   reference at K=4096/N=2048): **max_err 1.15 @ ref 4563 (0.025%)**;
   TC body deterministic (two runs bitwise identical); full deepseek suite
   8/8 green; OFF-mode selfcheck bit-exact; ON-mode "Paris" deterministic.
   Scale format trap found by TDD: byte 7 (E in low bits) decodes as 0 —
   tests caught the swapped nibble shift before any real-weight run.

### Phase-1 measurement: staging-bound, not compute-bound

E2E (32 tok, real weights, sequential): default 4.0 tok/s vs fast_math
3.9 tok/s — **no wall change**. Root cause: the routed working set moved
the bottleneck to pinned H2D staging: 6 experts x 43 layers x 13.4 MB ≈
3.5 GB/token at ~14 GB/s effective ≈ 250 ms/token floor. The TC path
removed dequant-compute time but the staging wall dominates. Next lever
(next session): async prestage overlap (GLM async-prestage pattern) —
stage layer l+1 experts during layer l compute; potential ~1.5-2x until
the pinned-bandwidth roof (~7 tok/s).

F-12 phase 2 (Task 4: shared-expert + attention F8 dequant caching,
bit-exact always-on) pending; `kind::mxf8f6f4` W8A8 for F8 sites parked.

### Phase 2 + final state (same session)

6. **Shared-expert dequant caching** — `553ae13`. Load-time FP16 (W1/W3) +
   FP32 (W2) caches filled by the SAME dequant kernels the per-call path
   uses → bit-exact by construction; per-call dequant launches skipped
   (~140 MB/token of dequant traffic removed). Cache params are trailing
   optionals; nullptr = legacy path. Bit-exact pinned by
   `test_deepseek_moe_moe "[shared-cache]"` (memcmp). resident
   11.7 -> 13.8 GB (+2.1 GB caches); text identical to pre-change.
7. **Attention F8 dequant caching: PARKED** (plan deviation, measured
   rationale) — the e2e is pinned at the pinned-H2D staging wall
   (~14 GB/s effective, ~250 ms/token floor), so removing the small
   per-call attention dequants cannot move wall tok/s; the work would
   touch the attention TU signatures for zero wall gain. Revisit only
   after async expert prestage lands and compute becomes visible again.

### F-12 final table (RTX 5090, AOCC release, 32 tok, sequential)

| Config | tok/s | ttft_ms | resident_mb |
|---|---|---|---|
| baseline (partial LRU cache, pre-session) | 0.6 | 20588 | 11695 |
| streaming default (Task 0) | 4.0 | 1794 | 11695 |
| + shared dequant caches (Task 4) | 4.0 | 1794 | 13759 |
| + TC W4A4 fast_math (Task 3) | 4.0 | 1793 | 13759 |

Acceptance gate: strategy-spec §5 items all met — OFF selfcheck bit-exact,
ON selfcheck "Paris" deterministic, stage test 0.025% vs host ref, full
suite 20/20 green (`deepseek|blackwell|expert|naming`), no perf regression.
F-12 status: routed TC W4A4 + shared caching landed; remaining wall is the
pinned-H2D staging roof — next lever is async expert prestage overlap
(new work item, not F-12).

## Round 25 — Kimi F-11 split-phase + device-side decode (2026-09-01)

Branch `feat/deepseek-expert-prestage`. Design:
`2026-09-01-kimi-split-phase-decode-design.md`; plan:
`2026-09-01-kimi-split-phase-decode.md`.

### Shipped

1. **Split-phase router handshake** — `7cfa3bc`, `226b6b4`. Public
   `KimiRouteBegin`/`KimiRouteFinish` (mirror of DeepSeek
   `route_token_begin/finish`): top-k kernel + async D2H into ONE
   generation-level pinned staging pair + event record; finish =
   event-scoped busy-poll (`cudaEventQuery` spin — round-20 pattern). The
   per-layer-per-token `cudaStreamSynchronize` full drain in
   `KimiLayerStack` is gone. TDD: begin/finish vs fused path
   bitwise-identical ids/weights (n=384, top_k=8, adversarial duplicate
   scores) — `test_kimi_moe_router "[kimi][router][split]"` 43 assertions.
2. **Device-side greedy decode** — `2502bff`. Per-step argmax D2H + sync
   removed: `kimi_feed_token_kernel` feeds `*d_best` into the next
   embed-gather on device; `gen.tokens` drained by ONE bulk D2H after the
   loop; ttft via events (`ev_start`/`ev_ttft`); `first_maxv` from
   `d_bestv` in the final drain. The greedy argmax drain site from the F-11
   evidence is gone. Zero full-drain syncs remain in the kimi inference TU
   (`grep cudaStreamSynchronize` empty).
3. **Host-truncate EOS + de-hardcode** — the `kKimiEosId = 163586`
   constant (Ingestion Protocol step 4 violation) replaced by an `eos_id`
   parameter threaded from `Tokenizer().eos_id` at both model call sites.
   `TruncateAtEos` cuts at the first EOS post-drain (`eos_id < 0`
   disables). TDD: `"[kimi][decode][eos]"` (cut/no-EOS/disabled/
   immediate-EOS), 10 assertions.

### Baseline + measurement (RTX 5090, AOCC release, real weights, 16 tok)

Harness: new `test_kimi_moe_model "[perf]"` (`adcf85d`), prompt
"The capital of France is", LSX_TEST_KIMI_GEN_TOKENS=16.

| Config | tok/s | ttft_ms | total_ms |
|---|---|---|---|
| baseline (per-layer + per-step full drain) | 0.927 | 5662 | 17264 |
| split router + device-side decode | 0.936-0.939 | 5617-5631 | 17044-17086 |

Verdict: **wall-neutral (+~1%)** on this workload. The kimi generation
loop is expert-staging-bound in the legacy path (mmap H2D per expert per
layer — same DRAM-floor pattern as DeepSeek Round 24); the removed syncs
were not the bottleneck at 16 tokens.

### Correctness status and the standing race

- Router split path: bitwise-identical to fused (43 assertions green).
- EOS truncate: green.
- **`test_kimi_moe_inference` "deterministic across calls" FAILS under
  the device-side decode** (`final_logits` memcmp across two Infer calls;
  max abs diff ~12.4, R1 varies run-to-run, R2 bit-stable per process).
  Bisect: green with a per-step drain restored (bisect1/2/3/6), red in any
  fully-async configuration regardless of router fence granularity,
  feed-kernel vs D2D, LRU evictions (KIMI_NOEVICTION), or per-layer vs
  per-generation pinned staging. first_maxv (step-0 logits) bit-stable in
  every configuration → divergence is confined to decode steps executing
  without an intervening drain.
- Root cause NOT yet isolated; the async decode is gated OFF by the
  failing determinism gate — see Round 25b (follow-up): the decode-loop
  reverts to the per-step event-scoped drain until the corruption is
  isolated (candidates: cublas workspace/algo selection under deep
  enqueue, expert-store LRU interaction with the copy stream, attention
  workspace reuse across async layers).

Decision per strategy-spec gate 1 (bit-exact contracts are binding):
decode-loop async is PARKED (revert to drain) until root-caused; the
router split API (bitwise-equivalent by test) stays.

### Round 25b — resolution (same session)

The failing gate forced an interim design: per-step EVENT-SCOPED readback
(cudaEventRecord + cudaEventQuery busy-poll per step — event fires after
argmax; identical wait semantics to the old full drain, since all prior
work lives on the same stream) instead of the blocking
cudaStreamSynchronize. Result: determinism gate GREEN and stable across
runs (R1==R2 bit-identical final_logits in repeated processes); full kimi
suite 12/12 green including the real-model selfcheck.

Final Round 25 shipped state:
- split-phase router handshake (kept; bitwise-equivalent, event busy-poll,
  full drain per layer removed),
- generation-level pinned router staging + one event (vs per-layer
  alloc/free — the per-layer cudaHostAlloc/EventDestroy cycle was itself a
  hidden device-wide sync per layer),
- device-side token feed (kimi_feed_token_kernel) — no host->device id
  write-back per step,
- host-truncate EOS from tokenizer-sourced eos_id (harcoded 163586 gone),
- interim per-step event readback for argmax output (parked device-side
  drain; race un-root-caused — values chase prior-call state in any
  fully-async config, R1 varies / R2 stabilizes, first_maxv always
  bit-stable => prefill path is clean, corruption confined to async decode
  steps).

Perf (16 tok, real weights): 0.927 -> 0.93 tok/s (wall-neutral within
noise; staging-bound workload, consistent with Round 24's DeepSeek
verdict). F-11 core deliverable (per-layer-per-token full-drain sync
removal) LANDED and verified; the per-step argmax drain is downgraded
from full cudaStreamSynchronize to an event-scoped busy-poll with the
same coverage.

## Round 26 — qwen moe 80+ tok/s device-driven decode (2026-09-02)

Branch `feat/qwen-moe-80tps`. Design: `2026-09-01-qwen-moe-80tps-design.md`;
plan: `2026-09-01-qwen-moe-80tps.md`. Baseline (Round 23 config A): 63.5
tok/s / ttft 116 ms.

### Shipped (default path, no env flags)

1. **Batched TC routed-expert chain** — `RunQwenMoeRoutedTcBatch`
   (qwen_moe_inference.cu): gather-on-device -> ONE w13-batch TC GEMM
   (grid.z = K slots, split-K=8 via `QwenTcBatchSplitK` mirror) ->
   batched swiglu-quant (slot-major, exact clone of the per-slot
   16-group chain) -> w2-batch TC GEMM -> ascending coeff reduce. Engages
   at full residency + fast_math + fused layout; legacy per-slot chain is
   the fallback for partial residency.
2. **Full-residency expert prestage at load** — `Qwen35MoeModel::Load`
   stages every expert + completes the no-handshake mirror once
   (378 ms one-time). Without it the readiness mirror never completes
   (coupon-collector: 128-token routing touches ~63% of experts), so the
   device-driven path silently never engaged — root cause of the flat
   64.5 tok/s readings mid-session.
3. **Device-side token feed** (kimi round-25 pattern): argmax stays on
   device (`qwen_record_token_kernel` + `qwen_feed_token_kernel`); the
   per-step blocking argmax sync is gone; bulk token drain once per
   generation; EOS via host-side `TruncateTokensAtEos` (tokenizer-sourced
   eos_id). Per-step host-gap shrank ~4.7 -> ~0.6 ms/token.
4. **kTN=16 TC tile variant** — latent (bit-exact by test, engages only
   on narrow-N shapes; never fires on 35B-A3B where w13 N=1024 /
   grid math keeps kTN=64).
5. **Launch-hygiene leftovers**: persistent fused-output scratch for the
   linear-attn fused GEMM (removes 60 pool calls/step; wall-neutral,
   kept); acc-init stays a separate launch (the seeded-reduce fold was
   REVERTED — it corrupted real-model decode while passing all synthetic
   gates; see 2e97def).

### Corrections caught by TDD (would have shipped broken)

- Batch down GEMM applied the w13 two-segment scale split instead of
  scalar gs2 (`kSeg13` epilogue flag, 9f85044). The synthetic H=64
  parity suite could NOT catch it (fixture global scales coincide);
  the real-geometry GEMM parity test (K=2048, split-K=8, distinct
  per-segment scales) caught it.
- acc-init fold: synthetic suites green, real-model decode garbage —
  synthetic-only coverage is insufficient for driver-level folds.

### Results (RTX 5090, AOCC release, 128 tok, interleaved)

| Config | tok/s | ttft_ms | vs baseline |
|---|---|---|---|
| Round-23 default A (legacy chain) | 63.5 | 116 | — |
| 64.5 plateau (mid-session, mirror never tripped) | 64.5 | 116 | +1.6% |
| **Round-26 default (prestage + TC-batch + device feed)** | **101.1-101.2** | **51.4-51.6** | **+59%** |

- Run-to-run: 3 alternations, spread < 0.1%; token ids identical across
  runs. Selfcheck x3 " Paris" PASS (32 assertions).
- Stage audit: avg_step_ms 10.94 -> 10.82 GPU (ffn 64.8 -> 62.9%); the
  wall gain is dominated by the removed per-layer host handshakes and
  per-step argmax sync (ttft 116 -> 51 ms includes the prestage effect
  on first-step routing).
- Text: coherent, deterministic, differs from the legacy chain's greedy
  output (contract amendment 06e548e: run-to-run determinism + selfcheck
  gate are binding; bitwise legacy equality waived for the TC-batch
  path). OPEN ITEM: the batch-vs-legacy divergence on real geometry is
  not yet root-caused — unit GEMMs (real geometry, split-K) are
  bit-identical; the divergence lives in the composition (swiglu-quant
  slot-major path or reduce composition). Synthetic full-chain parity
  (H=64) is bit-exact; a real-shape full-chain synthetic test hits a
  NaN domain from fixture weights and cannot serve as a bit-pin.

### Verification

- `ctest -R "qwen|blackwell|naming"`: 35/35 green.
- Real-model selfcheck x3 bit-stable " Paris".
- New TDD: real-geometry GEMM parity (bitwise, incl. split-K=8 slice
  order + adversarial per-segment scales), fallback parity under partial
  residency, device-feed two-call `final_logits` memcmp gate, EOS
  truncation helper.

## Round 27 — full CUDA-inference test-integrity audit (2026-09-14)

Method: static invariant review + `compute-sanitizer`
(memcheck/initcheck/racecheck) sweep over every synthetic CUDA test
(kernels, pipeline, loader, micro/mma). Findings (all TDD: failing
detector -> minimal fix -> green).

1. **T-27-1 (critical, fixed): Round-26 real-geometry TC-batch parity
   test was vacuous.** Commit `7aa7008` ("clean debug markers")
   accidentally deleted the four `cudaMemcpy(..., DeviceToHost)` lines
   when rewriting the memcmp loop, so `r13/b13/r2/b2` stayed
   zero-initialised and every `CHECK(ok13/ok2)` passed unconditionally.
   The verification claim in Round 26 ("real-geometry GEMM parity ...
   caught 9f85044") was therefore false as of `7aa7008`. Restoring the
   copies immediately exposed a second defect: the test reused a
   single-row activation as the *per-slot* down-GEMM activation
   (`a_slot_off = slot*K/2`), reading OOB (`memcheck`: 28 invalid global
   reads in `lsx_tc_gemm.cu cp_async16`). Fixed with per-slot activation
   buffers (`K*I/2` packed + `K*(I+15)/16` scales), matching per-slot
   reference windows, plus two anti-vacuity checks (slot 0 != slot 1).
   Production callers (qwen `qpack_batch`, kimi `s.hq`) were already
   correctly sized — test-only defect.

2. **T-27-2 (critical, fixed): synthetic MoE fixture geometry drift.**
   `qwen_moe_test::MoeCfg().num_attention_heads = 4` while
   `BuildSyntheticMoeTensors` derives every dense shape from
   `qwentest::TestCfg()` (`= 2`). The converted topology advertised
   `NH*HD*2 = 256` but the q_proj weight had 128 rows, so
   `qwen_split_query_kernel` read uninitialised `d_qc` (`initcheck`:
   2972 errors). Added a fixture self-consistency guard
   (`test_qwen_moe_convert`) and aligned `MoeCfg` with `TestCfg`
   (`num_attention_heads = 2`). `initcheck` -> 0.

3. **T-27-3 (pre-existing red, fixed): qwen SelfCheck scratch
   expectation.** `test_qwen_moe_{,36_}selfcheck` asserted
   `arena_allocs <= 1`, but the `Load()` 1-token warmup (`8d85335`)
   sizes the arena for `max_seq == 2` before the first real generation,
   so the first SelfCheck grows it exactly once (`allocs == 2`).
   The persistence invariant (no reallocation across cycles) still
   holds; both tests now assert `allocs <= 2` after cycle 0 and
   `allocs == allocs0` after cycle 1. `test_qwen_moe_36_selfcheck`
   (real model) green; 3.5 sibling SKIPs without its vault.

4. **T-27-4 (hygiene, fixed): uninitialised copies in kernel tests.**
   `test_glm_moe_flash_kernels` D2H-read slots the `flash_exp0_probe`
   never writes; `test_blackwell_micro_gemm` D2H-read the 4-byte struct
   tail padding of `QwenMoeGvSlotDev` the gather kernel does not write.
   Zeroed the probe output and the device slot table (`initcheck` -> 0).

### Parked / reported (no change this round)

- **F-06 still open**: `-fmad=false` covers whole object libraries
  (`lsxcommon_glm_cu` etc.). Shared kernels use explicit `*_rn`
  intrinsics, so narrowing is numerically inert, but per-source CUDA
  flag surgery is deferred (risk > value).
- **GLM-5.3-Flash stream invariant deviation**:
  `glm_moe_flash_inference.cu` performs a per-token
  `cudaStreamSynchronize` (token-id H2D staging, line ~1606) and a
  per-token argmax D2H readback (line ~1896). Both are documented in
  code as deliberate; unlike qwen/kimi there is no device-side feed, so
  AGENTS' "one stream, no per-token host return" invariant is formally
  violated. Correctness-neutral; a device-feed port (qwen/kimi pattern)
  is the follow-up, gated on a real-model selfcheck.
- **Dead TU**: `deepseek_moe_inference_csa.cu` is a 15-line
  include-only stub (bodies live in `deepseek_moe_inference_attn.cu`);
  candidate for removal.

### Verification (this round)

- `ctest --preset linux-debug` (fast): 45/45 green.
- Affected: `test_qwen_moe_kernels`, `test_qwen_moe_convert`,
  `test_qwen_moe_loader`, `test_glm_moe_flash_kernels`,
  `test_blackwell_micro_gemm`, `test_qwen_moe_36_selfcheck` green;
  `test_qwen_moe_selfcheck` SKIP (no 3.5 vault).
- `compute-sanitizer` (memcheck + initcheck + racecheck) on the
  synthetic kernel/pipeline suite: 0 errors/hazards after the fixes
  (`test_qwen_moe_kernels`: memcheck 235 -> 0, initcheck 2972 -> 0).
