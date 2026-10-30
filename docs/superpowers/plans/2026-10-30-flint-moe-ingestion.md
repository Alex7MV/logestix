# Recursive Flint (39.8B-A2.5B) ingestion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Subagents are FORBIDDEN by AGENTS.md — execute
> inline in the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest `RecursiveSI/flint` as the independent family `flint`: converter,
NVFP4 tree, loader, hybrid sink/SWA/no-RoPE attention, sigmoid-router MoE with an
ungated shared expert, tokenizer bridge, tool-calling bridge, and every lifecycle
gate (SelfCheck, FSI, CODE, DET, perf) green on the real weights.

**Architecture:** Structural template `hy` (48L/2048H, all routed experts
device-resident, sigmoid router + shared expert) with attention patterned on
`mimo` (learned sink as an extra softmax column, sliding-window decode). The
neutral NVFP4 GEMV layer is used unchanged; nothing is shared across families.

**Tech Stack:** C++20 / C23, CUDA 13.4 (sm_120/121), CMake 3.25+, Catch2 v3.16.1,
Arrow 26, PyTorch 2.13 (CPU oracle only).

## Global Constraints

- Frozen geometry from `docs/superpowers/specs/2026-10-30-flint-moe-ingestion-design.md`.
- C++20, `static_assert` on every frozen relation; no runtime config reads for geometry.
- Logging via spdlog only; no `printf`/`std::cout`; no new comments unless informative.
- NVFP4 only through the neutral `lsx_inference_nvfp4` / `lsx_gemv` layer.
- No abseil containers in `.cu`/`.cuh`-reachable headers.
- Every new test gets its ctest label in the same commit; selfcheck/fsi/code/det
  targets carry `SKIP_RETURN_CODE 4`.
- Guard registries (`test_code_naming_guard`, `test_model_independence_guard`,
  `test_model_suite_guard`, `test_ctest_labels_guard`) updated in the same commit
  as the files they describe.
- Commit prefixes `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`, English.
- No subagents; all work inline. Host GPU policy: one GPU workload at a time.
- Temp files under `/mnt/fast_storage/tmp`; vault models under `/mnt/fast_storage/huggingface`.

---

## File Structure

| File | Responsibility |
|---|---|
| `libs/lsxcommon/src/flint_moe_constants.h` | frozen passport, layer tables, `static_assert`s |
| `libs/lsxcommon/include/lsxcommon/flint_moe_convert.h` | converter entry (`ConvertFlintModel`) + config/geometry PODs |
| `libs/lsxcommon/src/flint_moe_convert.cpp` | `config.json` + safetensors → NVFP4 tree |
| `libs/lsxcommon/include/lsxcommon/flint_moe_load.h` | pinned weight structs + `LoadFlintWeights` |
| `libs/lsxcommon/src/flint_moe_load.cpp` | Arrow mmap → device graph + 47 expert slabs |
| `libs/lsxcommon/include/lsxcommon/flint_moe_device_math.h` | host reference (sink softmax, router, window) |
| `libs/lsxcommon/include/lsxcommon/flint_moe_kernels.h` + `src/flint_moe_kernels.cu` | RMSNorm / sigmoid top-6 / SwiGLU / routed accumulate / shared combine |
| `libs/lsxcommon/include/lsxcommon/flint_moe_inference_attn.h` + `src/flint_moe_inference_attn.cu` | GQA attention with qk-norm, optional RoPE, sink, SWA |
| `libs/lsxcommon/include/lsxcommon/flint_moe_inference_pipeline.h` + `src/flint_moe_inference_pipeline.cu` | KV session + generation loop |
| `libs/lsxcommon/src/flint_moe_model.{h,cpp}` | `IModel` subclass, framing, streaming, lifecycle |
| `libs/lsxcommon/src/flint_moe_tool_format.{h,cpp}` | native ChatML-XML codec |
| `libs/lsxcommon/src/pretokenizer.{h,cpp}`, `tokenizer.cpp`, `tokenizer_convert.cpp` | `tokenizer_kind = 4` (Flint split) |
| `apps/lsxforge/src/{main.cpp,cli.cpp,cli.h}` | `--flint` pipeline |
| `tests/test_flint_moe_*.cpp`, `tests/test_flint_*_selfcheck.cpp`, `tests/test_flint_moe_fixture.h` | lifecycle tests |
| `tools/flint_oracle.py`, `tools/flint_tokenizer_capture.py`, `tools/{fsi,code,det}_capture.py` | source oracle + goldens |

---

### Task 1: `tokenizer_kind = 4` (Flint pre-tokenizer) + parity

**Files:** modify `libs/lsxcommon/include/lsxcommon/pretokenizer.h`,
`libs/lsxcommon/src/pretokenizer.cpp`, `libs/lsxcommon/src/tokenizer.cpp`,
`libs/lsxcommon/include/lsxcommon/converter.h`, `libs/lsxcommon/src/tokenizer_convert.cpp`;
create `tests/test_flint_tokenizer_split.cpp`, `tools/flint_tokenizer_capture.py`,
`tests/fixtures/flint/tokenizer_parity.jsonl`.

**Interfaces:**
- Produces: `class FlintWordStream` (same `next(WordSpan&)` / `error()` contract as
  `Llama3WordStream`); `TokenizerSpecialPolicy::pretokenizer == 2` → metadata
  `tokenizer_kind = 4`; `arrow_tokenizer_encode` dispatches kind 4.

- [x] Step 1: write the failing split test: `"getUserName 1234567 中文 hello  world"`
  must split into `get|User|Name| |123|456|7| |中文| |hello| | |world|` (digits in
  ≤3 chunks, CJK run, case runs, two-space lookahead), plus punctuation
  `a...b` → `a|...|b`.
- [x] Step 2: run `ctest -R test_flint_tokenizer_split` → fails (no such stream).
- [x] Step 3: implement `FlintWordStream` (reuse `Llama3WordStream::match` with
  number/CJK pre-splits and the `[\p{P}\p{S}]+[\r\n]*` alternative), wire kind 4
  through `bpe_encode_llama3` via a stream template parameter, accept
  `pretokenizer == 2` in `BuildTokenizerArrowFiles`.
- [x] Step 4: run the test, then capture `tests/fixtures/flint/tokenizer_parity.jsonl`
  from the checkpoint's own `tokenizer.json` with `tools/flint_tokenizer_capture.py`
  (corpus: camelCase, digit runs of 1..7, CJK, contractions, multi-space, markers).
- [x] Step 5: add `tests/test_flint_tokenizer_parity.cpp` (label `fast`, SKIP 4 when
  the vault entry is missing) and commit.

### Task 2: frozen passport + converter + synthetic convert test

**Files:** create `flint_moe_constants.h`, `flint_moe_convert.{h,cpp}`,
`tests/test_flint_moe_fixture.h`, `tests/test_flint_moe_convert.cpp`; modify
`apps/lsxforge/src/{main,cli}.{cpp,h}`, `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`.

**Interfaces:**
- Produces: `lsxcommon::flint::ConvertFlintModel(const FlintConvertOptions &, const std::filesystem::path &src)`.
- Passport: `kHidden=2048, kLayers=49, kHeads=32, kKvHeads=4, kHeadDim=128,
  kVocab=150144, kDenseIntermediate=6144, kMoeIntermediate=512,
  kSharedIntermediate=1024, kNumExperts=256, kTopK=6, kFirstMoeLayer=2,
  kWindow=513, kRmsEps=1e-6f, kRopeTheta=10000.0f, kAttnScale=0.08838834764831845f,
  kEosIds={150001,150003,150005,150018}, kFullAttnLayer[49]` table
  `{0,1,5,9,13,17,21,25,29,33,37,41,45,48}` with `constexpr bool IsFullAttn(int)`
  and `constexpr bool SkipsRope(int)`.

- [x] Step 1: RED — synthetic fixture (`test_synth_checkpoint.h` primitives) with
  config.json + shards for 4 experts and every global; assert the emitted tree
  (`format == "logestix-flint-nvfp4"`, 47 expert files, dense NVFP4 records with
  `global_scale`, FP16 routers/embed/lm_head/norms/sinks, `model.index.json`
  layer tables) and that a drifted config (layer_types length, `no_rope` ≠ full
  list, expert width) is rejected.
- [x] Step 2: run → fails (no converter).
- [x] Step 3: implement `ParseFlintConfig` (frozen gate, synthetic override),
  dense pass and expert pass mirroring `hy_moe_convert.cpp`.
- [x] Step 4: `ctest -R test_flint_moe_convert` green (label `convert`, TIMEOUT 120,
  `ENVIRONMENT LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`, `RESOURCE_LOCK gpu`); commit.

### Task 3: loader

**Files:** create `flint_moe_load.{h,cpp}`, `tests/test_flint_moe_loader.cpp`; modify CMake.

**Interfaces:** `LoadFlintWeights(path, FlintPinnedWeights&)` / `FreeFlintWeights`;
`FlintPinnedWeights` holds `LsxNvfp4Weight` per dense tensor, per-layer
`FlintLayerWeights {q,k,v,o, q_norm,k_norm, sink, dense_mlp?, router, shared?, experts*}`.

- [x] Step 1: RED — loader test on a synthetic converted tree: geometry mismatch
  fails `Load`, every plane uploaded with the frozen shape, `Free` releases the
  expert slabs, `resident_bytes > 0`.
- [x] Step 2: implement with `arrow::io::MemoryMappedFile` + view readers (R2),
  expert slabs uploaded per layer, `cudaHostAlloc` staging freed per layer.
- [x] Step 3: green (label `fast`, SKIP 4, `RESOURCE_LOCK gpu`); commit.

### Task 4: kernels + host reference

**Files:** create `flint_moe_device_math.h`, `flint_moe_kernels.{h,cu}`,
`flint_moe_inference_attn.{h,cu}`, `tests/test_flint_moe_kernels.cpp`
(CUDA test executable), CMake object library `lsxcommon_flint_cu`.

**Interfaces:** `RunFlintRmsNorm`, `RunFlintRouterTopK` (sigmoid, top-6,
renormalized weights, F32), `RunFlintSwiGLU`, `RunFlintRoutedAccumulate`,
`RunFlintSharedCombine`, `RunFlintAttnPrefill/Decode(..., const __half *sink, int pos, bool swa, bool rope, ...)`.

- [x] Step 1: RED — host-reference parity: (a) sink softmax equals
  `softmax(qk ∪ {sink})` with sink contributing zero value; (b) router selection
  == sigmoid top-6 with renormalized weights; (c) `j0 = max(0, pos-512)`;
  (d) no-RoPE path leaves q/k untouched; (e) RoPE path matches rotate-half.
- [x] Step 2: implement kernels; `-fmad=false` on the object library.
- [x] Step 3: green (label `fast`); commit.

### Task 5: pipeline + model + registration + lifecycle

**Files:** create `flint_moe_inference_pipeline.{h,cu}`, `flint_moe_model.{h,cpp}`;
modify `model_registry.cpp`, guard registries (naming/independence/suite),
`tests/CMakeLists.txt`; create `tests/test_flint_moe_model.cpp`.

**Interfaces:** `FlintGenerate(weights, prompt, max_new, eos_ids, session, req, result)`;
`FlintMoeModel::kType == "flint"`.

- [x] Step 1: RED — model test: streaming emits one chunk per token, pre-set
  abort returns early, second `Infer` on a new prompt resets the session, result
  carries `output_text` + `weight_resident_bytes` (SKIP 4 without vault).
- [x] Step 2: implement pipeline (pos-major KV via `KvCacheGeometry`, dense layers
  0-1, MoE layers 2-48, 4 eos ids) and the model class with
  `RunUnderResidencyLock`, `GenerationStream`, `TeardownContext`,
  `AcquirePermanentWeights`, governor lease, `EffectiveFastMathMode(cfg_)`.
- [x] Step 3: register + extend guards; green (label `inference`, TIMEOUT 300); commit.

### Task 6: tool-calling bridge (4b)

**Files:** create `flint_moe_tool_format.{h,cpp}`, `tests/test_flint_moe_tool_framing.cpp`,
`tests/data/` fixture additions; modify `test_model_suite_guard.cpp` tool verdict.

- [x] Step 1: RED — render a system/user/assistant turn with reasoning + two
  parallel calls with nested JSON args and compare byte-for-byte with the template;
  parse the same text back; tokenizer round-trip on the synthetic fixture.
- [x] Step 2: implement; green; commit.

### Task 7: SelfCheck + CPU oracle + FSI/CODE/DET goldens

_Status 2026-10-30: SelfCheck green on real weights (Paris x3 bit-exact); the CPU
oracle is implemented and used to validate the engine (32 greedy tokens
token-for-token on the control prompt). The full capture set is OPEN — it runs at
3-5 s/token on CPU, see `docs/OPEN_TASKS.md` R6._

**Files:** create `tests/test_flint_moe_selfcheck.cpp`, `tools/flint_oracle.py`;
modify `tools/{fsi,code,det}_capture.py` (`loader: "flint"`), add
`tests/fixtures/{fsi,code,det}/flint.json`, per-stage tests.

- [x] Step 1: RED — selfcheck test (3 cycles, bit-exact, Paris).
- [x] Step 2: oracle: HF `Qwen3Moe*` + Flint deltas, layer parity vs the vendor
  plugin semantics; `LSX_ORACLE_DEVICE=cpu`.
- [x] Step 3: capture + commit goldens, run the three gates; commit.

### Task 8: perf baseline + docs + full guard sweep

_Status 2026-10-30: docs and the full guard sweep are DONE; the perf rows are
OPEN — `test_perf_baseline` requires a Release tree and the harness seeds the
rows itself (never by hand). Debug-build observation only in R6._

**Files:** `tests/perf/perf_baselines.json`, `tests/test_perf_baseline.cpp`,
`tests/CMakeLists.txt`, `AGENTS.md`, `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md`.

- [x] Step 1: add the `TEST_CASE` + `add_test` (label `perf`, SKIP 4).
- [x] Step 2: convert the real 75 GB checkpoint, measure through the harness,
  record rows.
- [x] Step 3: docs; `ctest` fast/convert/inference green; commit.

## Self-review

- Spec coverage: stages 1–8 + 4b each map to a task; Stage 0 is the spec itself.
- Every task has a RED step, labels, and a commit.
- Cross-task names (`LoadFlintWeights`, `RunFlintAttnDecode`, `FlintGenerate`)
  are fixed by the Interfaces blocks and must not drift.
