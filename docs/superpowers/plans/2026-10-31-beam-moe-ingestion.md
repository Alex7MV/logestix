# Beam-1.0-501B-A23B ingestion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.
> Subagents are FORBIDDEN by AGENTS.md — execute inline in the main session.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** ingest `reflection/Beam-1.0-501B-A23B` as the independent family
`beam`: converter, NVFP4 tree, streamed expert store, gated GQA with
SWA/NoPE-SSMax hybrid, depth-scaled residual + sandwich norms, group-limited
sigmoid-router MoE, tokenizer bridge, Harmony tool codec, and every lifecycle
gate that this host can execute (SelfCheck + per-layer oracle; FSI/CODE/DET
declared debt).

**Architecture:** loader/generation template `ling`/`deepseek_pro` (mmap
`ExpertStore` + device slot LRU + token-by-token session); kernel/norm/attention
template `flint` (hybrid SWA + NoPE full layers + per-head qk-norm + FP16
residual). Neutral NVFP4 GEMV and routed-FFN layers are used unchanged.

**Tech Stack:** C++20/C23, CUDA 13.4 (sm_120/121), CMake 3.25+, Catch2 v3.16.1,
Arrow 26, PyTorch CPU oracle.

## Global constraints

- Frozen geometry spec: `docs/superpowers/specs/2026-10-31-beam-moe-ingestion-design.md`.
- `static_assert` every frozen relation; no runtime geometry reads.
- spdlog only; NVFP4 only through `lsx_inference_nvfp4`; no absl containers in
  `.cu`/`.cuh`-reachable headers.
- Every new test gets its ctest label in the same commit; selfcheck/fsi/code/det
  targets carry `SKIP_RETURN_CODE 4`.
- Guard registries (`test_code_naming_guard`, `test_model_independence_guard`,
  `test_model_suite_guard`, `test_ctest_labels_guard`) updated in the same commit.
- Commit prefixes `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`; English.
- No subagents. Temp files in `/mnt/fast_storage/tmp`; vault in
  `/mnt/fast_storage/huggingface`.

## File structure

| File | Responsibility |
|---|---|
| `libs/lsxcommon/src/beam_moe_constants.h` | frozen passport, layer tables, static_asserts |
| `libs/lsxcommon/include/lsxcommon/beam_moe_convert.h` | `BeamTopology`, `ConvertBeamModel` |
| `libs/lsxcommon/src/beam_moe_convert.cpp` | config + safetensors → NVFP4 tree |
| `libs/lsxcommon/include/lsxcommon/beam_moe_load.h` | pinned weights + `LoadBeamWeights` |
| `libs/lsxcommon/src/beam_moe_load.cpp` | Arrow mmap → device graph + `ExpertStore` |
| `libs/lsxcommon/include/lsxcommon/beam_moe_device_math.h` | host reference (norms, router, residual, window) |
| `libs/lsxcommon/include/lsxcommon/beam_moe_kernels.h` + `src/beam_moe_kernels.cu` | norm/qk-norm/rope/ssmax/router/swiglu/residual/softcap |
| `libs/lsxcommon/include/lsxcommon/beam_moe_inference.h` + `src/beam_moe_inference.cu` | KV session, layer graph, generation |
| `libs/lsxcommon/src/beam_moe_model.{h,cpp}` | `IModel`, framing, streaming, lifecycle |
| `libs/lsxcommon/src/beam_moe_tool_format.{h,cpp}` | Harmony-XML codec |
| `libs/lsxcommon/src/pretokenizer.cpp`, `tokenizer.cpp`, `tokenizer_convert.cpp` | `tokenizer_kind = 6` |
| `apps/lsxforge/src/{main,cli}.{cpp,h}` | `--beam` pipeline |
| `tests/test_beam_moe_*.cpp`, `tests/test_beam_*_selfcheck.cpp`, `tests/test_beam_moe_fixture.h` | lifecycle tests |
| `tools/beam_layer_oracle.py`, `tools/beam_tokenizer_capture.py` | CPU oracle + parity capture |

---

### Task 1: `tokenizer_kind = 6` (Beam pre-tokenizer) + parity

- [x] RED: `tests/test_beam_tokenizer_split.cpp` — Qwen single-digit split:
  `"getUserName 1234567 中文 hello  world"` → `get|User|Name| |1|2|3|4|5|6|7| |中文| |hello| | |world|`
  plus `a...b` → `a|...|b` and `"a\n\nb"` → `a|` `\n\n` `|b`.
- [x] Implement `BeamWordStream` + `tokenizer_kind = 6` dispatch; `pretokenizer == 4`.
- [x] Capture `tests/fixtures/beam/tokenizer_parity.jsonl` from the checkpoint
  tokenizer; add `tests/test_beam_tokenizer_parity.cpp` (label `fast`, SKIP 4).

### Task 2: frozen passport + converter + synthetic convert test

- [x] RED: `test_beam_moe_convert.cpp` on a synthetic checkpoint:
  format tag, 49×4 expert files, dense NVFP4 records, FP16/F32 raw planes,
  layer tables, drifted config rejected.
- [x] Implement `ConvertBeamModel` (flint converter pattern; expert-major pass,
  w1=gate, w2=down, w3=up), `LSX_BEAM_SKIP_EXPERTS`, `--beam` CLI.
- [x] Label `convert`, TIMEOUT 120, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`.

### Task 3: loader

- [x] RED: `test_beam_moe_loader.cpp` — geometry gate, every plane uploaded with
  frozen shape, `ExpertStore` first_moe_layer=3, `Free` releases everything,
  `resident_bytes > 0`.
- [x] Implement with `MemoryMappedFile` + view readers (R1/R2); experts through
  `gpu::expert_store::ExpertStore<BeamExpertEntry>`; dense router kept F32.

### Task 4: kernels + host reference

- [x] RED: `test_beam_moe_kernels.cpp` against `beam_moe_device_math.h`:
  plain RMSNorm, per-head qk-norm, rotate-half RoPE, NoPE SSMax scale,
  window bounds, sigmoid group-limited router (bias/group/renorm),
  depth-scaled residual, gate bias broadcast, softcap.
- [x] Implement kernels; `-fmad=false` on the object library.

### Task 5: pipeline + model + registration + lifecycle

- [x] RED: `test_beam_moe_model.cpp` — streaming one Emit/token, pre-set abort,
  session reset, result fields (SKIP 4 without vault).
- [x] Implement pos-major KV session, dense layers 0-2, MoE 3-51, streamed
  experts with host-handshake staging (documented perf debt), 2 stop ids.
- [x] Register `kType="beam"`; extend the four guard registries.

### Task 6: tool-calling bridge (Harmony XML)

- [x] RED: `test_beam_tool_framing.cpp` — render/parse system+developer, analysis
  and final channels, `to=name` tool call with `rfl:parameter` bodies, parallel
  calls; tokenizer round-trip on a synthetic fixture.
- [x] Implement `beam_moe_tool_format.{h,cpp}`; suite-guard verdict row.

### Task 7: SelfCheck + numeric oracle + parity gates

- [x] RED: `test_beam_moe_selfcheck.cpp` (3 cycles, bit-exact, Paris, SKIP 4).
- [x] Numeric instrument: host-reference parity suite (`beam_moe_device_math.h`)
  per kernel + SelfCheck/determinism on the real weights (a per-layer CPU
  oracle over the 934 GB source is not runnable in 750 GB RAM).
- [x] FSI/CODE/DET: declared golden debt (source cannot run in 750 GB), recorded
  in `docs/OPEN_TASKS.md` + `docs/SELFCHECK_MATRIX.md`.

### Task 8: perf baseline + docs + guard sweep

- [x] perf `TEST_CASE` + `add_test` (label `perf`, SKIP 4); seed through the
  harness after the real conversion.
- [x] AGENTS.md Structure, `docs/OPEN_TASKS.md`, `docs/SELFCHECK_MATRIX.md`.
- [x] `ctest` fast/convert/inference green; guard tests green.

### Task 9: model report artifact (stage 9)

**Files:** create `docs/model-reports/2026-10-31-Beam-1.0-501B-A23B.md` +
`docs/model-reports/README.md`; modify `AGENTS.md` and the
`ingesting-a-model` skill (stage 9 rule + definition-of-done).

- [x] Publish-ready English post (what the checkpoint forced, the two artifact
  traps, measured numbers, what was not measured and why).
- [x] Evidence section: gates actually run, defects fixed, internal references.
- [x] Rule added to the ingestion protocol (closing stage 9) so every future
  model produces the same artifact.

## Self-review

- Stage 0 → spec; stages 1–9 + 4b → tasks 2–9; TDD RED step in every task.
- Cross-task names fixed: `BeamTopology`, `ConvertBeamModel`,
  `LoadBeamWeights`, `RunBeam*`, `BeamGenerate`, `BeamMoeModel::kType`.
