# Inkling-2-Mini Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task (AGENTS.md forbids subagents in this repo).
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt `thinkingmachines/Inkling-2-Mini` (`inkling2_loop`, looped MoE)
through all 9 Model Ingestion Protocol stages with real-weight gates.

**Architecture:** New model `inkling-2-mini` in family `inkling`, files
`inkling_moe_loop_*` modeled on the Inkling-2-Small implementation
(`inkling_moe_*`), with new loop machinery: 3 passes over layers [6,18),
per-pass KV/conv cache slots (48 total), rank-64 MessageGate controllers,
cross-pass message queries. Mixed expert codec preserved (NVFP4 0–19,
MXFP8 20–23). Manifest-driven registry; no hand-written guard rows.

**Tech Stack:** C++20/CUDA (sm_120 only), Arrow mmap, neutral ExpertStore,
lsx_nvfp4/lsx_gemv, Catch2, transformers 5.24.0 (`inkling2_loop`) as the
source reference at `/mnt/fast_storage/tmp/inkling2-mini/tf/x`.

**Spec:** `docs/superpowers/specs/2026-11-21-inkling-2-mini-ingest-design.md`
(geometry table, semantics list, decisions — every constant below comes from it).

## Global Constraints

- Branch `feat/inkling-2-mini-ingest`; scratch dir `/mnt/fast_storage/tmp/inkling2-mini`.
- TDD: failing test first, test + implementation + label in the SAME commit.
- Frozen geometry via `static_assert`; runtime topology cross-check fails Load on mismatch.
- spdlog only; no absl containers in headers reachable from .cu/.cuh; `StrFormat` is printf-style.
- NVFP4 GEMM only via `lsx_inference_nvfp4`; GEMM only via `lsx_gemv`; no atomicAdd feeding final_logits.
- One GPU workload at a time: check `nvidia-smi --query-compute-apps` before any GPU-labelled ctest run.
- Converter tests: synthetic checkpoints only, label `convert`, TIMEOUT ≤ 600 s, env `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` via test properties.
- Loader/model stages never read the real vault; selfcheck/parity/perf may (SKIP_RETURN_CODE 4).
- CUDA docs: read `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md` + `numeric-contract.md` before kernel tasks (operator-approved MCP mirror).
- Goldens/perf rows are never hand-edited to pass; perf seeded by running the harness.
- Commit prefixes: `feat:`/`test:`/`build:`/`docs:`; English messages.
- Naming: `loop` variant token must be added to `SchemeRegex()` in the same commit as the first `inkling_moe_loop_*` file.

## Review Focus

1. **Loop pass 2/3 state isolation** — a pass reading pass 1's KV/conv state instead of its own slot decodes fluent garbage; pinned by session/kernels tests (Task 6/7) and the 3-pass trace parity (Task 10).
2. **w13 row-interleave** — split-half keeps every byte but destroys knowledge (Small's week-long defect); converter test pins row pairing for BOTH codecs (Task 4).
3. **MessageGate order** — gates apply AFTER the AttnRes mix and BEFORE input norm; mlp gate after mlp_input mix; wrong order is silent (Task 7 kernel test vs host reference).
4. **Cross-pass query merge order** — q,mlp from `middle_cycle_cross_pass_queries`, k,v from `..._kv_queries`; a swap passes shape checks (Task 4 converter test).
5. **MXFP8 layers 20–23 assumed NVFP4** — dtype census drives the split, cross-checked against hf_quant_config.json; a uniform-codec converter decodes 1/6 of the stack as noise (Task 4).

---

### Task 1: Branch, stage-0 record, naming token

**Files:**
- Create: `docs/open-tasks/live/models/inkling-2-mini.md` (stage-0 triage record; content = spec's Stage 0 findings, status IN PROGRESS)
- Modify: `docs/OPEN_TASKS.md` (add live row)
- Modify: `tests/test_code_naming_guard.cpp` (`SchemeRegex()` var vocabulary: add `loop`; add a `m("inkling_moe_loop_constants")` positive case next to the existing inkling cases)

**Interfaces produced:** branch name; `loop` accepted by the naming guard.

- [ ] Step 1: `git checkout -b feat/inkling-2-mini-ingest` (workspace clean on main, verified).
- [ ] Step 2: Add the naming-guard positive case `REQUIRE(m("inkling_moe_loop_constants"));` — run `ctest --preset linux-debug -R test_code_naming_guard` → FAIL (regex rejects `loop`).
- [ ] Step 3: Add `loop` to the `var` alternation in `SchemeRegex()`.
- [ ] Step 4: Rebuild + rerun → PASS.
- [ ] Step 5: Write the stage-0 doc + OPEN_TASKS row; commit `test: accept the inkling loop variant token; stage-0 triage for inkling-2-mini`.

### Task 2: Frozen passport + constants test

**Files:**
- Create: `libs/lsxcommon/src/inkling_moe_loop_constants.h`
- Create: `tests/test_inkling_moe_loop_constants.cpp`
- Modify: `tests/CMakeLists.txt` (build wiring only, copy the `test_inkling_moe_base_constants` block)
- Create: `models/inkling-2-mini.json` (full manifest, all stages declared — see Task 9 for the final field values; write it NOW so `test_manifest_sync` drives every later task; targets whose sources don't exist yet make manifest validation fail, so initially declare only `model: [test_inkling_moe_loop_constants]` and grow the stages row-by-row as tasks land)

**Interfaces produced:** `namespace lsxcommon::inkling::loop::frozen` with:
`kModelType="inkling-2-mini"`, `kFormatTag="logestix-inkling-2-mini"`,
`kSourceModelType="inkling2_loop"`, `kTokenizerKind=2`,
`kHidden=2304, kLayers=24, kHeads=24, kKvHeads=8, kHeadDim=128, kGqaGroups=3`,
`kVocab=201024, kUnpaddedVocab=200058`,
`kExperts=128, kTopK=8, kMoeIntermediate=1152, kRouteScale=8.0`,
`kMlpGateSoftcap=4.0, kMlpUpSoftcap=25.0, kFinalLogitSoftcap=30.0`,
`kFullLayers={5,11,17,23}, kSlidingWindow=1024, kRelExtent=1024, kDRel=16`,
`kQkL2Clip=22.627416997969522, kRelL2Clip=8.0`,
`kLogScaleFloor=8192, kLogScaleAlpha=0.1, kRmsNormEps=1e-6, kConvKernel=4`,
`kAttnResBlock=6`,
`kLoopStart=6, kLoopEnd=18, kLoopPasses=3, kLoopGateRank=64`,
`kLoopLayers=kLoopEnd-kLoopStart` (12),
`kCacheLayers=kLayers+ (kLoopPasses-1)*kLoopLayers` (48),
`kNovfp4ExpertLayerCount=20, kMxfp8Block=32`,
`kEosId=200006, kPadId=199999, kStopIdAlt=199999, kBosId=-1`,
helpers `IsFullLayer(int)`, `IsLoopLayer(int)`, `CacheSlot(int layer, int pass)`
(= layer for pass 0; `kLayers + (pass-1)*kLoopLayers + (layer-kLoopStart)` for pass ≥ 1),
`static_assert`s: kCacheLayers==48, loop bounds are multiples of kAttnResBlock,
kLoopStart>0, codec split sums to kLayers, kHeads%kKvHeads==0.

- [ ] Step 1: Write `test_inkling_moe_loop_constants.cpp` asserting every constant above against the spec table, `CacheSlot(6,1)==24`, `CacheSlot(17,2)==47`, `IsLoopLayer(5)==false`, `IsLoopLayer(18)==false`, `IsFullLayer(23)==true`. Build wiring + manifest row (`model` stage only, `constants` target label `fast`).
- [ ] Step 2: Build; run → FAIL (header missing).
- [ ] Step 3: Write the header (comment block documents the loop semantics + mixed codec, mirroring `inkling_moe_constants.h` style but with Mini's numbers).
- [ ] Step 4: `ctest --preset linux-debug -R "test_inkling_moe_loop_constants|test_manifest_sync|test_code_naming_guard"` → PASS. Regenerate `docs/MODELS.md` (`python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`).
- [ ] Step 5: Commit `feat: inkling-2-mini frozen passport and manifest skeleton`.

### Task 3: Reference harness + loop semantics verification (host-side, no engine code)

**Files:**
- Create: `tools/inkling_loop_ref.py` (float64 per-layer reference from the artifact; extend `tools/inkling_layer_ref.py` patterns with loop scheduling, MessageGate, per-pass cache)
- Create: `tools/inkling_loop_transformers.py` (runs HF 5.24 `Inkling2LoopForCausalLM` on CPU with dequantized experts, greedy decode; PYTHONPATH=/mnt/fast_storage/tmp/inkling2-mini/tf/x)
- Create: `/mnt/fast_storage/tmp/inkling2-mini/probes/` outputs (not committed)

**Interfaces produced:** verified answers to the spec's "Semantics to verify" list; a source-side greedy generation for the Paris prompt + FSI/CODE/DET probe prompts (reused in Task 10); `tools/inkling_loop_transformers.py --prompt ... --max-gen N --out json` CLI.

- [ ] Step 1: Run the HF reference on "The capital of France is" (framed per chat template) with full dequantized weights on CPU; record top-1 continuation and logits hash inputs. Expected: coherent text containing "Paris" (verify; if the model answers differently, record the ACTUAL answer — the probe is data, not a constant).
- [ ] Step 2: Verify cross-pass query merge order + MessageGate math against `Inkling2LoopMergeMessageQueries` and `Inkling2LoopMessageGate` source; write findings into the stage-0 doc.
- [ ] Step 3: Verify the NVFP4 invariant `6*max(scale)*amax/2688 == amax` and the MXFP8 ue8m0 per-32 layout on Mini's real shards (same check as Small's converter header).
- [ ] Step 4: Commit `docs: inkling-2-mini loop semantics verified against transformers 5.24` (tools + stage-0 update).

### Task 4: Converter (stages 1–2)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_loop_convert.h` (`arrow::Status ConvertInklingLoopModel(const std::string &input_dir, const std::string &output_dir);`)
- Create: `libs/lsxcommon/src/inkling_moe_loop_convert.cpp`
- Create: `tests/test_inkling_moe_loop_fixture.h` (re-exports `synthtest::` primitives; synthetic Mini geometry 8 layers × 4 experts, loop [2,6) passes 2 — scaled-down but structurally complete)
- Create: `tests/test_inkling_moe_loop_convert.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add source), `tests/CMakeLists.txt` (build), `models/inkling-2-mini.json` (`convert` stage + override timeout 300, env synthetic)
- Modify: `apps/lsxforge/src/cli.h`/`cli.cpp`/`main.cpp` (`--inkling_loop` flag, `Pipeline::kInklingLoop`)

**Interfaces produced:** converted tree layout identical in shape to Small's
(`pinned/global.arrow`, `pinned/global_nvfp4.arrow`, `pinned/layer_<l>.arrow`,
`pinned/layer_<l>_nvfp4.arrow`, `experts/layer_<l>/exp_<e>.arrow`,
`model.index.json` with `"model": "Inkling-2-Mini"`, format tag, loop fields
`loop_start/loop_end/loop_passes/loop_gate_rank`, codec lists, tokenizer ids),
plus NEW pinned tensors: per-loop-layer controller triples
(`q/k/v/mlp_message_gate.{down,forget,gain}`) and merged
`message_queries [kLoopLayers][4][hidden]` FP32. Dense projections stay FP16
planes (as Small); embed/unembed FP16; norms FP16; queries/router-bias/controllers FP32
(controllers' forget/gain are FP32 in the reference; down is BF16→FP16).

- [ ] Step 1: Write the synthetic E2E test: globals present; BF16→FP16 dense; NVFP4 passthrough with per-expert global scale; MXFP8 layers with ue8m0 scales; **w13 de-interleave pinned row-by-row for both codecs** (gate row r == source row 2r, up row r == 2r+1, weights AND scales); attn_queries offset-by-1 rename; controllers emitted per loop layer; message queries merged in verified order (q,mlp / k,v); vision+MTP stripped and counted; index fields; leftover unknown tensor → error. Run → FAIL (converter missing).
- [ ] Step 2: Implement the converter following `inkling_moe_convert.cpp` structure (config parse via simdjson, frozen gates relaxed by `synthetic_geometry.h`, dtype-census codec split cross-checked vs hf_quant_config.json, `lsx_model_models_dir()` output root, atomic Arrow writes).
- [ ] Step 3: Wire lsxforge flag; build; `ctest --preset linux-debug -R test_inkling_moe_loop_convert` → PASS; guards (`manifest_sync`, `ctest_labels`, `naming`) → PASS.
- [ ] Step 4: Commit `feat: inkling-2-mini converter with loop controllers and mixed-codec experts`.

### Task 5: Loader

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_loop_load.h`
- Create: `libs/lsxcommon/src/inkling_moe_loop_load.cpp`
- Create: `tests/test_inkling_moe_loop_loader.cpp`
- Modify: CMake (lib+test), manifest (`load` stage + synthetic env)

**Interfaces produced:** mirror Small's loader API with `InklingLoop` prefix:
`InklingLoopTopology` (adds `loop_start/loop_end/loop_passes/loop_gate_rank`,
`CacheSlot()`), `InklingLoopExpertEntry`/`InklingLoopExpertDb` (neutral
`ExpertStore`, same 18-field view), `InklingLoopLayerWeights` (adds
`__half *msg_down[4]` FP16 [64,4608], `float *msg_forget[4]`, `float *msg_gain[4]`
[2304,64] — loop layers only), `InklingLoopPinnedWeights` (adds
`float *message_queries` [12][4][2304]),
`LoadInklingLoopTopology`, `InklingLoopTopologyMatchesFrozen`,
`LoadInklingLoopPinnedWeights`, `FreeInklingLoopPinnedWeights` (frees EVERY
plane incl. controllers), `LoadInklingLoopExpertIndex`,
`EnsureInklingLoopExpertResident`, `InklingLoopExpertWindowFloor`.

- [ ] Step 1: Failing loader test on the Task-4 synthetic converted fixture: topology fields incl. loop; frozen mismatch rejected (perturb hidden); every pinned tensor uploaded with right shape; controllers null on non-loop layers, present on loop layers; Free releases all (cuda mem delta ≈ 0); expert store views in-range. Run → FAIL.
- [ ] Step 2: Implement loader (mmap-only reads R2; views retained, no UnpackBatch past upload).
- [ ] Step 3: `ctest -R "test_inkling_moe_loop_loader|expert_store_guard"` → PASS.
- [ ] Step 4: Commit `feat: inkling-2-mini loader with per-pass cache topology`.

### Task 6: Session + scratch (48 cache slots)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_loop_inference_pipeline.h`
- Create: `tests/test_inkling_moe_loop_kernels.cu` (session cases first; kernels cases added in Task 7)
- Modify: CMake (test OBJECT lib `test_inkling_loop_cu`, `-fmad=false`), manifest (`kernels` stage, resource_lock gpu)

**Interfaces produced:** `InklingLoopSession` — KV via `KvCacheGeometry`
(pos-major), `kCacheLayers` slots; conv state `[kCacheLayers][4 convs]` FP32;
`EnsureCapacity(top, want, stream)` copies `kv_rows * per-layer row bytes`
(NOT LayerStride-with-capacity — Small's shipped defect, pinned red-first);
`Reset()` zeroes conv state + kv_rows + gen_tail. `InklingLoopScratch` extends
Small's arena with: `float *msg[4*hidden]` (retrieved messages),
`float *gate_h[2*hidden]`, `float *gate_rank[64]`, `float *msg_stack[...]`
(previous pass block states), pre-loop hidden/block snapshots.

- [ ] Step 1: Failing tests: session growth keeps every slot's rows intact (write pattern per slot, grow, verify — the EnsureCapacity trap); Reset zeroes all 48 slots' conv state; geometry reports pos-major; scratch Allocate/Free round-trip. Run → FAIL.
- [ ] Step 2: Implement session + scratch (allocation-site byte counting for `resident_bytes`).
- [ ] Step 3: `ctest -R test_inkling_moe_loop_kernels` → PASS (GPU idle check first).
- [ ] Step 4: Commit `feat: inkling-2-mini session with per-pass KV and conv slots`.

### Task 7: CUDA kernels (stage 5–6)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_loop_kernels.h`
- Create: `libs/lsxcommon/src/inkling_moe_loop_kernels.cu`
- Create: `libs/lsxcommon/src/inkling_moe_loop_inference_pipeline.cu`
- Modify: `tests/test_inkling_moe_loop_kernels.cu` (add kernel cases), lib CMake (OBJECT lib `lsxcommon_inkling_loop_cu`), docs mirror read first

**Interfaces produced:** kernel set mirroring Small's (rel-bias gather, L2 clip,
sink softmax, gated output, static/dynamic sconv, AttnRes mix, softcapped
SwiGLU MoE accumulate, NVFP4/MXFP8 expert GEMV via neutral layers, logits with
per-tensor global_scale and softcap 30) PLUS:
- `inkling_loop_message_gate_kernel` — `(1-tanh(forget(down)))*h + tanh(gain(down))*msg`, down input = concat(unweighted-RMSnorm(h), RMSnorm(msg)); FP32 accumulation; deterministic reduction (thread-0 RMW behind __syncthreads, no atomicAdd).
- `inkling_loop_message_query_kernel` — single-head softmax attention of `message_queries[layer][m]` over previous-pass block stack + final hidden (unweighted RMSnorm on states, scale hidden**-0.5), 4 messages per loop layer.
- `RunInklingLoopGeneration(...)` — same signature shape as Small's plus loop scheduling: for each token, pre-loop layers → pass 0 loop layers → passes 1,2 (restart from snapshots, retrieve messages, CacheSlot offset) → post-loop → final mix/norm/logits; one stream, no per-token cudaDeviceSynchronize; `GenerationStream` Emit per token; abort_flag polled between layers.
- `RunInklingLoopForwardTrace(tokens, out_states)` — 49 rows (post-embed + 48 applications) FP32 for parity.

- [ ] Step 1: Failing kernel tests vs host float references: message gate (rank-64 path, tanh both gates, norm-before-down order); message query retrieval (softmax over stack, 4 queries); rel-bias/clip/sconv/AttnRes parity at Mini geometry (reuse Small's test SHAPES, own fixture); MoE softcap accumulate; logits global_scale ≠ 1 fixture; MXFP8 GEMV vs host decode. Run → FAIL.
- [ ] Step 2: Implement kernels + pipeline (shared-memory sandbox rules; NVFP4 only via lsx_nvfp4; CCCL/cub for reductions).
- [ ] Step 3: `ctest -R test_inkling_moe_loop_kernels` → PASS.
- [ ] Step 4: Commit `feat: inkling-2-mini CUDA kernels and 3-pass generation pipeline`.

### Task 8: Model class + factory + token bridge + tool format (stages 3, 4, 4b)

**Files:**
- Create: `libs/lsxcommon/src/inkling_moe_loop_model.h` / `.cpp` (class `InklingMoeLoopModel final : ModelBase`, kType "inkling-2-mini"; Load/Infer/SelfCheck/Unload/Inspect/Tokenizer; BuildPromptInputIds frames the native user turn; lifecycle contract: RunUnderResidencyLock, GenerationStream Emit, abort, TeardownContext, AcquirePermanentWeights/AcquireExpertCacheLease, InferenceResult fill; LsxFastMathScope(EffectiveFastMathMode(cfg_)) in Infer; TraceForward)
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_loop_tool_format.h` + `src/inkling_moe_loop_tool_format.cpp` (verified Small-identical semantics: role markers, tool_declare XML block, thinking-effort line, `<|content_invoke_tool_json|>{"name":...,"args":{...}}` parse; markers special→token-id scan)
- Create: `tests/test_inkling_moe_loop_model.cpp` (synthetic converted fixture; streaming via tests/test_streaming_assert.h; pre-set abort case; frozen mismatch fails Load)
- Create: `tests/test_inkling_moe_loop_tool_framing.cpp` + `tests/data/synthetic_inkling_loop_tok.json` (own fixture with the sentinel added_tokens; render structure, parse, round-trip; NOT from the full framed prompt)
- Modify: lib CMake, tests CMake, manifest (`model` stage grows: constants/model/tool_framing; tool_framing label fast), `tests/test_model_suite_guard.cpp` tool-verdict table row `{"inkling-2-mini", {"test_inkling_moe_loop_tool_framing"}}`
- Factory registration is GENERATED from the manifest (`model_registry.inc`) — verify, do not hand-write.

**Interfaces produced:** `InklingMoeLoopModel` registered under "inkling-2-mini"; `BuildConversationInputIds`/`ParseToolCalls` overrides; framing helpers in `lsxcommon::inkling::loop::tools`.

- [ ] Step 1: Failing tool-framing test (assert against Mini's chat_template.jinja rendered by jinja2 in a one-off script; pin the exact marker sequence incl. effort line "Thinking effort level: 0.9"). Run → FAIL.
- [ ] Step 2: Implement tool format; run → PASS.
- [ ] Step 3: Failing model test (Load synthetic fixture, Infer streams ≥1 token per Emit, abort pre-set → 0 tokens, mismatch passport → Load false). Run → FAIL.
- [ ] Step 4: Implement model class; run → PASS; guards (independence, suite, manifest_sync) → PASS.
- [ ] Step 5: Commit `feat: inkling-2-mini model class, factory registration and native tool bridge`.

### Task 9: Real conversion + SelfCheck (stage 8 part 1)

**Files:**
- Create: `tests/test_inkling_moe_loop_selfcheck.cpp` (own TU like Small's: Paris ≥3 bit-exact cycles, env `LSX_TEST_INKLING_2_MINI_SELFCHECK_CYCLES`, knowledge probe, SKIP 4 when vault absent)
- Modify: tests CMake, manifest (`selfcheck` stage, form selfcheck, timeout 3600; probe fields `selfcheck_frame_prompt: true`, `selfcheck_bit_exact: true`, `selfcheck_max_gen` = MEASURED from Task 3's source run, not copied)
- Vault: run `lsxforge --inkling_loop` on the real checkpoint → `<vault>/logestix/`

**Interfaces produced:** converted real tree; measured Paris probe (top-1 " Paris" or the recorded actual answer), logits_hash, selfcheck_max_gen.

- [ ] Step 1: Failing selfcheck test (vault absent → SKIP 4; with vault → runs). Build; run → SKIP (expected red-equivalent for a gate: absence is not a pass).
- [ ] Step 2: Convert the real model (GPU idle check; ~20 GB read, minutes).
- [ ] Step 3: Run selfcheck → if Paris fails, follow the parity decision procedure (teacher-force fork, bug checklist, residual plane last resort). Iterate until 3 bit-exact cycles PASS.
- [ ] Step 4: Commit `feat: inkling-2-mini selfcheck gate green on real weights`.

### Task 10: FSI/CODE/DET goldens + gates (stage 8 part 2)

**Files:**
- Create: `tests/fixtures/{fsi,code,det}/inkling-2-mini.json` (captured via `tools/fsi_capture.py` / `tools/code_capture.py` / det capture per the three skills, source = HF 5.24 inkling2_loop CPU run from Task 3; `source` field records provenance)
- Modify: tests CMake (three `test_parity_gate.cpp` binaries with LSX_GATE_MODEL/KIND defines), manifest (`fsi`/`code`/`det` stages, form parity, timeout 3600)
- Load skills `fsi-selfcheck`, `code-selfcheck` before capture (det follows the same driver).

- [ ] Step 1: Capture goldens from the SOURCE (not the engine); commit fixtures.
- [ ] Step 2: Wire gate targets; run each → PASS or follow residual-plane procedure; a SKIP is not a pass.
- [ ] Step 3: Update `docs/SELFCHECK_MATRIX.md` row.
- [ ] Step 4: Commit `test: inkling-2-mini FSI/CODE/DET parity gates with source-derived goldens`.

### Task 11: Perf baseline (stage 8 part 3)

**Files:**
- Modify: `tests/test_perf_baseline.cpp` (add `TEST_CASE("perf baseline: inkling-2-mini", "[perf]")` next to the inkling cases)
- Modify: manifest (`perf` stage: target `test_perf_baseline_inkling_2_mini`, vault_subdir "Inkling-2-Mini", timeout 7200)
- Generated: `tests/perf/perf_baselines.json` row — seeded by RUNNING the target on a Release build (linux-release-znver4), never hand-edited.

- [ ] Step 1: Add TEST_CASE + manifest row; `ctest -R test_perf_baseline_logic` → PASS (pairing enforced).
- [ ] Step 2: Release build; run `test_perf_baseline "perf baseline: inkling-2-mini"` (2048 tokens, both modes, GPU idle) → row seeded with ttft/power.
- [ ] Step 3: Commit `test: inkling-2-mini perf baseline seeded through the harness`.

### Task 12: Full lifecycle verification + report (stage 9)

**Files:**
- Create: `docs/model-reports/2026-11-XX-Inkling-2-Mini.md` (date = real-gate measurement day; publish-ready post + evidence per the skill)
- Modify: `docs/model-reports/README.md` index, `docs/MODELS.md` (regenerated), stage-0 doc → COMPLETE, `docs/OPEN_TASKS.md` row → resolved
- Modify: manifest `report` field, `notes`, probe fields final values

- [ ] Step 1: Full sweeps: `ctest --preset linux-debug` (fast), `-R convert` sweep, inference sweep, perf; ALL guards green; `python3 tools/gen_model_manifest.py --strict --check` clean.
- [ ] Step 2: Write the report with measured numbers only; SKIP/debt stays debt.
- [ ] Step 3: Commit `docs: inkling-2-mini ingestion report and catalogue`.

---

## Self-review notes

- Spec coverage: stages 0–9 → Tasks 1–12; loop semantics verification → Task 3; R1/R2 → Task 5; lifecycle contract → Task 8; probe-is-data → Tasks 3/9.
- Type consistency: `InklingLoop*` prefix used throughout Tasks 5–8; `ConvertInklingLoopModel` in Task 4 consumed by Task 9's forge flag.
- The manifest grows stage-by-stage (Task 2 skeleton) because validation refuses targets without sources — this is the registry-driven equivalent of "label in the same commit".
- Proportion: bodies omitted where Small's implementation is the stated pattern; only Mini-specific deltas are spelled out.