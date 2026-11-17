# Inkling (base) ingestion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline; AGENTS.md forbids subagents). Steps use checkbox syntax. TDD per task: failing test first, watch it fail, minimal code, watch it pass, commit.

**Goal:** ingest `/mnt/fast_storage/huggingface/Inkling` (952B/28B-active, arch `inkling`) end to end: converter → NVFP4 tree → model class → tokenizer/tool bridge → CUDA kernels → all lifecycle gates → report.

**Architecture:** new family member `kType="inkling"`, files `inkling_moe_base_*`, namespace `lsxcommon::inkling::base` (passport `lsxcommon::inkling::base::frozen`). Plain-residual hybrid decoder: rel-bank attention (no RoPE), 4 static sconv/layer, MoE 256×top-6 + 2 shared-sink experts. Source is pure BF16 → standard NVFP4 quantization (dense + routed), norms/biases/rel-banks/sconv/router/embed stay high precision. Routed experts (~486 GiB) stream via neutral `ExpertStore` mmap views.

**Tech stack:** C++20, CUDA 13.5/sm_120, Arrow, simdjson, Catch2, neutral layers `lsx_nvfp4`/`lsx_gemv`/`ExpertStore`/`KvCacheGeometry`.

**Spec:** `docs/superpowers/specs/2026-11-17-inkling-base-ingest-design.md` (stage-0 numbers are frozen there; audited shapes in `/mnt/fast_storage/tmp/inkling_shapes.json`).

## Global Constraints

- Branch `feat/inkling-base-ingest`; worktree `/home/alexmv2025/.local/share/opencode/worktree/b5f640/feat/inkling-base-ingest`; temp `/mnt/fast_storage/tmp`; CUDA docs mirror `/home/alexmv2025/projects/cuda/docs` (read before kernel work).
- TDD: no production code without a failing test watched first; label in same commit.
- One GPU workload at a time: check `nvidia-smi --query-compute-apps` before any GPU-labelled ctest.
- Frozen geometry constants + static_assert; Load fails on passport mismatch.
- No absl containers in .cu-reachable headers; spdlog-only logging; printf-style StrFormat.
- Arrow reads mmap-only (`LoadTensorBatchViews`/`ExpertStore`); no owning UnpackBatch past upload scope.
- No hand-written registries: `models/inkling.json` + generator; guards read generated tables. Hand edits allowed ONLY: `SchemeRegex` variant token `base`, `AllowedCommon` (tool_format stem), independence-guard `consumers` list, suite-guard tool-verdict row, forge CLI flag, `test_perf_baseline.cpp` PerfModel+TEST_CASE.
- Convert test: synthetic checkpoint, label `convert`, TIMEOUT ≤ 600, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` via ENV.
- Loader/model tests: synthetic converted fixtures, never the vault.
- Goldens never rewritten to pass; perf rows seeded only by running the harness (Release).

## Review Focus

1. w13 slab layout: routed `mlp.experts.w13_weight` is HALVES [gate;up] (no Interleave op in `inkling_mm_model` mapping) while `shared_w13_weight` and dense `w13_dn` are ROW-INTERLEAVED (Interleave+Chunk). A halves-split on shared/dense (or de-interleave on routed) is byte-plausible garbage. → Task 2 fixture pins row identity by reconstruction, Task 8 knowledge gate is the oracle.
2. Attention scale is 1/head_dim = 1/128 (NOT 1/√128), q/k per-head weighted RMS-norm, no sinks/gates/L2-clip. Copying inkling-2 kernels halves logits silently. → Task 4 host-reference tests.
3. sconv is `conv(x)+x` with NO 1/√(k+1) normalizer and no dynamic taps; conv state is per-sequence and must reset per request. → Task 4/5 tests (two identical requests decode identically).
4. Router joint renorm: softmax over logsigmoid of the 6 selected routed + 2 shared logits (8-way), ×8.0×gate.global_scale; bias selection-only over 256; shared experts get their own 2 gammas. → Task 4 router host-reference test with asymmetric logits.
5. muP head: hidden/24.0 BEFORE lm_head, no softcap, logits sliced to 200058; embed_norm right after embed. → Task 5 pipeline test vs host reference.
6. KV heterogeneity: sliding layers 16 kv heads/window 512, full layers 8 kv heads/extent 1024 — per-layer geometry, growth copies `kv_rows*pos_stride` (not LayerStride). → Task 5 capacity-growth test.

---

### Task 1: branch baseline + frozen passport + naming token

**Files:**
- Create: `libs/lsxcommon/src/inkling_moe_base_constants.h` (namespace `lsxcommon::inkling::base::frozen`)
- Create: `tests/test_inkling_moe_base_constants.cpp`
- Modify: `tests/test_code_naming_guard.cpp` (var regex: add `base`)
- Modify: `models/` — NOT yet (manifest lands in Task 3 when class exists)

**Interfaces (produces):** `base::frozen::` constants used by every later task:
`kModelType="inkling"`, `kFormatTag="logestix-inkling-nvfp4"`, `kSourceModelType="inkling_mm_model"`, `kTokenizerKind=2`,
`kHidden=6144, kLayers=66, kHeads=64, kHeadDim=128, kKvHeadsFull=8, kKvHeadsSliding=16, kVocab=201024, kBaseVocab=200058`,
`kDRel=16, kRelExtent=1024, kSlidingWindow=512, kConvKernel=4, kRmsNormEps=1e-6, kAttnScaleDenom=128.0`,
`kFullLayerCount=11, kFullLayers={5,11,17,23,29,35,41,47,53,59,65}`, `IsFullLayer(l)`,
`kDenseMlpLayers=2` (layers 0,1 dense; 2..65 MoE), `kDenseIntermediate=24576`,
`kExperts=256, kTopK=6, kSharedExperts=2, kMoeIntermediate=3072, kRouteScale=8.0`,
`kLogScaleFloor=128000, kLogScaleAlpha=0.1, kLogitsMupDiv=24.0`,
`kQuantBlock=16`,
`kBosId=-1, kPadId=-1, kEosId=200006, kStopIdAlt=199999`, marker ids identical to inkling-2 (verified from tokenizer_config 2026-11-17): `kMarkerUser=200000, kMarkerModel=200001, kMarkerSystem=200002, kMarkerTool=200003, kMarkerText=200004, kMarkerEndSampling=200006, kMarkerThinking=200008, kMarkerEndMessage=200010, kMarkerXml=200024, kMarkerInvokeJson=200049`.
static_asserts: kHeads%kKvHeadsFull==0, kHeads%kKvHeadsSliding==0, kHidden==kHeads*kHeadDim, kMoeIntermediate%16==0, kDenseIntermediate==2*kHidden*2 (24576), kLayers==66.

- [ ] Step 1: write `tests/test_inkling_moe_base_constants.cpp` mirroring `test_inkling_moe_constants.cpp` structure with the values above (identity, geometry, hybrid table, MoE, muP/no-softcap facts, marker ids). Include the passport via relative path.
- [ ] Step 2: register target temporarily via `lsx_add_test(NAME test_inkling_moe_base_constants ... LABEL fast)` in tests/CMakeLists.txt; build; run → FAIL (header missing). (Manifest takes ownership in Task 3; remove the manual registration then.)
- [ ] Step 3: add `base` to the variant alternation in `SchemeRegex()`; write the passport header.
- [ ] Step 4: build; run test + `test_code_naming_guard` → PASS.
- [ ] Step 5: commit `feat: inkling base frozen passport + naming token`.

### Task 2: converter (stages 1–2) + synthetic E2E test

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_base_convert.h`, `libs/lsxcommon/src/inkling_moe_base_convert.cpp`
- Create: `tests/test_inkling_base_fixture.h` (namespace `inklingbasetest`, primitives from `test_synth_checkpoint.h`)
- Create: `tests/test_inkling_moe_base_convert.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add source), `apps/lsxforge/src/{main,cli}.cpp` + forge header (`--inkling-base` flag, Pipeline::kInklingBase)

**Interfaces (produces):**
```cpp
namespace lsxcommon::inkling::base {
struct ConvertTopology { /* parsed config fields incl. full/sliding lists,
  dense_mlp_idx, skipped_visual/audio/mtp counts, error_msg */ };
ConvertTopology ParseConfig(const std::string &input_dir);
enum class TensorKind { Unknown, Vision, Audio, Mtp, Embed, Unembed, EmbedNorm,
  FinalNorm, Wq, Wk, Wv, Wo, Wr, QNorm, KNorm, RelProj, KSconv, VSconv,
  AttnSconv, MlpSconv, AttnNorm, MlpNorm, DenseW13, DenseW2, DenseScale,
  ExpertW13, ExpertW2, SharedW13, SharedW2, RouterGate, RouterBias, RouterScale };
ParsedTensor ClassifyTensor(const std::string &name); // layer index too
arrow::Status ConvertInklingBaseModel(const std::string &input_dir,
                                      const std::string &output_root);
}
```
Artifact layout (`<out>/logestix/`): `pinned/global.arrow` (embed/unembed/norms raw BF16), `pinned/layer_<l>.arrow` (norms, q/k_norm, rel bank, 4 sconv taps, router weight F32-cast? keep source dtype BF16/F32 raw, router bias F32, mlp.global_scale, gate.global_scale), `pinned/layer_<l>_nvfp4.arrow` (wq/wk/wv/wo/wr; dense w13→`mlp.gate_proj.weight`+`mlp.up_proj.weight` de-interleaved then QuantizeSlice each; dense w2; shared → `mlp.shared_experts.<s>.gate_proj.weight`/`up_proj`/`down_proj` per s∈{0,1}, de-interleaved), `experts/layer_<l>/exp_<e>.arrow` for l≥2 via `PackExpertNVFP4` with w1=gate half, w3=up half of routed w13 (HALVES split — no de-interleave), w2 plain. Index `model.index.json`: format tag, `"model": "Inkling"`, geometry fields incl. `swa_num_key_value_heads`, `dense_mlp_layers`, `num_shared_experts`, `logits_mup_div`, `full_layers`, `sliding_layers`, ballast counts, tokenizer kind, VRAM block (`EstimateVram`, routed class for expert tensors, `has_residual=false`).
Config gate: frozen cross-check unless `SyntheticGeometryAllowed()`; Unknown tensor → error; Mtp/Vision/Audio counted, never applied. Expert w13 dtype must be BF16 (source is unquantized); anything else → error.
Tokenizer: `BuildTokenizerArrowFilesTikToken(tiktoken/tokenizer.model, tokenizer_config.json, pinned, {bos=-1, eos=200006, pad=-1})`; mandatory for real checkpoint.

- [ ] Step 1: fixture header: synthetic geometry H=64, L=6 (full={5}, dense MLP layers 0..1, MoE 2..5), heads 4, kv full 2 / sliding 4 (exercise BOTH kv widths), hd 16, d_rel 4, extent 8, window 8, E=4, topk=2, shared=2, I=32, dense I=128, vocab 256/unpadded 200. `WriteCheckpoint(dir)` emits config.json (text_config with local_layer_ids, dense_mlp_idx=1, swa_* fields, n_shared_experts=2, route_scale, logits_mup_width_multiplier=24.0, use_gate_bias etc.) + one safetensors shard with EVERY tensor kind, using `MakeBf16Values` with row-identifiable patterns (value = row*1000+col so a wrong split is detectable); shared_w13/dense w13 written ROW-INTERLEAVED, routed w13 written HALVES; router bias/gate global_scale F32; vision 2 + audio 1 + mtp 2 ballast tensors.
- [ ] Step 2: write convert test: run converter under env gate; assert index fields (format/model/full/sliding/dense layers/ballast counts incl. mtp), dense NVFP4 records exist for wq..wr, norms/sconv/rel/router raw records exist, expert files exist for l∈[2,6), no expert files for l<2. **Row-pairing pins:** decode `mlp.gate_proj.weight` record for layer 0 and shared expert records via the canonical codec (`nvfp4_scale.h`) and assert reconstruction correlates with the CORRECT source rows (relL2 < 0.2 vs correct rows, > 0.5 vs swapped) — this kills the halves/interleave confusion in both directions. Routed expert exp_1 of layer 3: decode w1/w3 and check gate rows come from the FIRST half, up rows from the SECOND half of the source slab.
- [ ] Step 3: register `lsx_add_test(... LABEL convert TIMEOUT 300 ENV LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)` (temporary; manifest owns it in Task 3); build; run → FAIL (converter missing).
- [ ] Step 4: implement converter + forge flag; build; run test → PASS. Run `test_infrastructure_guard`, `test_logging_guard`, `test_absl_format_guard`.
- [ ] Step 5: commit `feat: inkling base converter (NVFP4 dense + routed, shared-sink experts)`.

### Task 3: manifest + model class + factory (stage 3) + loader

**Files:**
- Create: `models/inkling.json`, `libs/lsxcommon/src/inkling_moe_base_model.h/.cpp`, `libs/lsxcommon/include/lsxcommon/inkling_moe_base_load.h`, `libs/lsxcommon/src/inkling_moe_base_load.cpp`
- Create: `tests/test_inkling_moe_base_loader.cpp`, `tests/test_inkling_moe_base_model.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt` (remove Task1/2 manual registrations), `tests/test_model_independence_guard.cpp` consumers += `inkling_moe_base_model.cpp`, `docs/MODELS.md` (regenerate)

**Interfaces (produces):** namespace `lsxcommon::inkling::base`:
`struct Topology` (like inkling's + `swa_kv_heads`, `dense_mlp_layers`, `logits_mup_div`; no attn_res/clip/softcap fields), `LoadTopology(path, out)`, `TopologyMatchesFrozen(top)`;
`struct ExpertEntry` (same field order as inkling's) + `using ExpertDb = gpu::expert_store::ExpertStore<ExpertEntry>`; `LoadExpertIndex(path, layers, experts, max_open)` with `first_moe_layer=2`, `scale_block_size=16`;
`struct DenseNvfp4 { LsxNvfp4Weight w; int rows, cols; }` per projection; `struct LayerWeights` { full flag; DenseNvfp4 wq,wk,wv,wo,wr; dense gate/up/down (l<2); shared[2] gate/up/down; __half* norms (attn_norm, mlp_norm, q_norm[128], k_norm[128]); float* rel_proj, 4 sconv taps, router_w [258,H], router_bias[256], router_gs, mlp_gs; float* conv_state (2*k*kv_dim + 2*k*H) };
`struct PinnedWeights` { top; __half* embed/head [201024,6144]; embed_norm/final_norm; layers; resident_bytes };
`LoadPinnedWeights`, `FreePinnedWeights` (release EVERY plane), `class InklingMoeBaseModel final : public ModelBase` with `kType="inkling"`, Load/Infer/SelfCheck/BuildPromptInputIds/BuildConversationInputIds/ParseToolCalls/TraceForward/TeardownWeights following the inkling-2 model shape (residency lock, GenerationStream, abort, AcquirePermanentWeights, InferenceResult fill). Infer/pipeline bodies land in Task 5; Task 3 lands Load + stubs that return a clear error, with the lifecycle contract present from the first commit.
Manifest: full stages block (`convert/load/model/kernels/selfcheck/fsi/code/det/perf` targets named `test_inkling_moe_base_*`, parity/selfcheck forms, perf `test_perf_baseline_inkling`), vault `{dir:"Inkling", env:"LSX_TEST_INKLING_ROOT"}`, aliases `["inkling"]`, forge_flag `--inkling-base`, tokenizer_kind 2, tool_roundtrip true, cooperative_cancel true, selfcheck_frame_prompt true, selfcheck_bit_exact true, selfcheck_max_gen 2 (provisional; re-measured in Task 8 BEFORE goldens), golden_debt {} initially with fsi/code/det debt strings until Task 8 captures (declare debt honestly, remove when goldens land), lifecycle_tu `["inkling_moe_base_model.cpp","inkling_moe_base_inference_pipeline.cu"]`, report path `docs/model-reports/2026-11-17-Inkling.md` (create placeholder at Task 9; until then omit `report` field).

- [ ] Step 1: loader test (synthetic converted fixture produced by running the Task-2 converter on the synthetic checkpoint): LoadTopology fields, frozen-match relaxation, LoadPinnedWeights uploads every plane with right shapes, Free releases (cudaMemGetInfo delta ≈ 0), expert store views resolve for layer≥2 and plane sizes match NVFP4 geometry. Model test: factory resolves "inkling" and alias; Load on synthetic tree succeeds; frozen mismatch (real mode) fails Load; pre-set-abort Infer returns promptly (streaming_assert.h pattern).
- [ ] Step 2: register both tests temporarily; build; watch FAIL.
- [ ] Step 3: implement load + model class + manifest; `python3 tools/gen_model_manifest.py --strict --docs docs/MODELS.md --root .`; reconfigure (generated tables); remove temporary registrations; build.
- [ ] Step 4: run loader/model tests + `test_manifest_sync`, `test_model_suite_guard`, `test_model_independence_guard`, `test_ctest_labels_guard`, `test_code_naming_guard` → PASS (suite guard may need the tool-verdict row + golden debt; add verdict row `{"inkling", {"test_inkling_moe_base_tool_framing"}}` when Task 4 lands — until then declare the model-stage tool test in stages so the guard sees the target).
- [ ] Step 5: commit `feat: inkling base model class, loader, manifest row`.

### Task 4: tokenizer + tool bridge (stages 4/4b)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_base_tool_format.h`, `libs/lsxcommon/src/inkling_moe_base_tool_format.cpp`
- Create: `tests/test_inkling_moe_base_tool_framing.cpp`, synthetic tokenizer fixture in `tests/data/` (reuse the pattern of the inkling-2 fixture; sentinels in added_tokens)
- Modify: `tests/test_code_naming_guard.cpp` AllowedCommon += `inkling_moe_base_tool_format`; `tests/test_model_suite_guard.cpp` tool-verdict row; CMakeLists source list

**Interfaces:** same segment API shape as inkling-2: `struct Segment{std::string text; bool is_control;}`, `RenderConversation(conv, out)`, `ParseToolCalls(text, out_calls)`, `MarkerId(tok, spelling)`, `AppendIds(tok, segs, out_ids)`. Grammar from THIS checkpoint's jinja (verified 2026-11-17): tool_declare system turn first; effort line `Thinking effort level: 0.9` before first non-system message; role markers; assistant thinking channel `<|message_model|><|content_thinking|>…<|end_message|>`; tool call frame `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"…","args":{…}}<|end_message|>`; assistant turn ends `<|content_model_end_sampling|>`; tool role header carries the tool name; generation prompt `<|message_model|>`. Differences vs inkling-2 renderer: tool-name header on the tool turn IS present here too; verify each frame against jinja output captured with the real tokenizer (python jinja2 render → expected ids fixture committed under tests/data/).

- [ ] Step 1: capture expected frames: render the checkpoint's chat_template.jinja with transformers/jinja2 for (a) single user prompt, (b) system+user+assistant+tool_calls+tool reply conversation, (c) tools declared; tokenize with the checkpoint tokenizer; commit id sequences as fixture.
- [ ] Step 2: write framing test (label fast): render structure equals fixture ids on the synthetic tokenizer; ParseToolCalls round-trips a generated call span (NOT the full prompt); marker ids resolve; empty/absent-marker fallback path.
- [ ] Step 3: watch FAIL; implement codec + model virtuals; watch PASS; guards green.
- [ ] Step 4: commit `feat: inkling base native chat/tool codec (step 4b)`.

### Task 5: CUDA kernels + pipeline (stages 5/6)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/inkling_moe_base_kernels.h`, `libs/lsxcommon/src/inkling_moe_base_kernels.cu`, `libs/lsxcommon/include/lsxcommon/inkling_moe_base_inference_pipeline.h`, `libs/lsxcommon/src/inkling_moe_base_inference_pipeline.cu`
- Create: `tests/test_inkling_moe_base_kernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (`lsxcommon_inkling_base_cu` OBJECT lib, FMAD off, link into lsxcommon consumers like inkling), model Infer wiring

**Read first:** `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md` + `numeric-contract.md`; skill nvfp4-blackwell-conventions.

**Kernel set (host-reference pinned, deterministic trees, no atomicAdd):**
- `RunRelBias` (per-head d_rel dot into bank, zero outside [0,extent)) — same math as inkling-2's, own TU.
- `RunQkHeadNorm`: weighted per-head RMSNorm over hd=128 with learned [128] weights (q_norm/k_norm), FP32.
- `RunKvAppend`/`RunAttention`: per-layer KvCacheGeometry (pos-major), NO sink, scale=1/128, sliding window 512 on sliding layers; rel bias added pre-softmax; full layers apply tau=log-scale to q AND bias (tau helper `LogScaleTau(top,pos)` host+device testable).
- `RunShortConv`: static taps only, out = conv(x)+x (inv_scale=1.0), ring state [k][channels], per-sequence reset.
- `RunRouter`: E=258 rows (256 routed + 2 shared), sigmoid scores; selection = top-6 of sigmoid+bias over first 256; weights = softmax over logsigmoid of the 6 selected + 2 shared logits; ×route_scale×global_scale; outputs 6 routed slots + 2 shared gammas. Deterministic thread-0 selection like inkling-2's.
- `RunSwiGLU` (plain silu(g)*u — no softcaps), `RunExpertAccumulate`, `RunSharedAccumulate` (adds shared outputs × gammas into the same FP32 acc), norm/cast/scale plumbing, `RunEmbedGather`, `RunNormF32In`, `RunMatVecF16` (FP16 tables for embed/head), logits path: final norm → ×(1/24) → head matvec → slice 200058 (NO softcap).
- Dense/shared/expert GEMV through `lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV` / `...ExpertBatch` ONLY.
- Pipeline Forward order per layer: attn_norm(x) → wq/wk/wv/wr → k/v sconv on RAW projections → q/k head-norm (AFTER conv) → (full: tau on q) → KV append → rel bias (full: tau on bias) → attention (window on sliding) → wo → attn_sconv → +residual → mlp_norm → [l<2: dense gate/up/down ×mlp_gs | l≥2: router → stage 6 experts (ExpertStore EnsureResident) → gate/up GEMV → SwiGLU → down GEMV → accumulate; shared 2 experts GEMV ×gammas added] → mlp_sconv → +residual. Head: final_norm → /24 → unembed. Session: KV per-layer geometry array (two KvCacheGeometry values: full/sliding), EnsureCapacity growth copies kv_rows*pos_stride per layer; conv rings reset per request; kMaxContext=8192.
- Scratch arena sized like inkling-2's minus AttnRes stack; expert staging slab 6 slots.

- [ ] Step 1: kernels test .cu with host references: rel-bias gather incl. out-of-extent zeros; head-norm vs double reference; attention (both kv widths, window boundary at 512, tau path) vs host softmax reference; short conv ring across 5 steps vs host conv, reset semantics; router joint-renorm with asymmetric logits + bias flipping selection but not weights; SwiGLU; shared accumulate. Register `lsx_add_cuda_test(... LABEL inference FMAD_OFF)`; watch FAIL.
- [ ] Step 2: implement kernels; watch PASS.
- [ ] Step 3: pipeline: extend model test with a synthetic-weights end-to-end decode (deterministic, 2 requests identical, abort mid-decode honoured, capacity growth across requests) + TraceForward API; watch FAIL then implement; PASS.
- [ ] Step 4: full fast+convert+inference contour rebuild; guards green; commit `feat: inkling base CUDA kernels + generation pipeline`.

### Task 6: real conversion + numeric validation vs reference

**Files:** none in-tree (scripts under `/mnt/fast_storage/tmp/inkling_base/`); output tree `/mnt/fast_storage/huggingface/Inkling/logestix/`.

- [ ] Step 1: run forge `--inkling-base` on the real checkpoint (background, log to /mnt/fast_storage/tmp). ~1.9 TB read; verify index counts: 66 layers, 64×256=16384 expert files, ballast counts (8 visual, 2 audio, 160 mtp), tree size ≈ 500 GB.
- [ ] Step 2: python reference script: `transformers.models.inkling` `InklingTextModel` with real dense weights (BF16, CPU/GPU split as fits) + on-demand expert dequant; compare engine TraceForward layer-by-layer on a short prompt (relL2 per layer; expect ~1e-2 NVFP4 body error, no structural divergence). Also float64 re-derivation of layer 0 and a full layer from raw safetensors.
- [ ] Step 3: knowledge sanity: framed "The capital of France is" → " Paris" top-1; mean NLL over a fixed paragraph < log(100). If red → bug checklist (skill stage-2 procedure), NOT golden edits.
- [ ] Step 4: record numbers for the report; commit any converter fixes test-first.

### Task 7: goldens + gates (stage 8)

**Files:** `tests/fixtures/{fsi,code,det}/inkling.json`, `tools/{fsi_capture,code_capture,det_capture}.py` (per-model overrides if needed), `models/inkling.json` (remove debt, set measured probe), `docs/SELFCHECK_MATRIX.md`.

- [ ] Step 1: measure the probe on the real tree: framed Paris, kOff, max_gen from measurement (start 2; extend only if the answer needs it); record logits_hash; set manifest fields BEFORE capturing goldens.
- [ ] Step 2: capture FSI/CODE/DET goldens. Source 952B not runnable locally → template-derived goldens via the capture tools' template path (inkling-2 precedent), `source` field states it. Use the checkpoint's own tokenizer+template for input_ids.
- [ ] Step 3: run gates one at a time (GPU lock discipline): selfcheck ×3 bit-exact, FSI 3/3, CODE 3/3, DET 3/3, tool round-trip. Red borderline → residual-plane procedure.
- [ ] Step 4: update matrix + manifest; guards green; commit `test: inkling base parity goldens + gates`.

### Task 8: perf baseline (Release)

**Files:** `tests/test_perf_baseline.cpp` (PerfModel kInklingBase + TEST_CASE "perf baseline: inkling"), `tests/perf/perf_baselines.json` (harness-seeded), manifest perf stage.

- [ ] Step 1: add PerfModel entry (vault "Inkling", canonical prompt, 2048) + TEST_CASE; fast `test_perf_baseline_logic` green.
- [ ] Step 2: configure/build `linux-release-znver4` (or AOCC tree per building-the-project); run `ctest -R test_perf_baseline_inkling` → seeds row with ttft/power. Expected band 0.5–1.5 t/s (read-bound ~30 GB/token).
- [ ] Step 3: commit `test: inkling perf baseline seeded via harness`.

### Task 9: report + docs close-out

**Files:** `docs/model-reports/2026-11-17-Inkling.md` (+README index), `docs/OPEN_TASKS.md` resolved entry, manifest `report` field, regenerate `docs/MODELS.md`.

- [ ] Step 1: write publish-ready post + evidence (measured numbers only; debt named as debt).
- [ ] Step 2: full verification: fast suite, convert sweep, inference sweep, guards; `python3 tools/gen_model_manifest.py --strict` clean.
- [ ] Step 3: commit `docs: inkling base ingestion report`; offer branch finish to operator.