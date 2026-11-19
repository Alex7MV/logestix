# Muse-Glimmer-30B Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (AGENTS.md forbids subagents in this repo — execution is native, in the main session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest Muse-Glimmer-30B end to end (converter → NVFP4 tree → model → tokenizer/ATEM bridge → kernels → all gates → report) on branch `feat/muse-glimmer-30b-ingest`.

**Architecture:** New dense family `muse` following the gemma-4-31b dense contour: own frozen passport, converter emitting pinned NVFP4 triples + FP16 embed/head/norms, single-stream CUDA generation loop on the neutral `lsx_nvfp4` layer, kind-3 tokenizer bridge, native ATEM tool codec. All ctest registration is GENERATED from `models/muse-glimmer-30b.json`.

**Tech Stack:** C++20/CUDA 13.5 (Blackwell sm_120), Arrow pinned IPC, Catch2, CMake presets `linux-debug` (dev) / `linux-release-znver4` (perf), HF transformers 5.15.1 CPU oracle for goldens.

**Spec:** `docs/superpowers/specs/2026-11-18-muse-glimmer-30b-ingest-design.md`

## Global Constraints

- TDD: every production change lands with a test that was watched failing first, label in the same commit.
- Frozen geometry: hidden 6656, 52 layers, I 19968, V 202048, 32 Q / 2 KV × 128, window 2048, full layers ≡ l%4==3, theta 500000 (NoPE on full), qk_scale 3.87, score scale 1/sqrt(128), softcap 20 with multiplier 0.19611613513818404, centered norms ×(1+w) (pre eps 1e-5, post eps 1e-8), final norm plain ×w eps 1e-5, weightless embed RMSNorm eps 1e-5, silu SwiGLU, bos 200000 / eos {200001, 200008} / pad 200018.
- NVFP4 GEMM only via `lsx_nvfp4::RunLsxNvfp4Gemm`; norms/biases never quantized; embed+lm_head FP16.
- One stream, no per-token `cudaDeviceSynchronize`; thread-0 shared-memory RMW behind `__syncthreads()`; no `atomicAdd` feeding final_logits; `-fmad=false` OBJECT lib; KV addressed only via `KvCacheGeometry` (pos-major).
- No abseil containers in headers reachable from `.cu`; spdlog-only logging; `absl::StrFormat` printf-style.
- Naming: `muse_dense_<role>` files; `muse` token added to SchemeRegex/FamilyStems and `muse_tool_format` to AllowedCommon in the SAME commit as first files.
- Converter tests: synthetic checkpoint only, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` via test ENV, label `convert`, TIMEOUT ≤ 600 s. Loader/model stages: synthetic fixtures only, never the vault.
- Manifest is the only hand-written registry; regenerate `docs/MODELS.md` via `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .` in the same commit as manifest edits.
- One GPU workload at a time: check `nvidia-smi --query-compute-apps` before any GPU-labelled run.
- Temp files in `/mnt/fast_storage/tmp`; NVIDIA docs mirror `/home/alexmv2025/projects/cuda/docs` (read NVFP4_MEMORY + numeric-contract before kernel work).
- Commit prefixes: build:/feat:/fix:/refactor:/test:/docs: — English messages.

## Review Focus

1. NoPE layers receiving RoPE anyway (or sliding layers losing it) — garbage beyond short contexts; pinned by per-layer rope/window table test (Task 2) + kernel parity (Task 6).
2. Attention gate applied AFTER o_proj or as two contiguous halves instead of elementwise sigmoid over 4096 dims — pinned by gated-attention host-reference parity (Task 6).
3. Centered (1+w) vs plain (w) norm mix-up between layer norms and final norm — pinned by norm parity cases with non-zero weights (Task 6).
4. Logits missing output_multiplier or global_scale (hot logits, greedy hides it) — head test with non-unit gs and multiplier (Task 6).
5. ATEM markers BPE'd as raw bytes instead of injected ids (special:true), and ParseToolCalls matching the example call inside the tool-defs block — framing round-trip on synthetic tokenizer, parse only from generated tail (Task 5).

---

### Task 1: Naming vocabulary + branch baseline

**Files:**
- Modify: `tests/test_code_naming_guard.cpp` (SchemeRegex fam list, FamilyStems, AllowedCommon)

**Interfaces:**
- Produces: `muse_dense_*` names accepted; `muse_tool_format` allowed.

- [ ] Step 1: Add a failing naming case: in the positive-match TEST_CASE add `CHECK(m("muse_dense_constants")); CHECK(m("muse_dense_inference_pipeline"));` and an AllowedCommon check for `muse_tool_format`.
- [ ] Step 2: Build+run `test_code_naming_guard` — verify FAIL on the new checks.
- [ ] Step 3: Add `muse` to fam regex + FamilyStems, `muse_tool_format` to AllowedCommon.
- [ ] Step 4: Rebuild, run guard + full fast sweep baseline (`ctest --preset linux-debug`); all green.
- [ ] Step 5: Commit `test: accept muse family naming vocabulary`.

### Task 2: Frozen passport + topology parser

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/muse_dense_constants.h` (namespace `lsxcommon::muse_dense`, `frozen` sub-namespace, `MuseDenseTopology` struct)
- Create: `libs/lsxcommon/include/lsxcommon/muse_dense_convert.h`, `libs/lsxcommon/src/muse_dense_convert.cpp` (classifier + `load_muse_dense_topology` only in this task)
- Test: `tests/test_muse_dense_constants.cpp` (label fast), `tests/test_muse_dense_convert.cpp` (label convert, TIMEOUT 120)
- Modify: `libs/lsxcommon/CMakeLists.txt` (add convert.cpp), `models/muse-glimmer-30b.json` (new, stages convert only + golden_debt for the rest), regenerate docs/MODELS.md

**Interfaces:**
- Produces: `frozen::kHiddenSize/kNumLayers/kIntermediateSize/kVocabSize/kNumHeads/kNumKVHeads/kHeadDim/kSlidingWindow/kRopeTheta/kQkScaleFactor/kAttentionScale/kOutputMultiplier/kFinalLogitSoftcap/kRmsNormEps/kPostNormEps/kBosId/kEosIdText/kEosIdTurn/kPadId`, `frozen::IsFullLayer(l) == (l % 4 == 3)`.
- Produces: `enum class TensorKind { Unknown, Visual, Embed, LmHead, FinalNorm, AttnQ, AttnK, AttnV, AttnO, AttnGate, MlpGate, MlpUp, MlpDown, InputNorm, PostAttnNorm, PreFfnNorm, PostFfnNorm }`, `ParsedTensor{kind,layer}`, `classify_muse_dense_tensor(name)`, `muse_dense_tensor_is_nvfp4(kind)` (true for the 8 projections only), `load_muse_dense_topology(config_path) -> MuseDenseTopology` (parses text_config + generation_config eos merge; error_msg on malformed).
- Produces: `kFormatTag = "logestix-muse-nvfp4"` in muse_dense_convert.h.
- Topology fields mirror gemma_dense: num_layers, layer_types vector, hidden/intermediate/vocab/heads/kv_heads/head_dim, sliding_window, rms_norm_eps, post_norm_eps, rope_theta, qk_scale_factor, output_multiplier, final_logit_softcapping, bos_id, eos_ids, pad_id, total_pinned_tensors, vram, error_msg + helpers `IsFull(l)`, `Window(l)`, `RopeTheta(l)` (0 on full layers).

- [ ] Step 1: Write `test_muse_dense_constants.cpp`: static values above; IsFullLayer(3/7/51)==true, (0/2/50)==false; 13 full + 39 sliding; kAttentionScale == 1/sqrt(128) within 1e-7; kOutputMultiplier == 0.19611613513818404f.
- [ ] Step 2: Write classifier cases in `test_muse_dense_convert.cpp`: all 16 text kinds classify with layer index (incl. `self_attn.gate_proj` → AttnGate NOT MlpGate); `model.vision_tower.*`/`vision_adapter`/`vision_projection` → Visual; unknown name → Unknown; nvfp4 policy true for 8 projections, false for Embed/LmHead/norms. Topology case reads the REAL config.json via vault env (`LSX_TEST_MUSE_GLIMMER_30B_ROOT`, SKIP when absent) and asserts every frozen value matches.
- [ ] Step 3: Create manifest `models/muse-glimmer-30b.json` (copy inkling.json shape): model_type `muse-glimmer-30b`, family `muse`, model_class `MuseGlimmer30BModel`, model_header `muse_dense_model.h`, passport the constants header, namespace `lsxcommon::muse_dense::frozen`, forge_flag `--muse`, tokenizer_kind 3, vault dir `Muse-Glimmer-30B`, stages.convert=[test_muse_dense_convert], golden_debt for missing stages ("ingestion in progress"), overrides: convert timeout 120 + synthetic env. Register the two test TUs in tests/CMakeLists.txt via `lsx_add_test` (constants label fast; convert label convert). Add muse_dense_convert.cpp to lsxcommon sources.
- [ ] Step 4: Build; run both tests — verify FAIL (missing header/functions).
- [ ] Step 5: Implement constants header + classifier + topology parser (pattern: gemma_dense_convert.cpp lines 1–250; simdjson DOM parse of text_config).
- [ ] Step 6: `python3 tools/gen_model_manifest.py --strict --docs docs/MODELS.md --root .` clean; rebuild; constants + convert tests PASS; `test_manifest_sync`, `test_model_suite_guard` (pending-stage tolerance), labels guard green.
- [ ] Step 7: Commit `feat: muse-glimmer frozen passport + tensor classifier`.

### Task 3: Converter pipeline + forge flag (stages 1–2, tokenizer stage 4)

**Files:**
- Modify: `libs/lsxcommon/src/muse_dense_convert.cpp`, `include/lsxcommon/muse_dense_convert.h` (add `ConvertMuseDenseModel`, `WriteMuseDenseModelIndex`)
- Modify: `apps/lsxforge/src/cli.h` (Pipeline::kMuse), `cli.cpp`, `main.cpp` (ABSL_FLAG muse + dispatch), `apps/lsxforge/tests/test_lsxforge_cli.cpp`
- Test: `tests/test_muse_dense_convert.cpp` (extend), `tests/test_muse_dense_fixture.h` (synthetic checkpoint builder on synthtest primitives)
- Modify: `models/muse-glimmer-30b.json` (no new stage yet — convert target grows cases)

**Interfaces:**
- Produces: `arrow::Status ConvertMuseDenseModel(input_path, output_root)` emitting `logestix/pinned/global.arrow` (embed FP16, lm_head FP16, final norm FP16 raw), `pinned/layer_N.arrow` (8 NVFP4 triples + 4 norms raw BF16→FP16), `pinned/tokenizer_*.arrow` (kind 3, policy.pretokenizer=1, bos/eos/pad from config), `logestix/model.index.json` (kFormatTag, geometry, layer_types, skipped_visual_tensors=809 on real source, vram block).
- Produces: fixture `musetest::WriteSyntheticCheckpoint(dir, layers=4, hidden=…small)` — every path: globals, all projections, norms, vision ballast tensors, both eos ids; BF16 via synthtest::MakeBf16*.
- BF16→FP16 conversion happens IN THE CONVERTER for embed/lm_head/norms (host loop, `__nv_bfloat16` bit shift or existing helper — reuse `bf16_convert` host path if present, else simple RNE host conversion pinned by a fixture case).

- [ ] Step 1: Extend convert test: synthetic E2E — ConvertMuseDenseModel on fixture; assert index format tag/geometry/ballast count; layer arrow has 8×3 NVFP4 records + 4 FP16 norms; global arrow has FP16 embed/lm_head/final norm with exact RNE values from known BF16 inputs; tokenizer arrows exist with kind 3, base_vocab_size 200000-style split (fixture uses small vocab), bos/eos/pad ids; unknown tensor makes conversion ERROR; NVFP4 decode of a known row ≈ source within codec tolerance (non-unit global_scale).
- [ ] Step 2: Add forge CLI case `--muse` → Pipeline::kMuse in test_lsxforge_cli.cpp; run — FAIL.
- [ ] Step 3: Run convert test — FAIL (ConvertMuseDenseModel undefined).
- [ ] Step 4: Implement converter (pattern gemma_dense_convert.cpp 500–730: ReadTensorData → quantize_tensor_gpu for projections; FP16 conversion for globals/norms; PinnedSchema batches; WriteMuseDenseModelIndex with vram estimate; tokenizer bridge via BuildTokenizerArrowFiles with pretokenizer=1 policy). Wire forge flag.
- [ ] Step 5: Rebuild; convert + forge CLI tests PASS; fast sweep green.
- [ ] Step 6: Commit `feat: muse-glimmer converter + forge --muse pipeline`.

### Task 4: Loader + model class + factory registration (stage 3)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/muse_dense_load.h`, `src/muse_dense_load.cpp`
- Create: `src/muse_dense_model.h`, `src/muse_dense_model.cpp`
- Test: `tests/test_muse_dense_loader.cpp`, `tests/test_muse_dense_model.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `models/muse-glimmer-30b.json` (load/model stages; model stage also claims constants test), tests/CMakeLists.txt

**Interfaces:**
- Produces: `MuseDenseLayerWeights{ __half* d_input_norm/d_post_attn_norm/d_pre_ffn/d_post_ffn; LsxNvfp4Weight attn_q,attn_k,attn_v,attn_o,attn_gate, mlp_gate,mlp_up,mlp_down; void* d_arena; size_t arena_bytes; void Free(); }`.
- Produces: `MuseDensePinnedWeights{ __half* d_embed; __half* d_lm_head; __half* d_final_norm; int vocab_size, hidden; std::vector<MuseDenseLayerWeights> layers; MuseDenseTopology top; uint64_t resident_bytes; void Free(); }` — resident_bytes counted at allocation sites.
- Produces: `cudaError_t LoadMuseDensePinnedWeights(model_path, pw, stream)` — mmap Arrow views, strict format/geometry gate (SyntheticGeometryAllowed bypass), hard error on missing file/tensor.
- Produces: `class MuseGlimmer30BModel final : public ModelBase` with `kType = "muse-glimmer-30b"`, Load/Infer/SelfCheck/TeardownWeights, ValidateGeometry, session kv_/tail under residency lock (Infer body lands Task 7; this task lands Load + geometry gate + factory row + stub Infer returning error "not implemented" is NOT allowed — instead Task 4 and Task 7 merge ordering: implement Load fully, Infer delegates to RunMuseDenseGeneration which Task 6/7 create; so Task 4's model test covers Load/Unload/Inspect/geometry-gate only, and the manifest model stage row lands here while lifecycle_tu inference.cu appears in Task 6 — declare golden_debt accordingly until then).
- Registration: `model_registry.inc` is generated — the manifest's model_class/model_header drive it; ensure `MuseGlimmer30BModel` ctor registered via the generated list.

- [ ] Step 1: Loader test on the Task-3 fixture converted tree: round-trip shapes/dtypes, norms FP16 exact, NVFP4 members non-null with right shapes, resident_bytes > 0, Free() idempotent; missing layer file → error; wrong format tag → error; frozen-geometry mismatch → error (synthetic env off).
- [ ] Step 2: Model test: factory creates `muse-glimmer-30b`; Load on synthetic converted fixture succeeds with synthetic env; geometry mismatch fails Load; Inspect reports type/vocab; Unload frees (second Load works).
- [ ] Step 3: Run — FAIL.
- [ ] Step 4: Implement loader + model skeleton (pattern gemma_dense_load.cpp / gemma_dense_model.cpp; governor leases AcquirePermanentWeights in Load, TeardownContext in TeardownWeights).
- [ ] Step 5: Update manifest stages load/model; regenerate; rebuild; tests + guards green.
- [ ] Step 6: Commit `feat: muse-glimmer loader + model class + factory row`.

### Task 5: Tokenizer verification + ATEM tool codec (stages 4/4b)

**Files:**
- Create: `libs/lsxcommon/src/muse_tool_format.h`, `src/muse_tool_format.cpp`
- Modify: `src/muse_dense_model.cpp` (BuildPromptInputIds / BuildConversationInputIds / ParseToolCalls)
- Test: `tests/test_muse_dense_tool_framing.cpp` (label fast), `tests/data/` synthetic tokenizer fixture with ATEM sentinel ids
- Modify: manifest (model stage claims tool_framing target; tool_roundtrip true), `tests/test_model_suite_guard.cpp` kTooling row `{"muse-glimmer-30b", {"test_muse_dense_tool_framing"}}`

**Interfaces:**
- Produces: `muse::RenderConversation(tok, conv, out_ids)` — default system turn (knowledge cutoff 2026-01-04, Reasoning strength: high, Valid recipients), `<|start|>role[ to=X]<|message|>` framing, `<|eom|>`/`<|eot|>` rules per the jinja, tool-defs block byte-faithful to render_tool_defs, generation prompt `<|start|>assistant`; markers injected via arrow_tokenizer_lookup ids (200000/200007/200008/200022/200023 on real vocab), free text via arrow_tokenizer_encode; bos prepended once.
- Produces: `muse::ParseAtemToolCalls(tok, gen_tokens, out_calls)` — scan ids for `<|start|>`…`to=` recipient channels, decode the ATEM XML between markers, parse `<atem:invoke name>`/`<atem:parameter name>` (values: raw string; JSON for lists/objects; true/false/null literals), fall back to tools::ParseToolCallsText when no markers.
- Round-trip tests NEVER parse the full rendered prompt (tool-defs embed an example call).

- [ ] Step 1: Write framing tests: (a) tokenizer kind-3 parity vs HF goldens captured with a small python script into tests/data/muse/ (encode 5 texts incl. contractions/digits/CRLF — capture from the REAL tokenizer.json, SKIP-free committed ids); (b) render structure assertions on synthetic fixture (channel markers as single ids, default system block, eom/eot placement); (c) parse: single call, multi-parameter, JSON list value, two calls, no-call text → empty; (d) render→parse round-trip on the ASSISTANT TAIL only.
- [ ] Step 2: Run — FAIL.
- [ ] Step 3: Implement codec + model virtuals; suite-guard kTooling row; manifest tool_roundtrip.
- [ ] Step 4: Rebuild; framing + guards green; fast sweep green.
- [ ] Step 5: Commit `feat: muse-glimmer ATEM conversation + tool bridge`.

### Task 6: CUDA kernels (stages 5–6)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/muse_dense_inference.h`, `src/muse_dense_inference.cu` (OBJECT lib `lsxcommon_muse_cu`, -fmad=false)
- Test: `tests/test_muse_dense_kernels.cu` (label inference)
- Modify: `libs/lsxcommon/CMakeLists.txt`, manifest kernels stage

**Interfaces:**
- Produces (all cudaError_t, one stream):
  - `RunMuseDenseCenteredNorm(stream, x, w, out, rows, hidden, eps)` — out = rms(x)·(1+w); `RunMuseDensePlainNorm(...)` — ·w (final norm); `RunMuseDenseEmbedNorm(...)` — weightless over gathered rows.
  - `RunMuseDenseQkNormScale(stream, q, k, seq, nh, nkv, hd, eps, q_scale)` — weightless per-head RMSNorm, Q ×3.87.
  - `RunMuseDenseRope(stream, pos, q, k, seq, nh, nkv, hd, theta)` — full-dim rotate-half; caller SKIPS it on NoPE layers.
  - `RunMuseDenseAttention(stream, q, k, v, out, seq, nh, nkv, hd, window, scale, pos_base, kgeom, vgeom)` + split-KV variant `RunMuseDenseAttentionSplit(..., partials, splits, ...)` — window>0 band, deterministic reductions (pattern: gemma_dense_inference.cu attention).
  - `RunMuseDenseAttnGate(stream, attn, gate, out, n)` — out = attn·sigmoid(gate) elementwise.
  - `RunMuseDenseSwiGLU(stream, gate, up, out, n)` — silu(gate)·up.
  - `RunMuseDenseEmbedGatherFp16(stream, embed, ids, out, seq, hidden)` — row copy (norm applied separately).
  - `RunMuseDenseLogitsArgmax(stream, lm_head_fp16, x, vocab, hidden, multiplier, cap, d_argmax, d_maxlogit, d_logits)` — FP16 GEMV via lsx_gemv/neutral path, logits = cap·tanh(raw·multiplier/cap), deterministic argmax lowest-index-tie (pattern: gemma_dense softcap kernel but FP16 head).
- Kernel tests use host references; every NVFP4-touching fixture uses non-unit global_scale; attention cases cover window band edge (pos exactly window), NoPE path (no rope call) vs sliding, GQA 16:1; norm cases use non-zero w to distinguish (1+w) from w; gate case verifies elementwise sigmoid placement pre-o_proj; logits case pins multiplier AND cap with hand-computed values; determinism: same inputs twice → bit-identical.

- [ ] Step 1: Read NVIDIA docs mirror (NVFP4_MEMORY, numeric-contract) — reconcile E2M1/E4M3 semantics before writing kernels.
- [ ] Step 2: Write kernel test file with the cases above — build, run, verify FAIL (kernels missing).
- [ ] Step 3: Implement kernels + OBJECT lib wiring.
- [ ] Step 4: Rebuild; kernels test PASS; manifest kernels stage + regenerate; guards green.
- [ ] Step 5: Commit `feat: muse-glimmer CUDA kernels (gated attention, NoPE, centered norms, softcap head)`.

### Task 7: Generation pipeline + lifecycle contract (stage 5 driver)

**Files:**
- Modify: `src/muse_dense_inference.cu` (add `MuseDenseGenContext`, `RunMuseDenseGeneration`), `include/lsxcommon/muse_dense_inference.h`, `include/lsxcommon/muse_dense_load.h` (MuseDenseKvCache), `src/muse_dense_load.cpp` (Allocate/Free), `src/muse_dense_model.cpp` (full Infer + SelfCheck via RunParisSelfCheck)
- Test: `tests/test_muse_dense_pipeline.cu` (synthetic full-geometry-shrunk model, determinism across repeats; label inference) — claimed by kernels stage in manifest
- Modify: manifest (lifecycle_tu, selfcheck stage via shared driver form=selfcheck, cooperative_cancel true, selfcheck probe fields placeholder → measured in Task 9)

**Interfaces:**
- Produces: `MuseDenseKvCache{dk,dv,geo per layer; Allocate(pw,max_total); Free();}` pos-major uniform 2×128 geometry.
- Produces: `RunMuseDenseGeneration(stream, pw, input_ids, max_gen, eos_ids, out, kv, ctx, sampler, gs)` — prefill+decode one stream; layer sandwich x + post_attn(o(gate·attn(qk_norm(rope?(…))))); x + post_ffn(down(silu(gate)·up(pre(x)))); embed norm on gather; final plain norm; softcap head; Emit per token, abort_flag polled per step; RAII workspace freeing on every error path.
- Model Infer: RunUnderResidencyLock whole decode; LsxFastMathScope(EffectiveFastMathMode(cfg_)); session kv_/tail created/published inside lock; FillInferenceResult with output_text + weight_resident_bytes.

- [ ] Step 1: Pipeline test: synthetic converted model (fixture, small geometry via synthetic env) → RunMuseDenseGeneration twice → identical tokens + final_logits; abort pre-set → zero tokens, ok=false-or-aborted; streaming Emit count == tokens.
- [ ] Step 2: Run — FAIL.
- [ ] Step 3: Implement driver + model Infer/SelfCheck; extend kLifecycle via manifest lifecycle_tu.
- [ ] Step 4: Rebuild whole contour; pipeline + model tests green; independence guard (LsxFastMathScope, lifecycle contract) green.
- [ ] Step 5: Commit `feat: muse-glimmer generation pipeline + lifecycle contract`.

### Task 8: Real conversion + goldens capture (CPU oracle)

**Files:**
- Create: `tools/muse_golden_oracle.py` (text-tower-only loader: MuseGlimmerTextModel from text_config, load language_model.* + lm_head weights, skip vision; CPU float32/bf16)
- Modify: `tools/fsi_capture.py`, `tools/code_capture.py`, `tools/det_capture.py` OVERRIDES: `"muse-glimmer-30b": {"loader": "muse_text", ...}` (import oracle; chat_template.jinja framing; ignore_eos where needed)
- Create: `tests/fixtures/{fsi,code,det}/muse-glimmer-30b.json` (captured goldens)

- [ ] Step 1: Convert the real checkpoint: `build/linux-debug/apps/lsxforge/lsxforge --muse --input /mnt/fast_storage/huggingface/Muse-Glimmer-30B --output <vault>/Muse-Glimmer-30B` (vault root from lsx_model_models_dir; conversion output under it). Verify index: 809 ballast counted, 627 text tensors converted, vram block.
- [ ] Step 2: Oracle smoke: Paris prompt greedy on CPU → confirm "Paris" reachable; record probe parameters (frame_prompt yes/no, max_gen, bit-exact kOff) from THIS run.
- [ ] Step 3: Capture FSI/CODE/DET goldens (CPU, greedy, chat-template framed; budgets per skill defaults, raise if reasoning truncates). Commit goldens + capture-tool changes.
- [ ] Step 4: Update manifest: selfcheck_frame_prompt/selfcheck_max_gen/selfcheck_bit_exact from measurement; drop golden_debt entries whose goldens now exist.
- [ ] Step 5: Commit `feat: muse-glimmer real conversion + source goldens (fsi/code/det)`.

### Task 9: Real-weight gates (stage 8)

**Files:**
- Modify: manifest (fsi/code/det stages form=parity, selfcheck form=selfcheck, timeouts, SKIP 4; tool_roundtrip already true)
- No new test files — shared drivers.

- [ ] Step 1: Check GPU idle; run `ctest --preset linux-debug-inference -R "muse"` — selfcheck (≥3 bit-exact Paris cycles + tool round-trip + cooperative cancel), fsi, code, det. A SKIP is not a pass.
- [ ] Step 2: Any red → systematic-debugging skill; borderline tie → stage-2 residual procedure (bug checklist FIRST: gs in head, Free() completeness, codec parity); record outcome honestly.
- [ ] Step 3: All green (or debt recorded in OPEN_TASKS + SELFCHECK_MATRIX) → commit `test: muse-glimmer real-weight gates green`.

### Task 10: Perf baseline (stage 8 perf)

**Files:**
- Modify: `tests/test_perf_baseline.cpp` (PerfModel row + TEST_CASE "perf baseline: muse-glimmer-30b"), `tests/perf/perf_baselines.json` (via harness run, never hand-edited), manifest perf stage (`test_perf_baseline_muse_glimmer_30b`, vault_subdir Muse-Glimmer-30B, timeout 1800)

- [ ] Step 1: Add TEST_CASE + manifest row; regenerate; fast guard `test_perf_baseline_logic` green (pairing).
- [ ] Step 2: Build `linux-release-znver4`; check GPU idle; run the perf target — seeds the JSON row with ttft/power evidence at the canonical 2048-token protocol.
- [ ] Step 3: Commit `perf: seed muse-glimmer-30b baseline`.

### Task 11: Docs + report (stage 9) + final sweep

**Files:**
- Create: `docs/model-reports/2026-11-18-Muse-Glimmer-30B.md` (publish-ready post + evidence)
- Modify: `docs/model-reports/README.md`, `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md` (entry + any debt), manifest report field, regenerate docs/MODELS.md

- [ ] Step 1: Write report AFTER gates: measured numbers only, checkpoint traps found (config/tokenizer/generation quirks), what was not measured and why.
- [ ] Step 2: Full verification: `ctest --preset linux-debug` (fast), `-convert`, `-inference -R muse`, guards; report links evidence.
- [ ] Step 3: Commit `docs: muse-glimmer-30b stage-9 report + catalogue`.