---
name: ingesting-a-model
description: Use when adding, adapting or ingesting a new model architecture into logestix, or porting an existing family to a new variant — lsxforge converter, NVFP4 packing, IModel + LsxModelFactory registration, tokenizer arrow bridge, CUDA inference kernels, SelfCheck harness, and the ctest lifecycle stages. Triggers - new model, model adaptation, Model Ingestion Protocol, MODEL INGEST INVARIANT, kType, frozen geometry, selfcheck Paris.
---

# Ingesting a new model into logestix

## Overview

**MODEL INGEST INVARIANT**: adding a model is a **Stage 0 triage** followed by a
fixed 8-stage sequence, plus the mandatory **stage 4b tool-calling bridge** and
the closing **stage 9 model report**. Every agent MUST follow the order;
skipping or reordering a stage is a broken build.
Reference implementations (real code anchors, not placeholders): GLM-5.2,
DeepSeek-V4-Flash, GLM-5.3-Flash, Qwen3.5-9B, Kimi-K2.7-Code.
Where the model came from, and whether it is even runnable here, is decided in
Stage 0 — not once the kernels are written. The operator-side counterpart (how
to LAUNCH and accept one of these models, in order, at maximum speed) is
`running-a-model` §2a/§2b.

This skill is the single source of truth for the protocol, the file-naming
convention and the guard registries. `AGENTS.md` only points here.

Work in **small verified steps**: each stage lands together with the test that
pins it, in the same commit (see TDD below). Verification commands are at the
end.

**REQUIRED SUB-SKILL:** use `test-driven-development` for every stage.

## Iron rules (apply to all stages)

- **TDD.** No production code without a failing test first. A new test MUST get
  its ctest label in the same commit, or `test_ctest_labels_guard` goes red.
- **Frozen geometry.** hidden_size / layers / heads / experts / top-k are locked
  as compile-time constants with `static_assert` (runtime config reads are
  legacy — new models do not do it).
- **Kernel-level rules** — one stream with no per-token `cudaDeviceSynchronize`,
  the shared-memory sandbox, NVFP4 GEMM only through the model-neutral
  `lsx_inference_nvfp4` layer, CCCL/cub priority, tile programming, alignment
  and the NVIDIA docs mirror: **REQUIRED SUB-SKILL**
  `nvfp4-blackwell-conventions` (do not restate them here).
- **No abseil containers in `.cu`/`.cuh`-reachable headers.** Containers and
  streams in device TUs stay `std::`.
- **Norms and biases are never quantized** — they stay FP16/BF16.
- **Host storage is the neutral `ExpertStore`, and Arrow is zero-copy.** A family
  NEVER writes its own routed-expert host container, and NEVER keeps Arrow
  payloads as owning `std::vector<uint8_t>` records when they are retained past
  the upload. The rule, the two sanctioned storage shapes and the measured cost
  of breaking it are in **"Host-side weight storage — ExpertStore and Arrow
  zero-copy"** below; it is the single largest host-RAM defect class in the repo.
- **Model independence.** Each model opens
  `LsxFastMathScope(EffectiveFastMathMode(cfg_))` inside its own `Infer`.
- **Shared lifecycle contract.** The model TU that owns the generation body
  must carry all of it: `RunUnderResidencyLock` around the whole decode,
  a `GenerationStream` (one `Emit` per decoded token) honouring
  `req.abort_flag`, `TeardownContext()` in `TeardownWeights`,
  `AcquirePermanentWeights`/`AcquireExpertCacheLease` in `Load`, and a filled
  `InferenceResult` (`output_text` + `weight_resident_bytes`). A
  `GenerationStream*` parameter that is accepted and then discarded
  (`(void)gs;`) does NOT satisfy it — `lsxfabric --serve` streams exclusively
  through `req.token_callback`, so a family that never emits delivers an empty
  stream and cannot be cancelled. Incremental-session state (KV cache +
  generated tail) is created and published INSIDE the lock. All of it is
  machine-enforced by `test_model_independence_guard.cpp::kLifecycle` —
  extend that registry in the same commit.

## Stage 0 — triage the source checkpoint (before stage 1)

The 8 stages assume you already decided to ingest this model. This is the step
that decides it, and it is where a model is won or lost: every number you skip
here comes back as a silently wrong token.

**0.1 Classify: new family, sibling variant, or unsupported?**
- Existing `model_type` in `models/<org>/<model>.json` and the same architecture → **sibling
  variant**: add a manifest row (same `family`, own `variant`, shared `namespace`
  is forbidden so give it its own passport or reuse the parent's class) and reuse
  the parent's loader/kernels. Do NOT write a new constants header when the
  geometry is identical (that is what model independence forbids).
- Same architecture family, different geometry (width, expert count, hybrid
  pattern, head layout) → **new family** → its own frozen header (stage 3).
- A tensor the classifier does not recognise → unsupported until you extend the
  classifier deliberately. Never "skip the unknown ones".

**0.2 Read the geometry from `config.json` and write it down.** The fields the
converters actually parse (grep your target converter to confirm): `hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`,
`vocab_size`, `intermediate_size`, `moe_intermediate_size`,
`shared_expert_intermediate_size`, `num_experts`/`num_local_experts`,
`num_experts_per_tok`, `rope_theta`, `max_position_embeddings`.
**The hybrid pattern is the most-missed item**: record which layers are full
attention vs linear/Mamba/GDN/KDA/DeltaNet, and any sink-bias, window, indexer
or compressor layers. GLM-5.3-Flash has `kda_layers`/`dsa_layers` lists, not a
uniform pattern — a single `layer_types` assumption silently decodes garbage.

**0.3 Tokenizer.** Kind: byte-level BPE or SentencePiece-BPE (SP needs
`tokenizer_kind="sp-bpe"` + `BuildTokenizerArrowFilesSentencePiece` — the alice
precedent). Record `added_tokens` and which of them are `special: true`, plus
`bos`/`eos`/`pad` ids and whether `tie_word_embeddings` is set. If
`chat_template.jinja` exists, read it NOW — the tool format (stage 4b) and the
single-prompt framing (a chat model must never see an unframed prompt) both come
from it.

**0.4 Ballast to strip — and to COUNT.** Vision/audio towers, MTP/speculative
heads: stored and counted, never applied (`mtp_layer`, `skipped_visual_tensors`
in the written index). Everything else must be classified; a leftover tensor
should make the converter error, not pass.

**0.5 Feasibility against THIS machine.** NVFP4 dense ≈ `numel/2` + `numel/16`
scale bytes (norms/biases stay FP16), plus KV at the target context, plus the
host-mapped expert store. Compare with card VRAM (expert slot cache is sized
from free VRAM minus a 2 GiB reserve) and host RAM — expert pinning is chunked
because a single >300 GiB `cudaHostAlloc` is refused, and needs
`ulimit -l unlimited`. A model that only fits at 2 GiB free VRAM is not a
candidate.

**0.6 Semantics to verify before trusting any output** (each is a source of
"answers, but not the model's answers"): attention scale and where it is applied
(some checkpoints fold it into the weights, some add it at QK), partial rotary
factor and a per-layer-type `rope_theta`, any RoPE scaling factor, sliding
window + learned sink bias, routed scaling / expert renormalisation, value
scale, and whether MTP is (not) applied.

**Gate:** do not start stage 1 until you can answer, in one commit message: which
registry rows you will extend, the geometry table above, the tokenizer verdict,
what you strip, and the feasibility numbers.

## The 8 stages

### 1. Topology and structure passport (lsxforge)

- Extend the **C++** pipeline in `apps/lsxforge` (never a Python tool) to parse
  the original `config.json` and the safetensors (pattern:
  `lsxcommon::qwen::ConvertQwenModel`, `libs/lsxcommon/src/qwen_dense_convert.cpp`).
- Emit a structural manifest stripped of multimodal ballast.
- **PATH DIRECTION**: results are saved in the ARTIFACT store, one directory per
  codec profile — `<LSX_ARTIFACT_ROOT>/<org>/<model>/<profile>/` holding
  `artifact.fb`, `model.index.json`, `pinned/`, `experts/`, the tokenizer and the
  chat template. There is no `logestix/` segment: the profile directory IS the
  artifact, and the org/model names come from `models/<org>/<model>.json`
  (`lsxforge::ModelIdentityFor` + `ArtifactProfileDir`), never from the path the
  operator typed. A converter writes nothing into the read-only checkpoint; the
  old central `model-metadata/` block is gone (host state such as the per-GPU
  perf calibration lives in `lsx_cache_dir()`). No hardcoded paths in converters.
- Test: see **Converter test rules** below — a `convert`-labelled synthetic E2E.

#### Converter test rules (fixed)

Reference: `tests/test_glm_moe_flash_convert.cpp` (synthetic E2E) +
`tests/test_synth_checkpoint.h` (family-neutral primitives
`synthtest::FakeTensor`/`WriteSafetensors`/`MakeBf16*`/`TensorNamesIn`; family
fixtures may re-export them, e.g. `qwentest::` in `test_qwen_dense_fixture.h`).

1. **Label `convert` in the same commit**, and a mandatory bounded TIMEOUT
   (≤600 s; canonically 120 s) — guard `CheckConvertTimeout`. The TIMEOUT is
   the mechanical defence against a test that actually converts a model.
2. **Converting a real model in a test is forbidden.** Run the converter E2E
   pipeline on a synthetic checkpoint: `config.json` + `*.safetensors` written
   with `WriteSafetensors` into a temporary directory.
3. **Synthetic geometry**: the converter's frozen gates are relaxed with
   `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` — a **runtime env var** set through
   `set_tests_properties(... PROPERTIES ENVIRONMENT ...)`, NOT a compile
   define (a compile define does not export env).
4. **The fixture must cover every pipeline path**: globals (embed / lm_head /
   final norm — without them the converter errors), BF16→NVFP4, FP8→NVFP4 with
   scale tiles, raw tensors, routed experts (all 3 projections), empty layers
   of the frozen range, and the `model.index.json` layout (format tag +
   kda/dsa/moe lists).
5. **FP8 fixture tensors**: sides a multiple of 128 (GPU kernel); scale tile
   `[ceil(R/128), ceil(C/128)]` F32; safetensors dtype strings `BF16`,
   `F8_E4M3`, `F32`.
6. **A real checkpoint is allowed only to read one light file** (e.g.
   `config.json` in `ParseFlashConfig`) with `SKIP()` when the model is absent;
   full weight reads/conversions never. Synthetic fixtures do not replace
   tests that validate the real format's field names.
   **Never spell the path**: the store is org-nested and only
   `models/<org>/<model>.json` knows the org, so a test asks
   `manifest_query::SourceDirFor("<model_type>")` for the checkpoint or
   `ConvertedDirFor("<model_type>")` for the artifact (it reads `pinned/`,
   `model.index.json`, `artifact.fb`) — never `lsx_model_models_dir() / "<Dir>"`.
   The SKIP message must name the path it looked at (`+ dir.string()`), or
   "absent" and "wrong path" read the same in a ctest log.

### 2. Quantization and compression (NVFP4)

- Configure the quantization engine for total 4-bit quantization of dense
  linear matrices: E2M1 nibbles (2 per byte) + per-block E4M3 scales + a float
  `global_scale` (pattern: `QwenNvfp4Weight`).
- Routed/shared MoE experts are packed per expert; keep the repo layout so the
  inference kernels can consume it unchanged (no repack at upload unless
  documented).
- Decide the **host** side of that expert set now, not at the end: on disk the
  layout is fixed, but whether it is read as mmap views or materialized into
  host RAM is the loader's decision, and it is the difference between a bounded
  and an unbounded host footprint. See **"Host-side weight storage —
  `ExpertStore` and Arrow zero-copy"** below (rules R1/R2).

#### Parity fix: the residual plane (the current way to fix quantization)

A parity gate that goes red on a *borderline* answer is not automatically a
quantization problem, and "make the weights more precise" is the last resort,
not the first move. Run this decision procedure (all of it is cheap once the
converter is deterministic — re-conversion of a 26B model is under a minute):

1. **Is it a tie or a bug?** Teacher-force the golden prefix through the
   converted model (`Infer` with `prompt + prefix[:k]`, read `final_logits`;
   temporarily force the final soft cap to a huge value so you see pre-softcap
   logits) and compare against the source model run the same way
   (`config.text_config.final_logit_softcapping = None`, bf16). Record the
   source margin and the converted margin at the fork. A flip of a >10-logit
   preference is not noise — look for a kernel defect first.
2. **Bug checklist before any precision work** (each of these has shipped at
   least once): the logits kernel must multiply the row dot by the table's
   per-tensor `global_scale` (test it with `gs != 1`; a `gs = 1` fixture cannot
   see a dropped scale — gemma-4-26b shipped 7.01x hot logits that way);
   `Free()` must release every allocation the loader added; the decode pass
   must use the same codec/scale path as the converter (decode the written
   record on the host and compare with the source tensor: `base relL2` of
   ~0.09 with `base+res` ~0.009 is the expected pair, and a residual that is
   zero/garbage shows up immediately).
3. **Only then, place a residual plane.** If the model tracks the source
   per layer (cos > 0.99 early, ~0.96 at the last layer) and the fork is a
   template near-tie, the cause is accumulated 4-bit body error. The fix is a
   **second NVFP4 plane carrying the residual** `w - dequant(base)`, decoded as
   one more GEMV with `acc = true`: ~8 effective bits for 2x the bytes, no new
   codec, schema or kernel.
4. **Measure the placement; never assume it.** A recipe that works on one
   member does not transfer to its sibling, and an offline HF-side simulation
   is only *qualitative* (its baseline error is not the engine's — on gemma-4
   the sim said "o+down suffices" for the dense 31B and the engine disagreed).
   Calibrate by re-converting the engine's own tree per candidate placement and
   probing the fork. Prefer the cheapest set that clears every gate, and prefer
   leaving the biggest/streamed tensors alone (routed experts) so the decode
   read volume is untouched.
5. **Budget it honestly.** Doubling a tensor's bytes is a real cost: report the
   resident size and re-seed the perf rows **through the harness** with its own
   TTFT/power evidence (never hand-edit a row). If no placement that fits the
   card clears the gate, say so and leave the golden red — that is a recorded
   debt, not a golden to rewrite.

Known placements (measured, do not re-derive from scratch):

| family | plane on | left 4-bit | cost |
|---|---|---|---|
| phi-4 (14.7B dense) | every projection; `embed_tokens`/`lm_head` stay FP16 | — | 16.6 GiB, 71 t/s |
| gemma-4-26b-a4b-it (MoE) | attention projections + dense MLP | routed experts, tied table | +1.5 GiB, 157 -> 128 t/s |
| gemma-4-31b-it (dense) | needs *every* linear (~33.7 GiB) -> does not fit; partial placements move away from the source | — | gate stays red (`docs/open-tasks/live/models/gemma-4-31b-it.md` §R) |

Implementation checklist (all of it is exercised by tests in the phi-4 and
gemma-4 changes — copy their shape):

- converter: `gpu::quantize_residual_tensor_gpu(...)` from the source bytes and
  the base triple; emit the residual as the **same** record schema with a
  `.res` name suffix (`<name>.res.{packed,scales,global_scale}`); write an
  index field (`"residual": true` / `"attn+mlp"`) for humans and the VRAM
  estimate; extend the family's `VramTensorPolicy::has_residual` so
  `EstimateVram` doubles exactly those tensors.
- loader: one `LsxNvfp4Weight` member per plane, uploaded from the suffix when
  present and **left null on a legacy 4-bit-only tree** (the decode skips the
  accumulate pass, so old artifacts keep loading); release every plane in
  `Free()`.
- inference: after each base GEMV/GEMM, one more call with `acc = true` into the
  same buffer (the kernels already support it, so the arithmetic is "base dot
  rounds, then the residual dot adds and rounds once more").
- tests, written red first: converter (records exist, `base+res` reconstruction
  at least 2x better than `base`, and *no* plane on norms/router/edge tensors
  you deliberately excluded), loader (members uploaded with the base shape, and
  null where the converter wrote none), and the arithmetic of the accumulate
  pass against a host reference that rounds between the two dots.

### 3. Registration in the factory (LsxModelFactory)

- Register the string `kType` in `libs/lsxcommon/src/model_registry.cpp` via
  `LsxModelFactory::Register(kType, ...)` (pattern: `QwenV5Model::kType =
  "qwen3.5-9b"`).
- Implement a `final` `IModel` subclass under `libs/lsxcommon/src`, with
  `Load` / `Infer` / `Unload` / `Inspect` / `SelfCheck` / `Tokenizer`.
- Lock the geometry with constants + `static_assert` **and** cross-check the
  loaded passport against them (a frozen-geometry mismatch must fail Load).
- Extend `test_model_independence_guard.cpp`'s `consumers` list with the new
  model file in the same commit.

### 4. Per-byte tokenization (Token Bridge)

- Register the BPE vocabulary through `BuildTokenizerArrowFiles`
  (`libs/lsxcommon/src/tokenizer_convert.cpp`); merges may be string `"a b"` or
  array `["a","b"]`.
- **ALIGNMENT INVARIANT**: read `bos_id`/`eos_id`/`pad_id` from the model config
  (do not hardcode in the kernel); generate the service prefixes exactly as the
  original tokenizer does. `bos_id >= 0` → a real begin-of-sentence id, else
  `bos_id = -1`.

### 4b. Tool-calling bridge (conversation + tool calls) — MANDATORY

A model served through the `taskmosaic` OpenAI/Anthropic bridge must answer the
tool-calling surface, not only single prompts. This is a sub-step of stage 4
(host-only token bridge; cited in code as "Model Ingestion Protocol step 4b").
An ingested model is **not done** without it.

- Implement the two `IModel` virtuals next to the model class:
  `BuildConversationInputIds(tok, ChatConversation, out_ids)` and
  `ParseToolCalls(tok, gen_tokens, out_calls)` (types in
  `libs/lsxcommon/include/lsxcommon/chat_conversation.h`). The `model.h`
  defaults are the **generic Hermes fallback** (`<tools>` block +
  `<tool_call>{json}</tool_call>`); they are a floor, not a pass.
- **Verify the native format before coding — do not guess.** Read the
  checkpoint's own `chat_template.jinja` (or the vendor encoder, e.g. DeepSeek's
  `encoding/encoding.py`) and inspect `tokenizer.json` `added_tokens` (which
  markers are `special`). Reproduce the verified format; a guessed template
  silently breaks tool calling.
- Inject control markers as their own vocab ids via `arrow_tokenizer_lookup` and
  BPE-encode only free text: `arrow_tokenizer_encode` is pure BPE and never
  resolves `added_tokens`. Drop the auto-prepended `bos_id` inside a ChatML frame.
- Parse from decoded text when the markers are `special:false` (they survive
  `arrow_tokenizer_decode`); scan token ids when a marker is `special:true`.
  Fall back to `tools::ParseToolCallsText` when the native markers are absent.
- A model that emits a BARE JSON object (`{"name": …, "arguments": …}`) —
  Granite-4.2 is the verified example — needs no override at all: the
  `model.h` default already tries `tools::ParseBareToolCallJson` after
  `ParseToolCallsText` (`include/lsxcommon/tool_format.h`). Do NOT add a
  "native" renderer for such a family without re-verifying the round-trip
  against real weights.
- Shared codecs live in model-neutral `tools::` (`tool_format.{h,cpp}`) or a
  guard-allowlisted family file (e.g. `deepseek_moe_tool_format.{h,cpp}` — add
  the stem to `AllowedCommon()` in the same commit).
- Reference implementations: Qwen3.5/3.6 ChatML XML (`qwen_tool_format.{h,cpp}`),
  GLM-5.3-Flash (`glm_moe_flash_model.cpp`), DeepSeek-V4/V4.1 DSML
  (`deepseek_moe_tool_format.{h,cpp}`), Gemma-4 (`gemma_tool_format.{h,cpp}`).
- Test (host-only, label `fast`): render structure, parse, and a tokenizer
  round-trip on a **synthetic tokenizer fixture** in `tests/data/` (add the
  sentinels to its `added_tokens`); reference `tests/test_qwen_tool_framing.cpp`,
  `tests/test_glm_flash_tool_framing.cpp`, `tests/test_deepseek_tool_framing.cpp`.
  Do NOT round-trip `ParseToolCalls` from the full rendered prompt — its
  instructions embed an example tool call that the parser would match.

### 5. Low-level CUDA kernels (inference.cu)

- Write/extend the inference kernels for the model's attention specifics
  (hybrid example: GatedDeltaNet + full attention in
  `libs/lsxcommon/src/qwen_inference.cu`).
- Use the model-neutral NVFP4 layer (rule above); no family-owned dispatcher.
- **Stream invariant** and **stride invariant**: all Q/K/V address offsets follow
  the physical tensor geometry — no overlap of adjacent time intervals in
  strided conv/recurrence buffers.
- **KV cache layout**: address the cache through `KvCacheGeometry`
  (`libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h`) — three strides, so any
  permutation of (head, pos) is a choice of two numbers — never a hand-rolled
  `* heads + h` index. `test_model_independence_guard` enforces this; its
  exemption list carries a reason per entry, and you delete your row when you
  migrate. The knob is `ModelInitConfig::kv_cache_layout` and
  `IModel::KvLayout()` must report what you actually built, so a family that
  ignores it is visible rather than silent.
- **The layout itself is a MEASURED choice, not an inherited default.** Ship
  pos-major (`[pos][head][dim]`) unless you have an end-to-end number saying
  otherwise. On Xing-4.0 head-major reads K/V 6-10% faster in isolation at ctx
  4096/8191 and is 2-6% **slower** end to end at 2048/4096/8191 tokens: with the
  rest of the step competing for L2, isolated read bandwidth does not predict the
  step. Both layouts are bit-exact (`test_xing_kv_layout_parity`), so trying the
  other is a config change — measure on the AOCC perf tree, don't assume. If your
  attention maps one THREAD per head rather than one block (as in glm), the
  coalescing trade-off differs again and needs its own analysis.
- Do NOT ask for a bandwidth measurement at ingestion: the layout is chosen here,
  its cost is measured in `perf-tuning-a-model`.
- Isolate CUDA `.cu` files in their own OBJECT library (`-fmad=false` when
  bit-exact math is required).

### 6. Memory synchronization (Shared Memory Sandbox)

- Apply the shared-memory sandbox and deterministic-reduction rules from
  `nvfp4-blackwell-conventions` (thread-0-only RMW behind `__syncthreads()`;
  no `atomicAdd` in anything feeding `final_logits`). Reference implementation:
  `full_attn_softmax_kernel`.

### 7. Token handoff — retired 2026-10-24 (nothing to do)

The binary **token-translation passport** (`LsxTokenTranslator`, `IoWorker`'s
per-thread mapping, `tools/lsx_token_packer`) has been removed. It had no
caller: TaskMosaic tokenizes with the *target model's own* tokenizer
(`EncodeConversation(model_path, ...)`) and the fabric server consumes those
ids as-is, so ids never cross vocabularies; a cross-model hop is a text hop.

What a new family must still do: build the tokenizer arrows (step 4) and let
`BuildConversationInputIds`/`ParseToolCalls` own its framing. Do **not**
reintroduce a cross-family id map without (a) a real consumer, (b) a defined
unk policy, and (c) an end-to-end test from one model's ids into the other's
server.

### 8. TDD and SelfCheck harness

- Implement an incremental `SelfCheck()` (1→2 tokens) (pattern:
  `QwenV5Model::SelfCheck`).
- The control prompt `"The capital of France is"` MUST deterministically answer
  `"Paris"` (top-1 on the first decoded token / short prefix),
  `r.selfcheck_passed == true`.
- Add `tests/test_<model>_selfcheck.cpp` running ≥3 stress cycles with
  bit-exact `final_logits` determinism and `exit 0`; cycle count overridable by
  an env var (e.g. `LSX_TEST_<MODEL>_SELFCHECK_CYCLES`).
- The selfcheck ctest target MUST set `SKIP_RETURN_CODE 4` (real weights are
  optional; absence skips, never fails).
- Add the **FSI parity gate** — a separate stage from SelfCheck (capture from
  the source safetensors, committed golden, separate
  `test_<family>_fsi_selfcheck` target, suite-guard `fsi` row):
  **REQUIRED SUB-SKILL** `fsi-selfcheck`.
- Add the **MCP payload gate** — a REQUIRED stage, the fourth parity gate
  (`mcp` is in the suite guard's `kRequired`): `test_<family>_mcp_selfcheck` +
  `tests/fixtures/mcp/<model_type>.json` + `"mcp"` in the manifest's `stages` +
  the `{"form": "parity", "kind": "mcp"}` override + one
  `lsx_add_parity_gate(... KIND mcp)` line. The golden is **protocol-defined**
  (JSON-RPC 2.0), so it is captured tokenizer-only with **no forward pass** —
  `python3 tools/mcp_capture.py --model <ckpt> --type <model_type> --out …` or
  `tools/mcp_capture_all.py` for every row. A row whose checkpoint is gone and
  has no artifact cannot produce the framed prompt: declare `golden_debt.mcp`
  with the reason in the same commit. Sub-skill **`mcp-selfcheck`**.
- Add the perf baseline — **three** things, not two: an entry in
  `tests/perf/perf_baselines.json`, a `TEST_CASE("perf baseline: <model_type>")`
  in `tests/test_perf_baseline.cpp`, **and** an
  `add_test(NAME test_perf_baseline_<model> COMMAND test_perf_baseline "perf
  baseline: <model_type>")` in `tests/CMakeLists.txt` with label `perf` and
  `SKIP_RETURN_CODE 4`. `test_perf_baseline_logic` machine-enforces the pairing in
  both directions, so a missing `add_test` fails the fast suite even though a
  bare `TEST_CASE` looks like coverage when you read the source and is invisible
  in ctest. Measure by running the target, not by editing the JSON: only that path
  captures `ttft_ms` and the NVML power samples. `xing`, `mimo` and `alice` shipped
  without the `add_test` and `ctest -L perf` ran 16 of 19 families unnoticed.
- The determinism/correctness discipline these tests implement (bit-exact vs
  tolerance class, canonical-length validation, revert-on-red-determinism,
  `SKIP_RETURN_CODE 4`) is `Verifying a numeric change` in
  `perf-tuning-a-model`.

### 9. Model report artifact (closing stage)

Every ingestion ends with one **publishable report** in
`docs/model-reports/YYYY-MM-DD-<model-card-name>.md` (date prefix = the day the
real-weight gates were measured; then the model card / vault directory name,
e.g. `2026-10-31-Beam-1.0-501B-A23B.md`; the index is
`docs/model-reports/README.md`). A later re-ingestion of the same model adds a
NEW dated file instead of overwriting the old one, so the history stays
readable. This is an artifact, not a status note: a
later session must be able to re-read it instead of replaying the ingestion,
and an operator must be able to publish it as-is.

It has two clearly separated parts:

- **`Publish-ready post`** — English, no repository-internal paths or
  codenames. State what was converted and on what hardware; what the
  checkpoint forced into the engine (the semantics that are easy to get
  wrong) and which reference was used; the artifact-level defects/traps found
  in the checkpoint (config/tokenizer/generation-config), each with the
  observed failure mode and a concrete suggestion to the authors; measured
  numbers (gates, determinism, tool round-trip, perf); and what was **not**
  measured and why.
- **`Evidence`** — internal references: the stage-0 design and plan,
  the `docs/OPEN_TASKS.md` row + its `docs/open-tasks/live/` file, `docs/SELFCHECK_MATRIX.md`, perf rows,
  and the exact gate results. A SKIP/debt entry stays a gap; never rewrite it
  as a pass.

Write it **after** the real-weight gates (stage 8) so the numbers in the post
are measured, not projected. Do not include secrets, customer data or
unverified claims; every number must point at the evidence section.

## Host-side weight storage — `ExpertStore` and Arrow zero-copy

The routed-expert set is the largest thing a model owns, and it is the one place
where a loader can silently trade tens of gigabytes of host RAM for nothing.
Two rules, both machine-visible in `expert_store.h`:

**R1 — the host expert store is the neutral `ExpertStore<TEntry>`, never a
family-private container.** Declare a view struct whose field order matches what
the shared `ExpertStore<TEntry>::DecodeEntry()` aggregate-fills, alias it
(`using XExpertDb = gpu::expert_store::ExpertStore<XExpertEntry>;`), and expose
`load_*_expert_index` / `ensure_*_expert_resident` / `free_*_expert_index`.
`ExpertStore` keeps the `mmap` + `RecordBatch` alive in `mmaps`/`batches` and
bounds how many entries stay mapped via `max_open`, so touching an expert costs
one mmap + IPC parse and **no heap copy**. The entry must stay a view: it is
`static_assert`ed trivially-copyable and ≤128 bytes. Do NOT reintroduce an
owning per-family map — that is precisely the `AliceExpertHost` shape that
existed in alice until 4d98fe2 (`std::unordered_map` of owning `PackedExpert`,
no eviction).

**R2 — Arrow reads are mmap-backed views unless the payload dies with the
upload.** Every file read goes through `arrow::io::MemoryMappedFile::Open`
(never `ReadableFile::Open`), and a retained payload must come from a view
reader — `LoadTensorBatchViews` / `LoadDenseNvfp4BatchViews` /
`LoadSharedExpertBatchViews`, or `ExpertStore::Load` — never from
`PinnedSchema::UnpackBatch` / `ExpertSchema::UnpackBatch` /
`DenseNvfp4Schema::UnpackBatch`, whose `TensorRecord`/`PackedExpert`/
`DenseNvfp4Record` each own a `std::vector<uint8_t>` **copy** of the tensor.
`UnpackBatch` is legitimate exactly when the record is transient (converted to
BF16/F32, uploaded, freed in the same scope). The reference implementation of
R2's validation is `tokenizer.cpp`: it `dynamic_pointer_cast`s the columns and
checks every buffer against the mapping bounds (`in_range`) before storing raw
pointers, so a non-mmap Arrow buffer fails loudly instead of being copied.

Two sanctioned storage shapes, both on the neutral store:

- **views + LRU** (default). `max_open` caps *mapped file*, so host memory is
  bounded by the cap, not by the model. Geometry (`num_layers`,
  `experts_per_layer`) is a **parameter** taken from the loaded
  `model.index.json` — never a second copy of the passport constants, or the
  synthetic fixture (8×4) breaks against the real model (48×512).
- **`PinChunked` views** (documented exception). When the expert set is small
  enough to pin and a pinned H2D staging buffer is genuinely wanted, pin the
  store's own chunks — `ExpertStore::PinChunked<kStream|kSplitFence>` — rather
  than materializing a second copy. Kimi (542 GiB) and Gemma (12.5 GiB) do this;
  both remain bounded and are not the pattern to copy blindly.

Forbidden, and the reason: a pinned arena that holds **every** expert for the
life of the session. That design shipped in five models and cost the expert
tree on disk in host RAM — qwen3.8-flash-next **64 GiB** (plus a second full
disk pass), mistral **61 GiB**, qwen3.6 / ornith **17 GiB** each — and survived
review because a 750 GB host makes `cudaMallocHost` of tens of gigabytes
succeed silently. Alice's identical-class bug (an unbounded owning map) cost
1.7 MB of heap per expert and showed as a monotonic RSS climb, 8.3 → 12.7 →
18.0 GB peak over 32/128/512 generated tokens. All of it was retired on
2026-10-14 (qwen3.6 17.5 → 7.0 GB at unchanged throughput; qwen3.8 63.6 →
10.9 GB) — `docs/open-tasks/2026-10-14-expert-store-arena-retired.md`.

Two consequences of the fix are traps for the next family; the first is recorded
as a negative result in
`docs/open-tasks/2026-10-14-qwen38-arena-required-negative.md`, the second was
refuted on 2026-10-23 in
`docs/open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md` (§V2):

- **§V1 — a pinned arena is not waste when the decode working set IS the
  model.** If every token routes to experts spread over the whole expert set,
  an LRU cap below that means nearly every touch is a re-mapped, unpinned
  expert, and pinning the cap buys nothing (measured: 9.2 t/s either way) while
  pinning everything costs 2x the arena because the views stay mapped beside the
  pinned copy. Check your working set against `max_open` before migrating; the
  store is a win only when it fits. And `cudaHostRegister` cannot pin a
  file-backed mapping at all, so a "cheap pin over the views" is not an option.
- **§V2 (closed, not reproduced) — `max_open` does bound RSS**, monotone and
  saturating at the whole tree (Mistral 3.2 / 7.0 / 31.1 GB at cap 64 / 512 /
  8192), and a geometry pass costs no payload pages because `DecodeEntry` takes
  Arrow views. If a family measures otherwise, repeat the measurement before
  believing it: the first version of that finding came from a single run and was
  an artifact.

## Model code file-naming convention

```
^<family>_<type>(_<variant>)?(_exp)?_<role>(_<subsystem>)?$
```
in `libs/lsxcommon/{src,include}` and `tests/`. The **regex in
`tests/test_code_naming_guard.cpp` (`SchemeRegex()`) is the contract** — the
token sets below are what it currently accepts.

- `family` ∈ {glm, qwen, gemma, kimi, deepseek}
- `type` ∈ {moe, dense} — always required
- `variant` ∈ {27b, flash, 36, v41} — only to disambiguate two models of one `type`
- `_exp` infix (after type/variant, before role): the Qwen3.8 "Flash-Next"
  (`qwen4_exp`) marker, e.g. `qwen_moe_exp_convert.cpp`
- `subsystem` — accepted ONLY on the `inference` role:
  {attn, csa, head, moe, pipeline, exp}
- `role` ∈ {convert, load, repack, load_internal, model, inference, kernels,
  expert_kernels, expert_cache, device_math, dequant, norm_kernels, fp8, constants}
- Family-prefixed files that intentionally carry NO type marker (allowlist in
  `AllowedCommon()`): `qwen_inference`, `qwen_base_model`,
  `qwen_inference_load`, `qwen_device_math`, `qwen_reduce`, `gemma_base_model`,
  `deepseek_moe_tool_format`.
- The legacy glued prefix `gemma4` is recognised.
- Model versions are NOT in file names; they live in class names and `kType`.
- Files without a family stem are model-neutral infrastructure and are not
  scanned by the naming guard.
- **A new family / variant / role / subsystem token must be added to
  `SchemeRegex()` in the SAME commit as the files that use it** — otherwise the
  guard rejects the file names.

## The manifest is the only registry you write by hand

`models/<org>/<model>.json` is the single source of truth for one ingested model.
Write it (stage 1), then everything the guards and ctest need is GENERATED at
configure time into `${CMAKE_BINARY_DIR}/generated/`:

| artifact | replaces |
|---|---|
| `model_manifest.h` | `kPassports`, `FamilyStems()`, `kGoldenDebt`, `Suites()` and `kLifecycle` — the guards now project this table |
| `model_targets.cmake` | every model target's `add_test` + label / timeout / `SKIP_RETURN_CODE` / `RESOURCE_LOCK` / environment |
| `model_registry.inc` | the `LsxModelFactory` registration list + vault-normalized aliases |

`test_manifest_sync` (label `fast`) fails when the compiled table disagrees with
the manifests, when `docs/MODELS.md` is stale (`--check`), when a test file is
claimed by no manifest (`--strict`) or when a `docs/SELFCHECK_MATRIX.md` row
names a model that no manifest declares.

**A manifest is not a form to fill in loosely.** `--self-test` proves five schema
rules and `--diff-cmake <oracle>` proves the resulting target set: 421 targets
reproduced with zero differences is the standard this registry was migrated at.

### What the manifest must contain

```jsonc
{
  "model_type": "flint",              // == T::kType == LsxModelFactory key
  "family": "flint",                  // shared across siblings; variants differ by "variant"
  "variant": null,
  "model_class": "FlintMoeModel",
  "model_header": "flint_moe_model.h",       // relative to libs/lsxcommon/src
  "passport": "libs/lsxcommon/src/flint_moe_constants.h",
  "namespace": "lsxcommon::flint::frozen",
  "format_tag": "logestix-flint-nvfp4",      // only where a constant exists
  "forge_flag": "--flint",
  "tokenizer_kind": 4,
  // The checkpoint's place in the ORG-NESTED store: the file lives at
  // models/<org>/<dir>.json, so org and dir are the path and the generator
  // REFUSES a row whose location disagrees with this block.
  "vault": { "org": "RecursiveSI", "dir": "flint", "env": "LSX_TEST_FLINT_ROOT" },
  // Invariant I4: one checkpoint, N native trees. Each profile is an artifact
  // with its OWN codecs and its OWN measured gates; `built` requires a digest,
  // `not-built` may carry no metrics at all.
  "preferred_profile": "nvfp4",
  "profiles": {
    "nvfp4": { "status": "not-built" },
    "bf16":  { "status": "not-built" }
  },
  "lifecycle_tu": ["flint_moe_model.cpp", "flint_moe_inference_pipeline.cu"],
  "stages": { "convert": [...], "load": [...], "model": [...], "kernels": [...],
              "selfcheck": [...], "fsi": [...], "code": [...], "det": [...],
              "perf": [...] },
  "perf": { "vault_subdir": "flint", "timeout": 3000, "resource_lock": "gpu" },
  "target_overrides": { "test_x": { "label": "inference", "timeout": 1800 } },
  "tool_roundtrip": true, "auto_path_determinism": true,
  "cooperative_cancel": true,        // the decode loop polls req.abort_flag
  "selfcheck_frame_prompt": false,   // Paris probe via BuildPromptInputIds
  "selfcheck_max_gen": 2,            // the family's own decode budget
  "selfcheck_bit_exact": true,       // FastMathMode::kOff, not kAuto
  "golden_debt": { "fsi": "why no source golden can exist" },
  "aliases": ["flint-1.0"],           // must be the lower-cased vault dir
  "report": "docs/model-reports/....md", "notes": "one line"
}
```

Validation refuses: an unknown stage, an empty stage without a `golden_debt`
reason, a missing passport or namespace, a `model_class` whose `kType` differs
from `model_type`, a target with no source, a shared vault directory, an alias
that is not the lower-cased vault dir, or a second manifest reusing a namespace.

### Which test files still exist per model

Most stages are ONE target with its own TU (`test_<family>_<stage>.cpp`). Three
are shared drivers parameterised by the manifest:

- FSI / CODE / DET → `tests/test_parity_gate.cpp`, compiled once per
  (model, kind) with `LSX_GATE_MODEL` / `LSX_GATE_KIND`. Do **not** add a
  per-family gate file: it is how three files came to SKIP with a neighbour's
  model name.
- SelfCheck → `tests/test_selfcheck_driver.cpp` with declared hooks
  (`paris_cycles` always; `tool_roundtrip`, `auto_path_determinism`,
  `cooperative_cancel`, `selfcheck_frame_prompt`, `selfcheck_max_gen`,
  `selfcheck_bit_exact` per flag). A family case that is not one of those hooks
  keeps its own file and target (`test_qwen_moe_exp_eos_selfcheck` is the
  precedent).

  **The probe is data, not a constant.** `selfcheck_frame_prompt`,
  `selfcheck_max_gen` and `selfcheck_bit_exact` are not decoration: `final_logits`
  is the LAST decode step, so `selfcheck_max_gen` alone decides the recorded
  `logits_hash`, and the math mode decides it too. Declare all three from
  measurement of the SOURCE family — never copy a sibling's numbers, and never
  "normalise" them, which is exactly how 35 families' hashes were silently
  re-baselined in the 2026-11-06 migration (`docs/open-tasks/2026-11-20-open-tasks-index-split-closed.md` §G5). Prove a
  probe change by replaying a binary built BEFORE it: same top-1, same `n`, same
  hash.

### Guards you still run (their data is generated, their rules are not)

| Guard (in `tests/`) | What it enforces now |
|---|---|
| `test_manifest_sync.cpp` | the generated table equals the manifests; the committed catalogue is current; no unclaimed test file; every row resolves through `LsxModelFactory` |
| `test_model_independence_guard.cpp` | unique frozen passport per model, no passport→passport includes, no cross-family geometry borrowing, `LsxFastMathScope` in `Infer`, and the shared lifecycle contract (residency lock, streaming/abort, context teardown, governor lease, result fill, session state under the lock) — all projected from `kModels`/`kLifecycle` |
| `test_model_suite_guard.cpp` | full lifecycle coverage per model, `SKIP_RETURN_CODE 4` on the gate targets, no cross-family fixtures, no vault loading in loader/model stages, every family target claimed, the stage-4b tool verdict, and the golden/debt pairing |
| `test_code_naming_guard.cpp` | file-naming scheme |
| `test_ctest_labels_guard.cpp` | one label per target (it reads the generated fragment too), skip-capable targets set 4, convert targets carry a bounded TIMEOUT |
| `test_absl_format_guard.cpp`, `test_logging_guard.cpp`, `test_expert_store_guard.cpp` | unchanged |

## Definition of done

- [ ] `kType` registered; `Load` forbids a frozen-geometry mismatch.
- [ ] Verification **scoped to this model**: `ctest --preset linux-debug` (fast
      guards, parallel) green, and every target of this row green —
      `ctest --test-dir build/linux-debug -R "<model_type>" -j 1`. The label
      sweeps are NOT part of a model ingestion (see "Verification"); if you ran
      one anyway, or skipped one, say which and why.
- [ ] Converter test (label `convert`, bounded TIMEOUT, synthetic checkpoint).
- [ ] Loader/model/kernels tests present and labelled.
- [ ] Streaming + cancellation wired: one `Emit` per decoded token through
      `req.token_callback`, `req.abort_flag` honoured between steps (pinned by
      `tests/test_streaming_assert.h` and a pre-set-abort case in the model
      stage; enforced by the `kLifecycle` guard).
- [ ] Tool-calling bridge implemented (stage 4b): native
      `BuildConversationInputIds`/`ParseToolCalls` verified against the
      checkpoint template, plus a host `fast` test on a synthetic tokenizer.
      The verdict (target name, or "generic") is written into the tool-verdict
      table of `test_model_suite_guard.cpp` in the same commit.
- [ ] `models/<org>/<model>.json` written and `python3
      tools/gen_model_manifest.py --strict` clean; `--docs docs/MODELS.md`
      regenerated in the same commit.
- [ ] SelfCheck ≥3 bit-exact cycles (the shared driver does it; a family case
      outside the hooks gets its own file).
- [ ] FSI/CODE/DET/MCP goldens captured and the gates **run** (a SKIP is not a
      pass). MCP's golden is protocol-defined, so it is captured tokenizer-only
      with no forward pass; if the checkpoint is gone and no artifact exists,
      declare `golden_debt.mcp` in the same commit. If a gate is red on a
      borderline answer, follow "Parity fix: the residual plane" in stage 2 —
      and record the outcome (green, or the measured reason it cannot be green on
      this card) in `docs/SELFCHECK_MATRIX.md` + `docs/OPEN_TASKS.md`.
- [ ] perf entry in `tests/perf/perf_baselines.json` + a `TEST_CASE` in
      `test_perf_baseline.cpp`; the `perf` stage target in the manifest supplies
      the ctest registration (label `perf`, `SKIP_RETURN_CODE 4`).
- [ ] No hand-written registry: if you edited a guard registry or a
      `tests/CMakeLists.txt` registration for a model target, stop — the
      manifest is missing a field.
- [ ] Routed experts are read through the neutral `ExpertStore` (R1/R2 below):
      `test_expert_store_guard` green, and a new family that needs the expert
      host store must NOT appear in that test's `kLegacyConsumers`.
- [ ] `docs/MODELS.md` regenerated (it is the per-model catalogue; `AGENTS.md` is
      a router and needs no model section); `test_model_suite_guard` and
      `test_code_naming_guard` green.
- [ ] Model report artifact (stage 9) written to
      `docs/model-reports/YYYY-MM-DD-<model-card-name>.md`, with a
      publish-ready English
      post and an evidence section; every number it quotes matches the gate
      files, and declared debt is named as debt.

## Verification

**Scope the sweeps to the model you are ingesting.** A model-local change cannot
regress another family, and the label sweeps are enormous: on 2026-11-22
`linux-debug-inference` held **350 targets across 43 models** (qwen 225, gemma 77,
…) and every device target carries `RESOURCE_LOCK "gpu"`, so the sweep is `-j1`
and runs for hours. Ingesting one model and then running all of it measures 340
targets the diff cannot touch, which is why a "full sweep" was once started three
times and abandoned three times. What a model ingestion actually has to prove:

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug

# 1. The guards that CAN be broken by any diff (parallel, ~3 min, no GPU lock).
#    This is the whole "did I break the repo" question, and it is label `fast`.
ctest --preset linux-debug -j 24

# 2. YOUR row's own lifecycle targets, by the manifest's model_type. This is the
#    model's convert → load → model → kernels → selfcheck → fsi/code/det → perf
#    set, derived from models/<org>/<model>.json — there is nothing to look up.
ctest --test-dir build/linux-debug -R "<model_type>" -j 1

# 3. Optionally the immediate siblings that share the TUs you edited. Name them
#    explicitly; never widen this to "the family" or "the sweep".
ctest --test-dir build/linux-debug -R "<model_type>|<sibling>" -j 1
```

Real weights: run the model's selfcheck/parity targets on the **Release/AOCC
tree**, not Debug. A Debug real-weight gate is 5-20x slower for the same verdict
(the varloop selfcheck is 15 s in `linux-release-znver4-aocc` and minutes in
`linux-debug`), and perf targets `SKIP` in Debug anyway.

```
ctest --preset linux-debug                                  # fast only
ctest --test-dir build/linux-debug -R "<model_type>" -j 1   # this model
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake   # dry-run router
```

The full sweeps stay available and are the right tool for **infrastructure**
work (a shared kernel layer, a schema, the router, the manifest generator):

```bash
ctest --preset linux-debug-convert      # converter sweep (infra changes)
ctest --preset linux-debug-inference    # every model/CUDA target (infra changes)
```

**Report what you did not run.** If you scope the sweeps, say so in the report
and name any red you saw in a scope you skipped — a green `-R <model_type>` is
not evidence about the other 42 rows, and a pre-existing red discovered during a
cancelled sweep belongs in `docs/OPEN_TASKS.md` (or in an existing row, if it is
already there — check before filing).

The guard tests above MUST be green before the model is considered
ingested. They are all label `fast`, so `ctest --preset linux-debug` is what
proves them — no sweep is needed for that, and none was ever needed.

## Common mistakes

- **Running a full label sweep to verify one model.** It is hours of GPU-serialised
  work on 340 targets the diff cannot touch; use the fast preset plus
  `-R "<model_type>"`.
- Adding kernels or a converter without the failing test first (TDD violation).
- Reading geometry from `config.json` at runtime instead of frozen constants.
- Violating a kernel-level rule that already lives in
  `nvfp4-blackwell-conventions` (a family-named NVFP4 dispatcher, a shared-memory
  race, an `atomicAdd` feeding `final_logits`).
- Forgetting a ctest label, or a selfcheck target without `SKIP_RETURN_CODE 4`
  (a skip then reports as a failure).
- Shipping a model without the stage 4b tool-calling bridge, relying on the
  generic Hermes default when the checkpoint has its own tool format.
- Accepting a `GenerationStream*` in the generation entry point and never using
  it: the model then streams nothing through `--serve` and ignores cancel.
- Mutating persistent session state (the KV cache, the generated tail) outside
  `RunUnderResidencyLock` — a concurrent `Infer` then frees the cache the other
  one decodes from.
- Counting a partial device footprint in `resident_bytes_` (it drives the
  governor lease and the `resident:` line of `lsxbenchmark --bench`); count at
  the allocation sites instead of re-deriving a formula.
- Freeing per-call device buffers only on the success path: CUDA errors are
  sticky, so one failure then leaks a full workspace per retry. Use RAII.
- Guessing the native tool format instead of reading `chat_template.jinja` /
  the vendor encoder and the tokenizer's `added_tokens`.
- Round-tripping `ParseToolCalls` from the full framed prompt — the instructions
  embed an example tool call that the parser then matches.
- Loading the real model vault inside a loader/model-stage test (forbidden —
  vault integration belongs to selfcheck/perf suites; loader/model stages use
  synthetic converted fixtures).
- Borrowing another family's test fixture instead of
  `tests/test_synth_checkpoint.h`.
- Writing a family-private routed-expert host container (a `std::unordered_map`
  of owning `PackedExpert`, a private pinned arena) instead of aliasing the
  neutral `ExpertStore`. This is the highest-cost mistake in the list and it is
  invisible on a big host: alice's version cost ~21 MB of RSS per generated
  token and the qwen/mistral arena loaders retain 17–64 GiB of pinned RAM each,
  yet all of them load without an error. See the host-storage section above.
- Keeping a `UnpackBatch` record alive past the scope that uploads it — the
  record owns a full `std::vector<uint8_t>` copy of the tensor. Read it as a
  view when the payload is retained.
- Adding a residual (or any extra) weight plane in the loader without adding it
  to `Free()`. The leak is invisible on one model load and shows up as an
  `out of memory` on the *next* one — on gemma-4-31b it surfaced as a SelfCheck
  failure in the following test case, not as a leak report.
- Treating an offline HF-side quantization simulation as a recipe oracle. It is
  qualitative: its all-4-bit baseline is not the engine's, so a placement it
  "proves" sufficient can be useless (or harmful) in the engine. Calibrate by
  re-converting the engine's own tree.
- Shipping without the stage 9 model report (or writing the report before the
  real-weight gates, so its numbers are projections and drift from the tree).
- Rewriting a golden, or a perf row, to make a gate pass. Goldens are source
  parity (`docs/OPEN_ENDEDNESS.md`); perf rows are re-seeded through the harness
  with its own evidence, never by hand.
