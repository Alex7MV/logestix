# Inference Unification — Design Specification

## Overview

Unify inference across the three engines (GLM-5.2, DeepSeek-V4-Flash,
Qwen3.5-9B) in `lsxcommon` and the four CLI surfaces (`lsxforge`, `lsxhome`,
`lsxfabric`, `lsxcli`). Three workstreams, one document, two implementation
phases:

- **Section 1 (items 1-6)** — eliminate the triplicated host-side scaffolding
  (tokenizer-db load, decode capacity-retry, SelfCheck "Paris", GLM framing,
  Infer-preamble guards, CLI lifecycle) into one `lsxcommon::runtime` layer.
- **Section 2 (items 7-10)** — a single immutable `gpu::InferenceResult`
  contract: `gen_tokens`, `output_text`, `final_logits`, precise CUDA/host
  timings, VRAM weight-residency accounting (Qwen resident-set baseline), and
  unified EOS routing.
- **Section 3 (items 11-15)** — the core loops: orchestrator ring closure,
  reusable pinned route-token staging (kills DeepSeek malloc-per-layer),
  isolated sampling module, unified `IModel::Benchmark`, and the
  prefill/KV-cache integration matrix.

Baseline anchor: **Qwen3.5-9B** is the residency and arena accounting reference
(per MODEL INGEST protocol). GLM is the single-allocation reference
(`GlmMoEArena::AllocateGeneration`); DeepSeek is the outlier to consolidate.

## Audit Findings (evidence)

| # | Finding | Evidence |
|---|---|---|
| A1 | Tokenizer-db mmap block triplicated verbatim | `glm_v5_model.cpp:35-50`, `deepseek_v4_flash_model.cpp:41-55`, `qwen_base_model.cpp:32-45` |
| A2 | Decode `-2` capacity-retry triplicated | `glm_v5_model.cpp:159-179`, `deepseek_inference_pipeline.cu:709-721` |
| A3 | SelfCheck "Paris" + `max(2,...)` forcing in 3 models | `glm_v5_model.cpp:207-246`, `qwen_base_model.cpp:136-188`, `deepseek_inference_pipeline.cu:804-823` |
| A4 | GLM chat framing re-implemented in lsxfabric | `prompt_framing.cpp:32-78` vs `lsxfabric/src/main.cpp:23-67` (`EncodeSelfCheckPrompt`) |
| A5 | Infer-preamble guards 6x; Qwen's `SelfCheck` misses 2 of 3 | `glm_v5_model.cpp:58-69+199-206`, `deepseek_v4_flash_model.cpp:63-74+94-105`, `qwen_base_model.cpp:72-83+139-142` |
| A6 | CLI lifecycle hand-rolled per app; lsxhome `--run` is a demo pump | `lsxforge/main.cpp:240-276`, `lsxcli/main.cpp:35-50`, `lsxfabric/main.cpp:95-156`, `lsxhome/main_win32.cpp:72-92` |
| A7 | Result contract gaps | Qwen drops `gen_tokens`/`output_text`/`final_logits`; GLM drops `kv_slots_used`; DeepSeek `ttft_ms = elapsed/total_steps` is wrong (`pipeline.cu:722`); `final_loss` never assigned anywhere |
| A8 | DeepSeek malloc-per-layer | 36 direct `cudaMalloc` + ~246 CSA + 2×(layer×token) route allocations per run (`deepseek_inference_pipeline.cu`, `deepseek_inference_moe.cu`) |
| A9 | Zero VRAM residency tracking | no `cudaMemGetInfo` anywhere; only GLM `kv_slots_used`, dropped at `glm_v5_model.cpp:152` |
| A10 | Orchestrator ring open | `DispatchOne` routes with empty `ModelRequest req` (`lsx_orchestrator.cpp:150,166`) — no tokenizer wiring, no result |
| A11 | No batched prefill; Qwen decode is full-recompute | `qwen_inference.cu:941`; GLM MoE collapses seq→row 0 (`glm_inference.cu:652-658`) |

---

## SECTION 1 — Code Duplication Elimination & Unification Pipeline

### Module layout

New host-only module (no CUDA, no absl containers — safe from `.cu` TUs):

- `libs/lsxcommon/include/lsxcommon/model_runtime.h`
- `libs/lsxcommon/src/model_runtime.cpp` — namespace `lsxcommon::runtime`
- `libs/lsxcommon/include/lsxcommon/model_base.h`
- `libs/lsxcommon/src/model_base.cpp` — thin `ModelBase` for the two patterns
  100% identical across all models (tokenizer-db member + preamble guard)

`ModelBase` state: `std::unique_ptr<ArrowTokenizerDb> tok_db_`, `bool loaded_`,
`bool weights_loaded_`, `ModelInitConfig cfg_`, `bool tokenizer_optional_`.

### Item 1 — Tokenizer db loading

| | |
|---|---|
| Dup sites | `glm_v5_model.cpp:35-50`, `deepseek_v4_flash_model.cpp:41-55`, `qwen_base_model.cpp:32-45` |
| Target | `bool ModelBase::LoadTokenizerDb(const std::string& model_path)` — mmap `logestix/pinned/tokenizer_vocab.arrow` + `tokenizer_merges.arrow`, `load_tokenizer_metadata` into `tok_db_`; false + spdlog on failure |
| Qwen divergence | tolerant path preserved via `tokenizer_optional_ = true` (warn + continue, `qwen_base_model.cpp:44`) |
| Deprecation | the 3 inline mmap blocks deleted; each `Load()` becomes `if (!LoadTokenizerDb(cfg_.model_path)) return false;` |

### Item 2 — Decode capacity-retry

| | |
|---|---|
| Dup sites | `glm_v5_model.cpp:159-179`, `deepseek_inference_pipeline.cu:709-721` |
| Target | `std::string runtime::DecodeGeneratedText(const ArrowTokenizerDb&, const std::vector<int32_t>& ids)` — internal capacity-grow loop (start 512, double on `-2`, ceiling 1 MiB), `""` + spdlog warn on persistent failure |
| Deprecation | both inline `out.resize(cap)` retry blocks deleted; DeepSeek `.cu` calls the host-only helper directly |

### Item 3 — SelfCheck "Paris"

| | |
|---|---|
| Dup sites | `glm_v5_model.cpp:207-246`, `qwen_base_model.cpp:136-188`, `deepseek_inference_pipeline.cu:804-823` |
| Target | `gpu::InferenceResult runtime::RunParisSelfCheck(IModel&, const ModelRequest&)` — force `max_gen_tokens = std::max(2, req.max_gen_tokens)`, delegate to `model.Infer`, 1→2-token prefix decode + `absl::StripAsciiWhitespace` + `== "Paris"`, fallback 8-token decode for `first_gen_text`; sets `selfcheck_passed`; logs `[selfcheck] top1_tok=... text='...' passed=...` |
| Deprecation | all 3 `SelfCheck()` bodies become `return runtime::RunParisSelfCheck(*this, req);`; `RunSelfCheckModel` in `deepseek_inference_pipeline.cu` deleted |

### Item 4 — GLM chat-template framing

| | |
|---|---|
| Dup sites | `prompt_framing.cpp:32-78` (canonical) vs lsxfabric `EncodeSelfCheckPrompt` `main.cpp:23-67` |
| Target | single canonical `prompt::BuildPromptInputIds`; add a static per-model framing descriptor table so a non-GLM model can register future framing without forking the helper |

```cpp
namespace lsxcommon::prompt {
// Ordered sentinel names resolved via arrow_tokenizer_lookup at encode time.
// A model with no registered entry (or an incomplete sentinel set) encodes raw.
struct FramingDescriptor {
    std::string_view model_type;          // e.g. "glm-5.2"
    std::string_view leading[3];          // [0]=gmask [1]=sop [2]=user  (pre-prompt)
    std::string_view trailing[3];         // asst, think, endthink       (post-prompt)
    std::size_t leading_count, trailing_count;
};
// Registry: compiled static table, keyed by model_type. GLM-5.2 is the only
// entry in Phase A.
const FramingDescriptor* LookupFraming(std::string_view model_type);
}
```
| Deprecation | `EncodeSelfCheckPrompt` deleted from `apps/lsxfabric/src/main.cpp`; `lsxfabric --testprompt/--selfcheck` route through the shared helper |

### Item 5 — Infer-preamble guards

| | |
|---|---|
| Dup sites | 6 blocks: `glm_v5_model.cpp:58-69+199-206`, `deepseek_v4_flash_model.cpp:63-74+94-105`, `qwen_base_model.cpp:72-83+139-142` |
| Target | `bool ModelBase::ValidateInferRequest(const ModelRequest&, gpu::InferenceResult&)` — canonical `error_msg` strings, in order: `"model not loaded"`, `"empty input_ids"`, `"max_gen_tokens must be positive"`; false short-circuits |
| Deprecation | all 6 inline blocks deleted; each `Infer`/`SelfCheck` begins `if (!ValidateInferRequest(req, r)) return r;` (fixes Qwen's missing 2 guards) |

### Item 6 — Unified CLI lifecycle

| | |
|---|---|
| Dup sites | `lsxforge/main.cpp:240-276`, `lsxcli/main.cpp:35-50`, `lsxfabric/main.cpp:95-156`; `lsxhome/main_win32.cpp:72-92` demo pump (no model) |
| Target | `int runtime::RunCliGeneration(GenerationCliArgs)`, `int runtime::RunCliSelfCheck(...)`, `int runtime::RunCliInspect(...)` — each encapsulates Create→Load→frame→run→`=== ... ===` print→Unload→Shutdown; returns process exit code |
| Result printing | single `runtime::PrintGenerationResult(const gpu::InferenceResult&)` replacing the bespoke `lsxforge/main.cpp:268-272` and `lsxfabric/main.cpp:143-149` blocks |
| lsxhome | `--run` gets real model wiring: compute thread drives `runtime::RunCliGeneration`-style loop and publishes `output_text` tokens through `GuiBridge` (`lsxhome/include/gui_bridge.h`); Phase A tail, D3D12-side token-buffer change |

---

## SECTION 2 — Unified Inference Result Contract

`gpu::InferenceResult` (`inference_gpu.h:17-40`) extended **additively** — no
field removal. Engine result structs (`GlmGenerationResult`,
`QwenGenerationResult`, DeepSeek loop) gain the timing/occupancy fields so the
model boundary copies, never recomputes.

### Contract (final shape)

```cpp
struct InferenceResult {
    // lifecycle
    bool ok = false;
    std::string error_msg;           // canonical: "model not loaded" | "empty input_ids"
                                     // | "max_gen_tokens must be positive" | engine error
    // streams
    std::vector<int32_t> gen_tokens; // generated stream, EOS-excluded; num_tokens == size()
    std::string output_text;         // DecodeGeneratedText(gen_tokens); generated only, no prompt echo
    std::vector<float> final_logits; // raw [vocab] row of final decode step
    int first_gen_tok_id = -1;
    float first_gen_maxv = 0.0f;
    std::string first_gen_text;
    bool selfcheck_passed = false;
    // timings (engine-recorded cudaEvent deltas)
    double ttft_ms = 0.0;            // stream start -> first generated token
    double total_time_ms = 0.0;      // stream start -> final token
    double tokens_per_sec = 0.0;     // num_tokens / (total_time_ms/1000)
    // residency + occupancy
    std::uint64_t weight_resident_bytes = 0;  // peak resident weight set over the run (Qwen baseline)
    int32_t kv_slots_used = 0;       // KV occupancy at run end (GLM already computes)
    // EOS
    bool eos_terminated = false;     // distinguishes EOS stop from max-token stop
    // engine
    int32_t num_layers = 0;
    float final_loss = 0.0f;         // unused; retained for ABI stability
};
```

### Item 7-9 — publish contract

- **`gen_tokens`**: all three publish; `num_tokens == gen_tokens.size()`.
- **`output_text`**: all three = `runtime::DecodeGeneratedText(gen_tokens)`.
  DeepSeek keeps its leading-token semantics (`c5df62b`); GLM already excludes
  the prompt; Qwen gains decode.
- **`final_logits`**: GLM already publishes (`glm_v5_model.cpp:153`); Qwen
  captures at `qwen_base_model.cpp:99` but must copy into `r.final_logits`;
  DeepSeek adds one device→host copy of the final step's `[vocab]` row
  (`vocab×4B`, off the hot loop, once per run).

### Item 10a — timings

- `ttft_ms` = prefill + first decode forward. GLM/Qwen wrap the engine call in
  `cudaEventRecord` pairs on the stream they create inside `Infer`.
  DeepSeek records `t_ttft` at `step == input_ids.size()-1` in the existing
  loop (`pipeline.cu:679-684`). Per-step argmax readback syncs unchanged —
  events add no barrier.
- DeepSeek `ttft_ms = elapsed/total_steps` (`pipeline.cu:722`) replaced by the
  event deltas.
- `tokens_per_sec = num_tokens / total_ms` uniformly.

### Item 10b — VRAM weight-residency (Qwen resident-set model)

- New `virtual std::uint64_t ResidentWeightBytes() const` on `IModel`
  (default `0` — additive, no break to test `FakeModel`) — thread-safe read of
  per-model accounting: pinned weight bytes (recorded at load) + per-run arena
  + staged-expert bytes.
- `InferenceResult::weight_resident_bytes` = peak over the run, filled by each
  model from its engine result.
- **Enforcement** = exact-equality invariant, verified in tests:
  `weight_resident_bytes == pinned_weight_bytes + arena.bytes (+ staged-expert
  bytes)`. GLM/Qwen arenas already report `bytes` (`GlmMoEArena::bytes`,
  `glm_inference_load.cu:246`); engines record `pinned_weight_bytes` at load.
  GLM staged-expert ceiling: `2 × kGlmExpertMatEl/2 × 3` bytes
  (`d_pack`+`d_scale`); DeepSeek after Section 3 consolidation: same shape.
  No magic budget constant — the arena's declared footprint is the budget.
- `kv_slots_used`: surfaced from `GlmGenerationResult.kv_slots_used`
  (`glm_v5_model.cpp:152` today dropped); DeepSeek/Qwen gain occupancy counters
  (Section 3 matrix).

### Item 10c — unified EOS routing

All engines: read `eos_id` from `ArrowTokenizerDb` (per `tokenizer_convert.cpp`
invariant — never hardcoded); stop on first generated EOS
(`step >= prompt_len`); exclude EOS from `gen_tokens`/`output_text`; set
`eos_terminated = true`. GLM (`glm_inference.h:404-406`) and DeepSeek
(`pipeline.cu:686-687`) already conform; Qwen verified against the same rule.

---

## SECTION 3 — Core Computational Architecture Loops

### Item 11 — orchestrator ring closure

`DispatchOne` (`lsx_orchestrator.cpp:119-182`) becomes the closed loop:

1. Parse `ChatCompletion cc` (unchanged, zero-copy).
2. `Classify(cc.model)` → resolve `IModel* m` (`glm_`/`deepseek_`, pre-resolved).
3. **Tokenize**: `prompt::BuildPromptInputIds(m->Type(), m->Tokenizer(), cc.content, input_ids)`.
4. `ModelRequest req{input_ids, cc.max_tokens or 64}` — `max_tokens` added to
   `ChatCompletion` (`openai_endpoint.h:20-30`).
5. `auto res = m->Infer(req)`.
6. **Serialize**: new `lsxnetwork::CompletionResponse { ok, output_text,
   gen_tokens, eos_terminated, ttft_ms, tokens_per_sec }` +
   `OpenAiEndpoint::WriteCompletion(const ChatCompletion&, const gpu::InferenceResult&, std::string&)`.
7. **Publish** the serialized JSON into a new SPSC `FrameQueue` response ring
   (same zero-copy chunk protocol, `frame_queue.h:17`); the IoWorker ZMQ writer
   drains it (one pinned consumer — `ServeTaskMosaicPinned` unchanged).
8. `SlotRetire` release unchanged; `kUnknown` → 404/503 semantics unchanged.

Invariant: response frame carries the request `id` (correlation via the
JSON `id` field — the ring has no per-slot correlation channel).

### Item 12 — route-token pinned staging (malloc-per-layer kill)

New `DeepSeekInferArena` mirroring `GlmMoEArena::AllocateGeneration`
(`glm_inference_load.cu:1054`): **one** device `cudaMalloc` + **one** pinned
host block, carving the 36 direct buffers (hidden/FFN/KV/logits) + CSA state +
routing staging. `route_token` (`deepseek_inference_moe.cu`) drops internal
`cudaMalloc/cudaFree` per (layer, token) and takes caller-owned pinned
`h_topk`/`h_top_weights` + arena `d_logits`. `ensure_expert_resident`
(fd-LRU, `deepseek_inference_load.cu:399-440`) unchanged. Result: per-run
allocation count drops from ~300+ to 2.

### Item 13 — isolated sampling module (greedy + strategy interface)

New host-only `include/lsxcommon/sampling.h` (`lsxcommon::sample`, zero CUDA
headers / absl containers — safe from `.cu`):

```cpp
namespace lsxcommon::sample {
class ISampler {
public:
    virtual ~ISampler() = default;
    virtual cudaError_t Sample(cudaStream_t, const void* d_logits, int vocab,
                               int32_t* d_out_token, float* d_out_maxv) = 0;
};
class GreedySampler : public ISampler { /* wraps existing fused argmax outputs */ };
}
```

- Engines keep fused argmax kernels (no added barrier); `GreedySampler` is the
  seam where a TopK/TopP sampler later reads the full row.
- EOS check stays host-side in the shared decode contract (item 10c).
- `ISampler&` threaded into each engine's decode loop; default instance
  `GreedySampler`.

### Item 14 — unified `IModel::Benchmark`

- New `virtual gpu::BenchmarkResult Benchmark(int num_tokens = 2048)`
  on `IModel` (default `{ok=false}` — additive, no break to test `FakeModel`);
  each model runs its own engine loop on synthetic/dummy tokens.
- Free `gpu::RunBenchmark(model_path)` (`benchmark_gpu.h:20`) becomes a thin
  shim: factory create → `Benchmark()` → destroy. `lsxfabric --benchmark` and
  `lsxforge --benchmark` call sites unchanged.
- `BenchmarkResult` (`benchmark_gpu.h:9-18`) gains `std::uint64_t resident_bytes`
  (aligns item 10b).

### Item 15 — batched prefill / KV-cache integration matrix

| Engine | Prefill today | Phase B target | KV today | Phase B target | Occupancy |
|---|---|---|---|---|---|
| GLM | attn seq=P, MoE collapses to row 0 (`glm_inference.cu:652-658`) | seq=P through MoE slices (kernels already take `seq`: `RunGlmMoERouter`/`RunGlmMoESlices`) | IndexShare, `kGlmIndexMaxSlots=1024` | — | `kv_slots_used` surfaced (item 10b) |
| DeepSeek | per-token (HC/CSA sequential state machine) | per-token retained, documented as inherent | window 128 + compressed, `d_kv_cache[43][...]` | arena-owned (item 12) | counter added |
| Qwen | full-recompute decode (`qwen_inference.cu:941`) | incremental per-call KV (append `d_k/d_v`, persist GatedDeltaNet recurrent state) | none (recompute) | per-call KV cache | counter added |

Matrix is the compatibility contract. `ModelRequest` stays batch=1 in Phase B;
multi-sequence batching explicitly deferred (would extend `ModelRequest` and
all three engines). GLM's `seq`-capable kernels make it the first prefill
integration.

---

## Phasing

- **Phase A (items 1-10):** Section 1 module + Section 2 contract. Pure
  host-side dedup + result-contract extension. No CUDA loop restructuring
  except DeepSeek `final_logits` readback and engine timing fields.
- **Phase B (items 11-15):** Section 3 loops. `DeepSeekInferArena`, sampling
  module, `IModel::Benchmark`, orchestrator ring closure, prefill/KV matrix
  (GLM seq-P MoE first, then Qwen incremental KV).

One spec, two writing-plans cycles (Phase A plan, then Phase B plan).

## Testing

- **Item 1-5:** host-only Catch2 tests for `LoadTokenizerDb` (existing
  `synthetic_tok.json` fixture pattern), `DecodeGeneratedText` (capacity grow,
  blank, failure), `RunParisSelfCheck` (against a fake `IModel`), framing
  descriptor regression, `ValidateInferRequest` (3 canonical errors).
- **Item 6:** CLI smoke tests asserting `=== --run Results ===` output + exit
  codes; `test_generation_cli` extended.
- **Section 2:** contract tests per engine — non-empty `gen_tokens`/`output_text`
  on synthetic input, `final_logits.size() == vocab_size`, `weight_resident_bytes`
  within budget, `eos_terminated` flags correctly, timing monotonicity
  (`ttft_ms <= total_time_ms`).
- **Section 3:** arena allocation-count assertion (2 allocs per DeepSeek run);
  sampler seam (GreedySampler == fused argmax); Benchmark parity
  (`IModel::Benchmark` vs `RunBenchmark` shim); orchestrator ring close
  (frame in → response frame out, `id` correlation — extend
  `tests/test_integrated_pipeline.cpp`); GLM seq-P prefill vs seq=1 parity.
- Real-model tests SKIP (CTest `SKIP_RETURN_CODE 4`) without checkpoints,
  matching existing GLM/DeepSeek real-weight suites.

## Files Changed (Phase A)

| File | Change |
|---|---|
| `libs/lsxcommon/include/lsxcommon/model_runtime.h` | new |
| `libs/lsxcommon/src/model_runtime.cpp` | new |
| `libs/lsxcommon/include/lsxcommon/model_base.h` | new |
| `libs/lsxcommon/src/model_base.cpp` | new |
| `libs/lsxcommon/CMakeLists.txt` | add both translation units |
| `libs/lsxcommon/include/lsxcommon/inference_gpu.h` | contract fields (item 7-10) |
| `libs/lsxcommon/include/lsxcommon/model.h` | `ResidentWeightBytes`, `Benchmark` |
| `libs/lsxcommon/src/glm_v5_model.cpp` | delegate to runtime/ModelBase; fill contract |
| `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` | delegate; contract fill |
| `libs/lsxcommon/src/qwen_base_model.cpp` | delegate; contract fill (gen_tokens/output_text/final_logits) |
| `libs/lsxcommon/src/deepseek_inference_pipeline.cu` | final_logits readback, ttft event, DecodeGeneratedText |
| `libs/lsxcommon/src/prompt_framing.cpp/.h` | framing descriptor |
| `apps/lsxforge/src/main.cpp` | `RunCliGeneration` |
| `apps/lsxfabric/src/main.cpp` | `RunCliSelfCheck`/helpers; delete `EncodeSelfCheckPrompt` |
| `apps/lsxcli/src/main.cpp` | `RunCliInspect` |
| `apps/lsxhome/src/main_win32.cpp` | real `--run` model wiring (Phase A tail) |
| `tests/` | contract + host-only tests, `test_generation_cli` extension |

## Out of Scope

- Multi-sequence batched prefill (deferred; `ModelRequest` stays batch=1).
- Stochastic sampling (TopK/TopP) — interface seam only.
- Changing the `GlmMoEArena`/Qwen arena allocation model.
- `final_loss` semantics (retained, unused).
