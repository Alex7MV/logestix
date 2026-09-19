# Qwen3.6-35B-A3B Ingestion Design

Date: 2026-09-07
Status: Approved
Goal: Adapt Qwen3.6-35B-A3B per the Model Ingestion Protocol (8 stages), autonomous
from the other models, reusing the qwen_moe family infrastructure and without new
CUDA kernels. Implemented via TDD on the `feat/qwen36-35b-a3b` branch.

## Reconnaissance (2026-09-07)

Source: `/mnt/llm_models/models/Qwen3.6-35B-A3B` — 67 GB, BF16, 26 shards
(`model-00001-of-00026.safetensors`), a full tokenizer (vocab.json, merges.txt,
tokenizer.json, tokenizer_config.json).

- `architectures = Qwen3_5MoeForConditionalGeneration`, `model_type = qwen3_5_moe` —
  architecturally identical to Qwen3.5-35B-A3B (already integrated as `qwen3.5-35b-a3b`).
- The text_config geometry **matches byte-for-byte** 3.5: hidden=2048, 40 layers
  (GDN hybrid, full_attention every 4th: i%4==3), heads=16, kv_heads=2, head_dim=256,
  linear 16/32/128/128, conv_kernel=4, vocab=248320, E=256, top-8, moe_i=512,
  shared_i=512, rms_eps=1e-6, partial_rotary=0.25, rope_theta=1e7.
- The tensor grammar of the main layers is identical to 3.5 (fused `mlp.experts.gate_up_proj`
  [256,1024,2048] / `mlp.experts.down_proj` [256,2048,512]).
- **The only structural difference is the MTP block**: in 3.6 the experts are fused
  (`mtp.layers.0.mlp.experts.gate_up_proj` / `down_proj`), in 3.5 they are per-expert
  (768 tensors). The generic Mtp branch of the existing converter covers both variants
  (MTP is stored, not applied — HF parity), no special handling is needed.
- Config delta 3.6: an explicit `text_config.bos_token_id = 248044` (matches
  eos_token_id), `pad_token_id = null`, `partial_rotary_factor = 0.25`,
  `tie_word_embeddings = false`; no `mlp_only_layers` (equivalent to empty).
- The tokenizer_config.json special tokens are identical to 3.5: no bos_token,
  eos_token = `<|im_end|>`, pad_token = `<|endoftext|>`.

## Operator decisions

| Question | Decision |
|---|---|
| Converter | Shared core: the body of `ConvertQwen35MoeModel` → `ConvertQwenMoeModelCommon(spec)`; 3.5 and 3.6 are thin wrappers. The 3.5 output is byte-identical |
| Step 7 (passport.bin) | Skip (like Qwen3.5-35B-A3B); the shared `passport/passport.bin` is taken by the GLM-5.3-Flash↔DeepSeek-V4-Flash bridge. The deviation is recorded here |
| Model class | Its own class `Qwen36MoeModel` + extraction of the Load/Infer/Benchmark lifecycle bodies into family helpers used by both models |
| New CUDA kernels | Not required (the architecture is identical to 3.5); `qwen_moe_inference.cu` is not changed |

## Invariant #1: Model autonomy

- Its own frozen passport: `qwen_moe_36_constants.h`, namespace `lsxcommon::qwen_moe_36::frozen`
  (a leaf header, no cross-includes of passports — independence guard).
- Its own kType `qwen3.6-35b-a3b`, its own `ValidateGeometry`, its own `Infer` with
  `LsxFastMathScope(cfg_.fast_math, ...)` (the independence-guard consumer list).
- The 3.5 files (`qwen_moe_*`) are touched only by the mechanical helper extraction
  (3.5 behaviour is unchanged, the regression shield is the existing
  `test_qwen_moe_{convert,loader,model,kernels,selfcheck}`) and by one-line
  legitimate touches: `model_registry.cpp`, `tests/test_code_naming_guard.cpp`,
  `tests/test_model_independence_guard.cpp`, CMakeLists, `apps/lsxforge/src/main.cpp`.
- The family infrastructure (`qwen_moe_load`, `qwen_moe_inference.cu`,
  the generation driver `RunQwenMoeGeneration`) is reused by reference —
  it is the allowed shared layer of the family (analogous to `qwen_inference` for dense+MoE).

## Frozen geometry (compile-time, static_assert)

Source of truth: `/mnt/llm_models/models/Qwen3.6-35B-A3B/config.json`
(text_config, model_type `qwen3_5_moe`). The values match 3.5, but the passport
is its own — independent (independence guard: a unique namespace per model).

```
hidden = 2048, vocab = 248320
Layers: 40, layer_types[i] = linear_attention (i%4!=3) / full_attention (i%4==3)
heads = 16, kv_heads = 2, head_dim = 256
linear: num_key_heads = 16, num_value_heads = 32, key_head_dim = 128,
        value_head_dim = 128, conv_kernel_dim = 4
MoE: num_experts = 256, num_experts_per_tok = 8, moe_intermediate = 512,
     shared_expert_intermediate = 512
rms_norm_eps = 1e-6, partial_rotary_factor = 0.25, rope_theta = 1e7
bos_token_id = 248044 (from config 3.6; eos_token_id = 248044, no pad)
MTP: store-not-apply (the MTP fused experts are quantized whole-tensor)
```

## Components (naming: family=qwen, type=moe, variant=36)

| File | Role |
|---|---|
| `libs/lsxcommon/src/qwen_moe_36_constants.h` | 3.6 frozen passport + static_assert |
| `libs/lsxcommon/src/qwen_moe_36_convert.cpp` + `include/lsxcommon/qwen_moe_36_convert.h` | `ConvertQwen36MoeModel` — a frozen gate against `qwen_moe_36::frozen` + a call to the shared core (format `logestix-qwen36moe`, model `Qwen3.6-35B-A3B`, record_bos) |
| `libs/lsxcommon/src/qwen_moe_36_model.cpp` + `.h` | `Qwen36MoeModel` (ModelBase), kType=`qwen3.6-35b-a3b` |

Modifications of existing files (legitimate touches):

| File | Change |
|---|---|
| `libs/lsxcommon/src/qwen_moe_convert.cpp` + `include/lsxcommon/qwen_moe_convert.h` | Body → `ConvertQwenMoeModelCommon(input, output, const QwenMoeConvertSpec&)`; `ParseQwenMoeConfig` parses `bos_token_id` (default -1); the index writes `bos_token_id` only when `spec.record_bos && top.bos_token_id >= 0` (the 3.5 output is byte-identical) |
| `libs/lsxcommon/src/qwen_moe_load.{h,cpp}` | Extract `LoadQwenMoeStandardGraph(...)` — the body of `Qwen35MoeModel::Load` (dense load, expert store, governor, cache sizing, prestaging), parameterized by a geometry gate |
| `libs/lsxcommon/src/qwen_moe_inference.cu` + `include/lsxcommon/qwen_moe_inference.h` | Extract the infer/benchmark drivers over `RunQwenMoeGeneration` (session policy, FillInferenceResult/FillBenchmarkResult) into family helpers |
| `libs/lsxcommon/src/qwen_moe_model.{h,cpp}` | Delegate to the extracted helpers (behaviour identical) |
| `libs/lsxcommon/src/model_registry.cpp` | `LsxModelFactory::Register<Qwen36MoeModel>()` |
| `apps/lsxforge/src/main.cpp` | Flag `--qwen36moe` → `ConvertQwen36MoeModel` |
| `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt` | New sources and test targets |
| `tests/test_code_naming_guard.cpp` | variant regex `(?:27b|flash)` → `(?:27b|flash|36)` |
| `tests/test_model_independence_guard.cpp` | The `kPassports` row (`qwen_moe_36_constants.h`, `lsxcommon::qwen_moe_36::frozen`), the `owner` map, `qwen_files`, the consumer `qwen_moe_36_model.cpp` |

## Converter

- `QwenMoeConvertSpec` (in `qwen_moe_convert.h`):
  `format_tag`, `model_name`, `bool (*check_frozen)(const QwenMoeTopology&, std::string&)`,
  `record_bos`.
- The classifier, fused-expert split (gate=rows[0,I), up=rows[I,2I)), the MTP branch,
  the tokenizer (`BuildAndVerifyTokenizer`, bos_required=false / eos_required=true),
  the unified `logestix/` layout — unchanged, shared by 3.5/3.6.
- 3.6 specifics: its own frozen gate; `"bos_token_id": 248044` is added to the index
  (3.6 only, controlled by `record_bos`).
- 3.6 MTP fused experts: the generic Mtp branch quantizes them whole-tensor
  ([256,1024,2048] / [256,2048,512]) — stored-not-applied, HF parity.
- Norms/biases/embed/lm_head/router/conv1d/A_log/dt_bias — without quantization
  (the protocol's precision invariant); linear matrices — NVFP4.

## Model class

- `Qwen36MoeModel final : ModelBase`:
  - `kType = "qwen3.6-35b-a3b"`;
  - ctor: `qwen::SetQwenTcFastMath(cfg_.fast_math)` (parity with 3.5);
  - `Load()`: `LoadQwenMoeStandardGraph(*this, geometry-gate-3.6, members)`;
  - `Infer()`: `LsxFastMathScope(cfg_.fast_math, cfg_.fast_math)` → family helper
    (session policy, `RunQwenMoeGeneration`, FillInferenceResult);
  - `SelfCheck()`: `runtime::RunParisSelfCheck` (prompt "The capital of France is"
    → "Paris", top-1);
  - `Benchmark()`: family helper, fresh session;
  - `ValidateGeometry()`: against `qwen_moe_36::frozen` (+ `SyntheticGeometryAllowed()`
    escape hatch for synthetic fixtures).
- The incremental session policy, the single end-to-end `cudaStream_t`, expert-cache
  sizing/prestaging — inherited from the extracted helpers (identical to 3.5).

## Metadata

The index and artifacts — the unified `<output_root>/logestix/` layout via
`lsxcommon::lsx_model_metadata(rel)`; no hardcoded paths in the converter.
The format tag in `model.index.json`: `logestix-qwen36moe`, `"model":
"Qwen3.6-35B-A3B"`, `bos_token_id: 248044`.

## Tests (TDD — tests first, then implementation)

| Executable | Content |
|---|---|
| `tests/test_qwen_moe_36_convert.cpp` | 3.6-grammar classifier (incl. MTP fused names → Mtp), frozen gate (accept the 3.6 config / reject a foreign geometry), synthetic roundtrip on the tiny fixture (`qwen_moe_test_fixture.h` pattern), index contents (format/model/bos), the record_bos condition |
| `tests/test_qwen_moe_36_model.cpp` | The registry contains `qwen3.6-35b-a3b`; ValidateGeometry accept/reject; synthetic lifecycle (SKIP pattern without CUDA/weights) |
| `tests/test_qwen_moe_36_selfcheck.cpp` | Real Paris selfcheck, 3 bit-exact cycles (`LSX_TEST_QWEN36_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4 without a converted model) — protocol step 8 |
| 3.5 regression | `test_qwen_moe_{convert,loader,model,kernels,selfcheck}` stay green; the 3.5 converter output is byte-identical (the guard — the existing 119 convert-test assertions) |
| Guards | `test_code_naming_guard.cpp` (variant `36`), `test_model_independence_guard.cpp` (passport/consumer) — extended in the same commit as the model files |

TDD order: convert tests (red) → shared core + 3.6 wrapper (green) → model tests
(red) → class + helpers (green) → guards/CMake (green) → real model conversion
(`lsxforge --qwen36moe`, 67 GB) → selfcheck test (red without a model → green).

## Risks

- The helper extraction touches `qwen_moe_model.cpp` — purely mechanical,
  behaviour pinned by the existing 3.5 test set.
- `bos_token_id == eos_token_id == 248044` in the 3.6 config: the generation EOS comes from
  the tokenizer db (`<|im_end|>`), as in 3.5 — no collision.
- The Windows test restriction (`ctest -R "qwen|gemma"`) already covers the new tests
  via the `qwen` prefix.
- Step 7 is skipped — a deliberate deviation, recorded in this spec; when a
  network bridge for the Qwen family appears, the passport is added as a separate
  file (not by overwriting the shared bridge).
