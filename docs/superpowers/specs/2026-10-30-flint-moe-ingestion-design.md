# Recursive Flint (39.8B-A2.5B) — ingestion design

Branch: `feat/flint-moe-ingest`
Date: 2026-10-30
Protocol: Model Ingestion Protocol (`ingesting-a-model`), Stage 0 + stages 1–8 + 4b.
Reference checkpoint: `/mnt/fast_storage/huggingface/flint` (BF16, 15 shards,
79.6 GB, 36 734 tensors). Source of truth for semantics: the checkpoint's own
`config.json`, `chat_template.jinja`, `tokenizer.json` and the vendor SGLang
plugin `sglang/flint_sglang-0.1.4-py3-none-any.whl` (`flint_sglang/model.py`,
`configuration_flint.py`, `tool_grammar.py`, `reasoning_parser.py`) plus
`sglang/fa4-learned-sink-fix.patch` (the vendor's own learned-sink fix).

## Stage 0 — triage

### 0.1 Classification: **new family `flint` (type `moe`)**

No `kType` shares this architecture. The combination — a dense 2-layer prefix,
hybrid SWA/full attention where the *full* layers carry **no positional
encoding**, a per-head learned sink logit, an ungated shared expert and a
256-expert sigmoid router — exists in no ingested family. Closest precedents are
used as structural templates, not as code to share: `hy` (48L/2048H, all
experts resident, sigmoid router + shared expert) and `mimo` (learned sink
column, sliding window).

Registry rows to extend (same commit as the files):

| registry | row |
|---|---|
| `test_code_naming_guard.cpp` | `SchemeRegex()` family token `flint` |
| `test_model_independence_guard.cpp` | `kPassports` (`flint` → `flint_moe_constants.h`, ns `lsxcommon::flint::frozen`) |
| `test_model_independence_guard.cpp` | `consumers` (`flint_moe_model.cpp`) |
| `test_model_independence_guard.cpp` | `kLifecycle` (model → `flint_moe_model.cpp`, `flint_moe_inference_pipeline.cu`) |
| `test_model_suite_guard.cpp` | `Suites()` row (convert→load→model→kernels→selfcheck→fsi→code→det→perf) + tool-verdict row |
| `test_ctest_labels_guard.cpp` | — (label every new test) |
| `test_expert_store_guard.cpp` | **no row** — all experts are device-resident, there is no host expert store |

### 0.2 Geometry (frozen, from `config.json` + tensor inventory)

| field | value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 49 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 4 |
| `head_dim` | 128 |
| `vocab_size` | 150144 |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 10000.0 (SWA layers only) |
| `max_position_embeddings` | 262144 (capacity only) |
| `sliding_window` | 513 = current token + 512 previous |
| `layer_types` | 14 `full_attention` at 0,1,5,9,…,45,48; 35 `sliding_attention` elsewhere |
| `no_rope_layer_indices` | **identical to the full-attention list** (verified) |
| `mlp_only_layers` | 0, 1 (dense) |
| `intermediate_size` | 6144 (dense MLP) |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 1024 |
| `num_experts` / `num_experts_per_tok` | 256 / 6 |
| `router_score_function` / `norm_topk_prob` | `sigmoid` / true |
| `softmax_type` | `learnable` (per-head sink) |
| `attention_bias` | false |
| `tie_word_embeddings` | false |
| `torch_dtype` | bfloat16 |

Weight shapes (authoritative, per layer): `q_proj [4096,2048]`,
`k_proj [512,2048]`, `v_proj [512,2048]`, `o_proj [2048,4096]`,
`q_norm/k_norm [128]`, `softmax_offset [32]`; dense `gate/up [6144,2048]`,
`down [2048,6144]`; router `mlp.gate.weight [256,2048]`; routed expert
`gate/up [512,2048]`, `down [2048,512]`; shared expert `gate/up [1024,2048]`,
`down [2048,1024]`; `embed_tokens`/`lm_head [150144,2048]` (not tied);
`model.norm.weight [2048]`. No `*.bias` tensors exist.

### 0.3 Tokenizer

- Byte-level BPE (`tokenizer.json`: `model.type=BPE`, vocab 150 000, 149 744
  merges, no normalizer, ByteLevel decoder), plus 144 added tokens at
  150000–150143: 132 `special:true` (`<|begin_of_text|>`, `<|end_of_text|>`,
  `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`,
  `<|developer|>`, `<|reserved_*|>`) and 12 `special:false`
  (`<think>`, `</think>`, `<tools>`, `</tools>`, `<tool_call>`, `</tool_call>`,
  `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`, `<tool_response>`,
  `</tool_response>`).
- `bos=150000`, `pad=150001`, `eos={150001,150003,150005,150018}`
  (`<|end_of_text|>`, `<|user|>`, `<|observation|>`, `<|developer|>`);
  `add_bos_token=false`, `add_eos_token=false`.
- **Pre-tokenizer is a 3-split Sequence**: `\p{N}{1,3}` Isolated → CJK
  (`[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+`) Isolated → a 6-alternative
  o200k **case-run** regex whose punctuation alternative is
  ` ?[\p{P}\p{S}]+[\r\n]*` (not o200k's `[^\s\p{L}\p{N}]+[\r\n/]*`).
  The repo has neither variant: `WordStream` is digit+CJK but letter-run,
  `Llama3WordStream` is case-run but digit/CJK-free. A third stream is required.

### 0.4 Ballast

None. The 15 shards carry exactly the language model (36 734 tensors accounted
for: 49×7 attention + 49×2 norms + 3 globals + 6 dense-MLP + 47×(768 experts +
router + 3 shared)). No vision tower, no MTP, no audio.

### 0.5 Feasibility on this machine (RTX 5090, 32 GB; 750 GB RAM)

| component | NVFP4 bytes |
|---|---|
| 47×256 routed experts (37.85B params) | 21.3 GB |
| attention + dense MLP + shared expert (~1.3B) | 0.75 GB |
| embed_tokens + lm_head (FP16, 2×307M params) | 1.23 GB |
| norms, router (FP16), sinks | ~0.05 GB |
| **resident total** | **~23.3 GB** |

Plus KV at the canonical 2048 context (14 full layers × 2048 × 512 × 2 × 2 B
≈ 60 MB) — the model fits entirely on the card, so routed experts ship as
device-resident per-layer slabs (hy precedent) and there is **no** host expert
store and no LRU. Converted tree ≈ 24 GB on a 6.9 TB volume.

### 0.6 Semantics to verify before trusting any token

- attention scale `1/sqrt(128)`, applied at QK only; no value scale;
- learned sink is a **per-head logit added to the softmax denominator with no
  value contribution** (FA4 patch: `row_sum += exp2(sink*log2e - max)`, the
  numerator is untouched; `row_max` also folds the sink in);
- **no positional encoding on the 14 full-attention layers** (RoPE is applied
  only on SWA layers), theta 10000, rotate-half pairing over all 128 dims,
  no scaling factor;
- q_norm / k_norm are per-head plain RMSNorm (weight form), eps 1e-6;
- routed weights: `sigmoid(Wx)` in F32, top-6, renormalize the selected scores
  (no bias tensor exists, so selection == plain sigmoid top-6), no routed
  scale factor;
- shared expert is **ungated** (plain SwiGLU) and is added to the routed sum;
- residual stream FP16, router/combine F32, logits F32 over an FP16 table
  (the vendor serves with `--enable-fp32-lm-head`);
- 4 stop ids; no final logit soft cap and no model-level temperature.

## Stage 1 — converter (`lsxforge --flint`, format `logestix-flint-nvfp4`)

New `flint_moe_convert.{h,cpp}` following `hy_moe_convert.cpp`:

1. `ParseFlintConfig` reads `config.json`, rejects any drift from the frozen
   passport (layers, widths, layer_types, no_rope list, mlp_only list, expert
   count, top-k, vocab, eps, theta, window) unless
   `SyntheticGeometryAllowed()`.
2. Dense graph → `pinned/*.arrow`: q/k/v/o, dense gate/up/down (NVFP4),
   shared expert per layer (NVFP4), router (FP16), norms/qk-norms/sinks (FP16),
   embed/lm_head (FP16), final norm.
3. Routed experts → `experts/layer_XX.arrow` in the repo's per-expert record
   layout `[gate | down | up]`, NVFP4, 256 experts per layer.
4. `model.index.json`: format tag, full geometry, `moe_layers` (2..48),
   `full_attention_layers`, `no_rope_layers`, eos list, `vram` block.
5. Test (`convert`, TIMEOUT 120, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`):
   synthetic 49-layer/4-expert fixture covering every path + a rejection test
   for a drifted config.

## Stage 2 — quantization

NVFP4 for every projection (attention, dense MLP, routed + shared experts).
Never quantized: norms, q/k head norms, router, sinks, embed/lm_head (FP16).
The residual-plane procedure from the skill is kept in reserve: only if a parity
gate goes red on a *measured* near-tie, add `<name>.res` planes (never rewrite a
golden).

## Stage 3 — registration + loader

- `FlintMoeModel` (`kType = "flint"`) registered in `model_registry.cpp`.
- `flint_moe_load.cpp`: pinned dense graph, 47 layers × 256 experts uploaded
  into per-layer device slabs directly from mmap views (no host copies, no
  ExpertStore), `resident_bytes_` counted at the allocation sites, `Free()`
  releases every slab. Frozen-geometry cross-check against `model.index.json`
  fails `Load`.
- KV: pos-major `KvCacheGeometry`, one cache slice per layer (49 total: 14
  full-attention slices read without a window, 35 SWA slices read through the
  513-row window). Correctness first: the SWA slices are full-length like the
  rest (100 KB per token overall, 205 MB at 2048, 3.3 GB at 32 768); a 513-row
  ring is a later memory optimisation, not part of ingestion.

## Stage 4 — tokenizer bridge

`converter.h`: `TokenizerSpecialPolicy::pretokenizer = 2` → `tokenizer_kind = 4`.
`pretokenizer.{h,cpp}`: `FlintWordStream` = number split (≤3 codepoints) → CJK
run → the 6-alternative case-run matcher with `\p{P}\p{S}` /
`[\r\n]*`. `tokenizer.cpp`: kind 4 dispatches to the kind-3 literal+BPE-merges
encoder with the new stream (template-parameterized gap encoder).
Parity: `tools/flint_tokenizer_capture.py` captures ids from the checkpoint's
own `tokenizer.json` for a corpus adversarial for digits/CJK/camelCase/
multi-space; the test SKIPs (4) when the vault entry is absent.

## Stage 4b — tool-calling bridge

`flint_moe_tool_format.{h,cpp}`, verified against `chat_template.jinja`:

- render: `<|system|>{s}<|developer|>{d}<|user|>{u}<|assistant|>` with markers
  inline (no separators), text sanitized exactly like `flint_text`
  (drop `\x00`, strip `<think>`/`</think>`, HTML-escape role markers);
- tool preamble as its own `<|system|>` turn when tools are present;
- assistant history: optional `<think>…</think>`, content, then
  `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value></tool_call>`
  (non-string values JSON-encoded, nested objects included);
- tool results: `<|observation|><tool_response>{…}</tool_response>`;
- parse: scan for `<tool_call>` … `</tool_call>`, name = text before the first
  `<arg_key>`, pairs of `arg_key`/`arg_value` (value is raw text, JSON-typed
  when it parses), parallel calls in one turn;
- generation prompt ends at `<|assistant|>`; all four config eos ids terminate.

Test (`fast`): render structure, parse (incl. parallel + nested JSON), and a
tokenizer round-trip on a synthetic fixture; never round-trip the full framed
prompt (its instructions embed an example call).

## Stage 5–6 — kernels

`flint_moe_inference_attn.{h,cu}` (family-owned, mimo-shaped): fused-QKV token
prefill/decode over a KV cache, GQA 8:1, per-head qk RMSNorm, optional
rotate-half RoPE (skipped on full layers), sink column in the softmax
denominator, `j0 = max(0, pos-512)` on SWA layers, FP16 out.
`flint_moe_kernels.{h,cu}`: RMSNorm, sigmoid top-6 router (F32), routed
accumulate, ungated shared combine (F32), SwiGLU. Both are tested RED-first
against a host reference in `flint_moe_device_math.h` (sink softmax, router
selection/renormalization, window bounds).

## Stage 8 — gates, oracle, perf

- SelfCheck: "The capital of France is" → `Paris`, ≥3 cycles bit-exact.
- **Oracle** (approved): `tools/flint_oracle.py` builds the reference from the
  installed HF `Qwen3Moe*` modules plus the Flint deltas, loads the BF16
  checkpoint on CPU (750 GB RAM), and is checked layer-by-layer against the
  vendor plugin's semantics. `tools/{fsi,code,det}_capture.py` gain a
  `loader: "flint"` override; goldens are captured from the source and
  committed. If a gate is red on a measured near-tie, apply the residual-plane
  procedure and record the outcome — never edit a golden.
- Perf: entry in `tests/perf/perf_baselines.json`, `TEST_CASE` in
  `test_perf_baseline.cpp`, `add_test(test_perf_baseline_flint)` (label `perf`,
  SKIP 4), measured through the harness at the 2048-token protocol.
- Docs: `AGENTS.md` Structure, `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md`.

## Open risks

1. **Sink + sliding mask interaction** — the sink is always in the denominator,
   even when the window is short; a host reference pins it.
2. **no-RoPE is per-layer, not per-type-by-accident** — the frozen table is
   `static_assert`ed against `config.json` by the converter.
3. **4 stop ids** — the pipeline takes `eos_ids` as a vector (gemma precedent).
4. **NVFP4 near-ties** on a coding model with long reasoning — measured, not
   assumed; residual planes are the documented fix.
5. **CPU oracle wall time** — 2.2B active per token on 160 cores; budget
   hours for the three capture sets, run last under the host resource policy.
