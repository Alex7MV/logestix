# Granite-Switch 4.1 30B ingestion design

## Context

`granite-switch-4.1-30b` (IBM, `/mnt/fast_storage/huggingface/granite-switch-4.1-30b`)
is a **composed** checkpoint: the Granite 4.1 30B dense decoder plus two
embedded act LoRA (aLoRA) adapters, control-token routing, and (in the source
package only) a separate 470M speech encoder.

The source is `GraniteSwitchForCausalLM` (`model_type="granite_switch"`), not a
plain `GraniteForCausalLM`. The reference implementation lives in the
`granite-switch` Python package; HuggingFace `transformers` does not ship this
model class. The checkpoint itself carries no modeling code, so the converter
must own the full semantic mapping.

### Verified geometry and semantics

| field | value |
|---|---|
| decoder layers | **64** (config `num_hidden_layers=66` = 64 decoder + 2 switch cache slots) |
| hidden_size | 4096 |
| num_attention_heads / num_key_value_heads | 32 / 8 |
| head_dim | 128 (`projection_head_dim`) |
| intermediate_size / shared_intermediate_size | 32768 / 32768 |
| vocab_size | **100355** (100352 base + `<|guardian-core|>` + `<|decision|>` + `<|audio|>`) |
| rms_norm_eps | 1e-5 |
| rope | default/rotate-half, full head_dim, theta = 5e7 |
| attention scale | `attention_multiplier = 0.0078125` (1/128) |
| embedding_multiplier | **12.0** |
| residual_multiplier | **0.175** |
| logits_scaling | **16.0** |
| tie_word_embeddings | **true** (no `lm_head` tensor) |
| bos / eos / pad | 100257 / 100257 / 100256 |
| adapters | `guardian-core` rank 16; `decision` rank 64 |
| LoRA `B` | **pre-scaled by `alpha/rank`** at composition time; no runtime alpha |
| LoRA `A` | zero-padded to `max_lora_rank=64`; rows beyond the adapter rank are zero |
| control tokens | 100352 (`guardian-core`), 100353 (`decision`) |
| substitute ids | 27 and 14711 (token exchange before embedding) |
| ASR | 470M encoder is a separate component; **not present in this checkpoint** |

Source file names (all 1284 tensors):

```
model.embed_tokens.weight
model.norm.weight
model.switch.codebook                 [2048, 64] F32
model.switch.control_to_substitute_lut [100355] I64
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.self_attn.qkv_proj.base_layer.weight
model.layers.N.self_attn.qkv_proj.lora_{A,B}_slices.{0,1,2}
model.layers.N.self_attn.o_proj.base_layer.weight
model.layers.N.self_attn.o_proj.lora_{A,B}
model.layers.N.shared_mlp.input_linear.base_layer.weight
model.layers.N.shared_mlp.input_linear.lora_{A,B}_slices.{0,1}
model.layers.N.shared_mlp.output_linear.base_layer.weight
model.layers.N.shared_mlp.output_linear.lora_{A,B}
```

`N` is 0..63. `qkv_proj` is fused `[Q | K | V]`; `input_linear` is fused
`[gate | up]` with `silu(first) * second` (Granite `GraniteMoeHybridMLP`).

### Switch routing — equivalence proven before implementation

The reference switch is a coded-memory attention over control tokens
(counting head + Kerdock codebook). `granite_switch.hf.switch.multi` was run
against 20 random sequences containing control tokens at arbitrary positions.
Its `adapter_indices` are exactly:

```
adapter_id[i] = id of the most recent control token at or before i, else 0
```

and its `modified_input_ids` are exactly the token-exchange LUT. The engine
therefore computes both with a host prefix scan; the codebook and counting
head are **not** needed at inference. This equivalence is a test invariant, not
an assumption.

### Numeric flow

```
x = E[exchange(ids)] * 12
per layer:
  n  = RMSNorm(x)
  qkv = W_qkv(n) + LoRA_a(n)              # 3 slices; adapter-specific rank
  q,k = RoPE(q,k)                          # full 128 dims, theta 5e7
  a  = causal GQA(q,k,v, scale=1/128)
  o  = W_o(a) + LoRA_a(a)
  x += 0.175 * o
  n  = RMSNorm(x)
  gu = W_gu(n) + LoRA_a(n)                 # [gate | up]
  d  = W_down(silu(gu[:I]) * gu[I:]) + LoRA_a(...)
  x += 0.175 * d
logits = (E(x)) / 16                       # tied embedding matrix
```

## Architecture decision

**Variant A (chosen): sibling files `granite_dense_switch_*`.** The 4.2
implementation stays untouched; the switch variant reuses the neutral NVFP4
layer, the shared kernel primitives, and the proven Granite RoPE/attention
wrappers. A new frozen passport is required because vocab, multipliers, tied
embeddings and the adapter surface are different.

Rejected: generalizing `granite_dense_*` (adds conditional branches to the hot
path of a green model), base-only ingestion (loses the model's purpose).

## Target layout

Format tag `logestix-granite-switch`; `kType = "granite-switch-4.1-30b"`;
`model.index.json` model name `Granite-Switch-4.1-30B`.

```
<out>/logestix/model.index.json
<out>/logestix/pinned/global.arrow               final norm + switch codebook (raw)
<out>/logestix/pinned/global_nvfp4.arrow         tied embed (one record; head aliases it)
<out>/logestix/pinned/layer_N.arrow              norms
<out>/logestix/pinned/layer_N_nvfp4.arrow        qkv / o / gate_up / down base weights
<out>/logestix/pinned/layer_N_adapters.arrow     LoRA A/B per adapter, BF16 raw, rank-trimmed
<out>/logestix/pinned/tokenizer_{vocab,merges}.arrow
```

Index additions (machine-readable, also the loader's source of truth):

```json
{
  "format": "logestix-granite-switch",
  "model": "Granite-Switch-4.1-30B",
  "num_layers": 64,
  "switch_cache_layers": 2,
  "tie_word_embeddings": true,
  "embedding_multiplier": 12.0,
  "residual_multiplier": 0.175,
  "logits_scaling": 16.0,
  "adapter_names": ["guardian-core", "decision"],
  "adapter_ranks": [16, 64],
  "adapter_token_ids": [100352, 100353],
  "adapter_substitute_token_ids": [27, 14711],
  "lora_b_prescaled": true
}
```

The converter reads the source `control_to_substitute_lut` and verifies it
equals the config-derived mapping (control id -> substitute, otherwise -1)
before dropping it. The codebook is stored as a raw F32 record (fidelity and
future reference) but is not uploaded to the device.

## Converter

- Parse the flat `config.json`; read `projection_head_dim` (fallback
  `hidden_size / num_attention_heads`).
- Validate `config.num_hidden_layers - switch_cache_layers == tensor_layers`.
- Classify every tensor; unknown names fail. The `base_layer.` and `shared_mlp`
  names map onto the canonical `qkv_proj` / `o_proj` / `gate_up_proj` /
  `down_proj` records (the source already ships fused qkv and gate/up).
- Quantize base projections to NVFP4. `embed` is quantized once; `head` aliases
  the same device record with `global_scale / 16`.
- Trim each LoRA pair to the adapter's rank from `adapter_ranks`; the trimmed
  bytes equal the source product because `A` rows beyond rank are exactly zero.
  Keep `B` as stored (already `alpha/rank`-scaled).
- Write adapter records only where the adapter targets the projection
  (`guardian-core` has no MLP records).
- Norms, final norm and codebook stay raw.
- Build the tokenizer with `bos=eos=100257`, `pad=100256`.

## Loader

`GraniteSwitchPinnedWeights` holds the base `LsxNvfp4Weight`s plus, per
projection site, `{A, B, rank}` per adapter (BF16 -> FP16 upload). `Free()`
releases both planes. `head` is a non-owning view of `embed` with
`global_scale = embed.global_scale / 16`. Control-token metadata comes from
`model.index.json`, never from a hardcoded table.

## Inference

- Host pre-processing per request: token exchange + adapter prefix scan over
  the **original** ids; remember `last_adapter` in the session for continuation
  calls (a decoded token inherits the adapter active at the control token).
- Embedding gather, then multiply by 12. Residual adds become
  `x += residual_multiplier * sublayer`.
- LoRA deltas: two-phase shrink/expand per projection slice. Shrink writes
  `y[token, r] = A x` only for tokens with `adapter>0`; expand adds
  `B y` into the projection output. Ranks 16 and 64 are supported; zero-rank
  sites are skipped. Decode remains one stream; no per-token sync.
- Base GEMV/GEMM stays on the neutral `lsx_nvfp4` layer. LoRA math is FP16;
  the reference is BF16, so adapter parity is tolerance-class, not bit-exact.
- Logits scaling is folded into the tied head's `global_scale` (power-of-two,
  exact in FP32 accumulation).
- KV cache: existing Granite layout/allocator, 64 layers.

## Chat, adapters, tools (stage 4b)

Native Granite-4.1 framing, byte-for-byte from `chat_template.jinja`:

```
<|start_of_role|>role<|end_of_role|>content<|end_of_text|>\n
```

Tools use the template's `<tools>` block with the OpenAI tool objects and the
instructional prefix/suffix; assistant calls are
`<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`. Parsing accepts
JSON inside `<tool_call>` and falls back to `tools::ParseToolCallsText` /
`ParseToolCallsBareJson`.

Adapter activation precedence (documented behaviour):

1. `ChatConversation.adapter_name` (populated from
   `chat_template_kwargs.adapter_name` in taskmosaic);
2. an explicit control token already present in the message text;
3. automatic detection of the invocation text in the last user message
   (`<guardian>` / `### DECISION`).

The insertion point reproduces the template: guardian drops the leading `<`
and keeps `guardian>`; decision drops `###` and keeps ` DECISION`. With no
invocation text, the control token goes immediately before the assistant role
marker.

`<|audio|>`/ASR is out of scope: the encoder is a separate artifact not
present in this checkpoint.

## Precision strategy

Start with the NVFP4 base + FP16 LoRA. Run every gate and an adapter-parity
probe against the CPU-BF16 source oracle. Only if a gate is red on a
borderline answer, add a `.res` residual plane to the measured subset (the
gemma/flint procedure) and re-measure VRAM; if no fitting placement clears the
gate, record the debt — never rewrite the golden.

## Oracle

`tools/granite_switch_oracle.py` loads
`GraniteSwitchForCausalLM` with `granite-switch==0.1.1` and
`transformers==5.17.0`, CPU BF16 (`device_map="cpu"`), and exposes
`adapter_name`. The capture scripts gain a `loader: "granite_switch"` branch.
Python 3.14 needs `pip install --ignore-requires-python` (the package metadata
caps at `<3.14`; the code runs). CPU BF16 is required: `device_map="auto"`
mixing GPU and CPU produced NaNs in the LoRA path, while full-CPU BF16 is
finite and deterministic.

The gate captures use the checkpoint's **native chat template** (verified on
the oracle: FSI 3/3 expected answers, chat mode, base adapter). Raw-body
framing remains only for a checkpoint without a template.

## Tests (TDD, one commit per stage pair)

| stage | test | assertions |
|---|---|---|
| convert | `test_granite_switch_convert` | synthetic checkpoint; index fields; base NVFP4 records; adapter rows trimmed/prescaled; LUT verified; no `lm_head` needed; TIMEOUT 120 |
| load | `test_granite_switch_loader` | base + adapter upload, head alias scale, rank-0 sites null, `Free()` |
| kernels | `test_granite_switch_kernels` | prefix scan vs reference switch table; token exchange; LoRA shrink/expand vs host reference; residual scale; rope/attention parity with granite |
| model | `test_granite_switch_model` | load/infer/stream/abort/continuation with adapter state; geometry rejection |
| selfcheck | `test_granite_switch_selfcheck` | Paris, >=3 cycles, bit-exact logits, SKIP 4 |
| parity | `test_granite_switch_fsi/code/det_selfcheck` | committed goldens from the CPU oracle, SKIP 4 |
| adapters | `test_granite_switch_adapter_parity` | base/guardian/decision first-token top-1 + routing, goldens from oracle |
| tokens | `test_granite_switch_tokenizer` | encode parity (27 prompts) and chat-template ids |
| tools | `test_granite_switch_tool_framing` | render/parse round-trip on a synthetic tokenizer |
| perf | `test_perf_baseline_granite_switch_4_1_30b` | harness-measured row, SKIP 4 |

Guard registries updated in the same commits: naming (`switch` variant,
`granite_switch_tool_format`), `kPassports`, `kLifecycle`, `Suites()`,
`kGoldenDebt` only if a capture is impossible, ctest labels, CMake targets.

## Risks and mitigations

- **4-bit body vs small LoRA deltas** — measure before adding residual planes.
- **32 GiB card** — 131k context KV alone is ~34 GiB; document the practical
  context ceiling and do not claim the full window.
- **Decoder-layer count mismatch (66 vs 64)** — the converter's arithmetic
  check (`cfg - 2 == tensors`) is the guard; the index stores 64.
- **Adapter-specific chat framing** — capture expected token ids from the
  reference tokenizer/template and pin them; do not guess. The base model's
  chat answer to a bare completion prompt can differ from its raw-completion
  answer; the gates use the native chat template, which the oracle cleared 3/3.
- **Perf** — correctness first. Base mode should land near Granite 4.2; adapter
  mode pays low-rank launches. The perf row is seeded through the harness, not
  hand-edited.

## Out of scope

ASR/`<|audio|>`, long-context (>32k) tuning, CUDA-graph decode, decision
probability/logprob API in taskmosaic (adapter generation still works; the
softmax-over-labels readout remains a client-side/readout feature).
