# Dory1.0-4B ingestion — plan (2026-11-10)

Model: `NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (`model_type: dory`, class
`DoryForCausalLM`). Vault source: `/mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`.

## Stage 0 triage (gate: answered before stage 1)

### 0.1 Classification — NEW FAMILY (`dory`)

Not a sibling: `dory` is a **normalized-GPT (nGPT) looped/latent-recurrent**
architecture with a single mixer per physical layer. Nothing in the repo
(`glm/qwen/gemma/kimi/deepseek/nemotron/…`) shares it: there is no RMSNorm
anywhere, the residual stream is L2-normalized after every mixer by `justnorm`,
and the middle block of layers is applied `n_recurrent_loops` times with the
input block's output re-injected. Own passport, own kernels, own model class.

Reference implementation (the only one; transformers 5.15.1 has no
`DoryForCausalLM`): `vllm-project/vllm#60149`,
`vllm/model_executor/models/dory.py`. Transcribed into `tools/dory_oracle.py`
(torch, BF16 weights + FP32 norms/softmax) and verified against the real
checkpoint: `"The capital of France is"` → `" Paris."` (greedy, 1st token 6993).
`transformers` cannot even build the tokenizer from the directory (config
validation rejects `model_type: dory`), so the oracle builds
`PreTrainedTokenizerFast` from `tokenizer.json` directly.

### 0.2 Geometry (config.json, frozen)

| field | value |
|---|---|
| hidden_size | 2560 |
| num_hidden_layers ("layers" = single mixers) | 72 |
| partition | input 16 + recurrent 40 + output 16 |
| n_recurrent_loops | 2 (`recurrent_kv_cache_mode: per_loop`) |
| mixers | attention at even indices (36), mlp at odd (36) |
| full vs sliding attention | `full_attention` iff `layer % 8 == 6` else `sliding_attention` (window 1024) |
| num_attention_heads / kv heads / head_dim | 16 / 4 / 256 (GQA 4:1) |
| intermediate_size | 7168 |
| vocab_size | 131072 (untied) |
| rope | per layer type: sliding θ=1e4, full θ=1e7, `partial_rotary_factor` 1.0, NeoX pairs |
| initializer_range (nGPT scale reference) | 0.0177 |
| ngpt_alpha_init | 0.05 |
| attention_output_gate_logit_scale_init | null → `sqrt(hidden_size)` |
| max_position_embeddings | 262144 |
| biases / tie / norms | none |

Effective depth = 16 + 40·2 + 16 = 112 mixers per token. Parameter count
(~3.97 B) matches the 7.95 GB BF16 shard.

### 0.3 Semantics that decide correctness (each is a "answers but not the
model's answers" trap)

1. `justnorm(x) = x · rsqrt(Σx² + 1e-6)` in FP32, cast back — **not** an
   RMSNorm: no mean, no learned weight, no sqrt(d) factor.
2. Residual: `slerp(justnorm(a), justnorm(b), lr)` with `lr = |alpha ·
   alpha_scale|`, `alpha_scale = 0.05/0.0177`. **abs()**, and a second
   normalization after the interpolation.
3. Attention: RoPE is applied to q,k **before** the nGPT qk-norm; q is scaled by
   `(sqk · sqrt(head_dim)/std)²` elementwise **after** a per-head `justnorm`, k
   is only `justnorm`ed. Softmax scale is the textbook `1/sqrt(head_dim)`.
4. Output gate: `out = out · sigmoid(g_proj(x) · (s_gate ·
   sqrt(sqrt(hidden_size))/std)²)` then a `justnorm` over the whole head stack
   *before* `o_proj`.
5. MLP: `g = gate_proj(x)·(suv_gate·sqrt(hidden))`, `u = up_proj(x)·(suv_up·
   sqrt(hidden))`, `justnorm(silu(g)·u)`, then `down_proj`.
6. Loop: `input_h` = output of the input block; `state = input_h`; per loop
   `h = justnorm(state + input_h)`, run the recurrent block, `state =
   justnorm(h)`; the output block consumes `state`. Loop 0 therefore starts at
   `justnorm(2·input_h)`, which is not a no-op.
7. Logits: **no final norm**; `logits = (h · W_headᵀ) · logit_scale[v]/std`
   (per-vocabulary-entry scale, 131072 values).
8. KV cache: `per_loop` = one independent cache per (recurrent attention layer,
   loop). 56 caches for 2 loops (8 input + 20·2 recurrent + 8 output).

### 0.4 Ballast

None. All 507 tensors are classified (`backbone.embeddings`, 72 mixers,
`lm_head.weight`, `lm_head.logit_scale`); a leftover tensor is a converter
error. No vision tower, no MTP head.

### 0.5 Feasibility on this machine (RTX 5090, 32 GiB; 750 GiB host)

NVFP4 body + residual plane ≈ 1 B/param for 2.6 B projections ≈ 2.7 GiB;
FP16 edges (embed + head, 2×671 MiB) ≈ 1.3 GiB; small nGPT vectors F32
≈ 5 MiB; KV 56 slots × 2048 pos × 1024 dims × 2 B ≈ 235 MiB. Total ≈ 4.3 GiB
— fits with a huge margin, and even the FP16 (non-quantized) body would fit
(8.0 GiB), which keeps a precision fallback available if a gate goes red.

### 0.6 Tokenizer

GPT-2-style **byte-level BPE** (`tokenizer.json`, 131072 vocab, 269443 merges,
`tokenizer_class: PreTrainedTokenizerFast`, pre-tokenizer kind 6 = the phi/qwen
regex). 1000 added tokens: 0 `<unk>`, 1 `<s>`, 2 `</s>`, 10 `<|im_start|>`,
11 `<|im_end|>`, 12-17 `<think> </think> <tool_call> </tool_call>
<tool_response> </tool_response>` (**special:false**), 18-999 `<SPECIAL_nn>`.
bos=1, eos=11, pad=11, `add_bos_token: false`, untied embeddings.
`chat_template.jinja` = Nemotron-v3 ChatML: empty `system` turn when absent, a
`# Tools` + `<tools>` XML preamble, Qwen3-Coder XML tool calls
(`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`),
`<think>`/`<think></think>` reasoning markers, tool results as
`<tool_response>` inside a user turn.

### 0.7 Registry rows to extend

- `models/dory-1.0-4b.json`: family `dory`, vault dir
  `NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, class `DoryDenseModel`,
  kType `dory-1.0-4b`.
- `test_code_naming_guard.cpp`: family `dory`, `dory_tool_format` in
  `AllowedCommon()`.
- `test_model_suite_guard.cpp`: tool verdict row.
- `docs/SELFCHECK_MATRIX.md`, `docs/MODELS.md` (generated),
  `tests/perf/perf_baselines.json` + `test_perf_baseline.cpp`.

## Stages

1. **Converter** (`dory_dense_convert.cpp`, `lsxforge --dory`): classifier over
   the 6 real tensor families; fuse `q|k|v` → `qkv_proj` and `gate|up` →
   `gate_up_proj`; NVFP4 + residual plane for every projection; edges
   (`backbone.embeddings.weight`, `lm_head.weight`) stay FP16; the nGPT vectors
   (`sqk`, `s_gate`, `suv_gate`, `suv_up`, `attn_alpha`, `mlp_alpha`) and
   `lm_head.logit_scale` stay raw BF16. Index: `logestix-dory-nvfp4`.
2. **Quantization**: NVFP4 E2M1 + per-16 E4M3 + global scale, plus the residual
   plane (`w - dequant(base)`, decoded with `acc=true`), per the stage-2 recipe.
   Norms/biases are never quantized (there are none besides the nGPT vectors).
3. **Registration**: `DoryDenseModel` (ModelBase), frozen passport
   `lsxcommon::dory::frozen`, geometry cross-check in `Load`.
4. **Token bridge**: `StartDoc`-free ChatML; the 8 control sentinels are
   injected as vocab ids; everything else is BPE. 4b: native
   `dory_tool_format.{h,cpp}` renderer + the Qwen3-Coder XML parser (the wire is
   byte-identical), verified against the checkpoint template.
5. **Kernels** (`dory_dense_inference.cu`): family-local `justnorm`,
   `slerp_alpha`, qk-norm, gated-attention-norm, Suv-SwiGLU-norm, and a
   sliding-window causal GQA attention (the neutral dense attention has no
   window). Projections go through the model-neutral `lsx_nvfp4` layer.
6. **Shared-memory sandbox**: block reductions behind `__syncthreads()`, no
   atomics feeding `final_logits`.
7. retired.
8. **Gates**: SelfCheck (Paris, ≥3 bit-exact cycles), FSI/CODE/DET goldens
   captured from the source with `tools/dory_oracle.py`, tool round-trip on the
   converted weights, perf baseline at the canonical 2048-token protocol.
9. **Report**: `docs/model-reports/2026-11-10-NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16.md`.

## Known deviations (recorded, not silent)

- Attention/output projection shapes: `hidden (2560) != num_heads·head_dim
  (4096)`, so `q/k/v/o` are rectangular — the passport pins it, the neutral
  dense kernels (which assume square) cannot be reused for attention.
- `per_loop` KV caching only (the checkpoint default). `last_loop` is documented
  as a memory mode but is not the ingested configuration.
