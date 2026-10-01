# phi-4 ingestion design

## Context

`phi-4` (Microsoft, `/mnt/llm_models/models/phi-4`) is a dense decoder-only
transformer (`Phi3ForCausalLM`) with a standard Llama-style block: RMSNorm, GQA
attention with full-dim RoPE, SwiGLU MLP. It has no MoE, no QK-norm, no linear
attention, no biases, and no config-level attention/residual multipliers — the
attention score scale is the textbook `1/sqrt(head_dim)`.

The source checkpoint stores the attention and MLP projections **already
fused**:

- `model.layers.N.self_attn.qkv_proj.weight` — `[(NH + 2*NKV)*HD, H]`, ordered
  `q | k | v`.
- `model.layers.N.mlp.gate_up_proj.weight` — `[2*I, H]`, ordered `gate | up`.

Frozen geometry (from `config.json`, 2026-10-01):

| field | value |
|---|---|
| hidden_size | 5120 |
| num_hidden_layers | 40 |
| num_attention_heads | 40 |
| num_key_value_heads | 10 |
| head_dim (hidden/heads) | 128 |
| intermediate_size | 17920 |
| vocab_size | 100352 |
| rms_norm_eps | 1e-5 |
| rope_theta | 250000 |
| attention scale | `1/sqrt(128)` (no config multiplier) |
| hidden_act | silu |
| tie_word_embeddings | false |
| bos_token_id / eos_token_id / pad_token_id | 100257 / 100265 / 100349 |
| special tokens | bos `<|endoftext|>`, eos `<|im_end|>`, pad `<|dummy_85|>` |

Chat template (embedded in `tokenizer_config.json`):

```
<|im_start|>{role}<|im_sep|>{content}<|im_end|>
```

Phi-4 ships **no native tool-calling syntax**; its chat template carries no
`tools` branch.

This is the FIRST `phi` family member. No backward compatibility is required
(Blackwell-only repo). The structural template is the `granite_dense_*` family
(also dense GQA + SwiGLU + full RoPE + NVFP4, already fused qkv/gate_up), so the
implementation mirrors it with a new geometry, a computed attention scale, the
source's fused tensor names, and phi's ChatML framing.

## Target layout (Model Ingestion Protocol, 8 stages)

Family `phi`, type `dense`. Format tag `logestix-phi`. Model name in
`model.index.json`: `Phi-4`. `kType` (factory) = `phi-4`.

Files:

```
libs/lsxcommon/include/lsxcommon/phi_dense_constants.h    frozen passport
libs/lsxcommon/include/lsxcommon/phi_dense_convert.h
libs/lsxcommon/src/phi_dense_convert.cpp                   stage 1/2
libs/lsxcommon/include/lsxcommon/phi_dense_load.h
libs/lsxcommon/src/phi_dense_load.cpp                      stage 3 (host)
libs/lsxcommon/include/lsxcommon/phi_dense_inference.h
libs/lsxcommon/src/phi_dense_inference.cu                  stage 5
libs/lsxcommon/src/phi_dense_model.h
libs/lsxcommon/src/phi_dense_model.cpp                     stage 3/4b/8
tests/test_phi_dense_fixture.h
tests/test_phi_dense_convert.cpp        label convert, TIMEOUT 120
tests/test_phi_dense_loader.cpp         label inference
tests/test_phi_dense_model.cpp          label inference
tests/test_phi_dense_kernels.cu         label inference
tests/test_phi_dense_selfcheck.cpp      label inference, SKIP_RETURN_CODE 4
tests/test_phi_dense_fsi_selfcheck.cpp  label inference, SKIP_RETURN_CODE 4
tests/test_phi_dense_code_selfcheck.cpp label inference, SKIP_RETURN_CODE 4
tests/test_phi_dense_det_selfcheck.cpp  label inference, SKIP_RETURN_CODE 4
tests/test_phi_tool_framing.cpp         label fast (host framing/parse)
```

On-disk converted layout mirrors `logestix-granite`:

```
<out>/logestix/model.index.json                  format + geometry + vram
<out>/logestix/pinned/global.arrow               final norm (raw BF16 -> FP16)
<out>/logestix/pinned/global_nvfp4.arrow         embed + lm_head (NVFP4)
<out>/logestix/pinned/layer_N.arrow              input/post norms
<out>/logestix/pinned/layer_N_nvfp4.arrow        qkv_proj + o_proj + gate_up_proj + down_proj
<out>/logestix/pinned/tokenizer_{vocab,merges}.arrow
```

## Stage details

1. **Converter** (`phi_dense_convert.cpp`, `lsxforge --phi`). Parse
   `config.json` (flat phi3 root: `hidden_size`, `num_hidden_layers`,
   `num_attention_heads`, `num_key_value_heads`, `intermediate_size`,
   `vocab_size`, `rms_norm_eps`, `rope_theta`, `bos/eos/pad_token_id`;
   `head_dim` derived = `hidden/heads`). Classify tensors:
   `model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`,
   `model.layers.N.{input_layernorm,post_attention_layernorm}.weight`,
   `self_attn.qkv_proj.weight`, `self_attn.o_proj.weight`,
   `mlp.gate_up_proj.weight`, `mlp.down_proj.weight`. Unknown names (e.g.
   a future `rotary_emb` buffer) are skipped, not fatal.
   Integrity: `VerifyAllTensorReads`, `CheckTensorCoverage` (expected names
   built from the frozen topology), `CheckLayerContiguity`,
   `CheckGeometryFields` (config vs tensor shapes), `BuildAndVerifyTokenizer`
   (bos+eos required), `VerifyConvertedModel`. Emits `logestix-phi`.
2. **NVFP4** every dense linear + embed/lm_head via `gpu::quantize_tensor_gpu`
   (E2M1 + per-16 E4M3 scales + global float scale). Norms stay FP16.
   `qkv_proj` and `gate_up_proj` are already fused in the source, so each is
   quantized as one record with a single global scale (the decode layout the
   granite loader expects: one streaming GEMV for q/k/v and one for gate/up).
3. **Loader** (`phi_dense_load.cpp`): NVFP4 pinned weights on one stream; the
   fused records are exposed as non-owning row-slices (`q|k|v`, `gate|up`);
   norms raw FP16; arena ownership.
4. **Registration + model** (`phi_dense_model.{h,cpp}`). `Phi4Model final :
   ModelBase`, `kType = "phi-4"`, frozen `static_assert`s, `ValidateGeometry`
   against `phi::PhiTopology` (skipped under `SyntheticGeometryAllowed()`).
   Registered in `model_registry.cpp`. `Infer` opens
   `LsxFastMathScope(EffectiveFastMathMode(cfg_))`.
5. **Tokenizer bridge**: reuse `BuildTokenizerArrowFiles`; phi is a byte-level
   GPT-2/tiktoken BPE (`tokenizer.json` + `vocab.json` + `merges.txt`);
   `bos_id=100257`, `eos_id=100265`, `pad_id=100349`. Control markers are
   injected as their own vocab ids.
6. **CUDA kernels** (`phi_dense_inference.cu`): dense loop using the
   model-neutral `lsx_nvfp4` GEMM/GEMV for all projections and the shared
   `lsx_kernels_{norm,act,math}.cuh` primitives; a phi RoPE kernel (full
   `head_dim`, theta=250000), a causal GQA attention kernel scaled by
   `1/sqrt(head_dim)`, SwiGLU MLP. One stream, no per-token sync.
7. **Shared-memory sandbox**: block-wide reductions via shared memory with
   `__syncthreads()`; no `atomicAdd` feeding `final_logits`.
8. **Passport** (`phi_dense_constants.h`, namespace `lsxcommon::phi::frozen`):
   frozen geometry is the single source of truth; no cross-includes.

### Step 4b — tool-calling bridge

Phi-4 has no native tool syntax, so the generic parser floor applies, but the
chat framing must be phi's own ChatML (not Hermes):

- `BuildPromptInputIds`: wrap a single user message as
  `<|im_start|>user<|im_sep|>{body}<|im_end|><|im_start|>assistant<|im_sep|>`.
- `BuildConversationInputIds`: emit the phi ChatML turn sequence; when tool
  schemas are present, render the generic tools block as system text (the
  Hermes-style schema text) inside an `im_start|system` turn, since phi has no
  native tool tags.
- `ParseToolCalls`: use the default `IModel` implementation
  (`tools::ParseToolCallsText` then `tools::ParseBareToolCallJson`).

Control markers (`<|im_start|>`, `<|im_sep|>`, `<|im_end|>`) are special
`added_tokens`; they are looked up with `arrow_tokenizer_lookup` and injected as
ids, with only free text BPE-encoded. The auto-prepended `bos_id` is dropped
inside the frame.

## Testing

TDD per stage: each production stage lands with the failing test that pins it,
and the ctest label in the same commit.

- **stage 1** `test_phi_dense_convert.cpp`: builds a synthetic tiny phi
  checkpoint (2 layers, hidden 64) with `test_synth_checkpoint.h` primitives,
  writes `qkv_proj`/`gate_up_proj` as pre-fused tensors + config + a synthetic
  tokenizer, runs the full converter, and asserts the index metadata (format,
  geometry, special ids), the pinned arrow files, the fused record names, and
  the VRAM block. `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` via
  `set_tests_properties(... ENVIRONMENT ...)`; bounded TIMEOUT 120.
- **stage 3** `test_phi_dense_loader.cpp` / `test_phi_dense_model.cpp`: load the
  converted synthetic fixture, assert geometry-mismatch rejection, registry
  round-trip via `LsxModelFactory`, and the fast-math scope marker.
- **stage 5** `test_phi_dense_kernels.cu`: RoPE / RMSNorm / causal GQA parity
  vs a host CPU reference, including the `1/sqrt(head_dim)` scale and the
  split-attention combine path.
- **stage 4b** `test_phi_tool_framing.cpp`: render structure, parse, and
  tokenizer round-trip on the synthetic tokenizer fixture (add the phi
  sentinels to its `added_tokens`).
- **stage 8** `test_phi_dense_selfcheck.cpp`: `"The capital of France is"` ->
  `"Paris"`, >=3 cycles bit-exact `final_logits`, `SKIP_RETURN_CODE 4`; plus the
  FSI/CODE/DET parity gates and a perf baseline target.
- **guards** (same commits): `test_code_naming_guard.cpp` (`phi` family token +
  accepts-test), `test_model_independence_guard.cpp` (passport row + consumer),
  `test_model_suite_guard.cpp` (convert/load/model/kernels/selfcheck/fsi/code/det
  rows), `test_ctest_labels_guard.cpp` (labels only).

## Real-weight verification

1. `cmake --preset linux-debug && cmake --build --preset linux-debug`
2. `lsxforge --phi --input /mnt/llm_models/models/phi-4 --output /mnt/llm_models/models/phi-4`
   (in-place `logestix/`).
3. Capture goldens from the source safetensors with
   `tools/{fsi,code,det}_capture.py --type phi-4`, commit to
   `tests/fixtures/{fsi,code,det}/phi-4.json`.
4. Run the selfcheck / FSI / CODE / DET targets on the converted model at
   `kOff`; a failure is a real finding, not a gate to loosen.
5. Run `test_perf_baseline_phi_4` at the canonical 2048-token protocol and
   record the measured t/s in `tests/perf/perf_baselines.json`.

GPU: RTX 5090 (CC 12.0, 32 GB). All GPU work observes the repo's
one-GPU-load-at-a-time rule.

## Out of scope

Multi-token KV-cache incremental decode optimization, Tensor-Core W4A4
attention fusion, and native tool-calling (phi-4 defines none — the generic
parser floor is correct and documented).

## Risks

- **Fused-order fidelity**: q/k/v and gate/up are already fused in the source;
  the split order (`q|k|v`, `gate|up`) is verified against the HF `Phi3`
  forward chunking before the loader slices are wired.
- **Attention scale**: phi applies `1/sqrt(head_dim)` with no config knob; this
  constant is pinned in the passport and asserted in the kernel parity test.
- **Chat-template vs golden**: FSI/CODE/DET goldens feed captured `input_ids`
  verbatim, so no HF<->logestix framing parity is assumed; the ChatML framing is
  tested separately for serving correctness.
