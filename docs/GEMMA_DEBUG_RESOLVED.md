# Gemma-4-26B-A4B-it: SelfCheck resolved (2026-08-24)

SelfCheck passes: 'The capital of France is' → "The capital of France is
**Paris**." (greedy continuation of this model; 'Paris' among the first tokens),
bit-exact determinism across cycles.

## Root causes (in order of discovery)

1. **Incomplete framing**: after `<|turn>model\n` the template (enable_thinking
   default=false) adds an EMPTY thinking channel
   `<|channel>thought\n<channel|>` (=100, 45518, 107, 101). Without it the model
   predicts garbage. Framing thus became **model-dependent**:
   `IModel::BuildPromptInputIds` (virtual, default = raw encode), GLM and
   Gemma own their descriptors (`glm::kFraming` / `gemma::kFraming`),
   there is no central registry anymore.

2. **Double BOS**: the encoder itself adds bos_id inside the prompt + SelfCheck
   inserted a second BOS at the start. Now the BOS policy is part of the
   framing descriptor (leading_bos): auto-BOS is moved to position 0.

3. **layer_scalar / router.per_expert_scale were dropped**: the converter stores
   them as BF16 (2 bytes/value), while the loader checked the size against F32 →
   the tensors were silently skipped, layer_scalar was not applied (hidden states
   grew ~15x). Fix: dtype-aware size guards (+ a regression in the fixture/test).

4. **Attention scale**: HF Gemma4TextAttention uses scaling = 1.0
   (q/k are per-head normalized); it was 1/sqrt(head_dim).

5. **__constant__ kE2M1Table = garbage in the new TU**: a header __constant__
   array across several object libraries produced a broken copy in
   lsxcommon_gemma_cu (RTX 5090 / CUDA 13.3). Replaced with an inline function
   `nvfp4::E2M1Value(nib)` with immediates — no global storage anymore,
   ODR aliasing impossible.

6. **cudaEventElapsedTime on cudaEventDisableTiming events** returned a
   sticky "invalid resource handle"; the first getLastError in the decode loop
   treated it as a generation error → abort after the first token. TTFT/total
   moved to std::chrono.

## Verification

- Absolute reference: real HF transformers 5.15.1 (CPU, bf16) on the same
  framed prompt — top-1 'The', 'Paris' top-2, greedy =
  "The capital of France is **…"; C++ vs HF logits corr=0.985, top-10
  overlap 8/10.
- Suite: gemma 9/9, light 53/53, glm/kimi/deepseek/integrated/blackwell
  26/26. Preexisting test_tokenizer failures (FIM SP normalization) are a separate
  branch bug, unrelated to these changes.

## Methodology

Env-gated dumps of hidden states per layer/stage + a fresh torch reference from
the original safetensors (BF16) + a decisive comparison against the real HF
forward as ground truth.
