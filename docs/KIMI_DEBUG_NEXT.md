# Kimi-K2.7-Code: SelfCheck debugging continuation plan

RESOLVED 2026-08-24. SelfCheck passes: 'The capital of France is' → 'Paris'
(top-1 tok=17374), 3+ cycles with bit-exact determinism of final_logits.

## Root causes (found by independent layer-1 validation)

1. **INT4 zero-point convention** (`kimi_int4_dequant_kernel`): compressed-tensors
   pack-quantized symmetric int4 stores an unsigned nibble with an implicit
   zero-point of 8 (offset binary), while the kernel decoded it as two's
   complement (`nib>7 ? nib-16 : nib`). Correct: `(nib - 8) * scale`.
   Diagnostics: the nibble histogram of the source checkpoint is symmetric around
   q=8; group-mean statistics are ~0 only under offset-binary;
   corr(NVFP4, INT4 twos-comp)=0.994 confirms the converter encoded wrong
   weights. The unit test encoded the same error (circular check) — rewritten to
   offset-binary. The model was reconverted.

2. **Non-deterministic float atomicAdd** in three RMSNorm kernels
   (`kimi_rmsnorm_rows_kernel`, `kimi_rmsnorm_rows_op_kernel`,
   `kimi_kv_postprocess_kernel`): the warp summation order changed between runs
   → final_logits bits drifted. Replaced with a deterministic reduction:
   partials in shared memory + fixed sequential summation by thread 0.

## What was NOT the cause

- yarn attention scale: softmax-scale = 1/sqrt(192) is correct for this config
  (mscale=mscale_all_dim=1.0); HF DeepseekV3Attention applies a mscale² boost
  via yarn_get_mscale(factor, mscale_all_dim) — but that is a separate quantity
  from the rotary attention_factor; verified numerically on layer 1 (attention
  residual converges with max|d|=0.0017).
- kv_b_proj absorption, router renorm (w = sigmoid/sum * 2.827 — matches
  HF to 1e-5), rope positional offsets, shared_experts (weight 1.0) — all
  compared element-by-element against a fresh torch reference from the original
  INT4.

## Methodology (for future models)

Env-gated dumps of the hidden state after prefill layer N + a fresh torch
reference of the layer from the original safetensors (not from our converted
files) localize the divergence within minutes: attention residual → MoE →
shared/routed → router.
