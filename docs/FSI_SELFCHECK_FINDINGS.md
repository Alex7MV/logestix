# FSISelfCheck — capture and parity findings

- **Date**: 2026-09-26
- **Scope**: pilot + rollout of FSISelfCheck
  (`docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`)
- **Status**: mechanism complete; two families (Qwen3.6-35B-A3B and
  Qwen3.8-Flash-Next) are wired into CI. The remaining families' tests SKIP
  until their goldens are captured and their divergences are resolved.

The FSI gate is **model-defined**: the golden is the original safetensors
model's own extracted answer (not the prompt template). The converted model is
required to reproduce that normalized answer at `FastMathMode::kOff`.

## Capture feasibility (original safetensors, HF transformers)

| Family | Result |
|---|---|
| Qwen3.6-35B-A3B | captured (all 3 exact) |
| Qwen3.8-Flash-Next | captured (all 3 exact; experimental `qwen4_exp` arch loaded with patches, see below) |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | captured (all 3 exact) |
| gemma-4-26B-A4B-it | captured (FSI-01/02 rephrased; template warnings) |
| granite-4.2-30b | rejected — FSI-02 extracted answer has unbalanced parentheses |
| ornith-1.5-35B-A3B | not capturable — config loads as `qwen3_5_moe`, but the MLX checkpoint uses `language_model.*` key names, so all HF params MISSING (needs a key-prefix remap) |
| DeepSeek-V4-Flash | recognized (`deepseek_v4`); FP8 loads after installing `accelerate` + `kernels==0.16.0`, but the FP8 linear kernel is CUDA-only and 149 G does not fit the 32 GB GPU (`device_map=auto` OOM) — needs a larger-VRAM host |
| GigaChat3.5-432B | recognized (`gigachat3_5`) + custom code; FP8 (416 G) — same CUDA/VRAM blocker as DeepSeek-V4-Flash, plus RAM pressure |
| DeepSeek-V4.1-Flash | not capturable locally — `deepseek_v41` is not in transformers 5.15.1 and the release ships its **own inference stack** (`inference/model.py`, `convert.py`, `generate.py`; tilelang FP8/FP4 kernels, tensor-parallel, target 2×80 GB GPU), not HF remote code. Do not confuse it with the flat FP8 + FP4 MoE checkpoint in the vault. |
| GLM-5.3-Flash | not capturable — `glm5_next` unknown to transformers 5.15.1 |
| GLM-5.2, Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-35B-A3B, Kimi-K2.7-Code | source absent from the vault |

### Qwen3.8-Flash-Next (`qwen4_exp`) capture recipe

The installed transformers 5.15.1 ships `Qwen4Exp*` classes, but the checkpoint
declares `model_type = qwen4_exp` and uses a layer type (`qwen_sparse_attention`)
that this build does not accept/route. Capture works after (documented in
`tools/fsi_capture.py`, override `loader = "qwen4_exp"`):

1. Extend `transformers.configuration_utils.ALLOWED_ATTN_LAYER_TYPES` with
   `qwen_sparse_attention`.
2. Register `qwen_sparse_attention` in `cache_utils.DYNAMIC_LAYER_TYPE_MAPPING`
   / `STATIC_LAYER_TYPE_MAPPING` (→ `Dynamic/StaticIndexedLayer`).
3. Build `Qwen4ExpForCausalLM(text_config)` in BF16 and load the checkpoint
   directly (source prefix `model.language_model.*` → `model.*`; skip the
   absent-from-text `model.visual.*` tower).
4. Fuse the PLE n-gram embedding: 128 `ngram_embedding.shard_N.weight` shards
   ([2500012, 160]) fill `ngram_embedding.weight` ([320001536, 160]) in order.

## Converted-model parity (source golden vs converted, `kOff`)

| Family | Result |
|---|---|
| Qwen3.6-35B-A3B | **PASS 3/3** (wired into CI) |
| Qwen3.8-Flash-Next | **PASS 3/3** (wired into CI) |
| gemma-4-26B-A4B-it | FSI-02 PASS after algebraic canonicalization; FSI-01 and FSI-03 diverge |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | FAIL 3/3 — converted model emits incoherent text |

### gemma-4-26B-A4B-it

- FSI-01: source `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)` (annuity-due) vs
  converted `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` (ordinary annuity) — genuinely
  different expression; greedy decoding diverged.
- FSI-03: source `Drawdown>Threshold` vs converted
  `Drawdown>Thresholdthought` — the converted model appended a continuation
  token inside the identifier.

### NVIDIA-Nemotron-3.5-Lightning-30B-A3B

- FSI-01 raw starts `The correct formula for the present value (PV)
  calculation is:` and the answer line does not match.
- FSI-02 raw is a long run of `0` digits (degenerate).
- FSI-03 raw is ``The requested output is: `0011` ``.

The existing "The capital of France is" -> "Paris" selfcheck passes on this
converted model, so the divergence only appears on the longer/complex FSI
prompts. This is a real converted-model finding, not a gate artifact.

#### Nemotron conversion / acceleration investigation (2026-09-26)

Checked whether the divergence is a bug introduced by the perf/"acceleration"
commits. Evidence (all negative for an acceleration bug):

| Check | Result |
|---|---|
| Routed-expert device LRU (`3fb1dbb`) | **Not the cause** — cache loads fully resident (`2944/2944` slots, only the prewarm misses) and the model still diverges. |
| Batched routed-expert GEMV (`1164799`) | **Not the cause** — at `kOff` it falls back to the pre-acceleration per-expert `RunLsxNvfp4Gemm`; `kAuto` and `kOff` fail identically. |
| Parallel router top-k (`242463f`) | Reviewed: same max/tie-break (lowest index) as the old serial scan; unit-tested (`test_nemotron_moe_kernels`). |
| FP16 relu2 / weighted-sum refactor | Numerically identical (relu^2 and the weighted sum are still computed in FP32; accumulation order unchanged). |
| Recurrent state leak across `Infer` | **No** — `ResetNemotronSession` zeroes mamba/conv/KV state at the start of every `Infer`. |
| KV/context overflow | **No** — session allocated for `kMaxSeq = 4096`. |
| Chat framing / special tokens | **Fine** — chat-framed "The capital of France is" answers correctly; high-id BPE tokens (` markdown` = 121468, ` Boundary` = 130036) match HF exactly. |
| Conversion/kernel/loader integrity | `test_nemotron_moe_{convert,kernels,loader,model,math}` all pass. |

Length sweep vs HF (greedy): exact match at 5 / 68 / 215 prompt tokens; at 425
tokens the converted model emits `"In"` where HF emits `"The"` (a near-tie
flip). On raw prompts the converted model matches HF token-for-token, including
the high-id tokens.

**Conclusion:** no evidence of an acceleration bug. The converted NVFP4 model
matches BF16 on short/medium and many long prompts, but greedy decoding
occasionally diverges when the state has accumulated quantization error, and on
some long/ambiguous FSI prompts that first flip cascades into a different
answer. This is a quantization-sensitivity limitation of the 4-bit conversion,
not the LRU/batched/router optimizations. A definitive proof would require a
higher-precision reference (BF16/FP16 resident) or per-layer hidden-state
parity against HF; the FSI gate correctly surfaces the divergence.

## DeepSeek-V4.1-Flash: API-derived golden experiment (2026-09-26)

The source cannot be run locally (own FP8/FP4 tilelang tensor-parallel stack).
As a stopgap the assistant (itself DeepSeek-V4.1-Flash) answered the three FSI
prompts and those answers were used as the golden, with `input_ids` produced by
the release's official `encoding/encoding.py` (`thinking_mode="chat"`) plus
`tokenizer.json`.

Result on the converted model (`kOff`): FSI-03 matched `Drawdown>Threshold`;
FSI-01 was one `)` short of the golden; FSI-02 ignored "use the provided
variable names" and substituted the numeric legs, additionally dropping zeros
(`100000` instead of `100000000`). Using the runtime's own framing (raw BPE via
`EncodeRawPrompt`, the default `BuildPromptInputIds`) was worse (garbage/empty),
so the divergence is model/runtime-level, not framing.

The golden was not committed, so `test_deepseek_moe_v41_fsi_selfcheck` SKIPs.
DeepSeek-V4-Flash is the same situation (its vault copy is a flat FP8+FP4
pre-quantized layout consumed by the repo's own loader, not an HF checkpoint).

## Normalization contract implemented here

- Preferred answer line: the first line where the anchor is immediately
  followed (spaces allowed) by `=` (arithmetic) or a comparison operator;
  fall back to the first line containing the anchor.
- Arithmetic answers are canonicalized by parsing the RHS (numbers,
  identifiers, `exp(...)` calls, `+ - * /`, parentheses, unary minus) and
  re-printing with minimal parentheses. Redundant parentheses therefore compare
  equal. Comparison answers are whitespace-normalized and bounded to the
  anchor, operator and one identifier.

As-built delta vs the spec: the golden `expected` is model-defined (spec
§3.2 updated); arithmetic canonicalization and operator-aware line selection
were added after observing real model outputs.

## Next steps

0. FP8 capture requires `accelerate` and `kernels` (`kernels==0.16.0`) installed
   and a host whose VRAM can hold the FP8 layers; pass `--device-map auto` on
   such a host. This session installed `accelerate 1.15.0` and `kernels 0.16.0`.
1. Wire each remaining family into CI by capturing its golden once the source
   is available/framework-loadable, and resolving any genuine divergence.
2. Nemotron: investigate the converted model on the FSI prompts (long-input
   path) independently of FSI.
3. Gemma FSI-01/FSI-03: determine whether the divergence is numeric or
   template-related.
