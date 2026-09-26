# FSISelfCheck — capture and parity findings

- **Date**: 2026-09-26
- **Scope**: pilot + rollout of FSISelfCheck
  (`docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`)
- **Status**: mechanism complete; a single family (Qwen3.6-35B-A3B) is wired
  into CI. The remaining families' tests SKIP until their goldens are captured
  and their divergences are resolved.

The FSI gate is **model-defined**: the golden is the original safetensors
model's own extracted answer (not the prompt template). The converted model is
required to reproduce that normalized answer at `FastMathMode::kOff`.

## Capture feasibility (original safetensors, HF transformers)

| Family | Result |
|---|---|
| Qwen3.6-35B-A3B | captured (all 3 exact) |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | captured (all 3 exact) |
| gemma-4-26B-A4B-it | captured (FSI-01/02 rephrased; template warnings) |
| granite-4.2-30b | rejected — FSI-02 extracted answer has unbalanced parentheses |
| ornith-1.5-35B-A3B | not capturable — config loads as `qwen3_5_moe`, but the MLX checkpoint uses `language_model.*` key names, so all HF params MISSING (needs a key-prefix remap) |
| DeepSeek-V4-Flash | recognized (`deepseek_v4`); FP8 loads after installing `accelerate` + `kernels==0.16.0`, but the FP8 linear kernel is CUDA-only and 149 G does not fit the 32 GB GPU (`device_map=auto` OOM) — needs a larger-VRAM host |
| GigaChat3.5-432B | recognized (`gigachat3_5`) + custom code; FP8 (416 G) — same CUDA/VRAM blocker as DeepSeek-V4-Flash, plus RAM pressure |
| DeepSeek-V4.1-Flash | not capturable — `deepseek_v41` unknown to transformers 5.15.1 (`accelerate` does not add arch support) |
| GLM-5.3-Flash | not capturable — `glm5_next` unknown to transformers 5.15.1 |
| Qwen3.8-Flash-Next | not capturable — `qwen4_exp` unknown to transformers 5.15.1 |
| GLM-5.2, Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-35B-A3B, Kimi-K2.7-Code | source absent from the vault |

## Converted-model parity (source golden vs converted, `kOff`)

| Family | Result |
|---|---|
| Qwen3.6-35B-A3B | **PASS 3/3** (wired into CI) |
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
