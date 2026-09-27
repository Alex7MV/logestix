# CODESelfCheck — capture and parity findings

- **Date**: 2026-09-27
- **Scope**: CODESelfCheck
  (`docs/superpowers/specs/2026-09-27-code-selfcheck-design.md`)
- **Status**: mechanism complete; Qwen3.6-35B-A3B, Qwen3.8-Flash-Next and
  GLM-5.3-Flash are wired into CI. The remaining families' tests SKIP until
  their goldens are captured and their divergences are resolved.

The CODE gate is **model-defined**: the golden is the original safetensors
model's own extracted answer (not the prompt template). The converted model is
required to reproduce that normalized answer **exactly** (no algebraic
canonicalization) at `FastMathMode::kOff`, so a syntactically different but
mathematically equivalent answer is a failure by design.

## Prompt set

| id | kind | anchor | expected |
|---|---|---|---|
| CODE-TEST-01 | json | `root` | `{"root":{"nodes":{"leaf":true}}}` |
| CODE-TEST-02 | assignment | `m` | `m=l+((h-l)/2)` |
| CODE-TEST-03 | regex | `[0-9A-F]` | `\b[0-9A-F]{4}\b` |

## Runs on the FSI-passing families

`test_<family>_fsi_selfcheck` currently passes for four families; CODESelfCheck
was run for each of them. Results on the converted model at `kOff`:

| Family | FSI | CODE golden | CODESelfCheck |
|---|---|---|---|
| Qwen3.6-35B-A3B | PASS | source-captured, committed | **PASS 3/3** |
| Qwen3.8-Flash-Next | PASS | source-captured (`qwen4_exp`), committed | **PASS 3/3** |
| GLM-5.3-Flash | PASS | template-derived (`--template-golden`, 512), committed | **PASS 3/3** |
| granite-4.2-30b | PASS | source-captured (not committed) | **FAIL 1/3** (CODE-01) |

Committed goldens:
`tests/fixtures/code/qwen3.6-35b-a3b.json`,
`tests/fixtures/code/qwen3.8-flash-next.json`,
`tests/fixtures/code/glm-5.3-flash.json`.

Capture commands:

```bash
python tools/code_capture.py --model /mnt/llm_models/models/Qwen3.6-35B-A3B \
    --type qwen3.6-35b-a3b --out tests/fixtures/code/qwen3.6-35b-a3b.json \
    --device-map auto --max-new-tokens 256

python tools/code_capture.py --model /mnt/llm_models/models/Qwen3.8-Flash-Next \
    --type qwen3.8-flash-next --out tests/fixtures/code/qwen3.8-flash-next.json \
    --max-new-tokens 256            # qwen4_exp loader is in OVERRIDES

# unrunnable source -> template golden (GLM-5.3-Flash only)
python tools/code_capture.py --model /mnt/llm_models/models/GLM-5.3-Flash \
    --type glm-5.3-flash --out tests/fixtures/code/glm-5.3-flash.json \
    --template-golden --max-new-tokens 512
```

## Capture feasibility (original safetensors, HF transformers 5.15.1)

| Family | Result |
|---|---|
| Qwen3.6-35B-A3B | captured (all 3 exact); converted PASS 3/3 |
| Qwen3.8-Flash-Next | captured with the `qwen4_exp` loader (all 3 exact); converted PASS 3/3 |
| granite-4.2-30b | captured (all 3 exact); converted **FAIL 1/3** (CODE-01) |
| gemma-4-26B-A4B-it | captured (all 3 exact); converted **FAIL 1/3** (CODE-02) |
| GLM-5.3-Flash | not capturable (`glm5_next` unknown); template-derived; converted **PASS 3/3** (512 + prompt-echo cut) |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | source present and capturable (FSI captured it); not captured — the converted model emits incoherent text under FSI |
| ornith-1.5-35B-A3B | not capturable — MLX `language_model.*` keys need a prefix remap |
| DeepSeek-V4-Flash | not capturable locally — FP8 149 G does not fit the 32 GB GPU |
| DeepSeek-V4.1-Flash | not capturable — ships its own inference stack (`deepseek_v41` unknown) |
| GigaChat3.5-432B | not capturable locally — FP8 416 G, same VRAM blocker |
| GLM-5.2, Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-35B-A3B, Kimi-K2.7-Code | source absent from the vault |

## Findings (divergences; goldens deliberately not committed)

A golden is committed only for a family whose converted model passes (same
discipline as FSISelfCheck). The divergences below are recorded here, not masked.

### granite-4.2-30b — CODE-TEST-01 (json)

Both source and converted wrap the answer in a short chain-of-thought. The
source emits the fully balanced object; the converted model drops the final
closing brace:

- source raw tail: `...</think>\n{"root":{"nodes":{"leaf":true}}}`
- converted raw tail: `...</think>\n{"root":{"nodes":{"leaf":true}}`

**Root cause (2026-09-27): a single-step argmax flip on a rare composite token,
not a launch nuance.** Ruled out one by one:

- **Not a token-budget truncation.** Re-running with `max_new_tokens=1024`
  reproduces the exact same 27-token output (the model stops on EOS far below
  the budget).
- **Not EOS/stop configuration.** Granite has a single `eos_id = 100257`
  (`<|im_end|>`), not the gemma-style multi-EOS list; `test_granite_dense_fsi_selfcheck`
  passes, and CODE-02/03 keep their final characters.
- **Not the sampler.** `DefaultSampler()` is `GreedySampler` (argmax), matching
  the source capture's `do_sample=False`.
- **Not framing/tokenization.** The golden `input_ids` are the source's
  HF-framed ids fed verbatim; the converted answer prefix is **token-identical**
  to the source up to the final brace.
- **Not decode/off-by-one.** The EOS token is appended before the loop stops and
  is dropped only by `skip_special_tokens`; FSI and CODE-02/03 keep their last
  tokens.

Converted token stream (27): `... 794(":") 1904("true") 3500("}}")
100257("<|im_end|>")`. Source token stream: `... 1904("true") 76642("}}}")
100257("<|im_end|>")`. The tokenizer maps the whole `}}}` to one token (76642)
and `}}` to 3500.

Source (BF16) per-step top-5 at the step after `true`:

| rank | token | text | logit |
|---|---|---|---|
| 1 | 76642 | `}}}` | 31.375 |
| 2 | 3500 | `}}` | 25.125 |
| 4 | 100257 | `<|im_end|>` | 17.625 |

The converted (NVFP4) model picks the source's **#2** (`}}`) and then EOS. So
the converted logits shifted enough to demote the rare `}}}` token below its
sub-token `}}` — a genuine quantization-sensitivity divergence (the same profile
as gemma/nemotron in `FSI_SELFCHECK_FINDINGS.md`), not an engine/budget/EOS
artifact. (A secondary CoT drift drops " No whitespace" in the converted
reasoning, reinforcing the quantization signature.) The gate is left strict and
no golden is committed.

### gemma-4-26B-A4B-it — CODE-TEST-02 (assignment): the extra parentheses

The source answers `m=l+((h-l)/2)`; the converted model answers
`m=(l+(h-l)/2)` — the same formula with the whole RHS additionally
parenthesized.

**Root cause (2026-09-27): a quantization argmax flip onto operator-plus-paren
composite tokens, not a launch nuance.** The token streams are identical up to
the first `m`; the divergence is the next token, and there is a second at the
`+`:

| step | source (BF16) top-1 | source top-2 | converted (NVFP4) picks |
|---|---|---|---|
| 1 (after `m`) | `=` id 236784, logit 23.0 | `=(` id 9036, logit 13.4 | **`=(`** |
| 3 | `+` id 236862, logit 23.1 | `+(` id 24064, logit 22.1 | **`+(`** |

Gemma's tokenizer has composite tokens that fuse the operator with an opening
paren (`=(`, `+(`); the converted logits demote the plain `=`/`+` below them, so
the model opens a parenthesis the source does not and closes it at the end. Step
1 flips a clear 9.6-logit margin to the source's #2; step 3 is already a near
tie (1.0). The raw is exactly `m=(l+(h-l)/2)` — the model's answer, no prompt
echo and no truncation, so `CutPromptEcho` does not apply.

CODE-TEST-01 and CODE-TEST-03 pass. Because CODESelfCheck compares exactly (no
associativity/parenthesis rewriting, unlike FSISelfCheck), the extra outer
parentheses are a failure by design; the golden stays uncommitted. Relaxing the
assignment kind to algebraic canonicalization would make it pass, but that
reverses the syntactic-strictness policy (see 2026-09-27 decision).

### GLM-5.3-Flash — resolved (resolved by budget + prompt-echo cut)

The source cannot run locally, so the golden is template-derived (`--template-
golden`: input_ids from the official `chat_template.jinja` + tokenizer, the
prompt's Expected Template as `expected`), matching the FSI stopgap.

The original failures were **mostly harness nuances**, now fixed:

- **Budget.** At `max_new_tokens=256` CODE-02 is still reasoning when the loop
  hits the cap, so nothing is extracted. The committed golden uses **512**; the
  model then emits `</think>m=l+((h-l)/2)` exactly. (At 256 CODE-03 "passed"
  only because the extractor found the template echo inside the reasoning — a
  false pass.)
- **Prompt-echo cut.** The converted model does not emit a turn-end token; after
  its answer it reproduces the prompt, with no separator. For CODE-03:
  `Output: \`\b[0-9A-F]{4}\b\`</think>\b[0-9A-F]{4}\b[CONTEXT]: CODE-Validation-Core-03 ...`.
  The regex extractor legitimately consumed `[CONTEXT]` (all regex-token chars).
  The answer itself was correct.

**Fix:** the golden now carries the prompt `body`, and the harness runs
`CutPromptEcho` (after the `</think>` cut) — if the body's first 32 chars occur
in the decoded text it truncates there. This is a harness bound, not a
relaxation of the exact comparison: the extracted answer is still compared
byte-for-byte. With 512 + the cut, GLM-5.3-Flash PASSes 3/3; the golden is
committed.

Underlying real behaviour to watch: the converted model never emits any of its
EOS tokens (`[154820 <|endoftext|>, 154827 <|user|>, 154829 <|observation|>]`,
correctly stored and checked as a set in `RunFlashDsaDecode`) and degenerates
into prompt repetition after the answer. `CutPromptEcho` contains it; a
terminating model would not need it.

### Qwen3.8-Flash-Next capture recipe (`qwen4_exp`)

The installed transformers 5.15.1 ships `Qwen4Exp*`, but the checkpoint declares
`model_type = qwen4_exp` and uses `qwen_sparse_attention`, and ships the PLE
n-gram embedding as 128 shards. `tools/code_capture.py` mirrors the FSI loader
(`OVERRIDES["qwen3.8-flash-next"] = {"loader": "qwen4_exp"}`): patch
`ALLOWED_ATTN_LAYER_TYPES` + the dynamic/static cache mappings, then fuse the
shards. Capture runs on CPU (~22 min) and the converted model passes 3/3.

### Other families

No golden is committed, so their `test_<family>_code_selfcheck` targets SKIP
(SKIP_RETURN_CODE 4), never fail.
