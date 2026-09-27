# CODESelfCheck — capture and parity findings

- **Date**: 2026-09-27
- **Scope**: pilot of CODESelfCheck
  (`docs/superpowers/specs/2026-09-27-code-selfcheck-design.md`)
- **Status**: mechanism complete; Qwen3.6-35B-A3B is wired into CI. The
  remaining families' tests SKIP until their goldens are captured and their
  divergences are resolved.

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

## Capture feasibility (original safetensors, HF transformers 5.15.1)

| Family | Result |
|---|---|
| Qwen3.6-35B-A3B | captured (all 3 exact); converted model PASS 3/3 |
| granite-4.2-30b | captured (all 3 exact); converted model **FAIL 1/3** (CODE-01) |
| gemma-4-26B-A4B-it | captured (all 3 exact); converted model **FAIL 1/3** (CODE-02) |
| qwen3.8-flash-next | capturable with the `qwen4_exp` custom loader (same patches as the FSI capture); not captured yet |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | source present and capturable (FSI captured it); not captured for CODE — the converted model emits incoherent text under FSI, so a golden would fail |
| GLM-5.3-Flash | not capturable — `glm5_next` unknown to transformers 5.15.1 |
| ornith-1.5-35B-A3B | not capturable — MLX `language_model.*` keys need a prefix remap |
| DeepSeek-V4-Flash | not capturable locally — FP8 149 G does not fit the 32 GB GPU |
| DeepSeek-V4.1-Flash | not capturable — ships its own inference stack (`deepseek_v41` unknown to transformers) |
| GigaChat3.5-432B | not capturable locally — FP8 416 G, same VRAM blocker |
| GLM-5.2, Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-35B-A3B, Kimi-K2.7-Code | source absent from the vault |

Capture command used for the pilot (granite needs low-effort thinking):

```bash
python tools/code_capture.py --model /mnt/llm_models/models/Qwen3.6-35B-A3B \
    --type qwen3.6-35b-a3b --out tests/fixtures/code/qwen3.6-35b-a3b.json \
    --device-map auto --max-new-tokens 256
```

## Pilot result — Qwen3.6-35B-A3B (committed golden)

Source and converted both answer all three prompts verbatim; the converted model
reproduces the golden at `kOff`. Golden:
`tests/fixtures/code/qwen3.6-35b-a3b.json`.

## Findings (divergences; goldens deliberately not committed)

A golden is committed only for a family whose converted model passes (same
discipline as FSISelfCheck). The divergences below are recorded here, not masked.

### granite-4.2-30b — CODE-TEST-01 (json)

Both source and converted wrap the answer in a short chain-of-thought. The
source emits the fully balanced object; the converted model drops the final
closing brace:

- source raw tail: `...</think>\n{"root":{"nodes":{"leaf":true}}}`
- converted raw tail: `...</think>\n{"root":{"nodes":{"leaf":true}}`

CODE-TEST-02 and CODE-TEST-03 pass. The missing `}` is a genuine converted-model
divergence (the extraction sees an unbalanced object and returns empty); it is
the class of failure the gate exists to catch. Investigate the granite decode /
EOS path before committing a golden.

### gemma-4-26B-A4B-it — CODE-TEST-02 (assignment)

The converted model emits an equivalent but differently-parenthesized
expression:

- expected: `m=l+((h-l)/2)`
- converted: `m=(l+(h-l)/2)`

CODE-TEST-01 and CODE-TEST-03 pass. Because CODESelfCheck compares exactly (no
associativity rewriting, unlike FSISelfCheck), the extra outer parentheses are a
failure. This is the intended "syntactic strictness" behaviour; if a future
decision relaxes the assignment kind to algebraic canonicalization, this case
would pass.

### Other families

No golden is committed, so their `test_<family>_code_selfcheck` targets SKIP
(SKIP_RETURN_CODE 4), never fail.
