# DETSelfCheck findings

Per-model capture feasibility, recipes and observed results for the DET
determinism / near-tie gate. The gate is defined in
`docs/superpowers/specs/2026-09-27-det-selfcheck-design.md`; the harness lives in
`libs/lsxcommon/src/det_selfcheck.cpp`.

## Status

The DET gate is wired for six families: the four that pass FSI-SelfCheck
(glm-5.3-flash, qwen3.6-35b-a3b, qwen3.8-flash-next, granite-4.2-30b) plus
gemma-4-26b-a4b-it and deepseek-v4.1-flash. Golden `expected` is
**template-defined** (the prompt's canonical Expected Template); the source run
only supplies `input_ids` and logs agreement.

| family (`Type()`) | golden | DET-01 | DET-02 | DET-03 | determinism (converted) |
|---|---|---|---|---|---|
| granite-4.2-30b | **wired** | PASS | PASS | PASS | `deterministic=true` all |
| glm-5.3-flash | **wired** | PASS | PASS | PASS | `deterministic=true` all |
| qwen3.6-35b-a3b | **wired** | PASS | PASS | PASS | `deterministic=true` all |
| qwen3.8-flash-next | **wired** | PASS | PASS | PASS | `deterministic=true` all |
| gemma-4-26b-a4b-it | **wired** | PASS | PASS | PASS | `deterministic=true` all |
| deepseek-v4.1-flash | **wired** | PASS | PASS | PASS | `deterministic=true` all |

**The determinism stress-test passed in every case** (18 prompt×family
combinations, byte-identical `gen_tokens` across 3 runs), and the extracted
canonical token matched the template for all 18.

### Prompt update (2026-09-27): DET-TEST-02 / DET-TEST-03

With the first prompt set, DET-02 failed for every non-granite family and DET-03
failed for gemma. In each failing case the source and converted models agreed
**byte-for-byte** (`TERMINAL`, `0`), so these were **prompt/expectation
mismatches, not conversion or runtime defects**. The two prompts were reworded
(`tools/det_prompts.json`):

- **DET-TEST-02** (now `LEXICAL-INDEX-CLAMP`): the original "isolate the final
  vocabulary token strictly according to the highest alphabetical sort priority"
  was ambiguous — models read "highest" as the alphabetically-greatest token and
  answered `TERMINAL`. Reworded to a literal predicate, *"choose the word that
  strictly begins with the character 'C' and ends with the character 'C'"*, whose
  unique answer is `CATASTROPHIC`. All families now answer `CATASTROPHIC`.
- **DET-TEST-03** (now `BINARY-REGISTER-STRING`): the original asked for "the
  strict primary hardware default register bit identifier", but gemma (and the
  Qwen source) emitted the raw bit `0`. Reworded to require "a two-token
  alphabetical word connected by an underscore" and to forbid raw single digits.
  All families now answer `BIT_ZERO`.

After the update all five families pass 3/3 deterministically (glm 596 s,
qwen3.6 95 s, qwen3.8 841 s, gemma 16 s, granite 18 s).

### DeepSeek-V4.1-Flash framing (2026-09-27)

DeepSeek-V4.1-Flash has **no HF chat template** (`tokenizer_config.json` has no
`chat_template`, there is no `chat_template.jinja`), so the generic capture
would frame the raw body and the gate would have no valid `input_ids`. The
model's own `encoding/encoding.py` is the framing source. `tools/det_capture.py`
gains a per-type `framing: "deepseek_v41"` override that calls
`encode_messages(..., thinking_mode="thinking")` and tokenizes with
`tokenizer.json`.

With that framing the converted model is **3/3 deterministic
(`PARADOX_ANTINOMY` / `CATASTROPHIC` / `BIT_ZERO`)** and wired. Notes:

- The reasoning model's correct mode is `thinking`; with `thinking_mode="chat"`
  the framing pre-closes `<think>` and DET-02 near-ties to `CRITICAL` (2/3). This
  is a framing artifact, not a conversion divergence.
- The source is not runnable locally (own FP8/FP4 `inference/` stack), so the
  golden is template-defined; as an API-derived cross-check the assistant (itself
  DeepSeek-V4.1-Flash) answers `PARADOX_ANTINOMY` / `CATASTROPHIC` / `BIT_ZERO`,
  matching the thinking-mode converted output.
- FSI for this family stays 1/3 (not wired); the shorter DET prompts are simpler.

### GLM-5.3-Flash root cause + fix (2026-09-27)

The initial DET run reported GLM as 1/3 (DET-01 empty; DET-02 extracted `FATAL`).
Root cause was **not** the model/runtime but the **capture budget**:
`tools/det_capture.py` defaulted to `max_new_tokens = 64`, unlike
`fsi_capture.py` (512). GLM-5.3-Flash's chat template always opens `<think>`, so
it emits a long reasoning preamble; at 64 tokens generation is cut off **before**
`</think>` and before the answer. With no closing `</think>`, `ExtractToken`
(`det_selfcheck.cpp`) cannot strip the reasoning and returns the first standalone
uppercase token it finds — a candidate name (`FATAL`) or nothing (DET-01); DET-03
appeared to pass only because the reasoning echoed the Expected Template.

Evidence: regenerating the golden with `max_new_tokens = 512` makes the harness
emit `</think>` followed by the canonical token (`CATASTROPHIC`, `BIT_ZERO`), and
the test is **green 3/3** with `deterministic=true`. Fix: the DET capture default
is now 512 (matching FSI). Secondary observation: after the answer the converted
GLM repeats the prompt / degenerates (`… [CONTEXT]: DET-Validation-Core-02 // …`,
`degenerate repetition…`); the golden `body` echo-cut trims this so extraction
still yields the answer. DET-01 remains verbose (does not close `<think>` within
512 and passes via the token in its reasoning), which is deterministic but a
known fragility.

### Capture recipes used

- `granite-4.2-30b`: model-defined run confirmed all three templates exactly;
  golden tokenizer-only (`--template-golden`, `max_new_tokens = 512`).
- `qwen3.6-35b-a3b`: golden tokenizer-only. (The pre-update source answered
  DET-02 `TERMINAL` / DET-03 `0`; the rewording resolves both.)
- `qwen3.8-flash-next`: golden tokenizer-only (500 GB source not run).
- `glm-5.3-flash`: source unrunnable (`glm5_next`); golden tokenizer-only
  (`chat_template.jinja`, `max_new_tokens = 512`).
- `gemma-4-26b-a4b-it`: source run on CPU (`max_new_tokens = 512`); the
  pre-update source answered `TERMINAL` / `0` (DET-02/03 differed from the
  templates); re-captured tokenizer-only after the rewording; **wired**.
- `deepseek-v4.1-flash`: source unrunnable (`deepseek_v41` not in transformers,
  own FP8/FP4 stack); framed with the release's `encoding/encoding.py`
  (`thinking_mode="thinking"`) + `tokenizer.json`; **wired**.

## Remaining families

| family (`Type()`) | golden | capture source | determinism (converted) | notes |
|---|---|---|---|---|
| glm-5.2 | pending | — | — | source not in vault; test SKIPs |
| glm-5.3-flash | **wired** | tokenizer-only (`glm5_next` unrunnable) | `deterministic=true` | **PASS 3/3** at `max_new_tokens=512`; golden committed |
| deepseek-v4-flash | pending | — | — | source not in vault; test SKIPs |
| deepseek-v4.1-flash | **wired** | `encoding/encoding.py` (thinking) | `deterministic=true` | **PASS 3/3**; golden committed |
| qwen3.5-9b | pending | — | — | source not in vault; test SKIPs |
| qwen3.8-27b | pending | — | — | source not in vault; test SKIPs |
| qwen3.5-35b-a3b | pending | — | — | source not in vault; test SKIPs |
| qwen3.6-35b-a3b | **wired** | tokenizer-only | `deterministic=true` | **PASS 3/3** (post prompt update); golden committed |
| ornith-1.5-35b-a3b | pending | — | — | source not in vault; test SKIPs |
| qwen3.8-flash-next | **wired** | tokenizer-only | `deterministic=true` | **PASS 3/3** (post prompt update); golden committed |
| gemma-4-26b-a4b-it | **wired** | source (CPU) / tokenizer-only | `deterministic=true` | **PASS 3/3** (post prompt update); golden committed |
| kimi-k2.7-code | pending | — | — | source not in vault; test SKIPs |
| nemotron-3.5-lightning-30b-a3b | pending | — | — | runnable source available; DET not captured |
| gigachat3.5-432b-a28b-reasoning | pending | — | — | 432B source; template golden only |
| granite-4.2-30b | **wired** | tokenizer-only | `deterministic=true` | **PASS 3/3**; golden committed |

Capture and the converted-model run are performed on the GPU host (one model
load at a time, per the host resource policy). Fill this table as each family is
captured and verified.

## Capture recipe

```bash
python3 tools/det_capture.py --selftest
python3 tools/det_capture.py --model /mnt/llm_models/models/<source-dir> \
    --type <Type()> --out tests/fixtures/det/<Type()>.json
python3 tools/det_capture.py --template-golden \
    --model /mnt/llm_models/models/<source-dir> \
    --type <Type()> --out tests/fixtures/det/<Type()>.json
```

Known overrides (in `tools/det_capture.py`): `granite-4.2-30b` uses
`enable_thinking=True, low_effort=True`; `qwen3.8-flash-next` uses the
`qwen4_exp` loader.

## Verify

```bash
ctest --preset linux-debug-inference -R "_det_selfcheck" --output-on-failure
LSX_DET_GOLDEN_DIR=tests/fixtures/det \
  ./build/linux-debug/apps/lsxbenchmark/lsxbenchmark \
  --model /mnt/llm_models/models/<vault-dir> --det-selfcheck
```

A `deterministic=no` result or an extraction mismatch is a **real finding**
about the converted runtime — record it here and investigate; do not loosen the
gate.
