---
name: fsi-selfcheck
description: Use when capturing, adding, running or debugging FSISelfCheck in logestix — the financial-services prompt-set parity gate, a separate Model Ingestion Protocol stage (not SelfCheck). Covers tools/fsi_capture.py, the tests/fixtures/fsi/<model>.json golden schema, extraction/canonicalization, the per-family test + CMake + suite-guard wiring, lsxbenchmark --fsi-selfcheck and LSX_FSI_GOLDEN_DIR. Triggers - FSI, fsi_selfcheck, --fsi-selfcheck, FSI golden, parity gate.
---

# FSISelfCheck (financial-services prompt-set parity gate)


> **Registration (2026-11-06):** the real-model target is still named
> `test_<family>_{fsi,code,det}_selfcheck` and still runs one per (model, stage),
> but its SOURCE is now the shared `tests/test_parity_gate.cpp`, compiled with
> `LSX_GATE_MODEL` / `LSX_GATE_KIND`. Do **not** create a per-family gate file:
> 105 of them were deleted because three shipped a SKIP naming a neighbour model,
> and the model row (vault dir, SKIP text, declared debt) now comes from
> `models/<model_type>.json`. See skill `ingesting-a-model` § "The manifest is the
> only registry you write by hand".

## Overview

FSISelfCheck is a **model-defined parity gate** and a first-class stage of the
Model Ingestion Protocol (step 8). It captures the **original safetensors**
model's own answer to a fixed 3-prompt financial set, freezes it as a committed
golden, and requires the **converted** NVFP4 model to reproduce the same
canonicalized answer *text* at `FastMathMode::kOff`.

**Core principle:** FSI is a separate functionality from `SelfCheck`, not a
variant of it. Do not touch `IModel::SelfCheck`, `runtime::RunParisSelfCheck`,
`selfcheck_passed` or `LSX_SELFCHECK_EXPECT_HASH`. It depends only on the neutral
`IModel` surface, `ArrowTokenizerDb` and `runtime::DecodeGeneratedText`.

The single source of truth for the design is
`docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md` (§3.5 decoupling
invariants, §4 extraction, §5 golden schema, §7 capture). Per-model capture
recipes and observed outcomes are in `docs/FSI_SELFCHECK_FINDINGS.md`.

## Relationship to the ingestion protocol

**REQUIRED SUB-SKILL:** `ingesting-a-model` (step 8, SelfCheck harness). FSI
extends step 8; the FSI test is a **new** target, never the existing
`test_<family>_selfcheck`.

Adding/adopting a model means extending `ingesting-a-model`'s step 8 with the
FSI row below, in the same commit as the model:

| Guard (in `tests/`) | What to extend | Rule |
|---|---|---|
| `test_model_suite_guard.cpp` | `Suites()`: add an `"fsi"` stage target per model | FSI lifecycle stage exists; `SKIP_RETURN_CODE 4` on `fsi` targets |
| `test_ctest_labels_guard.cpp` | label the new test `inference` | label in the same commit |
| `test_code_naming_guard.cpp` | — | `fsi_selfcheck.*` has no family stem (not scanned); test files are not scanned |

Do **not** add a new ctest label or lifecycle stage — `fsi` is a stage inside the
existing `inference` label (see `test-lifecycle`).

Full wiring is three files in the same commit: the golden, the test `.cpp`, and
the `tests/CMakeLists.txt` target.

## Decoupling invariants (FSI vs SelfCheck)

These are invariants, not guidelines. Violating one is a design regression.

| Dimension | SelfCheck | FSISelfCheck |
|---|---|---|
| Entry point | `IModel::SelfCheck` / `RunParisSelfCheck` | `fsi::RunFsiSelfCheck` |
| Input | hardcoded `"The capital of France is"` | golden `input_ids` from the source model |
| Reference | in-test literal `"Paris"` | committed golden JSON |
| Criterion | top-1 / short-prefix text (+ optional logits hash) | normalized answer `==` golden `expected` |
| Result type | `gpu::InferenceResult.selfcheck_passed` | `fsi::FsiResult.all_passed` |
| Numeric mode | `kAuto` | `kOff` |
| Env gates | `LSX_SELFCHECK_EXPECT_HASH` | `LSX_FSI_GOLDEN_DIR` only |
| CLI | `--selfcheck` | `--fsi-selfcheck` |
| Host unit test | `tests/test_model_runtime.cpp` | `tests/test_fsi_selfcheck.cpp` |
| Real-model test | `tests/test_<family>_selfcheck.cpp` | `tests/test_<family>_fsi_selfcheck.cpp` |

## The FSI prompt set

`tools/fsi_prompts.json` (`format: logestix-fsi-prompts`) is the single source of
truth. The C++ side never embeds the bodies — the golden carries the captured
`input_ids`.

| id | kind | anchor | canonical answer |
|---|---|---|---|
| FSI-TEST-01 | arithmetic | `PV` | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` |
| FSI-TEST-02 | arithmetic | `Balance_Alpha` | `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` |
| FSI-TEST-03 | comparison | `Drawdown` | `Drawdown>Threshold` |

## Workflow

1. **Capture** the golden from the source safetensors model.
2. **Commit** it under `tests/fixtures/fsi/<model_type>.json`.
3. **Add** `tests/test_<family>_fsi_selfcheck.cpp` + its `tests/CMakeLists.txt`
   target (label `inference`, `SKIP_RETURN_CODE 4`) and the suite-guard `fsi` row.
4. **Run** the target on the converted model; a failure is a real finding to
   investigate, not to mask.

### 1. Capture (`tools/fsi_capture.py`)

```bash
python tools/fsi_capture.py --selftest                 # no model; validates the set
python tools/fsi_capture.py --model <safetensors-dir|hf-id> \
    --type <model_type> --out tests/fixtures/fsi/<model_type>.json

# one prompt per run, merged into what an earlier run already wrote:
python tools/fsi_capture.py --model <dir> --type <model_type> \
    --out tests/fixtures/fsi/<model_type>.json \
    --only FSI-TEST-01 --resume [--max-new-tokens 512]
```

- **The capture is incremental and per-prompt-resumable.** Each completed case
  is written to `--out` immediately (atomically), the golden always carries the
  cases in prompt-set order, and `--only <ids>` + `--resume` capture a single
  prompt at a time and merge it into the file. A killed run therefore keeps
  every case it finished — the R21 Muse run that lost 1.5 h of CPU to a partial
  golden is the reason. A failed case is skipped (not fatal), its full raw is
  dumped to `<out>.<ID>.raw.txt`, and the process exits 3 with the ids to retry.

- Greedy (`do_sample=False`), `max_new_tokens` default 128, frames with
  `tokenizer.apply_chat_template(...)` when the tokenizer defines one.
- **If `tokenizer.chat_template` is unset, it loads `chat_template.jinja`** — this
  is required for granite/gemma/nemotron/GLM/Qwen; without it the raw body is
  encoded and the answer is malformed.
- Per-model quirks go in the `OVERRIDES` table keyed by `--type` (never implicit).
  Known ones: `granite-4.2-30b` → `{"enable_thinking": True, "low_effort": True}`;
  `qwen3.8-flash-next` → `{"loader": "qwen4_exp"}` (config allowlist + PLE shards).
  A custom loader may contribute its own capture facts through a
  `capture_opts()` function returning `template_kwargs` / `answer_channel` /
  `reasoning_channel` / `ignore_eos`; `tools/fsi_capture.py` merges it right
  after the loader is imported. `muse-glimmer-30b` is the worked example
  (`tools/muse_golden_oracle.py`): `reasoning_strength: low` plus the ATEM
  channel headers, see "ATEM channel framing" below.
- The golden `expected` is **model-defined**: the source model's own extracted
  answer, not the prompt template. A template difference is a warning, not a
  failure. Structural sanity (non-empty, balanced parens) must pass or capture
  aborts.
- `input_ids` are captured **after** framing and fed verbatim by the verifier, so
  no HF↔logestix chat-template parity is assumed.

**When the source cannot run locally** (own FP8/FP4 stack, unknown arch, too big),
take a **template/API-derived golden** and mark it in the `source` field
(e.g. GLM-5.3-Flash). This is a deliberate deviation from the model-defined rule,
forced by the unrunnable source; it still guards the converted path. See
`docs/FSI_SELFCHECK_FINDINGS.md`.

### 2. Golden schema (`tests/fixtures/fsi/<model_type>.json`)

```json
{ "format": "logestix-fsi-golden", "version": 1, "model_type": "<Type()>",
  "source": "...", "runtime": "transformers X / torch Y",
  "prompts": [ { "id": "FSI-TEST-01", "kind": "arithmetic", "anchor": "PV",
    "input_ids": [151644, 872, ...], "max_new_tokens": 128,
    "ignore_eos": true,
    "answer_channel": "assistant to=user",
    "reasoning_channel": "assistant to=self",
    "expected": "...", "raw": "..." } ] }
```

- Filename is `<model_type>.json` verbatim (dots allowed), matching `IModel::Type()`.
- `kind` ∈ {`arithmetic`,`comparison`}; `anchor`, `input_ids`, `expected`
  non-empty; a malformed file is a parse error, never a silent skip.
- `ignore_eos` (optional) forces the full token budget — set it for reasoning
  models that never emit EOS so the answer line is always produced.
- `answer_channel` / `reasoning_channel` (optional, both-or-none in practice) turn
  on **ATEM channel selection** before extraction: the harness reads the FIRST
  COMPLETE answer channel instead of the raw stream (`fsi::atem::SelectAnswerChannel`
  in `libs/lsxcommon/include/lsxcommon/atem_channel.h`, shared with the CODE and
  DET harnesses). The capture writes the same rule's result as `expected`
  (`tools/muse_golden_oracle.py::select_answer_channel`, pinned against the C++
  side by unit tests). Absent → historical behaviour, every existing golden is
  untouched.
- `raw` is diagnostic only (it is the un-truncated decode, so a looping model
  makes it large; nothing compares it).

### 2a. ATEM channel framing (reasoning checkpoints)

Some checkpoints (Muse-Glimmer) are channel models: `<|start|>assistant to=user<|message|>`
carries the answer, `assistant to=self` carries the reasoning, and the end markers
(`<|eom|>`, `<|eot|>`) are **special tokens**, so `skip_special_tokens=True` drops them
and the channel *headers* survive as literal text. Such a model also **re-emits** its
answer after looping into reasoning, so:

| stream shape | what the answer is |
|---|---|
| `…to=user<answerA> … to=self<reasoning> … to=user<answerA>` | `answerA` (first **complete** channel: another header follows it) |
| `…to=self<reasoning> … to=user<answerA` (budget ran out) | `answerA` (the lone channel is still the answer) |
| no channel header at all | the raw decode, unchanged |

The last answer channel is **not** trustworthy: the token budget can cut it
mid-answer (measured on Muse: `PARADOX_ANTINOM` for `PARADOX_ANTINOMY`). The rule is
implemented once per language and both are pinned by tests:
`fsi::atem::SelectAnswerChannel` (C++, shared by FSI/CODE/DET) and
`muse_golden_oracle.select_answer_channel` (Python capture). At the capture side the
channel text replaces the raw stream before `extract()`, so `expected` is exactly what
the verifier extracts from the same decode.

**The framing is part of the prompt.** Muse's chat template renders
`Reasoning strength: <reasoning_strength>.` into the system block, defaulting to
`high`, and the strength decides what the answer channel says: at `high` the
extraction read the prompt's numeric echo (`Balance_Alpha=100000000`), at `low`
all three answer channels are the canonical formula. Its capture therefore
passes `reasoning_strength: low` via `template_kwargs`; the kwarg is part of the
captured `input_ids`, so the converted model is replayed at the same strength and
the gate stays a parity check. Note `low` does **not** stop the channel loop (the
captured raws still hold 12/7/37 answer channels in 512 tokens) — the
first-complete rule above is what makes the repetition harmless, and a longer
budget is not a substitute for it (R21's 1024-token retry burned 1.5 h).

### 3. Test + CMake + guard

- Copy the family's vault-path-helper **pattern** into a self-contained
  `tests/test_<family>_fsi_selfcheck.cpp`; call only the `fsi::` harness — its header and implementation are
  `libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h` and
  `libs/lsxcommon/src/fsi_selfcheck.cpp` (host-only, model-neutral: the
  test owns the model and passes a `ModelRequest`; the harness never loads
  weights itself).
- Create the model with `cfg.fast_math_mode = kOff`.
- `SKIP` when the converted model **or** the golden is absent.
- CMake: `target_compile_definitions(... LSX_FSI_GOLDEN_DIR="${LSX_FSI_GOLDEN_DIR_FWD}")`,
  `add_test`, `set_tests_properties(... SKIP_RETURN_CODE 4)` and add the name to
  the `LABELS inference` list. Follow the existing FSI block in
  `tests/CMakeLists.txt`.
- `test_model_suite_guard.cpp::Suites()`: add the `"fsi"` stage entry.

### 4. Run

```bash
lsxbenchmark --model <vault-model-root> --fsi-selfcheck   # model type from the dir; forces kOff
ctest --preset linux-debug-inference -R "<family>_fsi_selfcheck"
LSX_FSI_GOLDEN_DIR=<dir> <test-binary>                    # ad-hoc golden
```

`lsxbenchmark` takes `--model <root>` (the vault model directory, not a
type); the type comes from the converted `model.index.json`.

## Extraction / canonicalization contract

`ExtractFsiFormula(kind, anchor, raw)` is pure host code
(`libs/lsxcommon/src/fsi_selfcheck.cpp`):

- Strip everything before the last `</think>` (reasoning CoT can hold competing
  formulas); unfence ``` blocks.
- Pick the **first** line where the anchor is directly followed (spaces allowed)
  by `=` (arithmetic) or a comparison operator; fall back to the first line
  containing the anchor.
- Arithmetic RHS is parsed and re-printed with minimal parentheses and
  equal-precedence chains left-associated, so association-only variants compare
  equal: `CF*((A)/(B))` == `CF*(A)/(B)`, `a*(b/c)` == `(a*b)/c`, `a-(b-c)` ==
  `a-b+c`. Each rewrite is an exact algebraic identity; commutativity is **not**
  applied. The `expected` is canonicalized by the same function.
- Comparison answers are whitespace-normalized and bounded to
  `anchor operator identifier`.

**Fragility to know:** the "first anchor+`=` line" rule can latch onto a reasoning
line that echoes the prompt (e.g. `Balance_Alpha = 100000000`), so a numeric
change that shifts greedy decoding can flip a case even when the model reasons
correctly. If a case fails this way, inspect `raw` before assuming a kernel bug.

## Common mistakes

- Extending/changing `SelfCheck` or `RunParisSelfCheck` to "share" FSI work
  (violates every decoupling invariant).
- Forgetting `SKIP_RETURN_CODE 4` (a missing weight/golden then fails instead of
  skipping) or the `inference` label; forgetting the suite-guard `fsi` row.
- Reusing `test_<family>_selfcheck.cpp` or its target for FSI.
- Encoding the prompt with the raw body because `chat_template.jinja` was not
  loaded (granite/gemma/nemotron/GLM/Qwen).
- Expecting the prompt template to equal the golden — the golden is the source
  model's own answer.
- Running the verifier at `kAuto` or comparing logits/timings; FSI compares
  answer text at `kOff`.
- Treating a real converted-model divergence as a gate artifact. FSI failures
  found the Nemotron quantization divergence and the GLM-5.3-Flash DSA
  self-only-attention bug — investigate the runtime/model, do not loosen the gate.

## Verification

```bash
ctest --preset linux-debug -R test_fsi_selfcheck                 # host unit tests (fast)
ctest --preset linux-debug-inference -R "_fsi_selfcheck"         # all real-model FSI
ctest --preset linux-debug -R "test_model_suite_guard|test_ctest_labels_guard"
```

Green means: host unit tests pass, the family target PASSes on the converted
model, and both guards are green. Findings (capture feasibility, per-model
recipes, resolved divergences) are recorded in
`docs/FSI_SELFCHECK_FINDINGS.md`.
