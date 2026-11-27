# Granite-Switch 4.1 8B — ingestion plan and execution record

Design and Stage-0 triage: `../specs/2026-11-07-granite-switch-4.1-8b-design.md`.
This file is the executable trail: the order the work was done in, the command at
each step, and what it printed. A later session should be able to reproduce the
ingestion from here without re-deriving the decisions.

## Why this model

`granite-switch-4.1-8b` is the first family member added **after** its sibling
under the manifest registry, so it doubles as a test of the ingestion protocol
itself: everything that the protocol assumes about "one model, one row" gets
exercised against a model that shares a converter, a kernel layer, a tokenizer
and a chat template with an already-ingested one. The defects this surfaced are
recorded in `docs/OPEN_TASKS.md` §G2 (a sibling "covered" by a test that
verifies its parent) and §G1 (the GPU lock).

## Stage 0 — triage

Recorded in the design doc. Summary: sibling variant, **different geometry**
(40 layers, FFN 12800, theta 1e7, residual 0.22), everything else identical
to the 30B including md5-identical tokenizer and chat template.

## Stage 1 — conversion

The family converter was already config-driven, so no conversion logic changed.
Two things were wrong for a second member, both found by TDD:

1. **The index identity was a literal.** `WriteGraniteSwitchModelIndex` wrote
   `"model": "Granite-Switch-4.1-30B"` regardless of the checkpoint. Red test
   first (`test_granite_switch_convert.cpp` → *"output identity is the checkpoint,
   not a literal"*), then `CheckpointName(input_path)` and a `model_name` field
   on `GraniteSwitchTopology`, with a second case for a trailing `/` (a path a
   shell script produces, and `fs::path("…/x/").filename()` is empty).

2. **The identity is now validated on load**, because a tree whose index claims a
   different model would otherwise be caught only by a frozen-geometry mismatch
   on a numeric field — a confusing message for what is a naming error.

Run:

```bash
build/linux-debug/apps/lsxforge/lsxforge --granite_switch \
  --input  /mnt/fast_storage/lsx_model/ibm-granite/granite-switch-4.1-8b \
  --output /mnt/fast_storage/lsx_model/ibm-granite/granite-switch-4.1-8b
```

## Stage 2 — NVFP4

No new work: the base projections are quantized by the shared path, the aLoRA
planes stay FP16, the tied head is the embedding. A residual plane is added only
if a gate goes red *and* the decision procedure in the ingestion skill says the
cause is accumulated 4-bit body error — not preemptively.

## Stage 3 — registration, the actual refactor

`GraniteSwitch41_30BModel` was a single `final` class with the passport baked
into `ValidateGeometry`. Two classes need that body, so:

- `GraniteSwitchDenseModel` (not `final`) holds the whole implementation and
  takes a `granite_switch::GraniteSwitchFrozen` — a plain struct of the
  expectations — in its constructor.
- `GraniteSwitch41_30BModel` and `GraniteSwitch41_8BModel` are thin `final`
  subclasses, each with its own `static const std::string kType` (the manifest
  generator resolves `kType` by grepping `<class>::kType = "…"`, so a template
  would not be resolvable) and each passing its own passport.
- `granite_switch::GraniteSwitchGeometryMismatch(frozen, top)` is the ONE
  comparison, in the load TU, host-testable without a GPU. It reports **every**
  differing field on its own line plus the identity, because a tree that is wrong
  twice must not need two runs to discover.
- The 8B passport is `granite_dense_switch_8b_constants.h`
  (`lsxcommon::granite_switch_8b::frozen`), a leaf that includes the family
  header for the shared struct types only.

Red-first: `test_granite_switch_prepare.cpp` gained
*"granite-switch 8B passport accepts only the 8B tree"* — it asserts the fixture
topology passes, that a 30B topology is rejected with `num_layers`, `rope_theta`
and `residual_multiplier` all named, and that the legacy `Granite-Switch-4.1-8B`
spelling folds to the same identity (`detect_model_type` folds case the same way).

Name-scheme: `granite_dense_switch_8b_constants.h` needed a second variant slot
(`switch` + `8b`), so `SchemeRegex()` now allows two, drawn from the same
vocabulary. Widening the grammar did not widen the accepted tokens.

## Stage 4 / 4b — tokenizer and tools

Nothing to implement: `tokenizer.json`, `merges.txt`, `vocab.json` and
`chat_template.jinja` are **md5-identical** to the 30B (measured, not assumed).
The stage-4b verdict is therefore the parent's and is declared in
`test_model_suite_guard.cpp`: the family framing test covers both members and the
runtime parse is the generic bare-JSON fallback (`tool_roundtrip: false`, an
explicit declaration, not a silent skip).

## Stage 5 / 6 — kernels, sandbox

Reused unchanged. The inference TU contains no `frozen::` reference at all — all
geometry arrives through `pw_.top` — so nothing in it knew about the 30B.

## Stage 8 — gates and goldens

Goldens are captured **from the 8B source**, never copied from the 30B, because
the answers differ (they are source parity, not template text):

```bash
.venv-gs/bin/python tools/fsi_capture.py  --model …/granite-switch-4.1-8b --type granite-switch-4.1-8b --out tests/fixtures/fsi/granite-switch-4.1-8b.json
.venv-gs/bin/python tools/code_capture.py --model … --type granite-switch-4.1-8b --out tests/fixtures/code/granite-switch-4.1-8b.json
.venv-gs/bin/python tools/det_capture.py  --model … --type granite-switch-4.1-8b --out tests/fixtures/det/granite-switch-4.1-8b.json
```

`.venv-gs` (documented in `tools/granite_switch_oracle.py`) now exists and is
gitignored; the oracle loads CPU-BF16 because the vendor class is not in
transformers and a mixed GPU/CPU placement produced NaNs in the aLoRA path.

A capture that disagrees with the prompt template prints a WARNING and stores the
model's **actual** answer — that is the intended behaviour (the golden records the
source), and it fired: the 8B answers FSI-02 with numbers where the 30B answers
symbolically, and DET-02 `CRITICAL` where the template says `CATASTROPHIC`. Both
are properties of the checkpoint, so they are recorded as measured, not "fixed".

## Stage 9 — report

`docs/model-reports/2026-11-07-Granite-Switch-4.1-8B.md`, written after the gates
so every number is measured.
