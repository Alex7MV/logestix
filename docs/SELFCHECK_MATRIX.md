# SelfCheck matrix — model x parity gate, as measured

Snapshot of **2026-10-24/25** on the reference host (RTX 5090, CUDA 13.4,
`linux-release-znver4-aocc`), one test at a time (GPU exclusivity).
Commit at capture: `f38fba84`.

This is a **measurement**, not a policy. The machine-checked registry is the
source of truth for which target covers which model and stage:
`tests/test_model_suite_guard.cpp::Suites()` (model passport -> stage ->
ctest target). A row here can go stale; the registry cannot drift silently,
because it fails when a registered target is missing.

## What the four gates mean

| gate | oracle | cases |
|---|---|---|
| **SelfCheck** | built-in prompt ("Paris"), token + text + last-logits determinism across stress cycles | 1 (+ family extras, e.g. `*_eos_selfcheck`) |
| **FSI** | generated answer vs a golden captured from the *source* safetensors (`tools/fsi_capture.py`) | 3 prompts |
| **CODE** | same, compiler-syntactic strictness (`tools/code_capture.py`), exact match, no algebraic canonicalisation | 3 prompts |
| **DET** | same, plus R>=3 byte-identical `gen_tokens` runs (`tools/det_capture.py`) | 3 prompts |

`SKIP` (ctest exit code 4) is **not** a pass: it means the converted tree or the
golden is absent. A golden that was never captured looks exactly like a passing
one in a ctest summary, which is why the debt rows are called out below.

## 1. Converted tree in the vault **and** goldens present — real verdicts

| model (passport) | SelfCheck | FSI | CODE | DET |
|---|---|---|---|---|
| alice-foundation-80b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| glm-5.3-flash | PASS | 3/3 | 3/3 | 3/3 |
| **gemma-4-26b-a4b-it** | PASS | 3/3 | 3/3 | 3/3 |
| **gemma-4-31b-it** | PASS | **2/3** (FSI-01) | 3/3 | 3/3 |
| granite-4.2-30b | PASS | 3/3 | 3/3 | 3/3 |
| ornith-1.5-35b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| phi-4 | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.5-9b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.6-35b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.8-27b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.8-flash-next | PASS (+ eos) | 3/3 | 3/3 | 3/3 |
| xing4.0-29b-a4b | PASS | 3/3 | 3/3 | 3/3 |

- SelfCheck **12/12**, FSI **11/12**, CODE **12/12**, DET **12/12**.
- The one red is the dense Gemma-4 member, and it is understood rather than
  mysterious: FSI-01 is a template near-tie where the dense body needs ~8 bits
  on *every* linear (33.7 GiB resident, over this card). Every partial
  placement that fits was measured and moves *away* from the source —
  `docs/OPEN_TASKS.md` §R and
  `open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md` §5.
- gemma-4-26b went green in this session (second NVFP4 plane on attn + dense
  MLP, routed experts left 4-bit): FSI/CODE/DET were 1/3, 2/3, 3/3 before.

## 2. Converted tree present, **no golden** — declared debt (FSI/CODE/DET SKIP)

| model | SelfCheck | FSI | CODE | DET | why |
|---|---|---|---|---|---|
| mistral-small-4-119b | PASS | SKIP | SKIP | SKIP | source is the Mistral "consolidated" layout (`params.json`, no `config.json`), so Transformers cannot run it and no oracle can be captured — `docs/OPEN_TASKS.md` §T |
| nemotron-3.5-lightning-30b-a3b-nvfp4 | PASS | SKIP | SKIP | SKIP | goldens never captured for the pre-quantized ModelOpt sibling (same geometry as the bf16 passport, which also has none) |

Both are recorded in the golden-debt list (§K) rather than left to look green.

## 3. Golden present, **no converted tree** — SKIP (needs the vault, not a code fix)

`deepseek-v4.1-flash`, `qwen3.5-35b-a3b`. Their FSI/CODE/DET targets run only
when the converted tree exists; SelfCheck skips with them.

## 4. Neither — SKIP

`glm-5.2`, `deepseek-v4-flash`, `kimi-k2.7-code`,
`nemotron-3.5-lightning-30b-a3b` (bf16 sibling), `gigachat3.5-432b-a28b-reasoning`,
`mimo-v2.6-flash-rl`. Registered stages exist; converted trees and goldens do
not.

## Reproduce

```bash
# 1. the real verdicts (12 models with a tree + goldens), one at a time
ctest --test-dir build/linux-release-znver4-aocc --timeout 1500 -R \
  "alice.*selfcheck|gemma_(moe|dense).*selfcheck|glm_moe_flash.*selfcheck|\
granite_dense.*selfcheck|nemotron_moe_nvfp4.*selfcheck|ornith_moe.*selfcheck|\
phi_dense.*selfcheck|qwen_dense.*selfcheck|qwen_moe_36.*selfcheck|\
qwen_moe_exp.*selfcheck|xing_moe.*selfcheck"

# 2. the debt rows (SelfCheck only; the three parity stages SKIP by design)
ctest --test-dir build/linux-release-znver4-aocc -R "mistral.*selfcheck"

# 3. the absent-vault families (all SKIP; proves the SKIP is the vault, not code)
ctest --test-dir build/linux-release-znver4-aocc -R \
  "deepseek_moe(_v41)?_selfcheck|glm_moe_selfcheck|kimi_moe_selfcheck|\
mimo_moe_selfcheck|nemotron_moe_selfcheck|qwen_moe_selfcheck|gigachat_moe_selfcheck"

# 4. the harness' own self-tests (no model): formula extraction/canonicalisation
ctest --test-dir build/linux-release-znver4-aocc -R \
  "test_(fsi|code|det)_selfcheck"
```

The parity stages share one rule: the golden is captured from the source
safetensors and is **never reconciled** to make a conversion pass. When a gate
goes red the question is which instrument is missing, not how to rewrite the
expectation (`docs/OPEN_ENDEDNESS.md`).
