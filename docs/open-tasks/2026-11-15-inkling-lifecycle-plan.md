# Inkling-2-Small — closing the lifecycle (2026-11-15)

Parity is green (`logestix_hash 323d30463a3d2b72`, ppl 30.4, " Paris"). What is
left is stage 8's remaining gates, stage 4 (throughput) and the registry.

| # | step | why | state |
|---|---|---|---|
| 1 | observe the raw FSI/CODE/DET answers from the converted model | the source is unrunnable (594B, own FP8/FP4 stack, 500 GB VRAM), so the goldens are template-derived; the observation fixes the token budget and proves the gate is not vacuous | todo |
| 2 | capture `tests/fixtures/{fsi,code,det}/inkling-2-small.json` (`--template-only` / `--template-golden`) | a golden per gate, marked template-derived in `source` | todo |
| 3 | `models/inkling-2-small.json` | the ONLY hand-written registry: it generates `model_manifest.h`, `model_targets.cmake`, `model_registry.inc` and the suite-guard row | todo |
| 4 | three `add_executable(... test_parity_gate.cpp)` targets + drop the 6 `kPendingSuiteTargets` / `_NON_MODEL_TESTS` entries | build reality is per target; the pending declarations must not outlive the row | todo |
| 5 | rename the format tag `logestix-inkling-2-small-nvfp4` -> `logestix-inkling-2-small` and re-convert | the artifact holds two expert codecs plus BF16 dense planes; "nvfp4" is now wrong | todo |
| 6 | perf row at the canonical 2048-token protocol + the VRAM profile | stage 4 | todo |
| 7 | `docs/MODELS.md` + the stage-9 report | generated catalogue + publishable artifact | todo |

Evidence for every step goes into `docs/open-tasks/2026-11-14-inkling-2-small-ingest.md`.
