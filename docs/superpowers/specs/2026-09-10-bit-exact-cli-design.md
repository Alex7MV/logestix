# Design: CLI bit-exact-only numeric policy (lsxfabric)

Date: 2026-09-10
Status: approved

## Problem

The inference engine launch parameters expose speed/performance overrides
(`--fast-math`, `--no-fast-math`, `--calibrate`). Policy change: the engine
always runs at maximum speed for the specific hardware (per-GPU auto perf
policy, already the default via `perf::ResolvePerfKnobs(gpu)` in
`model_engine.cpp`). Exactly one numeric-mode flag remains: bit-exactness for
100% answer verification.

## Non-goals

- Internal API is NOT collapsed: `FastMathMode` tri-state, `LsxFastMathScope`
  per-model policy, `ModelInitConfig::fast_math`/`fast_math_mode`, and
  `gpu_perf` knobs stay as-is. They are guarded by
  `tests/test_model_independence_guard.cpp` (per-call `LsxFastMathScope`
  contract) and exercised by the test suite (`LSX_TEST_FAST_MATH`).
- `gpu_perf::ResolvePerfKnobs` keeps its `cli_mode` parameter (documented
  precedence hook; exercised by `tests/test_gpu_perf.cpp`).
- Auto-calibration of unknown GPUs on first model Load is untouched. Only the
  developer A/B authoring CLI (`RunCliCalibrate`) is removed; the committed
  `BaseKnobs` table in `gpu_perf.cpp` remains the base source.

## CLI surface (apps/lsxfabric)

| Flag | Status |
|---|---|
| `--localmodel <path>` | kept |
| `--validate` | kept |
| `--benchmark` | kept (measurement tool; always auto-policy now) |
| `--testprompt` | kept |
| `--selfcheck` | kept |
| `--prompt <text>` | kept |
| `--max-gen <n>` | kept |
| `--bit-exact` | NEW — the single numeric-mode flag; forces
  `FastMathMode::kOff` (deterministic scalar path) for 100% verification |
| `--fast-math` | REMOVED (speed override) |
| `--no-fast-math` | REMOVED (superseded by `--bit-exact`) |
| `--calibrate` | REMOVED (perf knob authoring tool) |

Unknown arguments stay silently ignored (existing behavior); in particular
the removed flags are no longer recognized.

## Changes

| File | Change |
|---|---|
| `apps/lsxfabric/src/cli.h` (new) + `src/cli.cpp` (new) | `lsxfabric::FabricCliArgs` + `ParseFabricCliArgs(argc, argv)` — pure, testable parser. Fields: `model_path`, `validate`, `benchmark`, `testprompt`, `selfcheck`, `bit_exact`, `prompt`, `max_gen` |
| `apps/lsxfabric/src/main.cpp` | Uses the parser; calibrate branch deleted; generation/selfcheck map `bit_exact` → `GenerationCliArgs::fast_math_mode = FastMathMode::kOff`; benchmark calls `RunBenchmark(model_path, 2048)` |
| `libs/lsxcommon/include/lsxcommon/model_runtime.h` | `GenerationCliArgs`: drop legacy `bool fast_math` (its only producer was `--fast-math`; lsxforge never sets it); keep `fast_math_mode = kAuto` |
| `libs/lsxcommon/src/model_runtime.cpp` | `CreateCliModel`: drop the kOn legacy bool branch (cfg.fast_math stays false, mode assigned directly); `RunCliGeneration`/`RunCliSelfCheck` pass `args.fast_math_mode`; delete `RunCliCalibrate` + declaration |
| `libs/lsxcommon/include/lsxcommon/benchmark_gpu.h` + `src/benchmark_gpu.cpp` | `RunBenchmark(model_path, num_tokens = 2048)` — drop `bool fast_math`; benchmark always follows the auto perf policy |
| `AGENTS.md` | Update lsxfabric flag list (`--bit-exact` replaces `--fast-math`/`--no-fast-math`; `--calibrate` removed), update GPU Perf Auto-Tuning section (base-table authoring via `--calibrate` no longer exists) |

Untouched: all model files, `fast_math_mode.h`, `lsx_inference_nvfp4.*`,
`gpu_perf.*`, `test_gpu_perf.cpp`, `test_model_runtime.cpp` (already uses the
2-arg `RunBenchmark` form), lsxforge (defaults suffice).

## Semantics after the change

- Default: engine = maximum speed for the hardware. Global routing knobs come
  from the auto policy at `InitializeEngine`; per-call numerics stay kAuto
  (models don't touch global state).
- `--bit-exact`: model Infer runs under `LsxFastMathScope(kOff)` —
  deterministic scalar path, reproducible answers for verification.

## Testing (TDD)

New `apps/lsxfabric/tests/test_fabric_cli.cpp` (Catch2, label `fast`,
TIMEOUT 120 s, host-only) — written BEFORE the implementation:

1. Default parse: `bit_exact == false`, all mode flags false, default
   model path / prompt / max_gen.
2. `--bit-exact` → `bit_exact == true`.
3. Removed flags `--fast-math`, `--no-fast-math`, `--calibrate` are not
   recognized (no crash, `bit_exact` stays false, no calibrate field).
4. Kept flags parse: `--localmodel`, `--validate`, `--benchmark`,
   `--testprompt`, `--selfcheck`, `--prompt`, `--max-gen`.

Then implement parser + main.cpp rewiring + plumbing slimming until green;
existing suites (`test_model_runtime`, `test_gpu_perf`) must stay green.
