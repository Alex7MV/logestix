# Design: lsxbenchmark — centralized speed + bit-exactness app

Date: 2026-09-10
Status: approved

## Problem

Token-generation speed measurement is scattered across apps
(`lsxfabric --benchmark`, t/s prints in generation paths) and controlled by a
pile of flags (`--benchmark`, `--bit-exact`, `--selfcheck`). Policy change:
`lsxbenchmark` becomes the single owner of (a) generation-speed measurement and
(b) bit-exactness verification. Every other app builds for maximum inference
speed by default; the now-unneeded measurement flags are removed from them.

## Non-goals

- `FastMathMode` tri-state, `LsxFastMathScope` per-model policy,
  `ModelInitConfig::fast_math_mode`, and `gpu_perf` knobs stay as-is
  (guarded by `tests/test_model_independence_guard.cpp`).
- perf-labeled CTest harnesses (`test_blackwell_perf_*`) are untouched.
- `lsxforge --run` / `lsxfabric --testprompt` generation stays; the
  informational t/s line in `PrintGenerationResult` is not a measurement tool
  and stays.
- `lsxfabric --validate` (NVFP4/E4M3 validator) is untouched.

## New app: apps/lsxbenchmark

```
lsxbenchmark [--model <path>] [--bench] [--selfcheck] [--bit-exact]
             [--prompt <text>] [--max-gen <n>]
```

| Flag | Semantics |
|---|---|
| *(no mode flag)* | defaults to `--bench` |
| `--model <path>` | converted-model directory; default vault `DeepSeek-V4-Flash` via `lsx_model_models_dir()` (same convention as all apps) |
| `--bench` | speed measurement: real generation through `IModel::Infer` at maximum speed (`FastMathMode::kAuto`); reports tokens, t/s, ttft_ms, total_ms, resident MB |
| `--selfcheck` | Paris self-check via the existing `RunCliSelfCheck` harness |
| `--bit-exact` | numeric mode `FastMathMode::kOff` (deterministic scalar path); applies to the active mode (selfcheck -> kOff; bench -> kOff measurement) |
| `--prompt <text>` | bench prompt; default `"The capital of France is"` |
| `--max-gen <n>` | bench generation limit; default `2048` (matches the old `--benchmark`) |

Exit codes: `0` success, `1` failure (infer/selfcheck failed). Unknown flags
are silently ignored (lsxfabric parser pattern).

Speed measurement uses `IModel::Infer` (real prompt, `InferenceResult`
timings) — uniform across every model family, unlike the removed per-model
`IModel::Benchmark` (GLM-5.3-Flash never implemented it).

## lib change (libs/lsxcommon)

`runtime::RunCliBenchmark(const GenerationCliArgs &)` in
`model_runtime.{h,cpp}` next to `RunCliGeneration`/`RunCliSelfCheck`:
`CreateCliModel -> BuildPromptInputIds -> Infer -> print -> Unload`; reuses
`GenerationCliArgs::fast_math_mode`; prints a `=== lsxbenchmark: bench ===`
block (tokens, t/s, ttft, total, resident). Selfcheck mode reuses
`RunCliSelfCheck` unchanged. The app is a thin parser + dispatcher (lsxfabric
pattern).

## Removal of the internal benchmark API (no app consumers left)

| File | Change |
|---|---|
| `include/lsxcommon/benchmark_gpu.h`, `src/benchmark_gpu.cpp`, `src/benchmark_gpu.cu` | delete files (`BenchmarkResult`, `RunBenchmark` shim, legacy comment TU) |
| `include/lsxcommon/model.h` | delete `virtual Benchmark()` + `benchmark_gpu.h` include |
| `include/lsxcommon/model_base.h`, `src/model_base.cpp` | delete `RequireLoaded(BenchmarkResult &)` (Benchmark-only guard; Infer uses `ValidateInferRequest`) |
| `include/lsxcommon/model_runtime.h`, `src/model_runtime.cpp` | delete `FillBenchmarkResult` |
| `glm_moe_model`, `qwen_moe_model` (QwenMoeModelBase), `qwen_moe_exp_model`, `kimi_moe_model`, `deepseek_moe_model`, `glm_moe_flash_model`, `gemma_base_model`, `qwen_base_model` | delete Benchmark overrides (`.h` declaration + `.cpp` body) |
| `libs/lsxcommon/CMakeLists.txt` | drop `benchmark_gpu.cpp` / `benchmark_gpu.cu` from targets |
| `tests/test_model_runtime.cpp` | delete the 5 Benchmark-API test cases (`IModel::Benchmark default`, `RunBenchmark shim`, `FillBenchmarkResult computes t/s`, `RequireLoaded: canonical Benchmark guard`, `Concrete unloaded models report 'model not loaded' from Benchmark`) |

## apps/lsxfabric: maximum speed by default

| File | Change |
|---|---|
| `src/cli.h` | drop fields `benchmark`, `selfcheck`, `bit_exact`; keep `model_path`, `validate`, `testprompt`, `prompt`, `max_gen` |
| `src/cli.cpp` | stop parsing `--benchmark` / `--selfcheck` / `--bit-exact` (unknown tokens silently ignored, same as `--fast-math` before) |
| `src/main.cpp` | delete benchmark/selfcheck branches and the `fm` variable; `--testprompt` always runs `FastMathMode::kAuto` |
| `tests/test_fabric_cli.cpp` | rewritten (TDD-first): removed flags unrecognized; kept flags parse; defaults |

## New CMake wiring

- `apps/CMakeLists.txt`: `add_subdirectory(lsxbenchmark)`
- `apps/lsxbenchmark/CMakeLists.txt`: executable (`src/main.cpp`, `src/cli.cpp`),
  `lsxcommon` PRIVATE link, WIN32 arrow DLL POST_BUILD copies, `install(RUNTIME)`,
  `tests/` subdirectory when `BUILD_TESTING`
- `apps/lsxbenchmark/tests/CMakeLists.txt`: `test_lsxbenchmark_cli`, label
  `fast`, `TIMEOUT 120` (guard discipline), host-only

## Semantics after the change

- Every app except `lsxbenchmark` runs inference at maximum speed (`kAuto`
  auto perf policy); no numeric-mode flags remain outside `lsxbenchmark`.
- `lsxbenchmark --bench` = the single speed-measurement entry point.
- `lsxbenchmark --selfcheck [--bit-exact]` = the single correctness /
  bit-exactness verification entry point.

## Testing (TDD)

1. RED: rewrite `apps/lsxfabric/tests/test_fabric_cli.cpp` —
   `--benchmark`/`--selfcheck`/`--bit-exact` unrecognized, kept flags parse,
   defaults.
2. RED: new `apps/lsxbenchmark/tests/test_lsxbenchmark_cli.cpp` — default mode
   is bench, default model path/prompt/max_gen, every flag parses, combined
   flags, unknown flags ignored.
3. GREEN: implement `lsxbenchmark` (cli + main + CMake), `RunCliBenchmark`,
   prune lsxfabric.
4. Delete the `IModel::Benchmark` API; fix `test_model_runtime.cpp`.
5. AGENTS.md: new `apps/lsxbenchmark/` section, lsxfabric flag list, GPU Perf
   Auto-Tuning (`--bit-exact` now belongs to lsxbenchmark).
6. Verify: `ctest -L fast` green; labels/naming/suite guard tests green.
