---
name: test-lifecycle
description: Use when adding or running logestix tests — the fast/convert/inference/perf CTest labels, the test_router.cmake path rules and LSX_PRESET/LSX_LIST/LSX_DIFF, label discipline and the guard-test registry, preset commands, SKIP_RETURN_CODE 4 and converter TIMEOUTs. Triggers - добавить тест, label, ctest, test router, guard test, convert timeout.
---

# Test lifecycle

## Overview

Every CTest target in logestix belongs to exactly one of four label categories,
and the category decides *when* the test runs. The discipline is machine-enforced
by `test_ctest_labels_guard`, and the category for a changed path is chosen by the
router in `tools/test_router.cmake`. Get the label wrong and CI goes red; get the
category wrong and an inference test runs on every quick build.

Related: converter test *rules* (synthetic checkpoint, TIMEOUT) live in
`ingesting-a-model`; the perf suite and baselines live in `perf-tuning-a-model`;
the guard tests' registries also appear in `logestix-repo-map`.

## Categories

| Label | What it is | When it runs |
|---|---|---|
| `fast` (45) | host-only tests | every `ctest --preset` (default of all presets) |
| `convert` (13) | model converters | when a converter changed (router) or when debugging a model |
| `inference` (85) | CUDA / model contour | when the contour changed (router) or manually |
| `perf` (15) | perf benchmarks on real weights | separate command only, Release only |

Counts are a Linux snapshot (2026-09); `test_ctest_labels_guard` is the real
authority.

## Label discipline

A new test MUST carry its label **in the same commit**. `test_ctest_labels_guard`
fails the build otherwise. Classification rules for a changed path are the
`lsx_router_rule(...)` table in `tools/test_router.cmake`.

## Guard tests

| Guard (in `tests/`) | Rule it enforces |
|---|---|
| `test_code_naming_guard.cpp` | model file-naming scheme (`SchemeRegex()` / `AllowedCommon()`) |
| `test_model_independence_guard.cpp` | unique frozen-geometry passport per model, no cross-family borrowing, `LsxFastMathScope` in `Infer` (`kPassports`, `consumers`) |
| `test_model_suite_guard.cpp` | per-model lifecycle stages exist as ctest targets (`Suites()`); `SKIP_RETURN_CODE 4` on selfcheck; no vault loading in loader/model stages |
| `test_ctest_labels_guard.cpp` | every test target carries a label; every skip-capable target (`SKIP(` in its source) has `SKIP_RETURN_CODE 4`; `test_perf_baseline.cpp` carries an NDEBUG Release guard (all in addition to the label/TIMEOUT/preset/router checks) |
| `test_absl_format_guard.cpp` | `absl::StrFormat` is printf-style (`%s/%d/%u/%f`), never `{}` |
| `test_cuda_toolkit_guard.cu` | CUDA runtime ≥13.2, header/runtime match, `cuda_fp4` E2M1 semantics, Blackwell-only GPUs |

Extending a model means extending the relevant guard registry in the same commit —
the registries are described in `ingesting-a-model` and `logestix-repo-map`.

## Running the suites

```bash
ctest --preset linux-debug                  # fast only (Ubuntu 26.04)
ctest --preset linux-debug-convert          # convert sweep
ctest --preset linux-debug-inference        # inference sweep
ctest --preset linux-release-znver4-perf    # perf; Release-only
ctest -L "^(fast|convert|inference|perf)$"  # full run
```

Other OS profiles: `ctest --preset windows-debug` (fast). Preset names are built
from the base (`linux-debug`, `linux-release`, `linux-release-znver4`,
`windows-debug`, `windows-release`) plus a `-fast`/`-convert`/`-inference`/`-perf`
suffix that maps to the label filter. The base test preset filters `^fast$`.

- **Perf baselines require Release.** `test_perf_baseline` is NDEBUG-gated: a
  Debug run `SKIP`s (exit 4) before loading, because an unoptimized binary would
  be judged `DEGRADED` against the Release reference. There is no useful Debug
  perf run.
- **`SKIP_RETURN_CODE 4`** must be set on any target that can skip (real weights
  absent); otherwise CTest reports the skip as a failure. `test_ctest_labels_guard`
  enforces this for every target whose source calls `SKIP(`.
- **WINDOWS-ONLY RESTRICTION**: on Windows run model tests only for Qwen3.5-9B
  and gemma-4-26B-A4B-it — `ctest -R "qwen|gemma"`. Other model tests (GLM,
  DeepSeek, …) must not run on that OS.

## Test router

The router keeps "fast always, plus the affected contour" honest. It is driven by
`tools/test_router.cmake`; the rule table below is a snapshot — the file is the
source of truth.

```cmake
lsx_router_rule(<id> <labels> <regex>)   # FIRST match wins; order matters
```

| rule | labels | regex |
|---|---|---|
| `converters` | convert | `libs/lsxcommon/.*(converter\|_convert\|tokenizer_convert)` |
| `quant_engine` | convert inference | `libs/lsxcommon/.*safetensors_gpu` |
| `cuda` | inference | `\.cuh?$` |
| `lsxcommon_host` | fast | `libs/lsxcommon/.*(tokenizer\|unicode\|pretokenizer\|path_registry\|schema\|orchestrator)` |
| `lsxcommon` | inference | `libs/lsxcommon/` |
| `network` | fast | `libs/lsxnetwork/` |
| `apps` | fast | `apps/` |
| `tests` | convert inference | `^tests/` |
| `tooling` | fast | `^(CMakeLists\.txt$\|CMakePresets\.json$\|cmake/\|docs/\|tools/\|[^/]+\.md$)` |
| `default` | fast | `.` |

```bash
# autodetect labels from git diff (fast always; convert/inference when the
# matching contour changed):
cmake -DLSX_PRESET=linux-debug -P tools/test_router.cmake
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake  # dry-run
```

`LSX_DIFF` overrides the diff base; `LSX_PRESET` selects the target tree.

## Converter tests

Converter tests are `convert`-labelled and MUST have a bounded `TIMEOUT`
(≤600 s; canonically 120 s) and a synthetic checkpoint — converting a real model
in a test is forbidden. The full rules (synthetic geometry env gate, fixture
coverage, FP8 tiles) are the **Converter test rules** section of
`ingesting-a-model`.

## Common mistakes

- Adding a test without a label in the same commit → `test_ctest_labels_guard` red.
- Forgetting `SKIP_RETURN_CODE 4` on a test that cannot find real weights → the
  skip is reported as a failure.
- Running `-L perf` from a Debug tree and believing the numbers.
- Testing a converter by converting the real vault instead of a synthetic
  checkpoint (also a TIMEOUT waiting to happen).
- Adding an inference-labelled test for a host-only change, or vice versa —
  check the router rule for the path.
