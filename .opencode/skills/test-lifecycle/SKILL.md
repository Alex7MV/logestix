---
name: test-lifecycle
description: Use when adding or running logestix tests — the fast/convert/inference/perf CTest labels, the test_router.cmake path rules and LSX_PRESET/LSX_LIST/LSX_DIFF, label discipline and the guard-test registry, preset commands, SKIP_RETURN_CODE 4 and converter TIMEOUTs. Triggers - add test, label, ctest, test router, guard test, convert timeout.
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

**The one rule that costs the most time:** a model-local change is verified with
the `fast` preset plus that model's own targets — never with a label sweep. See
"Running the suites" below for the measured reason.

## Categories

| Label | What it is | When it runs |
|---|---|---|
| `fast` | host-only tests | every `ctest --preset` (default of all presets) |
| `convert` | model converters | when a converter changed (router) or when debugging a model |
| `inference` | CUDA / model contour | when the contour changed (router) or manually |
| `perf` | perf benchmarks on real weights | separate command only, Release only |

Counts move every week and a number in a skill rots silently, so read them
instead of trusting a table:

```bash
ctest --preset linux-debug -N                  # fast total (the preset default)
grep -oE 'LABELS "[a-z]+"' build/linux-debug/tests/CTestTestfile.cmake | sort | uniq -c
```

(linux-debug on 2026-10-02: fast 86 through the preset, convert 20,
inference 172, perf 22 targets — a target may carry two labels, so the label
sums exceed the fast total.) `test_ctest_labels_guard` is the authority for
"every target has a label".

## Label discipline

A new test MUST carry its label **in the same commit**. `test_ctest_labels_guard`
fails the build otherwise. Classification rules for a changed path are the
`lsx_router_rule(...)` table in `tools/test_router.cmake`.

## Registering a test

Use the helpers in `cmake/LsxTestHelpers.cmake` (included from the root
`CMakeLists.txt`) instead of repeating add_executable / link / defines /
add_test / set_tests_properties / the Windows Arrow-DLL copy:

```cmake
lsx_add_test(NAME test_foo
    LINK lsxcommon              # Catch2::Catch2WithMain is added automatically
    INCLUDE "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
    LABEL fast                  # omit when the grouped label list owns it
    TIMEOUT 120
    SKIP                        # SKIP_RETURN_CODE 4
    ENV "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

lsx_add_cuda_test(NAME test_bar_kernels     # OBJECT lib: test_bar_kernels_cu
    OBJECT_LINK lsxcommon_kimi_cu           # extra link into the OBJECT lib
    FMAD_OFF                                # bit-exact kernels
    LABEL inference)
```

`lsx_add_test` also copies the Arrow DLLs on Windows when `LINK` contains
`lsxcommon` / `lsxnetwork` / `lsx_dep_includes` / `arrow_shared`
(`NO_DLLS` / `ARROW_DLLS_ONLY` override). `lsx_add_cuda_test` builds the `.cu`
OBJECT library with `LSX_CUDA_ARCH`, the standard CCCL/MSVC defines,
`lsx_dep_includes` and `LSX_KERNEL_BUDGETS_FILE`, then links a CXX executable
against `lsxcommon`. Dependency include paths live only in `lsx_dep_includes`
(root `CMakeLists.txt`); never spell `${CMAKE_BINARY_DIR}/_deps/...`.

## Guard tests

| Guard (in `tests/`) | Rule it enforces |
|---|---|
| `test_infrastructure_guard` | no `${CMAKE_BINARY_DIR}/_deps` include paths, exactly one definition of the shared converter helpers (`ReadTensorData`, `WriteBatch`, `WriteExpertArrow`, `ExpertArrowPath`, `QuantizeSlice`), `CUDA_ARCHITECTURES` only `${LSX_CUDA_ARCH}`/`OFF` |
| `test_code_naming_guard.cpp` | model file-naming scheme (`SchemeRegex()` / `AllowedCommon()`) |
| `test_model_independence_guard.cpp` | unique frozen-geometry passport per model, no cross-family borrowing, `LsxFastMathScope` in `Infer` (`kPassports`, `consumers`) |
| `test_manifest_sync.cpp` | `models/*.json` is the registry: the compiled table equals the manifests, `docs/MODELS.md` is current (`--check`), no `tests/test_*.{cpp,cu}` file is unclaimed (`--strict`), every row resolves through `LsxModelFactory`, and no `docs/SELFCHECK_MATRIX.md` row names a model with no manifest |
| `test_manifest_generator_selftest` | the generator's own schema validator: mutates a live manifest and asserts all eight rules fire |
| `test_model_suite_guard.cpp` | the per-model suite is autonomous AND complete (`Suites()`, projected from the generated `kTargets`): every lifecycle stage (convert → load → model → kernels → selfcheck → fsi → code → det → **perf**) has a registered ctest target; every family test target is claimed by a row (no orphans) and every row declares the full stage set; `SKIP_RETURN_CODE 4` on the selfcheck/fsi/code/det targets; **loader/model stage tests must run on synthetic fixtures** — no vault path, and no cross-family fixture borrowing (the synthetic primitives live only in `tests/test_synth_checkpoint.h`). One case also pairs every row against `tests/fixtures/{fsi,code,det}`: a parity golden that is missing **and not declared in `kGoldenDebt`** fails, and a `kGoldenDebt` row whose golden now exists fails too |
| `test_ctest_labels_guard.cpp` | every test target carries a label; every skip-capable target (`SKIP(` in its source) has `SKIP_RETURN_CODE 4`; `test_perf_baseline.cpp` carries an NDEBUG Release guard (all in addition to the label/TIMEOUT/preset/router checks) |
| `test_absl_format_guard.cpp` | `absl::StrFormat` is printf-style (`%s/%d/%u/%f`), never `{}` |
| `test_logging_guard.cpp` | host logging goes through spdlog / `lsxcommon::log`: no `std::cout`/`std::cerr`/`std::clog`, no `printf`/`fprintf`/`vprintf`/`vfprintf`/`puts`/`putchar`/`perror` in `libs/` and `apps/`; the exception allowlist is exact-trimmed-line and two-way (an undeclared call and a stale entry both fail) |
| `test_cuda_toolkit_guard.cu` | CUDA runtime ≥13.2, header/runtime match, `cuda_fp4` E2M1 semantics, Blackwell-only GPUs |

Extending a model means adding a `models/<model_type>.json` row — the registries
those guards read are GENERATED from it at configure time, so there is no registry
to hand-edit. What a manifest needs is skill `ingesting-a-model`.

## Running the suites

**Scope a model change to that model.** Ingesting or fixing one model means:

```bash
ctest --preset linux-debug                      # the fast guards (parallel, ~3 min)
ctest --test-dir build/linux-debug -R "<model_type>" -j 1   # that model's targets
```

The label sweeps below are for **infrastructure** changes (a shared kernel layer,
a schema, the router, the manifest generator) — not for a model-local diff:

```bash
ctest --preset linux-debug-convert          # convert sweep
ctest --preset linux-debug-inference        # inference sweep
ctest --preset linux-release-znver4-perf    # perf; Release-only
ctest -L "^(fast|convert|inference|perf)$"  # full run
```

Measured 2026-11-22: `linux-debug-inference` is **350 targets across 43 models**
(qwen 225, gemma 77, kimi 29, inkling 29, glm 25, deepseek 25, …). Because every
device target carries `RESOURCE_LOCK "gpu"` the sweep must run `-j1`, so a model
ingestion that runs "the sweep" spends hours on targets its diff cannot reach.
Run the model's own row instead — the targets are derived from
`models/<model_type>.json`, so `-R` on the model_type is the whole set.

Other OS profiles: `ctest --preset windows-debug` (fast) — the same label
filters as Linux; a model test is not skipped or filtered by OS. Preset names are built
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

## Fuzzing

Fuzz targets are NOT ctests: a fuzzer needs a corpus and a time budget, and
its exit status is not a pass/fail signal, so it would break the label
discipline. Two layers instead:

- Always-on: the deterministic mutational fuzz inside the owning unit test
  (`tests/test_safetensors.cpp`, label `fast`) — fixed seeds, 1000-2000
  iterations, the same invariants (only validated descriptors escape; a
  successful read equals an independently computed file slice).
- Opt-in: a libFuzzer harness under `tests/fuzz/` (outside the `test_*`
  naming/label guards), enabled with `-DLSX_BUILD_FUZZERS=ON` on Clang. Build
  it in its own tree with the library instrumented
  (`-DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address"`), e.g.
  `build/fuzz/tests/fuzz_safetensors -max_total_time=60 -max_len=8192`.

## Converter tests

Converter tests are `convert`-labelled and MUST have a bounded `TIMEOUT`
(≤600 s; canonically 120 s) and a synthetic checkpoint — converting a real model
in a test is forbidden. The full rules (synthetic geometry env gate, fixture
coverage, FP8 tiles) are the **Converter test rules** section of
`ingesting-a-model`.

## Common mistakes

- **Running a full label sweep while ingesting ONE model.** `linux-debug-inference`
  is 350 targets across 43 models and must run `-j1` (GPU lock), i.e. hours of
  work on targets the diff cannot reach. Use `ctest --preset linux-debug` (the
  fast guards) plus `ctest -R "<model_type>"` for the model itself; the full
  sweeps are for infrastructure changes.
- Adding a test without a label in the same commit → `test_ctest_labels_guard` red.
- Forgetting `SKIP_RETURN_CODE 4` on a test that cannot find real weights → the
  skip is reported as a failure.
- Running `-L perf` from a Debug tree and believing the numbers.
- Testing a converter by converting the real vault instead of a synthetic
  checkpoint (also a TIMEOUT waiting to happen).
- Adding an inference-labelled test for a host-only change, or vice versa —
  check the router rule for the path.

### A shared driver means a SHARED rebuild

When many targets compile one source (`tests/test_selfcheck_driver.cpp` feeds 37
targets via `LSX_SELFCHECK_MODEL`, `tests/test_parity_gate.cpp` feeds every
FSI/CODE/DET cell), editing that source invalidates all of them at once. Rebuilding
"just the ones I am looking at" leaves a tree where some binaries carry the old
driver and some the new, and a sweep over it reports failures that belong to the
build, not to the code. Two invalid conclusions in one session came from exactly
this (2026-11-07: four targets red because they had not been rebuilt after the
driver gained per-family probe fields; one of them had also been linked from a
source file that was reverted without a rebuild).

Rule: after touching a shared driver or a shared `.cu`, build the whole contour
before measuring — `cmake --build --preset <preset>` (or the full target list), not
a hand-picked subset. If you must build a subset, say so in the report and re-run
the rest before quoting any verdict.

A binary in a stale tree is also the best AVAILABLE oracle for "was this green
before?" — release artifacts whose mtimes predate a refactor, plus a string check
that they lack the new code's symbols, let you replay the old probe. Verify the
strings; an mtime alone can be a relink of unchanged sources.
