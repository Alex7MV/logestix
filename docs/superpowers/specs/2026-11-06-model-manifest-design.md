# Model manifest — one file per model, everything else derived

Status: proposed (2026-11-06). Branch `refactor/model-manifest`.
Supersedes nothing; extends the Model Ingestion Protocol (see
`ingesting-a-model`) with a machine-readable model registry.

## 1. Intent and why

The operator's goal is not a smaller `AGENTS.md`. It is to make model
ingestion and adaptation **mechanical**: a new model should cost a converter, a
kernel set and a manifest — not six rounds of hand-editing registries, test
files, CMake and prose.

`AGENTS.md` is a symptom. It is 1109 lines / 77.8 KB (~21k tokens) loaded in
every session, and 849 of those lines (77%, 61.8 KB) are the `## Structure`
section: a per-model narrative that exists **because** the ingestion procedure is
manual and its results have to be written down somewhere. Two thirds of that
section's sentences are measured results, file inventories and session
narratives, none of which are machine-checked.

The cost of the manual procedure is measurable and already visible as defects:

| surface | today | consequence |
|---|---|---|
| FSI / CODE / DET gates | 105 files, 7 744 lines, `quick_ratio` 0.95 between siblings | 3 files ship a `SKIP` naming **another** model (`beam`, `flint` → `hy-memory-30b-a3b`, `minimax-moe` → `mistral-small-4-119b`) |
| guard registries | `Suites()` 38 rows, `kPassports` 35, `kLifecycle` 55, `kGoldenDebt` 36, `FamilyStems` 15 | `FamilyStems()` omits `beam`, `hy`, `minimax`, `mistral`, `xing` — the cross-family-borrowing check silently passes for 5 of 20 families |
| `tests/CMakeLists.txt` | 6 382 lines, ~10 lines per target, `lsx_add_test` used by 20 of ~450 targets | each model adds ~70 hand-written lines |
| `model_registry.cpp` | 35 `Register<T>()` + 35 includes | one missed line = a model that "does not exist" |
| `AGENTS.md` | 1 109 lines | `fast` is documented as "86/86"; it is 139 |
| `README.md` § Structure | third copy of the inventory | still describes `--benchmark`, `glm_v5_model`, `inference_*.cu` |

None of these are prose problems. They are five **hand-maintained projections of
the same fact set**, and every one of them has drifted. Adding a sixth guard test
to police the copies (as `test_parity_selfcheck_naming_guard` does today) treats
the symptom and pays interest on the debt.

## 2. What exists today (measured on `main`, 2026-11-06)

```
AGENTS.md                       1109 lines / 77 810 B   (Structure: 849 / 61 841)
tests/CMakeLists.txt            6382 lines
tests/test_*.{cpp,cu}          71 893 lines total
  fidelity gates (fsi/code/det)  7 744 lines, 105 files, 35 models x 3 kinds
per-family test suffixes        24 x {convert,load,model,selfcheck,fsi,code,det}
tools/*_tokenizer_capture.py    11 scripts, similarity 0.84
model classes registered        35 (plus one alias)
```

Per-model facts currently duplicated across 4–6 files: `model_type`, vault
directory name, `LSX_TEST_<X>_ROOT` env var, owning namespace, passport header,
format tag, `lsxforge` flag, tokenizer kind, stage→ctest-target map, golden-debt
reason, perf baseline key, lifecycle TUs.

## 3. Goals

1. **One file per model is the source of truth**: `models/<org>/<model>.json`.
2. Every other representation is **generated** from it and **cannot drift**:
   guard registries, ctest registration, parity-gate targets, `model_registry.cpp`,
   `docs/MODELS.md`, the `docs/SELFCHECK_MATRIX.md` table, `AGENTS.md` tables.
3. Ingestion is: manifest → converter → kernels → (generated everything else).
   A new model requires **zero** edits to `tests/CMakeLists.txt`, the guard
   registries, the registry TU, or `AGENTS.md`.
4. `AGENTS.md` becomes a **router**: invariants + pointers, ≤ 200 lines.
5. Drift is caught by a machine, not by review: a fast test regenerates every
   generated artifact in memory and fails on any difference.

## 4. Non-goals

- No change to converter/kernel/model **semantics**. This is a registry and
  build refactor; numeric behaviour and all goldens stay untouched.
- No generated *model code*. A family's converter, kernels, loader, pipeline and
  tool codec stay hand-written (that work is model-specific, not coordination).
- No guard is weakened or deleted for convenience. Guards keep their assertions;
  their *data* moves to the generated table.
- Measured numbers are not "generated": `t/s`, `logits_hash` and gate verdicts
  come from instrumented runs (`tests/perf/perf_baselines.json`, the gate
  binaries), and "not run" stays an honest value.
- A scaffolder that emits family skeletons (approach C) is **deferred** until B
  lands; whether it is worth building is decided afterwards (§12).

## 5. Anti-goals

Things that would make this a failure even if it builds:

- The manifest becoming a **second** thing to maintain by hand — a new model
  with a manifest that is *not* checked against the code (passport missing,
  `kType` mismatch, a stage target that does not exist) is worse than today.
- Losing ctest granularity. Per-model targets must survive: a failing suite must
  still name the model (`ctest -R test_flint_...`).
- Any generated file **committed to git** and then hand-edited.
- A silent behaviour change in the tests: same targets, same labels, same
  `SKIP_RETURN_CODE`, same timeouts, verified by diff (§9).

## 6. Constraints

- No subagents (operator directive). All work is done in the main session.
- C++20, spdlog-only logging, `absl::` host policy, one label per test in the
  same commit (`test_ctest_labels_guard`).
- CMake 3.25+ is the floor; `string(JSON …)` (3.19+) is therefore available
  without new dependencies.
- Windows preset must keep working (no POSIX-only generation; a Python tool and
  CMake-only paths only).
- The 8-stage Model Ingestion Protocol and its invariant are unchanged; this
  design only removes the *registration* half of stage 8.

## 7. Approaches considered

**A. Documentation diet.** Shrink `AGENTS.md`, move per-model facts to
`docs/MODELS.md`, generate the tables. Cheap, removes 21k tokens/session — but
the ingestion procedure stays manual, so the same drift returns in the new file.
Rejected as a *plan*; accepted as a *consequence* of B.

**B. Model manifest as the single source of truth.** One `models/<org>/<model>.json` per
model; generated registries, ctest targets, gate TUs, `model_registry.cpp` and
docs; a sync test that fails on drift. Removes the duplicate projections, makes
ingestion mechanical, and produces A for free. **Chosen.**

**C. Scaffold generator.** `tools/scaffold_model.py` prints a skeleton family
from a filled spec. Fast start, but multiplies code instead of collapsing it and
adds a new drift source. Deferred; only sensible on top of B.

## 8. Design

### 8.1 The manifest

Location: `models/<org>/<model>.json` in the repository root — nested by the
store's own org segment, one file per registered model, sorted by path for stable
diffs. `model_type` inside the file is the registry key (`T::kType`); the
directory is `vault.org` and the stem is `vault.dir`, so the path can be found by
`ls`-ing the source store and the row is checked against its own `vault` block
(`registry_path_error`, generator rule 0).

The registry describes a model as a SET of codec profiles, because one checkpoint
converts to several native trees (invariant I4) and each is a different artifact
with its own codecs and its own measured results:

```jsonc
"preferred_profile": "nvfp4",
"profiles": {
  "nvfp4": {
    "status": "built",
    "artifact": { "sha256": "<64 hex>", "bytes": 10824,
                  "tree_bytes": 17399424244, "planes": 83,
                  "store": "microsoft/phi-4/nvfp4" },
    "codecs": ["bf16", "nvfp4"], "codec_mask": 34, "target": "nvidia:sm_120",
    "gates": {
      "selfcheck": { "result": "pass", "cycles": 3, "top1_token": 12366,
                     "text": "Paris", "logits_hash": "cf51b9bbd5c913dd",
                     "date": "2026-11-28" },
      "fsi": { "result": "pass", "cases": "3/3", "date": "2026-11-28" },
      "code": { "...": "..." }, "det": { "...": "..." }, "mcp": { "...": "..." }
    }
  },
  "bf16": { "status": "not-built" }
}
```

Two rules make this evidence rather than decoration:

* `status: built` requires an artifact recording — a 64-hex `sha256` and a
  positive `bytes` — plus a non-empty `codecs` list, a `target`, and a `date` on
  every gate measurement. A verdict is pinned to a tree by its digest; without a
  digest there is nothing to pin it to.
* `status: not-built` may carry **nothing** but `notes`. Codecs, artifact or gate
  metrics on an unbuilt profile are a fabricated verdict — the failure mode that
  let a deleted tree keep a green row in the matrix.

`preferred_profile` must name a key of `profiles`; it is what the compiled table
exposes as `vault_profile`, i.e. the profile a gate looks for when the operator
does not choose one.

```jsonc
{
  "model_type": "flint",
  "display_name": "Recursive Flint",
  "family": "flint",
  "variant": null,
  "model_class": "FlintMoeModel",
  "model_header": "flint_moe_model.h",
  "namespace": "lsxcommon::flint::frozen",
  "passport": "libs/lsxcommon/src/flint_moe_constants.h",
  "format_tag": "logestix-flint-nvfp4",
  "forge_flag": "--flint",
  "tokenizer_kind": 4,
  "vault": {
    "org": "RecursiveSI",
    "dir": "flint",
    "env": "LSX_TEST_FLINT_ROOT"
  },
  "preferred_profile": "nvfp4",
  "profiles": { "nvfp4": { "status": "not-built" } },
  "lifecycle_tu": ["flint_moe_model.cpp", "flint_moe_inference_pipeline.cu"],
  "stages": {
    "convert":   ["test_flint_moe_convert"],
    "load":      ["test_flint_moe_loader"],
    "model":     ["test_flint_moe_model"],
    "kernels":   ["test_flint_moe_kernels"],
    "selfcheck": ["test_flint_moe_selfcheck"],
    "fsi":       ["test_flint_moe_fsi_selfcheck"],
    "code":      ["test_flint_moe_code_selfcheck"],
    "det":       ["test_flint_moe_det_selfcheck"],
    "perf":      ["test_perf_baseline_flint"]
  },
  "golden_debt": {
    "fsi": "goldens are template-derived; see OPEN_TASKS §R6",
    "code": "same",
    "det": "same"
  },
  "perf": { "baseline_key": "flint" },
  "report": "docs/model-reports/2026-10-2x-Recursive-Flint.md",
  "notes": "hybrid sink/SWA/no-RoPE MoE; learned attention sinks"
}
```

Rules the schema enforces (checked by the sync test, not by convention):

1. All nine `stages` keys are present. An empty list is a **declared gap**, and
   a `golden_debt`/`debt` reason is then required — silence is not allowed.
2. `passport` must exist **and** declare `namespace`; `model_class`'s `kType`
   must equal `model_type` (the generated registry asserts this at compile time).
3. `vault.env` is the only way a test learns where the model lives; no test
   spells a vault directory name.
4. Every stage target must be either an existing file `tests/<target>.cpp`
   (or `.cu`) or declared in `extra_sources`; an orphan target — registered in
   CMake but claimed by no manifest — fails.
5. `model_type` strings appear with metadata only here. Elsewhere they appear
   only as `T::kType` (compile-time) or in generated files.

### 8.2 Generation surfaces

| # | generated artifact | produced at | replaces |
|---|---|---|---|
| 1 | `generated/model_manifest.h` — `constexpr` table (type, family, namespace, passport, vault dir/env, tokenizer kind, stage presence, golden debt, lifecycle TUs) | configure time | `kPassports`, `FamilyStems`, `kGoldenDebt`, `Suites()`, `kLifecycle`, guard orphans |
| 2 | `generated/model_registry.inc` — `#include` + `LsxModelFactory::Register<T>()` | configure time | the 35-line hand list in `model_registry.cpp` |
| 3 | ctest registration for model targets (label, `SKIP_RETURN_CODE 4`, TIMEOUT, synthetic-geometry env chosen by stage) | configure time | ~2 300 hand-written CMake lines |
| 4 | `docs/MODELS.md` — one section per model (identity, passport, vault, gates, perf row, report link, debt) | generator + `--check` | `AGENTS.md` § Structure (849 lines), `README.md` § Structure |
| 5 | `docs/SELFCHECK_MATRIX.md` table rows (declared debt vs measured) | generator + `--check` | hand-updated matrix |

Two rules make this drift-proof:

- **Nothing generated is committed** except the two documentation files, and
  those are checked by a `--check` mode that regenerates into memory and fails on
  a byte difference. Build inputs go to `${CMAKE_BINARY_DIR}/generated/`.
- The generator is one tool (`tools/gen_model_manifest.py`) with three modes:
  `--cmake` (emit CMake fragment), `--header` (emit the C++ table),
  `--docs` (emit markdown), `--check` (all of the above, compare, exit 1).

### 8.3 The parity-gate driver

The 105 FSI/CODE/DET files collapse into **one** translation unit,
`tests/test_parity_gate.cpp`, compiled once per (model, kind):

```cmake
foreach(model ${LSX_MANIFEST_MODELS})
  foreach(kind fsi code det)
    lsx_add_test(NAME test_${model}_${kind}_selfcheck
        SOURCES test_parity_gate.cpp
        DEFINE "LSX_GATE_MODEL=\"${model}\"" "LSX_GATE_KIND=\"${kind}\""
        LABEL inference TIMEOUT 3600 SKIP)
  endforeach()
endforeach()
```

The driver reads its own row from `generated/model_manifest.h`, resolves the
vault path from the manifest's env var, `SKIP`s with a message built from the
manifest (so it can no longer name a neighbour), and calls the existing
model-neutral `fsi::RunFsiSelfCheck` / `code::…` / `det::…`. Per-model ctest
granularity is preserved; `test_parity_selfcheck_naming_guard` and its
`kKnownOffenders` debt become obsolete and are deleted along with their subject
(a guard whose defect class no longer exists must not linger green).

SelfCheck (the "Paris" gate) also runs through this driver, because it is the
last place a vault directory name and an env var are hardcoded per family. Its
family extras (EOS ids, cross-instance determinism, the single-framing cases)
are **declared hooks**, not copies: the manifest carries
`"selfcheck_extras": ["eos", "cross_instance"]`, and the neutral library
`lsxcommon/selfcheck_hooks.*` implements each hook once. A family that needs a
new hook adds it to the library (one implementation), not to its test file.

Two rules survive from `test_model_suite_guard` and stay static scans of the
test sources, unchanged: loader/model stage tests must not touch a vault path
(they run on synthetic fixtures), and no test may include another family's
fixture header.

### 8.4 `AGENTS.md` after the change

Target ≤ 200 lines, and every line survives only if it is an **invariant, a
pointer, or a rule an agent can violate**:

1. Project + architecture invariants (Blackwell, driver/runtime skew, no
   pre-Blackwell).
2. Dependency table (already single-sourced with `test_core.cpp` — keep).
3. Pointers: build → `building-the-project`; tests → `test-lifecycle`; models →
   `docs/MODELS.md`; code map → `logestix-repo-map`; open work →
   `docs/OPEN_TASKS.md`; gate status → `docs/SELFCHECK_MATRIX.md`.
4. The skill map table (unchanged — it is the router).
5. Cross-cutting rules that no skill owns (absl policy headline, logging,
   autonomous mode, git, NVIDIA docs discipline).

Everything currently in `## Structure` that is *invariant* (e.g. "the only GEMM
path is `lsx_gemm`", "no CLI speed flags", "MMAP-only Arrow reads") moves into
the numbered sections; everything that is *description of a model* moves to
generated `docs/MODELS.md`; everything that is *history* is already in
`docs/open-tasks/` and is replaced by a link.

### 8.5 The schema escape hatch (the only place convention is not enough)

Some targets do not follow `<target>.cpp` (a `.cu` kernel test, a target that
needs a define, a timeout above the stage default). The manifest expresses them
explicitly instead of CMake growing a special case:

```jsonc
"target_overrides": {
  "test_glm_moe_kernels":            { "source": "test_glm_moe_kernels.cu" },
  "test_mistral_moe_large4_kernels": { "source": "test_mistral_moe_large4_kernels.cu",
                                       "timeout": 300 },
  "test_hy_moe_prompt_framing":      { "env": { "LSX_HY_FIXTURE_DIR": "${SOURCE}/tests/fixtures/hy" } }
}
```

A target in a stage list that is neither convention-resolvable nor overridden
fails the sync test. There is no third path: what is not conventional is
declared, and what is declared is checked against the filesystem.

Windows: the operator rule "run model tests only for qwen/gemma" is encoded once
in the generator's CMake fragment (`if(WIN32)` filters the model loop), not in
`AGENTS.md` prose and not per target.

### 8.6 Where the narrative goes

Phase 6 is a migration, not a deletion. Every paragraph of today's
`## Structure` is routed by exactly one of four rules:

1. **Invariant / rule** (the only GEMM path is `lsx_gemm`; no CLI speed flags;
   MMAP-only Arrow reads) → a numbered section of the new `AGENTS.md`.
2. **Per-model fact** (identity, passport, vault, gates, perf row, format tag,
   debt, report link) → generated `docs/MODELS.md`. **Geometry is not among them**
   — see the amendment; the catalogue points at the passport instead.
3. **Measured result with a canonical home** (perf row, `logits_hash`, gate
   verdict) → stays where it is (`perf_baselines.json`, `SELFCHECK_MATRIX.md`)
   and is *linked*, never restated.
4. **Design rationale / session history** (why the LFU cache was rejected, the
   nsys decomposition behind a lever) → the model's report in
   `docs/model-reports/` or the dated archive in `docs/open-tasks/`, with a
   one-line pointer left in `docs/MODELS.md`.

The migrator commits the four-way routing as a table in the phase commit, so
"nothing was lost" is reviewable rather than claimed. `docs/MODELS.md` carries a
hard budget of 15 lines per model, enforced by the sync test — otherwise it
becomes the next `AGENTS.md`.

### 8.7 Migration phases

Each phase lands green (fast + the affected label sweep) and is independently
reviewable.

| phase | content | proof |
|---|---|---|
| **P1** | `models/<org>/<model>.json` for all 35 models, transcribed mechanically from the existing registries; generator `--docs`; `docs/MODELS.md` generated; `test_manifest_sync` (label `fast`) | generated doc matches hand knowledge; nothing else changes |
| **P2** | guards consume `generated/model_manifest.h`; delete `kPassports`, `FamilyStems`, `kGoldenDebt`, `Suites()`, `kLifecycle` hand rows; assertions unchanged | fixes the 5 missing families; guard tests keep passing |
| **P3** | one `test_parity_gate.cpp` + generated targets; delete 105 files; delete `test_parity_selfcheck_naming_guard` | target/property diff identical; the 3 stale `SKIP` strings disappear |
| **P4** | ctest registration for model targets from the manifest; delete the hand blocks | `ctest --test-dir … -N` output identical |
| **P5** | `model_registry.cpp` from `generated/model_registry.inc` | `LsxModelFactory::Create` resolves all 35 types |
| **P6** | generate `docs/MODELS.md` + matrix rows; rewrite `AGENTS.md`; `README.md` § Structure → pointer | AGENTS ≤ 200 lines; sync test green |

P2→P5 are behaviour-preserving by construction; the oracle is a **target dump**
(§9), not review.

## 9. Verification

1. **Coverage oracle (external, taken before P1).**
   `tools/dump_ctest_manifest.py` writes every target with its labels,
   properties, timeout and skip code to JSON. It is captured on `main` **before
   the first manifest is written** and re-diffed after every phase, so a
   transcription error in P1 cannot hide behind a self-consistent manifest.
   Committed under `tests/fixtures/ctest/` so the diff is reviewable, not
   asserted from memory.

   During P1–P3 the generator also runs in **shadow mode**: it emits the CMake
   fragment from the manifests and the result is diffed against the hand-written
   blocks still in `tests/CMakeLists.txt`. Only when the two agree does P4 delete
   the hand blocks. That is what makes "35 manifests transcribed by hand" a
   checkable claim rather than a hopeful one.
2. **Sync test.** `test_manifest_sync` (label `fast`) regenerates all artifacts
   in memory and fails on any difference; it also enforces the five schema rules
   of §8.1 (missing passport, `kType` mismatch, orphan target, undeclared gap,
   stale debt row).
3. **Guard suites unchanged.** `test_model_independence_guard`,
   `test_model_suite_guard`, `test_ctest_labels_guard`, `test_code_naming_guard`
   keep their assertions; only their data source changes.
4. **A negative test.** The sync test must fail on a deliberately mutated
   manifest (rename a stage target, drop a passport) — otherwise it proves
   nothing. Same shape as the existing two-way allowlist guards.
5. `ctest --preset linux-debug` (fast) green, plus one model sweep
   (`ctest --preset linux-debug-inference -R flint`) before/after P3.

## 10. What stays manual (honest statement)

The manifest removes **registration** toil, not model work. Still hand-written
per model: the passport's geometry VALUES (the header itself is the source of
truth and the catalogue links it), converter, NVFP4 packing rules, kernels, loader,
pipeline, tool codec, goldens capture, perf seeding. That is 6 of the 8 ingestion
stages — but it is the part that is genuinely different per model, and it is the
part the operator wants to spend time on.

Estimated effect of B: ~105 gate files → 1; ~2 300 CMake lines → ~40;
~150 registry rows → 0 hand rows; 35 registry statements → 0; `AGENTS.md`
1 109 → ≤ 200 lines; three stale-copy defect classes become impossible.

## 11. Risks and mitigations

| risk | mitigation |
|---|---|
| Manifest drifts from code | five two-way checks in §8.1, each failing loudly; `kType` cross-check is compile-time |
| Generator becomes a build dependency on Python | repository already requires Python for every capture tool; CMake-only fallback is the `--check` diff |
| Configure-time generation breaks the Windows preset | same CMake+Python path on both; generation is pure text, no POSIX calls |
| Silent test loss during P3/P4 | ctest dump oracle (§9.1), reviewed as a commit artifact |
| Manifest becomes a dumping ground | schema is closed (fixed keys), `notes` ≤ 120 chars, anything measured lives in the JSON/matrix instead |
| The repo compounds debt while B is in flight | phases are additive-first (P1/P2 add and read, only P3–P5 delete) |

## 12. Deferred: approach C (scaffolder)

Decided **after** B lands. B gives the generator the manifest schema and the
file-naming conventions; C then becomes "emit a manifest plus empty family TUs
that satisfy the sync test". Its value depends on how much of a new family is
boilerplate after P1–P6 — measurable at that point as "lines a fresh family
needs that are not converter/kernel logic". Not designed here.

## 13. What is left to the builder

- Exact CMake fragment shape for phase 4 (one `foreach` over models × stages vs
  per-stage loops) — must keep `lsx_add_test` as the single registration helper.
- Whether `golden_debt` reasons live in the manifest or as a `//` comment block
  in a generated header (default: manifest, since the sync test reads it).
- The precise split of `AGENTS.md` § Structure between "invariant" and
  "generated description"; the criterion is in §8.4.
- How `variants` (kimi `k3`/`k31_pro`, mistral `large4`, gemma dense/moe) share
  a family stem — one manifest each, `family` shared, `variant` naming the
  delta; the sync test requires the shared family to be identical.

---

## Amendment (2026-11-06): no geometry in the catalogue

The generated `docs/MODELS.md` deliberately does **not** restate each model's
geometry. The original §8.2 row asked for a "geometry one-liner"; that promise is
withdrawn here, for three measured reasons:

1. **There is no single schema.** The models fall into four architecture classes
   with genuinely different parameters: GQA+dense (5), GQA+MoE (16), MLA+MoE
   (4), and the KDA/GDN hybrids (10) where two attention kinds coexist in one
   model. A seven-field line would print blanks for half the catalogue.
2. **A plausible wrong number is worse than no line.** A synonym table wide
   enough to cover the three naming styles (`kHidden` / `kHiddenSize` /
   `kGemmaHidden`) maps `kKvLora = 512` to "kv heads: 512" for GLM-5.3-Flash -
   a latent rank read as a head count. This repository's own rule is that a
   claim needs an external oracle; a catalogue line has none.
3. **The passport already is the machine-checked source.** Every value is a
   `constexpr` under `static_assert`, which is stronger than prose: it cannot rot
   without failing the build.

So the catalogue links the passport (`- class …, passport …, namespace …`) and
the reader opens it, or the model's report, for numbers.
