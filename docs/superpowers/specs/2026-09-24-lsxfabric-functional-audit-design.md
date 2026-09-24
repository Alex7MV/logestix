# lsxfabric functional audit — design

Date: 2026-09-24
Branch: `audit/lsxfabric-functional-audit`
Status: approved (design), pending implementation

## 1. Goal

Audit the functionality of the `lsxfabric` app (`apps/lsxfabric/`) against its
stated contracts, using a test-driven approach. Deliver two things:

1. Tests that characterise current behaviour (green) and tests that encode the
   expected contract where it diverges from current behaviour (red, hidden).
2. A findings report (`docs/LSXFABRIC_AUDIT.md`) with evidence and
   recommendations.

**Non-goal:** changing production code. `validator.cpp`, `cli.cpp`, `main.cpp`
and `apps/lsxfabric/CMakeLists.txt` are not modified. Only test files, test
registration (`apps/lsxfabric/tests/CMakeLists.txt`) and documentation are
added.

## 2. Why this scope

The operator chose a **full functional audit**; divergences are **reported and
tested, not fixed**; both a green characterisation set and a red spec set are
required. Runtime integration (`--validate` over the real vault and real GPU
generation) is in scope as empirical evidence.

`lsxfabric` is small: CLI parsing (`cli.cpp`), a model validator
(`validator.cpp`), and a `main` that dispatches to either validation
(`--validate`) or generation (`--testprompt`). The generation path itself is
shared `lsxcommon` runtime code; only the dispatch wiring is `lsxfabric`'s own.

## 3. Surface inventory

| ID | Surface | Contract source |
|---|---|---|
| S1 | `ParseFabricCliArgs` — defaults, each flag, unknown-token policy, missing values, `--prompt` swallowing, `--max-gen` parsing | `cli.h` comments, `AGENTS.md`, `running-a-model` |
| S2 | `main` dispatch — mode precedence (`--testprompt` before `--validate`), return code mapping, no-op mode | `main.cpp`, `cli.h` |
| S3 | Validator primitives — `ValidateNVFP4Data`, `ValidateE4M3Scales`, `ValidateGlobalScale` | `validator.h`, NVFP4/E4M3 semantics |
| S4 | `LoadModelIndex` — JSON parsing, missing/typed fields, malformed input | `model.index.json` vault, converter |
| S5 | Schema validators — `ValidateDenseNvfp4Schema`, `ValidateExpert`, `ValidatePinnedTensors` | `lsxcommon/schema.h` |
| S6 | Orchestrator `RunValidator` — format allow-list, model layout, error/warning accounting, summary | `validator.cpp`, real vault formats |
| S7 | Integration — `--validate` over all vault models; real generation on 1–2 models | binaries + vault |

## 4. Test strategy (TDD)

Per surface, in order:

1. Derive the expected contract from the sources above.
2. Write a **green characterisation test** asserting current behaviour. It must
   pass on the unmodified tree.
3. Where current behaviour diverges from the contract, write a **red spec test**
   asserting the expected behaviour.

Red spec tests are Catch2 tests whose tags begin with `.` (`[.audit-spec]`), so
they are hidden from the default `ctest` run and the suite stays green. They are
run explicitly:

```
./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric "[.audit-spec]"
./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"
```

Each spec test carries a comment naming the finding id in the audit report.

Tests use synthetic fixtures and `temp_directory_path()` so the default suite
remains reproducible without a GPU, a vault, or network. `main` (S2) cannot be
linked into a test target, so it is audited by inspection plus the S7
integration runs.

### File organisation

No new CTest targets are added; new test translation units are attached to the
existing `test_lsxfabric` and `test_fabric_cli` targets via `target_sources`.
This keeps `test_ctest_labels_guard` satisfied (no new label obligations) and
avoids any GPU/vault coupling in the default suite.

Candidate files (final set fixed during implementation):

- `apps/lsxfabric/tests/test_validator_contract.cpp` — S3–S6 green + red.
- `apps/lsxfabric/tests/test_fabric_cli_edges.cpp` — S1 green + red.
- Extensions to existing `test_validator.cpp` / `test_fabric_cli.cpp` where a
  case naturally belongs to an existing `TEST_CASE`.

## 5. Integration procedure (S7)

Ordering obeys the host/GPU resource policy (`running-a-model` §8): `--validate`
is host-only and runs first; generation is GPU-exclusive, sequential, and last.

1. Build: `cmake --build --preset linux-debug`.
2. For each of the 10 converted vault models:
   `lsxfabric --localmodel <root> --validate`; capture return code,
   stdout and stderr.
3. Real generation, one at a time, no other GPU work:
   - `lsxfabric --localmodel <DeepSeek-V4-Flash> --testprompt --max-gen 64`
     (default model / default path).
   - `lsxfabric --localmodel <gemma-4-26B-A4B-it> --testprompt --max-gen 64`
     (fast, 4B active).
4. Raw commands and outputs go to the audit report appendix.

## 6. Findings report

`docs/LSXFABRIC_AUDIT.md` (follows the `docs/*_AUDIT.md` precedent):

1. Metadata: date, branch, commit, environment, scope, method, non-goals.
2. Surface inventory.
3. Findings table, sorted by severity:
   `id · severity · surface · expected vs actual · evidence · impact ·
   recommended fix (not applied)`.
4. Test inventory: `file · test name · tag · green/red · surface`.
5. How to run (default green + explicit red spec).
6. Integration evidence appendix.
7. Conclusions and recommendations.

Severity scale:

- **Critical** — silently wrong `PASS` on a corrupt/incomplete model, or a false
  `FAIL` rejecting valid data.
- **High** — contract broken for all real inputs.
- **Medium** — misleading accounting, ignored edge cases.
- **Low** — cosmetic / logging only.
- **Info** — observations without a defect.

## 7. Pre-audit hypotheses (to confirm or refute)

- F1: the format allow-list is stale — none of the 10 vault formats matches
  `logestix-v1` / `logestix-qwen` / `logestix-glm`.
- F2: `ValidateNVFP4Data` is a tautology (`lo > 0x0F || hi > 0x0F` can never be
  true for nibbles).
- F3: `ValidateE4M3Scales` requires a zero mantissa, rejecting scale values with
  a mantissa.
- F4: flags missing their value (`--localmodel`, `--max-gen`, `--prompt`) are
  silently ignored; `--prompt --validate` consumes the following flag as prompt
  text.
- F5: `--max-gen` uses `std::atoi` without validation.
- F6: dense-format pin schemas are only validated when the filename contains
  `nvfp4`; recent dense formats are not covered.
- F7: the expert path `experts/layer_N/exp_N.arrow` may not match every family
  layout.
- F8: missing pinned/expert files produce warnings, not errors, so an incomplete
  model can report `PASS`.
- F9: the summary's "Pinned tensors checked" prints the declared count, not the
  loaded count.
- F10: the `DenseNvfp4Schema` branch never increments `experts_checked`.

These are hypotheses; the audit confirms, refutes or refines each and may add
more.

## 8. Risks and mitigations

- **GPU exclusivity** — integration last, sequential; validate is host-only.
- **Vault coupling** — default suite stays synthetic/temp; vault runs are manual
  evidence, not registered tests.
- **Red tests fail by design** — hidden tags keep CI green; documented run
  command.
- **Guard tests** — no new targets/labels; verify with
  `ctest -R "lsxfabric|labels"` after registering tests.
- **Logging state** — multiple `Core::Initialize`/`Core::Shutdown` calls in
  tests follow the existing pattern; keep one initialise/shutdown per test that
  logs.

## 9. Out of scope

- Applying fixes.
- Auditing shared `RunCliGeneration` / `model_runtime` internals except through
  the S7 integration.
- Speed/performance measurement (lsxbenchmark's domain).
