# lsxfabric audit remediation — design

Date: 2026-09-24
Branch: `audit/lsxfabric-functional-audit`
Status: approved (design), pending implementation
Supersedes: the remediation part of
`2026-09-24-lsxfabric-functional-audit-design.md`

## 1. Goal

Remediate the High and Medium findings of `docs/LSXFABRIC_AUDIT.md`:

- High: F1 (stale format allow-list), F6 (dense path), F8 (missing expert is
  not an error).
- Medium: F4 (flag swallowing), F5 (`--max-gen` parsing), F7 (expert coverage),
  F12 (pinned schema), F14 (pinned count).

## 2. Root cause and decision

`apps/lsxfabric/src/validator.cpp` reimplements model knowledge — format tags,
expert/pinned/`shared`/`router`/`ple` layouts, schemas — that the inference
engine already owns in its registry and per-family loaders. That duplication is
why the allow-list and counts go stale.

Decision (operator): `--validate` must work **only through the inference
engine**, whose loaders already understand every model specific. The chosen
surface is a real model load: `InferenceEngine::CreateModel` → `IModel::Load`
→ `IModel::Unload`. The engine's registry
(`libs/lsxcommon/src/model_registry.cpp`) already registers every current family
(DeepSeek V4/V4.1, GLM V5/Flash, Qwen 3.5/3.6/3.8, Gemma, Ornith, Nemotron,
GigaChat, Granite, Kimi) and autodetects the type from `model.index.json`.

Consequence: `--validate` becomes a **load-only** run. It now allocates VRAM and
pins host stores like generation does; it is GPU-exclusive and no longer
host-only.

## 3. Architecture

- `--validate` = thin wrapper over the engine lifecycle:
  `CreateModel(cfg)` (autodetect) → `Load()` → `Unload()`. Exit code 0 iff
  `Load()` returned true; non-zero on an undetectable/unloadable model.
- `apps/lsxfabric/src/validator.{h,cpp}` shrinks to one function
  `bool RunModelValidation(const std::string &model_path)`. All
  model-specific helpers are deleted: `LoadModelIndex`, `ModelInfo`,
  `ValidationResult`, `ValidatePinnedTensors`, `ValidateExpert`,
  `ValidateDenseNvfp4Schema`, `ValidateNVFP4Data`, `ValidateE4M3Scales`,
  `ValidateGlobalScale`, `RunValidator`.
- `apps/lsxfabric/src/main.cpp` keeps its shape; the `--validate` branch calls
  the new wrapper.
- `apps/lsxfabric/src/cli.{h,cpp}` is fixed (F4/F5).

## 4. CLI semantics (F4/F5)

- A helper `IsKnownFlag(const char *)` recognises
  `--localmodel`, `--validate`, `--testprompt`, `--max-gen`, `--prompt`.
- Value-taking flags consume the next token **only if** it exists and is not a
  known flag. Thus `--prompt --validate` leaves the prompt at its default and
  sets `validate`.
- `--max-gen` is parsed strictly (`strtol`, whole token consumed, value in
  `[1, INT_MAX]`); an invalid or missing value keeps the default `64`. This
  replaces `std::atoi`.
- Unknown tokens remain silently ignored (unchanged policy).

## 5. Finding disposition

| id | severity | resolution |
|---|---|---|
| F1 | High | Fixed: format knowledge comes from the engine registry + autodetect |
| F6 | High | Fixed: dense models load directly, no fabricated expert loop |
| F7 | Medium | Fixed: the family loader loads the experts it needs |
| F8 | High | Fixed: a missing declared artefact fails `Load()` |
| F12 | Medium | Fixed: the loaders validate their own schemas |
| F14 | Medium | Fixed: lsxfabric no longer maintains a pinned count |
| F4 | Medium | Fixed in `cli.cpp` (no flag swallowing) |
| F5 | Medium | Fixed in `cli.cpp` (strict `--max-gen`) |
| F2 | Low | Moot: `ValidateNVFP4Data` is deleted |
| F9 | Low | Moot: `RunValidator` summary is deleted |
| F10 | Low | Moot: `RunValidator` summary is deleted |
| F13 | Info | Moot: `LoadModelIndex` is deleted |
| F11 | Low | Deferred: the silent no-op mode in `main` is out of scope |

Related observation (out of scope): `lsxcommon::integrity::VerifyConvertedModel`
(used by `lsxforge --verify` and by converters' post-conversion check) has its
own stale format list and counts only `pinned/`; it rejects
`logestix-ornithmoe`, `logestix-qwen38flashnext`, `logestix-glm5next`,
`logestix-v41-nvfp4`. Not fixed here because the operator scoped the work to
`lsxfabric`.

## 6. Tests

TDD order: change the tests to the expected behaviour (they fail), then change
the code (they pass).

- `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`: replace the characterisations
  that encode the old parser with the F4/F5-correct expectations; drop the
  hidden `[.audit-spec]` tag (the cases now pass):
  - `--max-gen abc` → `64`; `--max-gen 0` → `64`; `--max-gen -5` → `64`;
    `--max-gen 12` → `12`.
  - `--prompt --validate` → default prompt and `validate == true`.
  - `--localmodel --validate /tmp/x` → default path and `validate == true`.
  - `--prompt "hello world"` → prompt set (non-flag value still consumed).
  - Repeated flags last-wins and unknown-token ignoring stay.
- `apps/lsxfabric/tests/test_validator.cpp`: rewrite to the wrapper contract —
  `RunModelValidation` returns `false` for a nonexistent/undetectable path,
  driven inside an `InferenceEngine::Session`. No real weights are loaded.
- `apps/lsxfabric/tests/test_validator_contract.cpp`: deleted.
- `apps/lsxfabric/tests/CMakeLists.txt`: drop the `test_validator_contract.cpp`
  source; `validator.cpp` stays in `test_lsxfabric`.
- `apps/lsxfabric/tests/test_lsxfabric.cpp`: unchanged (`Core` lifecycle).
- No new CTest target or label is introduced, so `test_ctest_labels_guard` is
  unaffected.

## 7. Integration evidence (GPU-exclusive, last)

- Negative: `lsxfabric --validate --localmodel /nonexistent` → non-zero exit.
- Positive, three representatives, sequential:
  `DeepSeek-V4-Flash` (MoE, default), `granite-4.2-30b` (dense),
  `Ornith-1.5-35B-A3B-MLX` (previously an unknown format to the old validator).
  A full 10-model sweep is resource-bound (each load pins host stores) and is
  documented as such.
- Generation smoke: `--testprompt` on the fast Gemma model still works.

## 8. Report update

`docs/LSXFABRIC_AUDIT.md` gains a "Remediation" section recording the
disposition table (§5), the architecture change, the new integration evidence,
and the `VerifyConvertedModel` related observation. The "how to run" section
drops the hidden-spec commands.

## 9. Risks

- `--validate` is now expensive and GPU-exclusive. Mitigation: keep it out of
  the fast suite; document.
- Engine autodetection keys off the `model` field via the registry; the sampled
  models must be registered. Mitigation: verify each sampled type.
- Deleting validator APIs is safe: `grep` shows no non-test caller.
- Removing `validator.cpp`'s Arrow/simdjson use does not require CMake link
  changes (kept minimal), since `lsxcommon` exports the Arrow include paths.

## 10. Out of scope

`VerifyConvertedModel` / `lsxforge`, F2 semantics (code deleted), F11, and any
new validator functionality.
