# lsxfabric — Functional Audit

Date: 2026-09-24
Branch: `audit/lsxfabric-functional-audit`
Commit: `1902de3`
Build: `linux-debug`, CUDA 13.4, RTX 5090 (32.6 GB)
Scope: CLI contract, validator primitives, index parsing, schema validators,
`RunValidator` accounting, runtime integration.
Non-goals: applying fixes, shared `RunCliGeneration` internals, performance.

## Method

Contract-first TDD at the audit commit `1902de3` (production code frozen). Green
characterisation tests pinned current behaviour; hidden Catch2 tests (tag
`[.audit-spec]`) encoded the expected contract where it diverged. CLI 3/3 and
validator 3/3 hidden specs failed as intended; the default suite stayed green.

The findings below were subsequently remediated — see **Remediation**. The
hidden `[.audit-spec]` tests no longer exist after remediation.

## Findings

| id | severity | surface | expected | actual | evidence | impact |
|---|---|---|---|---|---|---|
| F1 | High | `RunValidator` format recognition | recognise the formats `lsxforge` currently produces | allow-list is only `logestix-v1`/`-qwen`/`-glm`; **none** of the 10 vault formats matches | `[.audit-spec] recognises logestix-v1-nvfp4`; all 10 vault runs log `Unknown format '…', continuing anyway` | every real model is flagged unknown and classified by fallback |
| F2 | Low | `ValidateNVFP4Data` | detect malformed packed data | `lo > 0x0F \|\| hi > 0x0F` is never true for nibbles; always returns true | `validator: ValidateNVFP4Data accepts every byte value` | packed-data validation is a no-op |
| F3 | Info | `ValidateE4M3Scales` | consistent with the codec | requires a zero mantissa, matching `nvfp4::FloatToE4m3P2` | `validator: ValidateE4M3Scales accepts E4M3 P2 codes only` | by design; refuted as a defect |
| F4 | Medium | `ParseFabricCliArgs` | flags must not swallow other flags | `--prompt --validate` sets prompt `= "--validate"` and leaves validate off | `fabric cli [spec]: --prompt does not swallow a known flag` | a mistyped invocation silently changes mode |
| F5 | Medium | `ParseFabricCliArgs` | reject/clamp invalid `--max-gen` | `std::atoi`: `abc`→0, `-5`→-5 | `fabric cli [spec]: --max-gen …` | invalid generation length reaches the runtime |
| F6 | High | `RunValidator` dense path | dense formats need no per-expert files; their NVFP4 tensors should be schema-validated | dense `logestix-granite` falls into the expert loop (64 `Expert not found` warnings) and its `*_nvfp4.arrow` files are never schema-checked | `[.audit-spec] dense formats need no expert files`; vault `granite-4.2-30b`: 65 warnings, PASS | corrupt dense quantized tensors pass validation; real warnings buried |
| F7 | Medium | `RunValidator` expert coverage | validate the declared expert set | checks only `experts/layer_N/exp_N.arrow` — `num_layers` files out of `total_experts` | vault `DeepSeek-V4-Flash`: `Experts checked: 43` of 11008; `GLM-5.3-Flash`: `0` of its experts (different layout) | corruption outside that 1-per-layer set is never seen |
| F8 | High | `RunValidator` missing files | a declared-but-missing expert is an error | emits a warning, `ok == true` (PASS) | `[.audit-spec] a missing declared expert is an error`; vault MoE runs PASS with `Expert not found` warnings | incomplete models report PASS |
| F9 | Low | `RunValidator` summary | report the loaded pinned count | prints the declared `total_pinned` | vault `Ornith`: declared 613, loaded 413, summary prints `Pinned tensors checked: 613` | misleading summary |
| F10 | Low | `RunValidator` summary | count dense-schema tensors | the `DenseNvfp4Schema` branch never increments `experts_checked` | code review | `Experts checked` is incomplete for qwen/glm |
| F11 | Low | `main` | report an inactive mode | no `--validate`/`--testprompt` runs nothing and returns 0 silently | integration no-op: `rc=0`, only header logs | a typo'd flag looks successful |
| F12 | Medium | `ValidatePinnedTensors` | validate pinned column schema | opens/reads/counts rows only | code review; vault dense `_nvfp4.arrow` files are read but never schema-checked | mistyped pinned shards pass |
| F13 | Info | `LoadModelIndex` | strict parsing | absent/mistyped fields silently default to 0/empty | `validator: LoadModelIndex defaults …` | a malformed index degrades instead of failing |
| F14 | Medium | `ValidatePinnedTensors` count | loaded == declared pinned count | 3 vault models mismatch (warning only) | `Ornith`/`Qwen3.6` 613 vs 413; `Qwen3.8` 1067 vs 821 | the declared count and the validator's accounting disagree, silently tolerated |

## Remediation

Root cause: `validator.cpp` reimplemented model knowledge (format tags,
expert/pinned/`shared`/`router`/`ple` layouts, schemas) that the inference
engine already owns in its registry and per-family loaders. The fix removes the
duplication rather than extending the lists.

`--validate` is now a thin wrapper over the engine load path:
`InferenceEngine::CreateModel` → `IModel::Load` → `Unload` (`RunModelValidation`
in `apps/lsxfabric/src/validator.cpp`). `validator.cpp`'s duplicated model logic
was deleted; `cli.cpp` gained a strict parser.

| id | status | note |
|---|---|---|
| F1 F6 F7 F8 F12 F14 | Fixed | delegated to the engine load path: no format allow-list, no fabricated expert loop, no pinned accounting |
| F4 F5 | Fixed | `cli.cpp`: a value flag never swallows a known flag; `--max-gen` is parsed strictly (`abc`/`0`/`-5` keep the default 64) |
| F2 F9 F10 F13 | Moot | the code that carried them was deleted |
| F11 | Deferred | the silent no-op mode in `main` is out of scope |

Follow-up (done after the audit): `lsxcommon::integrity::VerifyConvertedModel` —
the separate host-only converted-model verifier used by `lsxforge --verify` and
by the converters' post-conversion check — was itself stale (its format list
rejected `logestix-ornithmoe`, `logestix-qwen38flashnext`, `logestix-glm5next`,
`logestix-v41-nvfp4`, and it counted only `pinned/`). It was therefore removed,
together with its callers (`lsxforge --verify` and the final `vres` checks in
the converters). `--validate` (engine load) is now the single converted-model
validation path. The source-check helpers (`VerifyAllTensorReads`,
`CheckTensorCoverage`, `CheckLayerContiguity`, `CheckGeometryFields`) and
`BuildAndVerifyTokenizer` remain — they belong to conversion, not validation.

### Remediated integration evidence

Environment: RTX 5090 (32.6 GB), CUDA 13.4, `linux-debug`. Every `--validate`
run is a real model load, GPU-exclusive, run sequentially.

Negative case:

```
lsxfabric --validate --localmodel /nonexistent/model   →   rc=1
  Running validation...
  Validation failed: cannot create a model for /nonexistent/model
```

Positive sweep (one representative per shape; a full 10-model sweep is
resource-bound because each load pins host stores):

| model | old validator | new `--validate` |
|---|---|---|
| DeepSeek-V4-Flash (MoE, default) | unknown format warning, PASS | `rc=0` — `pinned weights: 13534 MiB in 901 ms` |
| granite-4.2-30b (dense) | 65 spurious warnings, PASS | `rc=0` — `pinned weights: 15706 MiB in 1610 ms`, 64 layers loaded |
| Ornith-1.5-35B-A3B-MLX (MoE, previously unknown format) | warnings, PASS | `rc=0` — `layer_39: 256 experts`, `loaded: resident=1409135616 bytes` |

Generation smoke (unchanged path):

```
lsxfabric --localmodel .../gemma-4-26B-A4B-it --testprompt --max-gen 32
  → rc=0, "The capital of France is **Paris**."  (32 tokens @ 127.1 t/s)
```

## Test inventory

| file | test | tag | status |
|---|---|---|---|
| test_fabric_cli_edges.cpp | flag without a value is ignored | `[lsxfabric][cli]` | green |
| test_fabric_cli_edges.cpp | a value flag does not swallow a known flag | `[lsxfabric][cli]` | green (F4) |
| test_fabric_cli_edges.cpp | non-numeric/non-positive `--max-gen` keeps the default | `[lsxfabric][cli]` | green (F5) |
| test_fabric_cli_edges.cpp | a non-flag value is still consumed | `[lsxfabric][cli]` | green |
| test_fabric_cli_edges.cpp | repeated flags, last one wins | `[lsxfabric][cli]` | green |
| test_fabric_cli_edges.cpp | help flags are unknown and ignored | `[lsxfabric][cli]` | green |
| test_fabric_cli.cpp | defaults / unknown measurement flags / kept flags | `[lsxfabric][cli]` | green |
| test_lsxfabric.cpp | Core init lifecycle | `[lsxfabric]` | green |
| test_validator.cpp | `RunModelValidation` rejects an undetectable path | `[lsxfabric][validate]` | green |

The audit-time green/red unit inventory (validator primitives, schemas,
`RunValidator` accounting, hidden `[.audit-spec]` specs) was removed by the
remediation: the functions it exercised no longer exist.

## Integration evidence (audit-time, pre-remediation)

Environment: RTX 5090 (32.6 GB), CUDA 13.4, `linux-debug`. This section records
the original host-only `--validate` behaviour of `validator.cpp`; the remediated
evidence is in **Remediation** above.

### `--validate` over every converted vault model

Every run returned `errors=0` and `Result: PASS`, including the unknown-format
warning for all ten:

| model | format | result | errors | warnings | experts checked |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | `logestix-v41-nvfp4` | PASS | 0 | 1 | 40 |
| DeepSeek-V4-Flash | `logestix-v1-nvfp4` | PASS | 0 | 1 | 43 |
| gemma-4-26B-A4B-it | `logestix-gemma` | PASS | 0 | 1 | 30 |
| GigaChat3.5-432B-A28B-Reasoning | `logestix-gigachat-nvfp4` | PASS | 0 | 4 | 37 |
| GLM-5.3-Flash | `logestix-glm5next` | PASS | 0 | 46 | 0 |
| granite-4.2-30b | `logestix-granite` | PASS | 0 | 65 | 0 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | `logestix-nemotron-nvfp4` | PASS | 0 | 30 | 23 |
| Ornith-1.5-35B-A3B-MLX | `logestix-ornithmoe` | PASS | 0 | 2 | 40 |
| Qwen3.6-35B-A3B | `logestix-qwen36moe` | PASS | 0 | 2 | 40 |
| Qwen3.8-Flash-Next | `logestix-qwen38flashnext` | PASS | 0 | 2 | 48 |

Representative lines:

```
DeepSeek-V4-Flash: Unknown format 'logestix-v1-nvfp4', continuing anyway
                  Pinned tensors checked: 2881 | Experts checked: 43 | Result: PASS
granite-4.2-30b:  Unknown format 'logestix-granite', continuing anyway
                  Expert not found: .../experts/layer_0/exp_0.arrow   (x64)
                  Result: PASS
Ornith:           Expected 613 pinned tensors, loaded 413
                  Pinned tensors checked: 613 | Experts checked: 40 | Result: PASS
```

`DeepSeek-V4-Flash` declares `total_experts=11008` and reports
`Experts checked: 43` (F7). `granite-4.2-30b` is dense (`num_layers=64`) but is
routed through the expert loop (F6).

### No-op mode (S2 / F11)

`lsxfabric --max-gen 1` → exit 0, no validation and no generation; only the
banner and `Model path:` line are logged.

### Real generation

- Default path (no `--localmodel`), `--testprompt --max-gen 64`: exit 0,
  `gen_tokens: 65`, `22.5 t/s`, output *" Paris. It is one of the most famous
  cities…"*.
- `--localmodel …/gemma-4-26B-A4B-it --testprompt --validate --max-gen 64`:
  exit 0, `64` tokens at `141.0 t/s`, no `Running validation…` line — confirms
  `--testprompt` takes precedence over `--validate` (S2).

## Conclusions

- The generation and dispatch surfaces (S1 kept flags, S2) behave as
  documented; generation works for the default model and for a fast MoE model.
- The validator is the weak surface. Its format allow-list is entirely stale
  (F1) and its model of expert layout is too shallow (F6, F7), so **every**
  real model reports `PASS` regardless of content (F1, F6, F8, F14). The
  primitives themselves are internally consistent (F3 refuted).
- Recommended follow-ups (not applied at audit time): update the format
  allow-list from the `logestix-*` format tags and classify dense vs MoE from
  the manifest; validate `ValidateDenseNvfp4Schema` for all dense
  `*_nvfp4.arrow` files; validate the full declared expert set; promote missing
  declared artifacts to errors; fix the CLI value handling (F4/F5).
- Resolved differently (see **Remediation**): rather than extending the
  duplicated lists in `validator.cpp`, validation was delegated to the engine's
  load path, and the CLI was fixed. F2/F9/F10/F13 became moot; F11 remains
  deferred; `VerifyConvertedModel` remains stale and out of scope.
