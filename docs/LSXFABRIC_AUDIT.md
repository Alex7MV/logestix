# lsxfabric — Functional Audit

Date: 2026-09-24
Branch: `audit/lsxfabric-functional-audit`
Commit: `1902de3`
Build: `linux-debug`, CUDA 13.4, RTX 5090 (32.6 GB)
Scope: CLI contract, validator primitives, index parsing, schema validators,
`RunValidator` accounting, runtime integration.
Non-goals: applying fixes, shared `RunCliGeneration` internals, performance.

## Method

Contract-first TDD. Green characterisation tests pin current behaviour; hidden
Catch2 tests (tag `[.audit-spec]`) encode the expected contract where it
diverges. The production code was not modified.

Run the hidden (red) spec set explicitly:

```bash
./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"
./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric "[.audit-spec]"
```

Result at this commit: CLI 3/3 fail; validator 3/3 fail (8 assertions, 5 pass,
3 fail). The default suite stays green.

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

## Test inventory

| file | test | tag | kind | surface |
|---|---|---|---|---|
| test_fabric_cli_edges.cpp | flag without value ignored | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | `--prompt` consumes flag token | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | non-numeric `--max-gen` → 0 | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | negative `--max-gen` verbatim | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | repeated flags last wins | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | flag-looking value literal | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | help flags ignored | `[lsxfabric][cli]` | green | S1 |
| test_fabric_cli_edges.cpp | `--max-gen` rejects non-numeric | `[.audit-spec]` | red | F5 |
| test_fabric_cli_edges.cpp | `--max-gen` positive | `[.audit-spec]` | red | F5 |
| test_fabric_cli_edges.cpp | `--prompt` no swallow | `[.audit-spec]` | red | F4 |
| test_validator_contract.cpp | `ValidateNVFP4Data` all bytes | `[validator]` | green | S3 |
| test_validator_contract.cpp | `ValidateE4M3Scales` P2 | `[validator]` | green | S3 |
| test_validator_contract.cpp | `ValidateGlobalScale` finite>0 | `[validator]` | green | S3 |
| test_validator_contract.cpp | `LoadModelIndex` full/sparse/malformed | `[validator]` | green | S4 |
| test_validator_contract.cpp | `ValidateExpert` (ok/cols/rows/global/missing) | `[validator]` | green | S5 |
| test_validator_contract.cpp | `ValidateDenseNvfp4Schema` | `[validator]` | green | S5 |
| test_validator_contract.cpp | `ValidatePinnedTensors` skip | `[validator]` | green | S5 |
| test_validator_contract.cpp | `RunValidator` index/unknown/experts/pinned | `[validator]` | green | S6 |
| test_validator_contract.cpp | recognises `logestix-v1-nvfp4` | `[.audit-spec]` | red | F1 |
| test_validator_contract.cpp | dense formats need no experts | `[.audit-spec]` | red | F6 |
| test_validator_contract.cpp | missing expert is an error | `[.audit-spec]` | red | F8 |

## Integration evidence

Environment: RTX 5090 (32.6 GB), CUDA 13.4, `linux-debug`. `--validate` is
host-only; the two generation runs were executed sequentially (GPU-exclusive).

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
- Recommended follow-ups (not applied here): update the format allow-list from
  the `logestix-*` format tags and classify dense vs MoE from the manifest;
  validate `ValidateDenseNvfp4Schema` for all dense `*_nvfp4.arrow` files;
  validate the full declared expert set; promote missing declared artifacts to
  errors; fix the CLI value handling (F4/F5).
