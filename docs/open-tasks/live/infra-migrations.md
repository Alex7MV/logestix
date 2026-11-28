# Infrastructure — build wiring, Arrow IPC, manifest gaps

### S. Arrow IPC residuals (2026-11-01)

Nothing here is a correctness defect: the audit closed A-01..A-07 and the
full-vault run (29 trees, 310 158 files) is green. These are measured-cost
decisions, not yet-worth-it work.

#### S1. Multi-expert per-file IPC packing — measure first
- **Open, instrument-first.** Packing N experts into one IPC file (N record
  batches, `ReadRecordBatch(i)`) divides inode/mmap/open/footer churn by N
  (~23k expert files per large tree) and changes nothing about the payload
  bytes read. It requires a format change plus re-conversion of all 29 trees
  (~4 TB) and pack-granular LRU bookkeeping.
- **Before any of that:** measure the `ExpertStore::ParseFile` share of the
  per-token cost on a high-miss model (DeepSeek-V4-Pro streams ~13.5 GB/token;
  Kimi does 277 misses/token). Migrate only if parse is material (>=10-20% of
  the step); for resident families it is already a one-time load cost.

#### S2. Write-path wall-time A/B
- **Open, low value.** The copy elimination is pinned by a deterministic
  tracked-pool budget (peak 2x -> 1x payload; 0 growing reallocations), which
  is stronger than a noisy timing. A wall-time A/B needs a second build of the
  pre-change commit in a worktree; do it only if a converter run looks
  memcpy-bound.

#### S3. Windows: atomic artifact writes
- **Open, platform-gated.** `WriteRecordBatchToFile` writes a temp file and
  renames; on Windows `std::filesystem::rename` cannot replace an existing
  file, so a remove-then-rename fallback is compiled under `_WIN32`. It has
  never run on the Windows preset (this host is Linux).

#### S4. Shared-expert size contract is not validated
- **Open, hardening.** `SharedExpertSchema` carries no encoding tag. Measured
  reality: DeepSeek-V4-Pro/V4-Flash store shared experts as raw F8_E4M3 with
  E8M0 128x128 tile scales (32 768 elements/scale), the NVFP4 families as
  per-16 E4M3. The schema-level check was removed on 2026-11-01 because it
  rejected the real trees; only offsets/ranges/global-scale are validated.
  A truncated shared-expert payload is therefore caught late (or not at all).
- **Fix shape:** write the shared-expert encoding/block into
  `model.index.json` (e.g. `"shared_expert": {"encoding": "nvfp4", "scale_block": 16}`
  or `{"encoding": "fp8_e8m0_tile", "tile": 128}`) and have
  `load_shared_experts_metadata` / `LoadSharedExpertBatchViews` enforce it.

#### S5. Routed-expert proportional truncation
- **Open, hardening.** `ExpertSchema` has no `shape`, so
  `CheckExpertScaleCount` infers `numel = packed*2` from the payload itself.
  It rejects an inconsistent truncation (packed and scales disagree) but not a
  *proportional* one (both halved). The loader knows the frozen geometry
  (`I`/`H` per family) and could cross-check `packed_bytes == ceil(I*H/2)`.
- **Fix shape:** per-family geometry check in the loader (no format change), or
  store the logical shape in the expert record (format change).

#### S6. Full-vault audit is manual
- **Open, tooling.** `test_arrow_tree_integrity` skips without
  `LSX_ARROW_AUDIT_ROOT` (a SKIP is not a pass), and the exhaustive
  `LSX_ARROW_AUDIT_FULL=1` run (29 trees, ~4 TB, 783 s) exceeds the `convert`
  label's 600 s cap, so it cannot live in CTest as-is. `lsxforge --verify` runs
  the same verifier but is a manual post-conversion step.
- **Fix shape:** a nightly/job that mounts the vault and runs the full verifier,
  plus wiring `--verify` into the conversion wrapper so a converted tree is
  never published unverified.

#### S7. The audit-matrix coverage rule is keyed off a hand-written list
- **Open, instrument.** `test_arrow_usage_guard` enforces R-A8 ("every adapted
  model has a row") as `by_model[model] == 1` for each entry of `kModels` plus
  `rows.size() == kModels.size()`. Both sides are hand-maintained, so a model
  present in **neither** is invisible and the rule silently passes over it —
  which is exactly what happened to `Kimi-K3.1-Pro` until 2026-11-05 (added in
  `0f919929`, after the variant had already landed with real-weight gates).
  The finding cost one audit, not a defect: the guard was not wrong about the
  rows it knew.
- **Fix shape:** derive the model set from a machine-checked registry instead of
  restating it — `test_model_suite_guard.cpp::Suites()` already enumerates every
  ingested model and is extended in the same commit as the model files, so the
  Arrow guard could assert `kModels ⊇ Suites()` (name mapping aside) and stop
  depending on a second list. Until then: adding a model means touching both,
  and the audit's own coverage claim rests on the list being complete.

### M. Test-registration migration (CMake boilerplate, 2026-11-03)

`cmake/LsxTestHelpers.cmake` now provides `lsx_add_test()` /
`lsx_add_cuda_test()` / `lsx_copy_arrow_dlls()`; `LSX_CUDA_ARCH` is the single
device-codegen arch and `lsx_dep_includes` is the only dependency-include
propagator (no `${CMAKE_BINARY_DIR}/_deps/...` at a call site).

Done in the first pass (verified by `ctest --show-only=json-v1` property
equality: 493 tests, 0 name/property diffs, plus a full build and the `fast` +
kernel-inference suites): 212 Windows Arrow-DLL blocks replaced by
`lsx_copy_arrow_dlls`; 130 `_deps` call-site paths removed (37 include blocks
now link `lsx_dep_includes`); the 14-test qwen CUDA loop and the 10 guard tests
use the helpers.

**Remaining, mechanical:** ~390 host registrations still spell out
`add_executable` / `target_link_libraries` / `add_test` /
`set_tests_properties`. Migrate them family by family in the same commit as
their next change, keeping the JSON property diff at zero (the helper already
covers LINK/INCLUDE/DEFINE/LABEL/TIMEOUT/SKIP/ENV and the DLL copy; add a
property to the helper only when a target genuinely needs one, e.g.
`RESOURCE_LOCK`).

### V. Codec profiles: the store supports N per model, no CONVERTER can produce the second one (2026-11-28; selection half closed 2026-11-28)

The hardware-native codec contract (branch `feat/lsxstorage-hardware-codec-contract`)
ships the STORAGE half of invariant I4 — `<ARTIFACT_ROOT>/<org>/<model>/<profile>/`,
one artifact per profile, `artifact.fb` per artifact — and the load gate names the
other profiles of a model when it refuses the current one. The selection half is
closed (below); ONE half remains open, found by the bootstrap conversion of real
weights (plan Task 9a step 4):

- **CLOSED 2026-11-28 (branch `build/ampere-aot-sm86-sm87`, plan Task 15): the
  converters know what a codec profile is.** `CodecProfile`, its spellings and one
  predicate (`ProfileUsesNvfp4Weights`) moved to
  `libs/lsxcommon/include/lsxcommon/codec_profile.h`, `ConvertPhiModel` /
  `ConvertQwenModel` take the profile and fold it into their existing
  `*_tensor_is_nvfp4(kind)` decision, and `lsxforge` passes the profile it parsed
  instead of only validating it. Measured: `microsoft/phi-4/bf16` (mask 2, 43
  planes) and `Qwen/Qwen3.5-9B/bf16` (mask 3, 35 planes), both `--verify` green,
  both registered in their manifests. **The remaining half is NOT this item** —
  the artifact now exists but no loader can read it: see `A3.13`
  (`docs/open-tasks/live/engine-defects.md`).
- **CLOSED 2026-11-28 (branch `build/ampere-aot-sm86-sm87`): `Resolve()` now
  SELECTS instead of guessing.** `lsx::storage::SelectProfile(org, model_dir,
  supported_mask, preference)` keeps the storage layer codec-agnostic by taking
  the capability as a mask and the precedence as an ordered name list;
  `lsxcommon` supplies both (`SupportedCodecs(ActiveTarget())` and
  `hw::CodecProfilePreference()` = int8-w8a16, nvfp4, mxfp8, bf16 — the single
  place spec §8.2 corrects once Branch 3 has measured). `Resolve(path, supported,
  preference)` applies it to a MODEL and never re-ranks an explicit profile path.
  `lsxfabric --explain-artifact` prints the target mask, every candidate's mask,
  rank and verdict, then the choice, without loading a model. Tests:
  `test_storage_select_profile` (6 cases) and `test_explain_artifact` (3).

Measured on the two bootstrap artifacts (both single-profile, both green): phi-4
`microsoft/phi-4/nvfp4` mask 34 (bf16|nvfp4), Qwen3.5-9B `Qwen/Qwen3.5-9B/nvfp4`
mask 35 (f32|bf16|nvfp4). Evidence and the gate verdicts:
`.superpowers/sdd/2026-11-24-ampere-hardware-native-codec-plan/progress.md`
(Task 9a) and `docs/SELFCHECK_MATRIX.md` §1.

### W. A checkpoint in the store that no registry row describes (2026-11-28)

`<LSX_SOURCE_ROOT>/baidu/ERNIE-5.5-Flash-Base-BF16` is a complete download (128
safetensors shards, `config.json`) and there is no
`models/baidu/ERNIE-5.5-Flash-Base-BF16.json`. The store is org-nested, so this is
not a layout defect — it is a model the registry cannot describe: no converter, no
passport, no geometry, therefore no row (`status: not-built` requires the row's
architecture fields, and inventing them would claim an ingestion that does not
exist).

Measured 2026-11-28: 23 org directories on disk, 22 in the registry; 39
checkpoints with weights on disk, 2 with a built artifact.

**Fix shape:** either ingest it (Stage 0 triage in `ingesting-a-model`) or evict
it, and add the reverse-direction check — "every checkpoint under the source root
belongs to a row, or is declared as an uningested download" — to
`test_model_suite_guard`, so the store and the registry cannot drift silently.

### Model manifest migration (2026-11-06) — what it did not cover

The registry moved from six hand-maintained projections to `models/<org>/<model>.json`.
Closed: the guard registries, the ctest registration, the factory list, the
parity-gate and selfcheck test files, the `AGENTS.md` model narrative
(1109 -> 164 lines) and the `README.md` inventory.

Still open, by construction:

- **Build wiring is still per model.** `add_executable`, the CUDA OBJECT
  libraries and per-target includes/defines stay in `tests/CMakeLists.txt`: only
  10 of 421 targets follow a convention, so a generated build graph would be a
  worse copy of the same information. Revisit only if a re-architecture makes
  the wiring conventional.
- **`format_tag` is declared for 8 of 35 models.** The rest have no
  machine-readable source (no `kFormatTag` constant; the tag lives inside a
  converter's index writer). The catalogue prints "not declared" rather than a
  guess. Closing it means exposing each converter's tag as a constant.
- **Per-model goldens and perf rows are still captured by hand.** The manifest
  names them and records debt; it cannot produce them.
- **Approach C (family scaffolder) is not built.** Decided after this work, on
  the measure "lines a fresh family needs that are not converter/kernel logic".

#### Stage-9 reports missing for 20 models

`docs/MODELS.md` links every model's report where one exists (15 of 35) and prints
**not written** where it does not — the line is stated rather than omitted, so a
missing report cannot read as "not applicable". The 20 without one:

`deepseek-v4-flash`, `deepseek-v4-pro-0813`, `deepseek-v4.1-flash`, `flint`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `glm-5.2`, `glm-5.3-flash`, `hy-memory-30b-a3b`, `kimi-k2.7-code`, `ling`, `mimo-v2.6-flash-rl`, `minimax-m3`, `nemotron-3.5-lightning-30b-a3b`, `ornith-1.5-35b-a3b`, `phi-4`, `qwen3.5-35b-a3b`, `qwen3.5-9b`, `qwen3.6-35b-a3b`, `xing4.0-29b-a4b`

The gap is not cosmetic: for these models the per-model *reasoning* (why a family
pins `FastMathMode::kOff`, why its residual plane exists, what the measured
bottleneck is) lives only in `AGENTS.md`'s git history and in `docs/open-tasks/`
sessions, not in a document a reader can be pointed at. `models/<org>/<model>.json:report`
now carries a validation rule — a declared path must exist — so writing one is a
one-line manifest change.

#### The ctest oracle is a dated snapshot, not a live target list

`tools/dump_ctest_manifest.py` and `tests/fixtures/ctest/2026-11-06-baseline.json`
(511 targets captured from the pre-migration `linux-debug` tree) are kept as the
migration's evidence, **not** as a gate and not as documentation of the current
target set.

- **What they prove.** `python3 tools/gen_model_manifest.py --root . --diff-cmake
  tests/fixtures/ctest/2026-11-06-baseline.json` compares the emitted model
  registration against what CMake produced *before* the manifests existed: 421 of
  the 511 targets are model targets (the other 90 are guard/kernel/infrastructure
  tests). Today it prints `compared 421 targets, 0 difference(s)`, which is the
  only record that the transcription into `models/<org>/<model>.json` was exact rather than
  approximately exact.
- **CI never runs it, and should not.** The snapshot describes the tree on
  2026-11-06. The first model added after that date makes the check report
  `manifest target absent from the baseline` and exit 1 — verified by deleting
  one target from a copy of the baseline, which yields
  `compared 420 targets, 1 difference(s)` and exit 1. That is expected growth,
  not a regression, so the check cannot live in the suite.
- **Do not re-capture it in place.** Re-dumping today yields 511 targets again, but
  not the same ones: `test_parity_selfcheck_naming_guard` is gone and
  `test_manifest_generator_selftest` (this migration's own self-test) is new.
  The model-target projection happens to be identical (421, 0 differences), so
  overwriting the file would look harmless and would silently destroy the only
  record of the tree as it was *before* the manifests. A snapshot is meaningful
  only as the "before" of a *future* migration, captured before its first edit.
- **Live authority is elsewhere.** If a current registration check is ever
  needed, it is the generated `model_targets.cmake` plus `test_manifest_sync`,
  not this JSON. The fixture answers "did the migration lose a target or change
  a label/timeout/env?", a question that has one date.
