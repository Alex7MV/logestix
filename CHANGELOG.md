# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-11-28

A converted tree is no longer "whatever a converter happened to leave in a
directory": it is a declared, verifiable artifact, admitted through a hardware
codec gate, living in a store whose layout mirrors the checkpoint's. MCP joins
SelfCheck / FSI / CODE / DET as a required lifecycle gate, and
`gemma-4-12B-it` is the 48th registered model.

**Breaking.** The registry layout, the artifact location and the environment
contract all changed, and every pre-existing converted tree was deleted, so
trees must be re-converted. The breaking items are marked below; the practical
summary is that `LSX_SOURCE_ROOT` and `LSX_ARTIFACT_ROOT` are now **required**
and both stores are org-nested.

### Added

**Storage and the artifact contract**

- `libs/lsxstorage` — a new layer **below** `lsxcommon` (links FlatBuffers +
  spdlog only). The load gate lives in `ModelBase::BeginLoad` and must read
  `artifact.fb`, so the artifact reader cannot depend on the model layer.
  Owns the two roots, `resolve.h`, the artifact reader/writer/verifier, and an
  in-tree SHA-256 (FIPS 180-4, pinned against the standard vectors including the
  1M-`a` streaming case) rather than a new OpenSSL dependency for one hash.
- `artifact.fb` (`libs/lsxstorage/schema/artifact.fbs`) — the single source of
  truth for what an artifact **contains**: codec mask, planes, per-layer codec
  map, tokenizer copy and provenance. `model.index.json` keeps geometry only.
  The reader copies into plain structs, so no consumer — and no `.cu` TU
  reaching `ModelBase` — ever includes FlatBuffers. Five refusal cases
  (`kMissing`, `kUnreadable`, `kMalformed`, `kBadVersion`) instead of a silent
  misread. `flatc` is now built and the header generated at build time, so a
  schema edit cannot leave a stale header behind.
- `VerifyArtifact` — rules D1-D5, each one a defect that would otherwise ship
  silently: a weight plane spelled `U8` is rejected naming `NVFP4`; `.scale` /
  `.scales` / `.global_scale` are companions and add no bit; `role=meta` adds no
  bit; a meta plane holding weight-like tensors is rejected; the mask is a
  **union, never a sum**. Plus: every declared plane exists, every file is
  declared or a sanctioned sidecar, a directory the load path cannot reach is a
  defect, `layer_codec` matches `layer_count`, and `deep` mode re-digests every
  plane. This is the only place a codec is inferred, and it exists to police the
  declaration; the runtime never infers.
- Emitter with self-verify and atomic publish (`tmp` + rename): it recomputes
  `used_codecs` from content rather than trusting the declaration, then
  re-reads and re-verifies what it wrote, because with every legacy tree deleted
  there is no reference artifact left to catch a converter bug. `PreflightDisk`
  refuses a conversion that cannot fit **before** writing a tree that would then
  be refused for a reason that reads like a codec bug.
- `--codec-profile` and the codec-profile table: `nvfp4` (NVFP4 weights,
  BF16/F32 rest -> sm_120/121), `bf16` (-> every target), `int8-w8a16`
  (declared, refused before any work — no quantizer yet). Validation happens
  before a byte is read; an unknown, unimplemented or root-less profile exits 2
  naming the remedy. `ScanArtifactTree` derives the planes **from the bytes** and
  the profile cross-checks them — two independent inputs, so agreement is
  evidence rather than a restatement.
- `ModelInitConfig::synthetic` — an explicit exemption for unit fixtures that
  build their own Arrow trees. Deliberately not an environment variable: the
  gate must not be bypassable by anything a shipped binary can set. Declared
  last, because the struct is brace-initialised positionally in several TUs.
- Per-layer seam for the codec-mixing families (`artifact()`,
  `LayerUsesCodec()`), cleared by `Unload`/`AbortLoad` so a stale map cannot
  outlive the weights it describes.

**Lifecycle gates**

- `MCPSelfCheck` (`--mcp-selfcheck`, `LSX_MCP_GOLDEN_DIR`,
  `libs/lsxcommon/include/lsxcommon/mcp_selfcheck.h`) — the fourth parity gate,
  on the operator's three validation prompts. It departs from its siblings in
  one deliberate way: FSI/CODE/DET are *model*-defined, so text equality is
  parity, while an MCP payload is *protocol*-defined — JSON-RPC 2.0 fixes the
  fields and says nothing about key order, so comparing bytes would measure
  whether the model guessed the template's layout. The gate extracts the first
  balanced object carrying the case anchor, minifies it, sorts keys and compares
  **documents**; `exact` records byte-identity and gates nothing. Measured, not
  assumed: `phi-4` (inside a JSON fence) and `qwen3.5-9b` (bare) both pass
  3/3; a golden with every `expected` re-keyed in reverse still passes with
  `exact=false`; `-32601` substituted for `-32602` fails with
  `diff='error.code: -32601 != -32602'`.
- MCP is a **required** stage, not a pilot: `OPTIONAL_STAGES = ("perf",)`,
  `mcp` in `test_model_suite_guard`'s `kRequired`, 48
  `lsx_add_parity_gate(... KIND mcp)` lines (45 new), 39 goldens captured by
  `tools/mcp_capture_all.py`. The remaining 9 rows are empty source-store
  placeholders and declare `golden_debt.mcp` — the same treatment the other
  gates get there, so a gap is never silent.
- `tools/mcp_capture.py`, `tools/mcp_capture_all.py`, `docs/MCP_PROMPTS.md`,
  `docs/MCP_SELFCHECK_FINDINGS.md`, the design spec, the rollout archive
  (`docs/open-tasks/2026-11-28-mcp-gate-rollout-closed.md`) and the
  `mcp-selfcheck` skill.

**Guards** — each with a non-vacuity case and each watched red first

- `test_path_shape_guard` — nothing in `libs/`, `apps/`, `tests/` or `tools/`
  builds a pre-split `logestix/<subdir>` path, `lsx_model_models_dir()` is
  called only from the resolver and the store layer, and no Python tool names a
  store root at all.
- `test_index_layout` — `layout` is present, the retired per-model tag must not
  come back, and golden readers are exempted by **class** rather than by file
  name (a file-name list is how the fourth gate tripped it on landing day).
- `test_index_fields` — a registry of the fields whose absence is a hard load
  failure, guarded in three narrow directions and failing **closed**. It
  immediately caught two required keys a manual audit had missed, which is the
  argument for a registry over a scan (four successive scans produced four
  different dead-key lists).
- `test_storage_roots`, `test_storage_artifact`,
  `test_storage_artifact_writer`, `test_storage_verify`,
  `test_load_codec_gate`, `test_hardware_target`, `test_bootstrap_artifacts`
  (the contract on real bytes: deep re-digest, self-sufficiency, admission on
  `sm_86`/`sm_87`/cpu versus `sm_120`/`sm_121`), `test_model_identity`,
  `test_forge_codec_profile`, `tools/test_lsx_paths.py`.

**Models**

- `gemma-4-12B-it` — 48 layers / 3840 hidden / 16 Q / 8 sliding + 1 global KV /
  15360 MLP, its own passport and `kType` over the shared dense body. Real
  weights: SelfCheck `**Paris**` x3 bit-exact `31d7a1672d89ccdc`, CODE 3/3,
  DET 3/3, MCP 3/3 byte-exact, FSI 2/3 (see *Known issues*). Throughput
  59.85 t/s auto / 48.77 t/s bit-exact at 2048 tokens, TTFT ~283 ms, 12.0 GB
  resident.
- `convert_residual_plane` as a **per-model manifest field**. One converter now
  serves both dense Gemma-4 sizes and the second NVFP4 plane's cost is
  proportional to the model: +5.7 GB at 12B (12.0 GB total, fits a 32 GB card)
  versus ~33.7 GiB at 31B, which does not. The value travels manifest ->
  generator -> `ModelRow` -> `ModelIdentityFor` -> `GemmaDenseConvertOptions`,
  and the converter **records what it wrote** in `model.index.json`
  (`"residual": "attn+mlp" | "none"`), so an index cannot claim a plane the tree
  does not hold.
- Placeholder-directory recognition (`ListPlaceholderDirs`): an evicted
  checkpoint's empty directory now says "the checkpoint is not downloaded"
  instead of advising the operator to check a spelling that is correct. Store:
  48 directories over 23 orgs — 39 checkpoints + 9 placeholders.

**Docs**

- `docs/model-reports/2026-11-28-Gemma-4-12B-IT.md`, the codec-contract design
  and its 25-task plan, the MCP design spec, and the reference to the fourth
  gate in the repo map.

### Changed

- **Breaking:** the model registry mirrors the store —
  `models/<org>/<model>.json` with `vault { org, dir, env }`,
  `preferred_profile`, and `profiles { <name>: { status, artifact, codecs,
  codec_mask, target, gates } }`. 48 rows over 22 orgs. Two machine-checked
  rules keep it evidence rather than decoration: `built` **requires** a 64-hex
  digest, a positive byte count, codecs, a hardware target and a dated verdict
  per gate; `not-built` may carry **nothing but notes**, so a verdict can never
  outlive the tree it described. The other 45 rows are `not-built` by fact.
- **Breaking:** `ModelRoot()` resolves the **artifact**
  (`<artifact root>/<org>/<dir>/<profile>`, env override still winning), not the
  checkpoint. This is what makes the registry true: the whole real-weight row
  for a model now runs with **no per-model environment variable set**, which was
  previously impossible.
- **Breaking:** the artifact path mirrors the checkpoint's, so moving between
  them means changing only the root —
  `<source root>/<org>/<model>` versus
  `<artifact root>/<org>/<model>/<profile>`. The profile directory is the
  artifact; the `logestix/<subdir>` segment is gone from 538 sites in 226 files.
- `ModelBase::BeginLoad` runs four stages instead of loading the tokenizer
  directly: `RESOLVE -> DECLARE -> GATE -> REBIND`. The ordering is the
  contract — resolution precedes the tokenizer because the tokenizer lives
  inside the artifact, and the rebind is last so a refused model cannot leave a
  mutated config for a retry on another target to resolve wrongly. A refusal
  names the missing codecs, the target **and** the profiles of that model this
  device *can* run.
- Codec classification is **shape-first**: a dense NVFP4 file carries the dtype
  it was quantized *from* (BF16), so trusting `dtype` mislabelled every
  quantized plane — the first real conversion declared `mask=2` while holding 40
  NVFP4 planes. `VerifyArtifact` and `ScanArtifactTree` now read the schema
  shape and fall back to `dtype` only for a 4-column pinned schema.
- Sibling identity moved to the layer that owns it:
  `ModelBase::IdentityAllows` compares the caller's `cfg_.model_type` against
  `artifact.fb` once for all families, instead of ten hand-written tag
  comparisons. Proven non-vacuous on the pair that needs it — `hy-memory` and
  `hy-simulmt` share a tower, and the retired tag was the only thing separating
  them, so the parent's loader would have answered as the sibling.
- One dense Gemma-4 body, one passport per checkpoint:
  `gemma_dense_constants.h` keeps only the shared types, and
  `gemma_dense_12b_constants.h` / `gemma_dense_31b_constants.h` hold the
  numbers. Arithmetic constants are **data from the tree that was loaded**
  (`embed_scale = bf16(sqrt(hidden))` — 62.0 at 3840, 73.5 at 5376;
  `attention_scale = 1.0`), because a shared literal is how one sibling ends up
  scaling every embedding by 1.19. A missing geometry field is a parse failure,
  not a default.
- Tests, perf rows, app defaults and Python tools resolve model paths **through
  the registry** (`SourceDirFor`, `ConvertedDirFor`, `tools/lsx_paths.py`),
  never from a store literal. Five helpers that returned the converted tree
  while being called `VaultRoot()` were renamed `ConvertedRoot()`; in this
  schema `vault` is the read-only source store.
- `FindConvertedTrees` detects a tree by `artifact.fb`, not by directory name —
  the last place in the tree that did.

### Fixed

- **38 sites built `<root>/<Model>` and dropped the org**, naming a directory
  that never existed: every one of them reported "the weights are not on disk"
  while the weights sat one level down. With it, 46 perf rows joining
  `perf.vault_subdir`, 15 Python tools that froze the root in a `default=`, 11
  passports and 29 parity fixtures carrying dead roots
  (`/mnt/llm_models/models/…`, `/mnt/fast_storage/huggingface/…`), two of which
  had not existed since 2026-10.
- `Resolve` accepts `<org>/<model>[/<profile>]`. An explicit profile is a
  **choice** — `Qwen/Qwen3.5-9B/bf16` reports "no profile 'bf16'" and never
  falls back to the nvfp4 sibling — and the org is part of the name, so
  `WrongOrg/Qwen3.5-9B` is refused instead of resolving to Qwen's tree by
  matching the leaf, which is the whole reason the store is nested.
- Artifact self-sufficiency: sidecars were looked for inside the artifact, where
  no converter writes them, so the copy step never ran on any real tree.
  Resolved from the source checkpoint first, with
  `ArtifactSpec::chat_template_text` for checkpoints such as `phi-4` that carry
  the template inline in `tokenizer_config.json` — recording `""` for a
  checkpoint that does have a template is a provenance loss, not honesty.
  `chat_template_sha256`, written since day one and compared by **nobody**, is
  now verified: a recorded template that is absent, or whose bytes moved, is a
  refusal.
- `VarloopTopologyMatches` returned `true` on synthetic geometry **before**
  checking the varloop flag, so a fixed-pass tree was accepted by the varloop
  row. The escape now sits after the capability check and before the numeric
  ones: a fixture may be 8 layers wide and declare 2 passes, never a different
  *kind* of checkpoint.
- Five loaders still read a retired `format` tag out of `top.error_msg`,
  re-wrapping a real layout refusal as "unsupported index format" and clearing
  every other topology error before anything could read it.
- `test_qwen_dense_kv_cache` had been **dead, not merely fragile** — it read
  `/mnt/llm_models/models/Qwen3.5-9B`, gone since 2026-10, so it SKIPped forever
  and read as "weights absent". It runs and passes now.
- `test_model_runtime` recovered its root by calling `parent_path()` twice,
  which after a path rewrite returned `temp_directory_path()` itself, so its
  cleanup tried to `remove_all("/tmp")`.
- `test_ctest_labels_guard`: a `)` inside a comment ended a helper's parsed
  body, so a correctly labelled target read as unlabelled; and the skip-code
  check understood only the literal `set_tests_properties(...)` spelling, not
  the helper's own `SKIP` keyword. It also now links the abseil symbols
  `StrJoin` needs — under clang (the AOCC Release tree) it had never run at all,
  because only GCC's inlining hid the missing reference.
- `mcp_capture_all.py` called `str.endswith` on `Path` objects, so its "real
  checkpoint versus empty placeholder" test could never run.
- `deepseek_moe_repack` spliced the new layout value with
  `regex_replace("\" + "2")` — group 12, which is empty — silently deleting the
  field; `kimi-k3` now tells K3 from K3.1-Pro by the index's own `indexer` flag
  instead of a format tag.
- `Resolve()` shared one `std::error_code` across three nested directory
  iterators, so a failure reading one model directory aborted the search for
  every other model.
- CODE logged its raw answer unescaped, truncating the one-line verdict for
  every line-oriented reader — the same bug DET had already documented.
- Gemma: the `gemma4_unified` ballast is claimed as stored-and-counted instead
  of unclassified, and the NVFP4 triple names its codec instead of the retired
  `U8` spelling (verifier rule D1 rejected the whole tree at publish).
- The reported version was a hand-written literal: `version()` printed
  `logestix v0.1.0` from `common.cpp`, while `version.h.in` was being generated
  from `project(VERSION ...)` and was included by **nobody** — so every binary
  announced 0.1.0 at tags v1.0.0 and v1.0.1. It now reports the generated
  `LSXCOMMON_VERSION_STRING` through `lsxcommon::version_string()`, the project
  declares `1.0.2`, and a test checks both halves — the header against
  `project(VERSION ...)`, and that no source file spells a version literal
  again — because each half looked plausible on its own.

### Removed

- `LegacyDir()`, `kModelRootBase` and the `lsx_model_models_dir()` forwarder.
  A root the binary invents is a path the operator never chose, and with the
  org-nested store it does not even fail loudly — it points at
  `<root>/<org>/<model>` for a root holding nothing, so every model reads as
  "weights absent" when the real problem is that no store was configured.
  `LSX_MODEL_ROOT` is still honoured as the pre-split spelling of the source
  root, second in precedence, and is **not** a default either.
- 18 per-model `format` tags (11 of them dead) and 21 dead `kFormatTag`
  constants, replaced by one integer `layout` generation with no fallback:
  a missing or different `layout` is refused with both numbers named.
- `perf.vault_subdir` / `LSX_PERF_VAULT_SUBDIR` — a second spelling of
  `vault.dir`, byte-identical in 45 of 47 manifests. The generator now refuses a
  row that re-declares it.
- The legacy-tree fallback in `lsxforge --verify`: it refuses a path without
  `artifact.fb` instead of falling back, because there is no legacy tree any
  more.

### Known issues

- **`gemma-4-12B-it` FSI-02 is red, and it is the source's coin flip, not the
  kernel's.** At the fork the source checkpoint itself prefers `1` (21.61) over
  `Rate` (21.05) — a 0.556-logit margin — so an ~8-bit body can move it.
  Measured rather than assumed: host relL2 0.0926 -> 0.0049 with the residual
  plane, verdict unchanged. Recorded as `R22`; closing it needs an operator
  decision (re-capture at a stable protocol, or >8 bits, ~25 GiB).
- 9 rows declare `golden_debt.mcp` pending checkpoint restoration.
- `baidu/ERNIE-5.5-Flash-Base-BF16` is a complete download with no registry row
  (23 org directories on disk, 22 in the registry) — recorded as `W`.
- `int8-w8a16` is declared and refused, not implemented; nothing produces a
  `bf16` profile yet (`--codec-profile bf16` is refused at the end of a full
  conversion), and profile selection is by `preferred_profile` rather than by
  "best compatible profile for this GPU".
- Pre-existing and unrelated to this release: `test_deepseek_moe_loader`,
  `test_deepseek_moe_pro_loader` and `test_gpu_perf` go red only inside a
  parallel sweep (the documented `I2` VRAM probes; each passes at `-j1`), and
  `tools/check_open_tasks.py` reports `G5` and `V` as both live and retired.

## [Unreleased]

## [1.0.1] - 2026-11-27

39 registered models became **47**, eight of them new, and the duplication audit
that had been running since v1.0.0 closed. 121 commits.

### Added

- Eight model ingestions, each through the full lifecycle (converter, loader,
  kernels, passport, tool bridge, SelfCheck / FSI / CODE / DET, perf row):
  - `GigaChat4-708B-A41B` — 710.3 B params, 720 GB FP8 -> 379 GB NVFP4
    (361 GB experts, 14 GB dense, 4.6 GB MTP); 64 decoder layers: 3 dense FFN +
    61 MoE of 256 top-8 experts, 16 gated MLA layers carrying the DSA indexer,
    48 GatedDeltaNet layers. The family core takes an explicit ingest
    expectation (`Gc4IngestSpec` / `Gc4FrozenExpect` / indexer dims) instead of
    reading the 37B passport, so a sibling needs no edit to the first model, and
    experts flush layer by layer during conversion because 387 GB resident in one
    map is an OOM, not a slowdown. SelfCheck PASS x3 bit-exact ("Paris").
  - `GigaChat4-37B-A4B` — hybrid GatedDeltaNet + DSA MLA + MoE.
  - `Hy4-flash` — hybrid sparse/SWA MLA, LatentMoE, iHC.
  - `inkling` — the family base (dense + routed NVFP4, shared-sink experts) with
    its own frozen passport, naming token, native tool codec and real-tree
    numeric probe.
  - `inkling-2-mini` — the looped `inkling2_loop` member: per-pass KV and conv
    slots, loop controllers, mixed-codec experts held device-resident, and a
    3-pass generation pipeline at 34.1 t/s.
  - `tml-d24-varloop-research` — the `is_varloop` sibling: runtime pass count
    instead of a compile-time one, source-captured goldens, 33.98 t/s.
  - `muse-glimmer-30b` — gated attention, NoPE, centered norms, softcap head,
    ATEM conversation and tool bridge; FSI golden captured at
    `reasoning_strength=low`, which closed `R21`.
  - `DeepSeek-V4.1-Flash-1017` — a `v4.1-flash` sibling variant.
- The duplication audit's phases 3-7 closed across all 41 models registered
  when it ran, with bit-identical behaviour as the criterion and net −4.3k LOC:
  the AttnRes depth mixer shared by alice/kimi-k3, one grid-stride cast kernel
  per direction in `lsx_kernels_common.cuh`, tool-format JSON helpers in
  `lsxcommon::tools`, the qwen/ornith MoE loader and generation driver templated
  on the passport (`lsx_moe_load`, `lsx_moe_inference_pipeline`), and nemotron
  lightning/ultra kernels + expert cache shared. New instrument
  `tools/dedup_scan.py` (kernel dup groups 52 → 21, host groups 26 → 7). Reports:
  `docs/open-tasks/2026-11-20-kernel-primitives-closed.md`,
  `docs/open-tasks/2026-11-20-dedup-full-audit-closed.md`.
- ATEM answer-channel selection for the FSI, CODE and DET parity gates, plus
  per-prompt **resumable** FSI capture, so a gate that dies on prompt 3 does not
  re-run prompts 1 and 2.
- `docs/OPEN_TASKS.md` became a routing index over `docs/open-tasks/live/`
  instead of one growing document.
- A memory-leak audit: `A3.1`-`A3.6` fixed and `A3.7`-`A3.9` filed.

### Changed

- Bumped simdjson v5.0.6 -> v5.2.0 (submodule `third_party/simdjson`, commit
  `a0d5ad97…`; the `test_core.cpp` version guard now asserts 5.2.0). On-demand
  correctness fixes upstream (malformed-value escapes, incomplete-JSON key
  cuts, `at_pointer` result handling); license texts unchanged.
- Bumped cpp-httplib v0.63.0 -> v0.66.0 (FetchContent pin `v0.66.0`; the
  `test_core.cpp` `CPPHTTPLIB_VERSION` guard now asserts 0.66.0).
- The README stopped being a second model catalogue: the register of models is
  `docs/MODELS.md`, generated from the manifests.

### Fixed

- `GigaChat4`'s DSA indexer rotated by `2*pos` instead of `pos`, because the
  decoder handed it an already-advanced rope pointer. Invisible below
  `index_topk` (where the selection is the identity), wrong above it, and
  out-of-bounds past position 2048. The shipped 37B's hashes are unchanged.
- The GigaChat4 converter classified its VRAM profile without the depth
  argument, so on a deeper checkpoint 30 of 61 MoE layers read as MTP ballast.
- `Load()` sized the expert slot cache before the fixed-size KV, whose 4 GiB
  reserve is smaller than this member's 5 GiB MLA KV at the 4096 session.
- `GigaChat4`'s `GatedNorm` gated on the unscaled state.
- `Hy4-flash`'s iHC post term multiplied the pre-norm MoE sum.
- Inkling's routed `w13` is row-interleaved, and its residual / conv / down / wo
  paths are FP32.
- Mixed-codec expert trees are verified rather than assumed to be NVFP4, and the
  inkling router dots, top-k selection and loop `MessageGate` are parallelised
  across rank rows.
- Four leaks on error paths: `EnsureWorkspace` released nothing when a chain
  failed, `XingSession` did not own its teardown, `RunKimiGeneration` leaked its
  per-call scratch on non-happy exits, and loaders kept partial allocations when
  they failed.
- The reported varloop pass override now names the checkpoint's declared count
  instead of silently disagreeing with it.
- ASCII-only string literals in the muse passport and converter, which AOCC
  refused with an escape warning.

## [1.0.0] - 2026-11-16

**LogestiX Synthesis v1.0.0 — Open-Source Sovereign C++20/CUDA Inference
Engine.** The first tagged release: 1987 commits of history, 39 registered models.

### Added

- `LSX_MODEL_ROOT` environment override for the model vault root
  (`libs/lsxcommon/src/path_registry.cpp`).
- `THIRD_PARTY_NOTICES`, `DEPENDENCIES.md` and a first-party license guard test.
- This changelog.
- Between-target GPU cooldown for `test_perf_baseline`: `perf::CoolDownGpu`
  (`libs/lsxcommon/src/gpu_power.cpp`, NVML read + a plateau wait) runs before
  each perf target, so a heat-soaked card no longer reads ~4% low and
  false-DEGRADEs against the references (`docs/OPEN_TASKS.md` §W).
- `LOGESTIX_BUILD_APPS` / `LOGESTIX_BUILD_TOOLS` / `LOGESTIX_BUILD_TESTS`
  switches so logestix can be consumed as a dependency without building the
  sibling applications, tools or the engine test suite.

### Changed

- Consolidated the duplicated dense-decoder kernels: a new model-neutral
  `lsxcommon::dense` layer (`lsx_dense_inference.{h,cu}`) now owns RMSNorm,
  full-dim RoPE, causal GQA attention (with the split-KV path), fused split-QKV
  and fused SwiGLU; granite, granite-switch and phi keep their layer drivers
  and delegate through their existing `RunGranite*`/`RunGraniteSwitch*`/`RunPhi*`
  entry points. Net -729 lines in `libs/`, -206 in `tests/`, zero behaviour
  change (`tools/dedup_provenance.py` proves the kernels moved verbatim;
  `logits_hash` unchanged on all three models).
- Bumped Apache Arrow 26.0.0 -> 26.0.1 (FetchContent pin `apache-arrow-26.0.1`;
  the IPC reader now returns `Invalid` for a malformed map array instead of
  crashing — GH-54936. No new APIs in our IPC/io/CUDA surface, so no call site
  changed).
- Bumped simdjson v5.0.5 -> v5.0.6 (submodule `third_party/simdjson`; the
  `test_core.cpp` version guard now asserts 5.0.6).
- Bumped cpp-httplib v0.61.0 -> v0.63.0 (FetchContent pin; the `test_core.cpp`
  `CPPHTTPLIB_VERSION` guard now asserts 0.63.0).
- Bumped Apache Arrow 25.0.1 -> 26.0.0 (FetchContent pin `apache-arrow-26.0.0`;
  Arrow now bundles Boost 1.92.0 and xsimd 14.2.0).
- Bumped simdjson v4.6.11 -> v5.0.5 (submodule `third_party/simdjson`; the
  `test_core.cpp` version guard now asserts 5.0.5).
- Bumped cpp-httplib v0.58.0 -> v0.61.0 (FetchContent pin; the `test_core.cpp`
  `CPPHTTPLIB_VERSION` guard now asserts 0.61.0).
- Bumped Catch2 v3.16.0 -> v3.16.1 (FetchContent pin; the `test_core.cpp`
  `CATCH_VERSION_PATCH` guard now asserts 1).
- NVIDIA CCCL is no longer a submodule: it is taken from the CUDA toolkit
  bundle (`<include>/cccl`) and **band-guarded** (configure-time in the root
  `CMakeLists.txt` + `test_core.cpp`). This removes the `-I` shadowing that made
  every `.cu` — including the toolkit's own transitively-included `<nv/target>` —
  compile against our copy, which could silently invert on a toolkit bump. The
  brief 3.4.2 -> 3.5.0 submodule bump is superseded by this.
- Translated `AGENTS.md`, `.opencode/skills/*` and documentation to English.
- Removed vendored NVIDIA documentation dumps; they are now referenced from the
  `nvidia-cuda-docs` MCP (`docs/NVIDIA_DOCS.md`).
- Added `abseil-cpp` to the README dependency table.

### Removed

- Tracked CTest artifact `Testing/Temporary/CTestCostData.txt` (now gitignored).
- `third_party/cccl` submodule — CCCL now comes from the CUDA toolkit bundle
  (band-guarded at configure time).
