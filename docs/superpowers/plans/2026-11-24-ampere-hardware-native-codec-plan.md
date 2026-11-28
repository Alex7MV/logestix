# Hardware-Native Codec + `lsxstorage` + Ampere AOT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine hardware-native and vendor-independent: one codec contract
(`PlaneCodec` × `HardwareTarget`) enforced at load, one artifact format
(`artifact.fb`) with split source/artifact storage in `libs/lsxstorage`, and native
NVIDIA Ampere (`sm_86` 3090, `sm_87` Orin NX) builds supporting `bf16` and
`int8-w8a16` profiles.

**Architecture:** Three independently mergeable branches. (1) `libs/lsxstorage`
becomes the bottom library owning roots, `artifact.fb` (FlatBuffers, mmap) and
resolution; `lsxcommon` gains a header-only `hardware_target.h` and enforces
`used_codecs ⊆ SupportedCodecs(active_target)` in the single `ModelBase::BeginLoad`
choke point. (2) The build accepts `86;87`, drops the arch-`a` object libs when no
12x arch is present, and the `bf16` codec profile is produced and validated on a
live 3090. (3) An `int8-w8a16` profile with a deterministic W8A16 GEMV + GEMM is
added and validated on a live 3090 and Orin NX.

**Tech Stack:** C++20, CUDA (13.5 on x86, 12.6 on JetPack 6.2.1), CMake 3.25+
presets, FlatBuffers (existing submodule), Apache Arrow 26 (mmap-only),
spdlog, Catch2 v3, simdjson.

**Spec:** `docs/superpowers/specs/2026-11-24-ampere-hardware-native-codec-design.md`
(committed alongside this plan). Executors read **both**.

**Revision 2026-11-27 (cold start):** every converted tree was deleted and the
store starts empty. This *removed* work rather than adding it, and the removals are
marked inline: the `dtype`→codec *mapping* table became five writer-side rules
(Task 7), per-profile ballast records and `drop_ballast` disappeared (Tasks 6, 8, 15),
`ResolveStatus::kLegacy` disappeared (Task 4, 9), the per-tree audit disappeared
(Task 9), and `PreflightDisk` demoted from a hard gate to a guard (Tasks 6, 8).
One task was **added**: **Task 9a, the bootstrap conversion**, because with zero
artifacts the real-weight suite is dark and a SKIP is not a pass.

**Revision 2026-11-27d (ownership):** answering "is `artifact.fb` per model or per
artifact?" found a second source of truth — `model.index.json` still declared codecs
(`mlp_nvfp4`, `attn_nvfp4`, `head_nvfp4`, `expert_nvfp4[]`, `expert_fp8[]`) and six
files read them to pick a kernel per layer. **Task 9b** closes it: the per-layer map
moves into `artifact.fb` (`Artifact::layer_codec[]`), the six readers go through
`ModelBase::LayerUsesCodec()`, and `--verify` fails an index that still carries those
keys. Both files are **per artifact (per model × profile)**; `model.index.json` keeps
geometry and `vram` only, and `vram` is legitimately profile-dependent.

**Revision 2026-11-27e (layout):** "why a `logestix/` folder inside the logestix
store?" — withdrawn. The intermediate path segment is removed, so
`ArtifactDir(...)` **is** the artifact directory:
`<ROOT>/<org>/<model_dir>/<profile>/{artifact.fb,pinned,experts,…}` (spec §2.4.2).
**New Task 4a** performs the mechanical rewrite (538 sites: 144 `libs/`, 3 `apps/`,
391 `tests/`), deletes the four dead `ResolvePinnedDir()` fallbacks, rewrites the two
"detect a tree by folder name" sites to test `artifact.fb`, and adds a `fast` guard
that the segment cannot come back.

**Revision 2026-11-28 (status audit — what is built, what is not).** The plan was
executed up to the end of Branch 1 and stopped there. This revision records the
*measured* state instead of re-planning it, and it corrects every task whose
implementation diverged from the text below: a later executor reads this document
as a description of the tree, so a stale interface here is a defect. Of the original
25 tasks, **nine are done** (1–7, 4a, 9a), **two are done-with-residue** (8, 9), **one
is partial** (9b) and **thirteen have no code at all** (10 and 11–22). Six tasks are
**new** (8a, 8b, 9c, 9d, 9e, 9f): each is a hole the finished code exposed, not a
change of scope.

> **Executor note (2026-11-28).** `AGENTS.md` forbids Task subagents, so the header's
> `subagent-driven-development` option is not available in this repository: execute
> inline with `superpowers:executing-plans`, one task per commit, and run the scoped
> verification AGENTS.md prescribes for each change (the `fast` preset plus that
> model's own targets; label sweeps only for shared-layer work).

Store and lifecycle facts, measured 2026-11-28 (method and evidence in
§ *Lifecycle audit*):

| quantity | measured value |
|---|---|
| registry rows (`models/<org>/<model>.json`) | **48**, each declaring all 10 stages and all 5 gate targets |
| checkpoints under `LSX_SOURCE_ROOT=/mnt/fast_storage/lsx_model` | **51 directories over 24 orgs**: 48 correspond to registry rows, **3 do not** (`baidu/ERNIE-5.5-Flash-Base-BF16`, `JetBrains/Mellum2-12B-A2.5B-Thinking`, `JetBrains/Mellum2.1-12B-A2.5B-Instruct` — open task `W`) |
| artifacts under `LSX_ARTIFACT_ROOT=/mnt/fast_storage/lsx_artifact` | **3** — `microsoft/phi-4/nvfp4`, `Qwen/Qwen3.5-9B/nvfp4`, `google/gemma-4-12B-it/nvfp4` |
| profiles other than `nvfp4` | **0** — no converter can produce one (**Task 9f**) |
| `lsxforge --verify` on those three | green: `declared_mask == actual_mask` (34 / 35 / 34), 0 problems, `--verify` deep-clean on `test_bootstrap_artifacts` |
| rows whose real-weight gates actually ran | **3 of 48**; the other 45 SKIP naming `<org>/<dir>/<profile>` |
| `ctest --preset linux-debug` (label `fast`) | **184/184 green**, 11 skipped, 166 s |

### Task status (2026-11-28)

`done` = the task's tests exist and pass; `residue` = the deliverable exists but a
hole inside it, named by a new task; `partial` = half the deliverable; `not
started` = no commit, no code.

| # | task | status | landed in | evidence / hole |
|---|---|---|---|---|
| 1 | `PlaneCodec` enum, names, parsing | **done** | `bb6612eb` | `hardware_target.h`; 17 cases |
| 2 | `HardwareTarget` + `SupportedCodecs` | **done** | `bb6612eb` | one `switch`, Ampere row = F32/BF16/F16/I8 |
| 3 | `CheckCodecs` + `ActiveTarget` | **done** | `bb6612eb` | probed once; `SetActiveTargetForTesting` is the only override |
| 4 | `libs/lsxstorage` skeleton + roots | **done** | `cef08396` | both roots now `std::optional` (`a7afd49e`) |
| 4a | remove the `logestix/` segment | **done** | `8c691a89` | `test_path_shape_guard` (6 cases, 2 allowlisted files) |
| 5 | `artifact.fb` schema, reader, verifier | **done** | `a7873320` | `artifact.fbs` v1 + `layer_codec[]`; 6 cases |
| 6 | `EmitArtifact` writer | **done** | `a6788275` | tmp+rename, digests, self-verify; 12 cases |
| 7 | `VerifyArtifact` + `--verify` | **done** | `2c27c7de` | 19 cases; D1–D5 enforced per *dtype column* only → **9c** |
| 8 | `--codec-profile` + emitter wired | **residue** | `1b8f324b` | emitter skipped for the default profile → **8a** |
| 9 | admission gate in `BeginLoad` | **residue** | `b4c05cb1` | 5 stages; profile choice not target-aware → **9e** |
| 9a | bootstrap conversion | **done, not as written** | `7575c87e` | 3 `nvfp4` artifacts, no `bf16`; slice never grew → **9f** |
| 9b | per-layer codec map leaves the index | **done** | `a7873320` + `132b683c` | `layer_codec[]` is now per-layer, derived by the scan |
| 10 | `MemoryModel` in the governor | **done** | `8effeaf4` | probed `prop.integrated`; one pool on unified |
| 11 | `LSX_CUDA_ARCH` accepts Ampere | **not started** | — | `CMakeLists.txt:38-53` |
| 12 | arch-`a` object libs only for 12x | **not started** | — | `libs/lsxcommon/CMakeLists.txt:651-671`, `tests/CMakeLists.txt:1440-1460` |
| 13 | toolchain-agnostic toolkit guard | **not started** | — | `test_cuda_toolkit_guard.cu:5,50-51,95-97` |
| 14 | Ampere presets + degradation audit | **not started** | — | `CMakePresets.json:10-75` has no `86`; no `cmake/toolchains/` |
| 15 | `bf16` profile for the slice | **not started** | — | `grep CodecProfile libs/lsxcommon/src` is empty |
| 16 | 3090/Jetson acceptance + docs | **not started** | — | `docs/NVIDIA_DOCS_AMPERE.md` has been **removed** from the tree — see the note in Task 16 |
| 17 | INT8 weight quantizer | **not started** | — | — |
| 18 | `int8-w8a16` converter side | **not started** | — | — |
| 19 | W8A16 decode GEMV | **not started** | — | — |
| 20 | W8A16 prefill GEMM | **not started** | — | — |
| 21 | per-profile goldens | **not started** | — | — |
| 22 | 3090 + Orin acceptance | **not started** | — | — |
| 8a | flagless conversion publishes an artifact | **done** | `04fecd45` | `ResolveCodecProfile`; emitter runs on every conversion |
| 8b | eight converters stop writing `mtp/` | **done** | `04fecd45` | all 8; drops are counted in the index |
| 9c | expert plane declares its codec | **done** | `85307f87` | `ExpertSchema` gains a trailing `dtype` (10 cols) |
| 9d | rule 5 enforced; index-codec readers gone | **done** | `132b683c` | `--verify` refuses the 5 keys; 0 readers left |
| 9e | target-aware profile selection + `--explain-artifact` | **done** | `dc19aed9` | `SelectProfile`; `lsxfabric --explain-artifact` |
| 9f | grow the artifact store + registry/store guard | **new, not started** | — | defect F1/F7/F8 |

## Global Constraints

- **No software emulation (I1).** A profile may only contain `PlaneCodec` values the
  target executes in hardware. NVFP4/FP8 are **never** in the `sm_86`/`sm_87` mask.
- **`artifact.fb` is mandatory.** No legacy codec inference from Arrow columns, file
  suffixes or `format` tags — at any stage, in any branch.
- **One mask, one place:** `used_codecs` in `artifact.fb`. The gate reads it; nothing
  recomputes it at runtime. **Ownership is per field, not per file:** `artifact.fb`
  owns codecs (including the per-layer map), `model.index.json` owns geometry and the
  `vram` profile and must contain **no** codec key (spec §2.4.3 rule 5).
- **The artifact IS the profile directory.** No intermediate `logestix/` path
  segment anywhere: `<LSX_ARTIFACT_ROOT>/<org>/<model_dir>/<profile>/`. An artifact is
  detected by `artifact.fb`, never by a directory name (spec §2.4.2).
- **The mask covers every shipped plane, and every shipped plane is reachable by the
  load path.** v1 ships **no ballast**: `mtp/` and any `deferred` subtree are not
  written by any profile (spec §1.4). No `dropped_planes`, no stored-vs-executed
  distinction.
- **Unambiguous `dtype` spellings are the writer's job (rules D1–D5, spec §1.3).**
  A quantized weight names its codec explicitly: `NVFP4`, `F8_E4M3`, `I8`, `BF16`,
  `F32`. **Never `U8` for a weight.** Companions are `*.scale`/`*.scales`/
  `*.global_scale` and carry the plane's own dtype; metadata is `I64`/`U8` under
  `role = "meta"`. `--verify` **fails** an ambiguous weight plane rather than
  resolving it. The mask is a set union, never a sum.
- **No legacy compatibility of any kind.** No dual-read, no codec inference from
  suffixes or `format` tags, no `kLegacy` resolve status, no migration. An artifact
  without a verified `artifact.fb` does not load.
- **No literal model/tree counts** in any test or runbook step: iterate
  `arrow_check::FindConvertedTrees()` and report what was found.
- **Disk:** the volume has 2.6 TB free. `lsxforge` still preflights the estimated
  artifact size and refuses a conversion it cannot finish — a guard against a
  half-written tree, not a capacity schedule.
- **Header reachable from `.cu`/`.cuh` must not include** `<absl/container/*>`,
  `<flatbuffers/*>`, `<arrow/*>`, or `lsxstorage` types. `hardware_target.h` is
  POD + `<string>`/`<cstdint>` only.
- Logging: `spdlog` / `lsxcommon::log` only. No `std::cout`, no `printf` in
  `libs/` or `apps/`.
- `absl::StrFormat` is printf-style: `%s`, `%d`; never `{}`.
- Arrow reads are mmap-only; Arrow writes atomic (tmp + rename).
- No CLI speed flags; the only diagnostic flag added in this plan is
  `lsxforge --codec-profile` and `lsxfabric --explain-artifact`.
- Device codegen only via `LSX_CUDA_ARCH` + `lsx_cuda_gencode_flags()`; every device
  target sets `CUDA_ARCHITECTURES OFF`; never spell `_deps/...` in a target.
- Every new ctest target: one label (`fast|convert|inference|perf`),
  `SKIP_RETURN_CODE 4` if it can skip, `RESOURCE_LOCK "gpu"` if it touches a device,
  registered through `cmake/LsxTestHelpers.cmake`.
- Dependency floors: library `CUDART_VERSION >= 12060`; `cuda_fp4.h` cases
  `>= 12080` via `__has_include`.
- Never run GPU-labelled ctest while another GPU run is in flight.
- Commits: English, `feat:`/`fix:`/`build:`/`test:`/`docs:`/`refactor:`.
  No subagents (AGENTS.md). Commit only when the operator asks per-branch.

## Review Focus

The failure modes most likely to bite a user of this software, each pinned by a test
in the task that owns the code. The first two are specific to the cold start: with
the old trees gone, there is no real data left to catch them, so the tests are the
only defence.

1. **A converter keeps an old spelling** — eight converters still write `mtp/`
   today, and several spell NVFP4 as a `U8` `.packed`/`.scales` pair (gemma, Muse,
   kimi, mimo, inkling). Carried into the new format, the first case ships planes
   the loader never reads (and now makes the conversion's own scan refuse the
   finished tree) and the second makes the codec unreadable without context.
   → Task 7 (rules D1–D5, `--verify` fails both) and **Task 8b** (the `mtp/`
   writers) + the converter edits in Task 8.
2. **A codec declaration survives in `model.index.json`** — five fields
   (`expert_nvfp4[]`, `expert_fp8[]`, `mlp_nvfp4`, `attn_nvfp4`, `head_nvfp4`) are
   still written by five converters and read by two inkling loaders **plus the
   arrow verifier**, which derives an exponent map from them. Left there, they can
   disagree with `artifact.fb` and the gate would pass a model that executes an
   undeclared codec. → **Task 9b**, and **Task 9d** for the residue.
3. **Nothing has ever been converted, so a wrong rule cannot be caught against
   reality** — the nine-row inventory in spec §1.3 was measured from data that is now
   deleted. → **Task 9a** treats the first real conversion as a contract test: two
   models, `--verify`, and any surprise written back into the spec.
4. **A tree that predates `artifact.fb` is loaded instead of refused** — a user
   points at an HF source checkout (which never contains an artifact) and a path
   that forgot the gate treats its `pinned/`-less directory as a model.
   → Task 9 (`kMissing` path, `kSourceOnly`).
5. **`cfg_.model_path` rebound before the gate passes** — a refused model leaves a
   mutated config, and a retry on another target resolves to the wrong profile.
   → Task 9 (rebind ordering test).
6. **The removed `logestix/` segment silently comes back, or survives somewhere** —
   one missed site sends a loader to a directory that no longer exists, and the
   failure reads as "missing weights" rather than "wrong path form". → **Task 4a**
   (a `fast` guard that no source file spells the segment, allowlisting only the
   per-user cache root).
7. **`sm_87` SASS missing from the fatbin** — Orin NX refuses to launch with "no
   kernel image for device" only at runtime, on hardware the operator may not have
   attached. → Task 11 (gencode flags test asserts both `code=sm_86` and
   `code=sm_87` in the emitted flags).

An eighth, operational: **a half-written tree with no `artifact.fb`** makes the gate
refuse for a reason that looks like a codec bug. → `PreflightDisk` (Tasks 6, 8).

Two more, added by the 2026-11-28 status audit because the shipped code shows they
are reachable today, not hypothetically:

9. **A tool reports success and publishes nothing** — `lsxforge` without
   `--codec-profile` writes a tree, exits 0 and emits no `artifact.fb`, so the tree
   the user just "converted" can never load. → **Task 8a**.
10. **The registry claims a profile the store does not hold** — `status: built` is
    hand-written and no compiled guard compares it with `IsArtifactDir()` or with
    the perf row's date, so a verdict can outlive the tree it described. →
    **Task 9f**.

---

# Branch 1 — `feat/lsxstorage-hardware-codec-contract`

Branch from `main`. Merges alone; produces working, testable software without
Branch 2 (all existing 5090 behaviour preserved after reconversion).

### Task 1: `PlaneCodec` enum, names, parsing

> **DONE (`bb6612eb`).** Steps 1–5 executed; `tests/test_hardware_target.cpp`
> carries **17 cases** (the three below plus the parse/format edges).
> **As built, differs from the text below:** the enum members *are* the bit
> values (`kF32 = 1ull << 0`, …) and `Mask()` is a `static_cast`, so there is no
> second `1 << n` table to drift; `kNone = 0` exists so a parsed name is never
> confused with "unset". `PlaneCodecName`/`ParsePlaneCodec` read one
> `kCodecNames[]` table, so a name cannot parse to one codec and render as
> another. The header carries no absl/Arrow/FlatBuffers, as required.

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/hardware_target.h`
- Create: `libs/lsxcommon/src/hardware_target.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add the `.cpp` to the host-only target)
- Test: `tests/test_hardware_target.cpp` (label `fast`, no GPU, no skip)

**Interfaces:**
- Produces: `namespace lsxcommon::hw`, `enum class PlaneCodec : uint64_t` with bits
  `kF32=1<<0, kBF16=1<<1, kF16=1<<2, kI8Sym=1<<3, kMxFp8=1<<4, kNvFp4=1<<5,
  kMxFp4=1<<6`; `using CodecMask = uint64_t`; `constexpr CodecMask Mask(PlaneCodec)`;
  `const char *PlaneCodecName(PlaneCodec)`; `bool ParsePlaneCodec(std::string_view,
  PlaneCodec *)`; `std::string FormatMask(CodecMask)`.

- [x] **Step 1: Write the failing test** — `tests/test_hardware_target.cpp`:

```cpp
TEST_CASE("plane codec bits are stable") {
    REQUIRE(Mask(PlaneCodec::kF32)   == 1ull);
    REQUIRE(Mask(PlaneCodec::kBF16)  == 2ull);
    REQUIRE(Mask(PlaneCodec::kF16)   == 4ull);
    REQUIRE(Mask(PlaneCodec::kI8Sym) == 8ull);
    REQUIRE(Mask(PlaneCodec::kMxFp8) == 16ull);
    REQUIRE(Mask(PlaneCodec::kNvFp4) == 32ull);
    REQUIRE(Mask(PlaneCodec::kMxFp4) == 64ull);
}
TEST_CASE("plane codec names round-trip through parse") {
    for (const char* n : {"f32","bf16","f16","i8","mxfp8","nvfp4","mxfp4"}) {
        PlaneCodec c{};
        REQUIRE(ParsePlaneCodec(n, &c));
        REQUIRE(ParsePlaneCodec(PlaneCodecName(c), &c));
    }
    PlaneCodec c{};
    CHECK_FALSE(ParsePlaneCodec("fp8", &c));   // never accepted: ambiguous
    CHECK_FALSE(ParsePlaneCodec("", &c));
}
TEST_CASE("FormatMask renders exactly the set bits, ascending") {
    REQUIRE(FormatMask(Mask(PlaneCodec::kBF16) | Mask(PlaneCodec::kNvFp4))
            == "bf16,nvfp4");
    REQUIRE(FormatMask(0) == "none");
}
```

- [x] **Step 2: Run it and confirm failure**

Run: `cmake --build build/linux-debug --target test_hardware_target -j` → FAIL to
compile (`hardware_target.h` not found).
Register the target first with `lsx_add_test(NAME test_hardware_target LABEL fast)`
so the failure is a link/compile failure of a known target, not a missing target.

- [x] **Step 3: Implement** `hardware_target.h` (POD only, `#include <cstdint>`,
  `<string>`, `<string_view>`; no absl/arrow/flatbuffers) and `hardware_target.cpp`.
  `PlaneCodecName` returns the lowercase names used by `--codec-profile` and
  `artifact.fb`; `ParsePlaneCodec` accepts exactly those plus `"i8"` for `kI8Sym`.
  Comment at the top: *storage encoding, not execution path* (spec §2.2).

- [x] **Step 4: Run and confirm pass** — `ctest --test-dir build/linux-debug -R test_hardware_target`

- [x] **Step 5: Commit** — `feat: add PlaneCodec storage-encoding enum with names and parsing`

### Task 2: `HardwareTarget` + the `SupportedCodecs` table

> **DONE (`bb6612eb`).** The table is one `switch` with the hardware reason per
> row. **As built, differs from the text below:** `TargetFromId` takes
> `std::string_view` (not `std::string`); `operator==`/`operator!=` were added so
> tests and callers compare targets without spelling `.Id()`. The CPU row is not
> a second table: `SupportedCodecs` derives the mask from the arch *string* via
> `CpuMaskForArch`, so `TargetFromId("cpu:avx512+bf16")` and
> `TargetFromCpuFeatures(true,true,…)` cannot disagree. `TargetFromCpuFeatures`
> with no usable ISA returns `Vendor::kUnknown` (gate fails closed) rather than a
> `cpu:` id with an empty mask. Ampere's row is
> `F32|BF16|F16|I8` with the GA10x reason in the comment
> (`mma.m16n8k16` f16/bf16, `mma.m16n8k32.s8`, no E4M3/E2M1) — consistent with
> the Ampere reference (§A2/§A4 of the MCP export).

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/hardware_target.h`, `src/hardware_target.cpp`
- Test: `tests/test_hardware_target.cpp`

**Interfaces:**
- Consumes: Task 1.
- Produces: `enum class Vendor : uint8_t {kNvidia, kCpu, kAmd, kUnknown}`;
  `struct HardwareTarget {Vendor vendor; std::string arch; std::string Id() const;}`;
  `CodecMask SupportedCodecs(const HardwareTarget &)`;
  `HardwareTarget TargetFromId(const std::string &, bool *ok)`;
  `HardwareTarget TargetFromComputeCapability(int major, int minor)`;
  `HardwareTarget TargetFromCpuFeatures(bool avx512, bool bf16, bool fp16, bool vnni,
  bool neon, bool dotprod)`.

- [x] **Step 1: Write the failing test** — the table, verbatim from spec §2.3:

```cpp
TEST_CASE("Blackwell executes all seven plane codecs") {
    const auto t = TargetFromComputeCapability(12, 0);
    REQUIRE(t.Id() == "nvidia:sm_120");
    REQUIRE(SupportedCodecs(t) == (Mask(PlaneCodec::kF32)|Mask(PlaneCodec::kBF16)
        |Mask(PlaneCodec::kF16)|Mask(PlaneCodec::kI8Sym)|Mask(PlaneCodec::kMxFp8)
        |Mask(PlaneCodec::kNvFp4)|Mask(PlaneCodec::kMxFp4)));
}
TEST_CASE("Ampere executes F32/BF16/F16/INT8 and never FP8 or FP4") {
    for (int minor : {6, 7}) {
        const auto t = TargetFromComputeCapability(8, minor);
        REQUIRE(t.Id() == (minor == 6 ? "nvidia:sm_86" : "nvidia:sm_87"));
        const CodecMask m = SupportedCodecs(t);
        REQUIRE(m == (Mask(PlaneCodec::kF32)|Mask(PlaneCodec::kBF16)
                     |Mask(PlaneCodec::kF16)|Mask(PlaneCodec::kI8Sym)));
        REQUIRE((m & Mask(PlaneCodec::kMxFp8)) == 0);
        REQUIRE((m & Mask(PlaneCodec::kNvFp4)) == 0);
        REQUIRE((m & Mask(PlaneCodec::kMxFp4)) == 0);
    }
}
TEST_CASE("Ada is not a capability row") {
    CHECK(SupportedCodecs(TargetFromComputeCapability(8, 9)) == 0);
}
TEST_CASE("cpu target is feature-derived, never assumed from the family name") {
    const auto bare = TargetFromCpuFeatures(true, false, false, false, false, false);
    REQUIRE(SupportedCodecs(bare) == Mask(PlaneCodec::kF32));
    const auto full = TargetFromCpuFeatures(true, true, true, true, false, false);
    REQUIRE(SupportedCodecs(full) == (Mask(PlaneCodec::kF32)|Mask(PlaneCodec::kBF16)
        |Mask(PlaneCodec::kF16)|Mask(PlaneCodec::kI8Sym)));
    CHECK(full.Id() == "cpu:avx512+bf16+fp16+vnni");
}
TEST_CASE("Jetson host side is neon, not avx512") {
    const auto t = TargetFromCpuFeatures(false, false, false, false, true, true);
    REQUIRE(t.Id() == "cpu:neon+dotprod");
    REQUIRE((SupportedCodecs(t) & Mask(PlaneCodec::kI8Sym)) != 0);
}
TEST_CASE("amd is reserved and supports nothing yet") {
    bool ok = false;
    const auto t = TargetFromId("amd:gfx942", &ok);
    REQUIRE(ok);
    CHECK(SupportedCodecs(t) == 0);
}
TEST_CASE("sm_86 and sm_87 are distinct targets") {
    CHECK(TargetFromComputeCapability(8, 6).Id() !=
          TargetFromComputeCapability(8, 7).Id());
}
TEST_CASE("unknown compute capability yields kUnknown, not a default") {
    CHECK(TargetFromComputeCapability(9, 0).vendor == Vendor::kUnknown);
    CHECK(TargetFromComputeCapability(8, 0).vendor == Vendor::kUnknown);  // sm_80 forbidden
}
```

- [x] **Step 2: Run and confirm failure** — unimplemented symbols.

- [x] **Step 3: Implement** the table as one `switch` on `(vendor, arch)` with the
  hardware reason as a comment per row (spec §2.3). `TargetFromComputeCapability`
  returns `kUnknown` for anything outside `{12.0, 12.1, 8.6, 8.7}` — including
  `8.0`, per AGENTS.md. `TargetFromCpuFeatures` builds the arch string from feature
  flags in a **fixed order**: `avx512`, `+bf16`, `+fp16`, `+vnni`; for neon,
  `neon`, `+dotprod`.

- [x] **Step 4: Run and confirm pass.**

- [x] **Step 5: Commit** — `feat: add HardwareTarget and the SupportedCodecs capability table`

### Task 3: `CheckCodecs` — I3 as a pure function — and `ActiveTarget`

> **DONE (`bb6612eb`).** `CheckCodecs` is the three-line function this task
> specified, with two refusal wordings added: an unknown vendor says *"no
> supported accelerator detected"* and a named-but-unimplemented target says
> *"not implemented in this build"* — both distinct from "you are missing codec
> X", which is what the operator needs to read. `ActiveTarget()` probes once
> under a `std::mutex` (not `std::once_flag`, because
> `ResetActiveTargetForTesting()` must be able to drop the cached probe);
> `hardware_target.cpp` may call `cudaGetDeviceCount`/`cudaGetDeviceProperties`
> because it is a host TU in `lsxcommon`, never reached from `.cu`. A present but
> untargeted device (sm_80/sm_89) deliberately does **not** fall through to the
> CPU row.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/hardware_target.h`, `src/hardware_target.cpp`
- Test: `tests/test_hardware_target.cpp`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `struct GateDecision {bool allowed; CodecMask missing; std::string
  message;}`; `GateDecision CheckCodecs(CodecMask used, const HardwareTarget &)`;
  `const HardwareTarget &ActiveTarget()`; `void SetActiveTargetForTesting(const
  HardwareTarget &)`; `void ResetActiveTargetForTesting()`.

- [x] **Step 1: Write the failing test:**

```cpp
TEST_CASE("gate allows a subset and reports the exact missing bits") {
    const auto sm86 = TargetFromComputeCapability(8, 6);
    CHECK(CheckCodecs(Mask(PlaneCodec::kBF16)|Mask(PlaneCodec::kF32), sm86).allowed);
    const auto d = CheckCodecs(Mask(PlaneCodec::kBF16)|Mask(PlaneCodec::kNvFp4), sm86);
    REQUIRE_FALSE(d.allowed);
    REQUIRE(d.missing == Mask(PlaneCodec::kNvFp4));
    CHECK(d.message.find("nvfp4") != std::string::npos);
    CHECK(d.message.find("nvidia:sm_86") != std::string::npos);
}
TEST_CASE("gate on an unimplemented target reports the target, not a codec") {
    bool ok = false;
    const auto amd = TargetFromId("amd:gfx942", &ok);
    const auto d = CheckCodecs(Mask(PlaneCodec::kBF16), amd);
    REQUIRE_FALSE(d.allowed);
    CHECK(d.message.find("not implemented") != std::string::npos);
}
TEST_CASE("gate refuses when the target is unknown") {
    REQUIRE_FALSE(CheckCodecs(Mask(PlaneCodec::kBF16), HardwareTarget{}).allowed);
}
TEST_CASE("ActiveTarget override is scoped to the test process") {
    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    REQUIRE(ActiveTarget().Id() == "nvidia:sm_86");
    ResetActiveTargetForTesting();
}
```

- [x] **Step 2: Run and confirm failure.**

- [x] **Step 3: Implement.** `CheckCodecs` is three lines: unknown vendor → refuse;
  `SupportedCodecs(t) == 0` → refuse "target not implemented"; otherwise
  `missing = used & ~supported`. `ActiveTarget()` resolves once under
  `std::once_flag`: `cudaGetDeviceProperties(&p, 0)` when a device exists (guard the
  CUDA call so host-only builds link), else CPUID-derived `cpu:*` features
  (`__builtin_cpu_supports("avx512f")`, `"avx512bf16"`, `"avx512fp16"`,
  `"avx512vnni"`; NEON via `__aarch64__` + `__ARM_FEATURE_DOTPROD`).

- [x] **Step 4: Run and confirm pass** on the 5090 host — `ActiveTarget().Id()` must
  be `nvidia:sm_120` there; add a `SKIP`-guarded case asserting that.

- [x] **Step 5: Commit** — `feat: add the codec admission gate as a pure function plus active-target resolution`

### Task 4: `libs/lsxstorage` skeleton and root resolution

> **DONE (`cef08396`, then reshaped).** `libs/lsxstorage` exists with
> `roots.h/.cpp`, links spdlog + FlatBuffers, and `lsxcommon::lsx_model_models_dir()` /
> `lsx_cache_dir()` / `lsx_cache_path()` are one-line forwarders. **As built,
> supersedes several details below — do not re-create them:**
>
> - `ArtifactDir(org, model_dir, profile)`, **not** `(family, model, profile)`:
>   the artifact store mirrors the source store `<org>/<model>/<profile>`
>   (`8c691a89`), so the first two segments are the *store's* spelling
>   (`microsoft`, `phi-4`), not the manifest's family/model_type.
> - Both roots are `std::optional<fs::path>` and **neither has a default**:
>   `LSX_SOURCE_ROOT` and `LSX_ARTIFACT_ROOT` are required configuration
>   (`a7afd49e`). The plan's "SourceRoot falls back to the platform default" is
>   withdrawn — a guessed store is how a loader reads the wrong tree.
> - `LegacyDir()` is **deleted** (the `logestix/` segment is gone, Task 4a).
> - Added by the re-measurement (`72904bef`, `1c6ad30a`, `a7afd49e`):
>   `FindSourceModel(name, why*)`, `ListSourceModels()`, `ListPlaceholderDirs()`
>   — the source store is org-nested, so finding a checkpoint by name is no
>   longer a path join, and a download that exists without weights must be
>   distinguishable from one that does not exist.
>
> The test block below is the *pre-org-nesting* contract and is kept for the
> precedence questions it asks (optional root, cache precedence, forwarders);
> the literal expectations are in `tests/test_storage_roots.cpp` (**26 cases**),
> which is the authority.

**Files:**
- Create: `libs/lsxstorage/CMakeLists.txt`
- Create: `libs/lsxstorage/include/lsxstorage/roots.h`, `src/roots.cpp`
- Modify: `libs/CMakeLists.txt` (add `add_subdirectory(lsxstorage)` **before** `lsxcommon`)
- Modify: `libs/lsxcommon/src/path_registry.cpp` (delegate), `include/lsxcommon/path_registry.h` (document)
- Test: `tests/test_storage_roots.cpp` (label `fast`)

**Interfaces:**
- Produces: `namespace lsx::storage`, `fs::path SourceRoot()`,
  `std::optional<fs::path> ArtifactRoot()`, `fs::path CacheRoot()`,
  `fs::path ArtifactDir(const std::string &family, const std::string &model, const std::string &profile)`,
  `fs::path LegacyDir(const std::string &model)` (diagnostic only — see below).
- Constraint: `lsxstorage` links FlatBuffers (from the `flatbuffers` target already
  in `third_party`) + spdlog only. **It must not link or include `lsxcommon`.**

- [x] **Step 1: Write the failing test** — use `setenv`/`unsetenv` in a RAII helper:

```cpp
TEST_CASE("artifact root is optional and never invented") {
    ScopedEnv e("LSX_ARTIFACT_ROOT", nullptr);
    CHECK_FALSE(ArtifactRoot().has_value());
}
TEST_CASE("artifact dir joins family/model_type/profile under the artifact root") {
    ScopedEnv e("LSX_ARTIFACT_ROOT", "/tmp/art");
    REQUIRE(ArtifactDir("phi", "phi-4", "bf16") ==
            fs::path("/tmp/art/phi/phi-4/bf16"));
    // model_type, NOT the vault dir name: "Qwen3.5-9B" -> "qwen3.5-9b".
    REQUIRE(ArtifactDir("qwen", "qwen3.5-9b", "bf16") ==
            fs::path("/tmp/art/qwen/qwen3.5-9b/bf16"));
}
TEST_CASE("source root falls back to LSX_MODEL_ROOT then the platform default") {
    { ScopedEnv a("LSX_SOURCE_ROOT", nullptr); ScopedEnv b("LSX_MODEL_ROOT", "/m");
      CHECK(SourceRoot() == fs::path("/m")); }
    { ScopedEnv a("LSX_SOURCE_ROOT", "/s"); ScopedEnv b("LSX_MODEL_ROOT", "/m");
      CHECK(SourceRoot() == fs::path("/s")); }
}
TEST_CASE("lsxcommon forwards to lsxstorage verbatim") {
    ScopedEnv a("LSX_MODEL_ROOT", "/m");
    CHECK(lsxcommon::lsx_model_models_dir() == "/m");
}
TEST_CASE("an unset artifact root is never silently replaced by a legacy path") {
    ScopedEnv a("LSX_ARTIFACT_ROOT", nullptr);
    ScopedEnv b("LSX_MODEL_ROOT", "/m");
    CHECK_FALSE(ArtifactRoot().has_value());
    // LegacyDir still answers, but only so Resolve can SAY "source, not artifact".
    CHECK(LegacyDir("phi-4") == fs::path("/m/phi-4/logestix"));
}
TEST_CASE("cache dir precedence is unchanged by the move") {
    ScopedEnv a("LSX_CACHE_DIR", "/c");
    CHECK(CacheRoot() == fs::path("/c"));
    CHECK(lsxcommon::lsx_cache_path("perf-cache/x.json") == "/c/perf-cache/x.json");
}
```

- [x] **Step 2: Run and confirm failure** (target absent).

- [x] **Step 3: Implement** by *moving* `path_registry.cpp`'s body into
  `lsx::storage` and leaving `lsxcommon::lsx_model_models_dir()` /
  `lsx_cache_dir()` / `lsx_cache_path()` as one-line forwarders. Update
  `libs/CMakeLists.txt` ordering. `libs/lsxcommon/CMakeLists.txt` gains
  `target_link_libraries(... lsxstorage)` on the host-only target that now needs it.

- [x] **Step 4: Run and confirm pass**, then prove no regression:
  `ctest --test-dir build/linux-debug -R "test_storage_roots|test_core|test_path" `.

- [x] **Step 5: Commit** — `refactor: move model-root resolution into libs/lsxstorage behind forwarders`

### Task 4a: Remove the `logestix/` path segment

> **DONE (`8c691a89`, folded together with the org-nested artifact path).**
> Measured now: `grep -rn '"logestix"' libs apps tests` returns 6 lines, all
> comments or the two allowlisted sites (`path_registry.cpp` for
> `~/.cache/logestix`, `tools/lsx_paths.py` for recognising a pre-split tree).
> `IsArtifactDir()` exists in `lsxstorage/artifact.h` and the three former
> folder-name detectors use it. `tests/test_path_shape_guard.cpp` has **6 cases**
> and scans `libs/`, `apps/`, `tests/` — plus the Python spelling, which the plan
> missed and a reference runbook actually uses.
>
> **As built:** the rewrite was not a blind scripted removal — the artifact path
> was *redefined* to mirror the source store in the same commit, so 538 sites did
> not become `fs::path(x)` but `ArtifactDir(org, model_dir, profile)` where an
> artifact was meant. Anything the plan describes as "one mechanical rewrite" is
> therefore historical.

**Why.** The artifact store made that segment redundant (spec §2.4.2), and it was
actively harmful because three detection sites used the folder *name* as the "is
this a converted tree?" marker instead of `artifact.fb`.

**Files:**
- Modify (`libs/` 144 sites / 84 files, `apps/` 3 / 1, `tests/` 391 / 142 — mechanical
  `fs::path(x) / "logestix"` → `fs::path(x)`; use a scripted rewrite then review)
- Modify: `libs/lsxcommon/src/tokenizer_arrows.cpp` — delete `ResolvePinnedDir()`'s
  four dead fallback branches (`logestix-qwen35moe`, `logestix-glm`,
  `logestix-kimi`, `logestix-gemma`); it becomes `model_path / "pinned"`
- Modify: `libs/lsxcommon/src/arrow_tree_verify.cpp:506` and
  `libs/lsxcommon/src/deepseek_moe_repack.cpp:83-85` — detect an artifact by
  `artifact.fb`, not by a directory name
- Test: `tests/test_path_shape_guard.cpp` (label `fast`)

**Interfaces:**
- Consumes: Task 4.
- Produces: the final on-disk path form —
  `<LSX_ARTIFACT_ROOT>/<org>/<model_dir>/<profile>/{artifact.fb,pinned,experts,…}`.
  No API change; `cfg_.model_path` stays the thing every join hangs off.

- [x] **Step 1: Write the failing guard test** (the shape of `test_logging_guard`:
  scan sources, allowlist the one legitimate use):

```cpp
TEST_CASE("no source file spells a logestix path segment") {
    // Allowlist: the per-user cache root in path_registry.cpp (~/.cache/logestix).
    const std::vector<std::string> allowed = {"path_registry.cpp"};
    for (const auto &dir : {"libs", "apps", "tests"})
        for (const auto &f : all_source_files(dir))
            if (!contains(allowed, f.filename()))
                CHECK_MESSAGE(no_logestix_segment(f), f.string());
}
TEST_CASE("the tree detectors key on artifact.fb, not on a directory name") {
    // Deliberately a bare file: Task 4a must not depend on EmitArtifact (Task 6).
    // Whether a file is a *valid* artifact is Task 5/9's business.
    TempDir d; make_minimal_tree(d.path());          // pinned/ + model.index.json
    CHECK_FALSE(IsArtifactDir(d.path()));            // no artifact.fb yet
    make_dir(d.path()/"logestix");                   // the OLD marker
    CHECK_FALSE(IsArtifactDir(d.path()));            // still not an artifact
    write_file(d.path()/"artifact.fb", "");          // presence is the marker
    CHECK(IsArtifactDir(d.path()));
}
```

`IsArtifactDir()` is a two-line `lsx::storage` helper
(`fs::exists(dir / "artifact.fb")`) introduced here so the three detector sites stop
testing a directory *name*. It deliberately says nothing about validity.

- [x] **Step 2: Run and confirm failure** — the guard finds 538 sites.

- [x] **Step 3: Implement** the rewrite. Order matters: (a) guard test in place, (b)
  scripted `/"logestix"` removal across the three trees, (c) hand-fix the three
  detector call sites (`IsArtifactDir`) and the `ResolvePinnedDir` fallbacks, (d)
  `ResolvePinnedDir` collapses to `model_path / "pinned"`, (e) run the full `fast`
  sweep.

- [x] **Step 4: Run and confirm pass** — `ctest --preset linux-debug`. Note honestly in
  the commit message that the *positive* path — a real converted artifact loading from
  the new layout — cannot be proven until Task 9a exists; what this task proves is
  that no site still spells the old segment and that the detectors key on
  `artifact.fb`.

- [x] **Step 5: Commit** — `refactor: drop the redundant logestix path segment from artifact layout`

### Task 5: `artifact.fb` schema, reader, verifier

> **DONE (`a7873320`).** `libs/lsxstorage/schema/artifact.fbs` (v1, identifier
> `LSXA`), `artifact.h`, `artifact.cpp`, `tests/test_storage_artifact.cpp`
> (**6 cases**). `ReadStatus` is exactly the five planned states — no `kLegacy`.
> **As built, differs from the text below:**
>
> - `ArtifactInfo` carries more than the plan listed, all of it load-bearing:
>   `org`/`model_dir` (so a profile can be enumerated from `artifact.fb` alone),
>   `layer_codec[]`, `chat_template_sha256` + `has_chat_template_digest` (written
>   since Task 6 and read by nothing until 2026-11-28 — a recorded digest that no
>   reader checks is a hollow field; `VerifyArtifact` now refuses a drifted
>   template), `source_dir`, `layer_count`, `pinned_tensors`, `expert_files`.
> - `IsArtifactDir()` lives here, not only in the verifier: the presence test is
>   what replaced "is there a directory named logestix?".
> - **No committed `.fb` fixtures.** Every case builds its artifact in a
>   `TempDir` through `EmitArtifact`/raw bytes, so the repo ships no binary blob
>   and a fixture cannot go stale against the schema. `tests/fixtures/storage/`
>   does not exist.

**Files:**
- Create: `libs/lsxstorage/schema/artifact.fbs` (schema from spec §2.4.3, verbatim)
- Create: `libs/lsxstorage/include/lsxstorage/artifact.h`, `src/artifact.cpp`
- Modify: `libs/lsxstorage/CMakeLists.txt` (generate the header with `flatc`, add its
  dir to that target only)
- Test: `tests/test_storage_artifact.cpp` (label `fast`), fixtures under
  `tests/fixtures/storage/`

**Interfaces:**
- Produces: `struct PlaneInfo`, `struct ArtifactInfo` (carrying `used_codecs` and
  `planes` — **no `dropped_planes`**: v1 ships no ballast, spec §1.4), `enum class
  ReadStatus {kOk,kMissing,kUnreadable,kBadVersion,kMalformed}`, `struct ReadResult
  {ReadStatus status; ArtifactInfo info; std::string detail;}`,
  `ReadResult ReadArtifact(const fs::path &artifact_dir)`,
  `ReadResult ReadArtifactFile(const fs::path &fb)`.

- [x] **Step 1: Write the failing test:**

```cpp
TEST_CASE("missing artifact.fb is kMissing, not an exception") {
    TempDir d;                                    // empty dir
    const auto r = ReadArtifact(d.path());
    REQUIRE(r.status == ReadStatus::kMissing);
    CHECK_FALSE(r.detail.empty());
}
TEST_CASE("a valid artifact round-trips mask, profile, target and planes") {
    const auto r = ReadArtifact(kFixtureDir / "phi4_bf16");
    REQUIRE(r.status == ReadStatus::kOk);
    CHECK(r.info.model_type == "phi-4");
    CHECK(r.info.family == "phi");
    CHECK(r.info.profile == "bf16");
    CHECK(r.info.target_id == "nvidia:sm_86");
    CHECK(r.info.used_codecs == (Mask(PlaneCodec::kBF16)|Mask(PlaneCodec::kF32)));
    REQUIRE(r.info.planes.size() >= 2);
    CHECK(r.info.planes[0].has_digest);
}
TEST_CASE("a truncated buffer is kUnreadable and a tampered one kMalformed") {
    TempDir d; write_truncated_fb(d.path());
    CHECK(ReadArtifact(d.path()).status == ReadStatus::kUnreadable);
    TempDir e; write_bitflipped_fb(e.path());
    CHECK(ReadArtifact(e.path()).status == ReadStatus::kMalformed);
}
TEST_CASE("format_version other than 1 is kBadVersion with the version in detail") {
    const auto r = ReadArtifact(kFixtureDir / "version_2");
    REQUIRE(r.status == ReadStatus::kBadVersion);
    CHECK(r.detail.find("2") != std::string::npos);
}
```

- [x] **Step 2: Run and confirm failure.**

- [x] **Step 3: Implement.** `ReadArtifact` = `MemoryMappedFile::Open` (mmap only, per
  AGENTS.md) → `flatbuffers::Verifier` over the mapped range → plain-struct copy-out.
  `artifact.h` **must not** include flatbuffers: the `.fb` accessors live in an
  anonymous namespace inside `artifact.cpp`. Generate the three fixtures as
  `.fb` files committed under `tests/fixtures/storage/` (small, produced by the
  writer in Task 6).

- [x] **Step 4: Run and confirm pass.**

- [x] **Step 5: Commit** — `feat: add artifact.fb FlatBuffers schema with mmap reader and verifier`

### Task 6: `EmitArtifact` — writer with digests and tokenizer copy

> **DONE (`a6788275`).** `artifact_writer.h/.cpp` + `src/sha256.{h,cpp}`;
> `tests/test_storage_artifact_writer.cpp` (**12 cases**): tmp+fsync+rename,
> digests, self-re-read, mask re-derivation, and no `artifact.fb` left behind on
> failure. **As built, differs from the text below:**
>
> - `ArtifactSpec` also carries `org`/`model_dir`/`source_dir`/`target_id` and
>   **`chat_template_text`**. The last one is the fix for the real-weight defect of
>   2026-11-28: `microsoft/phi-4` carries its template inline in
>   `tokenizer_config.json` with no `.jinja` beside the weights, so "copy the
>   `.jinja`" produced `chat_template == ""` on a checkpoint that does have a
>   template. The writer materializes the inline text as `chat_template.jinja`
>   and digests it like any other sidecar.
> - `PreflightDisk(dir, bytes, error*)` is forwarded from here and called by
>   `lsxforge` with a **conservative 8 GiB constant estimate**, not a per-model
>   estimate: it is a half-written-tree guard, not a capacity schedule.

**Files:**
- Create: `libs/lsxstorage/include/lsxstorage/artifact_writer.h`, `src/artifact_writer.cpp`
- Create: `libs/lsxstorage/src/sha256.{h,cpp}` (small, or reuse an existing digest util — check first; none found in recon)
- Test: `tests/test_storage_artifact_writer.cpp` (label `fast`)

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces: `struct PlaneDecl{std::string role; uint64_t codec_bit; std::string path;
  uint64_t bytes;}`, `struct ArtifactSpec{...}` (spec §2.4.5 — **no `drop_ballast`**;
  includes `std::vector<uint64_t> layer_codec`, the per-layer codec map that
  replaces the codec fields removed from `model.index.json`, spec §2.4.3 rule 5),
  `uint64_t EmitArtifact(const ArtifactSpec &, const fs::path &artifact_dir,
  std::string *error)` — returns the mask written, 0 on failure.
- Also produces: `bool PreflightDisk(const fs::path &dir, uint64_t bytes_needed,
  std::string *error)` — true iff free space ≥ `bytes_needed`. A guard against a
  half-written tree (2.6 TB is free), used by `lsxforge` before a conversion starts.

- [x] **Step 1: Write the failing test:**

```cpp
TEST_CASE("emit writes artifact.fb whose used_codecs is the OR of its planes") {
    TempDir d; make_minimal_tree(d.path());          // 2 planes: bf16 data, f32 data
    ArtifactSpec s = minimal_spec(d.path());
    std::string err;
    const uint64_t mask = EmitArtifact(s, d.path(), &err);
    REQUIRE(err.empty());
    REQUIRE(mask == (Mask(PlaneCodec::kBF16)|Mask(PlaneCodec::kF32)));
    const auto r = ReadArtifact(d.path());
    REQUIRE(r.status == ReadStatus::kOk);
    CHECK(r.info.used_codecs == mask);
}
TEST_CASE("emit copies chat_template.jinja and tokenizer.json and digests them") {
    TempDir d, src; write_file(src.path()/"chat_template.jinja", "{{x}}");
    write_file(src.path()/"tokenizer.json", "{}"); make_minimal_tree(d.path());
    ArtifactSpec s = minimal_spec(d.path());
    s.chat_template_src = (src.path()/"chat_template.jinja").string();
    s.tokenizer_json_src = (src.path()/"tokenizer.json").string();
    REQUIRE(EmitArtifact(s, d.path(), nullptr) != 0);
    CHECK(fs::exists(d.path()/"chat_template.jinja"));
    CHECK(fs::exists(d.path()/"tokenizer.json"));
    CHECK(ReadArtifact(d.path()).info.chat_template == "chat_template.jinja");
}
TEST_CASE("emit is atomic: a failure leaves no artifact.fb") {
    TempDir d; make_minimal_tree(d.path());
    ArtifactSpec s = minimal_spec(d.path());
    s.planes.push_back({"bogus", Mask(PlaneCodec::kMxFp8), "does_not_exist", 1});
    std::string err;
    REQUIRE(EmitArtifact(s, d.path(), &err) == 0);
    CHECK_FALSE(fs::exists(d.path()/"artifact.fb"));
    CHECK(err.find("does_not_exist") != std::string::npos);
}
TEST_CASE("emit re-verifies what it wrote and fails on a mask mismatch") {
    TempDir d; make_minimal_tree(d.path());
    ArtifactSpec s = minimal_spec(d.path());
    s.planes[0].codec_bit = Mask(PlaneCodec::kNvFp4);   // lie about the plane
    std::string err;
    REQUIRE(EmitArtifact(s, d.path(), &err) == 0);
    CHECK(err.find("mask") != std::string::npos);
}
```

- [x] **Step 2: Run and confirm failure.**

- [x] **Step 3: Implement**: write to `artifact.fb.tmp`, `fsync`, `rename`; digest
  every plane with SHA-256 over the mapped file; copy the tokenizer and template
  copies; then `ReadArtifact` + `VerifyArtifact(deep=false)` what was written and
  return 0 + `*error` on any disagreement.

- [x] **Step 4: Run and confirm pass.**

- [x] **Step 5: Commit** — `feat: add atomic artifact.fb writer with plane digests and tokenizer copy`

### Task 7: `VerifyArtifact` + `lsxforge --verify` integration

> **DONE (`2c27c7de`).** `lsxstorage/src/verify.cpp` (`VerifyArtifact`,
> `ScanArtifactTree`), `tests/test_storage_verify.cpp` (**19 cases**), and
> `lsxforge --verify` prints `declared_mask=… actual_mask=… problems=…` and
> exits non-zero on any problem. Measured on the three real artifacts: 34 / 35 /
> 34, 0 problems.
> **As built — and this is where the mask's honesty is decided:**
>
> - The codec of a file is taken from its **Arrow schema shape first**, then
>   from the `dtype` column only for the 4-column `PinnedSchema`. That is
>   deliberate: a `DenseNvfp4Schema` file records the dtype it was quantized
>   *from* (`"BF16"`), so trusting the column reported NVFP4 planes as BF16.
> - **Hole, owned by new Task 9c:** the 9-column `ExpertSchema` has no dtype
>   column, and the verifier hard-codes `kExpert → kNvFp4`. A BF16 or INT8 expert
>   plane is therefore unrepresentable, and a mixed NVFP4+MXFP8 expert tree
>   (Inkling-2-Small) verifies against a mask that is simply wrong. This blocks
>   the `bf16`/`int8-w8a16` profiles for every MoE family, so 9c is scheduled
>   before Task 15.
> - `ScanArtifactTree()` was added (not in the plan): forge cannot carry a
>   hand-written plane table per family for 40 pipelines, so it **scans** what
>   the converter wrote and the profile declaration becomes a cross-check
>   ("the tree holds no codec the profile did not promise"). The plan's
>   "declare, then verify" is now "scan, then cross-check both ways".
> - `--verify` does **not** yet reject a `model.index.json` that still carries
>   codec keys (rule 5) — that is residue of Task 9b, carried by Task 9d.

**Files:**
- Modify: `libs/lsxstorage/src/artifact.cpp` (+ `.h`: `VerifyReport`,
  `VerifyArtifact`)
- Modify: `libs/lsxcommon/src/arrow_tree_verify.cpp` — **verification only**; add a
  `VerifyAgainstArtifact(tree_root, ArtifactInfo)` entry that reuses `Classify()`
- Modify: `apps/lsxforge/src/main.cpp:263-290` (`--verify` path)
- Test: `tests/test_storage_verify.cpp` (label `fast`), extend
  `tests/test_arrow_tree_integrity.cpp`

**Interfaces:**
- Produces: `struct VerifyReport{bool ok; std::vector<std::string> problems;
  uint64_t declared_mask, actual_mask;}`, `VerifyReport VerifyArtifact(const fs::path
  &, bool deep)`; `lsxforge --verify` exits non-zero on any problem and prints
  `declared=… actual=…`.

- [x] **Step 1: Write the failing test:**

```cpp
TEST_CASE("verify recomputes the mask from the files and catches drift") {
    TempDir d; make_minimal_tree(d.path());
    REQUIRE(EmitArtifact(minimal_spec(d.path()), d.path(), nullptr) != 0);
    CHECK(VerifyArtifact(d.path(), true).ok);

    // A plane appears after the mask was declared (the exact I1 hole).
    write_file(d.path()/"pinned"/"extra_nvfp4.arrow", "not-really-arrow");
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.actual_mask != r.declared_mask);
}
TEST_CASE("verify flags a plane whose digest changed") {
    TempDir d; make_minimal_tree(d.path());
    REQUIRE(EmitArtifact(minimal_spec(d.path()), d.path(), nullptr) != 0);
    append_file(d.path()/"pinned"/"layer_0.arrow", "x");
    const auto r = VerifyArtifact(d.path(), true);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("sha256") != std::string::npos);
}
TEST_CASE("lsxforge --verify fails a tree with no artifact.fb") {
    TempDir d; make_minimal_tree(d.path());           // tree, no artifact.fb
    CHECK(run_forge({"--verify","--input",d.path().string()}) != 0);
}
// ── rules D1–D5 (spec §1.3): the reader REJECTS ambiguity, it never resolves it ──
TEST_CASE("a U8 weight plane is rejected: rule D1 forbids the old spelling") {
    TempDir d;
    write_pinned(d.path(), "down_proj.weight.packed", "U8", 64);
    write_pinned(d.path(), "down_proj.weight.scales", "U8", 4);
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("U8") != std::string::npos);
    CHECK(r.problems[0].find("NVFP4") != std::string::npos);  // says what to write
}
TEST_CASE("an explicitly named NVFP4 plane is accepted and sets the bit") {
    TempDir d;
    write_pinned(d.path(), "down_proj.weight", "NVFP4", 64);
    write_pinned(d.path(), "down_proj.weight.scale", "U8", 4);   // D2 companion
    CHECK(VerifyArtifact(d.path(), false).ok);
}
TEST_CASE("metadata and companions contribute no codec bit") {
    TempDir d;
    write_meta(d.path(), "indices", "I64", 24);        // D3: role = "meta"
    write_meta(d.path(), "flag", "BOOL", 1);
    CHECK(VerifyArtifact(d.path(), false).actual_mask == 0);
    TempDir e;
    write_pinned(e.path(), "x.weight", "F8_E4M3", 32);
    write_pinned(e.path(), "x.weight.scale", "U8", 1);  // E8M0 bytes, D2 companion
    CHECK(VerifyArtifact(e.path(), false).actual_mask == Mask(PlaneCodec::kMxFp8));
}
TEST_CASE("a metadata plane that looks like weights is rejected") {
    TempDir d;                                         // D4, the inverse direction
    write_meta(d.path(), "sneaky.weight", "BF16", 4096);
    REQUIRE_FALSE(VerifyArtifact(d.path(), false).ok);
}
TEST_CASE("one codec spelled twice in one tree is an OR, not a sum") {
    TempDir d;                                         // D5
    write_pinned(d.path(), "a.weight", "NVFP4", 64);
    write_pinned(d.path(), "b.weight", "NVFP4", 64);
    CHECK(VerifyArtifact(d.path(), false).actual_mask == Mask(PlaneCodec::kNvFp4));
}
TEST_CASE("a tree containing mtp/ is rejected: v1 ships no ballast") {
    TempDir d; make_minimal_tree(d.path());
    write_ballast(d.path(), "mtp/layer_032.arrow", "F8_E4M3", 6);
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("mtp") != std::string::npos);
    CHECK(r.problems[0].find("not reachable") != std::string::npos);
}
```

- [x] **Step 2: Run and confirm failure.**

- [x] **Step 3: Implement.** `VerifyArtifact` walks the tree, classifies each file with
  the existing schema-shape logic, and enforces **rules D1–D5 (spec §1.3)** — a
  *verification-only* check, never used at runtime:
  - D1: a weight plane with `dtype == "U8"` is a **failure** (message names `NVFP4`
    as the spelling to use). Only `NVFP4`, `F8_E4M3`, `I8`, `BF16`, `F32` are
    accepted for weights.
  - D2: `*.scale`/`*.scales`/`*.global_scale` are companions — they add no bit.
  - D3/D4: `role = "meta"` planes add no bit; a `role = "meta"` plane whose bytes
    look like weights is a failure.
  - D5: the mask is a union.
  - Every plane in `planes[]` must exist, and **every file in the tree must be a
    listed plane**: a `mtp/` directory (or any subtree the load path cannot reach)
    is a failure naming it as unreachable, because v1 ships no ballast.
  `deep` also re-digests. `--verify` requires `artifact.fb` and prints both masks.

- [x] **Step 4: Run and confirm pass**; then
  `ctest -R "test_storage_verify|test_arrow_tree_integrity"`.

- [x] **Step 5: Commit** — `feat: verify artifact.fb masks and digests against the converted tree`

### Task 8: `--codec-profile` in `lsxforge`, emitter wired once

> **DONE, WITH RESIDUE (`1b8f324b`).** `apps/lsxforge/src/codec_profiles.{h,cpp}`,
> `ABSL_FLAG(codec_profile)` + the `--codec-profile` spelling alias,
> `apps/lsxforge/tests/test_forge_codec_profile.cpp` (**8 cases**, label `fast`),
> `PreflightDisk` before any work, and one emitter call in the `finish` lambda.
> **As built, differs from the text below:**
>
> - `DeclaredPlanes(profile, PipelineKind, layers)` is **coarse on purpose** — two
>   kinds (`kDense`, `kMoe`) x 3 rows, not one row per pipeline. The real plane
>   list comes from `ScanArtifactTree(output_path)`; the declaration is the
>   cross-check (`scan.mask & ~promised == 0`), which is the only shape that
>   scaled to 40 pipelines.
> - `kDefault` is a **separate profile that emits nothing** (`finish` returns
>   before `EmitArtifact`, `main.cpp:538-541`). That is a contract hole — see
>   **Task 8a**.
> - **Step 3's "every profile stops writing `mtp/`" was not done.** Eight
>   converters still write `base / "mtp"`, which makes their trees unusable with
>   `--codec-profile` at all (`ContentDirs()` has no `mtp`, so `ScanArtifactTree`
>   refuses the finished tree). → **Task 8b**.
> - `--codec-profile` also requires `LSX_ARTIFACT_ROOT` and a manifest row for
>   the input; the output path is then `ArtifactProfileDir(identity, profile)` and
>   `--output` is ignored. Both refusals happen before a byte is read.

**Files:**
- Modify: `apps/lsxforge/src/main.cpp` (`ABSL_FLAG(codec_profile)`, plane tables,
  one `EmitArtifact` call per pipeline outcome)
- Modify: `apps/lsxforge/src/cli.cpp` / `cli.h` (validate the profile string)
- Create: `apps/lsxforge/src/codec_profiles.h` (per-pipeline plane table)
- Test: `tests/test_forge_codec_profile.cpp` (label `fast` for parsing;
  `convert` for one real tiny conversion)

**Interfaces:**
- Consumes: Task 6.
- Produces: `enum class CodecProfile {kDefault, kBf16, kNvFp4, kInt8W8A16}`,
  `bool ParseCodecProfile(std::string_view, CodecProfile *)`,
  `std::vector<PlaneDecl> DeclaredPlanes(CodecProfile, PipelineKind, int layers)`,
  and `bool PreflightDisk(const fs::path &, uint64_t bytes, std::string *)` forwarded
  from Task 6. **No `DropsBallast`**: every profile drops ballast identically
  (spec §1.4), so there is nothing to branch on.

- [x] **Step 1: Write the failing test:**

```cpp
TEST_CASE("unknown codec profile is rejected before any conversion starts") {
    CHECK_FALSE(ParseCodecProfile("fp8", nullptr));
    CHECK_FALSE(ParseCodecProfile("", nullptr));      // empty means default, not a name
    CHECK(ParseCodecProfile("bf16", nullptr));
    CHECK(ParseCodecProfile("nvfp4", nullptr));
    CHECK(ParseCodecProfile("int8-w8a16", nullptr));
}
TEST_CASE("the nvfp4 profile declares nvfp4 planes and the bf16 profile does not") {
    const auto nv = DeclaredPlanes(CodecProfile::kNvFp4, PipelineKind::kDense, 40);
    const auto bf = DeclaredPlanes(CodecProfile::kBf16,  PipelineKind::kDense, 40);
    CHECK(any_of(nv, [](auto&p){ return p.codec_bit == Mask(PlaneCodec::kNvFp4); }));
    CHECK(any_of(bf, [](auto&p){ return p.codec_bit == Mask(PlaneCodec::kBF16); }));
    CHECK(none_of(bf, [](auto&p){ return p.codec_bit == Mask(PlaneCodec::kNvFp4); }));
}
TEST_CASE("int8-w8a16 is not offered on a pipeline that has no quantizer yet") {
    CHECK(DeclaredPlanes(CodecProfile::kInt8W8A16, PipelineKind::kDense, 40).empty());
}
TEST_CASE("no profile declares a ballast plane, for any pipeline") {
    for (auto p : {CodecProfile::kBf16, CodecProfile::kNvFp4, CodecProfile::kInt8W8A16})
        for (auto k : {PipelineKind::kDense, PipelineKind::kMoe})
            for (const auto &pl : DeclaredPlanes(p, k, 40))
                CHECK(pl.path.rfind("mtp/", 0) != 0);
}
TEST_CASE("forge --codec-profile with a bad value exits non-zero and converts nothing") {
    CHECK(run_forge({"--phi","--input","/nonexistent","--codec-profile","bogus"}) != 0);
}
TEST_CASE("a conversion that cannot fit on disk is refused before it writes") {
    // PreflightDisk reports far less free space than the estimated artifact.
    CHECK_FALSE(PreflightDisk(kTempDir, 1ull << 40, &err));   // 1 TiB, never free
    CHECK(err.find("free") != std::string::npos);
}
```

- [x] **Step 2: Run and confirm failure.**

- [x] **Step 3: Implement.** Parse → validate → **`PreflightDisk`** against the
  estimated artifact size → run the selected pipeline → on `arrow::Status::OK()`,
  build the `ArtifactSpec` from `DeclaredPlanes(...)` and call `EmitArtifact` into
  the profile dir from `lsx::storage::ArtifactDir(family, model, profile)`. On
  emitter failure the conversion result becomes a failure (a tree without
  `artifact.fb` is not a product). One call site, 40 pipelines covered.
  **Every profile stops writing `mtp/`**: the nine converters that emit it lose that
  code path in this task, so no artifact can carry an unreachable plane.

- [x] **Step 4: Run and confirm pass**; one real conversion of a tiny model into a
  temp artifact root, then `lsxforge --verify` on it.

- [x] **Step 5: Commit** — `feat: add --codec-profile and emit artifact.fb from every forge pipeline`
  — *true for every explicit profile; the flagless `kDefault` path still returns
  before the emitter, which is Task 8a's defect.*

### Task 8a: The flagless conversion must still publish an artifact

> **DONE (`04fecd45`).** `ResolveCodecProfile()` collapses `kDefault` into
> `kNvFp4`, the emitter runs unconditionally, and an explicit `--codec-profile`
> still decides the destination (store profile dir vs `--output`). Measured: a
> flagless dory conversion wrote `artifact.fb` (mask 34, 147 planes) and
> `--verify` reported declared=34 actual=34, problems=0.

**Why this is a defect, not a preference.** `artifact.fb` is mandatory at load
(`Global Constraints`, §0.2), yet `lsxforge --phi --input <src>` — the shape in
every pre-contract runbook, the shape `ABSL_FLAG(output, "converted_model")`
defaults to — returns 0 and writes **no** `artifact.fb`, because `finish()`
short-circuits on `CodecProfile::kDefault` (`apps/lsxforge/src/main.cpp:538-541`).
The result is a tree that `lsxforge --verify` rejects and `BeginLoad` refuses
forever, from a tool that reported success. Two sources of truth for "what does a
conversion produce" is exactly the second-source-of-truth failure §0.5 forbids.

**Files:**
- Modify: `apps/lsxforge/src/main.cpp` (`finish`, the profile validation block)
- Modify: `apps/lsxforge/src/codec_profiles.{h,cpp}` (`kDefault` becomes an alias,
  or is deleted)
- Test: `apps/lsxforge/tests/test_forge_codec_profile.cpp` (extend, label `fast`)

**Interfaces:**
- Consumes: Task 8.
- Produces: `CodecProfile ResolveCodecProfile(CodecProfile p)` — `kDefault` maps to
  `kNvFp4`; there is no code path in `finish` that returns before emitting when
  the output is an artifact directory.

- [ ] **Step 1: Write the failing test** (label `fast`, no GPU — assert on the
  emitted file, not on a full conversion):

```cpp
TEST_CASE("the default profile is the nvfp4 profile, not a no-op profile") {
    CHECK(ResolveCodecProfile(CodecProfile::kDefault) == CodecProfile::kNvFp4);
    CHECK(CodecProfileName(ResolveCodecProfile(CodecProfile::kDefault)) ==
          std::string("nvfp4"));
}
TEST_CASE("the declared planes of the default profile are nvfp4's") {
    CHECK(DeclaredPlanes(CodecProfile::kDefault, PipelineKind::kDense, 4) ==
          DeclaredPlanes(CodecProfile::kNvFp4, PipelineKind::kDense, 4));
}
```

- [ ] **Step 2: Run it and confirm failure** — `DeclaredPlanes(kDefault, …)`
  currently answers nvfp4 rows (they share a `case`), so the second case may pass;
  the first fails to compile, which is the point.

- [ ] **Step 3: Implement.** In `finish`, delete the early return for `kDefault`.
  Resolve the profile once at flag-parsing time (`codec_profile =
  ResolveCodecProfile(codec_profile)`), so the emitter, the cross-check and the
  recorded `spec.profile` all speak `nvfp4`. A flagless conversion with no
  artifact root keeps writing to `--output` (existing behaviour) but now says so
  at `warn`: *"no LSX_ARTIFACT_ROOT: writing an artifact to --output, which no
  other tool will find"* — a diagnostic, not a refusal, because a scratch
  conversion is a legitimate use.

- [ ] **Step 4: Run and confirm pass**, then one scratch conversion of a 2-layer
  fixture with **no** `--codec-profile`: `artifact.fb` exists, `profile ==
  "nvfp4"`, `lsxforge --verify --input <dir>` green.

- [ ] **Step 5: Commit** — `fix: a flagless conversion publishes an nvfp4 artifact instead of a tree nothing can load`

### Task 8b: The eight converters that still write `mtp/` must stop

> **DONE (`04fecd45`).** All eight (alice, gigachat ×2, mimo, ornith, qwen_moe,
> qwen_moe_exp, xing, plus gigachat-708b's test). The dropped tensors are
> COUNTED (`dropped_mtp_tensors` / a log field), because a silent drop is data
> loss nobody can audit. `grep -rn mtp_dir libs/` is empty. Alice's fixture
> gained `WriteAliceConfigJson(mtp_layers)` / `WriteAliceShard(with_mtp)` — its
> MTP path had no test at all, so the fix would otherwise be unverified.

**Why this blocks the artifact path for MoE families.** `VerifyArtifact` and
`ScanArtifactTree` accept content only under `pinned|experts|router|shared|ple`
(`lsxstorage/src/verify.cpp:49-54`), because v1 ships no ballast (spec §1.4). Eight
converters still create `base / "mtp"` and write planes into it whenever the
checkpoint carries MTP tensors, so such a conversion is refused **after** the whole
tree is written, by the scan:

```
artifact scan: converted tree holds a file the load path cannot reach: mtp/…
               (v1 ships no ballast: an unreachable plane is not written)
```

The plan put this in Task 8 step 3 ("every profile stops writing `mtp/`"); it was
not done, and it is invisible to the `fast` suite because no test converts one of
these families with a profile.

| converter | `mtp_dir` write |
|---|---|
| `alice_moe_convert.cpp` | `:317`, `:554`, `:557` |
| `gigachat_moe_37b_convert.cpp` | `:598`, `:785` |
| `gigachat_moe_convert.cpp` | `:413`, `:579` |
| `mimo_moe_convert.cpp` | `:412`, `:542` |
| `ornith_moe_convert.cpp` | `:401`, `:715`, `:719` |
| `qwen_moe_convert.cpp` | `:332`, `:605`, `:609` |
| `qwen_moe_exp_convert.cpp` | `:495`, `:916`, `:920` |
| `xing_moe_convert.cpp` | `:693`, `:857` |

(`deepseek_moe_v41_convert.cpp:92,265,488` is the opposite case — it *excludes*
`mtp` with a counted record — and is the pattern to copy.)

**Files:**
- Modify: the eight converters above — delete the `mtp_dir` construction and its
  writes; keep the tensor-classification branch that routes `mtp.*` tensors away
  from `pinned/`, and log the count that was dropped (a silent drop is data loss
  nobody can audit)
- Test: `libs`-side `tests/test_arrow_tree_integrity.cpp` or a per-family `convert`
  target: a converted fixture of one MoE family must contain no `mtp/`

**Interfaces:**
- Consumes: Task 7 (`ScanArtifactTree`'s content-dir rule is the detector).
- Produces: no new API. The rule is a property of every converter: **a plane the
  load path cannot reach is not written, and what is skipped is counted.**

- [ ] **Step 1: Write the failing test** — convert one family's fixture **that
  carries MTP tensors** (the write is conditional on the bucket being non-empty,
  which is why a fixture without them would pass vacuously) and assert:

```cpp
TEST_CASE("a converted moe fixture carries no mtp ballast") {
    const auto a = convert_fixture("qwen_moe");
    REQUIRE(a.ok);
    CHECK_FALSE(fs::exists(a.dir / "mtp"));
    CHECK(lsx::storage::ScanArtifactTree(a.dir).ok);
    CHECK(VerifyArtifact(a.dir, /*deep=*/true).ok);
}
```

- [ ] **Step 2: Run it and confirm failure** — the tree contains `mtp/` and the
  scan reports the unreachable file.

- [ ] **Step 3: Implement** in the eight files; the dropped-tensor count goes into
  the same log line each converter already uses for its excluded categories
  (`deepseek_moe_v41_convert.cpp:488` is the wording to match).

- [ ] **Step 4: Run and confirm pass**, then each touched family's `convert`
  target: `ctest -R "alice|gigachat|mimo|ornith|qwen_moe|xing" -L convert -j1`.
  A `mtp/` still present means a write site was missed — grep for `mtp_dir` in
  `libs/lsxcommon/src` and expect zero hits.

- [ ] **Step 5: Commit** — `refactor: eight moe converters stop writing mtp ballast, and count what they drop`

### Task 9: The admission gate in `ModelBase::BeginLoad`

> **DONE, WITH RESIDUE (`b4c05cb1`).** `BeginLoad` is the five-stage pipeline —
> resolve → read → **identity** → gate → rebind → tokenizer — with the refusal
> texts this task specified; `tests/test_load_codec_gate.cpp` has **10 cases**
> (gate refusal on Ampere naming the codec and the target, the same artifact
> admitted on Blackwell, rebind ordering, source-only, absent path, synthetic
> escape, `LayerUsesCodec`, an artifact of another model, a mis-pinned
> `model_type`). `tests/test_artifact_migration_audit.cpp` was correctly never
> written (the per-tree audit was withdrawn with the cold start).
>
> **As built, differs from the text below (and from spec §2.5):**
>
> - **Stage (2b) IDENTITY is new.** `IdentityAllows(cfg_.model_type, info)`
>   compares the caller's declared `model_type` against `artifact.fb`'s, once for
>   every family, replacing ten hand-written `format`-tag comparisons in loaders
>   (`768b8a8a`). An **empty** declaration is the autodetect path and passes by
>   construction.
> - `Resolve(model_or_path)` takes **no target**; the plan's
>   `Resolve(path, ActiveTarget())` does not exist. The gate is still target-aware,
>   the *choice of profile* is not — that is **Task 9e**.
> - On refusal the gate lists the profiles this device *can* run, computed with
>   `ListProfiles(org, model_dir)` + `CheckCodecs` per sibling (a read of each
>   `artifact.fb`), not with the planned
>   `ListCompatibleProfiles(family, model, target)`.
> - `cfg_.synthetic` is the documented escape and is now set by 37 `model`-stage
>   fixtures; without it the gate refused every converted fixture in the tree
>   (2026-11-28 finding).

**Files:**
- Modify: `libs/lsxcommon/src/model_base.cpp` (`BeginLoad`)
- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h` (document the order)
- Modify: `libs/lsxcommon/include/lsxcommon/model.h` (`ModelInitConfig::synthetic`)
- Create: `libs/lsxstorage/src/resolve.cpp` + `.h` (`Resolve`, `Resolved`,
  `ResolveStatus`, `ListCompatibleProfiles`)
- Test: `tests/test_load_codec_gate.cpp` (label `fast`, no GPU — table-driven)
- Test: `tests/test_artifact_migration_audit.cpp` (label `fast`, `SKIP` when the
  vault is absent)

**Interfaces:**
- Consumes: Tasks 3–8.
- Produces: `lsx::storage::Resolved Resolve(const std::string &, const HardwareTarget &)`;
  `std::vector<std::string> ListCompatibleProfiles(const std::string &family,
  const std::string &model, const HardwareTarget &)`; `BeginLoad` order =
  resolve → read → gate → rebind → tokenizer.

- [x] **Step 1: Write the failing test:**

```cpp
TEST_CASE("a tree without artifact.fb is refused with the reconversion command") {
    FixtureArtifact f(FixtureKind::kNoArtifact);
    SetActiveTargetForTesting(TargetFromComputeCapability(12, 0));
    const auto r = f.load();
    REQUIRE_FALSE(r.ok);
    CHECK(r.log.find("artifact.fb") != std::string::npos);
    CHECK(r.log.find("lsxforge") != std::string::npos);
    CHECK(r.log.find("--codec-profile") != std::string::npos);
}
TEST_CASE("nvfp4 artifact on sm_86 is refused and names the codec and the target") {
    FixtureArtifact f(FixtureKind::kNvfp4Ok);
    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    const auto r = f.load();
    REQUIRE_FALSE(r.ok);
    CHECK(r.log.find("nvfp4") != std::string::npos);
    CHECK(r.log.find("nvidia:sm_86") != std::string::npos);
}
TEST_CASE("the same nvfp4 artifact loads on sm_120") {
    FixtureArtifact f(FixtureKind::kNvfp4Ok);
    SetActiveTargetForTesting(TargetFromComputeCapability(12, 0));
    CHECK(f.gate_passes());          // gate-level assertion; full Load may need GPU
}
TEST_CASE("bf16 artifact loads on sm_86, sm_87 and sm_120") {
    for (auto t : {TargetFromComputeCapability(8,6), TargetFromComputeCapability(8,7),
                   TargetFromComputeCapability(12,0)}) {
        FixtureArtifact f(FixtureKind::kBf16Ok);
        SetActiveTargetForTesting(t);
        CHECK(f.gate_passes());
    }
}
TEST_CASE("a refused model leaves cfg_.model_path untouched") {
    FixtureArtifact f(FixtureKind::kNvfp4Ok);
    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    const std::string before = f.config().model_path;
    (void)f.load();
    CHECK(f.config().model_path == before);
}
TEST_CASE("synthetic fixtures bypass resolve/read/gate explicitly") {
    ModelInitConfig cfg; cfg.model_path = kEmptyDir; cfg.synthetic = true;
    CHECK(BeginLoadForTest(cfg) || cfg.tokenizer_optional);   // documented escape
}
TEST_CASE("every model that HAS an artifact has a loadable profile on sm_120") {
    // Cold start: the store is empty, so this asserts a relation only for models
    // that were bootstrapped (Task 9a) — never "all 47 manifest rows are present".
    if (!lsx::storage::ArtifactRoot()) SKIP("no artifact root (cold start)");
    int checked = 0;
    for (const auto &m : kManifest) {
        const auto profiles = ListCompatibleProfiles(m.family, m.vault_dir,
                                                     TargetFromComputeCapability(12,0));
        if (profiles.empty()) continue;          // not converted yet: allowed
        ++checked;
        CHECK(std::is_sorted(profiles.begin(), profiles.end()));  // deterministic
    }
    INFO("models with artifacts: " << checked);
    CHECK(checked >= 2);                          // the Task 9a slice at minimum
}
```

- [x] **Step 2: Run and confirm failure.**

- [x] **Step 3: Implement** the five-stage `BeginLoad` from spec §2.5, with the exact
  refusal texts. `Resolve` precedence: (a) path already contains `artifact.fb`
  → `kArtifact`; (b) artifact root set and a compatible profile of that model exists
  → `kArtifact`; (c) source dir exists, or `LSX_MODEL_ROOT/<Model>/logestix` exists
  → `kSourceOnly` (the old vault layout is *not* a loadable tree: no `artifact.fb`);
  (d) `kNotFound`.
  There is **no `kLegacy`** — spec §2.4.1.

- [x] **Step 4: Run and confirm pass**, then the suite guards:
  `ctest -R "test_load_codec_gate|test_model_suite_guard|test_model_independence_guard|test_manifest_sync"`.
  Fix any fast test that went red because its fixture is not marked `synthetic` —
  that is the gate working, and it is the expected work in this task.

- [x] **Step 5: Add a guard that the empty store stays diagnostic.** With zero
  artifacts, the failure a user actually hits is "I pointed at a source dir". Assert
  the exact message names the source dir, says it is not a LogestiX artifact, and
  gives the `lsxforge` command — for a source dir, for an empty directory, and for a
  directory that contains `pinned/` and a `model.index.json` but no `artifact.fb`
  (a hand-rolled or partially-converted tree: the shape that would have been
  loadable before the contract, and must not be now).

- [x] **Step 6: Commit** — `feat: enforce the hardware codec gate in BeginLoad`

### Task 9a: Bootstrap conversion — the task that turns the real-weight suite back on

> **DONE, BUT NOT AS WRITTEN (`7575c87e`).** The converter/emitter was exercised
> against real checkpoints for the first time and three defects were fixed
> (documented in spec §1 revision 2026-11-28): the tokenizer/template copy never
> ran on a real tree because it looked under the *artifact* dir; an inline
> template in `tokenizer_config.json` needed materializing; and 37 `model`-stage
> fixtures needed `cfg_.synthetic`.
>
> **What differs from the steps below, and why it matters:**
>
> - The slice was converted **`nvfp4` first and only** — `microsoft/phi-4/nvfp4`,
>   `Qwen/Qwen3.5-9B/nvfp4` — plus a third model added later,
>   `google/gemma-4-12B-it/nvfp4` (`a8615734`). The plan's `bf16`-first order
>   existed to unblock Ampere acceptance; that ordering did not happen, and it is
>   why Branch 2 has nothing to load. **No `bf16` artifact exists anywhere.**
> - Step 4 (`--explain-artifact` showing a target-dependent pick) **could not be
>   executed**: the flag does not exist and there is no second profile to choose
>   between. It is carried by **Task 9e**.
> - `tests/test_bootstrap_artifacts.cpp` (label `inference`, `RESOURCE_LOCK "gpu"`)
>   exists with **5 cases** and hard-codes the two `nvfp4` paths on purpose, so it
>   cannot pass vacuously on an empty store. Measured 2026-11-28: green, 375 s
>   (deep re-digest of 34 GiB).
> - The gates that were dark now **run** for those three models and only those:
>   SelfCheck ×3 + FSI + CODE + DET + MCP, verdicts in `docs/SELFCHECK_MATRIX.md`
>   and in each row's `profiles.<p>.gates`. Growing the set beyond three is
>   **Task 9f**.

**Why this is a task and not a follow-up.** With every converted tree deleted, 22 test
files that resolve `LSX_TEST_*_ROOT` and 27 that touch a converted tree SKIP, and per
AGENTS.md *a gate that SKIPs is not a pass*. Until two artifacts exist, nothing in
Branch 1 is verified against real weights. This task is a Branch-1 exit criterion.

**Files:**
- Create: `tests/test_bootstrap_artifacts.cpp` (label `inference`,
  `RESOURCE_LOCK "gpu"`, `SKIP_RETURN_CODE 4` when no artifact root)
- Modify: `docs/OPEN_TASKS.md` (record the cold start and the new artifact root)
- Test: the existing per-model targets for the two bootstrap models

**Interfaces:**
- Consumes: Tasks 4–9.
- Produces: no new API. It is the first real use of `EmitArtifact`, and any defect it
  exposes is fixed here or written back into the spec (§1.3).

- [x] **Step 1: Convert the two slice models, `bf16` first**

```
export LSX_ARTIFACT_ROOT=<store>
lsxforge --phi  --input <SRC>/phi-4      --codec-profile bf16
lsxforge --qwen --input <SRC>/Qwen3.5-9B --codec-profile bf16
lsxforge --verify --input <store>/phi/phi-4/bf16
lsxforge --verify --input <store>/qwen/qwen3.5-9b/bf16
```

Expected: `--verify` green on both; masks `bf16,f32`; **no `mtp/`**; `artifact.fb`,
`tokenizer.json` and `chat_template.jinja` present inside each artifact. A failure
here is a converter defect — fix the converter, not the check.

- [x] **Step 2: Assert the contract on real bytes (this is the test that cannot be
  faked by fixtures)**

```cpp
TEST_CASE("bootstrapped artifacts satisfy the codec contract on real weights") {
    const auto root = lsx::storage::ArtifactRoot();
    if (!root) SKIP("LSX_ARTIFACT_ROOT unset — bootstrap conversion not run");
    for (const auto &rel : {"phi/phi-4/bf16", "qwen/qwen3.5-9b/bf16"}) {
        const auto dir = *root / rel;
        const auto r = ReadArtifact(dir);
        REQUIRE(r.status == ReadStatus::kOk);
        CHECK((r.info.used_codecs & Mask(PlaneCodec::kNvFp4)) == 0);
        CHECK((r.info.used_codecs & Mask(PlaneCodec::kMxFp8)) == 0);
        CHECK_FALSE(fs::exists(dir / "mtp"));
        CHECK(VerifyArtifact(dir, /*deep=*/true).ok);
    }
}
TEST_CASE("a bf16 artifact gates clean on sm_86 and on sm_120") {
    if (!lsx::storage::ArtifactRoot()) SKIP("no artifact root");
    const auto info = ReadArtifact(*lsx::storage::ArtifactRoot() / "phi/phi-4/bf16");
    REQUIRE(info.status == ReadStatus::kOk);
    CHECK(CheckCodecs(info.info.used_codecs, TargetFromComputeCapability(8, 6)).allowed);
    CHECK(CheckCodecs(info.info.used_codecs, TargetFromComputeCapability(12, 0)).allowed);
}
```

- [x] **Step 3: Run the gates that were dark**, on the 5090, sequentially
  (one GPU workload at a time):

```
ctest --test-dir build/linux-debug -R "phi_4|qwen3.5-9b" -j1
```

Expected: SelfCheck ×3, FSI, CODE, DET and the perf row now **run** instead of
SKIPping. Record the verdicts; a red is a finding for the report, not a reason to
weaken a gate.

- [ ] **Step 4: Convert the `nvfp4` siblings and prove invariant 4**

```
lsxforge --phi --input <SRC>/phi-4 --codec-profile nvfp4
lsxfabric --localmodel phi-4 --explain-artifact
```

Expected: two candidates listed; on `nvidia:sm_120` the `nvfp4` one is chosen, on
`nvidia:sm_86` the `bf16` one — the first real demonstration of multi-artifact
autoselect.

- [x] **Step 5: Commit** — `feat: bootstrap the artifact store with the bf16 and nvfp4 slices`

### Task 9b: Per-layer codec map leaves `model.index.json`

> **PARTIAL (`a7873320` delivered the storage half; no commit migrated the
> readers).** Exists: `Artifact::layer_codec[]` in `artifact.fbs`; forge writes
> it (a uniform copy of the scanned mask, `main.cpp:590-591`); `ModelBase::artifact()`
> and `ModelBase::LayerUsesCodec(layer, codec)` (`model_base.cpp:151-158`); a case
> in `test_load_codec_gate.cpp:197` proving the answer comes from the artifact.
> **Does not exist:**
>
> | residue | where | owner |
> |---|---|---|
> | 5 converters still **write** codec keys | `inkling_moe_convert.cpp:1012`, `inkling_moe_loop_convert.cpp:1149`, `muse_dense_convert.cpp:341-342`, `gemma_dense_convert.cpp:385-387`, `gemma_moe_convert.cpp:336` | 9d |
> | 2 loaders still **read** `expert_nvfp4`/`expert_fp8` | `inkling_moe_load.cpp:131-143`, `inkling_moe_loop_load.cpp:146-158` | 9d |
> | the arrow verifier infers expert codec **from the index** (`LoadExpertCodec`) | `arrow_tree_verify.cpp:63-140` | 9c + 9d |
> | `--verify` has **no** rule-5 check | `lsxstorage/src/verify.cpp` (grep `model.index.json` → sidecar list only) | 9d |
>
> The plan's cost estimate here was wrong in both directions and is corrected by
> spec §2.4.3: the codec fields are read by **two files in one family** (inkling),
> not six files in two; but **five** converters write them, not four — `muse` and
> both `gemma` writers were missed.

**Why this exists.** `model.index.json` currently declares codecs
(`mlp_nvfp4`, `attn_nvfp4`, `head_nvfp4`, `expert_nvfp4[]`, `expert_fp8[]`) and six
files read them to pick a kernel per layer. That is a **second source of truth**
beside `artifact.fb`, and if the two disagree the gate passes a model that then
executes an undeclared codec — the exact hole I1 closes. Spec §2.4.3 rule 5.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h` — `const
  lsx::storage::ArtifactInfo *artifact() const`, `bool LayerUsesCodec(int,
  hw::PlaneCodec) const` (spec §2.4.6)
- Modify: `libs/lsxcommon/src/model_base.cpp` — store the `ArtifactInfo` from
  `BeginLoad` stage 2; clear it in `Unload`/`AbortLoad`
- Modify the 6 readers: `inkling_moe_load.cpp`, `inkling_moe_loop_load.cpp`,
  `inkling_moe_inference_pipeline.cu`, `inkling_moe_loop_inference_pipeline.cu`,
  `gemma_moe_load.cpp`, `gemma_moe_inference.cu`
- Modify the converters that write those fields: `inkling_moe_convert.cpp`,
  `inkling_moe_loop_convert.cpp`, `gemma_moe_convert.cpp`, `gemma_dense_convert.cpp`
  — stop emitting them, emit `ArtifactSpec::layer_codec` instead
- Test: `tests/test_layer_codec_source.cpp` (label `fast`)

**Interfaces:**
- Consumes: Tasks 5, 6, 9.
- Produces: `const lsx::storage::ArtifactInfo *ModelBase::artifact() const`;
  `bool ModelBase::LayerUsesCodec(int layer, hw::PlaneCodec) const`.

- [ ] **Step 1: Write the failing test:**

```cpp
TEST_CASE("an artifact whose model.index.json still declares codecs is rejected") {
    TempDir d; make_minimal_tree(d.path());
    write_index_with(d.path(), R"({"num_layers":2,"expert_nvfp4":[true,true]})");
    REQUIRE(EmitArtifact(minimal_spec(d.path()), d.path(), nullptr) != 0);
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("expert_nvfp4") != std::string::npos);
}
TEST_CASE("LayerUsesCodec answers from the artifact, not from model.index.json") {
    // artifact declares layer 0 = NVFP4, layer 1 = MXFP8
    FixtureArtifact f(FixtureKind::kMixedLayers);
    SetActiveTargetForTesting(TargetFromComputeCapability(12, 0));
    REQUIRE(f.load().ok);
    CHECK(f.model().LayerUsesCodec(0, PlaneCodec::kNvFp4));
    CHECK_FALSE(f.model().LayerUsesCodec(0, PlaneCodec::kMxFp8));
    CHECK(f.model().LayerUsesCodec(1, PlaneCodec::kMxFp8));
}
TEST_CASE("LayerUsesCodec is false before Load and after Unload") {
    FixtureArtifact f(FixtureKind::kMixedLayers);
    CHECK_FALSE(f.model().LayerUsesCodec(0, PlaneCodec::kNvFp4));
    f.load(); f.model().Unload();
    CHECK_FALSE(f.model().LayerUsesCodec(0, PlaneCodec::kNvFp4));
}
TEST_CASE("a mixed-layer artifact is still gated as a whole") {
    // the per-layer map must not become a way around the model-level gate
    FixtureArtifact f(FixtureKind::kMixedLayers);       // nvfp4 + mxfp8
    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    CHECK_FALSE(f.load().ok);                            // refused, as before
}
```

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** Remove the five codec keys from the four converters'
  index writers and populate `layer_codec` instead; replace the six readers'
  `root["expert_nvfp4"]…` / `mlp_nvfp4` lookups with `LayerUsesCodec(...)`, keeping
  their existing hard-failure behaviour for an out-of-range layer. `--verify` gains
  the rule-5 check (fail if any of the five keys is present).

- [ ] **Step 4: Run and confirm pass**, plus the two families' own suites:
  `ctest -R "inkling|gemma" -j1` (one GPU workload at a time).

- [ ] **Step 5: Commit** — `refactor: move the per-layer codec map from model.index.json into artifact.fb`

### Task 9c: An expert plane declares its codec — the verifier may not assume NVFP4

> **DONE (`85307f87`).** `PackedExpert::dtype` (default `NVFP4`) is written as a
> trailing `dtype` field of `ExpertSchema`/`SharedExpertSchema` (10 columns; the
> nine indices are a wire contract for the .cu staging offsets, so the column was
> appended, not prepended). `PackExpertNVFP4` sets `F8_E4M3` on the verbatim MXFP8
> passthrough, both inkling converters say so for their 8-bit layers, and the
> verifier reads the codec from that column — refusing the 9-column form outright.

**Why this blocks Branches 2 and 3.** `VerifyArtifact` and `ScanArtifactTree` take
a file's codec from its Arrow **schema shape** (`lsxstorage/src/verify.cpp`), and
`SchemaShape::kExpert` — the 9-column `ExpertSchema`
(`w1_weight … w3_global_scale`) — is hard-coded to `kNvFp4` with no dtype column
to consult. Consequences, all live:

1. A **BF16** expert plane (Branch 2's `bf16` profile for any MoE family) verifies
   as NVFP4, so the mask says `nvfp4` while the bytes are BF16 → the gate refuses
   the artifact on Ampere for a codec it does not contain, and forge's own
   cross-check (`scan.mask & ~promised`) refuses the conversion.
2. An **INT8** expert plane (Branch 3) is unrepresentable for the same reason.
3. A **mixed** NVFP4+MXFP8 expert tree (Inkling-2-Small, Inkling-2-Mini) verifies
   against a mask that is simply wrong — `--verify` cannot catch a codec drift on
   the one family that has a per-layer mix.

This is free to fix **now** and expensive later: the three existing artifacts
carry no expert planes, so no artifact is invalidated by changing the schema.

**Files:**
- Modify: `libs/lsxstorage/src/verify.cpp` (`ClassifyShape`, the `kExpert` branch
  of both `VerifyArtifact` and `ScanArtifactTree`)
- Modify: `libs/lsxcommon/src/converter.cpp` (the `ExpertSchema` writer) and the
  family expert writers that build that schema themselves
  (`inkling_moe_convert.cpp`, `inkling_moe_loop_convert.cpp`, `gemma_moe_convert.cpp`)
- Modify: `libs/lsxcommon/src/arrow_tree_verify.cpp` — `LoadExpertCodec` and the
  `elements_per_byte` map are deleted in favour of the declared column (Task 9d
  removes the last caller)
- Test: `tests/test_storage_verify.cpp` (extend, label `fast`)

**Interfaces:**
- Consumes: Task 5, 6, 7.
- Produces: `ExpertSchema` = the 9 weight columns **plus** `dtype:string`, one row
  per expert, values exactly the D1 spellings (`NVFP4`, `F8_E4M3`, `I8`, `BF16`);
  `ClassifyShape` returns `kExpert` only for the 10-column shape; a 9-column
  expert file is a verification **failure** naming the missing declaration (D1/D4:
  ambiguity is a conversion defect).

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST_CASE("a bf16 expert plane sets the bf16 bit, not nvfp4") {
    TempDir d;
    write_expert(d.path(), "experts/layer_0/exp_0.arrow", /*dtype=*/"BF16");
    const auto r = VerifyArtifact(d.path(), false);
    CHECK((r.actual_mask & Mask(PlaneCodec::kBF16)) != 0);
    CHECK((r.actual_mask & Mask(PlaneCodec::kNvFp4)) == 0);
}
TEST_CASE("a mixed expert tree reports both bits and both layers") {
    TempDir d;                       // layer 0 NVFP4, layer 1 E4M3
    write_expert(d.path(), "experts/layer_0/exp_0.arrow", "NVFP4");
    write_expert(d.path(), "experts/layer_1/exp_0.arrow", "F8_E4M3");
    CHECK(VerifyArtifact(d.path(), false).actual_mask ==
          (Mask(PlaneCodec::kNvFp4) | Mask(PlaneCodec::kMxFp8)));
}
TEST_CASE("a 9-column expert file is refused: it declares no codec") {
    TempDir d;
    write_legacy_expert(d.path(), "experts/layer_0/exp_0.arrow");
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("dtype") != std::string::npos);
}
```

- [ ] **Step 2: Run them and confirm failure** — the first two fail because
  `kExpert` ignores the column; the third fails because the 9-column shape is
  still accepted.

- [ ] **Step 3: Implement.** `ClassifyShape` requires all ten columns; the codec
  comes from the `dtype` column through the same `CodecForDtype()` the pinned path
  uses (so `"U8"` stays a failure, D1), with `IsCompanionName()` still deciding
  companions. Then write the column: one string per row, constant inside a file
  (a file is one plane). Families whose expert store is not the 9-column schema
  (`experts/layer_N/…` written by a bespoke writer) get the same column from their
  own writer — the rule is "one plane, one declared codec", not "one schema".

- [ ] **Step 4: Run and confirm pass** — `ctest -R "test_storage_verify|test_arrow_tree_integrity"`,
  then the two families' convert targets: `ctest -R "inkling|gemma" -R "convert" -j1`.
  The three existing artifacts must still verify (they ship no expert planes):
  `ctest -R test_bootstrap_artifacts`.

- [ ] **Step 5: Commit** — `feat: an expert plane declares its codec, so the verifier stops assuming NVFP4`

### Task 9d: Rule 5 is enforced, and the last index-codec readers go

> **DONE (`132b683c`), together with 9b's other half.** `--verify` parses
> `model.index.json` (mmap + simdjson keys, not a text scan) and fails on any of
> the five fields; the two inkling topology parsers stopped reading them and the
> models project `artifact.fb::layer_codec` through
> `lsxcommon/layer_codec_map.h`; the five writers (inkling ×2, gemma ×2, muse)
> stopped emitting them; and `ScanArtifactTree` now derives a **per-layer** map
> (forge used to record the tree mask repeated N times, which is a lie for a mixed
> tree). Found on real data: gemma-4-12B-it's index still carried three of the
> five keys — re-converted, verifies clean.

**Why.** Task 9b's whole point was that `model.index.json` must carry **no** codec
key (spec §2.4.3 rule 5), so `artifact.fb` cannot be contradicted. Today nothing
enforces it: `--verify` treats `model.index.json` as a sidecar and never opens it,
five converters still write the keys, two inkling loaders still read
`expert_nvfp4`/`expert_fp8`, and `arrow_tree_verify.cpp:63-140` derives an
`elements_per_byte` map from them — a second source of truth in the one place that
must not have one.

**Files:**
- Modify: `libs/lsxstorage/src/verify.cpp` — parse `model.index.json` (mmap +
  simdjson, the tree's existing JSON path) and fail on any of
  `mlp_nvfp4`, `attn_nvfp4`, `head_nvfp4`, `expert_nvfp4`, `expert_fp8`
- Modify: `libs/lsxcommon/src/inkling_moe_load.cpp:131-143`,
  `inkling_moe_loop_load.cpp:146-158` → `LayerUsesCodec(layer, …)`
- Delete: `libs/lsxcommon/src/arrow_tree_verify.cpp`'s `LoadExpertCodec` +
  `ExpertCodecMap` (their codec knowledge came from the index)
- Modify: the five converters listed in Task 9b's table — stop emitting the keys
- Test: `tests/test_storage_verify.cpp`, `tests/test_layer_codec_source.cpp`
  (create, label `fast`), the inkling/gemma `convert`+`load`+`model` targets

**Interfaces:**
- Consumes: Tasks 7, 9c.
- Produces: no new API. `VerifyReport::problems` gains the exact key name, e.g.
  `"model.index.json still declares codec 'expert_nvfp4': the codec map lives in
  artifact.fb's layer_codec"`.

- [ ] **Step 1: Write the failing test** (the case the plan put in Task 9b's step 1,
  which was never written):

```cpp
TEST_CASE("an artifact whose model.index.json still declares codecs is rejected") {
    TempDir d; make_minimal_tree(d.path());
    write_index_with(d.path(), R"({"num_layers":2,"expert_nvfp4":[true,true]})");
    REQUIRE(EmitArtifact(minimal_spec(d.path()), d.path(), nullptr) != 0);
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("expert_nvfp4") != std::string::npos);
    CHECK(r.problems[0].find("layer_codec") != std::string::npos);
}
```

- [ ] **Step 2: Run and confirm failure** — the artifact verifies clean today.

- [ ] **Step 3: Implement.** Readers first (so the writers cannot remove a key that
  something still needs): the inkling loaders ask `LayerUsesCodec`; the
  `arrow_tree_verify` exponent map is deleted once 9c lands the declared column.
  Then stop writing the keys. Then the `--verify` rule, with the message above.

- [ ] **Step 4: Run and confirm pass**, then the two families on their own labels
  (one GPU workload at a time): `ctest -R "inkling|gemma" -j1`. A red here is a
  family defect the migration exposed, not a reason to relax the rule.

- [ ] **Step 5: Commit** — `refactor: the codec map lives only in artifact.fb, and --verify enforces it`

### Task 9e: Target-aware profile selection + `--explain-artifact`

> **DONE (`dc19aed9`).** `lsx::storage::SelectProfile(org, model_dir, supported,
> preference)` keeps the capability as a mask and the rank as a name list;
> `Resolve(path, supported, preference)` prefers a compatible profile for a MODEL
> and never re-ranks an explicit profile; `BeginLoad` supplies
> `SupportedCodecs(ActiveTarget())` and `hw::CodecProfilePreference()`
> (int8-w8a16, nvfp4, mxfp8, bf16 — the one place spec §8.2 says to correct from
> measurement); `lsxfabric --explain-artifact` prints the target mask, every
> candidate's verdict and rank, and the choice, without loading a model.

**Why.** Invariant I4 says `lsxfabric` **selects** the best compatible artifact. The
storage half ships; the selection half does not, and the code is actively
misleading about it:

- `lsx::storage::Resolve(model)` picks *"the first profile that IS an artifact"*
  by iterating `fs::directory_iterator` — unspecified order — while the comment
  beside it claims *"in the same sorted order ListProfiles uses"*
  (`lsxstorage/src/resolve.cpp:135-154`). With one profile per model this is
  invisible; with two it is filesystem-dependent.
- Nothing consults `SupportedCodecs(ActiveTarget())` when choosing. The gate then
  refuses the arbitrarily chosen profile and prints the usable list — a good
  error, but not a selection (open task `V`).
- `lsxfabric` has no `--explain-artifact` flag (spec §2.7 names it as the only new
  diagnostic flag), so "why did it pick that one" is unanswerable from the CLI.

**Files:**
- Modify: `libs/lsxstorage/include/lsxstorage/resolve.h`, `src/resolve.cpp` — add
  `SelectProfile`, fix `Resolve`'s model-name branch, delete the false comment
- Modify: `libs/lsxcommon/src/model_base.cpp` — `BeginLoad`'s resolve stage calls
  `SelectProfile` with the active mask and the preference table
- Modify: `apps/lsxfabric/src/main.cpp` + `cli.{h,cpp}` — `--explain-artifact`
- Test: `tests/test_storage_resolve_profile.cpp` (create, label `fast`, no GPU)

**Interfaces:**
- Consumes: Tasks 4, 5, 9.
- Produces:
  `std::optional<Resolved> SelectProfile(const std::string &org, const std::string &model_dir, hw-codec-mask-as-uint64_t supported, const std::vector<std::string> &preference);`
  — storage stays codec-agnostic by taking the mask and an ordered name list;
  `lsxcommon` supplies them (`SupportedCodecs(ActiveTarget())` and
  `{"nvfp4","int8-w8a16","mxfp8","bf16"}` — operator ruling: compressed-native
  first on the 5090). Ties break on profile name ascending, so two runs agree.

- [ ] **Step 1: Write the failing test:**

```cpp
TEST_CASE("selection prefers the compressed profile the target can execute") {
    TempDir root; make_profile(root.path(), "microsoft/phi-4", "bf16",  kBF16);
    make_profile(root.path(), "microsoft/phi-4", "nvfp4", kNvFp4|kBF16);
    const auto blackwell = SelectProfile("microsoft", "phi-4",
        Mask(PlaneCodec::kF32)|Mask(PlaneCodec::kBF16)|Mask(PlaneCodec::kNvFp4),
        kPreference);
    REQUIRE(blackwell.has_value());
    CHECK(blackwell->profile == "nvfp4");
    const auto ampere = SelectProfile("microsoft", "phi-4",
        Mask(PlaneCodec::kF32)|Mask(PlaneCodec::kBF16), kPreference);
    REQUIRE(ampere.has_value());
    CHECK(ampere->profile == "bf16");
}
TEST_CASE("no compatible profile is nullopt, never the first one on disk") {
    TempDir root; make_profile(root.path(), "microsoft/phi-4", "nvfp4", kNvFp4);
    CHECK_FALSE(SelectProfile("microsoft", "phi-4", Mask(PlaneCodec::kBF16),
                              kPreference).has_value());
}
TEST_CASE("the same store answers the same profile twice") { /* 10 iterations */ }
TEST_CASE("an explicit profile path is a choice and is never re-ranked") {
    TempDir root; make_profile(root.path(), "microsoft/phi-4", "bf16", kBF16);
    make_profile(root.path(), "microsoft/phi-4", "nvfp4", kNvFp4|kBF16);
    const auto r = Resolve((root.path()/"microsoft/phi-4/bf16").string());
    REQUIRE(r.status == ResolveStatus::kArtifact);
    CHECK(r.profile == "bf16");            // even with nvfp4 present
}
```

- [ ] **Step 2: Run and confirm failure** — `SelectProfile` does not exist; the
  fourth case already passes and must keep passing.

- [ ] **Step 3: Implement.** `ListProfiles` already returns a name-sorted vector of
  `Resolved`; `SelectProfile` reads each candidate's `used_codecs`, keeps
  `used & ~supported == 0`, sorts by `(preference index, profile name)` and returns
  the head. `Resolve`'s model-name branch delegates to it with the preference table
  from its caller — storage takes the list as a parameter, so it has no opinion of
  its own. `--explain-artifact` prints one line per candidate
  (`profile mask=… compatible=yes/no rank=N`) then the choice, and exits 0 without
  loading a model.

- [ ] **Step 4: Run and confirm pass** — `ctest -R "test_storage_resolve_profile|test_load_codec_gate|test_storage_roots"`,
  then `lsxfabric --localmodel phi-4 --explain-artifact` on this host (one
  candidate until Task 9f produces a second).

- [ ] **Step 5: Commit** — `feat: select an artifact profile by target capability, and explain the choice`

### Task 9f: Grow the artifact store beyond three models

**Why this is now the largest open item in the plan.** Measured 2026-11-28: **3 of
48** registry rows have an artifact, so **45 rows'** real-weight stages SKIP — and
per AGENTS.md a gate that SKIPs is not a pass. Every stage target exists
(`docs/MODELS.md` declares all 10 for all 48 rows), the goldens exist or are
declared debt, and nothing can run. The plan's Task 9a was written to end this with
two models; that was the right *first* step, not the end state.

**Files:**
- Modify: `models/<org>/<model>.json` — `profiles.<p>.status`/`artifact`/`gates`
  for each model converted
- Modify: `docs/SELFCHECK_MATRIX.md`, `docs/MODELS.md` (regenerate via
  `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`)
- Modify: `tests/perf/perf_baselines.json` — re-seed the perf row for a model
  whose artifact was reconverted (a row that predates its artifact is a claim about
  a tree that no longer exists)
- Test: `tests/test_registry_store_consistency.cpp` (create, label `fast`,
  `SKIP` when the artifact root is unset)

**Interfaces:**
- Consumes: Tasks 8a, 9c, 9d (a conversion that publishes nothing, or a verifier
  that mislabels an expert plane, would put wrong rows in the registry).
- Produces:
  - `tools/gen_model_manifest.py` + `cmake/LsxModelManifest.cmake` emit a **new
    table** `lsxmanifest::kProfiles[]`:
    `struct ProfileRow { const char *model_type; const char *profile; const char *status; unsigned long long codec_mask; const char *store_path; };`
    Today the generated table carries only `vault_profile` (the *preferred* one),
    so no compiled guard can see that a model has two profiles — which is why the
    registry can drift from the store unnoticed.
  - No runtime API. The guard below is the deliverable that keeps the store and the
    registry from drifting.

- [ ] **Step 0: Extend the generator first**, because the guard cannot be written
  against a table that does not exist: one `kProfiles[]` row per
  `models/<org>/<model>.json` → `profiles.<name>`, with `status`, `codec_mask` and
  `artifact.store` (empty for `not-built`). `test_manifest_generator_selftest`
  gets a case for the new table, and `test_manifest_sync` already fails on drift
  between the compiled table and the manifests.

- [ ] **Step 1: Write the failing guard** — the registry can currently claim
  `status: built` with no artifact on disk, or ship an artifact no row mentions:

```cpp
TEST_CASE("every row that claims a built profile has that artifact, and vice versa") {
    if (!lsx::storage::ArtifactRoot()) SKIP("LSX_ARTIFACT_ROOT unset");
    for (std::size_t i = 0; i < lsxmanifest::kProfileCount; ++i) {
        const auto &p = lsxmanifest::kProfiles[i];
        const auto *row = manifest_query::FindModel(p.model_type);
        REQUIRE(row != nullptr);
        const auto dir = *lsx::storage::ArtifactRoot() / row->vault_org /
                         row->vault_dir / p.profile;
        const bool on_disk = IsArtifactDir(dir);
        INFO(p.model_type << " " << p.profile << " " << p.status);
        CHECK(on_disk == (std::string_view(p.status) == "built"));
        if (!on_disk) continue;
        const auto r = ReadArtifact(dir);
        REQUIRE(r.status == ReadStatus::kOk);
        CHECK(r.info.used_codecs == p.codec_mask);   // the row cites the mask
    }
}
TEST_CASE("the perf row of a built profile is not older than its artifact") {
    if (!lsx::storage::ArtifactRoot()) SKIP("LSX_ARTIFACT_ROOT unset");
    // artifact.fb's created_utc vs perf_baselines.json's date for that model
}
```

- [ ] **Step 2: Run and confirm failure** — phi-4 and Qwen3.5-9B carry perf rows
  dated 2026-10-24 / 2026-11-05 against artifacts created 2026-11-28, so the
  second case fails, which is the point. (Their `nvfp4` codec layout is unchanged,
  so re-seeding is a re-run, not a new measurement — say so in the commit.)

- [ ] **Step 3: Convert, in this order, recording each row as it lands**
  1. `bf16` for `phi-4` and `Qwen3.5-9B` — blocked on Task 15, and the artifact
     Branch 2 needs.
  2. The cheap dense families that already have a full lifecycle suite and small
     weights (`dory-1.0-4b`, `granite-switch-4.1-8b`) — width of coverage, not
     depth.
  3. The MoE families (`inkling`, `gemma`, `qwen35moe`, `ornith`) after Task 9c,
     because their expert planes need the declared codec — and the eight families
     in Task 8b cannot be converted at all until their `mtp/` write is gone.
  4. The large rows last, and only when a gate needs them: a conversion is
     minutes-to-hours and the artifact is target-independent, so there is no
     ordering hazard (spec §1.9).

- [ ] **Step 4: Run and confirm pass** — the guard green, then the new rows' own
  stage targets (`ctest -R "<model_type>" -j1`, one GPU workload at a time), then
  `ctest --preset linux-debug` for the registry/generator guards.

- [ ] **Step 5: Commit** — `feat: grow the artifact store and guard it against the registry`

### Task 10: `MemoryModel` (unified memory) in the governor

> **DONE (`8effeaf4`).** `MemoryModel{kDiscrete,kUnified}` is probed from
> `prop.integrated` through `HardwareQueries::unified_shared`; on unified memory
> `GrantableVramLocked()` returns the shared pool and all lease kinds charge ONE
> counter (a `BudgetFor()` view shared by Acquire/Resize/Release, so a lease is
> never returned to a counter it was not taken from). Two overloads of
> `ExpertCacheReserveBytes`: `(budget)` is the spec's min(8 GiB, 25%);
> `(HardwareProfile)` is what the three expert-cache call sites use and returns
> the calibrated constant EXACTLY for kDiscrete — min() applied to a 5090's free
> reading would move every baseline.
>
> **What the original note said, for the record.** `grep -n "MemoryModel\|unified_shared_bytes"
> libs/lsxcommon/{include/lsxcommon/resource_governor.h,src/resource_governor.cpp}`
> is empty. **Anchors as of today:** `HardwareProfile` is
> `resource_governor.h:12-20`; `HardwareQueries` (the injectable probes) is
> `:24-29`, so a `unified_shared_bytes` probe must be added there too or the unit
> tests cannot substitute a fake; `lifecycle::kExpertCacheVramReserveBytes` is
> `:181` and the `ExpertCacheSlots(..., reserve)` parameter that consumes it is
> `:192`. Nothing in `lifecycle_strategy.cpp` names the constant — the two call
> sites are `resource_governor.h` itself and the tests, so the plan's third file
> is a no-op as the tree stands.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/resource_governor.h`
  (`enum class MemoryModel`, `HardwareProfile::memory_model`,
  `unified_shared_bytes`, `ExpertCacheReserveBytes(budget)`)
- Modify: `libs/lsxcommon/src/resource_governor.cpp` (probe + budget arithmetic)
- Modify: `libs/lsxcommon/src/lifecycle_strategy.cpp` (use the new reserve function)
- Test: `tests/test_resource_governor.cpp` (extend, label `fast`)

**Interfaces:**
- Produces: `MemoryModel{kDiscrete,kUnified}`;
  `std::uint64_t ExpertCacheReserveBytes(std::uint64_t budget)` returning
  `min(8 GiB, budget / 4)`; `kDiscrete` keeps `kExpertCacheVramReserveBytes` exactly.

- [ ] **Step 1: Write the failing test:**

```cpp
TEST_CASE("discrete reserve is unchanged at 8 GiB for a 32 GB card") {
    REQUIRE(ExpertCacheReserveBytes(32ull<<30) == (8ull<<30));
}
TEST_CASE("a unified 16 GB budget reserves a quarter, not half") {
    REQUIRE(ExpertCacheReserveBytes(16ull<<30) == (4ull<<30));
}
TEST_CASE("unified memory is detected from the device, never guessed from the name") {
    HardwareProfile p; p.memory_model = MemoryModel::kUnified;
    p.unified_shared_bytes = 16ull<<30; p.vram_free_bytes = 8ull<<30;
    GovernorOptions o; o.override_hw = true; o.hw_override = p;
    auto &g = ResourceGovernor::Instance(); g.Initialize(o);
    CHECK(g.hardware().memory_model == MemoryModel::kUnified);
    g.Shutdown();
}
TEST_CASE("a permanent-weights lease on unified memory is counted once") {
    // budget 16 GiB; a 10 GiB weight lease then a 10 GiB pinned lease must fail.
}
```

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** Probe `prop.integrated`; on unified, `GrantableVramLocked()`
  uses `unified_shared_bytes` and shares one budget with RAM leases.

- [ ] **Step 4: Run and confirm pass** — `ctest -R "test_resource_governor|test_lifecycle|test_vram"`.

- [ ] **Step 5: Commit** — `feat: model unified memory in the resource governor`

**Branch 1 exit criteria — measured 2026-11-28:**

| criterion | state |
|---|---|
| `ctest --preset linux-debug` green (all `fast` guards) | **met** — 184/184, 166 s, 11 legitimate skips |
| Task 9a: two artifacts converted, verified, gating on `sm_120` and `sm_86` | **partly met** — three `nvfp4` artifacts exist and verify; the *Ampere* half is unproven because no `bf16` profile is producible (Task 15) |
| Task 9a step 3: the gates that were dark now RUN (SelfCheck ×3, FSI, CODE, DET, MCP) | **met for 3 of 48 rows** — `phi-4`, `qwen3.5-9b` all green; `gemma-4-12b-it` FSI red (documented source near-tie, `R22`) |
| Task 9a step 4: `nvfp4` sibling + `--explain-artifact` shows the target-dependent pick | **not met** — no second profile and no flag (Task 9e, Task 9f) |
| Task 9b: no `model.index.json` carries a codec key; inkling/gemma read the artifact | **not met** — 5 writers, 2 readers, 1 verifier (Task 9d) |
| Task 4a: path-shape guard green, artifacts at `<ROOT>/<org>/<model_dir>/<profile>/` | **met** — 6-case guard, 3 artifacts in that shape |
| spec committed | **met** |
| `docs/numeric-contract.md` §1/§1.1 amended | **not met** — no codec/profile scoping in the contract yet (moved into Task 16, which owns the doc amendments) |
| `docs/OPEN_TASKS.md` records the cold start and the artifact root | **met indirectly** — items `V` and `W` state the measured store position; the plan's own evidence lives here and in `docs/SELFCHECK_MATRIX.md` |

So: the *code* half of Branch 1 is done and green; the half that needed real
artifacts is done for three models and is the input to Branch 2. **Branch 1 is
closed with three carry-over tasks (9c, 9d, 9e, 8a) and one coverage task (9f),
not reopened.**

---

# Branch 2 — `build/ampere-aot-sm86-sm87`

> **NOT STARTED (2026-11-28).** No commit exists for Tasks 11–16: the build accepts
> only `120`/`121`, both arch-`a` object libs are unconditional, the toolkit guard
> asserts Blackwell by name, there are no Ampere presets, no converter emits `bf16`,
> and `docs/NVIDIA_DOCS_AMPERE.md` — which Task 16 was going to amend — has been
> removed from the tree. Each task below carries a *status* note with the exact
> current anchors and the traps the intervening commits created.

Branches from Branch 1 (needs `SupportedCodecs`). Scope: build + `bf16` profile +
live 3090 acceptance. **No new GEMM** (operator option c).

### Task 11: `LSX_CUDA_ARCH` accepts Ampere and emits both SASS

> **NOT STARTED (2026-11-28).** The validator is `CMakeLists.txt:38-53`: one regex
> `^12[01][af]?$`, a FATAL naming Blackwell only, and
> `--generate-code=arch=compute_${arch},code=[compute_${arch},sm_${arch}]` per
> entry. A `86;87` configure fails today with the Blackwell-only message.
>
> **Anchors and traps as the tree stands:**
>
> - The acceptance rule must be added **inside `lsx_cuda_gencode_flags()`** and
>   nowhere else: `test_infrastructure_guard` fails any `CUDA_ARCHITECTURES` that
>   carries an arch spelling and any literal `--generate-code=` outside this
>   function, which is the mechanism that keeps CMake 4.x from substituting
>   `compute_75`.
> - `tests/test_infrastructure_guard.cpp:572` asserts the default literal
>   `set(LSX_CUDA_ARCH "120;121"` still in the file — keep the default and add the
>   Ampere case beside it, do not change the default.
> - `FindNonFamilyGencodeLiterals` (`:517-537`) matches only
>   `compute_(12[0-9])`, so a hand-rolled `--generate-code=arch=compute_86` would
>   slip past the guard. Widen the pattern to `(12[0-9]|8[67])` **in the same
>   commit**, or the new arch gets a second, unchecked way in.
> - `sm_86a`/`sm_87a`/`sm_86f` do not exist (no `-a` family on GA10x): a suffix on
>   an 8x entry must fail with its own message, not fall into the Blackwell text.
> - Suggested regex: `^(12[01][af]?|8[67])$`, FATAL text
>   *"supported: 120/121 (Blackwell consumer), 86 (GA102), 87 (GA10B)"*.

**Files:**
- Modify: `CMakeLists.txt:35-54`
- Modify: `tests/test_infrastructure_guard.cpp` (its expectations mention `12[01]`)
- Test: `tests/test_infrastructure_guard.cpp` (extend, label `fast`)

- [ ] **Step 1: Failing test** — a CMake-level test that configures a scratch binary
  dir with `-DLSX_CUDA_ARCH=86;87` and asserts configure succeeds and the emitted
  flags contain **both** `code=sm_86` and `code=sm_87`:

```cpp
TEST_CASE("LSX_CUDA_ARCH accepts 86;87 and emits both SASS images") {
    const auto r = configure_scratch({"86;87"});
    REQUIRE(r.ok);
    CHECK(r.gencode_flags.find("code=sm_86") != std::string::npos);
    CHECK(r.gencode_flags.find("code=sm_87") != std::string::npos);
}
TEST_CASE("a forbidden arch still fails configure with a named reason") {
    for (const char* a : {"90","89","80","75"}) {
        const auto r = configure_scratch({a});
        CHECK_FALSE(r.ok);
        CHECK(r.err.find("supported") != std::string::npos);
    }
}
```

- [ ] **Step 2: Run, confirm failure** (`FATAL_ERROR: not a Blackwell consumer target`).

- [ ] **Step 3: Implement.** Accept `^1(2[01]|8[67])[af]?$` with a per-family suffix
  rule: `f`/`a` allowed only on 12x (there is no `sm_86a`/`sm_86f`). Update the
  FATAL message to "supported: 120/121 (Blackwell), 86/87 (Ampere)".

- [ ] **Step 4: Run, confirm pass**; also re-run the unchanged `linux-debug` configure.

- [ ] **Step 5: Commit** — `build: accept Ampere sm_86/sm_87 in LSX_CUDA_ARCH`

### Task 12: arch-`a` object libs are emitted only for 12x

> **NOT STARTED (2026-11-28).** Both object libs hard-code the `a` targets:
> `libs/lsxcommon/CMakeLists.txt:651-671` (`lsxcommon_tc_cu`,
> `--generate-code=arch=compute_120a,code=sm_120a;--generate-code=arch=compute_121a,code=sm_121a`)
> and the test-side twin at `tests/CMakeLists.txt:1440-1460`. Neither is
> conditional, so an `86;87` configure asks nvcc for `compute_120a` on a
> toolchain/target set that does not contain it.
>
> **Anchors:** the three `prop.major` probes that are the only callers are
> `lsx_tc_gemm.cu:908`, `:1086`, `:1148` — they return `cudaErrorNotSupported` /
> `false` off-12x, which is why the link can succeed without the object libs.
> The `nm` check the plan asks for is on the final `lsxbenchmark` binary of an
> 86-only build: `nm -C | grep lsx_tc` must be empty.
>
> **As the guard stands:** `test_infrastructure_guard` deliberately *sanctions*
> literal `compute_12xa` lines (`:545-556`), so wrapping them in an `if()` does not
> trip it. Add the new assertion ("`lsxcommon_tc_cu` absent when no 12x arch is
> targeted") to that same guard, because a configure-scratch assertion cannot see a
> target-registration decision any other way.

**Files:**
- Modify: `libs/lsxcommon/CMakeLists.txt:640-660`
- Modify: `tests/CMakeLists.txt:1377-1392`
- Test: `tests/test_infrastructure_guard.cpp` (extend)

- [ ] **Step 1: Failing test** — a configure-scratch assertion that with
  `LSX_CUDA_ARCH=86;87` the target `lsxcommon_tc_cu` does **not** exist, and with
  `120;121` it does.

- [ ] **Step 2: Run, confirm failure** (the literal `120a` flags break the 86/87
  configure: `nvcc fatal: Unsupported gpu architecture 'compute_120a'` for a
  pre-12.0 toolkit, or an unused-arch object on a 12-x toolkit targeting only 86/87).

- [ ] **Step 3: Implement** with a `lsx_arch_has_12x()` helper returning true iff any
  `LSX_CUDA_ARCH` entry starts with `12`; wrap both object libs in
  `if(lsx_arch_has_12x())`. The `lsx_tc` symbols must not be referenced outside
  `prop.major == 12` probes — verify by `nm` on the final binary in a 86-only build.

- [ ] **Step 4: Run, confirm pass** for both arch lists.

- [ ] **Step 5: Commit** — `build: emit arch-a tensor-core objects only when a 12.x arch is targeted`

### Task 13: `test_cuda_toolkit_guard` becomes toolchain-agnostic

> **NOT STARTED (2026-11-28).** Four concrete blockers, all in
> `tests/test_cuda_toolkit_guard.cu`:
>
> | line | today | why it breaks on 12.6 / Ampere |
> |---|---|---|
> | `5` | `#include <cuda_fp4.h>` unconditional | the TU does not even compile under CUDA 12.6, so the planned `SKIP` can never run — the include must move **inside** `__has_include` |
> | `50-51` | `CUDART_VERSION >= kMinCudaRuntime` **and** `CUDART_VERSION / 1000 == 13` | the major-equality assertion is a Blackwell/CUDA-13.5 assumption |
> | `95-97` | `REQUIRE(prop.major == 12)` for every present device | the supported set is `{12.0, 12.1, 8.6, 8.7}` |
> | `59-93` | three `cuda_fp4.h` cases | keep verbatim, but behind `__has_include` |
>
> **Refinement:** the device case must be written against
> `lsxcommon::hw::TargetFromComputeCapability(prop.major, prop.minor)` and assert
> `vendor != Vendor::kUnknown`, not against a literal `major ==`. Then the guard
> and the capability table cannot disagree. The driver-stamp cases
> (`:104-172`) are toolkit-version arithmetic and stay unchanged — they are what
> catches a stale build dir after a toolkit upgrade, and that matters more on a
> second machine than on this one.

**Files:**
- Modify: `tests/test_cuda_toolkit_guard.cu:14,50-51,59-85,88-96`
- Test: itself

- [ ] **Step 1: Failing tests** — rewrite the four assertions:

```cpp
TEST_CASE("cuda_toolkit_guard: runtime floor 12.6, any major") {
    REQUIRE(CUDART_VERSION >= 12060);
}
TEST_CASE("cuda_toolkit_guard: cuda_fp4.h cases only where the header exists") {
#if __has_include(<cuda_fp4.h>)
    // the three existing e2m1/pack cases stay verbatim
#else
    SKIP("cuda_fp4.h is absent before CUDA 12.8 (JetPack 6.2.1 = 12.6)");
#endif
}
TEST_CASE("cuda_toolkit_guard: present devices are in the supported set") {
    // {12.0, 12.1, 8.6, 8.7}; anything else fails and names itself
}
```

- [ ] **Step 2: Run, confirm failure** on the 13.5 host (the `== 13` assertion still
  passes there, but the new fp4 case must compile with `__has_include` and the
  supported-set case must fail against the current `prop.major == 12`).

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run, confirm pass** on the 5090.

- [ ] **Step 5: Commit** — `test: make the CUDA toolkit guard toolchain- and arch-agnostic`

### Task 14: Ampere build presets + the runtime degradation audit

> **NOT STARTED (2026-11-28).** `CMakePresets.json` has six configure presets
> (`windows-debug`, `windows-release`, `linux-debug`, `linux-release`,
> `linux-release-znver4`, `linux-release-znver4-aocc`) and none sets
> `LSX_CUDA_ARCH`. `cmake/toolchains/` does not exist.
>
> **Anchors and traps:**
>
> - A configure preset in this file is not self-contained: `CMakePresets.json`
>   also carries test presets that name each build preset, and `ctest --preset`
>   resolves through them. A new build preset without its test presets (at least
>   `-fast`, `-inference`, `-convert` + the plain one) is a half-added preset.
> - The Jetson toolchain file must pin `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc`,
>   `LSX_LTO=OFF`, `LSX_ARCH_ZNVER4=OFF`, no `-march`, and must **not** inherit the
>   AOCC preset's x86-only `CMAKE_INSTALL_RPATH` entries.
> - The degradation audit is three probes (`lsx_tc_gemm.cu:908, :1086, :1148`);
>   each already returns `cudaErrorNotSupported`/`false` off-12x. Record in the
>   header comment that with the Branch-1 gate an NVFP4 artifact cannot reach an
>   Ampere device at all, so these are defence in depth, not the gate.
> - Keep the accepted-arch regex in **one** file that both the CMake function and
>   `tests/test_arch_support_guard.cpp` read (a small `cmake/LsxArch.cmake` holding
>   the pattern as a variable). Two hand-copied regexes drift the moment someone
>   adds `89`.

**Files:**
- Modify: `CMakePresets.json` (add `linux-release-sm86`, `jetson-orin-release`)
- Create: `cmake/toolchains/jetson-orin.cmake` (CUDA 12.6 root, `aarch64`, no LTO,
  no `-march`)
- Modify: `libs/lsxcommon/src/lsx_tc_gemm.cu` (comment the probe contract)
- Test: `tests/test_arch_support_guard.cpp` (label `fast`) — asserts
  `SupportedCodecs(nvidia:sm_86)` is non-empty (pins the policy against a future
  revert of AGENTS.md)

- [ ] **Step 1: Failing test** — the guard test above plus a preset-parsing test that
  asserts every preset's `LSX_CUDA_ARCH` is accepted by the validator.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** presets + toolchain file. Audit the three
  `prop.major != 12` probes: each must return `cudaErrorNotSupported` (capability),
  and every caller must tolerate it (the two dispatch sites already do at
  `lsx_inference_nvfp4.cu:957-975,1179`). Record in the header comment that with the
  Branch-1 gate an NVFP4 artifact cannot reach an Ampere device at all, so these
  probes are defence in depth, not the gate.

- [ ] **Step 4: Run, confirm pass** — `ctest -R "test_arch_support_guard|test_infrastructure_guard"`.

- [ ] **Step 5: Commit** — `build: add sm_86 and JetPack 6.2.1 presets with an arch-support guard`

### Task 15: `bf16` codec profile for the 2-model slice

> **NOT STARTED (2026-11-28) — and unimplementable as written until one thing
> moves.** `grep -rn "CodecProfile" libs/lsxcommon/src` is empty: the two dense
> converters quantize unconditionally, and the enum they would branch on lives in
> **`apps/lsxforge/src/codec_profiles.h`**, which `libs/` cannot include.
>
> **Refinement 1 — the enum moves down.** Put `enum class CodecProfile` in
> `libs/lsxcommon/include/lsxcommon/codec_profile.h` (POD, `<cstdint>` only, safe
> from `.cu`). `lsxforge` keeps the *policy* it owns (`DeclaredPlanes`,
> `ParseCodecProfile`, the CLI) and re-exports the enum. Without this the
> converter signature cannot express a profile and the task turns into a
> per-family copy of the table — the exact thing spec §2.6 forbids.
>
> **Refinement 2 — the branch point is one predicate, not the whole converter.**
> In `phi_dense_convert.cpp` the decision is `phi_tensor_is_nvfp4(kind)`
> (`:202-213`, used at `:383` and `:508/:513`); in `qwen_dense_convert.cpp` the
> twin is `qwen_tensor_is_nvfp4` (`:206`, used at `:606`). The `bf16` profile
> makes that predicate answer `false` for every kind, so weights flow through the
> existing `PinnedSchema` passthrough with `dtype = "BF16"` — **no quantizer, no
> new schema, no new Arrow batch type**. Entry points to change:
> `ConvertPhiModel(input, output)` (`phi_dense_convert.h:53`, call site
> `main.cpp:946`) and its Qwen twin take a `CodecProfile`.
>
> **Refinement 3 — the mask must come out right the first time.** The scanned mask
> for a `bf16` tree is `bf16,f32`; if the converter keeps writing
> `pinned/global_nvfp4.arrow` the scan reports `nvfp4` and forge's own cross-check
> refuses the conversion at the end of a multi-hour run. So the plan's step 1 case
> (`CHECK_FALSE(fs::exists(a.dir/"pinned"/"layer_0_nvfp4.arrow"))`) is the cheap
> canary — run it on a 2-layer fixture **before** touching the real checkpoint.

**Files:**
- Modify: `libs/lsxcommon/src/phi_dense_convert.cpp`, `qwen_dense_convert.cpp`
  (accept `CodecProfile::kBf16`; write BF16 planes, no `_nvfp4` files)
- Modify: `apps/lsxforge/src/codec_profiles.h` (enable `kInt8W8A16` = still empty;
  enable `kBf16` for `PipelineKind::kDense`)
- Test: `tests/test_phi_dense_convert.cpp`, `tests/test_qwen_dense_convert.cpp`
  (extend, label `convert`)

- [ ] **Step 1: Failing test** — convert a 2-layer synthetic fixture in both profiles
  and assert the plane inventory and masks:

```cpp
TEST_CASE("phi bf16 profile produces no nvfp4 plane and a bf16-only mask") {
    const auto a = convert_fixture(CodecProfile::kBf16);
    REQUIRE(a.ok);
    CHECK(a.artifact_mask == (Mask(PlaneCodec::kBF16)|Mask(PlaneCodec::kF32)));
    CHECK_FALSE(fs::exists(a.dir/"pinned"/"layer_0_nvfp4.arrow"));
    CHECK(VerifyArtifact(a.dir, true).ok);
}
TEST_CASE("a bf16 artifact of a model with an FP8 MTP head carries no mxfp8 bit") {
    // GigaChat4-37B/708B shape: the MTP head is F8_E4M3 and inference never reads it.
    const auto a = convert_fixture_with_mtp(CodecProfile::kBf16, /*fp8_mtp=*/6);
    REQUIRE(a.ok);
    CHECK((a.artifact_mask & Mask(PlaneCodec::kMxFp8)) == 0);
    CHECK_FALSE(fs::exists(a.dir/"mtp"));   // v1 ships no ballast
    CHECK(VerifyArtifact(a.dir, true).ok);  // and verify agrees
    // And it therefore gates clean on Ampere, which the nvfp4 sibling does not.
    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    CHECK(CheckCodecs(a.artifact_mask, ActiveTarget()).allowed);
}
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** the BF16 write path in both converters (the source is
  already BF16 for these two families, so this is a pass-through plane, not a
  quantizer). Any `mtp/` write in the family's converter is deleted here, not
  gated — v1 has no ballast for any profile (spec §1.4), and Task 8 already removed
  the shared path.

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit** — `feat: add the bf16 codec profile to the phi and qwen dense converters`

### Task 16: Live 3090 acceptance + documentation amendments

> **NOT STARTED (2026-11-28), and one file in its list no longer exists.**
> `docs/NVIDIA_DOCS_AMPERE.md` was **removed from the repository** as NVIDIA
> copyrighted material (`docs/NVIDIA_DOCS.md` records the removal and the
> decision); editing `:22-27` of it is impossible and re-adding it would undo a
> deliberate legal choice. The offline export the operator keeps is
> `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_AMPERE.md` — the same 2026-09
> snapshot §A1–A8 of which this plan already cites for sm_86/sm_87 limits.
>
> **Where the amendments actually go:**
>
> | amendment | file |
> |---|---|
> | Ampere is a supported target (policy) | `AGENTS.md` — **operator-reviewed text, applied by them**, as the plan already says |
> | what Ampere executes and why (first-party words) | a new `docs/ampere-support.md`: the capability table row, the `mma.m16n8k16`/IMMA reasons, the 99 KB sm_86 / 163 KB sm_87 smem tiers, no FP8/FP4 |
> | R3 scoping to a target set | `docs/numeric-contract.md` §1/§1.1 (spec §4's amendment — still unapplied, and this is the task that owns it) |
> | per-profile goldens | `docs/numeric-contract.md` §4 + Task 21's resolver |
> | measured numbers | `docs/model-reports/2026-11-24-ampere-bf16-slice.md` + `tests/perf/perf_baselines.json` |
>
> **Do not** reproduce NVIDIA text in the repo: `docs/NVIDIA_DOCS.md`'s rule is
> that anything a kernel depends on is re-derived in first-party words and pinned
> in code or a regression test. The CUDA facts this branch needs are already
> pinned that way in `hardware_target.cpp`'s table comments.

**Files:**
- Modify: `AGENTS.md` (Ampere policy — **operator-reviewed text, applied by them**)
- Modify: `docs/NVIDIA_DOCS_AMPERE.md:22-27` ("never add 86-real/87-real" → supported)
- Create: `docs/model-reports/2026-11-24-ampere-bf16-slice.md`
- Test: existing per-model targets against `LSX_ARTIFACT_ROOT`

- [ ] **Step 1: Configure and build on the 3090 host**

Run: `cmake --preset linux-release-sm86 && cmake --build build/linux-release-sm86 -j`
Expected: complete build, no `compute_120a` error, `test_cuda_toolkit_guard` green.

- [ ] **Step 2: Convert the slice** (already done in Task 9a on the 5090; this run is
  on the 3090 host, and the artifact tree is target-independent, so it may be copied
  rather than rebuilt)

Run: `LSX_ARTIFACT_ROOT=<store> lsxforge --phi --input <SRC>/phi-4 --codec-profile bf16`
and the same for `--qwen --input <SRC>/Qwen3.5-9B --codec-profile bf16`.
Expected: `artifact.fb` present at the profile dir root, `--verify` green, mask `bf16,f32`.

- [ ] **Step 3: Load and gate on the 3090**

Run: `lsxbenchmark --localmodel <store>/phi/phi-4/bf16 --selfcheck`
Expected: gate passes (`bf16,f32 ⊆ nvidia:sm_86`), SelfCheck "Paris" ×3 bit-identical,
throughput recorded (GEMV-only prefill is expected to be slow; record the number, do
not tune it in this branch).

- [ ] **Step 4: Prove the refusal**

Run: `lsxbenchmark --localmodel <store>/phi/phi-4/nvfp4 --selfcheck`
Expected: non-zero exit, message naming `nvfp4`, `nvidia:sm_86`, and the
`--codec-profile bf16` remedy.

- [ ] **Step 5: Build on Jetson (sm_87, CUDA 12.6)**

Run: `cmake --preset jetson-orin-release && cmake --build build/jetson-orin-release -j4`
Expected: build green; `test_cuda_toolkit_guard` green with the fp4 cases SKIPped and
the reason printed.

- [ ] **Step 6: Commit** — `docs: record the Ampere bf16 slice acceptance on sm_86 and sm_87`

**Branch 2 exit criteria:** both toolchains build; `bf16` artifacts verified; 3090
SelfCheck green; the NVFP4 refusal demonstrated on 3090; Jetson build + guard green;
the model report committed with measured throughput.

---

# Branch 3 — `feat/int8-w8a16-sm86-sm87`

> **NOT STARTED (2026-11-28).** No file this branch names exists:
> `int8_quant.{h,cu}`, `lsx_gemv_int8.cu`, `lsx_gemm_w8a16.cu`,
> `tests/test_int8_quant.cu`, `tests/test_lsx_gemv_int8.cu`,
> `tests/test_gemm_w8a16.cu`. The branch also inherits two prerequisites that did
> not exist when it was written: `CodecProfile` must be visible to `libs/`
> (Task 15, refinement 1) and expert planes must declare their codec (Task 9c),
> or an INT8 MoE tree verifies as NVFP4 and is refused on Ampere.

Branches from Branch 2. Scope: `int8-w8a16` profile + deterministic W8A16 decode GEMV
and prefill GEMM, validated on a live 3090 and Orin NX 16 GB.

**Numeric note that shapes every task:** W8A16 has no mixed s8×f16 tensor-core
instruction on GA10x. It is a **storage** win: weights are `int8`, dequantized to
f16/bf16 in the epilogue, accumulated in f32. Activations stay f16/bf16 — **no
activation quantization**, so the numerics are bounded by per-channel weight
rounding and the path stays deterministic and device-property-independent (R3 within
the `{sm_86, sm_87}` set).

### Task 17: Per-channel symmetric INT8 weight quantizer

> **NOT STARTED (2026-11-28).** The scope note below still holds exactly:
> `converter.cpp:660` (`(dtype == "I8" || dtype == "U8") && !src_scale.empty()`)
> carries int8 planes through, `converter.cpp:309` accepts `I8` as a source dtype,
> and nothing quantizes. **Anchor:** the quantizer's surface is the same as
> `gpu::quantize_tensor_gpu` (used at `phi_dense_convert.cpp:391`, `:428`), so
> `RunInt8QuantizePerChannel` should mirror its signature and its launch geometry
> conventions rather than invent a second style; the reference it must beat for
> determinism is `RunLxGemmVec`'s fixed warp-per-row geometry (`lsx_gemv.h:26`).

**Scope note (2026-11-27 re-measurement).** `kI8Sym` **storage already exists**: 768
int8 tensors ship in `DeepSeek-V4-Flash`, written by the generic passthrough at
`converter.cpp:660` (`(dtype == "I8" || dtype == "U8") && !src_scale.empty()`), and
`converter.cpp:309` already accepts `I8` as a source dtype. What does not exist is a
BF16 → I8 **quantizer** and the **inference kernel**. This task adds the quantizer
only; it does not invent a storage format.

**Files:**
- Create: `libs/lsxcommon/src/int8_quant.h`, `src/int8_quant.cu`
- Create: `libs/lsxcommon/include/lsxcommon/int8_quant.h`
- Test: `tests/test_int8_quant.cu` (label `inference`, `RESOURCE_LOCK "gpu"`,
  `SKIP_RETURN_CODE 4` when no device)

**Interfaces:**
- Produces: `struct Int8Plane {const int8_t *q; const float *scale; int rows, cols;
  int64_t q_bytes, scale_bytes;}`;
  `cudaError_t RunInt8QuantizePerChannel(cudaStream_t, const void *src, int rows,
  int cols, LxDtype src_dtype, Int8Plane out);`
  `float Int8MaxAbsErrorBound(float max_abs_per_channel)`.

- [ ] **Step 1: Failing test:**

```cpp
TEST_CASE("per-channel symmetric quantize round-trips within half a step") {
    // 4x64 fp16 weights with per-row max |w| = 1, 0.5, 0.25, 2
    // assert scale[r] == max_abs[r] / 127
    // assert max |dequant - w| <= max_abs[r] / 254 for every element
}
TEST_CASE("a zero row quantizes to scale 0 and all-zero q, with no NaN") { … }
TEST_CASE("determinism: two runs on identical input are bit-identical") { … }
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** the kernel: one block per row, warp max reduction (fixed
  geometry, no atomics), `scale = max_abs / 127`, `q = round(w / scale)` clamped to
  [-127, 127] (symmetric, so -128 unused), zero row → scale 0.

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit** — `feat: add deterministic per-channel symmetric int8 weight quantizer`

### Task 18: `int8-w8a16` codec profile — converter side

> **NOT STARTED (2026-11-28).** Depends on Task 17 (quantizer), Task 15's enum move
> (`CodecProfile` must be visible to `libs/`) and Task 9c if the family is MoE. The
> `PinnedSchema` part of the note below is accurate — `dtype = "I8"` already passes
> the verifier's `CodecForDtype()` and clears D1 — so this task is a converter
> change plus a mask assertion, not a format change.

**Files:**
- Modify: `libs/lsxcommon/src/phi_dense_convert.cpp`, `qwen_dense_convert.cpp`
- Modify: `apps/lsxforge/src/codec_profiles.h` (`kInt8W8A16` for `kDense`)
- Test: extend `tests/test_phi_dense_convert.cpp`, `tests/test_qwen_dense_convert.cpp`

**Interfaces:**
- Consumes: Task 17. Uses the **existing** `PinnedSchema` (`name,shape,dtype,data`) —
  `dtype = "I8"` for the plane plus a sibling `<name>.scale` tensor with
  `dtype = "F32"`. **No Arrow schema change.**

- [ ] **Step 1: Failing test:**

```cpp
TEST_CASE("int8 profile writes I8 planes plus F32 scale companions and declares i8") {
    const auto a = convert_fixture(CodecProfile::kInt8W8A16);
    REQUIRE(a.ok);
    CHECK(a.artifact_mask == (Mask(PlaneCodec::kI8Sym)|Mask(PlaneCodec::kBF16)
                             |Mask(PlaneCodec::kF32)));
    CHECK(tensor_dtype(a.dir, "layer_0.arrow", "mlp.up.weight") == "I8");
    CHECK(tensor_dtype(a.dir, "layer_0.arrow", "mlp.up.weight.scale") == "F32");
    CHECK(VerifyArtifact(a.dir, true).ok);
}
TEST_CASE("int8 profile byte size is ~half the bf16 profile for the same model") {
    // assert int8_tree_bytes < 0.55 * bf16_tree_bytes
}
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** the quantize-and-write path; keep norms/embeddings in
  BF16/F32 (only 2-D linear weights are quantized).

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit** — `feat: add the int8-w8a16 codec profile to the phi and qwen dense converters`

### Task 19: W8A16 decode GEMV

> **NOT STARTED (2026-11-28).** `lsx_gemv.h` currently exposes exactly two entry
> points — `RunLxGemmVec` (`:26`) and `RunLxGemmVecWT` (`:31`) — so the new
> `RunLxGemmVecInt8` is the third and must take the same `(stream, w, x, …)` shape
> with the per-row scale inserted after `w`; a different argument order is how a
> family ends up with two GEMV conventions. One kernel serves sm_86 and sm_87: the
> geometry must fit the **99 KB** sm_86 shared-memory tier, since the 163 KB sm_87
> tier is not a licence to size for Orin.

**Files:**
- Create: `libs/lsxcommon/src/lsx_gemv_int8.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_gemv.h` (add the entry point)
- Modify: `libs/lsxcommon/CMakeLists.txt`
- Test: `tests/test_lsx_gemv_int8.cu` (label `inference`, `RESOURCE_LOCK "gpu"`)

**Interfaces:**
- Produces: `cudaError_t RunLxGemmVecInt8(cudaStream_t, const int8_t *w,
  const float *w_scale, const void *x, int seq, int N, int K, LxDtype x_dtype,
  float alpha, void *y, LxDtype y_dtype, float beta);` — same contract and the same
  fixed warp-per-row geometry as `RunLxGemmVec`.

- [ ] **Step 1: Failing test:**

```cpp
TEST_CASE("w8a16 gemv matches a dequantized-f32 reference within 1e-4 relative") {
    // N=64, K=256, random int8 weights + per-row scale, bf16 activations
    // reference: f32 accumulation over dequantized weights
}
TEST_CASE("w8a16 gemv is bit-identical across 3 runs (R3 within the target set)") { … }
TEST_CASE("w8a16 gemv is bit-identical between two launch shapes of the same problem") { … }
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** as a variant of the existing fixed-geometry GEMV: load
  `int8` bytes, convert to f32 via the scale, multiply f32, warp-shuffle reduce in a
  fixed order. Deliberately **no** algorithm choice based on device properties.

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit** — `feat: add deterministic W8A16 decode GEMV`

### Task 20: W8A16 prefill GEMM (the one new MMA kernel)

> **NOT STARTED (2026-11-28). One numeric fact to state before writing the kernel:**
> W8A16 dequantizes the weight tile to `__half`/`__nv_bfloat16` in shared memory and
> then uses **`mma.sync.aligned.m16n8k16` (f16/bf16, f32 accumulate)** — the int8
> `mma.sync` family is for W8A8 and is *not* used here, because activations stay
> 16-bit (that is what keeps the path deterministic and the error bounded by
> per-channel weight rounding, per the numeric note above). Consequences for the
> tile: the smem budget is a 16-bit tile, not an int8 one; `cp.async` needs only
> 4-byte alignment (16-byte `.cg` for the aligned bulk of the tile); and the tile
> must fit the **99 KB sm_86** tier so one kernel serves both 3090 and Orin.
> Everything else in this task's step 3 — fixed tile schedule, no TMA, no
> mbarrier, no tcgen05 — stands.

**Files:**
- Create: `libs/lsxcommon/src/lsx_gemm_w8a16.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_tc_gemm.h` (Ampere section)
- Test: `tests/test_gemm_w8a16.cu` (label `inference`, `RESOURCE_LOCK "gpu"`)

**Interfaces:**
- Produces: `cudaError_t RunW8A16Gemm(cudaStream_t, const int8_t *w,
  const float *w_scale, const void *x, int seq, int K, int N, LxDtype x_dtype,
  void *y, bool accumulate);`

- [ ] **Step 1: Failing test:**

```cpp
TEST_CASE("w8a16 gemm agrees with the w8a16 gemv path at seq==1") { … }
TEST_CASE("w8a16 gemm agrees with the w8a16 gemv path per row at seq==8") {
    // tile path vs 8 sequential gemv calls, tolerance 1e-4 relative
}
TEST_CASE("w8a16 gemm is deterministic across runs and batch shapes") { … }
TEST_CASE("w8a16 gemm returns NotSupported on a non-Ampere device it was not built for") { … }
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement**: `cp.async`-staged tiles (available since sm_80, and
  already used in `lsx_tc_gemm.cu:86-135`), dequantize the `int8` tile to `__half` /
  `__nv_bfloat16` in shared memory, `mma.sync.aligned.m16n8k16` (f16/bf16), f32
  accumulate, fixed tile schedule. **No TMA, no mbarrier, no tcgen05** — none exist on
  GA10x. Smem budget must fit the 99 KB sm_86 tier (not the 163 KB sm_87 tier) so one
  kernel serves both.

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit** — `feat: add the W8A16 prefill GEMM for sm_86/sm_87`

### Task 21: Per-profile goldens and the numeric declaration

> **NOT STARTED (2026-11-28), and the file list is incomplete.** Golden resolution
> happens in **two** places, not one, and they must agree or `lsxbenchmark
> --fsi-selfcheck` and `ctest -R test_*_fsi_selfcheck` will read different files:
>
> | site | code |
> |---|---|
> | CLI | `libs/lsxcommon/src/model_runtime.cpp:297` (fsi), `:331` (code), `:366` (mcp), `:401` (det) — each builds `<dir>/<Type()>.json` |
> | test driver | `tests/test_parity_gate.cpp:60-76` (`GoldenDir()`/`GoldenPath()`), one implementation for all four gates |
>
> There is no `ResolveGolden()` function to modify: the plan's Task 21 names
> `fsi_selfcheck.cpp` / `code_selfcheck.cpp` / `det_selfcheck.cpp` /
> `selfcheck_common.cpp`, and those hold the *comparison*, not the path. The rule
> belongs next to each path construction — one small helper in `lsxcommon`
> (`GoldenPathFor(model_type, profile, dir)`) called by all five sites, with the
> test driver calling the same helper so the two cannot diverge.
>
> **Profile discovery order** (the helper's contract): the loaded artifact's
> `profile`, else the manifest's `vault_profile` when the model is not an artifact
> yet. Then `<model_type>@<profile>.json` if it exists, else `<model_type>.json`
> (the historical `nvfp4` spelling, which is why no existing golden moves).

**Files:**
- Modify: `libs/lsxcommon/src/fsi_selfcheck.cpp`, `code_selfcheck.cpp`,
  `det_selfcheck.cpp`, `selfcheck_common.cpp` (resolve `<type>@<profile>.json`,
  fall back to the bare name)
- Modify: `docs/numeric-contract.md` (the §4 amendment, if not already in Branch 1)
- Modify: `docs/SELFCHECK_MATRIX.md` (generated — regenerate via the manifest tool)
- Test: `tests/test_golden_profile_resolution.cpp` (label `fast`)

- [ ] **Step 1: Failing test:**

```cpp
TEST_CASE("a profile-suffixed golden is preferred and the bare name still resolves") {
    // golden dir has phi-4.json and phi-4@bf16.json
    CHECK(ResolveGolden("phi-4", "bf16") == "phi-4@bf16.json");
    CHECK(ResolveGolden("phi-4", "nvfp4") == "phi-4.json");   // historical profile
}
TEST_CASE("a missing profile golden is reported as debt, not as a pass") { … }
```

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** the resolution rule and the debt reporting (the existing
  declared-debt mechanism from `OPEN_TASKS` item `K` already enforces both
  directions — reuse it, do not invent a second one).

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit** — `feat: resolve parity goldens per codec profile`

### Task 22: Live acceptance on RTX 3090 and Orin NX 16 GB

> **NOT STARTED (2026-11-28).** It also depends on numbers that do not exist yet:
> the artifact store holds three `nvfp4` artifacts and no second profile, so the
> "third profile of the same two models — the multi-artifact case in full" starts
> from one profile, not two. Task 9f schedules the conversions in the order that
> makes this task's steps executable.

**Files:**
- Create: `docs/model-reports/2026-11-24-ampere-int8-w8a16.md`
- Modify: `tests/perf/perf_baselines.json` (seeded rows for the Ampere targets)
- Test: existing per-model gate targets against `LSX_ARTIFACT_ROOT`

- [ ] **Step 1: Convert and verify on the 3090 host**

The volume has 2.6 TB free (spec §1.9); phi-4 `int8-w8a16` is ≈15 GB and
Qwen3.5-9B ≈9 GB. `lsxforge` still preflights the estimate and refuses a conversion
it cannot finish — a guard against a half-written tree, not a capacity problem.

Run: `LSX_ARTIFACT_ROOT=<store> lsxforge --phi --input <SRC>/phi-4 --codec-profile int8-w8a16`
then `--qwen --input <SRC>/Qwen3.5-9B --codec-profile int8-w8a16`; `--verify` each.
Expected masks: `i8,bf16,f32`; **no `mtp/` in any tree** (v1 has no ballast); the
`bf16` and `nvfp4` siblings already exist from Task 9a, so this is the third profile
of the same two models — the multi-artifact case in full.

- [ ] **Step 2: Gate + parity on the 3090**

Run: `lsxbenchmark --localmodel <store>/phi/phi-4/int8-w8a16 --selfcheck`,
`--fsi-selfcheck`, `--code-selfcheck`, `--det-selfcheck`.
Expected: mask includes `i8`, gate passes on `nvidia:sm_86`; SelfCheck "Paris" ×3
bit-identical; FSI/CODE/DET verdicts recorded (a red is a finding for the report,
not a reason to change the gate).

- [ ] **Step 3: Perf row on the 3090** — canonical 2048-token protocol, record t/s and
  the `int8` vs `bf16` delta (this is the measurement that decides the profile rank
  in spec §8.2).

- [ ] **Step 4: Orin NX 16 GB**

Run the same conversion (copy the artifact tree; it is target-independent by
`used_codecs`) and the same four gates on `nvidia:sm_87`, with the unified-memory
governor active. Record: load success, resident bytes, expert-cache slots chosen,
t/s, and whether the 4 GiB reserve (16 GiB / 4) held.

- [ ] **Step 5: Orin Nano 8 GB (best effort)** — load `dory-1.0-4b` (smallest slice)
  in `int8-w8a16`; record pass/fail. A failure here is reported, not fixed, in this
  branch.

- [ ] **Step 6: Commit** — `docs: record the int8-w8a16 acceptance on sm_86 and sm_87`

**Branch 3 exit criteria:** `int8-w8a16` artifacts verify; 3090 SelfCheck green and
FSI/CODE/DET verdicts recorded; Orin NX load + gates recorded with measured memory;
perf rows seeded for both Ampere targets; profile rank corrected from measurement if
it disagrees with the assumption.

---

## Lifecycle audit (2026-11-28)

This is the evidence behind the revision note and the task statuses above. It asks
one question per AGENTS.md's lifecycle gate — *did every stage actually run, and is
a SKIP distinguishable from a pass?*

**Method** (all reproducible on this host):

```bash
export LSX_SOURCE_ROOT=/mnt/fast_storage/lsx_model
export LSX_ARTIFACT_ROOT=/mnt/fast_storage/lsx_artifact
ctest --preset linux-debug                                   # 184/184 fast, 166 s
lsxforge --verify --input <root>/microsoft/phi-4/nvfp4       # declared=34 actual=34
ctest -R "test_phi_dense|test_phi_tool_framing|test_perf_baseline_phi_4|\
test_bootstrap_artifacts|test_load_codec_gate" -j1           # 12 pass, 1 skip (perf)
ctest -R "test_dory_dense_selfcheck|test_dory_dense_fsi_selfcheck" -j1 -V
# registry vs store, per row: count rows, profiles, status, and which artifacts
# exist — the script below walks models/*/*.json and $LSX_ARTIFACT_ROOT
python3 - <<'PY'
import glob, json, os
root = os.environ['LSX_ARTIFACT_ROOT']
for f in sorted(glob.glob('models/*/*.json')):
    d = json.load(open(f))
    for name, p in d.get('profiles', {}).items():
        store = p.get('artifact', {}).get('store')
        on_disk = bool(store) and os.path.isfile(os.path.join(root, store, 'artifact.fb'))
        print(d['model_type'], name, p.get('status'), 'on_disk' if on_disk else '-')
PY
# -> 48 rows, 1 profile each, 3 built == 3 on disk, 45 not built
```

### A. The registry and the store agree on what exists, and disagree with reality

| check | result |
|---|---|
| rows declaring all 10 stages + all 5 gate targets | **48 / 48** — the *structure* of the lifecycle gate is complete for every model |
| rows with `status: built` | **3** (`phi-4`, `qwen3.5-9b`, `gemma-4-12b-it`), all `nvfp4` |
| artifacts on disk | **3**, each at the exact `<org>/<model_dir>/<profile>` path its row names, all deep-verify green |
| registered rows with no artifact | **45** — every real-weight stage for them SKIPs |
| checkpoints on disk with no registry row | **3** — `baidu/ERNIE-5.5-Flash-Base-BF16` (534 GB, open task `W`) **and two the registry has never seen**: `JetBrains/Mellum2-12B-A2.5B-Thinking`, `JetBrains/Mellum2.1-12B-A2.5B-Instruct` (23 GB each, both with `config.json` + weights) |
| registry rows with a profile count > 1 | **0** — every row declares exactly one profile; `preferred_profile` is `nvfp4` for all 48 |

The spec's inventory (§1.1: "39 checkpoints over 18 orgs") is now **51 over 24**.
The two Mellum downloads are new since it was written; task `W`'s fix shape ("every
checkpoint under the source root belongs to a row or is declared as an uningested
download") is what would have caught them, and it is now a three-model gap, not one.

### B. The three artifact-backed rows, stage by stage

| stage | `phi-4` (`nvfp4`) | `qwen3.5-9b` (`nvfp4`) | `gemma-4-12b-it` (`nvfp4`) |
|---|---|---|---|
| convert | pass | pass | pass |
| `--verify` | 34/34, 0 problems | 35/35, 0 problems | 34/34, 0 problems |
| load / model / kernels | pass | pass | pass |
| SelfCheck "Paris" ×3 | pass, `cf51b9bbd5c913dd` | pass, `a2af669548e59c73` | pass, `31d7a1672d89ccdc` |
| FSI | **pass 3/3** | **pass 3/3** | **FAIL 2/3** (FSI-02 red) — documented `R22`: the *source* is a 0.556-logit near-tie at the fork |
| CODE | pass 3/3 | pass 3/3 | pass 3/3 |
| DET | pass 3/3 | pass 3/3 | pass 3/3 |
| MCP | pass 3/3 | pass 3/3 | pass 3/3 (byte-exact) |
| perf row (2048 tokens) | **71.83 t/s, dated 2026-10-24** | **160.92 t/s, dated 2026-10-23** | 59.85 t/s, 2026-11-28 |
| stage-9 report | **missing** (`"report": ""`) | **missing** (`"report": ""`) | present |
| tool round-trip hook | not declared | not declared | declared |

Two real defects in this table, neither of which any test catches:

1. **The perf rows of `phi-4` and `qwen3.5-9b` predate their artifacts.** Both were
   re-converted on 2026-11-28; the numbers are from 2026-10-24 / 2026-10-23. The
   `nvfp4` codec layout is unchanged, so the numbers are probably still true — but
   "probably" is what a registry exists to avoid. Owner: **Task 9f**, step 1.
2. **Two of the three bootstrap artifacts have no stage-9 report.** The lifecycle
   gate ends in a publishable report (`docs/model-reports/YYYY-MM-DD-<name>.md`);
   `phi-4` and `qwen3.5-9b` have an empty `report` field while their gates are
   recorded. Owner: Task 9f when those rows are touched, or a `docs:` commit of its
   own.

### C. A SKIP is distinguishable from a pass — verified

`test_dory_dense_selfcheck` and `test_dory_dense_fsi_selfcheck` (a model with no
artifact) skip with

```
dory-1.0-4b converted model not present in nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/nvfp4
```

— it names the model, the org, the directory and the profile it looked in, and the
skip is `SKIP_RETURN_CODE 4` with a declared stage target, so it cannot be mistaken
for green. That is the behaviour the cold start was designed to produce, and it is
what makes the 45 missing artifacts an honest gap rather than a silent one.

The four declared-debt counters agree with the matrix: `fsi` 11, `code` 11, `det`
11, `mcp` 9 rows without a golden — a *declared* absence, enforced in both
directions by `test_model_suite_guard`.

**One rule the fast suite cannot see.** Spec §1.4 forbids ballast, and
`ContentDirs()` (`lsxstorage/src/verify.cpp:49-54`) is where that becomes
enforceable: `{pinned, experts, router, shared, ple}`. `grep -rn 'mtp_dir'
libs/lsxcommon/src` returns **8 files / 33 hits**, and each of them writes planes
into `base / "mtp"`. So eight families' `--codec-profile` conversions are refused
by the scan *after* the whole tree is written — a failure no `fast` test can reach,
because no `fast` test converts a MoE family. That is finding F10 and Task 8b.

### D. Findings and their owners

| # | finding | owner |
|---|---|---|
| F1 | 45 of 48 rows have no artifact, so their lifecycle gates SKIP | **Task 9f** |
| F2 | No converter can produce a second profile; the `bf16` profile that unblocks Ampere does not exist | **Task 15** (+ 8a, 9c) |
| F3 | `VerifyArtifact` assumes an expert plane is NVFP4 (no dtype column), so `bf16`/`int8` MoE trees cannot be verified or gated | **Task 9c** |
| F4 | `model.index.json` still holds a codec map in 5 writers, 2 readers and the arrow verifier; `--verify` does not enforce rule 5 | **Task 9d** |
| F5 | Profile selection is directory-order, not target-aware; `--explain-artifact` does not exist | **Task 9e** |
| F6 | A flagless `lsxforge` run exits 0 and publishes no `artifact.fb` — a tree nothing can load | **Task 8a** |
| F7 | Three checkpoints on disk have no registry row (ERNIE + two Mellum) | Task 9f + open task `W` |
| F8 | `phi-4` and `qwen3.5-9b` perf rows predate their artifacts; both lack a stage-9 report | **Task 9f** |
| F9 | Branch 2/3 have no code at all: no `86/87` build, no `bf16`/`int8` profile, no Ampere presets, and the doc the plan amends was deleted from the tree | Tasks 11–22, 16 |
| F10 | eight converters still write `mtp/`, which `ScanArtifactTree` refuses — so those families cannot be converted with a profile at all, after a full run | **Task 8b** |

## Self-review of this plan

- **Spec coverage:** I1 → Tasks 2, 9 (mask excludes, gate refuses). I2 → Tasks 1–3.
  I3 → Task 9 (+ Task 7 for mask integrity). I4 → Tasks 8, 9 (`Resolve`,
  `ListCompatibleProfiles`), 15/18 (per-profile conversion), and the `lsxfabric`
  autoselect is specified in spec §2.7 and scheduled as part of Task 9's
  `ListCompatibleProfiles` + the fabric CLI work noted there. I5 → Tasks 4–8, 10.
  Dependency registry → Tasks 11–14 + 13. Unified memory → Task 10, 22.
- **Gap found and closed:** the spec's autoselect (§2.7) had no owning task when
  first drafted. It is now covered by `ListCompatibleProfiles` (Task 9) plus the
  `--explain-artifact` flag; the fabric wiring itself is deliberately *not* a
  separate task because `apps/lsxfabric/src/main.cpp` already routes through
  `LsxModelFactory::Create` → `BeginLoad`, so ranking is the only new code and it
  lands in Task 9.
- **Type consistency checked:** `CodecMask`, `Mask()`, `PlaneCodec`, `CodecProfile`,
  `ArtifactInfo`, `VerifyReport`, `Resolved`, `ReadStatus`, `MemoryModel` are spelled
  identically in every task that uses them.
- **Proportion:** no kernel bodies are transcribed; the only code blocks are test
  assertions carrying the spec's exact values, and signatures.

### Re-review after the cold start (2026-11-27b)

- **What the empty store simplified, verified against the plan text:** `dropped_planes`
  is gone from Tasks 5/6/7/15; `DropsBallast` is gone from Task 8 (replaced by a test
  that *no* profile declares a ballast plane); `kLegacy` is gone from Tasks 4/9;
  the per-tree audit is gone from Task 9 (replaced by a guard on the empty-store
  diagnostic); `PreflightDisk` survives in Tasks 6/8 but as a guard, and the 2×
  staging rule is deleted.
- **What it added, and the plan now says it in the exit criteria:** **Task 9a**, the
  bootstrap conversion. Without it, Branch 1 can be "complete" while every real-weight
  gate SKIPs — the exact failure AGENTS.md calls out. `Task 9a` is therefore a
  Branch-1 exit criterion, not a convenience.
- **Coverage of the new spec sections:** §1.3 rules D1–D5 → Task 7's six cases (the
  `U8`-rejection case is the one that pins the retirement of the old spelling);
  §1.4 "no ballast in any profile" → Tasks 7, 8, 15; §1.1.1 deletion verified
  complete → Task 9 step 5 (the "looks like a tree, is not an artifact" case);
  §1.9 demoted constraint → Global Constraints + Tasks 6, 8.
- **One risk is now structural rather than residual:** nothing can validate the
  contract against real data any more, because the data is deleted — and with the
  last leftover tree gone (verified: 0 `.arrow` in the vault), not even a specimen
  survives. Task 9a handles it by treating the first two conversions as the contract
  test; `EmitArtifact`'s self-re-verify (Task 6) and the declared-vs-written mask
  check (Task 8) make a bad emitter fail loudly instead of producing a plausible
  artifact. The nine-row inventory stays in the spec as the only provenance record.
- **Test-count sanity:** 25 tasks (22 + 4a + 9a + 9b). Of them, 13 add `fast` tests
  that run with no GPU (Tasks 1–9, 4a, 9b, 11, 14, 21) — the half of the contract
  provable on a machine with no Ampere hardware attached. Task 4a is in that set: the
  path-shape guard is the only thing that can catch a missed rewrite without a
  converted artifact to load.
- **Defect found by answering "is `artifact.fb` per model?" (spec §9.3):**
  `model.index.json` still declared codecs and six files read them, giving the design
  a second source of truth and a live I1 hole. Closed by **Task 9b** (per-layer map
  moves into `artifact.fb`, readers go through `ModelBase::LayerUsesCodec`), at a
  stated cost of 6 files in 2 families — and the plan now records that the split
  between the two files is settled **per field**, not per file.

### Re-review after the status audit (2026-11-28)

Run against the tree, not against the text — every claim below was checked with a
command or a test run, and the commands are in § *Lifecycle audit*.

- **Spec coverage, re-stated honestly.** I1 → Tasks 2 + 9 (done) **and 9c** (the
  verifier currently *contradicts* I1 on expert planes by assuming NVFP4). I2 →
  Tasks 1–3 (done). I3 → Task 9 (done). I4 → **storage half done** (Tasks 4–9),
  selection half open (**9e**), production half open (**15**, **9f**). I5 → Tasks
  4–8, 10 (done except Task 10), with §1.4's no-ballast rule still violated by
  eight converters (**8b**). The §4 numeric-contract amendment is still
  unapplied and is now owned explicitly by **Task 16**. So the design is not
  "implemented"; it is "implemented for one codec profile on one target family,
  with named owners for the rest".
- **Two planned interfaces were unimplementable as written and are corrected
  above**, which is the strongest argument for recording the audit in the plan
  rather than in a chat log: (a) Task 15 needs `CodecProfile` **inside `libs/`**,
  but the enum lives in `apps/lsxforge`, so the task as written would have produced
  a per-family copy of the profile table — the exact thing spec §2.6 forbids;
  (b) Task 16 amends `docs/NVIDIA_DOCS_AMPERE.md`, which no longer exists and must
  not return.
- **Type consistency re-checked across the new tasks.** `CodecProfile` moves to
  `lsxcommon/codec_profile.h` and `lsxforge` re-exports it, so Task 8's
  `ParseCodecProfile`/`DeclaredPlanes` keep their signatures. `SelectProfile` takes
  the capability as a **`uint64_t` mask plus an ordered name list**, not as
  `hw::CodecMask`/`hw::PlaneCodec`: `lsxstorage` must stay below `lsxcommon`
  (Task 4's constraint), and the mask crossing the boundary as an integer is
  already how `artifact.fb` stores it. `PlaneCodec`'s numeric values (1<<0..1<<6)
  and `artifact.fbs`'s `bit_flags` **indices** (0..6) are the same seven bits by
  two different spellings — the `.fbs` comment says so, and
  `test_manifest_sync`-style drift here would reinterpret every artifact, so
  Task 5's case is the guard.
- **Review focus, updated.** The original seven failure modes stand, and the audit
  found that two of them are live rather than hypothetical: #1 (a converter keeping
  an old spelling — the *verifier* now keeps one instead, for expert planes: Task
  9c) and #2 (a codec declaration outside `artifact.fb` — five writers and two
  readers still do it: Task 9d). Added: #9 "the tool reports success and publishes
  nothing" (Task 8a) and #10 "the registry says `built` and the store disagrees"
  (Task 9f).
- **Proportion.** The audit added ~12 KB of status, evidence and six tasks; it
  removed no design. The plan is now 31 tasks — 25 original, of which 9 are done,
  2 done-with-residue, 1 partial and 13 not started — and it says which is which at
  the top, so a reader no longer has to diff the document against the tree.
