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
`<ROOT>/<family>/<model_type>/<profile>/{artifact.fb,pinned,experts,…}` (spec §2.4.2).
**New Task 4a** performs the mechanical rewrite (538 sites: 144 `libs/`, 3 `apps/`,
391 `tests/`), deletes the four dead `ResolvePinnedDir()` fallbacks, rewrites the two
"detect a tree by folder name" sites to test `artifact.fb`, and adds a `fast` guard
that the segment cannot come back.

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
  segment anywhere: `<LSX_ARTIFACT_ROOT>/<family>/<model_type>/<profile>/`. An artifact is
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

1. **A converter keeps an old spelling** — nine converters today write `mtp/`, and
   several spell NVFP4 as a `U8` `.packed`/`.scales` pair (gemma, Muse, kimi, mimo,
   inkling). Carried into the new format, the first case ships planes the loader
   never reads (inflating the mask) and the second makes the codec unreadable
   without context. → Task 7 (rules D1–D5, `--verify` fails both) and the converter
   edits in Task 8.
2. **A codec declaration survives in `model.index.json`** — five fields
   (`expert_nvfp4[]`, `expert_fp8[]`, `mlp_nvfp4`, `attn_nvfp4`, `head_nvfp4`) are
   read by six files to pick a kernel per layer. Left there, they can disagree with
   `artifact.fb` and the gate would pass a model that executes an undeclared codec.
   → **Task 9b**.
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

---

# Branch 1 — `feat/lsxstorage-hardware-codec-contract`

Branch from `main`. Merges alone; produces working, testable software without
Branch 2 (all existing 5090 behaviour preserved after reconversion).

### Task 1: `PlaneCodec` enum, names, parsing

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

- [ ] **Step 1: Write the failing test** — `tests/test_hardware_target.cpp`:

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

- [ ] **Step 2: Run it and confirm failure**

Run: `cmake --build build/linux-debug --target test_hardware_target -j` → FAIL to
compile (`hardware_target.h` not found).
Register the target first with `lsx_add_test(NAME test_hardware_target LABEL fast)`
so the failure is a link/compile failure of a known target, not a missing target.

- [ ] **Step 3: Implement** `hardware_target.h` (POD only, `#include <cstdint>`,
  `<string>`, `<string_view>`; no absl/arrow/flatbuffers) and `hardware_target.cpp`.
  `PlaneCodecName` returns the lowercase names used by `--codec-profile` and
  `artifact.fb`; `ParsePlaneCodec` accepts exactly those plus `"i8"` for `kI8Sym`.
  Comment at the top: *storage encoding, not execution path* (spec §2.2).

- [ ] **Step 4: Run and confirm pass** — `ctest --test-dir build/linux-debug -R test_hardware_target`

- [ ] **Step 5: Commit** — `feat: add PlaneCodec storage-encoding enum with names and parsing`

### Task 2: `HardwareTarget` + the `SupportedCodecs` table

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

- [ ] **Step 1: Write the failing test** — the table, verbatim from spec §2.3:

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

- [ ] **Step 2: Run and confirm failure** — unimplemented symbols.

- [ ] **Step 3: Implement** the table as one `switch` on `(vendor, arch)` with the
  hardware reason as a comment per row (spec §2.3). `TargetFromComputeCapability`
  returns `kUnknown` for anything outside `{12.0, 12.1, 8.6, 8.7}` — including
  `8.0`, per AGENTS.md. `TargetFromCpuFeatures` builds the arch string from feature
  flags in a **fixed order**: `avx512`, `+bf16`, `+fp16`, `+vnni`; for neon,
  `neon`, `+dotprod`.

- [ ] **Step 4: Run and confirm pass.**

- [ ] **Step 5: Commit** — `feat: add HardwareTarget and the SupportedCodecs capability table`

### Task 3: `CheckCodecs` — I3 as a pure function — and `ActiveTarget`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/hardware_target.h`, `src/hardware_target.cpp`
- Test: `tests/test_hardware_target.cpp`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `struct GateDecision {bool allowed; CodecMask missing; std::string
  message;}`; `GateDecision CheckCodecs(CodecMask used, const HardwareTarget &)`;
  `const HardwareTarget &ActiveTarget()`; `void SetActiveTargetForTesting(const
  HardwareTarget &)`; `void ResetActiveTargetForTesting()`.

- [ ] **Step 1: Write the failing test:**

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

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** `CheckCodecs` is three lines: unknown vendor → refuse;
  `SupportedCodecs(t) == 0` → refuse "target not implemented"; otherwise
  `missing = used & ~supported`. `ActiveTarget()` resolves once under
  `std::once_flag`: `cudaGetDeviceProperties(&p, 0)` when a device exists (guard the
  CUDA call so host-only builds link), else CPUID-derived `cpu:*` features
  (`__builtin_cpu_supports("avx512f")`, `"avx512bf16"`, `"avx512fp16"`,
  `"avx512vnni"`; NEON via `__aarch64__` + `__ARM_FEATURE_DOTPROD`).

- [ ] **Step 4: Run and confirm pass** on the 5090 host — `ActiveTarget().Id()` must
  be `nvidia:sm_120` there; add a `SKIP`-guarded case asserting that.

- [ ] **Step 5: Commit** — `feat: add the codec admission gate as a pure function plus active-target resolution`

### Task 4: `libs/lsxstorage` skeleton and root resolution

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

- [ ] **Step 1: Write the failing test** — use `setenv`/`unsetenv` in a RAII helper:

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

- [ ] **Step 2: Run and confirm failure** (target absent).

- [ ] **Step 3: Implement** by *moving* `path_registry.cpp`'s body into
  `lsx::storage` and leaving `lsxcommon::lsx_model_models_dir()` /
  `lsx_cache_dir()` / `lsx_cache_path()` as one-line forwarders. Update
  `libs/CMakeLists.txt` ordering. `libs/lsxcommon/CMakeLists.txt` gains
  `target_link_libraries(... lsxstorage)` on the host-only target that now needs it.

- [ ] **Step 4: Run and confirm pass**, then prove no regression:
  `ctest --test-dir build/linux-debug -R "test_storage_roots|test_core|test_path" `.

- [ ] **Step 5: Commit** — `refactor: move model-root resolution into libs/lsxstorage behind forwarders`

### Task 4a: Remove the `logestix/` path segment

**Why.** Every converted tree used to live at `<base>/<Model>/logestix/` inside the
source vault, so 538 sites spell `fs::path(x) / "logestix"`. The artifact store makes
that segment redundant (spec §2.4.2), and it is actively harmful because three
detection sites use the folder *name* as the "is this a converted tree?" marker
instead of `artifact.fb`.

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
  `<LSX_ARTIFACT_ROOT>/<family>/<model_type>/<profile>/{artifact.fb,pinned,experts,…}`.
  No API change; `cfg_.model_path` stays the thing every join hangs off.

- [ ] **Step 1: Write the failing guard test** (the shape of `test_logging_guard`:
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

- [ ] **Step 2: Run and confirm failure** — the guard finds 538 sites.

- [ ] **Step 3: Implement** the rewrite. Order matters: (a) guard test in place, (b)
  scripted `/"logestix"` removal across the three trees, (c) hand-fix the three
  detector call sites (`IsArtifactDir`) and the `ResolvePinnedDir` fallbacks, (d)
  `ResolvePinnedDir` collapses to `model_path / "pinned"`, (e) run the full `fast`
  sweep.

- [ ] **Step 4: Run and confirm pass** — `ctest --preset linux-debug`. Note honestly in
  the commit message that the *positive* path — a real converted artifact loading from
  the new layout — cannot be proven until Task 9a exists; what this task proves is
  that no site still spells the old segment and that the detectors key on
  `artifact.fb`.

- [ ] **Step 5: Commit** — `refactor: drop the redundant logestix path segment from artifact layout`

### Task 5: `artifact.fb` schema, reader, verifier

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

- [ ] **Step 1: Write the failing test:**

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

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** `ReadArtifact` = `MemoryMappedFile::Open` (mmap only, per
  AGENTS.md) → `flatbuffers::Verifier` over the mapped range → plain-struct copy-out.
  `artifact.h` **must not** include flatbuffers: the `.fb` accessors live in an
  anonymous namespace inside `artifact.cpp`. Generate the three fixtures as
  `.fb` files committed under `tests/fixtures/storage/` (small, produced by the
  writer in Task 6).

- [ ] **Step 4: Run and confirm pass.**

- [ ] **Step 5: Commit** — `feat: add artifact.fb FlatBuffers schema with mmap reader and verifier`

### Task 6: `EmitArtifact` — writer with digests and tokenizer copy

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

- [ ] **Step 1: Write the failing test:**

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

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement**: write to `artifact.fb.tmp`, `fsync`, `rename`; digest
  every plane with SHA-256 over the mapped file; copy the tokenizer and template
  copies; then `ReadArtifact` + `VerifyArtifact(deep=false)` what was written and
  return 0 + `*error` on any disagreement.

- [ ] **Step 4: Run and confirm pass.**

- [ ] **Step 5: Commit** — `feat: add atomic artifact.fb writer with plane digests and tokenizer copy`

### Task 7: `VerifyArtifact` + `lsxforge --verify` integration

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

- [ ] **Step 1: Write the failing test:**

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

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** `VerifyArtifact` walks the tree, classifies each file with
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

- [ ] **Step 4: Run and confirm pass**; then
  `ctest -R "test_storage_verify|test_arrow_tree_integrity"`.

- [ ] **Step 5: Commit** — `feat: verify artifact.fb masks and digests against the converted tree`

### Task 8: `--codec-profile` in `lsxforge`, emitter wired once

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

- [ ] **Step 1: Write the failing test:**

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

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** Parse → validate → **`PreflightDisk`** against the
  estimated artifact size → run the selected pipeline → on `arrow::Status::OK()`,
  build the `ArtifactSpec` from `DeclaredPlanes(...)` and call `EmitArtifact` into
  the profile dir from `lsx::storage::ArtifactDir(family, model, profile)`. On
  emitter failure the conversion result becomes a failure (a tree without
  `artifact.fb` is not a product). One call site, 40 pipelines covered.
  **Every profile stops writing `mtp/`**: the nine converters that emit it lose that
  code path in this task, so no artifact can carry an unreachable plane.

- [ ] **Step 4: Run and confirm pass**; one real conversion of a tiny model into a
  temp artifact root, then `lsxforge --verify` on it.

- [ ] **Step 5: Commit** — `feat: add --codec-profile and emit artifact.fb from every forge pipeline`

### Task 9: The admission gate in `ModelBase::BeginLoad`

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

- [ ] **Step 1: Write the failing test:**

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

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement** the five-stage `BeginLoad` from spec §2.5, with the exact
  refusal texts. `Resolve` precedence: (a) path already contains `artifact.fb`
  → `kArtifact`; (b) artifact root set and a compatible profile of that model exists
  → `kArtifact`; (c) source dir exists, or `LSX_MODEL_ROOT/<Model>/logestix` exists
  → `kSourceOnly` (the old vault layout is *not* a loadable tree: no `artifact.fb`);
  (d) `kNotFound`.
  There is **no `kLegacy`** — spec §2.4.1.

- [ ] **Step 4: Run and confirm pass**, then the suite guards:
  `ctest -R "test_load_codec_gate|test_model_suite_guard|test_model_independence_guard|test_manifest_sync"`.
  Fix any fast test that went red because its fixture is not marked `synthetic` —
  that is the gate working, and it is the expected work in this task.

- [ ] **Step 5: Add a guard that the empty store stays diagnostic.** With zero
  artifacts, the failure a user actually hits is "I pointed at a source dir". Assert
  the exact message names the source dir, says it is not a LogestiX artifact, and
  gives the `lsxforge` command — for a source dir, for an empty directory, and for a
  directory that contains `pinned/` and a `model.index.json` but no `artifact.fb`
  (a hand-rolled or partially-converted tree: the shape that would have been
  loadable before the contract, and must not be now).

- [ ] **Step 6: Commit** — `feat: enforce the hardware codec gate in BeginLoad`

### Task 9a: Bootstrap conversion — the task that turns the real-weight suite back on

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

- [ ] **Step 1: Convert the two slice models, `bf16` first**

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

- [ ] **Step 2: Assert the contract on real bytes (this is the test that cannot be
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

- [ ] **Step 3: Run the gates that were dark**, on the 5090, sequentially
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

- [ ] **Step 5: Commit** — `feat: bootstrap the artifact store with the bf16 and nvfp4 slices`

### Task 9b: Per-layer codec map leaves `model.index.json`

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

### Task 10: `MemoryModel` (unified memory) in the governor

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

**Branch 1 exit criteria:** `ctest --preset linux-debug` green (all `fast` guards,
~3 min); **Task 9a done** — two `bf16` artifacts converted, verified, gating clean on
both `sm_120` and `sm_86`, with their SelfCheck ×3 + FSI + CODE + DET running (not
SKIPping) on the 5090; the `nvfp4` siblings exist and `--explain-artifact` shows the
target-dependent pick; **Task 9b done** — no `model.index.json` in the tree carries a
codec key, and the inkling/gemma suites are green reading the artifact instead;
**Task 4a done** — the path-shape guard is green and the bootstrapped artifacts live at
`<ROOT>/<family>/<model_type>/<profile>/` with no intermediate segment;
`docs/superpowers/specs/…-design.md` committed; `docs/numeric-contract.md` §1/§1.1
amended; `docs/OPEN_TASKS.md` records the cold start and the artifact root.

---

# Branch 2 — `build/ampere-aot-sm86-sm87`

Branches from Branch 1 (needs `SupportedCodecs`). Scope: build + `bf16` profile +
live 3090 acceptance. **No new GEMM** (operator option c).

### Task 11: `LSX_CUDA_ARCH` accepts Ampere and emits both SASS

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

Branches from Branch 2. Scope: `int8-w8a16` profile + deterministic W8A16 decode GEMV
and prefill GEMM, validated on a live 3090 and Orin NX 16 GB.

**Numeric note that shapes every task:** W8A16 has no mixed s8×f16 tensor-core
instruction on GA10x. It is a **storage** win: weights are `int8`, dequantized to
f16/bf16 in the epilogue, accumulated in f32. Activations stay f16/bf16 — **no
activation quantization**, so the numerics are bounded by per-channel weight
rounding and the path stays deterministic and device-property-independent (R3 within
the `{sm_86, sm_87}` set).

### Task 17: Per-channel symmetric INT8 weight quantizer

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
