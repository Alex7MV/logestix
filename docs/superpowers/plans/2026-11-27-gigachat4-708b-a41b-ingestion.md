# GigaChat4-708B-A41B ingestion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest `GigaChat4-708B-A41B` (710 B params, 720 GB FP8 source) into
logestix as a new sibling row of the existing GigaChat4 family, with every
Model Ingestion Protocol stage (1–9) green on synthetic fixtures and on the real
converted weights.

**Architecture:** The family layer (decode pipeline, attention, GDN, norm, DSA
indexer) is already runtime-parameterised by `GcPipelineDims` and serves both
shipped GigaChat checkpoints. This plan adds a third row by (a) giving the two
remaining frozen touch points — the converter's geometry gate and the loader's
passport check — explicit *expectation* parameters, and (b) adding a per-model
frozen passport, converter entry point and `IModel` TU. No kernel is duplicated,
and the new row reads no constant from another model's passport.

**Tech Stack:** C++20 host code (absl), CUDA 13.5 / SM 12.0 device code (not
touched), Arrow IPC weight format, Catch2 tests, ctest via
`tools/gen_model_manifest.py`, NVFP4 (E2M1 + E4M3 block-16 scales).

**Spec:** `docs/superpowers/specs/2026-11-27-gigachat4-708b-a41b-ingest-design.md`

## Global Constraints

- Target is Blackwell SM 12.0 only (one RTX 5090, 32 GB). No pre-Blackwell code.
- Logging only through `lsxcommon::log`; `absl::StrFormat` is printf-style.
- No abseil containers in headers reachable from `.cu`/`.cuh`.
- Frozen geometry is compile-time (`static_assert`), never read from
  `config.json` at run time; `Load` must fail on a passport mismatch.
- One GPU workload at a time (`RESOURCE_LOCK "gpu"` is per ctest process — check
  `nvidia-smi --query-compute-apps` before starting a device run).
- Real-weight gates run on the Release/AOCC tree (`build/linux-release-znver4-aocc`),
  Debug only for the model-scoped correctness tests.
- Never rewrite a golden or a perf row to make a gate pass.
- Model-scoped verification only: `ctest --preset linux-debug` (guards) plus
  `-R "gigachat" -j 1`. No label sweeps.
- Conversion output root is the vault dir:
  `/mnt/fast_storage/huggingface/GigaChat4-708B-A41B/logestix` (861 GB free,
  ~410 GB artifact).
- Temp scratch on the fast disk: `/mnt/fast_storage/tmp`.

## Commit strategy

The repository forces an atomic model row: `tools/gen_model_manifest.py` (the
only registry) refuses a manifest whose `model_class`/targets do not exist, the
generated `model_registry.inc` cannot compile without the model TU, and the
orphan check rejects a `tests/test_*.cpp` that no manifest claims. Therefore:

- **Commit 1 (`refactor:`)**: the shared-family-core parameterisation (Tasks
  1). Self-consistent: no new test files, no manifest change; the two shipped
  rows prove behaviour is unchanged.
- **Commit 2 (`feat:`)**: Tasks 2–7 together — the whole 708B row plus its tests
  and manifest. Each task is developed red-first; the row lands in one commit
  because the guards require it.
- **Commit 3 (`test:`)**: the perf row seeded through the harness (Task 10).
- **Commit 4 (`docs:`)**: stage-9 report and the open-task/matrix rows.

## File Structure

| file | responsibility |
|---|---|
| `libs/lsxcommon/src/gigachat_moe_708b_constants.h` | the 708B frozen passport (leaf header) |
| `libs/lsxcommon/include/lsxcommon/gigachat_moe_708b_convert.h` | 708B converter entry declaration + its frozen spec accessor |
| `libs/lsxcommon/src/gigachat_moe_708b_convert.cpp` | `ConvertGigaChat4_708BModel()` → shared core with the 708B `Gc4IngestSpec` |
| `libs/lsxcommon/include/lsxcommon/gigachat_moe_708b_model.h` | `GigaChat4Moe708BModel` `IModel` declaration |
| `libs/lsxcommon/src/gigachat_moe_708b_model.cpp` | Load / Infer / Unload / Inspect / SelfCheck / Tokenizer + stage-4b bridge |
| `libs/lsxcommon/src/gigachat_moe_37b_convert.{h,cpp}` | shared converter core (`Gc4IngestSpec`, per-layer expert flush, vram classifier fix) |
| `libs/lsxcommon/src/gigachat_moe_37b_load.{h,cpp}` | shared loader + `Gc4FrozenExpect` |
| `libs/lsxcommon/src/gigachat_moe_37b_inference_indexer.{h,cu}`, `..._inference_pipeline.cu` | indexer dims knobs (no foreign passport read) |
| `tests/test_gigachat4_fixture.h` | renamed family fixture (`test_gigachat_37b_fixture.h` → this) + the 708B index writer |
| `tests/test_gigachat_moe_708b_{kernels,convert,loader,weight_graph,model,tokenizer_parity}.cpp` | per-stage tests |
| `models/gigachat4-708b-a41b.json` | the only hand-written registry |

## Interfaces (exact, later tasks depend on these)

```cpp
// gigachat_moe_37b_convert.h  (shared family core)
struct Gc4FrozenGeometry {
    int hidden, layers, vocab, heads, kv_heads, num_experts, top_k,
        moe_intermediate, dense_intermediate, first_k_dense,
        q_lora, kv_lora, qk_nope, qk_rope, v_head;
};
struct Gc4IngestSpec {
    const char *format;                       // index "format"
    const char *model_name;                   // index "model"
    Gc4FrozenGeometry frozen;                 // gate + parse defaults
    std::vector<int> full_attention_layers;   // expected layer map
    float attn_scale;                         // index "attn_scale"
};
Gc4IngestSpec Gc4DefaultIngestSpec();         // the 37B passport's spec
Gc4Topology ParseGigaChat4Config(const std::string &config_json_text,
                                 const Gc4FrozenGeometry &defaults);
arrow::Status ConvertGigaChat4ModelWithSpec(const std::string &input_dir,
                                           const std::string &output_root,
                                           const Gc4IngestSpec &spec);

// gigachat_moe_37b_load.h  (shared family loader)
struct Gc4FrozenExpect {
    const char *format;
    Gc4FrozenGeometry geo;
    std::vector<int> full_attention_layers;
    int gate_rank;
};
Gc4FrozenExpect Gc4DefaultExpect();           // the 37B passport's expectation
bool LoadGigaChat4IndexWithFormat(const std::string &model_path,
                                 const std::string &expected_format,
                                 Gc4Index &out, std::string &error);
bool ValidateGigaChat4GeometryAgainst(const Gc4Index &idx,
                                     const Gc4FrozenExpect &expect,
                                     std::string &error);
bool LoadGigaChat4WeightsWithExpect(const std::string &model_path,
                                   const Gc4Index &idx, Gc4Weights &w,
                                   cudaStream_t stream,
                                   const Gc4FrozenExpect &expect,
                                   std::string &error);

// gigachat_moe_708b_convert.h
Gc4IngestSpec Gc708BIngestSpec();     // spec built from the 708B passport
arrow::Status ConvertGigaChat4_708BModel(const std::string &input_dir,
                                        const std::string &output_root);

// gigachat_moe_708b_model.h
class GigaChat4Moe708BModel final : public ModelBase { /* kType = "gigachat4-708b-a41b" */ };
```

## Review Focus

- **A 37B constant left in a 708B path.** The converter's VRAM profile and any
  `frozen::` read inside a shared TU are the two places this happens silently;
  the symptom is a model that loads and answers with wrong widths. Pinned by the
  `vram` block assertion (Task 3) and the derived-width test (Task 2).
- **An expert set that does not flush per layer.** Accumulating 386.9 GB in one
  `flat_hash_map` is an OOM at the end of a multi-hour conversion, not at the
  start. Pinned by the per-layer flush counter in the convert test (Task 3).
- **A tree written with the wrong format tag.** A 708B row must never accept a
  37B tree (or vice versa) even when the geometry check is bypassed for a
  synthetic fixture. Pinned by the cross-rejection cases (Task 4).
- **Session cap vs. VRAM.** `max_seq_ = 4096` costs 5.4 GB of KV on this card;
  at 2048 the canonical 2048-token perf protocol no longer fits prompt +
  generation. Pinned by the model-stage context-limit case (Task 5).
- **A SKIP read as a pass.** The parity goldens and the perf row depend on a
  410 GB artifact; every gate that skips must be named as a gap in the report.

---

### Task 1: Parameterise the shared family core (no behaviour change)

**Files:**
- Modify: `libs/lsxcommon/src/gigachat_moe_37b_convert.cpp:243-356` (`ParseGigaChat4Config`), `:426-807` (`ConvertGigaChat4Model` body → `ConvertGigaChat4ModelWithSpec`), `:698-721` (vram classifier)
- Modify: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_convert.h`
- Modify: `libs/lsxcommon/src/gigachat_moe_37b_load.cpp:150-170` (format gate), `:190-280` (`ValidateGigaChat4Geometry` → `...Against`), `:316-321` (gate rank)
- Modify: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_load.h`
- Modify: `libs/lsxcommon/src/gigachat_moe_37b_inference_indexer.cu:429,449`
- Modify: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_inference_indexer.h` (`GcDsaIndexerDims`)
- Modify: `libs/lsxcommon/include/lsxcommon/gigachat_moe_inference_pipeline.h` (`GcPipelineDims`), `libs/lsxcommon/src/gigachat_moe_inference_pipeline.cu:372-383`
- Test: existing `tests/test_gigachat_moe_37b_convert.cpp`, `test_gigachat_moe_37b_loader.cpp`, `test_gigachat_moe_37b_weight_graph.cpp`, `test_gigachat_moe_selfcheck.cpp` (unchanged, must stay green)

**Interfaces:**
- Produces: `Gc4FrozenGeometry`, `Gc4IngestSpec`, `Gc4DefaultIngestSpec()`,
  `ParseGigaChat4Config(text, defaults)`, `ConvertGigaChat4ModelWithSpec(...)`,
  `Gc4FrozenExpect`, `Gc4DefaultExpect()`, `LoadGigaChat4IndexWithFormat(...)`,
  `ValidateGigaChat4GeometryAgainst(...)`, `LoadGigaChat4WeightsWithExpect(...)`,
  `GcPipelineDims::index_onorm_eps`, `GcPipelineDims::index_score_scale`,
  `GcDsaIndexerDims::onorm_eps`, `GcDsaIndexerDims::score_scale`.

- [ ] **Step 1: Record the green baseline**

Run: `ctest --test-dir build/linux-debug -R "gigachat" -j 1`
Expected: the 37B and 3.5 rows green; note the 37B selfcheck `logits_hash` from
its log for the re-check in Step 6.

- [ ] **Step 2: Add the spec/expect types and keep the old entry points as wrappers**

`ConvertGigaChat4Model(dir, root)` → `ConvertGigaChat4ModelWithSpec(dir, root,
Gc4DefaultIngestSpec())`; `ParseGigaChat4Config(text)` →
`ParseGigaChat4Config(text, Gc4DefaultIngestSpec().frozen)`;
`LoadGigaChat4Index` → `LoadGigaChat4IndexWithFormat(..., frozen::kFormat, ...)`;
`ValidateGigaChat4Geometry` → `...Against(idx, Gc4DefaultExpect(), error)`;
`LoadGigaChat4Weights` → `...WithExpect(..., Gc4DefaultExpect(), ...)`. The
frozen gate inside the spec path compares `top.*` against `spec.frozen` and
`top.full_attention_layers` against `spec.full_attention_layers`, and writes
`spec.format` / `spec.model_name` / `spec.attn_scale` into the index.

- [ ] **Step 3: Fix the two defects the 708B row would inherit**

(a) The `VramProfile` classifier lambdas (`gigachat_moe_37b_convert.cpp:713-720`)
must call `classify_gigachat4_tensor(n, top.layers)`; the one-argument overload
reads the passport depth and would classify every layer ≥ 32 as MTP ballast.
(b) `RunGigaChatDsaIndexerStep` must take `eps` from `d.onorm_eps` and the score
scale from `d.score_scale`; `GcPipeline::Impl::Init` fills both from
`dims.index_onorm_eps` / `dims.index_score_scale`, which default to
`1e-6f` and `0.08838834764831845f`.

- [ ] **Step 4: Flush experts per layer in the shared converter**

Replace the single `experts` map sweep with: accumulate one layer's experts,
write `experts/layer_<L>/exp_<E>.arrow`, then erase that layer's entries.
Peak growth becomes 256 × 3 records instead of 15 616. Keep the
`incomplete expert` error and the `w1=gate, w3=up, w2=down` slot convention.

- [ ] **Step 5: Build both trees**

Run: `cmake --build --preset linux-debug --target lsxcommon lsxforge test_gigachat_moe_37b_convert test_gigachat_moe_37b_loader test_gigachat_moe_37b_weight_graph`
Expected: builds clean (no new warnings).

- [ ] **Step 6: Prove the refactor changed no behaviour**

Run: `ctest --test-dir build/linux-debug -R "gigachat" -j 1`
Expected: identical results to Step 1; the 37B selfcheck `logits_hash` is the
one recorded in `docs/open-tasks/live/models/gigachat4-37b-a4b.md`.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_37b_* libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_* \
        libs/lsxcommon/src/gigachat_moe_inference_pipeline.cu \
        libs/lsxcommon/include/lsxcommon/gigachat_moe_inference_pipeline.h
git commit -m "refactor: parameterise the GigaChat4 family core on ingest expectations"
```

---

### Task 2: The 708B frozen passport and the naming token

**Files:**
- Create: `libs/lsxcommon/src/gigachat_moe_708b_constants.h`
- Create: `tests/test_gigachat_moe_708b_kernels.cpp`
- Modify: `tests/test_code_naming_guard.cpp` (`SchemeRegex()` variant list + one acceptance `CHECK`)

**Interfaces:**
- Produces: `namespace lsxcommon::gigachat4_708b::frozen` with `kType =
  "gigachat4-708b-a41b"`, `kFormat = "logestix-gigachat4-708b-nvfp4"`,
  `kModelName = "GigaChat4-708B-A41B"`, `kHidden 7168`, `kLayers 64`,
  `kVocab 128256`, `kFirstKDense 3`, `kGateRank 16`, `kLogitSoftcap 30.0f`,
  `kFullAttnLayers[16] = {3,7,…,63}`, `kNumFullAttn 16`, `kMtpLayer 64`,
  `kHeads 64`, `kKVHeads 64`, `kQkNopeHeadDim 128`, `kQkRopeHeadDim 64`,
  `kQkHeadDim 192`, `kVHeadDim 128`, `kQLoraRank 1536`, `kKvLoraRank 512`,
  `kRopeTheta 1e5f`, `kYarn*`, `kAttnScale 0.13086079996295008f`,
  `kAlphaQ 2.160246899469287f`, `kAlphaKv 3.7416573867739413f`,
  `kLinearNumKeyHeads 32`, `kLinearNumValueHeads 64`, `kLinearKeyDim 4096`,
  `kLinearValueDim 8192`, `kLinearConvDim 16384`, `kLinearQkvzDim 24576`,
  `kLinearBaDim 128`, `kNumRouted 256`, `kTopK 8`, `kMoeIntermediate 2048`,
  `kSharedIntermediate 2048`, `kDenseIntermediate 18432`, `kRoutedScale 2.5f`,
  `kSwigluLimit 10.0f`, `kIndexHeads 64`, `kIndexHeadDim 128`,
  `kIndexTopk 2048`, `kIndexRope 64`, token ids `kBosId 1`, `kEosId 2`,
  `kPadId 2`, `kRoleSepId 128000`, `kMessageSepId 128001`, `kThinkId 128014`,
  `kThinkEndId 128015`.

- [ ] **Step 1: Write the failing test**

`tests/test_gigachat_moe_708b_kernels.cpp`:

```cpp
// TDD — GigaChat4-708B-A41B passport derivations and the shared variant math.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gigachat_moe_708b_constants.h"
#include "gigachat_moe_device_math.h"
#include <cmath>

namespace frozen = lsxcommon::gigachat4_708b::frozen;

TEST_CASE("gigachat4-708b derived geometry matches the checkpoint", "[gigachat708b][fast]")
{
    CHECK(frozen::kQkHeadDim == 192);
    CHECK(frozen::kLinearKeyDim == 4096);
    CHECK(frozen::kLinearValueDim == 8192);
    CHECK(frozen::kLinearConvDim == 16384);
    CHECK(frozen::kLinearQkvzDim == 24576);
    CHECK(frozen::kLinearBaDim == 128);
    CHECK(frozen::kSharedIntermediate == 2048);
    CHECK(frozen::kIndexHeads * frozen::kIndexHeadDim == 8192);
    CHECK(frozen::kNumFullAttn == 16);
    CHECK(frozen::kMtpLayer == 64);
    // alpha = sqrt(hidden / rank) — the two values that differ from the 37B row.
    CHECK(frozen::kAlphaQ == Catch::Approx(std::sqrt(7168.0 / 1536.0)).margin(1e-6));
    CHECK(frozen::kAlphaKv == Catch::Approx(std::sqrt(7168.0 / 512.0)).margin(1e-6));
    CHECK(frozen::kAlphaQ == Catch::Approx(2.160246899469287).margin(1e-9));
    CHECK(frozen::kAlphaKv == Catch::Approx(3.7416573867739413).margin(1e-9));
    // geometry-independent: same formula as the 37B row.
    const double mscale = 0.1 * std::log(32.0) + 1.0;
    CHECK(frozen::kAttnScale ==
          Catch::Approx(std::pow(192.0, -0.5) * mscale * mscale).margin(1e-6));
}

TEST_CASE("gigachat4-708b variant math is the 4.0 form at 708b widths", "[gigachat708b][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    CHECK(dev::GatedNormScale(1.0f, 2.0f, 0.0f) == Catch::Approx(2.0f));
    CHECK(dev::GdnOutputNormElementOffset(1.0f, 2.0f, 0.0f, 1e-6f, 2.0f, 0.0f) ==
          Catch::Approx(2.0f).margin(1e-6f));
    CHECK(dev::LogitSoftcapElement(1.0e6f, frozen::kLogitSoftcap) ==
          Catch::Approx(30.0f).margin(1e-3f));
    CHECK(frozen::kMoeIntermediate != 768);   // a 37B value must not survive here
    CHECK(frozen::kSharedIntermediate == frozen::kMoeIntermediate);
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_708b_kernels`
Expected: FAIL — `gigachat_moe_708b_constants.h: No such file` (and the target
does not exist yet, so add the `add_executable` block from Task 7 Step 3 first
if the configure has not run).

- [ ] **Step 3: Write the passport**

Transcribe the constants listed in **Interfaces** into
`gigachat_moe_708b_constants.h` with a header comment naming the measured
source (`config.json` fields, the 61 MoE layers, the one MTP layer, the 16
full-attention layers) and `static_assert`s mirroring the 37B header's list plus
`kLayers == 64`, `kNumFullAttn == 16`, `kMoeIntermediate == 2048`,
`kLinearQkvzDim == 24576`, `kLinearConvDim == 16384`, `kHidden % 128 == 0`.

- [ ] **Step 4: Add the variant token to the naming scheme**

In `tests/test_code_naming_guard.cpp`, add `708b` to the `var` alternation and a
`CHECK(m("gigachat_moe_708b_constants"))` (plus `_convert`, `_model`) to the
acceptance block.

- [ ] **Step 5: Run the two tests**

Run: `ctest --test-dir build/linux-debug -R "gigachat708b|code_naming" -j 1`
Expected: PASS.

- [ ] **Step 6: Commit** — with Task 7 (see Commit strategy).

---

### Task 3: Converter entry point, per-layer flush and the frozen gate

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_708b_convert.h`, `libs/lsxcommon/src/gigachat_moe_708b_convert.cpp`
- Create: `tests/test_gigachat_moe_708b_convert.cpp`
- Modify: `tests/test_gigachat4_fixture.h` (rename from `test_gigachat_37b_fixture.h`; 5 include sites)
- Modify: `apps/lsxforge/src/main.cpp` (`ABSL_FLAG(gigachat4_708b)`, dispatch), `apps/lsxforge/src/cli.{h,cpp}` (`--gigachat4-708b`)

**Interfaces:**
- Consumes: `Gc4IngestSpec`, `ConvertGigaChat4ModelWithSpec`, `classify_gigachat4_tensor`.
- Produces: `Gc708BIngestSpec()`, `ConvertGigaChat4_708BModel(dir, root)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gigachat_moe_708b_convert.cpp` (label `convert`, synthetic only):

```cpp
TEST_CASE("gigachat4-708b config defaults are the frozen passport", "[gigachat708b][convert]")
{
    // An EMPTY config object exposes the parse defaults: they must be the 708B
    // passport, never the 37B's.
    const auto t = ParseGigaChat4Config("{\"hidden_size\":7168,\"num_hidden_layers\":64,"
                                        "\"vocab_size\":128256}", Gc708BIngestSpec().frozen);
    CHECK(t.hidden == 7168);  CHECK(t.layers == 64);
    CHECK(t.num_experts == 256); CHECK(t.moe_intermediate == 2048);
    CHECK(t.dense_intermediate == 18432); CHECK(t.first_k_dense == 3);
    CHECK(t.q_lora == 1536);  CHECK(t.kv_lora == 512);
    CHECK(t.heads == 64);     CHECK(t.kv_heads == 64);
}

TEST_CASE("gigachat4-708b converter writes the 708b index", "[gigachat708b][convert]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    // synthetic checkpoint (tiny geometry) -> convert -> read model.index.json
    // CHECK format == "logestix-gigachat4-708b-nvfp4"
    // CHECK model  == "GigaChat4-708B-A41B"
    // CHECK attn_scale == 0.13086079996295008
    // CHECK deferred.mtp_layer == <synthetic layers>, deferred.dflash2 mentions "not ingested"
    // CHECK every expert has gate/up/down; count == layers_with_moe * experts
}

TEST_CASE("gigachat4-708b vram profile counts all 61 MoE layers", "[gigachat708b][convert]")
{
    // Synthetic checkpoint with FROZEN 708B layer count is too big to write, so
    // this case pins the classifier contract directly: for a 708-shaped index
    // every layer >= 32 must classify as a decoder layer, not as MTP ballast.
    CHECK(classify_gigachat4_tensor("model.layers.40.mlp.experts.0.up_proj.weight", 64).kind
          == Gc4TensorKind::ExpertUp);
    CHECK(classify_gigachat4_tensor("model.layers.64.eh_proj.weight", 64).kind
          == Gc4TensorKind::Mtp);
    CHECK(classify_gigachat4_tensor("model.layers.32.eh_proj.weight", 32).kind
          == Gc4TensorKind::Mtp);
}

TEST_CASE("gigachat4-708b frozen gate rejects the 37B geometry", "[gigachat708b][convert]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0", 1);
    // A config carrying the 37B geometry (32 layers / 2560 hidden) must be
    // refused by ConvertGigaChat4_708BModel with "frozen geometry mismatch".
    // (Write the config only — the gate fires before any shard is read.)
}

TEST_CASE("gigachat4-708b experts are flushed per layer", "[gigachat708b][convert]")
{
    // Two MoE layers in the synthetic checkpoint: the converter must produce
    // every expert file AND keep peak residency to one layer. The observable
    // contract is the file set + a conversion that completes on the synthetic
    // checkpoint; the flush itself is pinned by LSX_GC4_CONVERT_PEAK_EXPERTS
    // (a log line the test greps for: "peak experts in flight: N" with
    // N <= experts_per_layer).
}
```

- [ ] **Step 2: Run and watch them fail**

Run: `ctest --test-dir build/linux-debug -R "test_gigachat_moe_708b_convert" -j 1`
Expected: FAIL to compile — `gigachat_moe_708b_convert.h` missing.

- [ ] **Step 3: Implement the entry point**

`Gc708BIngestSpec()` returns `{frozen::kFormat, frozen::kModelName, {7168, 64, 128256, 64, 64, 256, 8, 2048, 18432, 3, 1536, 512, 128, 64, 128}, {3,7,…,63}, frozen::kAttnScale}`.
`ConvertGigaChat4_708BModel` is a one-line delegation to
`ConvertGigaChat4ModelWithSpec(dir, root, Gc708BIngestSpec())`.

- [ ] **Step 4: Rename the family fixture and add the 708B index writer**

`git mv tests/test_gigachat_37b_fixture.h tests/test_gigachat4_fixture.h`, update
the five includes, and add:

```cpp
inline void WriteGigaChat4_708BIndex(const std::filesystem::path &model_root,
                                     int hidden_override = 7168,
                                     const std::string &format_tag = "logestix-gigachat4-708b-nvfp4")
```
mirroring `WriteGigaChat4Index` with `num_layers 64`, the 16-layer map,
`num_experts 256`, `moe_intermediate_size 2048`, `first_k_dense_replace 3`,
`deferred.mtp_layer 64`.

- [ ] **Step 5: Run the convert tests**

Run: `ctest --test-dir build/linux-debug -R "test_gigachat_moe_708b_convert" -j 1`
Expected: PASS (5 cases).

- [ ] **Step 6: Commit** — with Task 7.

---

### Task 4: Loader expectations — accept 708B, reject 37B

**Files:**
- Create: `tests/test_gigachat_moe_708b_loader.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/gigachat_moe_708b_model.h` (declares `Gc708BExpect()` if it lives beside the model)

**Interfaces:**
- Consumes: `LoadGigaChat4IndexWithFormat`, `ValidateGigaChat4GeometryAgainst`, `Gc4FrozenExpect`.
- Produces: `Gc4FrozenExpect Gc708BExpect()` (format `logestix-gigachat4-708b-nvfp4`, the 708 geometry, gate rank 16).

- [ ] **Step 1: Write the failing test**

```cpp
TEST_CASE("gigachat4-708b index matches its own passport", "[gigachat708b][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc708_idx_ok");
    gigachat4test::WriteGigaChat4_708BIndex(root);
    Gc4Index idx; std::string err;
    REQUIRE(LoadGigaChat4IndexWithFormat(root.string(), frozen::kFormat, idx, err));
    CHECK(idx.num_layers == 64); CHECK(idx.hidden_size == 7168);
    CHECK(idx.num_experts == 256); CHECK(idx.moe_intermediate_size == 2048);
    CHECK(idx.first_k_dense_replace == 3);
    CHECK(idx.full_attention_layers.size() == 16);
    CHECK(ValidateGigaChat4GeometryAgainst(idx, Gc708BExpect(), err));
}

TEST_CASE("gigachat4-708b refuses the 37B tree and vice versa", "[gigachat708b][load]")
{
    const auto r37 = gigachat4test::MakeTempDir("gc708_idx_37");
    gigachat4test::WriteGigaChat4Index(r37);
    Gc4Index idx; std::string err;
    // format tag mismatch
    CHECK_FALSE(LoadGigaChat4IndexWithFormat(r37.string(), frozen::kFormat, idx, err));
    CHECK(err.find("format") != std::string::npos);
    // and even with the tag forced, the geometry gate refuses it
    REQUIRE(LoadGigaChat4IndexWithFormat(r37.string(), "logestix-gigachat4-nvfp4", idx, err));
    CHECK_FALSE(ValidateGigaChat4GeometryAgainst(idx, Gc708BExpect(), err));
    CHECK(err.find("num_layers") != std::string::npos);
}
```

- [ ] **Step 2: Run and watch it fail** — `cmake --build --preset linux-debug --target test_gigachat_moe_708b_loader`; expected: compile error, `Gc708BExpect` unknown.

- [ ] **Step 3: Implement `Gc708BExpect()`** (returns the 708 format + geometry + gate rank 16) in `gigachat_moe_708b_model.cpp` with its declaration in the model header.

- [ ] **Step 4: Run** — `ctest --test-dir build/linux-debug -R "test_gigachat_moe_708b_loader" -j 1`; expected PASS.

- [ ] **Step 5: Commit** — with Task 7.

---

### Task 5: The `IModel` — load, decode, stream, teardown

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_708b_model.h`, `libs/lsxcommon/src/gigachat_moe_708b_model.cpp`
- Create: `tests/test_gigachat_moe_708b_weight_graph.cpp`, `tests/test_gigachat_moe_708b_model.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add the two sources to `lsxcommon`)

**Interfaces:**
- Consumes: everything from Tasks 1–4, `gigachat::GcPipeline`, `InitGigaChatExpertCache`.
- Produces: `GigaChat4Moe708BModel` with `static const std::string kType`,
  `Load/Infer/Unload/Inspect/SelfCheck/Tokenizer/BuildPromptInputIds/
  BuildConversationInputIds/ParseToolCalls`.

- [ ] **Step 1: Write the failing weight-graph test** — convert the synthetic checkpoint with `ConvertGigaChat4_708BModel` (synthetic bypass on), `LoadGigaChat4IndexWithFormat` + `ValidateGigaChat4GeometryAgainst(idx, Gc708BExpect())` under the bypass, `LoadGigaChat4WeightsWithExpect`, assert: the DSA indexer planes are non-null on full-attention layers and null on GDN layers; one routed expert resolves through the slot LRU; `w.resident_bytes > 0`.

- [ ] **Step 2: Run it, watch it fail** (target missing).

- [ ] **Step 3: Write the model TU**

Copy `gigachat_moe_37b_model.cpp` with: namespace `gigachat4_708b`, `kType` from
its passport, `gigachat4_708b::` / `Gc708BExpect()` for the loader calls, the 708
passport for `kQkNopeHeadDim`/`kQkRopeHeadDim`/`kVHeadDim`/`kLinear*HeadDim`/
`kRopeTheta`/`kYarn*`/`kIndexRope`, `max_seq_ = 4096`, and log prefix
`[gigachat4-708b]`. Everything else — `RunUnderResidencyLock`, the
`GenerationStream`/abort loop, `TeardownContext()`, `AcquirePermanentWeights`/
`AcquireExpertCacheLease`, `weight_resident_bytes`, the chatml-v5
`BuildPromptInputIds`, and the stage-4b delegations to
`RenderGigaChatConversation`/`AppendGigaChatIds`/`ParseGigaChatTokenCalls` — is
the same code (the template is byte-identical).

- [ ] **Step 4: Write the failing model test**

```cpp
TEST_CASE("gigachat4-708b load + decode streams one token per step", "[gigachat708b][model]")
{ /* synthetic tree; req.max_gen_tokens = 3; collect req.token_callback ticks;
     CHECK(ticks == 3); CHECK(r.ok); CHECK(r.num_layers == 2);
     CHECK(r.weight_resident_bytes > 0); */ }

TEST_CASE("gigachat4-708b honours a pre-set abort flag", "[gigachat708b][model]")
{ /* abort_flag set before Infer -> 0 decoded tokens, no crash */ }

TEST_CASE("gigachat4-708b refuses prompt + max_gen above the session cap", "[gigachat708b][model]")
{ /* input_ids = 4096 ids, max_gen = 2 -> error_msg mentions "context length exceeded" */ }
```

- [ ] **Step 5: Run both tests** — expected PASS after Step 3.

- [ ] **Step 6: Commit** — with Task 7.

---

### Task 6: Tokenizer parity + tool-calling verdict

**Files:**
- Create: `tests/test_gigachat_moe_708b_tokenizer_parity.cpp`
- Modify: `tests/test_model_suite_guard.cpp` (tool verdict row)

**Interfaces:**
- Consumes: the vault env `LSX_TEST_GIGACHAT4_708B_A41B_ROOT`; the shared `gigachat_tool_format` codec.

- [ ] **Step 1: Write the failing test** — clone the 37B tokenizer-parity test with the 708 env var and the 708 vault dir; cases: real-vault `<s>`/`</s>`/`<|role_sep|>\n`/`<|message_sep|>\n\n`/`<think>`/`</think>` ids equal the passport (SKIP when the vault is absent), and the synthetic-fixture BPE round-trip of `The capital of France is`.

- [ ] **Step 2: Run it** — expected PASS (SKIP is acceptable before Task 8).

- [ ] **Step 3: Add the tool verdict**

In `kTooling`, add `{"gigachat4-708b-a41b", {"test_gigachat_tool_framing_37b"}}`
with a one-line comment recording that `chat_template.jinja` and `tokenizer.json`
are byte-identical to the 37B's (sha256-verified at stage 0). No new codec.

- [ ] **Step 4: Run** — `ctest --test-dir build/linux-debug -R "test_model_suite_guard" -j 1`; expected PASS once Task 7 lands the manifest.

- [ ] **Step 5: Commit** — with Task 7.

---

### Task 7: The manifest, the executables and the generated catalogue

**Files:**
- Create: `models/gigachat4-708b-a41b.json`
- Modify: `tests/CMakeLists.txt` (`add_executable` blocks for the six new test TUs)
- Modify: `tests/perf/perf_baseline.cpp`? — no: `tests/test_perf_baseline.cpp` (model table row, Task 10)
- Regenerate: `docs/MODELS.md`

**Interfaces:**
- Produces: the `gigachat4-708b-a41b` ctest target set, the factory registration
  (`model_registry.inc`), the generated `model_targets.cmake`.

- [ ] **Step 1: Add the ten `add_executable` blocks**

Six source-backed TUs (convert, loader, weight_graph, model, kernels,
tokenizer_parity) with `lsx_copy_arrow_dlls(... ARROW_ONLY)` where the 37B block
does it, plus the four **shared drivers** the manifest's `form` does not create
by itself: `add_executable(test_gigachat_moe_708b_selfcheck test_selfcheck_driver.cpp)`
with `"LSX_SELFCHECK_MODEL=\"gigachat4-708b-a41b\""`, and the three parity gates
from `test_parity_gate.cpp` with `LSX_GATE_MODEL="gigachat4-708b-a41b"`,
`LSX_GATE_KIND="fsi|code|det"` and `LSX_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/{fsi,code,det}"`,
copying the `test_nemotron_ultra_*` blocks verbatim in shape. Each shared-driver
target needs `target_include_directories(... "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")`.

- [ ] **Step 2: Write the manifest**

```jsonc
{
  "model_type": "gigachat4-708b-a41b",
  "display_name": "gigachat4-708b-a41b",
  "family": "gigachat", "variant": "708b",
  "model_class": "GigaChat4Moe708BModel",
  "model_header": "../include/lsxcommon/gigachat_moe_708b_model.h",
  "passport": "libs/lsxcommon/src/gigachat_moe_708b_constants.h",
  "namespace": "lsxcommon::gigachat4_708b::frozen",
  "format_tag": "logestix-gigachat4-708b-nvfp4",
  "forge_flag": "--gigachat4-708b",
  "tokenizer_kind": 3,
  "vault": { "dir": "GigaChat4-708B-A41B" },
  "lifecycle_tu": ["gigachat_moe_708b_model.cpp"],
  "stages": {
    "convert": ["test_gigachat_moe_708b_convert"],
    "load": ["test_gigachat_moe_708b_loader", "test_gigachat_moe_708b_weight_graph",
             "test_gigachat_moe_708b_tokenizer_parity"],
    "model": ["test_gigachat_moe_708b_model"],
    "kernels": ["test_gigachat_moe_708b_kernels"],
    "selfcheck": ["test_gigachat_moe_708b_selfcheck"],
    "fsi": ["test_gigachat_moe_708b_fsi_selfcheck"],
    "code": ["test_gigachat_moe_708b_code_selfcheck"],
    "det": ["test_gigachat_moe_708b_det_selfcheck"],
    "perf": ["test_perf_baseline_gigachat4_708b"]
  },
  "perf": { "target": "test_perf_baseline_gigachat4_708b",
            "vault_subdir": "GigaChat4-708B-A41B", "timeout": 5400 },
  "target_overrides": {
    "test_gigachat_moe_708b_loader": { "label": "fast", "env": [] },
    "test_gigachat_moe_708b_kernels": { "label": "fast" },
    "test_gigachat_moe_708b_tokenizer_parity": { "label": "fast", "env": [] },
    "test_gigachat_moe_708b_selfcheck": { "form": "selfcheck" },
    "test_gigachat_moe_708b_fsi_selfcheck": { "form": "parity", "kind": "fsi" },
    "test_gigachat_moe_708b_code_selfcheck": { "form": "parity", "kind": "code" },
    "test_gigachat_moe_708b_det_selfcheck": { "form": "parity", "kind": "det" }
  },
  "tool_roundtrip": true, "auto_path_determinism": false,
  "cooperative_cancel": true,
  "selfcheck_frame_prompt": false, "selfcheck_max_gen": 2, "selfcheck_bit_exact": false,
  "golden_debt": {},
  "aliases": ["gigachat4-708b-a41b"],
  "notes": "GigaChat4 708B-A41B: 64 layers, 61 MoE (256 experts, top-8), 16 full-attention DSA layers; expert set streams from host RAM",
  "report": "docs/model-reports/2026-11-27-GigaChat4-708B-A41B.md"
}
```

- [ ] **Step 3: Regenerate and validate**

Run: `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`
Run: `python3 tools/gen_model_manifest.py --strict --root .`
Expected: no errors; `docs/MODELS.md` gains the row.

- [ ] **Step 4: Reconfigure and build the row**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug`
Expected: clean; `LsxModelFactory` now resolves `gigachat4-708b-a41b`.

- [ ] **Step 5: Run the fast guards**

Run: `ctest --preset linux-debug -j 24`
Expected: green, including `test_manifest_sync`, `test_model_suite_guard`,
`test_model_independence_guard`, `test_ctest_labels_guard`,
`test_code_naming_guard`, `test_expert_store_guard`.

- [ ] **Step 6: Run the model row**

Run: `ctest --test-dir build/linux-debug -R "gigachat" -j 1`
Expected: the two shipped siblings plus the new convert/load/model/kernels tests
green (selfcheck/parity/perf SKIP without the vault — they become the Task 9
work).

- [ ] **Step 7: Commit** — the whole row, with Tasks 2–6.

```bash
git add libs/lsxcommon/src/gigachat_moe_708b_* libs/lsxcommon/include/lsxcommon/gigachat_moe_708b_* \
        libs/lsxcommon/CMakeLists.txt tests/test_gigachat_moe_708b_* tests/test_gigachat4_fixture.h \
        tests/CMakeLists.txt tests/test_code_naming_guard.cpp tests/test_model_suite_guard.cpp \
        apps/lsxforge/src models/gigachat4-708b-a41b.json docs/MODELS.md
git commit -m "feat: GigaChat4-708B-A41B row — passport, converter, loader, model, tests, manifest"
```

---

### Task 8: Convert the real checkpoint (720 GB)

**Files:** none (artifact under `/mnt/fast_storage/huggingface/GigaChat4-708B-A41B/logestix`)

- [ ] **Step 1: Check the card is idle** — `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`; no compute apps, else wait.

- [ ] **Step 2: Build the release/AOCC tree** — `cmake --preset linux-release-znver4-aocc && cmake --build --preset linux-release-znver4-aocc --target lsxforge lsxbenchmark`

- [ ] **Step 3: Convert**

```bash
cd /mnt/fast_storage/projects/cuda/logestix
time build/linux-release-znver4-aocc/apps/lsxforge --gigachat4-708b \
  --input /mnt/fast_storage/huggingface/GigaChat4-708B-A41B \
  --output /mnt/fast_storage/huggingface/GigaChat4-708B-A41B 2>&1 | tee /mnt/fast_storage/tmp/gc708b-convert.log
```
Expected: `GigaChat4 conversion: 64 layers, 15616 experts, 11 MTP tensors stored`
and a ~410 GB tree; monitor `df -h /mnt/fast_storage` (must not go below ~50 GB).

- [ ] **Step 4: Verify the artifact**

```bash
python3 - <<'EOF'
import json; p='/mnt/fast_storage/huggingface/GigaChat4-708B-A41B/logestix/model.index.json'
d=json.load(open(p)); print(d['format'], d['num_layers'], d['num_experts'], d['deferred'])
EOF
du -sh /mnt/fast_storage/huggingface/GigaChat4-708B-A41B/logestix
ls /mnt/fast_storage/huggingface/GigaChat4-708B-A41B/logestix/experts | wc -l   # 61
```
Expected: format `logestix-gigachat4-708b-nvfp4`, 64 layers, 256 experts, 61
expert dirs, ~410 GB.

---

### Task 9: Real-weight gates

**Files:** `tests/fixtures/{fsi,code,det}/gigachat4-708b-a41b.json`

- [ ] **Step 1: Capture the parity goldens (template-derived, 37B precedent)**

```bash
python3 tools/fsi_capture.py --model /mnt/fast_storage/huggingface/GigaChat4-708B-A41B \
  --type gigachat4-708b-a41b --template-only \
  --out tests/fixtures/fsi/gigachat4-708b-a41b.json
# same for tools/code_capture.py -> tests/fixtures/code/, tools/det_capture.py -> tests/fixtures/det/
```
Expected: three golden files whose `model_type` is `gigachat4-708b-a41b` and
whose `source` records the template-derived deviation.

- [ ] **Step 2: SelfCheck on the release tree**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R "test_gigachat_moe_708b_selfcheck" -j 1 --output-on-failure`
Expected: PASS, ≥3 bit-exact cycles, top-1 `Paris`; record the `logits_hash`.

- [ ] **Step 3: Parity gates** — run the three `test_gigachat_moe_708b_{fsi,code,det}_selfcheck` targets one at a time. Expected 3/3 each. Any red is recorded as debt in `docs/open-tasks/live/models/gigachat4-708b-a41b.md` and named in the report, never patched into the golden.

- [ ] **Step 4: Source-side localisation (if a gate is red or SelfCheck is not `Paris`)** — run `python3 tools/gigachat4_layer_probe.py` on a two-layer prefix of the checkpoint and compare per-stage cosines against the engine's `LSX_GC4_DUMP_LAYERS` dump, exactly as the 37B `G1` item documents.

- [ ] **Step 5: Commit the goldens**

```bash
git add tests/fixtures/fsi/gigachat4-708b-a41b.json tests/fixtures/code/gigachat4-708b-a41b.json \
        tests/fixtures/det/gigachat4-708b-a41b.json
git commit -m "test: GigaChat4-708B-A41B parity goldens (template-derived)"
```

---

### Task 10: Perf row, measured

**Files:** `tests/test_perf_baseline.cpp` (model table + `TEST_CASE`), `tests/perf/perf_baselines.json` (written by the harness)

- [ ] **Step 1: Add the table row and the test case**

`{"gigachat4-708b-a41b", "GigaChat4-708B-A41B", "The capital of France is", 2048}`
and `TEST_CASE("perf baseline: gigachat4-708b-a41b", "[perf]")` following the
37B case.

- [ ] **Step 2: Run it on the release tree** — `ctest --test-dir build/linux-release-znver4-aocc -R "test_perf_baseline_gigachat4_708b" -j 1 --output-on-failure`. Expected: the canonical 2048-token run completes inside the 5400 s timeout (~12–20 min at the projected 2–3 t/s) and writes the row.

- [ ] **Step 3: Commit** — `git commit -m "test: seed the GigaChat4-708B-A41B perf row from the release run"`.

---

### Task 11: Report and bookkeeping

**Files:** `docs/model-reports/2026-11-27-GigaChat4-708B-A41B.md`, `docs/model-reports/README.md`, `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md`, `docs/open-tasks/live/models/gigachat4-708b-a41b.md`

- [ ] **Step 1: Write the report** with a publish-ready English post (no repo paths/codenames) and an Evidence section; every number must match the gate logs and the perf row; SKIPs and reds named as gaps.
- [ ] **Step 2: Add the matrix row, the open-task row and the per-model file.**
- [ ] **Step 3: Re-run `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .` and `ctest --preset linux-debug -j 24`; both clean.**
- [ ] **Step 4: Commit** — `git commit -m "docs: GigaChat4-708B-A41B ingestion report and open items"`.

---

## Self-review notes (spec → plan)

- Spec §Identity → Tasks 2, 4, 7 (passport, expect, manifest).
- Spec §Converter memory → Task 1 Step 4 + Task 3 the flush assertion.
- Spec §Feasibility / VRAM → Task 7 Step 6 (context-limit case) and Task 10
  (measured perf, no hand-written row).
- Spec §Non-goals (no source-side full-model goldens) → Task 9 Step 1 records the
  deviation in the golden's `source`, Step 4 gives the per-layer localisation.
- Spec §"review focus" items each own a test: 37B constant leak (Task 2 Step 1,
  Task 3 vram case), no per-layer flush (Task 3), wrong format tag (Task 4), the
  session cap (Task 5 Step 4), SKIP-as-pass (Task 11 Step 1).
