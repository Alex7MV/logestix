# Kimi-K3.1-Pro ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native; `AGENTS.md` forbids subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest `Kimi-K3.1-Pro` as a sibling variant of the `kimi` family (`kimi-k3.1-pro`) whose 24 full-attention layers carry a Gated-DSA Lightning Indexer, with the indexer converted, pinned, scored and applied above `index_topk = 2048` and byte-identical to Kimi-K3's dense path at or below it.

**Architecture:** The variant gets its own frozen passport (`kimi_moe_pro_constants.h`, ns `lsxcommon::kimi_pro::frozen`), its own converter entry (`--kimi_k31_pro`), and its own `IModel` (`kimi_moe_pro_model.{h,cpp}`). The K3 converter/loader/kernels/pipeline stay shared and learn one optional component: five BF16 indexer tensors per full-attention layer, an indexer-key cache, and an optional selected-position list on the MLA attention kernel.

**Tech Stack:** C++20 host code (absl `StrFormat`, spdlog via `lsxcommon::log`), CUDA 13.4 device code (`sm_120/121`), Arrow IPC weights, Catch2 v3 tests, CMake presets (`linux-debug`).

**Spec:** `docs/superpowers/specs/2026-11-05-kimi-k31-pro-ingestion-design.md`

## Global Constraints

- TDD: every production change lands with the test that pins it, in the same commit; every new test gets its ctest label in the same commit.
- Frozen geometry: no runtime `config.json` read at inference; the loaded `model.index.json` is cross-checked against the passport and a mismatch fails `Load`.
- Norms, routers, gates and **indexer weights** are never quantized (BF16/FP16/F32).
- Kernels: one stream, no per-token `cudaDeviceSynchronize`, NVFP4 only through `lsx_inference_nvfp4`, shared-memory sandbox (thread-0 RMW behind barriers), no `atomicAdd` in anything feeding `final_logits`, determinism must not depend on scheduling.
- No abseil containers in `.cu`/`.cuh`-reachable headers.
- Naming: `^<family>_<type>(_<variant>)?_<role>(_<subsystem>)?$`, family `kimi`, type `moe`, variant `pro` (already in `SchemeRegex()`), roles `constants|convert|load|model|inference|kernels`.
- Host logging only through `lsxcommon::log`; `absl::StrFormat` is printf-style (`%s/%d/%llu`), never `{}`.
- Commit prefixes `feat:`/`test:`/`fix:`/`docs:`/`build:`; commit only on explicit request — this plan's commits are authorised by the operator's "делай в отдельной ветке" (branch `feat/kimi-k31-pro`).
- Real-model weights live in the vault (`/mnt/fast_storage/huggingface`), converted in place under `<source>/logestix/`; temporary files go to `/mnt/fast_storage/tmp`.
- One GPU workload at a time; `lsxbenchmark` runs with strict GPU exclusivity and perf last.

## Review Focus

- **`n_pos` exactly `2048` and `2049`** — the dormancy boundary must be exact: at `2048` the dense path runs (byte-identical to K3), at `2049` the indexer must select 2048 of 2049 keys with the *newest* key never dropped by a tie-break.
- **Ties in the top-k** — equal scores must resolve by ascending position, and a repeated run of the same session must produce the same list.
- **A K3 tree loaded by the Pro model, and vice versa** — both must fail loudly at `Load`/convert, never decode.
- **`max_context` above the indexer key cache** — the cache is sized from `max_context`; a `prompt + max_new` that overflows must be a clean error (existing behaviour) and must not read past the cache.
- **Indexer absent on a KDA layer / present twice** — conversion must error, not silently drop.

---

## File Structure

| file | responsibility |
|---|---|
| `libs/lsxcommon/src/kimi_moe_pro_constants.h` | frozen passport for the variant (geometry restated, DSA table, indexer shapes, ids) |
| `libs/lsxcommon/include/lsxcommon/kimi_moe_pro_convert.h` | `ConvertKimiK31ProModel(...)` declaration |
| `libs/lsxcommon/src/kimi_moe_pro_convert.cpp` | thin pro entry (variant discriminator + delegation) |
| `libs/lsxcommon/include/lsxcommon/kimi_moe_pro_model.h` | `KimiK31ProModel : IModel` |
| `libs/lsxcommon/src/kimi_moe_pro_model.cpp` | registry, `Load`/`Infer`/`SelfCheck`, framing reuse, lifecycle |
| `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_convert.h` | +`KimiVariant`, +`ConvertKimiK3Variant`, +indexer kinds/topology fields |
| `libs/lsxcommon/src/kimi_moe_k3_convert.cpp` | +indexer classification, raw-BF16 write, index `dsa_layers`, variant gates |
| `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_load.h` | +`dsa`/indexer fields on topology and per-layer weights |
| `libs/lsxcommon/src/kimi_moe_k3_load.cpp` | +topology parse, +indexer upload, +frozen cross-check |
| `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_kernels.h` | +indexer key/query/score/top-k, +optional `sel` on MLA attention |
| `libs/lsxcommon/src/kimi_moe_k3_kernels.cu` | the same, device side |
| `libs/lsxcommon/src/kimi_moe_k3_inference_pipeline.cu` | +indexer key cache, +DSA branch, +selection list |
| `apps/lsxforge/src/{cli.h,cli.cpp,main.cpp}` | `--kimi_k31_pro` |
| `tests/test_kimi_pro_*.cpp` / `.cu` | the gates below |

---

### Task 1: Frozen passport for the variant

**Files:**
- Create: `libs/lsxcommon/src/kimi_moe_pro_constants.h`
- Test: `tests/test_kimi_pro_constants.cpp` (built INTO the existing `test_kimi_k3_constants` target — the sibling's constants stage is the family constants gate, so the `family test target is claimed by a suite row` guard stays green while the pro `Suites()` row cannot exist yet; see Step 5)
- Modify: `tests/CMakeLists.txt` (second source on `test_kimi_k3_constants`, label `fast` unchanged)
- Modify: `tests/test_model_independence_guard.cpp` (kPassports row — **deferred to Task 6**)
- Modify: `tests/test_model_suite_guard.cpp` (Suites() + tool-verdict rows — **deferred to Task 7**)

**Interfaces:**
- Produces: `namespace lsxcommon::kimi_pro::frozen` with `kModelType = "kimi-k3.1-pro"`, `kFormatTag = "logestix-kimi-k31-pro-nvfp4"`, all K3 geometry constants restated under the same names (`kHidden`, `kLayers`, `kMlaLayers`, `kExperts`, `kTopK`, `kLatentDim`, `kSharedIntermediate`, `kSituBeta`, `kSituLinearBeta`, `kVocab`, `kBosId`, `kEosId`, `kStopIdAlt`, `kPadId`, `kTokenizerKind = 2`, `kRmsNormEps`), plus `kDsaLayerCount = 24`, `kIndexHeads = 32`, `kIndexHeadDim = 128`, `kIndexTopK = 2048`, `kIndexRopeDim = 0`, `kIndexQLora = 1536`, `kIndexKvLen = 7168`, `constexpr bool IsDsaLayer(int)`, and `IsMlaLayer(int)`.

- [x] **Step 1: Write the failing test**

`tests/test_kimi_pro_constants.cpp` — one `TEST_CASE("kimi-k3.1-pro passport: identity", "[kimi-pro][constants]")` asserting `kModelType == "kimi-k3.1-pro"`, `kFormatTag == "logestix-kimi-k31-pro-nvfp4"`; one `TEST_CASE("kimi-k3.1-pro passport: DSA table equals the MLA table", ...)` asserting `kDsaLayerCount == kMlaLayerCount` and `IsDsaLayer(l) == IsMlaLayer(l)` for every `l` in `[0, kLayers)`; one `TEST_CASE("kimi-k3.1-pro passport: indexer geometry", ...)` asserting `kIndexHeads * kIndexHeadDim == 4096`, `kIndexRopeDim == 0`, `kIndexTopK == 2048`, `kIndexQLora == kQLoraRank`, `kIndexKvLen == kHidden`; one `TEST_CASE("kimi-k3.1-pro passport: geometry matches Kimi-K3", ...)` asserting `kHidden == 7168 && kLayers == 93 && kExperts == 896 && kTopK == 16 && kLatentDim == 3584 && kMoeIntermediate == 3072 && kVocab == 163840`.

- [x] **Step 2: Run test to verify it fails** — `cmake --build --preset linux-debug --target test_kimi_k3_constants` → FAIL: `kimi_moe_pro_constants.h: No such file or directory` (observed).

- [x] **Step 3: Write `kimi_moe_pro_constants.h`** — self-contained (it must NOT include `kimi_moe_k3_constants.h`; the independence guard forbids passport→passport includes), with the same `static_assert` invariant block as K3 plus `static_assert(kIndexTopK == 2048, ...)` and `static_assert(kDsaLayerCount == kMlaLayerCount, ...)`.

- [x] **Step 4: Run test to verify it passes** — 615 assertions / 12 cases green in `test_kimi_k3_constants` (pro file contributes 352 assertions / 5 cases); full `ctest --preset linux-debug` 134/134.

- [x] **Step 5: Extend the independence guard — DEFERRED to Task 6.** The guard cross-checks `kPassports` against `kLifecycle` ("no kLifecycle registry row"), and a `kLifecycle` row needs a readable model TU. Adding the passport row here would leave the `fast` suite red for four commits, so both rows (`kPassports` + `kLifecycle`) land together with `kimi_moe_pro_model.cpp` in Task 6. Executed 2026-11-05: row added, guard failed on the missing lifecycle row, row reverted; deviation recorded here.

- [x] **Step 6: Commit** — `feat(kimi-k31-pro): frozen passport for the DSA variant`.

---

### Task 2: Converter — indexer tensors, variant gate, index fields

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_convert.h` (kinds, `KimiVariant`, topology fields, `ConvertKimiK3Variant`)
- Modify: `libs/lsxcommon/src/kimi_moe_k3_convert.cpp`
- Create: `libs/lsxcommon/include/lsxcommon/kimi_moe_pro_convert.h`, `libs/lsxcommon/src/kimi_moe_pro_convert.cpp`
- Modify: `apps/lsxforge/src/cli.h`, `apps/lsxforge/src/cli.cpp`, `apps/lsxforge/src/main.cpp`, `apps/lsxforge/tests/test_lsxforge_cli.cpp`
- Test: `tests/test_kimi_pro_convert.cpp` (label `convert`, `TIMEOUT 120`, `ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1"`)

**Interfaces:**
- Consumes: Task 1's passport constants.
- Produces:
  - `enum class KimiK3TensorKind { ..., IndexerWqB, IndexerWk, IndexerKNormW, IndexerKNormB, IndexerWeightsProj }`
  - `enum class KimiVariant { k3, k31_pro };`
  - `arrow::Status ConvertKimiK3Variant(const std::string &input_dir, const std::string &output_root, KimiVariant variant);` (the existing `ConvertKimiK3Model` becomes a `k3` delegate)
  - `KimiK3ConvertTopology` gains `std::string model_type; std::vector<int> dsa_layers; int index_heads = 0, index_head_dim = 0, index_topk = 0, index_rope_dim = 0;`
  - `lsxcommon::kimi_pro::ConvertKimiK31ProModel(const std::string &input_dir, const std::string &output_root)` → `ConvertKimiK3Variant(..., KimiVariant::k31_pro)`
  - `--kimi_k31_pro` flag mapped to `Pipeline::kKimiK31Pro`.

- [ ] **Step 1: Write the failing test** — `tests/test_kimi_pro_convert.cpp` builds a synthetic checkpoint with `synthtest::WriteSafetensors` (K3 fixture shape, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`) whose MLA layers also carry the five `self_attn.indexer.*` tensors, then asserts: (a) `model.index.json` has `"format": "logestix-kimi-k31-pro-nvfp4"`, `"model": "kimi-k3.1-pro"`, `"dsa_layers"` listing exactly the synthetic MLA layers; (b) each indexer tensor exists in `pinned/` with `dtype == "BF16"` and its original shape (not quantized); (c) the same fixture **without** indexer tensors makes `ConvertKimiK31ProModel` return an invalid status; (d) an indexer tensor on a KDA layer makes it fail; (e) a fixture whose `model_type` is `kimi_k3` fails with the variant message.

- [ ] **Step 2: Run to verify it fails** — target links, every assertion fails.

- [ ] **Step 3: Implement** — add the five kinds and the classifier suffixes `self_attn.indexer.wq_b.weight`, `.wk.weight`, `.k_norm.weight`, `.k_norm.bias`, `.weights_proj.weight`; extend `ParseKimiK3Config` with `text_config.model_type` + `dsa_config` (`dsa_layers` 1-based → 0-based, `index_topk`, `index_head_dim`, `index_num_heads`, `index_rope_dim`); route indexer kinds through the existing `pinned_raw` (`TensorRecord`) path so the bytes stay BF16; in `ConvertKimiK3Variant` gate on `variant`: `k31_pro` requires `model_type == "kimi_k31"`, `dsa_layers == mla_layers`, and exactly 5 indexer tensors on every DSA layer and none elsewhere; `k3` rejects any `Indexer*` kind. Write `"model"`, `"dsa_layers"`, `"index_head_dim"`, `"index_num_heads"`, `"index_topk"`, `"index_rope_dim"`, `"indexer": true/false` into `model.index.json`. Add the `--kimi_k31_pro` flag end to end (cli enum `kKimiK31Pro`, `main.cpp` dispatch, cli test table).

- [ ] **Step 4: Run to verify it passes** — `ctest -R test_kimi_pro_convert --preset linux-debug-convert` and `test_lsxforge_cli`.

- [ ] **Step 5: Commit** — `feat(kimi-k31-pro): converter keeps the DSA indexer, variant-gated`.

---

### Task 3: Loader — topology, indexer weights, frozen cross-check

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_load.h`
- Modify: `libs/lsxcommon/src/kimi_moe_k3_load.cpp`
- Test: `tests/test_kimi_pro_loader.cpp` (label `fast`)

**Interfaces:**
- Produces: `KimiK3Topology` gains `bool dsa = false; std::vector<int> dsa_layers; int index_heads = 0, index_head_dim = 0, index_topk = 0, index_rope_dim = 0;` and `bool IsDsa(int layer) const`. `KimiK3LayerWeights` gains `__half *idx_wqb = nullptr, *idx_wk = nullptr, *idx_knorm_w = nullptr, *idx_knorm_b = nullptr, *idx_wproj = nullptr;`. `KimiK3TopologyMatchesFrozen` keeps its signature; the pro model passes its own expected values through a new `bool KimiK3TopologyMatchesFrozen(const KimiK3Topology&, bool expect_dsa, int expect_topk)` overload (the existing one-argument form delegates with `expect_dsa = false`, preserving K3).

- [ ] **Step 1: Write the failing test** — a synthetic converted tree (the Task 2 fixture with indexers) loaded through `LoadKimiK3Topology` reports `dsa == true`, `dsa_layers == mla_layers`, `index_topk == 2048`, `index_heads == 32`, `index_head_dim == 128`; `KimiK3TopologyMatchesFrozen(top, true, 2048)` is true and `(top, false, 2048)` is false; a K3 tree (no indexers) reports `dsa == false` and has null `idx_wqb` after `LoadKimiK3PinnedWeights`; the Pro tree uploads `idx_wqb` with `rows == 4096, cols == 1536` and `idx_wk` with `rows == 128, cols == 7168`; `FreeKimiK3PinnedWeights` leaves every indexer pointer null (checked twice to catch a double free).

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — parse the new index fields (`decoded` in `LoadKimiK3Topology`), tolerate their absence (`dsa = false`) so a K3 tree still loads; in the pinned-weight pass, look up the five tensors per DSA layer and upload through the existing BF16→FP16 raw helper; `Free` releases them; extend the resident-byte accounting by exactly the bytes allocated (counted at the allocation site).

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `feat(kimi-k31-pro): loader pins the indexer weights and cross-checks the variant`.

---

### Task 4: Kernels — indexer key/query/score, exact top-k, optional selection

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_kernels.h`
- Modify: `libs/lsxcommon/src/kimi_moe_k3_kernels.cu`
- Test: `tests/test_kimi_pro_indexer.cu` via `lsx_add_cuda_test`, label `inference`

**Interfaces:**
- Produces (exact signatures):
```cpp
cudaError_t RunKimiK3IndexerKey(cudaStream_t, const __half *x,
                                const __half *wk, const __half *knorm_w,
                                const __half *knorm_b, __half *k_out, int kv_len,
                                int index_dim, float eps);
cudaError_t RunKimiK3IndexerQuery(cudaStream_t, const __half *qa,
                                  const __half *x, const __half *wq_b,
                                  const __half *wproj, __half *q_out,
                                  __half *w_out, int q_lora, int kv_len,
                                  int index_heads, int index_dim,
                                  float q_scale, float w_scale);
cudaError_t RunKimiK3IndexerScores(cudaStream_t, const __half *q,
                                   const __half *w, const __half *k_cache,
                                   int n_pos, int index_heads, int index_dim,
                                   float *score);
cudaError_t RunKimiK3IndexerTopK(cudaStream_t, const float *score, int n_pos,
                                 int k, int32_t *out_idx, int *out_n);
cudaError_t RunKimiK3MlaAttention(cudaStream_t, const __half *latent, int n_pos,
                                  const __half *q_abs, const __half *q_rot,
                                  const uint8_t *wv_packed,
                                  const uint8_t *wv_scales, float wv_gs,
                                  __half *out, int heads, int latent_dim,
                                  int rope, int v_head, int rows_per_head,
                                  int row_offset, int q_rot_stride, float scale,
                                  int wv_row_stride_packed,
                                  int wv_row_stride_scales,
                                  const int32_t *sel, int n_sel);  // new tail
```
Calling `RunKimiK3MlaAttention` with `sel == nullptr, n_sel == 0` reproduces today's behaviour exactly.

- [ ] **Step 1: Write the failing test** — in `tests/test_kimi_pro_indexer.cu`:
  1. `TEST_CASE("kimi-k3.1-pro indexer: key LayerNorm matches the host reference")` — 4 positions of 128 dims, `wk` a deterministic pseudo-random BF16 matrix, assert `max |dev - host| <= 1e-2` after FP16 rounding.
  2. `TEST_CASE("kimi-k3.1-pro indexer: query and head weights match the host reference")` — asserts the `128^-0.5` and `32^-0.5` scales are applied (`CHECK(rel_err < 2e-2)` against a host double reference with the same scales).
  3. `TEST_CASE("kimi-k3.1-pro indexer: scores are the head-weighted relu sum")` — compares against a host double reference; also asserts a negative pre-activation contributes exactly `0`.
  4. `TEST_CASE("kimi-k3.1-pro indexer: top-k is exact and deterministic")` — `n_pos = 3000`, `k = 2048`, a score vector with deliberate exact ties at the boundary: the selected set equals the host's `std::partial_sort` set, ties resolve by ascending position, and two consecutive runs are identical.
  5. `TEST_CASE("kimi-k3.1-pro indexer: identity selection is bit-identical to dense attention")` — `sel = {0..n-1}`, `n = 37`: the output equals the `sel == nullptr` output bit for bit.
  6. `TEST_CASE("kimi-k3.1-pro indexer: masked attention matches the host reference")` — an arbitrary ascending subset of positions, compared against a host attention over the same subset.

- [ ] **Step 2: Run to verify it fails** (unresolved symbols).

- [ ] **Step 3: Implement** — key: one block of 128 threads, `dot → mean/var → normalise → *w + b`; query: two GEMVs through the existing dequant-dot helpers; scores: one block per position, `kBlock` threads reducing `w[h]*max(0, q[h]·k[s])` in FP32 with `block_sum`, no atomics; top-k: monotone-uint32 radix select (4 × 8-bit histogram passes over one block, then emit `> threshold` in ascending order and fill the remainder from `== threshold` in ascending order); attention: `t = sel ? sel[i] : i` inside the existing position loop.

- [ ] **Step 4: Run to verify it passes** — `ctest -R test_kimi_pro_indexer --preset linux-debug-inference`.

- [ ] **Step 5: Commit** — `feat(kimi-k31-pro): Lightning-Indexer kernels and selectable MLA attention`.

---

### Task 5: Pipeline — indexer key cache and the DSA branch

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_k3_inference_pipeline.cu`
- Test: covered by Task 6's model test executed on a synthetic tree with `index_topk = 4` (below).

**Interfaces:**
- Consumes: Task 4's kernels, Task 3's weights.
- Produces: no API change (`RunKimiK3Generation` keeps its signature).

- [ ] **Step 1: Write the failing test** — extend `tests/test_kimi_pro_model.cpp` (Task 6) with `TEST_CASE("kimi-k3.1-pro pipeline: the DSA branch engages above the top-k")`: a synthetic tree with `index_topk = 4` and a prompt of 8 tokens must (a) run `Infer` successfully, (b) produce, for the same weights, exactly the same logits as an otherwise identical tree with `index_topk = 8` (all keys selected — the identity), because with 4 selected keys the *most recent* keys dominate; assert equality only for the identity direction: `index_topk >= n_pos` must equal the dense reference bit for bit. The kernel-level >topk equivalence is Task 4's test.

- [ ] **Step 2: Run to verify it fails** (no DSA branch → different code path missing).

- [ ] **Step 3: Implement** — allocate `index_k` (`dsa_layers.size() * max_context * index_dim` FP16) and `index_sel` (`index_topk` int32) + `index_score` (`max_context` FP32) in `Workspace`; in the MLA block, when `lw.idx_wk != nullptr`: compute the current position's indexer key into `index_k[slot][pos]`, then if `n_pos > top.index_topk` compute query/weights/scores/top-k and pass `(sel, n)` to the attention, else pass `(nullptr, 0)`. Nothing else changes on the dense path.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `feat(kimi-k31-pro): pipeline indexer cache and dormant-at-tp<=topk DSA branch`.

---

### Task 6: Model, registry, tool bridge, lifecycle

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/kimi_moe_pro_model.h`, `libs/lsxcommon/src/kimi_moe_pro_model.cpp`
- Modify: `libs/lsxcommon/src/model_registry.cpp` (register `kimi-k3.1-pro`)
- Modify: `tests/test_model_independence_guard.cpp` (`consumers`, `kLifecycle`)
- Test: `tests/test_kimi_pro_model.cpp` (label `fast`)

**Interfaces:**
- Produces: `class KimiK31ProModel final : public ModelBase` with `static const std::string kType;` (`"kimi-k3.1-pro"`), `Load`, `Infer`, `SelfCheck`, `Unload`, `Inspect`, `BuildConversationInputIds`, `BuildPromptInputIds`, `ParseToolCalls`, `TeardownWeights`; framing delegates to `kimi_moe_k3_tool_format.h` (XTML, identical tokenizer).

- [ ] **Step 1: Write the failing test** — `tests/test_kimi_pro_model.cpp`: `TEST_CASE("kimi-k3.1-pro model: registry")` asserts `LsxModelFactory::Create("kimi-k3.1-pro")` is non-null and `KimiK31ProModel::kType == "kimi-k3.1-pro"`; `TEST_CASE("kimi-k3.1-pro model: a Kimi-K3 tree is rejected")` loads the K3 synthetic tree and asserts `Load()` is false with the variant message; `TEST_CASE("kimi-k3.1-pro model: frozen mismatch fails Load")` flips `index_topk` in the index and asserts failure; `TEST_CASE("kimi-k3.1-pro model: lifecycle")` asserts `Infer` on an unloaded model returns a clean error, and (synthetic tree) a 2-token `Infer` returns `ok`, `weight_resident_bytes > 0`, `output_text` decoded, `token_callback` invoked once per token, and a pre-set `abort_flag` stops after zero or one token; `TEST_CASE("kimi-k3.1-pro pipeline: the DSA branch engages above the top-k")` from Task 5.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — copy the K3 model shape: `LsxFastMathScope(EffectiveFastMathMode(cfg_))` in `Infer`, `RunUnderResidencyLock` around generation, `GenerationStream` wired from `req.token_callback`/`req.abort_flag`, `AcquirePermanentWeights`, `FinishLoad`, `TeardownContext` in `TeardownWeights`, env knobs `LSX_KIMI_K31_PRO_MAX_OPEN` / `_MAX_CONTEXT`; register the type in `model_registry.cpp`; add the `kPassports` row here (see Task 1 Step 5) together with the `kLifecycle` row.

- [ ] **Step 4: Run to verify it passes** — plus `ctest -R "test_model_independence_guard|test_code_naming_guard"`.

- [ ] **Step 5: Commit** — `feat(kimi-k31-pro): IModel, registry, XTML bridge and the lifecycle contract`.

---

### Task 7: SelfCheck, parity goldens, perf baseline

**Files:**
- Create: `tests/test_kimi_pro_selfcheck.cpp`, `tests/test_kimi_pro_{fsi,code,det}_selfcheck.cpp`
- Create: `tests/fixtures/{fsi,code,det}/kimi-k3.1-pro.json`
- Modify: `tools/kimi_k3_template_goldens.py` (a `--model kimi-k3.1-pro` key so the goldens are keyed to the variant)
- Modify: `tests/test_perf_baseline.cpp`, `tests/perf/perf_baselines.json`, `tests/CMakeLists.txt`
- Modify: `tests/test_model_suite_guard.cpp` (`Suites()` row, tool verdict, golden case)

**Interfaces:**
- Produces: ctest targets `test_kimi_pro_selfcheck`, `test_kimi_pro_fsi_selfcheck`, `test_kimi_pro_code_selfcheck`, `test_kimi_pro_det_selfcheck`, `test_perf_baseline_kimi_k31_pro` — all `inference`/`perf` with `SKIP_RETURN_CODE 4`; `test_kimi_pro_selfcheck` runs ≥3 cycles, overridable by `LSX_TEST_KIMI_PRO_SELFCHECK_CYCLES`.

- [ ] **Step 1: Write the failing tests** — the four targets mirroring `test_kimi_k3_*_selfcheck.cpp` with `kModelType = "kimi-k3.1-pro"`; the perf test is `TEST_CASE("perf baseline: kimi-k3.1-pro")` reading the `perf_baselines.json` entry.
- [ ] **Step 2: Run to verify they fail** (missing goldens/targets).
- [ ] **Step 3: Implement** — generate the three template goldens, add the suite row (declaring that the tokenizer-parity and tool-framing stages are shared with `kimi-k3`), add the perf `add_test`.
- [ ] **Step 4: Run to verify** — `ctest -R "test_kimi_pro_.*_selfcheck" --preset linux-debug-inference` (SKIP 4 without the vault is a pass; the vault run is Task 9).
- [ ] **Step 5: Commit** — `test(kimi-k31-pro): selfcheck, template parity goldens and the perf row wiring`.

---

### Task 8: Docs and guard hygiene

**Files:** `AGENTS.md` (Structure), `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md` §R16, `docs/model-reports/README.md`.

- [ ] **Step 1** Write the AGENTS structure entry, the matrix row (all gates `not run` until Task 9), §R16 with the inferred-indexer debt, and the report index row.
- [ ] **Step 2** `ctest --preset linux-debug` green (fast suite).
- [ ] **Step 3 Commit** — `docs(kimi-k31-pro): structure, matrix row and the inferred-indexer debt`.

---

### Task 9: Real weights (vault)

- [ ] **Step 1** Convert: `lsxforge --kimi_k31_pro --input /mnt/fast_storage/huggingface/Kimi-K3.1-Pro --output ...` in the background (1.56 TB, ~2 h); verify the index (`dsa_layers` = 24 layers, `total_experts == 82 432`, `total_pinned_tensors` includes 120 indexer tensors).
- [ ] **Step 2** `lsxbenchmark --selfcheck --model /mnt/fast_storage/huggingface/Kimi-K3.1-Pro --model-type kimi-k3.1-pro` → Paris ×3 bit-exact, record `logits_hash`.
- [ ] **Step 3** Tool round-trip `get_weather {"city":"Paris"}` through `lsxbenchmark --selfcheck` + `ParseToolCalls`.
- [ ] **Step 4** Run FSI/CODE/DET time-boxed; record measured outcomes (a non-finishing run is debt, never a pass).
- [ ] **Step 5** Seed perf rows through `lsxbenchmark --bench` at 2048 tokens (`perf` + `r3`) or record why not.
- [ ] **Step 6** Update the matrix + `perf_baselines.json` (harness-written) + write `docs/model-reports/2026-11-05-Kimi-K3.1-Pro.md`.
- [ ] **Step 7 Commit** — `docs(kimi-k31-pro): real-weight gates and the model report`.
