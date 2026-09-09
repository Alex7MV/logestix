# Gemma NVFP4 Decode Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise gemma-4-26B-A4B-it decode throughput from ~38.7 t/s to ≥70 t/s by moving dense MLP, attention projections, and tied embed/lm_head to NVFP4, and removing the per-layer router handshake.

**Architecture:** Three independent phases, each a separate commit. Phase 1 quantizes dense MLP to NVFP4 (converter writes packed payloads into the existing `logestix-gemma` pinned schema; loader detects them; `ForwardLayer` runs a GEMV chain). Phase 2 does the same for attention q/k/v/o and the tied embed/lm_head table. Phase 3 removes the host router handshake with a device-composed slot table.

**Tech Stack:** CUDA 13.3 (sm_120), C++20, Arrow IPC, Catch2 v3, existing `nvfp4_gemv_math.cuh` / `lsx_inference_nvfp4.h` NVFP4 machinery.

## Global Constraints

- Format tag stays `logestix-gemma`; existing BF16 conversions must keep loading (loader branches, converter writes both variants' schema-compatible files).
- Norms, `layer_scalar`, router weights/scales, q/k_norm remain BF16/F32 (precision invariant).
- All reduces keep fixed order (ascending-k, ascending chunk); selfcheck must stay bit-exact across 3 cycles.
- No new parallel GEMM dispatcher; use `RunLsxNvfp4Gemm` / gemma routed GEMV chain (`nvfp4_gemv_math.cuh` contract).
- Build: `cmake --build build/linux-release-znver4-aocc -j 160 --target <target>`; real model at `/mnt/llm_models/models/gemma-4-26B-A4B-it`.
- Synthetic gemma tests need `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`.
- Environment rollback flags: `LSX_GEMMA_MLP_BF16=1`, `LSX_GEMMA_ATTN_BF16=1`, `LSX_GEMMA_HEAD_BF16=1` must restore the previous path.
- Perf gate per phase: `--testprompt --max-gen 64` t/s must not regress; final ≥70 t/s.

---

### Task 1: Converter — dense MLP NVFP4 records in pinned schema

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_convert.cpp` (MLP branch of `ConvertGemmaModel`, tensor classification near `classify_gemma_tensor`)
- Test: `tests/test_gemma_moe_convert.cpp`

**Interfaces:**
- Consumes: `gpu::quantize_tensor_gpu(raw, numel, "BF16", nullptr, shape, {}, packed, scales, gs)` (`libs/lsxcommon/include/lsxcommon/safetensors_gpu.h:17`); `PinnedSchema::MakeBatch` (4-col: name/shape/dtype/data).
- Produces: pinned records named `mlp.gate_proj.weight.packed`, `mlp.gate_proj.weight.scales`, `mlp.gate_proj.weight.global_scale` (same triple for `up_proj`, `down_proj`); `packed` is `large_binary`, `scales` is `large_binary`, `global_scale` is a 4-byte F32 `large_binary` (bit-copied). Written into the same `layer_N.arrow` batch. Index flag: `"mlp_nvfp4": true` written by `WriteGemmaModelIndex`.

- [ ] **Step 1: Write failing converter test**

In `tests/test_gemma_moe_convert.cpp` add (after existing includes, inside the file's namespace):

```cpp
TEST_CASE("gemma converter emits NVFP4 records for dense MLP",
          "[gemma][convert][nvfp4]") {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_convert_mlp_nvfp4";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string()).ok());

    auto batch_res = lsxcommon::LoadTensorBatchViews(
        out / "logestix" / "pinned" / "layer_0.arrow");
    REQUIRE(batch_res.ok());
    auto names = batch_res->names;  // std::vector<std::string>
    auto has = [&](const std::string &s) {
        return std::find(names.begin(), names.end(), s) != names.end();
    };
    REQUIRE(has("mlp.gate_proj.weight.packed"));
    REQUIRE(has("mlp.gate_proj.weight.scales"));
    REQUIRE(has("mlp.gate_proj.weight.global_scale"));
    // packed size = numel/2; scales = numel/16
    const auto &cfg = gemmatest::TestCfg();
    auto packed_rec = batch_res->by_name.at("mlp.gate_proj.weight.packed");
    CHECK(packed_rec.bytes == static_cast<size_t>(cfg.intermediate_size *
                                                 cfg.hidden_size / 2));
    auto scales_rec = batch_res->by_name.at("mlp.gate_proj.weight.scales");
    CHECK(scales_rec.bytes ==
          static_cast<size_t>(cfg.intermediate_size * cfg.hidden_size / 16));
    fs::remove_all(tmp);
}
```

Note: check the actual `LoadTensorBatchViews` return type in `libs/lsxcommon/include/lsxcommon/schema.h` first — adapt `names`/`by_name` access to whatever the real view struct exposes (it may need a second `PinnedSchema::UnpackBatch` call instead). The test asserts the three records exist with correct byte sizes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_convert && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_convert`
Expected: FAIL — `.packed` records absent.

- [ ] **Step 3: Implement converter change**

In `gemma_moe_convert.cpp`, where dense-MLP tensors are appended to `raw_by_layer` (locate via `classify_gemma_tensor` returning the MLP kinds), branch: when `mlp_nvfp4` is enabled (constant `constexpr bool kMlpNvfp4 = true;` for this converter build), replace the three raw `TensorRecord`s with six records:

```cpp
namespace {
struct Nvfp4Triple {
    std::vector<uint8_t> packed, scales;
    float global_scale = 1.0f;
};
bool QuantizeBf16ToNvfp4(const std::vector<uint8_t> &raw,
                         const std::vector<int64_t> &shape,
                         Nvfp4Triple &out) {
    return gpu::quantize_tensor_gpu(raw.data(),
                                    static_cast<int64_t>(raw.size()) / 2,
                                    "BF16", nullptr, shape, {},
                                    out.packed, out.scales,
                                    out.global_scale);
}
void AppendNvfp4Records(const std::string &base, const Nvfp4Triple &t,
                        const std::vector<int64_t> &shape,
                        std::vector<TensorRecord> &dst) {
    TensorRecord pk, sc, gs;
    pk.name = base + ".packed";  pk.dtype = "U8";   pk.shape = shape; pk.data = t.packed;
    sc.name = base + ".scales";  sc.dtype = "U8";   sc.shape = shape; sc.data = t.scales;
    gs.name = base + ".global_scale"; gs.dtype = "F32"; gs.shape = {1};
    gs.data.assign(reinterpret_cast<const uint8_t *>(&t.global_scale),
                   reinterpret_cast<const uint8_t *>(&t.global_scale) + 4);
    dst.push_back(std::move(pk));
    dst.push_back(std::move(sc));
    dst.push_back(std::move(gs));
}
} // namespace
```

Call `QuantizeBf16ToNvfp4` on the MLP raw data, then `AppendNvfp4Records("mlp.gate_proj.weight", ...)`, same for up/down. In `WriteGemmaModelIndex` add `"mlp_nvfp4": true` to the JSON body.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_convert && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_convert`
Expected: PASS (all pre-existing cases too).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gemma_moe_convert.cpp tests/test_gemma_moe_convert.cpp
git commit -m "feat: gemma converter emits NVFP4 dense-MLP records in pinned schema"
```

---

### Task 2: Loader — detect and upload NVFP4 dense MLP

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/gemma_moe_load.h` (`GemmaLayerWeights`)
- Modify: `libs/lsxcommon/src/gemma_moe_load.cpp` (`LoadLayer` + a new `LoadNvfp4LayerRecords`)
- Test: `tests/test_gemma_moe_loader.cpp`

**Interfaces:**
- Consumes: Task 1 records; `DenseNvfp4BatchViews`/`LoadDenseNvfp4BatchViews` (`libs/lsxcommon/include/lsxcommon/schema.h:52,869`); `LsxNvfp4Weight` (`libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h:29`); `fastload::UploadNvfp4Weight` (`libs/lsxcommon/src/fast_load_internal.h:244`) — requires the weight type to expose `rows/cols/global_scale/packed_bytes/scale_bytes/d_packed/d_scales/Free()`, which `LsxNvfp4Weight` does.
- Produces: `GemmaLayerWeights::mlp_gate/mlp_up/mlp_down` of type `LsxNvfp4Weight` (non-null `d_packed` when NVFP4 present), plus `bool mlp_nvfp4` on the layer. Loader keeps BF16 fields null in that case. `GemmaPinnedWeights` unchanged.

- [ ] **Step 1: Write failing loader test**

In `tests/test_gemma_moe_loader.cpp` add:

```cpp
TEST_CASE("gemma loader uploads NVFP4 dense MLP when present",
          "[gemma][loader][nvfp4]") {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_loader_mlp_nvfp4";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string()).ok());

    lsxcommon::gemma::GemmaPinnedWeights pw;
    auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw, nullptr);
    REQUIRE(rc == cudaSuccess);
    REQUIRE(!pw.layers.empty());
    const auto &lw = pw.layers[0];
    REQUIRE(lw.mlp_nvfp4);
    REQUIRE(lw.mlp_gate.d_packed != nullptr);
    CHECK(lw.mlp_gate.rows == gemmatest::TestCfg().intermediate_size);
    CHECK(lw.mlp_gate.cols == gemmatest::TestCfg().hidden_size);
    CHECK(lw.d_mlp_gate == nullptr);  // BF16 path not populated
    pw.Free();
    fs::remove_all(tmp);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_loader && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_loader`
Expected: FAIL — `mlp_nvfp4` member does not compile / is absent.

- [ ] **Step 3: Implement loader**

`gemma_moe_load.h`: add to `GemmaLayerWeights`:

```cpp
#include "lsxcommon/lsx_inference_nvfp4.h"
// ...
    // NVFP4 dense MLP (Task 1 records). Non-null d_packed when the
    // conversion provided them; BF16 d_mlp_* fields stay null then.
    lsxcommon::lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    bool mlp_nvfp4 = false;
```

(Note the header must stay CUDA-safe: `lsx_inference_nvfp4.h` is a model-neutral device header, safe in this TU — verify with `rg 'flat_hash' libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h` returning nothing.)

`gemma_moe_load.cpp` in `LoadLayer`: before building `jobs`, scan `tensors` for the six `.packed/.scales/.global_scale` records. Reconstruct F32 global scales via `std::memcpy`. For each of the three projections, assemble a temporary `DenseNvfp4View` (name/shape/packed/scales/global_scale) and call `fastload::UploadNvfp4Weight(view, lw.mlp_gate, stream, ring, bytes_acc, nullptr)`. If all three uploaded, set `lw.mlp_nvfp4 = true` and drop the three BF16 `make_job` entries; otherwise leave BF16 jobs intact (legacy files). Record the lookup helper:

```cpp
static const TensorRecord *FindTensorSuffix(const std::vector<TensorRecord> &ts,
                                            std::string_view suffix) {
    for (const auto &t : ts)
        if (t.name.size() > suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &t;
    return nullptr;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_loader && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_loader`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/gemma_moe_load.h libs/lsxcommon/src/gemma_moe_load.cpp tests/test_gemma_moe_loader.cpp
git commit -m "feat: gemma loader uploads NVFP4 dense MLP weights"
```

---

### Task 3: Inference — dense MLP GEMV chain (decode + prefill)

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_inference.cu` (`ForwardLayer` dense-MLP branch, lines ~1454-1478; new static kernel wrappers near the routed GEMV chain)
- Test: `tests/test_gemma_moe_moe.cpp`

**Interfaces:**
- Consumes: Task 2 `lw.mlp_gate/mlp_up/mlp_down` (`LsxNvfp4Weight`); `RunLsxNvfp4GemmSeq1GemmV(stream, x, packed, scales, gs, y, K, N, acc)` (`libs/lsxcommon/src/lsx_inference_nvfp4.cu:430`); existing `gemma_gemv_act_batch_kernel<GEGLU>` geGLU core.
- Produces: in `ForwardLayer`, when `lw.mlp_nvfp4` and `seq == 1` and no `LSX_GEMMA_MLP_BF16`, the dense-MLP branch computes into `ws.d_qr` exactly as before (post-MLP contents identical role: d1 = rms(mlp_out)). No signature changes.

- [ ] **Step 1: Write failing MoE/kernel test**

In `tests/test_gemma_moe_moe.cpp` add a parity test comparing the NVFP4 dense-MLP GEMV chain against a host reference on synthetic tensors (following the file's existing host-reference pattern):

```cpp
TEST_CASE("gemma dense MLP NVFP4 GEMV chain parity vs host reference",
          "[gemma][moe][nvfp4]") {
    const int H = 256, I = 128;   // 32-multiple shapes
    // quantize random BF16 gate/up/down on GPU via lsxcommon quantizer
    // (reuse the file's existing helpers for making weights; if none exist,
    // fill host vectors and call gpu::quantize_tensor_gpu through the
    // public header).
    std::vector<uint8_t> gate_raw(I * H * 2), up_raw(I * H * 2), down_raw(H * I * 2);
    FillPatternBf16(gate_raw); FillPatternBf16(up_raw); FillPatternBf16(down_raw);
    std::vector<uint8_t> gp, gsc, up, usc, dp, dsc;
    float ggs, ugs, dgs;
    std::vector<int64_t> gate_shape{I, H}, down_shape{H, I};
    REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
        gate_raw.data(), (int64_t)I * H, "BF16", nullptr, gate_shape, {},
        gp, gsc, ggs));
    REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
        up_raw.data(), (int64_t)I * H, "BF16", nullptr, gate_shape, {},
        up, usc, ugs));
    REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
        down_raw.data(), (int64_t)H * I, "BF16", nullptr, down_shape, {},
        dp, dsc, dgs));

    // device buffers, x = pattern
    // gate = GEMV(gp,gsc,ggs); up = GEMV(up,usc,ugs)
    // h[j] = gelu_tanh(g[j]) * u[j]   (FP16 rounding parity: compute in fp32, cast half)
    // y = GEMV(dp,dsc,dgs) over h
    // acc[j] = sum_c x_c * gate_row_j[c]  -- host reference with the SAME
    //   E2M1/E4M3 decode as nvfp4_codec.h (E4M3ScaleToFloat ignores man bits;
    //   e4==0 -> 0). Compare within 0.05 abs.
    REQUIRE(diff < 0.05f);
}
```

Implementation detail for the device side (shown here because it is the actual production code path being exercised before wiring into ForwardLayer):

```cpp
__half *xh, *gu, *hbuf, *y;
cudaMalloc(&xh, H * 2); cudaMalloc(&gu, (size_t)I * 2);
cudaMalloc(&hbuf, (size_t)I * 2); cudaMalloc(&y, (size_t)H * 2);
// bf16->half conversion of x via kernels::Bf16ToHalf or cudaMemcpy of __half
RunLsxNvfp4GemmSeq1GemmV(stream, xh, gp.data()... /* device copies */, ggs,
                         gu, H, I, /*acc=*/false);
// geGLU in-place: reuse gemma_gemv_act_batch_kernel<true> via a tiny host
// wrapper if exported, else run a 5-line elementwise kernel in the test
RunLsxNvfp4GemmSeq1GemmV(stream, hbuf, dp, dsc, dgs, y, I, H, false);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_moe && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_moe`
Expected: FAIL or compile error (wrapper missing).

- [ ] **Step 3: Wire ForwardLayer**

In `ForwardLayer`, replace the dense-MLP `Matmul` block with:

```cpp
    static const bool mlp_bf16 =
        std::getenv("LSX_GEMMA_MLP_BF16") != nullptr;
    if (lw.mlp_nvfp4 && !mlp_bf16) {
        // NVFP4 weight-streaming chain (decode + prefill share the core).
        e = RunGemmaDenseMlpNvfp4(stream, lw.mlp_gate, lw.mlp_up,
                                  lw.mlp_down, ws, seq, H, I);
    } else {
        // legacy cuBLAS BF16 branch (unchanged)
        ...
    }
```

`RunGemmaDenseMlpNvfp4` (new static function in the anonymous namespace of `gemma_moe_inference.cu`):

```cpp
cudaError_t RunGemmaDenseMlpNvfp4(cudaStream_t stream,
                                  const lsxcommon::lsx_nvfp4::LsxNvfp4Weight &gate,
                                  const lsxcommon::lsx_nvfp4::LsxNvfp4Weight &up,
                                  const lsxcommon::lsx_nvfp4::LsxNvfp4Weight &down,
                                  GenWorkspace &ws, int seq, int64_t H, int64_t I) {
    // x: ws.d_normed (BF16) -> ws.d_xh (FP16, EnsureWorkspace-added buffer
    // sized seq*H). Uses gemma_bf16_to_half_kernel with a proper grid.
    // gate/up: RunLsxNvfp4GemmSeq1GemmV per seq row (seq==1 fast path);
    // for seq>1 launch the same core row-by-row (prefill correctness first).
    // geGLU: gemma_gemv_act_batch_kernel<true> over I elements.
    // down: GEMV into FP16 out, then gemma_f32_to_bf16-style cast into ws.d_qr.
}
```

Workspace: extend `EnsureWorkspace` with `d_xh` (FP16 mirror of `d_normed`, `S*H*2` bytes) and `d_mlp_h` (FP16 `S*I_max*2`) — both scratch, grown with existing capacity checks.

- [ ] **Step 4: Run tests + real-model selfcheck**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_moe test_gemma_moe_model && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_moe && ./build/linux-release-znver4-aocc/apps/lsxfabric/lsxfabric --localmodel /mnt/llm_models/models/gemma-4-26B-A4B-it --selfcheck`
NOTE: the real model on disk is BF16-converted — `mlp_nvfp4` absent → legacy path; selfcheck must still pass (regression gate only). Re-convert the model: `./build/linux-release-znver4-aocc/apps/lsxforge/lsxforge --gemma --in /mnt/llm_models/models/gemma-4-26B-A4B-it --out /mnt/llm_models/models/gemma-4-26B-A4B-it` (check lsxforge's actual gemma flags with `--help` first) into a NEW directory `gemma-4-26B-A4B-it-nvfp4` to avoid clobbering, then point `LSX_TEST_GEMMA_MODEL_PATH` at it for the selfcheck test.
Expected: parity test PASS; selfcheck passed=yes.

- [ ] **Step 5: Measure + commit**

```bash
./build/linux-release-znver4-aocc/apps/lsxfabric/lsxfabric --localmodel <converted-dir> --testprompt --max-gen 64
# record t/s; expect small gain (~+8-15%)
git add libs/lsxcommon/src/gemma_moe_inference.cu tests/test_gemma_moe_moe.cpp
git commit -m "feat: gemma dense-MLP NVFP4 weight-streaming decode chain"
```

---

### Task 4: Converter/Loader — attention q/k/v/o NVFP4

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_convert.cpp` (extend Task 1 branch to `self_attn.*_proj`)
- Modify: `libs/lsxcommon/include/lsxcommon/gemma_moe_load.h`, `libs/lsxcommon/src/gemma_moe_load.cpp` (attn NVFP4 upload)
- Test: `tests/test_gemma_moe_loader.cpp`

**Interfaces:**
- Consumes: Task 1 `AppendNvfp4Records`/`QuantizeBf16ToNvfp4`; Task 2 loader NVFP4 detection.
- Produces: `GemmaLayerWeights::attn_q/attn_k/attn_v/attn_o` (`LsxNvfp4Weight`) + `bool attn_nvfp4`; `q_rows/kv_rows/o_cols` still set (from record shapes) so attention geometry code keeps working. Index flags: `"attn_nvfp4": true`.

- [ ] **Step 1: Extend the failing test** (add to Task 2's test case or a new one)

```cpp
TEST_CASE("gemma loader uploads NVFP4 attention projections",
          "[gemma][loader][nvfp4]") {
    // same fixture/conversion as the MLP test
    // after Load:
    const auto &lw = pw.layers[0];
    REQUIRE(lw.attn_nvfp4);
    CHECK(lw.attn_q.d_packed != nullptr);
    CHECK(lw.attn_q.rows == /* sliding: 16*256=4096, or fixture value */);
    CHECK(lw.attn_o.cols == lw.attn_q.rows);
    // and on the full-attention layer (last layer in fixture):
    const auto &fw = pw.layers.back();
    CHECK(fw.attn_q.rows == /* fixture global q rows */);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/linux-release-znver4-aocc -j 160 --target test_gemma_moe_loader && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_gemma_moe_loader`
Expected: FAIL.

- [ ] **Step 3: Implement**

Converter: in the same branch as Task 1, also quantize `self_attn.q_proj.weight`, `k_proj`, `v_proj` (skipped when absent — full-attention layers have no v_proj), `o_proj`. Loader: mirror Task 2's detection for the four projections; keep `q_rows/kv_rows/o_cols` assignment by reading the record shape before quantization records replaced the raw ones (record the shapes from the `.packed` record's shape list — same [rows, cols]).

- [ ] **Step 4: Run tests**

Run: same as Step 2. Expected: PASS; all prior cases green.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gemma_moe_convert.cpp libs/lsxcommon/include/lsxcommon/gemma_moe_load.h libs/lsxcommon/src/gemma_moe_load.cpp tests/test_gemma_moe_loader.cpp
git commit -m "feat: gemma attention projections in NVFP4 (converter + loader)"
```

---

### Task 5: Inference — attention projections via NVFP4 GEMV

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_inference.cu` (`ForwardLayer` steps 2/7 — q/k/v projections and o_proj)
- Test: `tests/test_gemma_moe_attn.cpp`

**Interfaces:**
- Consumes: Task 4 weights; Task 3's `d_xh` FP16 workspace + BF16→FP16 conversion; `RunLsxNvfp4GemmSeq1GemmV`.
- Produces: `ws.d_qr/ws.d_kr/ws.d_vr` contents identical role as the BF16 path (BF16 GEMV outputs were BF16; NVFP4 chain yields FP16 → cast to BF16, or keep the QK-norm kernels reading FP16 — decision: cast to BF16, keeping all downstream kernels unchanged). No API changes.

- [ ] **Step 1: Write failing test** — projection GEMV parity (reuse Task 3's pattern with q_proj shape [4096, 2816]-class from fixture).

```cpp
TEST_CASE("gemma attention projection NVFP4 GEMV parity",
          "[gemma][attn][nvfp4]") {
    // q_proj [q_rows, H]: quantize, GEMV, host reference (same decode as
    // Task 3), tolerance 0.05.
}
```

- [ ] **Step 2: Run to verify fail.** Build + run `test_gemma_moe_attn`. Expected: FAIL (test-only helper references the new wrapper).

- [ ] **Step 3: Wire ForwardLayer**

Replace step-2 `Matmul` calls when `lw.attn_nvfp4 && seq == 1 && !attn_bf16`:

```cpp
    static const bool attn_bf16 =
        std::getenv("LSX_GEMMA_ATTN_BF16") != nullptr;
    const bool attn_nvfp4 = lw.attn_nvfp4 && !attn_bf16;
    // q: RunLsxNvfp4GemmSeq1GemmV(stream, ws.d_xh, lw.attn_q..., qf16, q_rows, H)
    // cast FP16 -> BF16 into ws.d_qr via existing conversion kernel
    // k: same into ws.d_kr; v: tied-layers keep the k-copy shortcut
    // o: same after attention, output cast into ws.d_qr
```

Prefill (seq>1) keeps the cuBLAS BF16 branch by default (rows-tiled GEMV can follow later; correctness first). Cast kernels: add `gemma_f16_to_bf16_kernel` (mirror of `gemma_f32_to_bf16_kernel`).

- [ ] **Step 4: Run tests + selfcheck (real NVFP4-converted model from Task 3).** Expected: PASS.

- [ ] **Step 5: Measure + commit**

```bash
git add libs/lsxcommon/src/gemma_moe_inference.cu tests/test_gemma_moe_attn.cpp
git commit -m "feat: gemma attention projections through NVFP4 GEMV in decode"
```

---

### Task 6: Converter/Loader — tied embed/lm_head NVFP4 (global_nvfp4.arrow)

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_convert.cpp` (write `global_nvfp4.arrow` with `embed_tokens.weight`)
- Modify: `libs/lsxcommon/include/lsxcommon/gemma_moe_load.h` (`GemmaPinnedWeights::embed_nvfp4/head_nvfp4`)
- Modify: `libs/lsxcommon/src/gemma_moe_load.cpp` (upload)
- Test: `tests/test_gemma_moe_loader.cpp`

**Interfaces:**
- Consumes: `DenseNvfp4Schema`/`LoadDenseNvfp4BatchViews` (`schema.h:147,869`); `UploadNvfp4Weight`.
- Produces: `GemmaPinnedWeights::embed_nvfp4` + `head_nvfp4` (aliasing the same `LsxNvfp4Weight`, `view_only=true` semantics documented on the struct); `vocab_size/hidden` still set. `d_embed` null when NVFP4 head present. Index flag `"head_nvfp4": true`.

- [ ] **Step 1: Failing test**

```cpp
TEST_CASE("gemma loader uploads NVFP4 tied embed/lm_head",
          "[gemma][loader][nvfp4]") {
    // fixture + convert
    REQUIRE(pw.embed_nvfp4.d_packed != nullptr);
    CHECK(pw.embed_nvfp4.rows == gemmatest::TestCfg().vocab_size);
    CHECK(pw.embed_nvfp4.cols == gemmatest::TestCfg().hidden_size);
    CHECK(pw.d_embed == nullptr);  // BF16 table not uploaded
}
```

- [ ] **Step 2: Verify fail.** Build loader test. Expected: FAIL.

- [ ] **Step 3: Implement**

Converter: in the global-tensor loop, route `embed_tokens.weight` to a `DenseNvfp4Record` list; write `pinned/global_nvfp4.arrow` via `DenseNvfp4Schema::MakeBatch` (mirror qwen converter's `global_nvfp4` handling at `qwen_moe_convert.cpp:389,489`). Loader: `LoadDenseNvfp4BatchViews(global_nvfp4.arrow)` + `UploadNvfp4Weight` into `embed_nvfp4`; `head_nvfp4` copies the struct (both wrap the same device buffers; `Free()` only once — set a flag or free only `embed_nvfp4` and leave `head_nvfp4` as a read-only copy with `d_packed` pointing at the same buffers, documented).

- [ ] **Step 4: Verify pass.** Loader test PASS; legacy fixture (no `global_nvfp4.arrow`) still loads BF16 `d_embed`.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gemma_moe_convert.cpp libs/lsxcommon/include/lsxcommon/gemma_moe_load.h libs/lsxcommon/src/gemma_moe_load.cpp tests/test_gemma_moe_loader.cpp
git commit -m "feat: gemma tied embed/lm_head stored as NVFP4 (global_nvfp4.arrow)"
```

---

### Task 7: Inference — NVFP4 token embedding + LM head

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_inference.cu` (embedding gather kernel; `RunGemmaLogitsArgmax` NVFP4 rows)
- Test: `tests/test_gemma_moe_model.cpp` (synthetic end-to-end) + real-model selfcheck

**Interfaces:**
- Consumes: Task 6 `pw.embed_nvfp4/head_nvfp4`; `E2M1Value`/`E4M3ScaleToFloat` (`nvfp4_codec.h`).
- Produces: `RunGemmaGeneration` works with `d_embed == nullptr && embed_nvfp4.d_packed != nullptr`; `RunGemmaLogitsArgmax` gains an NVFP4 overload (packed/scales/gs + rows/cols instead of BF16 table). No signature change for BF16 callers.

- [ ] **Step 1: Failing test** — synthetic model end-to-end already covers this once converter emits NVFP4 head (Task 6 converts the fixture). Add an assertion to the existing `factory creates registered gemma model` case:

```cpp
    // after Infer succeeds, verify the run used the NVFP4 head when present
    // (log the branch or check via Inspect); the functional gate is that
    // Infer works and is deterministic — existing CHECKs suffice.
```

Concretely: no new test file; the existing two `test_gemma_moe_model` cases ARE the failing tests once Task 6's fixture conversion flips the loader (they will fail if the generation path still requires `d_embed`).

- [ ] **Step 2: Verify fail.** Build `test_gemma_moe_model`; run with `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`. Expected: FAIL (nullptr embed dereference / explicit check).

- [ ] **Step 3: Implement**

Embedding gather kernel:

```cpp
__global__ void gemma_embed_nvfp4_kernel(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float gs, const int32_t *__restrict__ ids,
    __nv_bfloat16 *__restrict__ out, int hidden) {
    const int t = blockIdx.x;
    const int token = ids[t];
    const int64_t row_packed = static_cast<int64_t>(token) * (hidden >> 1);
    const int64_t row_scales = static_cast<int64_t>(token) * (hidden >> 4);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const uint8_t byte = packed[row_packed + (d >> 1)];
        const int nib = (d & 1) ? (byte >> 4) : (byte & 0xF);
        const float w = lsxcommon::gpu::nvfp4::E2M1Value(nib) *
                        lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(
                            scales[row_scales + (d >> 4)]) * gs;
        out[static_cast<int64_t>(t) * hidden + d] = __float2bfloat16(w);
    }
}
```

In `RunGemmaGeneration`, branch the two embed launches on `pw.embed_nvfp4.d_packed`; also skip the `sqrt(H)` embedding-scale kernel? NO — the scale multiply stays (it is applied after embedding in both paths).

LM head: add an NVFP4 rows kernel variant of `gemma_logits_rows_kernel` — same structure, but each vocab row dot decodes E2M1/E4M3 chunk-wise (borrow the inner loop of `GemvDotRow` in scalar form: 32-element chunks, `sb = 2*c` scale-block pair, ascending-d accumulation to keep bit-identical logits ordering); keep `softcap`, `blk_best/blk_idx` plumbing identical. `RunGemmaLogitsArgmax` dispatches on a new optional `const LsxNvfp4Weight *head` parameter (nullptr = legacy BF16).

- [ ] **Step 4: Verify pass.** Synthetic model tests + real NVFP4-converted model selfcheck PASS (3 cycles bit-exact).

- [ ] **Step 5: Measure + commit**

```bash
git add libs/lsxcommon/src/gemma_moe_inference.cu
git commit -m "feat: gemma NVFP4 token embedding gather + NVFP4 LM head rows kernel"
```

---

### Task 8: Device no-handshake routing

**Files:**
- Modify: `libs/lsxcommon/src/gemma_moe_inference.cu` (`EnsureGemmaExpertCache` — device meta mirror; `ForwardLayer` — gather + branch)
- Test: `tests/test_gemma_moe_moe.cpp`

**Interfaces:**
- Consumes: `g_gen_ctx->ec` (`ExpertDeviceCache`); `QwenMoeGvSlotDev` (`qwen_moe_inference.h:47`); `RunRoutedGemvBatchSeg`.
- Produces: `GemmaGenContext` gains `QwenMoeGvSlotDev *d_flat_meta = nullptr` (array [total_experts], uploaded at cache setup, updated on slot Insert) and `bool no_handshake` (true iff `slots == total_experts`). `ForwardLayer` when `no_handshake && seq == 1`: skips the `h_sel` D2H + `ev_router` busy-poll for the MoE block (the D2H copy for LRU bookkeeping also skipped — no misses possible).

- [ ] **Step 1: Failing test** — gather-kernel unit test:

```cpp
TEST_CASE("gemma no-handshake gather composes slot table on device",
          "[gemma][moe][nvfp4]") {
    // 4 fake experts in slots; d_flat_meta uploaded with known gs/coeff
    // d_sel = {2,0}, d_selw = {0.7f, 0.3f}
    // run gather kernel; copy back tbl[2]; expect tbl[k].dev_slot ==
    // meta[sel[k]].dev_slot etc.
}
```

- [ ] **Step 2: Verify fail.** Build. Expected: FAIL (kernel absent).

- [ ] **Step 3: Implement**

Gather kernel (in `gemma_moe_inference.cu`, near the routed GEMV chain):

```cpp
__global__ void gemma_gemv_gather_table_kernel(
    const int32_t *__restrict__ d_sel, const float *__restrict__ d_selw,
    const lsxcommon::qwen_moe::QwenMoeGvSlotDev *__restrict__ flat_meta,
    int K, lsxcommon::qwen_moe::QwenMoeGvSlotDev *__restrict__ tbl) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int expert = d_sel[k];
    lsxcommon::qwen_moe::QwenMoeGvSlotDev v = flat_meta[expert];
    v.coeff = d_selw[k];
    tbl[k] = v;
}
```

`EnsureGemmaExpertCache`: `cudaMalloc` `d_flat_meta` [total_experts]; fill host-side from `db.entries` (gs1/gs3/gs2; `dev_slot = -1` until resident). On every `ec.Insert(flat)` in both staging paths, also update the device mirror entry (small async H2D of one 24-byte struct — or maintain the whole table and re-upload on change; insertion is rare after prewarm).

`ForwardLayer` MoE block when `g_gen_ctx->no_handshake && seq == 1 && batched_path`: skip the `cudaMemcpyAsync(h_sel...)` pair + busy-poll; run gather kernel into `d_tbl` (persistent per-context buffer, not `cudaMallocAsync`); run `RunRoutedGemvBatchSeg` with the device table. The staged-slot `for` loop (miss handling) is skipped entirely — prewarm filled all slots. Guard: `h_sel` consumers later in the pipeline — none (selection only used for staging/weights).

- [ ] **Step 4: Verify.** Gather test PASS; real-model selfcheck (NVFP4 conversion) PASS 3 bit-exact cycles; legacy path (small `LSX_GEMMA_EXPERT_CACHE_SLOTS=64`) still generates correctly via handshake fallback.

- [ ] **Step 5: Measure + commit**

```bash
git add libs/lsxcommon/src/gemma_moe_inference.cu tests/test_gemma_moe_moe.cpp
git commit -m "feat: gemma device-composed routed-GEMV slot table (no router handshake)"
```

---

### Task 9: Full validation + performance report

**Files:**
- Test: all gemma suites; no new files

- [ ] **Step 1: Full synthetic suite**

```bash
for t in test_gemma_moe_convert test_gemma_moe_loader test_gemma_moe_model \
         test_gemma_moe_moe test_gemma_moe_attn test_gemma_moe_norm \
         test_gemma_moe_rotary; do
  LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/$t || exit 1
done
./build/linux-release-znver4-aocc/tests/test_qwen_nvfp4_gemv || exit 1
```

Expected: all PASS.

- [ ] **Step 2: Real-model selfcheck (3 cycles, bit-exact)**

```bash
LSX_TEST_GEMMA_MODEL_PATH=<nvfp4-converted-dir> \
  ./build/linux-release-znver4-aocc/tests/test_gemma_moe_selfcheck
```

Expected: PASS.

- [ ] **Step 3: Performance A/B**

```bash
# new: NVFP4 conversion
./build/linux-release-znver4-aocc/apps/lsxfabric/lsxfabric \
  --localmodel <nvfp4-dir> --testprompt --max-gen 64   # 3 runs
# rollback: same binary, legacy weights dir
./build/linux-release-znver4-aocc/apps/lsxfabric/lsxfabric \
  --localmodel /mnt/llm_models/models/gemma-4-26B-A4B-it --testprompt --max-gen 64
```

Expected: ≥70 t/s on the NVFP4 conversion; report the numbers in the commit message.

- [ ] **Step 4: Commit (report + any fixups)**

```bash
git add -A
git commit -m "perf: gemma NVFP4 decode -- final validation and numbers"
```
