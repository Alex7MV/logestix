// DeepSeek-V4-Pro-0813 loader stage (Model Ingestion Protocol stage 5).
//
// Covers the family load path on a synthetic converted layout:
//   * the neutral ExpertSchema round-trip (shared wire format),
//   * E4M3 subnormal handling used by the NVFP4 scale codec,
//   * pinned-weight upload + topology-driven layer sizing,
//   * strict error handling and idempotent Free(),
//   * the family-local pending-copy drain hook,
//   * the frozen-passport gate in Model::Load.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/schema.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/api.h>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "deepseek_moe_pro_load.cuh"
#include "test_deepseek_moe_pro_fixture.h"

namespace fs = std::filesystem;

namespace {

lsxcommon::PackedExpert MakeTestExpert()
{
    lsxcommon::PackedExpert e;
    e.w1_packed.resize(1024);
    for (int i = 0; i < 1024; ++i)
        e.w1_packed[i] = static_cast<uint8_t>(i);
    e.w2_packed.resize(1024);
    for (int i = 0; i < 1024; ++i)
        e.w2_packed[i] = static_cast<uint8_t>(255 - i);
    e.w3_packed.resize(512);
    for (int i = 0; i < 512; ++i)
        e.w3_packed[i] = (i % 2 == 0) ? uint8_t(0xAA) : uint8_t(0xBB);
    e.w1_scales = {0x30, 0x38, 0x40, 0x40};
    e.w2_scales = {0x28, 0x38};
    e.w3_scales = {0x38};
    e.w1_global_scale = 1.0f;
    e.w2_global_scale = 1.0f;
    e.w3_global_scale = 1.0f;
    return e;
}

fs::path TempExpertPath()
{
    return fs::temp_directory_path() / "logestix_pro_test_expert.arrow";
}

uint16_t RefBf16ToHalf(uint16_t bits)
{
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    __half h = __float2half(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

} // namespace

TEST_CASE("deepseek-v4-pro ExpertSchema round-trip is a view",
          "[deepseek][pro][loader]")
{
    auto expert = MakeTestExpert();
    auto batch_result = lsxcommon::ExpertSchema::MakeBatch(expert);
    REQUIRE(batch_result.ok());
    auto batch = *batch_result;
    REQUIRE(batch->ValidateFull().ok());

    auto path = TempExpertPath();
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, path).ok());

    auto mmap_res = arrow::io::MemoryMappedFile::Open(
        path.string(), arrow::io::FileMode::READ);
    REQUIRE(mmap_res.ok());
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    REQUIRE(reader_res.ok());
    auto batch_back = (*reader_res)->ReadRecordBatch(0);
    REQUIRE(batch_back.ok());

    auto loaded_result = lsxcommon::ExpertSchema::UnpackBatch(*batch_back);
    REQUIRE(loaded_result.ok());
    const auto &loaded = *loaded_result;

    CHECK(loaded.w1_packed.size() == 1024);
    CHECK(loaded.w2_packed.size() == 1024);
    CHECK(loaded.w3_packed.size() == 512);
    CHECK(memcmp(loaded.w1_packed.data(), expert.w1_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w2_packed.data(), expert.w2_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w3_packed.data(), expert.w3_packed.data(), 512) == 0);
    CHECK(memcmp(loaded.w1_scales.data(), expert.w1_scales.data(), 4) == 0);
    CHECK(memcmp(loaded.w2_scales.data(), expert.w2_scales.data(), 2) == 0);
    CHECK(memcmp(loaded.w3_scales.data(), expert.w3_scales.data(), 1) == 0);

    mmap_res.ValueOrDie().reset();
    fs::remove(path);
}

TEST_CASE("deepseek-v4-pro e4m3 codec keeps subnormals safe",
          "[deepseek][pro][loader][e4m3]")
{
    CHECK(lsxcommon::float_to_e4m3(0.0f) == 0);
    CHECK(lsxcommon::e4m3_to_float(0) == 0.0f);
    CHECK(lsxcommon::float_to_e4m3(0x1.0p-9f) == 0x01);
    CHECK(lsxcommon::float_to_e4m3(0x1.0p-7f) == 0x04);
    CHECK(lsxcommon::float_to_e4m3(0x1.0p-10f) == 0x00);
    CHECK(lsxcommon::e4m3_to_float(0x01) == 0x1.0p-9f);
    CHECK(lsxcommon::e4m3_to_float(0x04) == 0x1.0p-7f);
}

TEST_CASE("deepseek-v4-pro pinned load uploads the synthetic model",
          "[deepseek][pro][loader][fastload]")
{
    auto base = fs::temp_directory_path() / "logestix_ds_pro_loader";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsprofixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::deepseek_pro::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::deepseek_pro::gpu::load_pinned_weights(
                model.string(), pw, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    CHECK(pw.vocab_size == dsprofixtest::kVocab);
    CHECK(pw.layers.size() == dsprofixtest::kNumLayers);
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.resident_bytes > 0);
    // The passport grid: layer 0 ratio 128, layer 2 ratio 4.
    CHECK(pw.layers[0].compress_ratio == 128);
    CHECK(pw.layers[2].compress_ratio == 4);
    CHECK(pw.layers[2].d_idx_wq_b != nullptr);

    {
        uint16_t got = 0;
        REQUIRE(cudaMemcpy(&got, pw.d_embed, 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        CHECK(got == RefBf16ToHalf(0x4248));
    }

    // Idempotent re-load.
    REQUIRE(lsxcommon::deepseek_pro::gpu::load_pinned_weights(
                model.string(), pw, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(pw.layers.size() == dsprofixtest::kNumLayers);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("deepseek-v4-pro loader errors on a missing layer file",
          "[deepseek][pro][loader][strict]")
{
    auto base = fs::temp_directory_path() / "logestix_ds_pro_loader_missing";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsprofixtest::BuildSyntheticModel(model).ok());
    fs::remove(model / "pinned" / "layer_1.arrow");

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::deepseek_pro::gpu::PinnedWeights pw;
    CHECK(lsxcommon::deepseek_pro::gpu::load_pinned_weights(
              model.string(), pw, stream) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    pw.Free();
    CHECK(cudaGetLastError() == cudaSuccess);
    pw.Free();  // idempotent
    CHECK(cudaGetLastError() == cudaSuccess);

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("deepseek-v4-pro loader releases partial globals on failure (A3.3)",
          "[deepseek][pro][loader][strict][leak]")
{
    // Same contract as the V4-Flash loader (A3.3): a failed load releases
    // its own partial globals. Dropping hc_head_scale keeps the batch
    // valid, so embed/head/norm/hc_fn/hc_base are cudaMalloc'd before the
    // strict presence check fails. Cycle 1 frees via pw.Free(); cycle 2
    // uses a fresh struct — a loader that skips its error-path release
    // leaks the whole set into free-VRAM terms on cycle 2.
    auto base = fs::temp_directory_path() / "logestix_ds_pro_loader_partial";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsprofixtest::BuildSyntheticModel(model).ok());
    {
        auto recs = dsprofixtest::GlobalRecords();
        recs.pop_back();   // hc_head_scale missing
        REQUIRE(dsprofixtest::WriteGlobalArrow(model, recs).ok());
    }

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    {
        lsxcommon::deepseek_pro::gpu::PinnedWeights warm;
        (void)lsxcommon::deepseek_pro::gpu::load_pinned_weights(
            model.string(), warm, stream);
        warm.Free();
        (void)cudaGetLastError();
    }

    lsxcommon::deepseek_pro::gpu::PinnedWeights pw;
    CHECK(lsxcommon::deepseek_pro::gpu::load_pinned_weights(
              model.string(), pw, stream) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    pw.Free();

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    lsxcommon::deepseek_pro::gpu::PinnedWeights pw2;
    CHECK(lsxcommon::deepseek_pro::gpu::load_pinned_weights(
              model.string(), pw2, stream) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    REQUIRE(pw2.d_embed == nullptr);
    REQUIRE(pw2.d_head == nullptr);
    REQUIRE(pw2.d_final_norm == nullptr);
    REQUIRE(pw2.d_hc_head_fn == nullptr);
    REQUIRE(pw2.d_hc_head_base == nullptr);

    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    const std::int64_t delta = static_cast<std::int64_t>(free_before) -
                               static_cast<std::int64_t>(free_after);
    INFO("free VRAM delta after a second failed load: " << delta << " bytes");
    REQUIRE(delta < 64 * 1024);
    pw2.Free();

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("deepseek-v4-pro topology reads the converted layout",
          "[deepseek][pro][loader]")
{
    auto dir = fs::temp_directory_path() / "lsx_ds_pro_loader_topo";
    fs::remove_all(dir);
    REQUIRE(dsprofixtest::BuildSyntheticModel(dir).ok());
    const auto topo =
        lsxcommon::deepseek_pro::gpu::load_topology(dir.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.num_layers == dsprofixtest::kNumLayers);
    CHECK(topo.experts_per_layer == 2);
    CHECK(topo.format == "logestix-v1-nvfp4");
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro expert store drains through the family hook",
          "[deepseek][pro][loader]")
{
    // The neutral ExpertStore owns LRU eviction and munmaps a mapping only
    // after the pending-prefetch-copy registry released it. Pro's prefetch
    // copies live in the *family* registry, so the store must call the hook
    // instead of the V4-Flash global one.
    auto dir = fs::temp_directory_path() / "lsx_ds_pro_loader_store";
    fs::remove_all(dir);
    REQUIRE(dsprofixtest::BuildSyntheticModel(dir).ok());
    auto db = lsxcommon::deepseek_pro::gpu::load_expert_index(
        dir.string(), dsprofixtest::kNumLayers, 2);
    CHECK(db.entries.size() ==
          static_cast<size_t>(dsprofixtest::kNumLayers) * 2);
    CHECK(db.pending_drain ==
          &lsxcommon::deepseek_pro::PendingCopyDrain);
    lsxcommon::deepseek_pro::gpu::free_expert_index(db);
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro Model::Load rejects a passport mismatch",
          "[deepseek][pro][loader][passport]")
{
    auto base = fs::temp_directory_path() / "logestix_ds_pro_passport";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsprofixtest::BuildSyntheticModel(model).ok());

    {
        std::ofstream f(model / "model.index.json");
        f << "{\n"
          << "  \"model\": \"DeepSeek-V4-Pro-0813\",\n"
          << "  \"num_layers\": "
          << lsxcommon::deepseek_pro::kLayers + 1 << ",\n"
          << "  \"experts_per_layer\": "
          << lsxcommon::deepseek_pro::kNumRouted << ",\n"
          << "  \"total_experts\": "
          << ((lsxcommon::deepseek_pro::kLayers + 1) *
              lsxcommon::deepseek_pro::kNumRouted)
          << ",\n"
          << "  \"vram\": {\n"
          << "    \"min_bytes\": 1024,\n"
          << "    \"optimal_bytes\": 2048,\n"
          << "    \"max_bytes\": 4096,\n"
          << "    \"context_tokens\": 2048\n"
          << "  }\n"
          << "}\n";
    }

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.model_type = "deepseek-v4-pro-0813";
    lsxcommon::EnsureModelRegistry();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK_FALSE(m->Load());
    fs::remove_all(base);
}
