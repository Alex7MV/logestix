#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/schema.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/api.h>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static lsxcommon::PackedExpert MakeTestExpert()
{
    lsxcommon::PackedExpert e;

    // w1: 1024 bytes of [0..255] repeated
    e.w1_packed.resize(1024);
    for (int i = 0; i < 1024; ++i)
        e.w1_packed[i] = static_cast<uint8_t>(i);

    // w2: 1024 bytes of [255..0] repeated
    e.w2_packed.resize(1024);
    for (int i = 0; i < 1024; ++i)
        e.w2_packed[i] = static_cast<uint8_t>(255 - i);

    // w3: 512 bytes of alternating 0xAA/0xBB
    e.w3_packed.resize(512);
    for (int i = 0; i < 512; ++i)
        e.w3_packed[i] = (i % 2 == 0) ? uint8_t(0xAA) : uint8_t(0xBB);

    // E4M3-encoded scales (power-of-2): e4 = ceil(log2(s)) + 7, byte = (e4 << 3)
    // 0.5  → ceil(log2(0.5)) = -1 → e4=6 → 0x30
    // 1.0  → ceil(log2(1.0)) =  0 → e4=7 → 0x38
    // 2.0  → ceil(log2(2.0)) =  1 → e4=8 → 0x40
    e.w1_scales = {0x30, 0x38, 0x40, 0x40};
    e.w2_scales = {0x28, 0x38};
    e.w3_scales = {0x38};
    e.w1_global_scale = 1.0f;
    e.w2_global_scale = 1.0f;
    e.w3_global_scale = 1.0f;

    return e;
}

static fs::path TempExpertPath()
{
    return fs::temp_directory_path() / "logestix_test_expert.arrow";
}

TEST_CASE("Loader round-trip with synthetic PackedExpert", "[deepseek][loader]")
{
    auto expert = MakeTestExpert();

    auto batch_result = lsxcommon::ExpertSchema::MakeBatch(expert);
    REQUIRE(batch_result.ok());
    auto batch = *batch_result;
    REQUIRE(batch->ValidateFull().ok());

    auto path = TempExpertPath();
    auto st = lsxcommon::WriteRecordBatchToFile(batch, path);
    REQUIRE(st.ok());

    // Read the batch back and decode through the canonical ExpertSchema
    // path (the legacy per-model Loader was removed as prod-dead).
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
    CHECK(loaded.w1_scales.size() == 4);
    CHECK(loaded.w2_scales.size() == 2);
    CHECK(loaded.w3_scales.size() == 1);

    CHECK(memcmp(loaded.w1_packed.data(), expert.w1_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w2_packed.data(), expert.w2_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w3_packed.data(), expert.w3_packed.data(), 512) == 0);

    CHECK(memcmp(loaded.w1_scales.data(), expert.w1_scales.data(), 4) == 0);
    CHECK(memcmp(loaded.w2_scales.data(), expert.w2_scales.data(), 2) == 0);
    CHECK(memcmp(loaded.w3_scales.data(), expert.w3_scales.data(), 1) == 0);

    CHECK(loaded.w1_global_scale == expert.w1_global_scale);
    CHECK(loaded.w2_global_scale == expert.w2_global_scale);
    CHECK(loaded.w3_global_scale == expert.w3_global_scale);
    // `loaded` is a view into the mapping, so the mapping may only be dropped
    // after the last read. On Windows an open mapping holds a file lock that
    // makes remove() fail; on Linux unlinking an open mmap is legal, which is
    // why this only surfaces here.
    mmap_res.ValueOrDie().reset();
    fs::remove(path);
}

TEST_CASE("float_to_e4m3 handles subnormals safely", "[deepseek][schema][e4m3]")
{
    SECTION("zero returns 0") {
        CHECK(lsxcommon::float_to_e4m3(0.0f) == 0);
        CHECK(lsxcommon::e4m3_to_float(0) == 0.0f);
    }
    SECTION("values below half the min subnormal round to zero") {
        // E4M3FN (matching cuda_fp8 / torch.float8_e4m3fn): the smallest
        // subnormal is 2^-9, so anything below 2^-10 rounds to 0x00.
        float tiny = 1.4e-45f;
        uint8_t e4 = lsxcommon::float_to_e4m3(tiny);
        CHECK(e4 == 0x00);
        CHECK(lsxcommon::e4m3_to_float(e4) == 0.0f);
    }
    SECTION("E4M3 subnormal range round-trips") {
        CHECK(lsxcommon::float_to_e4m3(0x1.0p-9f) == 0x01);  // 2^-9
        CHECK(lsxcommon::float_to_e4m3(0x1.0p-7f) == 0x04);  // 2^-7 = 4*2^-9
        CHECK(lsxcommon::float_to_e4m3(0x1.0p-10f) == 0x00); // 2^-10 ties to 0
        CHECK(lsxcommon::e4m3_to_float(0x01) == 0x1.0p-9f);
        CHECK(lsxcommon::e4m3_to_float(0x04) == 0x1.0p-7f);
    }
    SECTION("normal powers of 2 encode correctly") {
        CHECK(lsxcommon::float_to_e4m3(0.0625f) == 0x18);
        CHECK(lsxcommon::float_to_e4m3(0.125f)  == 0x20);
        CHECK(lsxcommon::float_to_e4m3(0.25f)   == 0x28);
        CHECK(lsxcommon::float_to_e4m3(0.5f)    == 0x30);
        CHECK(lsxcommon::float_to_e4m3(1.0f)    == 0x38);
        CHECK(lsxcommon::float_to_e4m3(2.0f)    == 0x40);
        CHECK(lsxcommon::float_to_e4m3(4.0f)    == 0x48);
        CHECK(lsxcommon::float_to_e4m3(8.0f)    == 0x50);
    }
    SECTION("non-power-of-2 rounds to nearest") {
        uint8_t e4 = lsxcommon::float_to_e4m3(3.0f);
        CHECK(e4 == 0x44);
        CHECK(lsxcommon::e4m3_to_float(e4) == 3.0f);
    }
    SECTION("decode matches encode round-trip") {
        float vals[] = {0.0f, 1.0e-10f, 1.0e-40f, 0.1f, 0.5f, 1.0f, 10.0f, 100.0f};
        for (float v : vals) {
            uint8_t e4 = lsxcommon::float_to_e4m3(v);
            float dec = lsxcommon::e4m3_to_float(e4);
            CHECK(dec >= 0.0f);
            // Values at/below half the min subnormal (2^-10) round to zero.
            if (v > 0x1.0p-10f) CHECK(dec > 0.0f);
        }
    }
}

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "deepseek_moe_load.cuh"
#include "test_deepseek_moe_fixture.h"

namespace {

uint16_t RefDsBf16ToHalf(uint16_t bits)
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

TEST_CASE("Fast pinned load uploads synthetic model bit-exact",
          "[deepseek][loader][fastload]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_fast";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    CHECK(pw.vocab_size == dsfixtest::kVocab);
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.resident_bytes > 0);

    // embed[0] must equal BF16(0x4248) converted to FP16 bit-exactly.
    {
        uint16_t got = 0;
        REQUIRE(cudaMemcpy(&got, pw.d_embed, 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        CHECK(got == RefDsBf16ToHalf(0x4248));
    }
    {
        auto &lw = pw.layers[1];
        REQUIRE(lw.d_ffn_norm != nullptr);
        uint16_t got = 0;
        REQUIRE(cudaMemcpy(&got, lw.d_ffn_norm, 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        CHECK(got == RefDsBf16ToHalf(0x3C00));
    }

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader sizes layers from topology",
          "[deepseek][loader][fastload]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_topo";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Topology-driven sizing (index.json says 3; was hardcoded 43).
    CHECK(pw.layers.size() == dsfixtest::kNumLayers);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader errors on missing layer file",
          "[deepseek][loader][strict]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_missing";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());
    fs::remove(model / "pinned" / "layer_1.arrow");

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    CHECK(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
          == cudaErrorInvalidValue);
    // Drop any sticky error the failed load may have left so the Free() checks
    // below reflect Free() alone (independent of load's internal cleanup).
    (void)cudaGetLastError();
    // load_pinned_weights already releases its partial allocation on the error
    // path; the caller's Free() must be a no-op, not a double free. A leaked
    // device error here means the pointers were not nulled after the first
    // release.
    pw.Free();
    CHECK(cudaGetLastError() == cudaSuccess);
    pw.Free();  // Free() is idempotent
    CHECK(cudaGetLastError() == cudaSuccess);

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("Load expert with empty w1 fields", "[deepseek][loader][edge]")
{
    auto expert = MakeTestExpert();
    expert.w1_packed.clear();
    expert.w1_scales.clear();

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

    CHECK(loaded.w1_packed.empty());
    CHECK(loaded.w1_scales.empty());

    CHECK(loaded.w2_packed.size() == 1024);
    CHECK(loaded.w3_packed.size() == 512);
    CHECK(loaded.w2_scales.size() == 2);
    CHECK(loaded.w3_scales.size() == 1);

    CHECK(memcmp(loaded.w2_packed.data(), expert.w2_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w3_packed.data(), expert.w3_packed.data(), 512) == 0);

    CHECK(loaded.w1_global_scale == expert.w1_global_scale);
    CHECK(loaded.w2_global_scale == expert.w2_global_scale);
    CHECK(loaded.w3_global_scale == expert.w3_global_scale);
    mmap_res.ValueOrDie().reset();   // see the round-trip case above
    fs::remove(path);
}

namespace {

bool DsInRange(const void *p, const void *base, size_t bytes)
{
    return p >= base &&
           static_cast<const uint8_t *>(p) <
               static_cast<const uint8_t *>(base) + bytes;
}

} // namespace

TEST_CASE("Arena-backed DeepSeek load: containment + idempotency",
          "[deepseek][loader][fastload]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_arena";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(pw.layers.size() == dsfixtest::kNumLayers);

    for (int l = 0; l < dsfixtest::kNumLayers; ++l) {
        auto &lw = pw.layers[l];
        REQUIRE(lw.d_arena != nullptr);
        REQUIRE(lw.arena_bytes > 0);
        CHECK(DsInRange(lw.d_ffn_norm, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_gate_weight, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_wq_a, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_wq_a_scales, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_hc_attn_fn, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_attn_sink, lw.d_arena, lw.arena_bytes));
    }

    // Idempotent re-load: frees the previous set internally, no leak.
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(pw.layers.size() == dsfixtest::kNumLayers);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader errors on malformed tensor",
          "[deepseek][loader][strict]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_badtensor";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    // Corrupt global.arrow: truncate model.norm.weight payload below its
    // declared shape size.
    {
        auto recs = dsfixtest::GlobalRecords();
        recs[2].data.resize(2);   // shape says kHidden elements
        REQUIRE(dsfixtest::WriteGlobalArrow(model, recs).ok());
    }

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    CHECK(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
          == cudaErrorInvalidValue);
    pw.Free();

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader releases partial globals on a failed load (A3.3)",
          "[deepseek][loader][strict][leak]")
{
    // The missing-hc_head_scale load fails AFTER embed/head/norm/hc_fn/
    // hc_base were cudaMalloc'd. The loader's contract (LayerWeights::Free
    // comment) is that load_pinned_weights releases its partial allocation
    // on error paths — a direct loader caller must not have to know which
    // globals made it in. Probe: free VRAM after the failed load +
    // pw.Free() must return to the pre-load level.
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_partial";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());
    {
        auto recs = dsfixtest::GlobalRecords();
        recs.pop_back();   // hc_head_scale missing: batch stays valid, so
                           // embed/head/norm/hc_fn/hc_base are cudaMalloc'd
                           // before the strict presence check fails.
        REQUIRE(dsfixtest::WriteGlobalArrow(model, recs).ok());
    }

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // Warm-up: one failed load + Free so allocator/context bookkeeping
    // settles before the measured cycles.
    {
        lsxcommon::gpu::PinnedWeights warm;
        (void)lsxcommon::gpu::load_pinned_weights(model.string(), warm, stream);
        warm.Free();
        (void)cudaGetLastError();
    }

    // Cycle 1: the failed load must release its own partial globals. If it
    // does not, pw.Free() here frees them and the VRAM probe reads clean —
    // but cycle 2 uses a FRESH struct with no Free(), so any allocation the
    // loader left behind accumulates in free-VRAM terms.
    lsxcommon::gpu::PinnedWeights pw;
    CHECK(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
          == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    pw.Free();

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw2;
    CHECK(lsxcommon::gpu::load_pinned_weights(model.string(), pw2, stream)
          == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    // Contract (LayerWeights::Free comment): the loader releases its
    // partial allocation on error paths; the caller's Free() is a no-op.
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
    // embed+head are kVocab*kHidden*2 B each (~64 KiB here); the leaked
    // set is >128 KiB. 64 KiB slack absorbs allocator granularity only.
    REQUIRE(delta < 64 * 1024);
    pw2.Free();

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek model Load rejects topology mismatching the passport",
          "[deepseek][loader][passport]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_passport";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    // Rewrite the index with a layer count that violates the frozen passport
    // of the real architecture.
    {
        std::ofstream f(model / "model.index.json");
        f << "{\n"
          << "  \"model\": \"DeepSeek-V4-Flash\",\n"
          << "  \"num_layers\": "
          << lsxcommon::gpu::kNumLayers + 1 << ",\n"
          << "  \"experts_per_layer\": "
          << lsxcommon::gpu::kNumRouted << ",\n"
          << "  \"total_experts\": "
          << ((lsxcommon::gpu::kNumLayers + 1) * lsxcommon::gpu::kNumRouted)
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
    cfg.model_type = "deepseek-v4-flash";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK_FALSE(m->Load());  // hard failure: topology vs passport mismatch
    fs::remove_all(base);
}
