#include <catch2/catch_test_macros.hpp>

#include "test_kimi_moe_fixture.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/kimi_moe_load.h"

#include <absl/strings/str_cat.h>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace lsxcommon;
using lsxcommon::kimi::ensure_kimi_expert_resident;
using lsxcommon::kimi::free_kimi_expert_index;
using lsxcommon::kimi::load_kimi_expert_index;

namespace {

// Converts the tiny synthetic model once and returns the output root.
std::string BuildConvertedFixture()
{
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "kimi_loader_in";
    auto out = fs::temp_directory_path() / "kimi_loader_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);

    kimitest::WriteKimiConfig(in);
    kimitest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               kimitest::BuildSyntheticKimiModel(3));
    kimitest::WriteKimiTokenizerFiles(in);
    REQUIRE(lsxcommon::kimi::ConvertKimiModel(in.string(), out).ok());
    return out.string();
}

} // namespace

TEST_CASE("LoadKimiPinnedWeights uploads dense and MoE layers",
          "[kimi][loader]")
{
    auto model = BuildConvertedFixture();

    lsxcommon::kimi::KimiPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaError_t err = LoadKimiPinnedWeights(model, pw, stream);
    REQUIRE(err == cudaSuccess);
    if (err == cudaSuccess) {
        REQUIRE(pw.layers.size() == 3);
        // Dense layer 0: MLP weights present, router absent.
        REQUIRE(pw.layers[0].d_mlp_gate != nullptr);
        REQUIRE(pw.layers[0].d_mlp_down != nullptr);
        REQUIRE(pw.layers[0].d_mlp_up != nullptr);
        REQUIRE(pw.layers[0].d_router_gate == nullptr);
        // MoE layers 1-2: router + gate bias (fixture bias has 16 entries),
        // no dense MLP.
        for (int l = 1; l <= 2; ++l) {
            INFO("layer " << l);
            REQUIRE(pw.layers[l].d_router_gate != nullptr);
            REQUIRE(pw.layers[l].h_gate_bias != nullptr);
            REQUIRE(pw.layers[l].gate_bias_n == 16);
            REQUIRE(pw.layers[l].d_mlp_gate == nullptr);
        }
        // Globals uploaded.
        REQUIRE(pw.d_embed != nullptr);
        REQUIRE(pw.d_head != nullptr);
        REQUIRE(pw.d_final_norm != nullptr);
        REQUIRE(pw.vocab_size == 64);    // fixture embed rows
        REQUIRE(pw.hidden == 16);
    }
    pw.Free();
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaStreamDestroy(stream);
}

TEST_CASE("LoadKimiPinnedWeights errors on missing layer file",
          "[kimi][loader]")
{
    auto model = BuildConvertedFixture();
    auto missing =
        std::filesystem::path(model) / "logestix" / "pinned" / "layer_2.arrow";
    std::filesystem::rename(missing, missing.string() + ".bak");

    lsxcommon::kimi::KimiPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaError_t err = LoadKimiPinnedWeights(model, pw, stream);
    REQUIRE(err == cudaErrorInvalidValue);
    REQUIRE(pw.layers.empty());   // Free() ran on failure

    std::filesystem::rename(missing.string() + ".bak", missing);
    cudaStreamDestroy(stream);
}

TEST_CASE("LoadKimiPinnedWeights releases partial globals on failure (A3.2)",
          "[kimi][loader][leak]")
{
    // LoadGlobal uploads embed, head, then norm. Removing norm.weight from
    // global.arrow keeps the batch valid, so embed+head are cudaMalloc'd
    // before the strict presence check fails. Contract: the loader
    // releases its partial allocation on error paths (the missing-layer
    // case above pins pw.layers.empty()); the globals must follow it.
    // Cycle 1 frees via pw.Free(); cycle 2 uses a fresh struct — a loader
    // that skips the release leaks embed+head into free-VRAM terms.
    namespace fs = std::filesystem;
    auto model = BuildConvertedFixture();
    const fs::path global = fs::path(model) / "logestix" / "pinned" /
                            "global.arrow";
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(global);
        REQUIRE(batch.ok());
        auto recs = lsxcommon::PinnedSchema::UnpackBatch(*batch);
        REQUIRE(recs.ok());
        std::vector<lsxcommon::TensorRecord> keep;
        for (auto &r : *recs)
            if (r.name.find("norm.weight") == std::string::npos)
                keep.push_back(std::move(r));
        REQUIRE(keep.size() == recs->size() - 1);
        auto out = lsxcommon::PinnedSchema::MakeBatch(keep);
        REQUIRE(out.ok());
        REQUIRE(lsxcommon::WriteRecordBatchToFile(*out, global).ok());
    }

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    {
        lsxcommon::kimi::KimiPinnedWeights warm;
        (void)LoadKimiPinnedWeights(model, warm, stream);
        warm.Free();
        (void)cudaGetLastError();
    }

    lsxcommon::kimi::KimiPinnedWeights pw;
    CHECK(LoadKimiPinnedWeights(model, pw, stream) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    pw.Free();

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    lsxcommon::kimi::KimiPinnedWeights pw2;
    CHECK(LoadKimiPinnedWeights(model, pw2, stream) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    REQUIRE(pw2.d_embed == nullptr);
    REQUIRE(pw2.d_head == nullptr);

    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    const std::int64_t delta = static_cast<std::int64_t>(free_before) -
                               static_cast<std::int64_t>(free_after);
    INFO("free VRAM delta after a second failed load: " << delta << " bytes");
    // Fixture embed/head are 64*16*2 B each; cudaMalloc granularity is
    // ~2 MiB per block, so the leaked pair reads ~4 MiB. 1 MiB slack.
    REQUIRE(delta < 1024 * 1024);
    pw2.Free();

    cudaStreamDestroy(stream);
}

TEST_CASE("Kimi expert index resolves routed experts", "[kimi][loader]")
{
    auto model = BuildConvertedFixture();
    auto db = load_kimi_expert_index(model, /*num_layers=*/3,
                                     /*experts_per_layer=*/2);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    ensure_kimi_expert_resident(db, /*layer=*/1, /*expert=*/0, stream);
    const auto &e = db.entries[1 * 2 + 0];
    REQUIRE(e.host_w1 != nullptr);
    REQUIRE(e.host_w2 != nullptr);
    REQUIRE(e.host_w3 != nullptr);
    REQUIRE(e.host_w1_scales != nullptr);
    // Fixture gate proj is [4,32]: packed nibbles 4*16=64 bytes.
    CHECK(e.w1_bytes == 64);
    // Pinned store (r3): blobs live in locked-host 16 GiB chunks (the driver
    // fails a single >300 GiB cudaHostAlloc); pointers stay valid for async
    // H2D without the file-backed VMA penalty. Block layout
    // [w1|w3][s1|s3][w2][s2] enables the split-fence upload.
#if defined(__unix__) || defined(__APPLE__)
    REQUIRE(!db.pinned_pools.empty());
    const auto *chunk0 = static_cast<const uint8_t *>(db.pinned_pools.front());
    constexpr int64_t kChunk = int64_t{16} << 30;
    const auto aln = [](int64_t x) { return (x + 255) / 256 * 256; };
    REQUIRE(e.pinned_blob != nullptr);
    REQUIRE(e.pinned_blob >= chunk0);
    REQUIRE(e.pinned_blob < chunk0 + kChunk);
    REQUIRE(e.blob_a_bytes == aln(e.w1_bytes) + e.w3_bytes);
    REQUIRE(e.blob_b_bytes == aln(e.w1_scales_bytes) + e.w3_scales_bytes);
    REQUIRE(e.blob_c_bytes == e.w2_bytes);
    REQUIRE(e.blob_d_bytes == e.w2_scales_bytes);
    // upload source pointers were re-pointed INTO the pinned chunks
    REQUIRE(e.host_w1 >= chunk0);
    REQUIRE(e.host_w1 < chunk0 + kChunk);
#else
    // Expert pinning is not implemented on Windows (docs/open-tasks/2026-09-30-..., item B1):
    // PinChunked returns false, pinned_pools stays empty, and the loader falls
    // back to pageable MemoryMappedFile views — correctness-neutral, just a
    // slower H2D path. The pointers above are the file-backed fallback views.
    WARN("expert pinning unavailable on this platform; using pageable views");
    CHECK(db.pinned_pools.empty());
#endif
    free_kimi_expert_index(db);
    cudaStreamDestroy(stream);
}

TEST_CASE("kimi expert pinning: LSX_EXPERT_PIN=0 falls back to mmap",
          "[kimi][loader][pin]")
{
    auto model = BuildConvertedFixture();
#ifdef _WIN32
    _putenv_s("LSX_EXPERT_PIN", "0");
#else
    setenv("LSX_EXPERT_PIN", "0", 1);
#endif
    auto db = load_kimi_expert_index(model, /*num_layers=*/3,
                                     /*experts_per_layer=*/2);
    REQUIRE(db.pinned_pools.empty());
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    ensure_kimi_expert_resident(db, /*layer=*/1, /*expert=*/0, stream);
    REQUIRE(db.entries[1 * 2 + 0].host_w1 != nullptr);   // mmap view works
    free_kimi_expert_index(db);
    cudaStreamDestroy(stream);
#ifdef _WIN32
    _putenv_s("LSX_EXPERT_PIN", "");
#else
    unsetenv("LSX_EXPERT_PIN");
#endif
}

// Round-2 Task 4: the r2 loader uploads MLA projections as NVFP4 pairs
// (packed/scale device buffers + global scale) and hard-rejects pre-r2
// indexes.
TEST_CASE("kimi loader r2: MLA NVFP4 roundtrip + old rev rejected",
          "[kimi][loader][r2]")
{
    auto model = BuildConvertedFixture();   // converter now emits r2

    // (a) Happy path: projections arrive as device NVFP4 pairs.
    {
        lsxcommon::kimi::KimiPinnedWeights pw;
        cudaStream_t stream;
        REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
        cudaError_t err = LoadKimiPinnedWeights(model, pw, stream);
        REQUIRE(err == cudaSuccess);
        REQUIRE(pw.layers.size() == 3);
        for (int l = 0; l < 3; ++l) {
            INFO("layer " << l);
            const auto &lw = pw.layers[l];
            REQUIRE(lw.d_q_a_packed != nullptr);
            REQUIRE(lw.d_q_a_scale != nullptr);
            REQUIRE(lw.q_a_gs > 0.f);
            // fixture q_a [16,16] BF16 -> 128 packed bytes / 16 scale bytes
            REQUIRE(lw.q_a_packed_bytes == 128);
            REQUIRE(lw.q_a_scale_bytes == 16);
            REQUIRE(lw.d_q_b_packed != nullptr);
            REQUIRE(lw.d_kv_a_packed != nullptr);
            REQUIRE(lw.d_kv_b_packed != nullptr);
            REQUIRE(lw.d_o_packed != nullptr);
            // norms untouched (BF16 pointers)
            REQUIRE(lw.d_q_a_norm != nullptr);
            REQUIRE(lw.d_kv_a_norm != nullptr);
        }
        pw.Free();
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    // (b) Pre-r2 index is rejected with a strict "reconvert" error.
    {
        namespace fs = std::filesystem;
        auto idx = fs::path(model) / "logestix" / "model.index.json";
        std::string json;
        {
            std::ifstream f(idx);
            json.assign((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        }
        const std::string old_tag = "\"logestix-kimi-r2\"";
        const auto pos = json.find(old_tag);
        REQUIRE(pos != std::string::npos);
        json.replace(pos, old_tag.size(), "\"logestix-kimi\"   ");
        {
            std::ofstream f(idx);
            f << json;
        }
        lsxcommon::kimi::KimiPinnedWeights pw;
        cudaStream_t stream;
        REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
        cudaError_t err = LoadKimiPinnedWeights(model, pw, stream);
        REQUIRE(err == cudaErrorInvalidValue);
        REQUIRE(pw.layers.empty());
        cudaStreamDestroy(stream);
    }
}
