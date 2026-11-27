// TDD — Nemotron-3-Ultra-550B index loader, frozen-geometry gate and device
// weight graph.
//
// The frozen-geometry gate is the one thing standing between "the wrong
// converted tree" and "plausible garbage at 0.9 t/s": Ultra and the 30B
// Lightning share a source layout but no width, so a tree built by the sibling
// converter must be refused BY NAME, not by whatever shape happens to overflow
// first.
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_ultra_fixture.h"

#include "lsxcommon/nemotron_moe_ultra_convert.h"
#include "lsxcommon/index_layout.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/nemotron_moe_ultra_load.h"
#include "lsxcommon/schema.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_ultra;

namespace {

// Synthetic geometry bypass, set for the whole binary: these cases run on the
// tiny fixture, and every production path keeps the gate closed.
struct EnableSyntheticGeometry {
    EnableSyntheticGeometry()
    {
#ifdef _WIN32
        _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
#else
        setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
#endif
    }
} g_enable_synthetic;

// Builds a converted Ultra tree from the fixture and returns its model root.
fs::path ConvertedFixture(const std::string &tag)
{
    const auto src = ultratest::MakeTempDir(tag + "_src");
    ultratest::WriteUltraCheckpoint(src);
    const auto dst = ultratest::MakeTempDir(tag + "_dst");
    const auto st = ConvertNemotronUltraModel(src.string(), dst.string());
    REQUIRE(st.ok());
    return dst;
}

bool HaveCuda()
{
    int devs = 0;
    return cudaGetDeviceCount(&devs) == cudaSuccess && devs > 0;
}

} // namespace

TEST_CASE("ultra index: a converted tree loads and validates", "[ultra][load]")
{
    const auto tree = ConvertedFixture("nu_idx");
    UltraIndex idx;
    std::string err;
    const bool loaded = LoadUltraIndex(tree.string(), idx, err);
    INFO("index error: " << err);
    REQUIRE(loaded);
    CHECK(err.empty());
    CHECK(idx.layout == lsxcommon::kIndexLayout);
    CHECK(idx.num_layers == 3);
    CHECK(idx.moe_latent_size == ultratest::kTLatent);
    CHECK(idx.n_group == ultratest::kTGroup);
    CHECK(idx.topk_group == ultratest::kTTopkGroup);
    CHECK(idx.num_experts == ultratest::kTExperts);
    CHECK(idx.mtp_tensors == ultratest::kTMtpTensors);
    // The checkpoint declares eos_token_id as a list; both spellings must load
    // and the scalar is the first stop.
    REQUIRE_FALSE(idx.eos_token_ids.empty());
    CHECK(idx.eos_token_ids.front() == 2);
    CHECK(idx.eos_token_id == 2);
    CHECK(ValidateUltraGeometry(idx, err));
}

TEST_CASE("ultra index: a foreign format tag is rejected by name", "[ultra][load]")
{
    const auto tree = ConvertedFixture("nu_tag");
    // Rewrite the tag to the 30B sibling's: same layout family, different schema.
    {
        std::ifstream in(tree / "model.index.json");
        std::stringstream ss;
        ss << in.rdbuf();
        std::string json = ss.str();
        const std::string from = "\"layout\": 2";
        const auto at = json.find(from);
        REQUIRE(at != std::string::npos);
        json.replace(at, from.size(), "\"layout\": 1");
        std::ofstream out(tree / "model.index.json");
        out << json;
    }
    UltraIndex idx;
    std::string err;
    CHECK_FALSE(LoadUltraIndex(tree.string(), idx, err));
    CHECK(err.find("layout") != std::string::npos);
}

TEST_CASE("ultra geometry: a 30B-shaped tree is rejected by field", "[ultra][load]")
{
    // The synthetic bypass would hide exactly this, so close it for the case.
    const char *prev = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    const std::string saved = prev ? prev : "";
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0", 1);

    UltraIndex idx;
    idx.layout = lsxcommon::kIndexLayout;
    idx.num_layers = 52;      // the sibling's
    idx.hidden_size = 2688;   // the sibling's
    idx.vocab_size = 131072;
    idx.num_experts = 128;
    std::string err;
    const bool ok = ValidateUltraGeometry(idx, err);
    CHECK_FALSE(ok);
    CHECK(err.find("num_layers") != std::string::npos);

    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", saved.c_str(), 1);
}

TEST_CASE("ultra geometry: a partial tree fails naming the first missing layer",
          "[ultra][load]")
{
    UltraIndex idx;
    idx.layout = lsxcommon::kIndexLayout;
    idx.num_layers = 3;
    idx.partial = true;
    idx.layers_converted = {1, 2};
    std::string err;
    CHECK_FALSE(ValidateUltraGeometry(idx, err));
    CHECK(err.find("layer 0") != std::string::npos);
}

TEST_CASE("ultra weights: the latent bridge is mandatory when the index declares it",
          "[ultra][load]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_lat");
    UltraIndex idx;
    std::string err;
    const bool loaded = LoadUltraIndex(tree.string(), idx, err);
    INFO("index error: " << err);
    REQUIRE(loaded);

    // Overwrite the layer's NVFP4 record set with an EMPTY batch: the file opens
    // and the layer's other weights still resolve, so the only thing that can
    // fail is the latent-bridge check. That is the case a config/tensor mismatch
    // produces in the wild (latent declared, projections never converted).
    REQUIRE(lsxcommon::WriteBatch(lsxcommon::DenseNvfp4Schema::MakeBatch({}),
                       tree / "pinned" / "layer_1_nvfp4.arrow")
                .ok());

    UltraWeights w;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const bool ok = LoadUltraWeights(tree.string(), idx, w, stream, err);
    cudaStreamDestroy(stream);
    CHECK_FALSE(ok);
    CHECK(err.find("fc1_latent_proj") != std::string::npos);
    w.Free();
}

TEST_CASE("ultra weights: a missing layer file is named, not fudged", "[ultra][load]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_miss");
    UltraIndex idx;
    std::string err;
    const bool loaded = LoadUltraIndex(tree.string(), idx, err);
    INFO("index error: " << err);
    REQUIRE(loaded);

    fs::remove(tree / "pinned" / "layer_2_nvfp4.arrow");

    UltraWeights w;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const bool ok = LoadUltraWeights(tree.string(), idx, w, stream, err);
    cudaStreamDestroy(stream);
    CHECK_FALSE(ok);
    CHECK(err.find("layer_2_nvfp4.arrow") != std::string::npos);
    w.Free();
}

TEST_CASE("ultra weights: the synthetic tree uploads, exposes the latent widths and frees",
          "[ultra][load]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_w");
    UltraIndex idx;
    std::string err;
    const bool loaded = LoadUltraIndex(tree.string(), idx, err);
    INFO("index error: " << err);
    REQUIRE(loaded);
    REQUIRE(ValidateUltraGeometry(idx, err));

    UltraWeights w;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(LoadUltraWeights(tree.string(), idx, w, stream, err));
    CHECK(w.resident_bytes > 0);
    REQUIRE(w.layers.size() == 3);

    // Mamba layer: in_proj is [35072, 8192] on the real model, [kTMambaIn, kTH]
    // here; the loader takes the shape from the record, never from the passport.
    CHECK(w.layers[0].in_proj.rows == ultratest::kTMambaIn);
    CHECK(w.layers[0].in_proj.cols == ultratest::kTH);
    // MoE layer at the LATENT width.
    CHECK(w.layers[1].latent_up.rows == ultratest::kTLatent);
    CHECK(w.layers[1].latent_up.cols == ultratest::kTH);
    CHECK(w.layers[1].latent_down.rows == ultratest::kTH);
    CHECK(w.layers[1].latent_down.cols == ultratest::kTLatent);
    CHECK(w.layers[1].shared_up.rows == ultratest::kTShared);
    // Attention layer: q at full width, k/v at the KV width (GQA 2:1).
    CHECK(w.layers[2].q.rows == ultratest::kTAH * ultratest::kTAHD);
    CHECK(w.layers[2].k.rows == ultratest::kTKVH * ultratest::kTAHD);

    w.Free();
    CHECK(w.d_embed == nullptr);
    CHECK(w.d_head == nullptr);
    CHECK(w.layers.empty());
    CHECK(w.resident_bytes == 0);
    cudaStreamDestroy(stream);
}
