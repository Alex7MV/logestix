// TDD — GigaChat4-708B-A41B device weight graph. Converts the synthetic
// checkpoint through the 708B entry point, then loads it with the 708B
// expectation: the shared loader must accept the member's own passport, upload
// the whole graph (DSA indexer planes included) and resolve a routed expert
// through the shared store. GPU-gated.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"

#include "test_gigachat4_fixture.h"

#include "gigachat_moe_708b_constants.h"
#include "lsxcommon/gigachat_moe_708b_convert.h"
#include "lsxcommon/gigachat_moe_708b_model.h"
#include "lsxcommon/gigachat_moe_37b_load.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/schema.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using namespace lsxcommon::gigachat4;
namespace gc708 = lsxcommon::gigachat4_708b;
namespace frozen708 = lsxcommon::gigachat4_708b::frozen;

namespace {

void SetSynthetic(bool on)
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0", 1);
#endif
}

std::filesystem::path BuildConverted()
{
    SetSynthetic(true);
    const auto src = gigachat4test::MakeTempDir("gc708_wg_src");
    const auto dst = gigachat4test::MakeTempDir("gc708_wg_dst");
    gigachat4test::WriteGigaChat4Checkpoint(src);
    const auto st = gc708::ConvertGigaChat4_708BModel(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());
    return dst;
}

} // namespace

TEST_CASE("gigachat4-708b weight graph loads under its own expectation",
          "[gigachat708b][weight_graph]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const auto root = BuildConverted();
    // The synthetic tree is tiny, so the geometry gate is bypassed; the EXPECT
    // (gate rank, format tag) still travels through the loader.
    SetSynthetic(true);

    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4IndexWithFormat(root.string(), lsxcommon::kIndexLayout, idx,
                                         err));
    REQUIRE(idx.num_layers == 2);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gigachat::GcWeights w;
    REQUIRE(LoadGigaChat4WeightsWithExpect(root.string(), idx, w, stream,
                                           gc708::Gc708BExpect(), err));
    CHECK(err.empty());
    CHECK(w.hidden == gigachat4test::kTH);
    CHECK(w.vocab == gigachat4test::kTVocab);
    CHECK(w.layers.size() == 2);
    CHECK(w.resident_bytes > 0);
    CHECK(w.final_norm.rank == gigachat4test::kTGateRank);

    // GDN layer: no indexer planes.
    const lsxcommon::gigachat::GcLayerWeights &l0 = w.layers[0];
    CHECK_FALSE(l0.full_attention);
    CHECK(l0.indexer.wq_b.d_packed == nullptr);
    // MLA layer: the DSA indexer is part of the member's graph.
    const lsxcommon::gigachat::GcLayerWeights &l1 = w.layers[1];
    CHECK(l1.full_attention);
    REQUIRE(l1.indexer.d_wk != nullptr);
    REQUIRE(l1.indexer.d_weights_proj != nullptr);
    REQUIRE(l1.indexer.d_k_norm != nullptr);
    CHECK(l1.indexer.wq_b.rows == gigachat4test::kTIndexQ);

    // Free must release every plane: a second full load has to succeed.
    w.Free();
    CHECK(LoadGigaChat4WeightsWithExpect(root.string(), idx, w, stream,
                                         gc708::Gc708BExpect(), err));
    CHECK(w.resident_bytes > 0);
    w.Free();
    cudaStreamDestroy(stream);
    SetSynthetic(false);
}

TEST_CASE("gigachat4-708b expert store serves a routed expert",
          "[gigachat708b][weight_graph]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const auto root = BuildConverted();
    SetSynthetic(true);

    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4IndexWithFormat(root.string(), lsxcommon::kIndexLayout, idx,
                                         err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gigachat::GcExpertCache cache;
    const bool init_ok = lsxcommon::gigachat::InitGigaChatExpertCache(
        root.string(), idx.num_layers, idx.first_k_dense_replace,
        idx.num_experts, idx.num_experts_per_tok, /*slots_override=*/2, stream,
        cache, err);
    INFO("init error: " << err);
    REQUIRE(init_ok);
    CHECK(cache.moe_layers == 1);
    CHECK(cache.layout.ok);
    lsxcommon::gigachat::GcExpertView v =
        lsxcommon::gigachat::ResolveGcExpertView(cache, 1, 0, stream, err);
    CHECK(v.gate_packed != nullptr);
    CHECK(v.up_packed != nullptr);
    CHECK(v.down_packed != nullptr);

    lsxcommon::gigachat::FreeGcExpertCache(cache);
    cudaStreamDestroy(stream);
    SetSynthetic(false);
}
