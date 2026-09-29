// TDD — GigaChat3.5 device weight graph + expert-cache bring-up (Plan 2b
// Task 1). Converts the synthetic checkpoint, uploads the dense graph and
// resolves one routed expert through the slot LRU. GPU-gated.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"

#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_load.h"
#include "lsxcommon/schema.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using namespace lsxcommon::gigachat;

namespace {

void SetSyntheticGeometry(bool on)
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0", 1);
#endif
}
void ForceSyntheticGeometry() { SetSyntheticGeometry(true); }

std::filesystem::path BuildConverted()
{
    ForceSyntheticGeometry();
    const auto src = gigachattest::MakeTempDir("gc_lf_src");
    const auto dst = gigachattest::MakeTempDir("gc_lf_dst");
    gigachattest::WriteGigaChatCheckpoint(src);
    const auto st = ConvertGigaChatModel(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());
    return dst;
}

} // namespace

TEST_CASE("gigachat loader rejects a transposed gated-norm gate projection",
          "[gigachat][load][strict]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    ForceSyntheticGeometry();
    const auto src = gigachattest::MakeTempDir("gc_badnorm_src");
    const auto dst = gigachattest::MakeTempDir("gc_badnorm_dst");
    gigachattest::WriteGigaChatCheckpoint(src, /*verified=*/false,
                                          /*corrupt_gate_down=*/true);
    const auto st = ConvertGigaChatModel(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(dst.string(), idx, err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    GcWeights w;
    CHECK_FALSE(LoadGigaChatWeights(dst.string(), idx, w, stream, err));
    CHECK(err.find("gated norm") != std::string::npos);
    cudaStreamDestroy(stream);
    w.Free();
}

TEST_CASE("gigachat loader enforces the frozen gate rank",
          "[gigachat][load][strict]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    ForceSyntheticGeometry();  // convert stage accepts the synthetic geometry
    const auto src = gigachattest::MakeTempDir("gc_badrank_src");
    const auto dst = gigachattest::MakeTempDir("gc_badrank_dst");
    gigachattest::WriteGigaChatCheckpoint(src, /*verified=*/false,
                                          /*corrupt_gate_down=*/false,
                                          /*gate_rank=*/8);
    const auto st = ConvertGigaChatModel(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    // Enable the frozen gate: the rank branch (not the shape branch) must fire.
    SetSyntheticGeometry(false);
    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(dst.string(), idx, err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    GcWeights w;
    CHECK_FALSE(LoadGigaChatWeights(dst.string(), idx, w, stream, err));
    CHECK(err.find("gate rank") != std::string::npos);
    cudaStreamDestroy(stream);
    w.Free();
}

TEST_CASE("gigachat device weight graph loads the converted layout",
          "[gigachat][load][inference]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const auto root = BuildConverted();

    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(root.string(), idx, err));
    REQUIRE(idx.num_layers == 2);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GcWeights w;
    REQUIRE(LoadGigaChatWeights(root.string(), idx, w, stream, err));
    CHECK(err.empty());
    CHECK(w.hidden == gigachattest::kTH);
    CHECK(w.vocab == gigachattest::kTVocab);
    CHECK(w.layers.size() == 2);
    CHECK(w.resident_bytes > 0);
    REQUIRE(w.h_embed != nullptr);
    REQUIRE(w.d_lm_head != nullptr);
    CHECK(w.final_norm.dim == gigachattest::kTH);
    CHECK(w.final_norm.rank == gigachattest::kTGateRank);

    const GcLayerWeights &l0 = w.layers[0];
    CHECK_FALSE(l0.full_attention);
    CHECK_FALSE(l0.moe);
    CHECK(l0.input_norm.dim == gigachattest::kTH);
    CHECK(l0.input_norm.rank == gigachattest::kTGateRank);
    REQUIRE(l0.d_gdn_in_ba != nullptr);
    REQUIRE(l0.d_gdn_norm != nullptr);
    REQUIRE(l0.d_conv1d != nullptr);
    REQUIRE(l0.d_A_log != nullptr);
    REQUIRE(l0.d_dt_bias != nullptr);
    CHECK(l0.gdn_in_qkvz.rows == gigachattest::kTGdnQkvz);
    CHECK(l0.gdn_in_qkvz.cols == gigachattest::kTH);
    CHECK(l0.gdn_out_proj.rows == gigachattest::kTH);
    CHECK(l0.gdn_out_proj.cols == gigachattest::kTGdnValue);
    CHECK(l0.conv_dim == gigachattest::kTGdnConv);
    CHECK(l0.conv_kernel == 4);
    CHECK(l0.gdn_value_heads == 2);
    CHECK(l0.gdn_ba_rows == gigachattest::kTGdnBa);
    CHECK(l0.gdn_norm_dim == 128);
    CHECK(l0.mlp_gate.rows == gigachattest::kTDenseI);
    CHECK(l0.mlp_gate.cols == gigachattest::kTH);
    CHECK(l0.mlp_down.rows == gigachattest::kTH);
    CHECK(l0.mlp_down.cols == gigachattest::kTDenseI);

    const GcLayerWeights &l1 = w.layers[1];
    CHECK(l1.full_attention);
    CHECK(l1.moe);
    REQUIRE(l1.d_router != nullptr);
    REQUIRE(l1.d_router_bias != nullptr);
    CHECK(l1.q_a_norm.dim == 128);
    CHECK(l1.q_a_norm.rank == gigachattest::kTGateRank);
    CHECK(l1.kv_a_norm.dim == 128);
    CHECK(l1.q_a.rows == 128);
    CHECK(l1.q_a.cols == gigachattest::kTH);
    CHECK(l1.q_b.rows == gigachattest::kTQkOut);
    CHECK(l1.q_b.cols == 128);
    CHECK(l1.kv_a.rows == gigachattest::kTKvA);
    CHECK(l1.kv_a.cols == gigachattest::kTH);
    CHECK(l1.kv_b.rows == gigachattest::kTKvB);
    CHECK(l1.attn_gate.rows == gigachattest::kVOut);
    CHECK(l1.o_proj.cols == gigachattest::kVOut);
    CHECK(l1.num_experts == gigachattest::kTExperts);
    CHECK(l1.shared_gate.rows == gigachattest::kTMoeI);
    CHECK(l1.shared_gate.cols == gigachattest::kTH);
    CHECK(l1.shared_down.rows == gigachattest::kTH);
    CHECK(l1.shared_down.cols == gigachattest::kTMoeI);

    // Byte identity: the pinned host embed must equal the arrow payload.
    auto views = lsxcommon::LoadTensorBatchViews(
        root / "logestix" / "pinned" / "global.arrow");
    REQUIRE(views.ok());
    constexpr std::string_view kEmbed = "model.embed_tokens.weight";
    const lsxcommon::TensorView *emb = nullptr;
    for (const auto &t : views->tensors)
        if (t.name.size() >= kEmbed.size() &&
            t.name.compare(t.name.size() - kEmbed.size(), kEmbed.size(),
                           kEmbed) == 0)
            emb = &t;
    REQUIRE(emb != nullptr);
    CHECK(std::memcmp(w.h_embed, emb->data, static_cast<size_t>(emb->bytes)) ==
          0);

    w.Free();
    cudaStreamDestroy(stream);
}

TEST_CASE("gigachat expert cache brings up the routed store",
          "[gigachat][load][inference]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const auto root = BuildConverted();

    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(root.string(), idx, err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GcExpertCache cache;
    const bool init_ok =
        InitGigaChatExpertCache(root.string(), idx.num_layers,
                                idx.first_k_dense_replace, idx.num_experts,
                                idx.num_experts_per_tok, /*slots_override=*/2,
                                stream, cache, err);
    INFO("init error: " << err);
    REQUIRE(init_ok);
    CHECK(cache.total_experts == 4);
    CHECK(cache.moe_layers == 1);
    REQUIRE(cache.moe_index_of_layer.size() == 2);
    CHECK(cache.moe_index_of_layer[0] == -1);
    CHECK(cache.moe_index_of_layer[1] == 0);
    CHECK(cache.layout.ok);
    CHECK(cache.per_expert_bytes > 0);

    GcExpertView v = ResolveGcExpertView(cache, 1, 0, stream, err);
    CHECK(v.gate_packed != nullptr);
    CHECK(v.up_packed != nullptr);
    CHECK(v.down_packed != nullptr);
    CHECK(v.gate_scales != nullptr);
    CHECK(v.up_scales != nullptr);
    CHECK(v.down_scales != nullptr);

    FreeGcExpertCache(cache);
    cudaStreamDestroy(stream);
}
