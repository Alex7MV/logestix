// TDD — GigaChat4-37B-A4B device weight graph. Converts the synthetic
// checkpoint, uploads the dense graph (including the DSA indexer weights the
// 3.5 sibling does not have) and resolves one routed expert through the slot
// LRU. GPU-gated.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"

#include "lsxcommon/gigachat_moe_37b_convert.h"
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

namespace {

void SetSyntheticGeometry(bool on)
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0", 1);
#endif
}

std::filesystem::path BuildConverted()
{
    SetSyntheticGeometry(true);
    const auto src = gigachat4test::MakeTempDir("gc4_wg_src");
    const auto dst = gigachat4test::MakeTempDir("gc4_wg_dst");
    gigachat4test::WriteGigaChat4Checkpoint(src);
    const auto st = ConvertGigaChat4Model(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());
    return dst;
}

} // namespace

TEST_CASE("gigachat4 device weight graph loads the converted layout",
          "[gigachat4][weight_graph]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const auto root = BuildConverted();

    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4Index(root.string(), idx, err));
    REQUIRE(idx.num_layers == 2);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    Gc4Weights w;
    REQUIRE(LoadGigaChat4Weights(root.string(), idx, w, stream, err));
    CHECK(err.empty());
    CHECK(w.hidden == gigachat4test::kTH);
    CHECK(w.vocab == gigachat4test::kTVocab);
    CHECK(w.layers.size() == 2);
    CHECK(w.resident_bytes > 0);
    REQUIRE(w.h_embed != nullptr);
    REQUIRE(w.d_lm_head != nullptr);
    CHECK(w.final_norm.dim == gigachat4test::kTH);
    CHECK(w.final_norm.rank == gigachat4test::kTGateRank);

    const lsxcommon::gigachat::GcLayerWeights &l0 = w.layers[0];
    CHECK_FALSE(l0.full_attention);
    CHECK_FALSE(l0.moe);
    CHECK(l0.input_norm.dim == gigachat4test::kTH);
    CHECK(l0.input_norm.rank == gigachat4test::kTGateRank);
    REQUIRE(l0.d_conv1d != nullptr);
    REQUIRE(l0.d_A_log != nullptr);
    REQUIRE(l0.d_dt_bias != nullptr);
    REQUIRE(l0.d_gdn_in_ba != nullptr);
    REQUIRE(l0.d_gdn_norm != nullptr);
    CHECK(l0.gdn_in_qkvz.rows == gigachat4test::kTGdnQkvz);
    CHECK(l0.gdn_in_qkvz.cols == gigachat4test::kTH);
    CHECK(l0.gdn_out_proj.rows == gigachat4test::kTH);
    CHECK(l0.gdn_out_proj.cols == gigachat4test::kTGdnValueDim);
    CHECK(l0.conv_dim == gigachat4test::kTGdnConv);
    CHECK(l0.conv_kernel == 4);
    CHECK(l0.gdn_value_heads == 2);
    CHECK(l0.gdn_ba_rows == gigachat4test::kTGdnBa);
    CHECK(l0.gdn_norm_dim == 128);
    // Dense layer: the BF16 dense MLP became NVFP4 and is resident.
    CHECK(l0.mlp_gate.rows == gigachat4test::kTDenseI);
    CHECK(l0.mlp_gate.cols == gigachat4test::kTH);
    CHECK(l0.mlp_down.rows == gigachat4test::kTH);
    CHECK(l0.mlp_down.cols == gigachat4test::kTDenseI);
    // The dense layer carries no indexer.
    CHECK(l0.indexer.wq_b.d_packed == nullptr);
    CHECK(l0.indexer.d_wk == nullptr);

    const lsxcommon::gigachat::GcLayerWeights &l1 = w.layers[1];
    CHECK(l1.full_attention);
    CHECK(l1.moe);
    REQUIRE(l1.d_router != nullptr);
    REQUIRE(l1.d_router_bias != nullptr);
    CHECK(l1.q_a_norm.dim == gigachat4test::kTQlora);
    CHECK(l1.q_a_norm.rank == gigachat4test::kTGateRank);
    CHECK(l1.kv_a_norm.dim == gigachat4test::kTKvLora);
    CHECK(l1.q_a.rows == gigachat4test::kTQlora);
    CHECK(l1.q_a.cols == gigachat4test::kTH);
    CHECK(l1.q_b.rows == gigachat4test::kTQkOut);
    CHECK(l1.q_b.cols == gigachat4test::kTQlora);
    CHECK(l1.kv_a.rows == gigachat4test::kTKvA);
    CHECK(l1.kv_a.cols == gigachat4test::kTH);
    CHECK(l1.kv_b.rows == gigachat4test::kTKvB);
    CHECK(l1.kv_b.cols == gigachat4test::kTKvLora);
    CHECK(l1.attn_gate.rows == gigachat4test::kTVOut);
    CHECK(l1.attn_gate.cols == gigachat4test::kTH);
    CHECK(l1.o_proj.rows == gigachat4test::kTH);
    CHECK(l1.o_proj.cols == gigachat4test::kTVOut);
    // DSA indexer: wq_b is NVFP4 on the q-lora latent, the rest stays BF16.
    CHECK(l1.indexer.wq_b.rows == gigachat4test::kTIndexQ);
    CHECK(l1.indexer.wq_b.cols == gigachat4test::kTQlora);
    REQUIRE(l1.indexer.d_wk != nullptr);
    REQUIRE(l1.indexer.d_weights_proj != nullptr);
    REQUIRE(l1.indexer.d_k_norm != nullptr);
    CHECK(l1.num_experts == gigachat4test::kTExperts);
    CHECK(l1.shared_gate.rows == gigachat4test::kTMoeI);
    CHECK(l1.shared_gate.cols == gigachat4test::kTH);
    CHECK(l1.shared_down.rows == gigachat4test::kTH);
    CHECK(l1.shared_down.cols == gigachat4test::kTMoeI);

    // Byte identity: the pinned host embed must equal the arrow payload.
    auto views = lsxcommon::LoadTensorBatchViews(
        root / "pinned" / "global.arrow");
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
    // Free() must release every plane: a second full load has to succeed.
    CHECK(LoadGigaChat4Weights(root.string(), idx, w, stream, err));
    CHECK(w.resident_bytes > 0);
    w.Free();
    cudaStreamDestroy(stream);
}

TEST_CASE("gigachat4 expert cache brings up the routed store",
          "[gigachat4][weight_graph]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const auto root = BuildConverted();

    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4Index(root.string(), idx, err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gigachat::GcExpertCache cache;
    const bool init_ok =
        lsxcommon::gigachat::InitGigaChatExpertCache(
            root.string(), idx.num_layers, idx.first_k_dense_replace,
            idx.num_experts, idx.num_experts_per_tok, /*slots_override=*/2,
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

    lsxcommon::gigachat::GcExpertView v =
        lsxcommon::gigachat::ResolveGcExpertView(cache, 1, 0, stream, err);
    CHECK(v.gate_packed != nullptr);
    CHECK(v.up_packed != nullptr);
    CHECK(v.down_packed != nullptr);

    lsxcommon::gigachat::FreeGcExpertCache(cache);
    cudaStreamDestroy(stream);
}
