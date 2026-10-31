// Ling-3.1-flash loader stage (Model Ingestion Protocol step 3): converts a
// synthetic checkpoint, uploads the dense graph and the expert-store geometry,
// and proves Free() releases every allocation (a second load succeeds).
#include <catch2/catch_test_macros.hpp>

#include "ling_moe_constants.h"
#include "lsxcommon/ling_moe_convert.h"
#include "lsxcommon/ling_moe_load.h"
#include "test_ling_moe_fixture.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>

using namespace lsxcommon::ling;
namespace fs = std::filesystem;

namespace {

bool HasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct Tree {
    fs::path in;
    fs::path out;
    Tree()
    {
        in = fs::temp_directory_path() / "lsx_ling_loader_in";
        out = fs::temp_directory_path() / "lsx_ling_loader_out";
        fs::remove_all(in);
        fs::remove_all(out);
        ling_test::WriteCheckpoint(in);
        auto st = ConvertLingModel(in.string(), out.string());
        REQUIRE(st.ok());
    }
    ~Tree()
    {
        std::error_code ec;
        fs::remove_all(in, ec);
        fs::remove_all(out, ec);
    }
};

} // namespace

TEST_CASE("LoadLingIndex reads the produced manifest", "[ling-moe-loader]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    Tree t;
    LingIndex top = LoadLingIndex(t.out.string());
    INFO(top.error_msg);
    REQUIRE(top.error_msg.empty());
    CHECK(top.format == frozen::kFormatTag);
    CHECK(top.num_layers == ling_test::TestCfg().layers);
    CHECK(top.num_experts == ling_test::TestCfg().experts);
    CHECK(top.hidden == ling_test::TestCfg().hidden);
    REQUIRE(top.mla_layers.size() == 1);
    CHECK(top.mla_layers[0] == 7);
    CHECK(top.kda_layers.size() == 7);
    CHECK(top.moe_layers.size() == 7);
    CHECK(top.dense_layers.size() == 1);
    CHECK(top.first_moe_layer == 1);
    CHECK(top.eos_token_id == ling_test::TestCfg().eos);
    CHECK(top.pad_token_id == ling_test::TestCfg().pad);
    CHECK(top.total_experts == 56);
}

TEST_CASE("LoadLingDenseWeights uploads the graph and frees it",
          "[ling-moe-loader]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    Tree t;
    LingPinnedWeights pw;
    std::string error;
    const bool ok = LoadLingDenseWeights(t.out.string(), pw, nullptr, error);
    INFO(error);
    REQUIRE(ok);
    const auto &c = ling_test::TestCfg();
    const int KD = c.heads * c.head_dim;

    REQUIRE(pw.layers.size() == static_cast<size_t>(c.layers));
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);

    const auto &kda = pw.layers[0];
    CHECK_FALSE(kda.is_mla);
    REQUIRE(kda.kda_q.d_packed != nullptr);
    CHECK(kda.kda_q.rows == KD);
    CHECK(kda.kda_q.cols == c.hidden);
    CHECK(kda.kda_o.rows == c.hidden);
    CHECK(kda.kda_o.cols == KD);
    CHECK(kda.kda_f.rows == KD);
    CHECK(kda.kda_g.rows == KD);
    CHECK(kda.d_conv_q != nullptr);
    CHECK(kda.d_a_log != nullptr);
    CHECK(kda.d_dt_bias != nullptr);
    CHECK(kda.d_kda_o_norm != nullptr);
    CHECK(kda.dense_gate.d_packed != nullptr);
    CHECK(kda.dense_gate.rows == c.dense_inter);

    const auto &mla = pw.layers[7];
    CHECK(mla.is_mla);
    REQUIRE(mla.mla_q_a.d_packed != nullptr);
    CHECK(mla.mla_q_a.rows == c.q_lora);
    CHECK(mla.mla_q_a.cols == c.hidden);
    CHECK(mla.mla_q_b.rows == c.heads * (c.qk_nope + c.qk_rope));
    CHECK(mla.mla_q_b.cols == c.q_lora);
    CHECK(mla.mla_kv_a.rows == c.kv_lora + c.qk_rope);
    CHECK(mla.mla_kv_a.cols == c.hidden);
    CHECK(mla.mla_kv_b.rows == c.heads * (c.qk_nope + c.v_head));
    CHECK(mla.mla_kv_b.cols == c.kv_lora);
    CHECK(mla.mla_dense.rows == c.hidden);
    CHECK(mla.mla_dense.cols == c.heads * c.v_head);
    CHECK(mla.d_q_a_norm != nullptr);
    CHECK(mla.d_kv_a_norm != nullptr);
    CHECK(mla.d_mla_g != nullptr);

    const auto &moe = pw.layers[1];
    CHECK(moe.is_moe);
    CHECK(moe.d_router != nullptr);
    CHECK(moe.router_bytes == static_cast<int64_t>(c.experts) * c.hidden * 2);
    CHECK(moe.d_router_bias != nullptr);
    CHECK(moe.shared_gate.rows == c.shared_inter);
    CHECK(moe.shared_gate.cols == c.hidden);
    CHECK(moe.shared_up.rows == c.shared_inter);
    CHECK(moe.shared_down.rows == c.hidden);
    CHECK(moe.shared_down.cols == c.shared_inter);

    auto &store = pw.expert_store;
    CHECK(store.num_layers == c.layers);
    CHECK(store.num_experts == c.experts);
    REQUIRE(store.records.size() == static_cast<size_t>(c.layers));
    const auto &r = store.records[1][0];
    CHECK(r.w1_bytes == static_cast<int64_t>(c.moe_inter) * c.hidden / 2);
    CHECK(r.w2_bytes == static_cast<int64_t>(c.moe_inter) * c.hidden / 2);
    CHECK(r.w3_bytes == static_cast<int64_t>(c.moe_inter) * c.hidden / 2);
    CHECK(r.gs1 != 1.0f);
    CHECK(store.d_seg_wgs != nullptr);

    const lsxcommon::KvCacheGeometry kv = LingKvGeometry(pw.top, 16);
    CHECK(kv.layers == 1);
    CHECK(kv.kv_heads == c.heads);
    CHECK(kv.head_dim == c.qk_nope + c.qk_rope + c.v_head);
    CHECK(kv.Elements() == 1 * c.heads * 16 *
                                (c.qk_nope + c.qk_rope + c.v_head));

    pw.Free();
    CHECK(pw.layers.empty());
    CHECK(pw.expert_store.num_layers == 0);

    // Second load in the same process: a leak shows up as OOM here.
    REQUIRE(LoadLingDenseWeights(t.out.string(), pw, nullptr, error));
    pw.Free();
}
