// Step-5-Preview loader stage (Model Ingestion Protocol stage 3): the pinned
// dense graph (NVFP4 attention/dense-MLP/shared experts + F32 norms + BF16
// router), the neutral streamed expert store and the device slot cache, all
// from a synthetic converted tree (never the vault).
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/step_moe_load.h"
#include "step_moe_constants.h"
#include "test_step_moe_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace lsxcommon::step;
namespace fs = std::filesystem;

namespace {

fs::path ConvertFixture(const char *tag)
{
    fs::path in = fs::temp_directory_path() / (std::string("lsx_step_l_in_") + tag);
    fs::path out = fs::temp_directory_path() / (std::string("lsx_step_l_out_") + tag);
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    step_test::BuildSyntheticCheckpoint(in);
    auto st = ConvertStepMoeModel(in.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

} // namespace

TEST_CASE("Step-5 topology parses the converted index", "[step-moe-loader]")
{
    fs::path out = ConvertFixture("topo");
    auto top = LoadStepTopology(out.string());
    // A successfully parsed index of this generation reports NO error. This used
    // to assert `error_msg == frozen::kFormatTag`: the loader smuggled the
    // per-model tag through the error field and cleared it at the consumer. The
    // tag is retired (design §2.4.3), and leaving the assertion behind is what
    // made the half-migrated consumer invisible.
    CHECK(top.error_msg.empty());
    const auto &c = step_test::TestCfg();
    CHECK(top.num_layers == c.layers);
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_experts == c.experts);
    CHECK(top.first_moe_layer == c.first_moe_layer);
    REQUIRE(top.mlp_types.size() == static_cast<size_t>(c.layers));
    REQUIRE(top.layer_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.layer_types[3] == "full_attention");
    CHECK(top.eos_token_id == 1);
    CHECK(top.eos_token_id2 == 2);
    std::error_code ec;
    fs::remove_all(out, ec);
}

TEST_CASE("Step-5 loader uploads the dense graph and opens the expert store",
          "[step-moe-loader]")
{
    fs::path out = ConvertFixture("load");
    const auto &c = step_test::TestCfg();

    StepPinnedWeights pw;
    std::string error;
    const bool ok = LoadStepDenseWeights(out.string(), pw, nullptr, error);
    if (!ok) WARN(error);
    REQUIRE(ok);

    CHECK(pw.embed_elems == static_cast<int64_t>(c.vocab) * c.hidden);
    CHECK(pw.head_elems == static_cast<int64_t>(c.vocab) * c.hidden);
    REQUIRE(pw.layers.size() == static_cast<size_t>(c.layers));
    for (int l = 0; l < c.layers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        CHECK(lw.loaded);
        CHECK(lw.attn.q.rows == c.heads * c.head_dim);
        CHECK(lw.attn.q.cols == c.hidden);
        CHECK(lw.attn.k.rows == c.kv_heads * c.head_dim);
        CHECK(lw.attn.o.rows == c.hidden);
        CHECK(lw.attn.g.rows == c.heads);
        CHECK(lw.d_input_norm != nullptr);
        CHECK(lw.d_post_norm != nullptr);
        CHECK(lw.attn.d_q_norm != nullptr);
        CHECK(lw.attn.d_k_norm != nullptr);
        CHECK(lw.is_moe == (l >= c.first_moe_layer));
        if (lw.is_moe) {
            CHECK(lw.d_router != nullptr);
            CHECK(lw.d_router_bias != nullptr);
            CHECK(lw.shared_gate.rows == c.shared_inter);
            CHECK(lw.shared_down.rows == c.hidden);
        } else {
            CHECK(lw.mlp_gate.rows == c.dense_inter);
            CHECK(lw.mlp_down.rows == c.hidden);
        }
    }
    CHECK(pw.resident_bytes > 0);
    // The neutral store is view-backed and bounded, never an arena.
    CHECK(pw.expert_store.experts.entries.size() ==
          static_cast<size_t>(c.layers) * c.experts);
    CHECK(pw.expert_store.experts.first_moe_layer == c.first_moe_layer);
    CHECK(pw.expert_store.records.size() == static_cast<size_t>(c.layers));
    CHECK(pw.expert_cache.DeviceBytes() > 0);
    CHECK(pw.expert_cache_bytes > 0);
    // Byte geometry derived from the source tensors.
    const auto &rec =
        pw.expert_store.records[static_cast<size_t>(c.first_moe_layer)][0];
    const int64_t n = static_cast<int64_t>(c.moe_inter) * c.hidden;
    CHECK(rec.w1_bytes == n / 2);
    CHECK(rec.w2_bytes == n / 2);
    CHECK(rec.w3_bytes == n / 2);
    CHECK(rec.s1_bytes == n / 16);

    pw.Free();
    CHECK(pw.layers.empty());
    CHECK(pw.expert_cache.d_base == nullptr);
    std::error_code ec;
    fs::remove_all(out, ec);
}

TEST_CASE("Step-5 loader refuses an index generation it cannot read",
          "[step-moe-loader]")
{
    // The generation check is the loader's own; without this case a consumer
    // that ignored `error_msg` (which is exactly what the retired format carrier
    // did) would still look green here.
    fs::path out = ConvertFixture("layout");
    const fs::path idx = out / "model.index.json";
    std::string json;
    {
        std::ifstream f(idx);
        json.assign(std::istreambuf_iterator<char>(f),
                    std::istreambuf_iterator<char>());
    }
    const std::string from = "\"layout\": 2";
    REQUIRE(json.find(from) != std::string::npos);
    json.replace(json.find(from), from.size(), "\"layout\": 1");
    {
        std::ofstream f(idx);
        f << json;
    }

    StepPinnedWeights pw;
    std::string error;
    CHECK_FALSE(LoadStepDenseWeights(out.string(), pw, nullptr, error));
    CHECK(error.find("layout 1") != std::string::npos);
    CHECK(error.find("reads layout 2") != std::string::npos);
    std::error_code ec;
    fs::remove_all(out, ec);
}
