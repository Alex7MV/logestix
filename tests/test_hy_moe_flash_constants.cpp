// Hy4-flash passport (Model Ingestion Protocol step 3): the frozen geometry in
// hy_moe_flash_constants.h must equal the checkpoint's own config.json, field by
// field. The header's static_asserts pin the internal relations; this test is
// what pins the header to the SOURCE — otherwise a transcription slip is a
// compile-time-green lie.
//
// Host-only and cheap: it reads exactly one 10 KB file. Absent checkpoint ->
// SKIP (the vault is optional everywhere), never a failure.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include "test_manifest_query_root.h"
#include "lsxcommon/config_json.h"
#include "hy_moe_flash_constants.h"

namespace {

namespace hy4 = lsxcommon::hy4;

// The SOURCE checkpoint (read-only store) — what 'vault' means in the
// registry schema; the converted tree is ConvertedDirFor/ModelRoot.
std::filesystem::path VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_HY4_FLASH_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(manifest_query::SourceDirFor("hy4-flash"));
}

// The text_config object of config.json. The DOM parser is used on purpose:
// simdjson's ondemand API is forward-only, and this test reads the document out
// of order (scalars, then three arrays, then the nested rope objects).
struct TextConfig {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object text;

    TextConfig(const std::filesystem::path &, const std::string &json)
    {
        REQUIRE(parser.parse(simdjson::padded_string(json)).get(doc) ==
                simdjson::SUCCESS);
        REQUIRE(doc["text_config"].get_object().get(text) == simdjson::SUCCESS);
    }
};

std::string ReadText(const std::filesystem::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

std::vector<int> IntArray(simdjson::dom::object &o, const char *key)
{
    std::vector<int> out;
    simdjson::dom::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return out;
    for (auto e : arr) {
        int64_t v = 0;
        if (e.get_int64().get(v) == simdjson::SUCCESS)
            out.push_back(static_cast<int>(v));
    }
    return out;
}

std::vector<std::string> StrArray(simdjson::dom::object &o, const char *key)
{
    std::vector<std::string> out;
    simdjson::dom::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return out;
    for (auto e : arr) {
        std::string_view sv;
        if (e.get_string().get(sv) == simdjson::SUCCESS)
            out.emplace_back(sv);
    }
    return out;
}

int Int(simdjson::dom::object &o, const char *key, int dflt)
{
    int64_t v = 0;
    if (o[key].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
}

bool Dbl(simdjson::dom::object &o, const char *key, double &out)
{
    return o[key].get_double().get(out) == simdjson::SUCCESS;
}

bool GetBool(simdjson::dom::object &o, const char *key)
{
    bool b = false;
    if (o[key].get_bool().get(b) != simdjson::SUCCESS) return false;
    return b;
}

} // namespace

TEST_CASE("hy4 passport: geometry equals the checkpoint config", "[hy4][fast]")
{
    manifest_query::RequireSourceRoot();
    const auto model_dir = VaultRoot();
    const auto cfg_path = model_dir / "config.json";
    if (!std::filesystem::exists(cfg_path))
        SKIP("Hy4-flash checkpoint not present at " + model_dir.string());

    std::string json = ReadText(cfg_path);
    TextConfig tc{cfg_path, json};
    auto &t = tc.text;

    CHECK(Int(t, "hidden_size", -1) == hy4::kHidden);
    CHECK(Int(t, "num_hidden_layers", -1) == hy4::kLayers);
    CHECK(Int(t, "vocab_size", -1) == hy4::kVocab);
    CHECK(Int(t, "intermediate_size", -1) == hy4::kDenseIntermediate);
    CHECK(Int(t, "moe_intermediate_size", -1) == hy4::kMoeIntermediate);
    CHECK(Int(t, "routed_expert_hidden_size", -1) == hy4::kLatent);
    CHECK(Int(t, "n_routed_experts", -1) == hy4::kNumRouted);
    CHECK(Int(t, "num_experts_per_tok", -1) == hy4::kTopK);
    CHECK(Int(t, "n_shared_experts", -1) == 1);
    CHECK(Int(t, "q_lora_rank", -1) == hy4::kQLora);
    CHECK(Int(t, "kv_lora_rank", -1) == hy4::kKvLora);
    CHECK(Int(t, "qk_nope_head_dim", -1) == hy4::kQkNope);
    CHECK(Int(t, "qk_rope_head_dim", -1) == hy4::kQkRope);
    CHECK(Int(t, "v_head_dim", -1) == hy4::kVHead);
    CHECK(Int(t, "num_attention_heads", -1) == hy4::kHeadsDsa);
    CHECK(Int(t, "swa_num_attention_heads", -1) == hy4::kHeadsSwa);
    CHECK(Int(t, "sliding_window", -1) == hy4::kSwaWindow);
    CHECK(Int(t, "hc_mult", -1) == hy4::kHcMult);
    CHECK(Int(t, "index_n_heads", -1) == hy4::kIndexNHeads);
    CHECK(Int(t, "index_head_dim", -1) == hy4::kIndexHeadDim);
    CHECK(Int(t, "index_topk", -1) == hy4::kIndexTopk);
    CHECK(Int(t, "compressed_indexer_compress_ratio", -1) == hy4::kPoolRatio);
    CHECK(Int(t, "compressed_indexer_group_topk", -1) == hy4::kGroupTopk);
    CHECK(Int(t, "bos_token_id", -1) == hy4::kBosId);
    CHECK(Int(t, "eos_token_id", -1) == hy4::kEosId);
    CHECK(Int(t, "pad_token_id", -1) == hy4::kPadId);

    // Float fields compare in float space: the passport holds floats and the
    // config holds the decimal literal (1e-05 is not exactly 1e-5f in double).
    double d = 0.0;
    REQUIRE(Dbl(t, "rms_norm_eps", d));
    CHECK(static_cast<float>(d) == hy4::kRmsEps);
    REQUIRE(Dbl(t, "hc_eps", d));
    CHECK(static_cast<float>(d) == hy4::kHcEps);
    REQUIRE(Dbl(t, "hc_magnitude", d));
    CHECK(static_cast<float>(d) == hy4::kHcMagnitude);
    REQUIRE(Dbl(t, "routed_scaling_factor", d));
    CHECK(static_cast<float>(d) == hy4::kRouteScale);
    REQUIRE(Dbl(t, "swiglu_limit", d));
    CHECK(static_cast<float>(d) == hy4::kSwigluLimit);

    // The hybrid pattern, layer by layer, against the checkpoint's own lists.
    const auto layer_types = StrArray(t, "layer_types");
    REQUIRE(layer_types.size() == static_cast<size_t>(hy4::kLayers));
    for (int l = 0; l < hy4::kLayers; ++l) {
        const bool dsa = layer_types[static_cast<size_t>(l)] ==
                         "deepseek_sparse_attention";
        const bool swa = layer_types[static_cast<size_t>(l)] ==
                         "sliding_attention";
        CHECK((dsa || swa));
        CHECK(dsa == hy4::IsDsa(l));
        CHECK(swa == hy4::IsSwa(l));
    }

    const auto mlp_types = StrArray(t, "mlp_layer_types");
    REQUIRE(mlp_types.size() == static_cast<size_t>(hy4::kLayers));
    for (int l = 0; l < hy4::kLayers; ++l)
        CHECK((mlp_types[static_cast<size_t>(l)] == "sparse") ==
              hy4::IsMoeLayer(l));

    // The config's `engram_layer_ids` is 1-BASED: it lists [2,12,22,32] while
    // the engram TENSORS and the model card put the four layers at 1,11,21,31.
    // The passport follows the tensors; this case pins the offset so a future
    // reader does not "fix" the passport to agree with the field.
    const auto engram_layers = IntArray(t, "engram_layer_ids");
    std::vector<int> expected_engram;
    for (int l = 0; l < hy4::kLayers; ++l)
        if (hy4::IsEngramCut(l)) expected_engram.push_back(l);
    std::vector<int> shifted = expected_engram;
    for (int &v : shifted) ++v;
    CHECK((engram_layers == expected_engram || engram_layers == shifted));
    CHECK(shifted == engram_layers);

    // The two rope regimes, read out of rope_parameters.
    simdjson::dom::object rp;
    REQUIRE(t["rope_parameters"].get_object().get(rp) == simdjson::SUCCESS);
    simdjson::dom::object swa_rp, dsa_rp;
    REQUIRE(rp["sliding_attention"].get_object().get(swa_rp) ==
            simdjson::SUCCESS);
    REQUIRE(rp["deepseek_sparse_attention"].get_object().get(dsa_rp) ==
            simdjson::SUCCESS);
    double swa_theta = 0.0, dsa_theta = 0.0;
    REQUIRE(swa_rp["rope_theta"].get_double().get(swa_theta) ==
            simdjson::SUCCESS);
    REQUIRE(dsa_rp["rope_theta"].get_double().get(dsa_theta) ==
            simdjson::SUCCESS);
    CHECK(swa_theta == static_cast<double>(hy4::kRopeThetaSwa));
    CHECK(dsa_theta == static_cast<double>(hy4::kRopeThetaDsa));

    // MRoPE sections.
    const auto sections = IntArray(dsa_rp, "mrope_section");
    REQUIRE(sections.size() == 3);
    CHECK(sections[0] == hy4::kMropeSectionT);
    CHECK(sections[1] == hy4::kMropeSectionH);
    CHECK(sections[2] == hy4::kMropeSectionW);
}

TEST_CASE("hy4 passport: no rope scaling, untied head, fp32 head flag",
          "[hy4][fast]")
{
    manifest_query::RequireSourceRoot();
    const auto model_dir = VaultRoot();
    const auto cfg_path = model_dir / "config.json";
    if (!std::filesystem::exists(cfg_path))
        SKIP("Hy4-flash checkpoint not present at " + model_dir.string());

    std::string json = ReadText(cfg_path);
    TextConfig tc{cfg_path, json};
    using namespace lsxcommon::config;

    // No YaRN: the attention scale is 256^-0.5 with no mscale multiplier. A
    // family that ported DeepSeek's mscale would silently rescale every logit.
    // rope_theta lives per layer type; a top-level value would mean the
    // checkpoint switched to a uniform regime.
    double d = 0.0;
    CHECK_FALSE(Dbl(tc.text, "rope_theta", d));
    CHECK_FALSE(GetBool(tc.text, "tie_word_embeddings"));
    CHECK(GetBool(tc.text, "enable_lm_head_fp32"));
    CHECK(Int(tc.text, "num_nextn_predict_layers", -1) == 0);
    CHECK(GetBool(tc.text, "gated_mla"));
    CHECK(GetBool(tc.text, "learnable_sink"));
    CHECK(GetBool(tc.text, "sandwich_norm"));
    CHECK(GetBool(tc.text, "use_mla"));
    CHECK(Int(tc.text, "o_groups", 0) == 1);
}
