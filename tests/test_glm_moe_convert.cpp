#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/glm_moe_convert.h>
#include <lsxcommon/schema.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "test_glm_moe_fixture.h"

namespace fs = std::filesystem;

using lsxcommon::glm::TensorKind;
using lsxcommon::glm::ParsedTensor;
using lsxcommon::glm::classify;

TEST_CASE("GLM map gate/up/down expert proj to w1/w2/w3", "[glm][convert]")
{
    auto gate = classify("model.layers.42.mlp.experts.128.gate_proj.weight");
    REQUIRE(gate.kind == TensorKind::RoutedExpert);
    CHECK(gate.layer == 42);
    CHECK(gate.expert == 128);
    CHECK(gate.weight_id == 1);

    auto down = classify("model.layers.3.mlp.experts.0.down_proj.weight");
    REQUIRE(down.kind == TensorKind::RoutedExpert);
    CHECK(down.layer == 3);
    CHECK(down.expert == 0);
    CHECK(down.weight_id == 2);

    auto up = classify("model.layers.78.mlp.experts.255.up_proj.weight");
    REQUIRE(up.kind == TensorKind::RoutedExpert);
    CHECK(up.layer == 78);
    CHECK(up.expert == 255);
    CHECK(up.weight_id == 3);
}

TEST_CASE("GLM shared expert projection is routed per layer", "[glm][convert]")
{
    auto gate = classify("model.layers.10.mlp.shared_experts.gate_proj.weight");
    REQUIRE(gate.kind == TensorKind::SharedExpert);
    CHECK(gate.layer == 10);
    CHECK(gate.weight_id == 1);

    auto up = classify("model.layers.10.mlp.shared_experts.up_proj.weight");
    REQUIRE(up.kind == TensorKind::SharedExpert);
    CHECK(up.layer == 10);
    CHECK(up.weight_id == 3);
}

TEST_CASE("GLM router gate + e_score bias classify as router", "[glm][convert]")
{
    auto gate = classify("model.layers.8.mlp.gate.weight");
    REQUIRE(gate.kind == TensorKind::Router);
    CHECK(gate.layer == 8);

    auto bias = classify("model.layers.8.mlp.gate.e_score_correction_bias");
    REQUIRE(bias.kind == TensorKind::Router);
    CHECK(bias.layer == 8);
    CHECK(bias.base == "e_score_correction_bias");
}

TEST_CASE("GLM dense MLP proj (layers 0-2) classified, not experts", "[glm][convert]")
{
    auto gate = classify("model.layers.0.mlp.gate_proj.weight");
    REQUIRE(gate.kind == TensorKind::DenseMlp);
    CHECK(gate.layer == 0);
    CHECK(gate.base == "gate_proj");

    auto up = classify("model.layers.2.mlp.up_proj.weight");
    REQUIRE(up.kind == TensorKind::DenseMlp);
    CHECK(up.layer == 2);
    CHECK(up.base == "up_proj");
}

TEST_CASE("GLM MLA attention tensors classify per layer", "[glm][convert]")
{
    for (const auto *base : {
            "q_a_proj", "q_a_layernorm", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj"}) {
        auto p = classify(
            std::string("model.layers.5.self_attn.") + base + ".weight");
        CHECK(p.kind == TensorKind::Attention);
        CHECK(p.layer == 5);
        CHECK(p.base == base);
    }
}

TEST_CASE("GLM indexer tensors classify on full layers", "[glm][convert]")
{
    auto wq_b = classify(
        "model.layers.6.self_attn.indexer.wq_b.weight");
    REQUIRE(wq_b.kind == TensorKind::Indexer);
    CHECK(wq_b.layer == 6);
    CHECK(wq_b.base == "wq_b");

    auto k_norm = classify(
        "model.layers.6.self_attn.indexer.k_norm.bias");
    REQUIRE(k_norm.kind == TensorKind::Indexer);
    CHECK(k_norm.base == "k_norm");
}

TEST_CASE("GLM layer norms classify per layer", "[glm][convert]")
{
    auto in = classify("model.layers.40.input_layernorm.weight");
    REQUIRE(in.kind == TensorKind::LayerNorm);
    CHECK(in.layer == 40);

    auto post = classify("model.layers.40.post_attention_layernorm.weight");
    REQUIRE(post.kind == TensorKind::LayerNorm);
    CHECK(post.layer == 40);
}

TEST_CASE("GLM MTP layer 78 extra tensors classify", "[glm][convert]")
{
    auto enorm = classify("model.layers.78.enorm.weight");
    REQUIRE(enorm.kind == TensorKind::LayerNorm);
    CHECK(enorm.layer == 78);

    auto hnorm = classify("model.layers.78.hnorm.weight");
    REQUIRE(hnorm.kind == TensorKind::LayerNorm);
    CHECK(hnorm.layer == 78);

    auto eh = classify("model.layers.78.eh_proj.weight");
    REQUIRE(eh.kind == TensorKind::Attention);
    CHECK(eh.layer == 78);

    auto head_norm = classify("model.layers.78.shared_head.norm.weight");
    REQUIRE(head_norm.kind == TensorKind::LayerNorm);
    CHECK(head_norm.layer == 78);
}

TEST_CASE("GLM global tensors classify", "[glm][convert]")
{
    auto embed = classify("model.embed_tokens.weight");
    REQUIRE(embed.kind == TensorKind::Global);
    CHECK(embed.base == "embed_tokens");

    auto head = classify("lm_head.weight");
    REQUIRE(head.kind == TensorKind::Global);
    CHECK(head.base == "lm_head");

    auto norm = classify("model.norm.weight");
    REQUIRE(norm.kind == TensorKind::Global);
    CHECK(norm.base == "model.norm");
}

TEST_CASE("GLM unknown / non-GLM names classify as Unknown", "[glm][convert]")
{
    auto deepseek = classify("model.layers.0.ffn.experts.1.w1.weight");
    CHECK(deepseek.kind == TensorKind::Unknown);

    auto garbage = classify("totally.not.a.glm.tensor");
    CHECK(garbage.kind == TensorKind::Unknown);
}

/* ── Topology (extended model.index.json) ─────────────────────────────── */

static fs::path TempIndexPath()
{
    return fs::temp_directory_path() / "logestix_glm_test_index.json";
}

static void PutIndex(const std::string &json)
{
    auto p = TempIndexPath();
    std::ofstream f(p, std::ios::trunc);
    f << json;
}

TEST_CASE("GLM topology parses extended index (79 layers)", "[glm][topology]")
{
    PutIndex(R"({
      "format": "logestix-glm",
      "model": "GLM-5.2",
      "num_layers": 79,
      "experts_per_layer": 256,
      "moe_layers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                     23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                     41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,
                     59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78],
      "dense_layers": [0,1,2],
      "indexer_layers": [0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78],
      "total_pinned_tensors": 1024,
      "total_experts": 19456,
      "vram": {
        "min_bytes": 1000000,
        "optimal_bytes": 2000000,
        "max_bytes": 3000000,
        "context_tokens": 2048
      }
    })");

    auto topo = lsxcommon::glm::load_topology(TempIndexPath().string());
    REQUIRE(topo.error_msg.empty());
    CHECK(topo.num_layers == 79);
    CHECK(topo.experts_per_layer == 256);
    CHECK(topo.moe_layers.size() == 76);
    CHECK(topo.moe_layers.front() == 3);
    CHECK(topo.moe_layers.back() == 78);
    CHECK(!topo.is_moe_layer(0));
    CHECK(!topo.is_moe_layer(2));
    CHECK(topo.is_moe_layer(3));
    CHECK(topo.is_moe_layer(78));
    CHECK(topo.dense_layers.size() == 3);
    CHECK(topo.is_dense_layer(0));
    CHECK(topo.is_dense_layer(2));
    CHECK(!topo.is_dense_layer(3));
    CHECK(topo.indexer_layers.size() == 22);
    CHECK(topo.is_indexer_layer(0));
    CHECK(topo.is_indexer_layer(6));
    CHECK(topo.is_indexer_layer(78));
    CHECK(!topo.is_indexer_layer(3));
    CHECK(topo.total_experts == 19456);
    CHECK(topo.vram.min_bytes == 1000000);
    CHECK(topo.vram.optimal_bytes == 2000000);
    CHECK(topo.vram.max_bytes == 3000000);
    CHECK(topo.vram.context_tokens == 2048);
}

TEST_CASE("GLM topology rejects malformed index", "[glm][topology]")
{
    PutIndex(R"({ "format": "logestix-glm", "model": "GLM-5.2" })");
    auto topo = lsxcommon::glm::load_topology(TempIndexPath().string());
    CHECK(!topo.error_msg.empty());

    PutIndex("not json at all {{{");
    auto topo2 = lsxcommon::glm::load_topology(TempIndexPath().string());
    CHECK(!topo2.error_msg.empty());
}

TEST_CASE("GLM topology rejects index without vram block", "[glm][topology]")
{
    PutIndex(R"({
      "format": "logestix-glm",
      "model": "GLM-5.2",
      "num_layers": 79,
      "experts_per_layer": 256,
      "moe_layers": [],
      "dense_layers": [],
      "indexer_layers": [],
      "total_pinned_tensors": 1024,
      "total_experts": 19456
    })");
    auto topo = lsxcommon::glm::load_topology(TempIndexPath().string());
    CHECK(!topo.error_msg.empty());
    CHECK(topo.error_msg.find("vram") != std::string::npos);
}

/* ── Converter pipeline (synthetic safetensors → logestix layout) ───────── */

// Builds the synthetic input dir used by the converter pipeline tests.
static fs::path GlmBuildInput(fs::path base, bool with_tok,
                              const std::vector<std::string> &drop_names = {})
{
    fs::remove_all(base);
    auto in = base / "model";
    fs::create_directories(in);

    auto tensors = glmtest::BuildSyntheticModel();
    std::vector<glmtest::FakeTensor> keep;
    for (auto &t : tensors) {
        bool drop = false;
        for (auto &n : drop_names)
            if (t.name.find(n) != std::string::npos) { drop = true; break; }
        if (!drop) keep.push_back(t);
    }
    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors", keep);
    glmtest::WriteConfig(in);
    if (with_tok)
        fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                      in / "tokenizer.json");
    return in;
}

TEST_CASE("GLM converter writes logestix layout from synthetic shards",
          "[glm][convert][pipeline]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_pipeline";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors", glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");

    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    // Extended index
    auto index_path = out / "model.index.json";
    REQUIRE(fs::exists(index_path));
    auto topo = lsxcommon::glm::load_topology(index_path.string());
    REQUIRE(topo.error_msg.empty());
    CHECK(topo.num_layers == 79);
    CHECK(topo.experts_per_layer == 2);
    CHECK(topo.total_experts == 152);   // 2 experts x MoE layers 3..78
    CHECK(std::find(topo.moe_layers.begin(), topo.moe_layers.end(), 3) != topo.moe_layers.end());
    CHECK(std::find(topo.moe_layers.begin(), topo.moe_layers.end(), 6) != topo.moe_layers.end());
    CHECK(std::find(topo.moe_layers.begin(), topo.moe_layers.end(), 78) != topo.moe_layers.end());
    CHECK(std::find(topo.moe_layers.begin(), topo.moe_layers.end(), 0) == topo.moe_layers.end());
    CHECK(std::find(topo.dense_layers.begin(), topo.dense_layers.end(), 0) != topo.dense_layers.end());
    CHECK(std::find(topo.indexer_layers.begin(), topo.indexer_layers.end(), 4) != topo.indexer_layers.end());
    CHECK(std::find(topo.indexer_layers.begin(), topo.indexer_layers.end(), 5) == topo.indexer_layers.end());
    CHECK(std::find(topo.indexer_layers.begin(), topo.indexer_layers.end(), 78) == topo.indexer_layers.end());

    // VRAM profile: mandatory, internally consistent.
    CHECK(topo.vram.min_bytes > 0);
    CHECK(topo.vram.optimal_bytes >= topo.vram.min_bytes);
    CHECK(topo.vram.max_bytes > topo.vram.optimal_bytes); // routed experts exist
    CHECK(topo.vram.context_tokens > 0);

    // Pinned files
    REQUIRE(fs::exists(out / "pinned" / "global.arrow"));
    auto globals = glmtest::TensorNamesIn(out / "pinned" / "global.arrow");
    CHECK(glmtest::TensorNamesIn(out / "pinned" / "global.arrow").size() == 3);
    CHECK(globals[0] == "model.embed_tokens.weight");

    // Dense layer 0 file: contains dense MLP + attention, no experts
    REQUIRE(fs::exists(out / "pinned" / "layer_0.arrow"));
    auto l0 = glmtest::TensorNamesIn(out / "pinned" / "layer_0.arrow");
    CHECK(std::find(l0.begin(), l0.end(),
        "model.layers.0.mlp.gate_proj.weight") != l0.end());
    CHECK(std::find(l0.begin(), l0.end(),
        "model.layers.0.self_attn.o_proj.weight") != l0.end());
    CHECK(l0.size() == 12);   // 2 norms + 7 attn + 3 dense

    // MoE layer 3 file: router + attn + norms; routed/shared experts separate
    REQUIRE(fs::exists(out / "pinned" / "layer_3.arrow"));
    auto l3 = glmtest::TensorNamesIn(out / "pinned" / "layer_3.arrow");
    CHECK(std::find(l3.begin(), l3.end(),
        "model.layers.3.mlp.gate.weight") != l3.end());
    CHECK(std::find(l3.begin(), l3.end(),
        "model.layers.3.mlp.gate.e_score_correction_bias") != l3.end());
    CHECK(std::find(l3.begin(), l3.end(),
        "model.layers.3.self_attn.o_proj.weight") != l3.end());
    CHECK(std::find(l3.begin(), l3.end(),
        "model.layers.3.mlp.experts.0.gate_proj.weight") == l3.end());
    CHECK(std::find(l3.begin(), l3.end(),
        "model.layers.3.mlp.shared_experts.gate_proj.weight") == l3.end());

    // Routed experts: exp_0 / exp_1 for each of layers 3, 6, 78
    REQUIRE(fs::exists(out / "experts" / "layer_3" / "exp_0.arrow"));
    REQUIRE(fs::exists(out / "experts" / "layer_3" / "exp_1.arrow"));
    REQUIRE(fs::exists(out / "experts" / "layer_6" / "exp_0.arrow"));
    REQUIRE(fs::exists(out / "experts" / "layer_6" / "exp_1.arrow"));
    REQUIRE(fs::exists(out / "experts" / "layer_78" / "exp_0.arrow"));
    REQUIRE(fs::exists(out / "experts" / "layer_78" / "exp_1.arrow"));

    // Shared experts arrow (one slot per layer 0..78; 3, 6, 78 filled)
    REQUIRE(fs::exists(out / "pinned" / "shared_experts.arrow"));
    auto se_batch = lsxcommon::ReadRecordBatchFromFile(out / "pinned" / "shared_experts.arrow");
    REQUIRE(se_batch.ok());
    auto se_vec = lsxcommon::SharedExpertSchema::UnpackBatch(*se_batch);
    REQUIRE(se_vec.ok());
    CHECK(se_vec->size() == 79);   // one slot per layer 0..78 (only 3, 6, 78 filled)
    CHECK(!(*se_vec)[3].w1_packed.empty());
    CHECK(!(*se_vec)[6].w1_packed.empty());
    CHECK(!(*se_vec)[78].w1_packed.empty());
    CHECK((*se_vec)[3].w2_packed.size() == (*se_vec)[3].w1_packed.size());
    CHECK((*se_vec)[3].w2_packed.size() == (*se_vec)[3].w1_packed.size());
    CHECK((*se_vec)[0].w1_packed.empty());

    fs::remove_all(base);
}

TEST_CASE("GLM converter aborts on missing required tensor", "[glm][convert][integrity]")
{
    auto base = fs::temp_directory_path() / "logestix_glm_integrity";
    auto in = GlmBuildInput(base, true, {"model.layers.3.mlp.gate.weight"});
    auto out = base / "converted";

    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    INFO("ConvertGlmModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("missing tensors") != std::string::npos);
    fs::remove_all(base);
}

TEST_CASE("GLM converter aborts on missing tokenizer.json", "[glm][convert][integrity]")
{
    auto base = fs::temp_directory_path() / "logestix_glm_integrity_tok";
    auto in = GlmBuildInput(base, false);
    auto out = base / "converted";

    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    INFO("ConvertGlmModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("tokenizer.json") != std::string::npos);
    fs::remove_all(base);
}