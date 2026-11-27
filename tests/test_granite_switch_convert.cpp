#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_switch_convert.h>
#include <lsxcommon/schema.h>

#include "nvfp4_scale.h"
#include "test_granite_switch_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace lsxcommon::granite_switch;

namespace {

std::vector<std::string> DenseNvfp4NamesIn(const std::filesystem::path &p)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(p);
    if (!batch.ok()) return {};
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!recs.ok()) return {};
    std::vector<std::string> names;
    for (auto &r : *recs) names.push_back(r.name);
    return names;
}

// Decode one written NVFP4 record back to floats, exactly as the inference
// codec does: E2M1 magnitudes, one E4M3 block scale per 16 values, one global
// scale for the tensor.
std::vector<float> DecodeRecord(const lsxcommon::DenseNvfp4Record &r)
{
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const int64_t n = static_cast<int64_t>(r.packed.size()) * 2;
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t byte = r.packed[static_cast<size_t>(i >> 1)];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        const float mag = kMag[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
        const float scale =
            lsxcommon::nvfp4::E4m3ToFloat(r.scales[static_cast<size_t>(i >> 4)]);
        out[static_cast<size_t>(i)] = mag * scale * r.global_scale;
    }
    return out;
}

float RelL2(const std::vector<float> &a, const std::vector<float> &b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size() && i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return den > 0.0 ? static_cast<float>(std::sqrt(num / den)) : 0.0f;
}

// Arrow record name -> source tensor name. The converter renames the composed
// projections (`qkv_proj.base_layer.weight` -> `qkv_proj.weight`), so a test that
// compares a record against the checkpoint must undo that rename.
std::string SourceNameFor(const std::string &arrow)
{
    std::string s = arrow;
    if (s.find(".res") != std::string::npos) s.erase(s.find(".res"));
    for (const auto &kv :
         {std::pair<std::string, std::string>{"self_attn.qkv_proj.weight",
                                             "self_attn.qkv_proj.base_layer.weight"},
          {"self_attn.o_proj.weight", "self_attn.o_proj.base_layer.weight"},
          {"shared_mlp.gate_up_proj.weight",
           "shared_mlp.input_linear.base_layer.weight"},
          {"shared_mlp.down_proj.weight",
           "shared_mlp.output_linear.base_layer.weight"}}) {
        const auto at = s.find(kv.first);
        if (at != std::string::npos) {
            s.replace(at, kv.first.size(), kv.second);
            break;
        }
    }
    return s;
}

std::vector<float> SourceBf16(const std::vector<gswtest::FakeTensor> &ts,
                              const std::string &name)
{
    for (const auto &t : ts) {
        if (t.name != name) continue;
        std::vector<float> out(static_cast<size_t>(t.data.size() / 2));
        for (size_t i = 0; i < out.size(); ++i) {
            const uint16_t bits = static_cast<uint16_t>(
                t.data[2 * i] | (static_cast<uint16_t>(t.data[2 * i + 1]) << 8));
            const uint32_t f32 = static_cast<uint32_t>(bits) << 16;
            float v;
            std::memcpy(&v, &f32, sizeof(v));
            out[i] = v;
        }
        return out;
    }
    return {};
}

bool Contains(const std::vector<std::string> &v, const std::string &s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

} // namespace

TEST_CASE("granite-switch tensor classifier", "[granite_switch][convert]")
{
    auto check = [](const std::string &n, GraniteSwitchTensorKind k,
                    int layer = -1, int slice = -1) {
        auto p = classify_granite_switch_tensor(n);
        CHECK(p.kind == k);
        if (layer >= 0) CHECK(p.layer == layer);
        if (slice >= 0) CHECK(p.slice == slice);
    };

    check("model.embed_tokens.weight", GraniteSwitchTensorKind::Embed);
    check("model.norm.weight", GraniteSwitchTensorKind::FinalNorm);
    check("model.switch.codebook", GraniteSwitchTensorKind::SwitchCodebook);
    check("model.switch.control_to_substitute_lut",
          GraniteSwitchTensorKind::SwitchLut);
    check("model.layers.3.input_layernorm.weight",
          GraniteSwitchTensorKind::InputNorm, 3);
    check("model.layers.3.post_attention_layernorm.weight",
          GraniteSwitchTensorKind::PostNorm, 3);
    check("model.layers.3.self_attn.qkv_proj.base_layer.weight",
          GraniteSwitchTensorKind::AttnQkvBase, 3);
    check("model.layers.3.self_attn.qkv_proj.lora_A_slices.2",
          GraniteSwitchTensorKind::AttnQkvLoraA, 3, 2);
    check("model.layers.3.self_attn.qkv_proj.lora_B_slices.1",
          GraniteSwitchTensorKind::AttnQkvLoraB, 3, 1);
    check("model.layers.3.self_attn.o_proj.base_layer.weight",
          GraniteSwitchTensorKind::AttnOBase, 3);
    check("model.layers.3.self_attn.o_proj.lora_A",
          GraniteSwitchTensorKind::AttnOLoraA, 3);
    check("model.layers.3.self_attn.o_proj.lora_B",
          GraniteSwitchTensorKind::AttnOLoraB, 3);
    check("model.layers.3.shared_mlp.input_linear.base_layer.weight",
          GraniteSwitchTensorKind::MlpGateUpBase, 3);
    check("model.layers.3.shared_mlp.input_linear.lora_A_slices.1",
          GraniteSwitchTensorKind::MlpGateUpLoraA, 3, 1);
    check("model.layers.3.shared_mlp.input_linear.lora_B_slices.0",
          GraniteSwitchTensorKind::MlpGateUpLoraB, 3, 0);
    check("model.layers.3.shared_mlp.output_linear.base_layer.weight",
          GraniteSwitchTensorKind::MlpDownBase, 3);
    check("model.layers.3.shared_mlp.output_linear.lora_A",
          GraniteSwitchTensorKind::MlpDownLoraA, 3);
    check("model.layers.3.shared_mlp.output_linear.lora_B",
          GraniteSwitchTensorKind::MlpDownLoraB, 3);

    check("model.layers.3.self_attn.q_norm.weight",
          GraniteSwitchTensorKind::Unknown);
    check("lm_head.weight", GraniteSwitchTensorKind::Unknown);

    CHECK(granite_switch_tensor_is_nvfp4(GraniteSwitchTensorKind::Embed));
    CHECK(granite_switch_tensor_is_nvfp4(GraniteSwitchTensorKind::AttnQkvBase));
    CHECK_FALSE(granite_switch_tensor_is_nvfp4(
        GraniteSwitchTensorKind::AttnQkvLoraA));
    CHECK_FALSE(granite_switch_tensor_is_nvfp4(
        GraniteSwitchTensorKind::InputNorm));
    CHECK_FALSE(granite_switch_tensor_is_nvfp4(
        GraniteSwitchTensorKind::SwitchCodebook));
}

TEST_CASE("granite-switch converter builds the pinned layout",
          "[granite_switch][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_conv_test";
    remove_all(tmp);
    create_directories(tmp);

    gswtest::BuildConverted(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");

    REQUIRE(ConvertGraniteSwitchModel(tmp.string(), tmp.string()).ok());

    auto top = load_granite_switch_topology(
        (tmp / "model.index.json").string());
    INFO(top.error_msg);
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == 2);
    CHECK(top.switch_cache_layers == 2);
    CHECK(top.hidden_size == 64);
    CHECK(top.intermediate_size == 128);
    CHECK(top.vocab_size == 72);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.head_dim == 16);
    CHECK(top.adapter_ranks == std::vector<int>({4, 8}));
    CHECK(top.adapter_token_ids == std::vector<int>({64, 65}));
    CHECK(top.adapter_substitute_token_ids == std::vector<int>({6, 7}));
    CHECK(top.adapter_names ==
          std::vector<std::string>({"guardian-core", "decision"}));
    CHECK(top.tie_word_embeddings);
    CHECK(top.embedding_multiplier == Catch::Approx(12.0));
    CHECK(top.residual_multiplier == Catch::Approx(0.175));
    CHECK(top.logits_scaling == Catch::Approx(16.0));
    CHECK(top.vram.min_bytes > 0);

    const auto pinned = tmp / "pinned";
    CHECK(exists(pinned / "global.arrow"));
    CHECK(exists(pinned / "global_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_0_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_1_nvfp4.arrow"));
    CHECK(exists(pinned / "layer_0_adapters.arrow"));

    auto gn = DenseNvfp4NamesIn(pinned / "global_nvfp4.arrow");
    CHECK(Contains(gn, "model.embed_tokens.weight"));
    CHECK_FALSE(Contains(gn, "lm_head.weight"));

    auto ln = DenseNvfp4NamesIn(pinned / "layer_0_nvfp4.arrow");
    CHECK(Contains(ln, "model.layers.0.self_attn.qkv_proj.weight"));
    CHECK(Contains(ln, "model.layers.0.self_attn.o_proj.weight"));
    CHECK(Contains(ln, "model.layers.0.shared_mlp.gate_up_proj.weight"));
    CHECK(Contains(ln, "model.layers.0.shared_mlp.down_proj.weight"));

    auto an = gswtest::TensorNamesIn(pinned / "layer_0_adapters.arrow");
    CHECK(Contains(an,
                   "model.layers.0.self_attn.qkv_proj.lora_A_slices.0.adapter0"));
    CHECK(Contains(an,
                   "model.layers.0.self_attn.qkv_proj.lora_B_slices.2.adapter1"));
    CHECK_FALSE(Contains(
        an, "model.layers.0.shared_mlp.input_linear.lora_A_slices.0.adapter0"));
    CHECK(Contains(
        an, "model.layers.0.shared_mlp.input_linear.lora_A_slices.0.adapter1"));
    CHECK(Contains(an,
                   "model.layers.0.shared_mlp.output_linear.lora_B.adapter1"));

    auto rn = gswtest::TensorNamesIn(pinned / "global.arrow");
    CHECK(Contains(rn, "model.norm.weight"));
    CHECK(Contains(rn, "model.switch.codebook"));

    remove_all(tmp);
}

TEST_CASE("granite-switch output identity is the checkpoint, not a literal",
          "[granite_switch][convert]")
{
    // ONE converter serves every member of the family, so a baked-in name is
    // wrong the moment a second member exists: the index would claim to be the
    // 30B, `detect_model_type` would resolve the 30B class, and the checkpoint
    // would only fail later at frozen-geometry validation. The family
    // convention (kimi_moe_convert.cpp) is the checkpoint directory name, which
    // is what the vault and the manifest's `vault.dir` pin.
    using namespace std::filesystem;
    const auto root = temp_directory_path() / "granite_switch_identity";
    remove_all(root);
    const auto in = root / "granite-switch-4.1-8b";
    create_directories(in);
    gswtest::BuildConverted(in);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              in / "tokenizer.json");
    REQUIRE(ConvertGraniteSwitchModel(in.string(), in.string()).ok());

    auto top = load_granite_switch_topology(
        (in / "model.index.json").string());
    CHECK(top.error_msg.empty());
    CHECK(top.model_name == "granite-switch-4.1-8b");
    remove_all(root);
}

TEST_CASE("granite-switch identity survives a trailing separator",
          "[granite_switch][convert]")
{
    // lsxforge is driven by shell scripts; "<vault>/granite-switch-4.1-8b/" is a
    // path a user types. fs::path::filename() of that spelling is empty, which
    // would emit an empty model name and make the tree unloadable in a way that
    // looks like a corrupt index rather than a truncated argument.
    using namespace std::filesystem;
    const auto root = temp_directory_path() / "granite_switch_identity_slash";
    remove_all(root);
    const auto in = root / "granite-switch-4.1-8b";
    create_directories(in);
    gswtest::BuildConverted(in);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              in / "tokenizer.json");
    REQUIRE(ConvertGraniteSwitchModel(in.string() + "/", (in.string() + "/"))
                .ok());
    auto top = load_granite_switch_topology(
        (in / "model.index.json").string());
    CHECK(top.model_name == "granite-switch-4.1-8b");
    remove_all(root);
}

TEST_CASE("granite-switch emits the residual plane on the measured placement",
          "[granite_switch][convert]")
{
    // FSI-01 of the 8B forks at a template choice (`CF` vs the evaluated
    // constant) that the 4-bit body cannot hold: measured 2026-11-07, the source
    // margin is +6.9 logits and the converted model's is -1.3, while the
    // reconstruction error is a healthy relL2 0.094-0.095 on every tensor class.
    // The fix is the documented one: a second NVFP4 plane carrying
    // `w - dequant(base)`, decoded as one more GEMV with acc=true (~8 effective
    // bits for 2x the bytes, no new codec).
    //
    // Placement is MEASURED, not assumed (skill step 4). Four trees were
    // converted from the same checkpoint and probed at the fork; margin =
    // logit(CF) - logit(250), source +6.88:
    //   none      +0.9 GB  -1.27  red
    //   attn      +0.9 GB  +1.58  green  <-- shipped (cheapest that clears, and
    //                                     it leaves the MLP, 79% of the layer's
    //                                     bytes, at 4 bits)
    //   mlp       +3.3 GB  +2.19  green
    //   attn+mlp  +4.2 GB  +7.14  green
    // So the assertion is asymmetric on purpose: PRESENT on attention, ABSENT on
    // the MLP. A test that only checked "some .res records exist" would pass for
    // any placement and pin nothing.
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_residual";
    remove_all(tmp);
    create_directories(tmp);
    gswtest::BuildConverted(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    REQUIRE(ConvertGraniteSwitchModel(tmp.string(), tmp.string()).ok());

    const auto nv = tmp / "pinned" / "layer_0_nvfp4.arrow";
    const auto names = DenseNvfp4NamesIn(nv);
    for (const char *t : {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight"}) {
        const std::string base = std::string("model.layers.0.") + t;
        INFO(base);
        CHECK(Contains(names, base));
        CHECK(Contains(names, base + ".res"));
    }
    for (const char *t : {"shared_mlp.gate_up_proj.weight",
                          "shared_mlp.down_proj.weight"}) {
        const std::string base = std::string("model.layers.0.") + t;
        INFO(base);
        CHECK(Contains(names, base));
        CHECK_FALSE(Contains(names, base + ".res"));
    }

    // Excluded on purpose: the embedding/head (already FP16 in the loader's
    // view of the tied table), the norms and the switch codebook. A fake plane
    // on any of them would cost bytes and change nothing the gates read.
    for (const std::string &n : names) {
        if (n.find(".res") == std::string::npos) continue;
        INFO(n);
        CHECK(n.find("embed_tokens") == std::string::npos);
        CHECK(n.find("model.norm") == std::string::npos);
        CHECK(n.find("codebook") == std::string::npos);
        CHECK(n.find("layernorm") == std::string::npos);
    }

    // The index records the placement, so a reader of the tree can tell a
    // residual build from a legacy one without decoding a weight.
    std::ifstream f(tmp / "model.index.json");
    std::string idx((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    // The tag names the placement, so a reader can tell the trees apart without
    // decoding a weight — and so a future ablation is visible in the artifact.
    CHECK(idx.find("\"residual\": \"attn\"") != std::string::npos);

    remove_all(tmp);
}

TEST_CASE("granite-switch residual plane halves the 4-bit reconstruction error",
          "[granite_switch][convert]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_residual_err";
    remove_all(tmp);
    create_directories(tmp);
    auto tensors = gswtest::BuildModel();
    gswtest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gswtest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    REQUIRE(ConvertGraniteSwitchModel(tmp.string(), tmp.string()).ok());

    const auto nv = tmp / "pinned" / "layer_0_nvfp4.arrow";
    auto batch = lsxcommon::ReadRecordBatchFromFile(nv);
    REQUIRE(batch.ok());
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(recs.ok());

    // Same two tensors the shipped placement covers; the MLP has no plane by
    // measurement, so requiring one here would contradict the artifact.
    for (const char *t : {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight"}) {
        const std::string base = std::string("model.layers.0.") + t;
        const lsxcommon::DenseNvfp4Record *b = nullptr;
        const lsxcommon::DenseNvfp4Record *r = nullptr;
        for (const auto &x : *recs) {
            if (x.name == base) b = &x;
            if (x.name == base + ".res") r = &x;
        }
        REQUIRE(b != nullptr);
        REQUIRE(r != nullptr);
        const auto src = SourceBf16(tensors, SourceNameFor(base));
        REQUIRE(!src.empty());
        const auto dq_base = DecodeRecord(*b);
        const auto dq_res = DecodeRecord(*r);
        REQUIRE(dq_base.size() == src.size());
        REQUIRE(dq_res.size() == src.size());
        std::vector<float> sum(src.size());
        for (size_t i = 0; i < sum.size(); ++i) sum[i] = dq_base[i] + dq_res[i];
        const float e_base = RelL2(dq_base, src);
        const float e_res = RelL2(sum, src);
        INFO(base << ": base relL2=" << e_base << " base+res=" << e_res);
        CHECK(e_base > 0.0f);
        CHECK(e_res < e_base * 0.5f);
    }
    remove_all(tmp);
}
