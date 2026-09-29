#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "test_qwen_moe_exp_fixture.h"
#include "lsxcommon/qwen_moe_exp_convert.h"

#include <filesystem>

using namespace lsxcommon::qwen_moe_exp;
namespace fs = std::filesystem;

TEST_CASE("ParseQwen4ExpConfig reads qwen4_exp fields", "[qwen-exp-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_qwen_exp_cfg";
    fs::create_directories(dir);
    qwen_moe_exp_test::WriteExpConfig(dir);
    auto top = ParseQwen4ExpConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = qwen_moe_exp_test::ExpCfg();
    REQUIRE(top.hidden_size == c.hidden_size);
    REQUIRE(top.num_layers == c.num_layers);
    REQUIRE(top.num_experts == c.num_experts);
    REQUIRE(top.num_experts_per_tok == c.num_experts_per_tok);
    REQUIRE(top.moe_intermediate_size == c.moe_intermediate_size);
    REQUIRE(top.shared_expert_intermediate_size ==
            c.shared_expert_intermediate_size);
    REQUIRE(top.hc_count == c.hc_count);
    REQUIRE(top.hc_lowrank == c.hc_lowrank);
    REQUIRE(top.indexer_n_heads == c.indexer_n_heads);
    REQUIRE(top.indexer_budget == c.indexer_budget);
    REQUIRE(top.indexer_compress_ratio == c.indexer_compress_ratio);
    REQUIRE(top.indexer_head_dim == c.indexer_head_dim);
    REQUIRE(top.ngram_size == c.ngram_size);
    REQUIRE(top.heads_per_ngram == c.heads_per_ngram);
    REQUIRE(top.ple_embed_dim == c.ple_embed_dim);
    REQUIRE(top.split_ngram_parts == c.split_ngram_parts);
    REQUIRE(top.ngram_vocab_size_base == c.ngram_vocab_size_base);
    REQUIRE(top.ple_layer_ids == std::vector<int>{2});
    REQUIRE(top.eos_token_id == c.eos_token_id);
    REQUIRE(top.rms_norm_eps == Catch::Approx(1e-6));
    REQUIRE(top.partial_rotary_factor == Catch::Approx(0.25));
    REQUIRE(top.rope_theta == Catch::Approx(10000.0));
    REQUIRE(top.layer_types.size() == 4);
    REQUIRE(top.layer_types[3] == "full_attention");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("ParseQwen4ExpConfig fails on missing config", "[qwen-exp-convert]")
{
    auto top = ParseQwen4ExpConfig("/nonexistent_dir_lsx_qwen_exp");
    REQUIRE_FALSE(top.error_msg.empty());
}

TEST_CASE("qwen4_exp tensor classification", "[qwen-exp-convert]")
{
    using K = ExpTensorKind;
    auto cl = [](const std::string &n) {
        return classify_qwen4_exp_tensor(n);
    };
    CHECK(cl("model.language_model.embed_tokens.weight").kind == K::Embed);
    CHECK(cl("lm_head.weight").kind == K::LmHead);
    CHECK(cl("model.language_model.hyper_connection_mixer.hc_norm.weight")
              .kind == K::GlobalMixerNorm);
    CHECK(cl("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight")
              .kind == K::GlobalMixerDown);
    CHECK(cl("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight")
              .kind == K::GlobalMixerUp);
    CHECK(cl("model.language_model.layers.3.self_attn.q_proj.weight").kind ==
          K::QsaQ);
    CHECK(cl("model.language_model.layers.3.self_attn.q_proj.weight").layer == 3);
    CHECK(cl("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight")
              .kind == K::IndexerQkProj);
    CHECK(cl("model.language_model.layers.3.self_attn.indexer.q_layernorm.weight")
              .kind == K::IndexerQNorm);
    CHECK(cl("model.language_model.layers.3.self_attn.indexer.k_layernorm.weight")
              .kind == K::IndexerKNorm);
    CHECK(cl("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
              .kind == K::GdnQkv);
    CHECK(cl("model.language_model.layers.0.linear_attn.A_log").kind ==
          K::GdnALog);
    CHECK(cl("model.language_model.layers.0.linear_attn.dt_bias").kind ==
          K::GdnDt);
    CHECK(cl("model.language_model.layers.0.attn_hyper_connection.hc_norm.weight")
              .kind == K::AttnHcNorm);
    CHECK(cl("model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight")
              .kind == K::AttnHcBlockInject);
    CHECK(cl("model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight")
              .kind == K::MlpHcNorm);
    CHECK(cl("model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight")
              .kind == K::MlpHcDown);
    CHECK(cl("model.language_model.layers.0.mlp_hyper_connection.block_inject_weight.weight")
              .kind == K::MlpHcBlockInject);
    CHECK(cl("model.language_model.layers.1.ple.key_proj.weight").kind ==
          K::PleKeyProj);
    CHECK(cl("model.language_model.layers.1.ple.value_proj.weight").kind ==
          K::PleValueProj);
    CHECK(cl("model.language_model.layers.1.ple.norm_key.weight").kind ==
          K::PleNormKey);
    CHECK(cl("model.language_model.layers.1.ple.norm_conv.weight").kind ==
          K::PleNormConv);
    CHECK(cl("model.language_model.layers.1.ple.conv1d.weight").kind ==
          K::PleConv);
    CHECK(cl("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
              .kind == K::PleNgramShard);
    CHECK(cl("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
              .shard == 0);
    CHECK(cl("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_17.weight")
              .shard == 17);
    CHECK(cl("model.language_model.layers.1.ple.ple_embedding.layer_multipliers")
              .kind == K::PleMeta);
    CHECK(cl("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets")
              .kind == K::PleMeta);
    CHECK(cl("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes")
              .kind == K::PleMeta);
    CHECK(cl("model.language_model.layers.0.mlp.gate.weight").kind ==
          K::RouterGate);
    CHECK(cl("model.language_model.layers.0.mlp.experts.gate_up_proj").kind ==
          K::FusedExpertsGateUp);
    CHECK(cl("model.language_model.layers.0.mlp.experts.down_proj").kind ==
          K::FusedExpertsDown);
    CHECK(cl("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight")
              .kind == K::SharedGateProj);
    CHECK(cl("model.language_model.layers.0.mlp.shared_expert.up_proj.weight")
              .kind == K::SharedUpProj);
    CHECK(cl("model.language_model.layers.0.mlp.shared_expert.down_proj.weight")
              .kind == K::SharedDownProj);
    CHECK(cl("model.language_model.layers.0.mlp.shared_expert_gate.weight")
              .kind == K::SharedGate);
    CHECK(cl("mtp.layers.0.mlp.gate.weight").kind == K::Mtp);
    CHECK(cl("mtp.fc.weight").kind == K::Mtp);
    CHECK(cl("model.visual.blocks.0.attn.qkv.weight").kind == K::Visual);
    CHECK(cl("visual.blocks.0.norm1.weight").kind == K::Visual);
    CHECK(cl("model.language_model.layers.0.weird.weight").kind == K::Unknown);
    CHECK(cl("totally.unknown").kind == K::Unknown);
}

TEST_CASE("qwen4_exp classifier tolerates malformed layer indices",
          "[qwen-exp-convert]")
{
    // A malformed tensor name (empty/non-numeric layer index) must classify as
    // Unknown, never throw std::invalid_argument out of the converter.
    CHECK_NOTHROW(classify_qwen4_exp_tensor(
        "model.language_model.layers..self_attn.q_proj.weight"));
    CHECK_NOTHROW(classify_qwen4_exp_tensor(
        "model.language_model.layers.self_attn.q_proj.weight"));
    CHECK(classify_qwen4_exp_tensor(
              "model.language_model.layers..self_attn.q_proj.weight").kind ==
          ExpTensorKind::Unknown);
    CHECK(classify_qwen4_exp_tensor(
              "model.language_model.layers.self_attn.q_proj.weight").kind ==
          ExpTensorKind::Unknown);

    // A PLE n-gram shard index with an implausible magnitude must classify as
    // Unknown, not overflow/throw std::out_of_range out of parse_ple_shard_index.
    const std::string huge_shard =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        "shard_99999999999999999999.weight";
    CHECK_NOTHROW(classify_qwen4_exp_tensor(huge_shard));
    CHECK(classify_qwen4_exp_tensor(huge_shard).kind == ExpTensorKind::Unknown);
}

TEST_CASE("qwen4_exp NVFP4 policy", "[qwen-exp-convert]")
{
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::QsaQ));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::QsaK));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::QsaV));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::QsaO));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::IndexerQkProj));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnQkv));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnZ));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnA));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnB));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnOut));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::AttnHcDown));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::AttnHcUp));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::AttnHcBlockInject));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::MlpHcDown));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::MlpHcUp));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::MlpHcBlockInject));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GlobalMixerDown));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GlobalMixerUp));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::FusedExpertsGateUp));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::FusedExpertsDown));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::SharedGateProj));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::SharedUpProj));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::SharedDownProj));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleKeyProj));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleValueProj));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::Embed));
    CHECK(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::LmHead));

    // Invariants: norms, router, scalar gate, A_log/dt_bias, PLE norms,
    // indexer layernorms stay high precision.
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::AttnHcNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::MlpHcNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GlobalMixerNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::RouterGate));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::SharedGate));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnALog));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnDt));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnConv));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::GdnNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::QsaQNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::QsaKNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::IndexerQNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::IndexerKNorm));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleNormKey));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleNormQuery));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleNormConv));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleConv));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleNgramShard));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::PleMeta));
    CHECK_FALSE(qwen4_exp_tensor_is_nvfp4(ExpTensorKind::FinalNorm));
}

TEST_CASE("splitmix64 multipliers bit-exact vs HF reference", "[qwen-exp-convert]")
{
    // Ground truth: stored in checkpoint layer_multipliers, verified 2026-09-02
    // against _build_layer_multipliers(seed=1234, ple_index=0, vocab=248320).
    int64_t m[3];
    Splitmix64Multipliers(1234, 0, 248320, 3, m);
    CHECK(m[0] == 23703573157769LL);
    CHECK(m[1] == 20109073645365LL);
    CHECK(m[2] == 8052911324071LL);
}

TEST_CASE("prime table matches checkpoint heads", "[qwen-exp-convert]")
{
    // Ground truth: ngram_heads_vocab_sizes [16] from checkpoint shard 37.
    CHECK(FindNthPrimeAfter(20000000 - 1, 1) == 20000003);
    CHECK(FindNthPrimeAfter(20000000 - 1, 2) == 20000023);
    CHECK(FindNthPrimeAfter(20000000 - 1, 3) == 20000033);
    CHECK(FindNthPrimeAfter(20000000 - 1, 4) == 20000047);
    CHECK(FindNthPrimeAfter(20000000 - 1, 8) == 20000077);
    CHECK(FindNthPrimeAfter(20000000 - 1, 16) == 20000171);
}

TEST_CASE("synthetic checkpoint conversion produces logestix layout",
          "[qwen-exp-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_qwen_exp_convert";
    fs::path out = fs::temp_directory_path() / "lsx_qwen_exp_convert_out";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::remove_all(out, ec);
    qwen_moe_exp_test::ExpSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const int written = qwen_moe_exp_test::BuildSyntheticExpCheckpoint(dir);
    REQUIRE(written > 0);
    auto st = ConvertQwen38FlashNextModel(dir.string(), out.string());
    REQUIRE(st.ok());
    REQUIRE(fs::exists(out / "logestix" / "model.index.json"));
    REQUIRE(fs::exists(out / "logestix" / "pinned" / "global_nvfp4.arrow"));
    REQUIRE(fs::exists(out / "logestix" / "pinned" / "layer_0_nvfp4.arrow"));
    // hc records live in the per-layer dense bundles (layer_0.arrow raw +
    // layer_0_nvfp4.arrow quantized); global mixer bundle present.
    REQUIRE(fs::exists(out / "logestix" / "ple" / "meta.arrow"));
    REQUIRE(fs::exists(out / "logestix" / "ple" / "shard_0.arrow"));
    // experts layout per layer
    REQUIRE(fs::exists(out / "logestix" / "experts" / "layer_0" / "exp_0.arrow"));
    REQUIRE(fs::exists(out / "logestix" / "experts" / "layer_3" / "exp_7.arrow"));
    // router + shared
    REQUIRE(fs::exists(out / "logestix" / "router" / "layer_0.arrow"));
    REQUIRE(fs::exists(out / "logestix" / "shared" / "layer_0_nvfp4.arrow"));
    // mtp subtree stored
    REQUIRE(fs::exists(out / "logestix" / "mtp"));
    qwen_moe_exp_test::ExpUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    fs::remove_all(dir, ec);
    fs::remove_all(out, ec);
}

TEST_CASE("real-model conversion hard-fails on synthetic geometry",
          "[qwen-exp-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_qwen_exp_gate";
    fs::path out = fs::temp_directory_path() / "lsx_qwen_exp_gate_out";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::remove_all(out, ec);
    qwen_moe_exp_test::BuildSyntheticExpCheckpoint(dir);
    // Frozen gate must reject the tiny fixture: clear the escape hatch
    // explicitly (ctest presets it for the synthetic cases; Catch2 shuffles
    // case order, so this case cannot rely on another case's cleanup).
    qwen_moe_exp_test::ExpUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    auto st = ConvertQwen38FlashNextModel(dir.string(), out.string());
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("frozen geometry") != std::string::npos);
    fs::remove_all(dir, ec);
    fs::remove_all(out, ec);
}

#include "qwen_moe_exp_device_math.h"

#include <cmath>
#include <random>
#include <vector>

TEST_CASE("hostref GatedResidual finite and sane", "[qwen-exp-hostref]")
{
    namespace hr = lsxcommon::qwen_moe_exp::hostref;
    const int H = 2, hc = 2, lr = 2, n = H * hc;
    std::vector<float> x(n), down(lr * n, 0.5f), up(n * lr, 0.5f),
        inject(hc * n, 0.1f);
    for (int i = 0; i < n; ++i) x[i] = 0.25f * (i + 1);
    std::vector<float> mixed(H), inj(hc);
    hr::GatedResidualHost(x.data(), down.data(), up.data(), inject.data(),
                          H, hc, lr, 1e-6, mixed.data(), inj.data());
    for (float v : mixed) REQUIRE(std::isfinite(v));
    for (float v : inj) REQUIRE(v > 0.f);
    // Zero-norm edge: all-zero input -> rsqrt(0 + eps) path stays finite.
    std::fill(x.begin(), x.end(), 0.f);
    hr::GatedResidualHost(x.data(), down.data(), up.data(), inject.data(),
                          H, hc, lr, 1e-6, mixed.data(), inj.data());
    for (float v : mixed) REQUIRE(std::isfinite(v));
    for (float v : inj) REQUIRE(std::isfinite(v));
}

TEST_CASE("hostref PLE gate matches HF formula", "[qwen-exp-hostref]")
{
    namespace hr = lsxcommon::qwen_moe_exp::hostref;
    const int H = 2, hc = 2;
    std::vector<float> k(H * hc, 1.f), q(H * hc, 1.f);
    // s = hc * H / sqrt(H) = sqrt(H) * hc > 0 -> positive gate.
    const float g = hr::PleGateHost(k.data(), q.data(), H, hc);
    REQUIRE(g > 0.f);
    k[0] = -1.f;  // flip one stream contribution negative
    const float g2 = hr::PleGateHost(k.data(), q.data(), H, hc);
    REQUIRE(std::isfinite(g2));
    // All-negative sum -> negative gate sign preserved.
    std::fill(k.begin(), k.end(), -1.f);
    const float g3 = hr::PleGateHost(k.data(), q.data(), H, hc);
    REQUIRE(g3 < 0.f);
}

TEST_CASE("hostref GatedResidual matches closed-form hand computation",
          "[qwen-exp-hostref]")
{
    namespace hr = lsxcommon::qwen_moe_exp::hostref;
    // Deterministic tiny case verified by an independent derivation:
    // H=1, hc=2, lr=1. hc-norm group_size == H == 1 -> per-element RMSNorm:
    // normed_i = x_i * rsqrt(x_i^2 + eps) = sign(x_i) (magnitude ~1).
    const double eps = 1e-6;
    const float x[2] = {3.f, 4.f};
    const float down[2] = {0.5f, 1.5f};
    const float up[2] = {2.f, -1.f};
    const float n0 = static_cast<float>(3.0 / std::sqrt(9.0 + eps));
    const float n1 = static_cast<float>(4.0 / std::sqrt(16.0 + eps));
    const double v = (0.5 * n0 + 1.5 * n1) / 2.0;          // acc / hc
    const double silu = v / (1.0 + std::exp(-v));          // silu(v)
    const float g0 = 1.f / (1.f + std::exp(-static_cast<float>(up[0] * silu)));
    const float g1 = 1.f / (1.f + std::exp(-static_cast<float>(up[1] * silu)));
    const float want_mixed = 0.5f * (g0 * n0 + g1 * n1);
    // inject weights w_i = {0.1, 0.2} (rows over 2 elems)
    const float inject[4] = {0.1f, 0.2f, 0.1f, 0.2f};
    const double s0 = (0.1 * n0 + 0.2 * n1) / 2.0;
    const double s1 = s0;  // same weights per branch in this construction
    const float want_i0 = static_cast<float>(2.0 / (1.0 + std::exp(-s0)));
    const float want_i1 = static_cast<float>(2.0 / (1.0 + std::exp(-s1)));

    float mixed = 0.f, inj[2] = {0.f, 0.f};
    hr::GatedResidualHost(x, down, up, inject, /*H=*/1, /*hc=*/2, /*lr=*/1,
                          eps, &mixed, inj);
    REQUIRE(mixed == Catch::Approx(want_mixed).epsilon(1e-5));
    REQUIRE(inj[0] == Catch::Approx(want_i0).epsilon(1e-6));
    REQUIRE(inj[1] == Catch::Approx(want_i1).epsilon(1e-6));
}
