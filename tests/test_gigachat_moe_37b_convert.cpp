// TDD — GigaChat4-37B-A4B converter (Model Ingestion Protocol stage 1/2).
// Synthetic checkpoint only: the real model is never converted in a test; the
// bounded ctest TIMEOUT is the mechanical guard against that.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"

#include "lsxcommon/gigachat_moe_37b_convert.h"
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat4;
namespace fs = std::filesystem;

TEST_CASE("gigachat4 tensor classifier", "[gigachat4][convert]")
{
    auto k = [](const char *n) { return classify_gigachat4_tensor(n).kind; };
    CHECK(k("model.embed_tokens.weight") == Gc4TensorKind::Embed);
    CHECK(k("lm_head.weight") == Gc4TensorKind::LmHead);
    CHECK(k("model.norm.weight") == Gc4TensorKind::FinalNormWeight);
    CHECK(k("model.norm.gate_up_projection.weight") ==
          Gc4TensorKind::FinalNormGateUp);
    CHECK(k("model.layers.3.input_layernorm.weight") ==
          Gc4TensorKind::LayerNormWeight);
    CHECK(k("model.layers.7.post_feedforward_layernorm.gate_down_projection.weight") ==
          Gc4TensorKind::LayerNormGateDown);
    CHECK(k("model.layers.3.self_attn.q_a_proj.weight") ==
          Gc4TensorKind::QaProj);
    CHECK(k("model.layers.3.self_attn.kv_a_proj_with_mqa.weight") ==
          Gc4TensorKind::KvaProj);
    CHECK(k("model.layers.3.self_attn.gate_proj.weight") ==
          Gc4TensorKind::AttnGateProj);
    // DSA indexer — the tensors the 3.5 sibling does not have.
    CHECK(k("model.layers.3.self_attn.indexer.wq_b.weight") ==
          Gc4TensorKind::IndexWqB);
    CHECK(k("model.layers.3.self_attn.indexer.wk.weight") ==
          Gc4TensorKind::IndexWk);
    CHECK(k("model.layers.3.self_attn.indexer.weights_proj.weight") ==
          Gc4TensorKind::IndexWeightsProj);
    CHECK(k("model.layers.3.self_attn.indexer.k_norm.weight") ==
          Gc4TensorKind::IndexKNorm);
    CHECK(k("model.layers.4.self_attn.in_proj_qkvz.weight") ==
          Gc4TensorKind::GdnInProjQkvz);
    CHECK(k("model.layers.4.self_attn.in_proj_ba.weight") ==
          Gc4TensorKind::GdnInProjBa);
    CHECK(k("model.layers.4.self_attn.conv1d.weight") ==
          Gc4TensorKind::GdnConv1d);
    CHECK(k("model.layers.4.self_attn.A_log") == Gc4TensorKind::GdnALog);
    CHECK(k("model.layers.4.self_attn.norm.weight") ==
          Gc4TensorKind::GdnNormWeight);
    CHECK(k("model.layers.5.mlp.gate.weight") == Gc4TensorKind::RouterWeight);
    CHECK(k("model.layers.5.mlp.gate.e_score_correction_bias") ==
          Gc4TensorKind::RouterBias);
    CHECK(k("model.layers.5.mlp.shared_experts.gate_proj.weight") ==
          Gc4TensorKind::SharedGate);
    CHECK(k("model.layers.5.mlp.experts.100.up_proj.weight") ==
          Gc4TensorKind::ExpertUp);
    CHECK(k("model.layers.0.mlp.down_proj.weight") == Gc4TensorKind::MlpDown);
    CHECK(k("model.layers.32.eh_proj.weight") == Gc4TensorKind::Mtp);
    CHECK(k("model.layers.32.shared_head.head.weight") == Gc4TensorKind::Mtp);
    CHECK(k("garbage") == Gc4TensorKind::Unknown);

    auto e = classify_gigachat4_tensor("model.layers.5.mlp.experts.100.up_proj.weight");
    CHECK(e.layer == 5);
    CHECK(e.expert == 100);

    // Linears that carry a `_scale_inv` sibling and BF16 linears both become
    // NVFP4; norms / router / scalars never do.
    CHECK(gigachat4_tensor_is_nvfp4(Gc4TensorKind::IndexWqB));
    CHECK(gigachat4_tensor_is_nvfp4(Gc4TensorKind::SharedGate));
    CHECK(gigachat4_tensor_is_nvfp4(Gc4TensorKind::MlpDown));
    CHECK_FALSE(gigachat4_tensor_is_nvfp4(Gc4TensorKind::IndexKNorm));
    CHECK_FALSE(gigachat4_tensor_is_nvfp4(Gc4TensorKind::RouterWeight));
    CHECK_FALSE(gigachat4_tensor_is_nvfp4(Gc4TensorKind::GdnConv1d));
}

TEST_CASE("gigachat4 config parser reads the authoritative fields",
          "[gigachat4][convert]")
{
    const char *cfg = R"({
      "hidden_size": 2560, "num_hidden_layers": 32, "vocab_size": 128256,
      "num_attention_heads": 32, "num_key_value_heads": 32,
      "n_routed_experts": 192, "num_experts_per_tok": 8,
      "moe_intermediate_size": 768, "intermediate_size": 8192,
      "first_k_dense_replace": 1, "rms_norm_eps": 1e-06,
      "routed_scaling_factor": 2.5, "swiglu_limit": 10,
      "final_logit_softcapping": 30, "q_lora_rank": 1536,
      "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
      "v_head_dim": 128, "index_n_heads": 64, "index_head_dim": 128,
      "index_topk": 2048, "full_attention_layers": [3,7,11,15,19,23,27,31],
      "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2
    })";
    auto t = ParseGigaChat4Config(cfg);
    CHECK(t.error_msg.empty());
    CHECK(t.hidden == 2560);
    CHECK(t.layers == 32);
    CHECK(t.vocab == 128256);
    CHECK(t.num_experts == 192);
    CHECK(t.top_k == 8);
    CHECK(t.dense_intermediate == 8192);
    CHECK(t.q_lora == 1536);
    CHECK(t.kv_lora == 512);
    CHECK(t.index_heads == 64);
    CHECK(t.index_topk == 2048);
    CHECK(t.logit_softcap == 30.0);
    REQUIRE(t.full_attention_layers.size() == 8);
    CHECK(t.full_attention_layers[0] == 3);
    CHECK(t.full_attention_layers[7] == 31);

    auto bad = ParseGigaChat4Config("{ not json ");
    CHECK_FALSE(bad.error_msg.empty());
}

TEST_CASE("gigachat4 converter emits the logestix-gigachat4-nvfp4 layout",
          "[gigachat4][convert]")
{
    const auto src = gigachat4test::MakeTempDir("gc4_src");
    const auto dst = gigachat4test::MakeTempDir("gc4_dst");
    gigachat4test::WriteGigaChat4Checkpoint(src);
    gigachattest::WriteTokenizerJson(src);

    auto st = ConvertGigaChat4Model(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    std::ifstream in(dst / "model.index.json");
    REQUIRE(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string json = ss.str();

    CHECK(json.find("\"layout\": 2") !=
          std::string::npos);
    CHECK(json.find("\"full_attention_layers\": [1]") != std::string::npos);
    CHECK(json.find("\"index_topk\": 2") != std::string::npos);
    CHECK(json.find("\"index_heads\": 2") != std::string::npos);
    CHECK(json.find("\"logit_softcap\": 30") != std::string::npos);
    CHECK(json.find("\"mtp_layer\": 2") != std::string::npos);
    CHECK(json.find("\"dflash2\"") != std::string::npos);
    CHECK(json.find("\"total_experts\": 2") != std::string::npos);

    // Dense/pinned output.
    CHECK(fs::exists(dst / "pinned" / "global.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_000_nvfp4.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_001_nvfp4.arrow"));
    // BF16 dense MLP and the BF16 attention gate must still land in the NVFP4
    // batch (they are quantized, not passed through raw).
    CHECK(fs::exists(dst / "pinned" / "layer_000.arrow"));
    // Routed experts: one file per expert, all three projections.
    CHECK(fs::exists(dst / "experts" / "layer_1" / "exp_0.arrow"));
    CHECK(fs::exists(dst / "experts" / "layer_1" / "exp_1.arrow"));
    // MTP layer is counted, never applied — and in v1 never written: a plane
    // the load path cannot reach makes the whole artifact unpublishable.
    CHECK_FALSE(fs::exists(dst / "mtp"));

    // Tokenizer bridge (stage 4): byte-level BPE of the Llama-3 case-run kind.
    CHECK(json.find("\"tokenizer\": \"arrow-bpe\"") != std::string::npos);
    CHECK(fs::exists(dst / "pinned" / "tokenizer_vocab.arrow"));
    CHECK(fs::exists(dst / "pinned" / "tokenizer_merges.arrow"));
    auto vf = arrow::io::MemoryMappedFile::Open(
        (dst / "pinned" / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (dst / "pinned" / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    lsxcommon::ArrowTokenizerDb db;
    auto meta = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
    INFO(meta.ToString());
    REQUIRE(meta.ok());
    CHECK(db.tokenizer_kind == 3);
    CHECK(db.eos_id == 2);
    CHECK(db.pad_id == 2);
}

TEST_CASE("gigachat4 converter rejects an unclassified tensor",
          "[gigachat4][convert]")
{
    const auto src = gigachat4test::MakeTempDir("gc4_bad_src");
    const auto dst = gigachat4test::MakeTempDir("gc4_bad_dst");
    // A leftover tensor must fail the conversion instead of being skipped.
    gigachat4test::WriteGigaChat4Checkpoint(src, /*verified=*/false,
                                            /*extra_unknown=*/true);
    gigachattest::WriteTokenizerJson(src);
    auto st = ConvertGigaChat4Model(src.string(), dst.string());
    CHECK_FALSE(st.ok());
}
