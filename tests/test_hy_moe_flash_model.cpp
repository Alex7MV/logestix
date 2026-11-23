// Hy4-flash model stage: factory resolution, the frozen-geometry and format
// gates, and the request guards — all host-only (label `fast`), so they run
// without the 90 GB artifact.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"

namespace fs = std::filesystem;

namespace {

fs::path WriteIndex(int layers, const std::string &format,
                    const std::string &model)
{
    const fs::path root =
        fs::temp_directory_path() /
        fs::path("hy4-model-" + std::to_string(::getpid()) + "-" +
                 std::to_string(layers) + "-" + std::to_string(format.size()));
    fs::remove_all(root);
    fs::create_directories(root / "logestix" / "pinned");
    std::ofstream f(root / "logestix" / "model.index.json");
    f << "{\"format\": \"" << format << "\", \"model\": \"" << model
      << "\", \"num_layers\": " << layers
      << ", \"experts_per_layer\": 512, \"total_experts\": 20992,"
      << " \"total_pinned_tensors\": 0, \"hidden_size\": 3072,"
      << " \"vocab_size\": 120832, \"num_experts_per_tok\": 16,"
      << " \"moe_intermediate_size\": 1536, \"latent_size\": 1536,"
      << " \"shared_intermediate_size\": 1536, \"dense_intermediate_size\": 13824,"
      << " \"num_attention_heads\": 32, \"swa_num_attention_heads\": 64,"
      << " \"q_lora_rank\": 1024, \"kv_lora_rank\": 512,"
      << " \"qk_nope_head_dim\": 192, \"qk_rope_head_dim\": 64,"
      << " \"v_head_dim\": 256, \"sliding_window\": 512, \"hc_mult\": 4,"
      << " \"index_n_heads\": 16, \"index_head_dim\": 128, \"index_topk\": 2048,"
      << " \"bos_token_id\": 120000, \"eos_token_id\": 120025,"
      << " \"pad_token_id\": 120002,"
      << " \"vram\": {\"pinned_bytes\": 0, \"expert_bytes\": 0, \"kv_bytes\": 0,"
      << " \"workspace_bytes\": 0, \"total_bytes\": 0},"
      << " \"deferred\": {\"engram\": 28, \"vision\": 266, \"dfly\": 74},"
      << " \"layers\": [";
    for (int l = 0; l < layers; ++l) {
        const bool dsa = l <= 1 || l % 4 == 1;
        f << "{\"is_dsa\": " << (dsa ? "true" : "false") << ", \"heads\": "
          << (dsa ? 32 : 64) << ", \"is_moe\": " << (l > 0 ? "true" : "false")
          << ", \"has_engram\": false}";
        if (l + 1 < layers) f << ",";
    }
    f << "]}\n";
    return root;
}

} // namespace

TEST_CASE("hy4 model: the factory resolves the registry id", "[hy4][fast]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "hy4-flash";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "hy4-flash");
    // The MLA latent cache is pos-major: one latent row per position.
    CHECK(model->KvLayout() == lsxcommon::KvCacheLayout::kPosMajor);
}

TEST_CASE("hy4 model: Infer before Load is refused, not crashed", "[hy4][fast]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "hy4-flash";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2, 3};
    const auto r = model->Infer(req);
    CHECK(!r.ok);
    CHECK(!r.error_msg.empty());
}

TEST_CASE("hy4 model: Load rejects a foreign format and a foreign card",
          "[hy4][fast]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "hy4-flash";

    const fs::path wrong_format = WriteIndex(42, "logestix-hy-nvfp4", "Hy4-flash");
    cfg.model_path = wrong_format.string();
    auto m1 = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m1 != nullptr);
    CHECK(!m1->Load());

    const fs::path wrong_card = WriteIndex(42, "logestix-hy4-nvfp4", "Hy-Memory-30B-A3B");
    cfg.model_path = wrong_card.string();
    auto m2 = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m2 != nullptr);
    CHECK(!m2->Load());

    const fs::path wrong_layers = WriteIndex(41, "logestix-hy4-nvfp4", "Hy4-flash");
    cfg.model_path = wrong_layers.string();
    auto m3 = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m3 != nullptr);
    CHECK(!m3->Load());

    fs::remove_all(wrong_format);
    fs::remove_all(wrong_card);
    fs::remove_all(wrong_layers);
}

TEST_CASE("hy4 model: a pre-set abort flag yields no tokens", "[hy4][fast]")
{
    // The flag is checked between steps; with no loaded model the request is
    // refused first, which is the contract this case pins.
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "hy4-flash";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    std::atomic<bool> abort{true};
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2, 3};
    req.abort_flag = &abort;
    const auto r = model->Infer(req);
    CHECK(!r.ok);
    CHECK(r.gen_tokens.empty());
}

// Temporary diagnostic (tagged so it is never part of the default run): prints
// what the converted tree actually answers, which is the only way to tell a
// semantic bug from a numeric one when no reference implementation exists.
TEST_CASE("hy4 model: control-prompt diagnostic", "[.hy4diag]")
{
    const char *env = std::getenv("LSX_TEST_HY4_FLASH_ROOT");
    const fs::path root = env ? fs::path(env)
                              : fs::path("/mnt/fast_storage/huggingface/Hy4-flash");
    if (!fs::exists(root / "logestix" / "model.index.json"))
        SKIP("converted Hy4-flash tree absent");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "hy4-flash";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    std::vector<int32_t> ids(256, 0);
    size_t cap = ids.size();
    const char *prompt = "The capital of France is";
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        model->Tokenizer(), prompt, std::strlen(prompt), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 8;
    const auto r = model->Infer(req);
    std::string toks;
    for (auto t : r.gen_tokens) toks += std::to_string(t) + " ";
    WARN("ok=" << r.ok << " err='" << r.error_msg << "' first_tok="
               << r.first_gen_tok_id << " maxv=" << r.first_gen_maxv
               << " tokens=[" << toks << "] text='" << r.output_text << "'");
    model->Unload();
}
