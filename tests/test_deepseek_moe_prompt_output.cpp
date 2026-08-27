// TDD — DeepSeek-V4-Flash meaningful prompt output.
// Real-model test: SKIPs when no DeepSeek checkpoint is present (CTest
// SKIP_RETURN_CODE 4, same as the other real-weight tests).
//
// Contract under test (DeepSeekV4FlashModel::Infer):
//   - the raw "The capital of France is" must generate a non-empty
//     tokenizer-decoded output_text,
//   - the FIRST generated token is the argmax at step prompt_len-1 (the last
//     prompt token); the output decode must include it, so the trimmed
//     output_text starts with "Paris",
//   - r.ok == true && !r.gen_tokens.empty().
#include <catch2/catch_test_macros.hpp>

#include <absl/strings/ascii.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace {

std::filesystem::path DeepSeekModelPath()
{
    const char *env = std::getenv("LSX_TEST_DEEPSEEK_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "DeepSeek-V4-Flash";
}

// Encodes the shared "capital of France" self-check prompt with the raw BPE
// encode (DeepSeek consumes the raw prompt; no chat-template framing).
std::vector<int32_t> EncodePrompt(lsxcommon::ArrowTokenizerDb &tok)
{
    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE(
    "DeepSeek-V4-Flash Infer: output_text starts with 'Paris' on the test prompt",
    "[deepseek][model][prompt][real-model]")
{
    const std::filesystem::path model = DeepSeekModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("DeepSeek real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "deepseek-v4-flash");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodePrompt(m->Tokenizer());
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 8;

    auto r = m->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());
    REQUIRE(!r.output_text.empty());

    std::string trimmed = r.output_text;
    absl::StripAsciiWhitespace(&trimmed);
    REQUIRE(trimmed.rfind("Paris", 0) == 0);

    // Unified runtime contract (Task 6): timings, occupancy, token count.
    REQUIRE(r.total_time_ms >= r.ttft_ms);
    REQUIRE(r.kv_slots_used > 0);
    REQUIRE(r.num_tokens == static_cast<int>(r.gen_tokens.size()));

    m->Unload();
}
