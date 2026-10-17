// OPEN_TASKS §E1c — qwen3.8-flash-next early-EOS gate.
//
// This lives in its OWN binary on purpose. The driver keeps process-global
// generation state (the `static` embed dequant and generation arena in
// qwen_moe_exp_inference.cu, plus the `thread_local` decode scratches), and
// measured 2026-10-17 on the real weights: a request returns a DIFFERENT answer
// depending on which request ran before it in the same process. The identical
// prompt at an identical 64-token budget answers with a bare EOS after 5 tokens
// when it is the first request in the process, and with a 64-token explanation
// when a 2-token request ran first (OPEN_TASKS §E1d). Pinning the expert-cache
// residency and warming the arena at the same budget both fail to remove it.
//
// Putting the gate in its own executable gives it a fresh process, so it tests
// the driver's EOS contract and not the process history. §E1d is the open item.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

#include "qwen_moe_exp_model.h"

namespace {

// Sets an env var for the enclosing scope and restores it on destruction, so a
// failing REQUIRE cannot leak the override into any later code path.
struct ScopedEnv {
    std::string name;
    ScopedEnv(const char *n, const char *v) : name(n)
    {
#ifdef _WIN32
        _putenv_s(n, v);
#else
        setenv(n, v, 1);
#endif
    }
    ~ScopedEnv()
    {
#ifdef _WIN32
        _putenv_s(name.c_str(), "");
#else
        unsetenv(name.c_str());
#endif
    }
};

// Converted root: unified vault convention <model>/logestix.
std::filesystem::path ConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_QWEN38_MODEL_DIR"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "Qwen3.8-Flash-Next";
}

} // namespace

// §E1c: RunExpGeneration used to decode the whole max_gen_tokens budget and
// truncate at the first EOS afterwards, so everything after EOS was computed and
// thrown away and a streaming consumer saw nothing until the budget ran out.
// This prompt reaches EOS on the 5th generated token, so a 64-token budget
// separates "stopped on EOS" from "ran the budget and cut afterwards" cleanly.
TEST_CASE("qwen3.8-flash-next stops on EOS instead of running the budget",
          "[qwen38][model][lifecycle][real-model]")
{
    const std::filesystem::path model_dir = ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Qwen3.8-Flash-Next converted model directory not present");

    // Pin the expert-cache residency so the gate is a pure function of its own
    // inputs. Load sizes the cache from the free VRAM it can see, so without
    // pinning the slot count - and the memory left for the generation arena -
    // depends on the host. Measured on this host: 8192 slots makes even the
    // Paris selfcheck fail, 1024 and 5350 pass.
    const ScopedEnv slots("LSX_QWEN38_EXPERT_CACHE_SLOTS", "1024");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    cfg.model_type = lsxcommon::Qwen38FlashNextModel::kType;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.max_gen_tokens = 64;
    REQUIRE(m->BuildPromptInputIds(m->Tokenizer(), "Say nothing at all.",
                                   req.input_ids));

    std::vector<std::uint32_t> streamed;
    req.token_callback = [&](const lsxcommon::TokenChunk &c) {
        streamed.push_back(c.token_id);
    };

    const auto r = m->Infer(req);
    INFO("error: " << r.error_msg << " gen=" << r.gen_tokens.size()
                   << " eos=" << r.eos_terminated
                   << " kv_slots=" << r.kv_slots_used
                   << " text=[" << r.output_text.substr(0, 200) << "]");
    REQUIRE(r.ok);
    REQUIRE(r.eos_terminated);

    // The decisive assertion, and the reason the gate lives in its own binary:
    // `gen_tokens.size() < max_gen_tokens` CANNOT show the loop stopped early,
    // because the post-loop truncation makes the list short in both cases. It
    // was confirmed to pass with the early break disabled. KV growth is the
    // observable that counts executed steps - one row per step - so an early
    // stop lands near prompt + generated rather than near prompt + budget.
    CHECK(r.kv_slots_used <=
          static_cast<int>(req.input_ids.size() + r.gen_tokens.size()) + 2);

    // The publish contract: EOS is recorded but never streamed, and every token
    // before it was published in order. ExpectStreamMatchesTokens does not
    // apply here - the stream is deliberately one token shorter than gen_tokens.
    REQUIRE(streamed.size() + 1 == r.gen_tokens.size());
    for (std::size_t i = 0; i < streamed.size(); ++i)
        CHECK(streamed[i] == static_cast<std::uint32_t>(r.gen_tokens[i]));

    m->Unload();
}
