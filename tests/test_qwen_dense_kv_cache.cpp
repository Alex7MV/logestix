#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "test_manifest_query_root.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/qwen_base_model.h"

using namespace lsxcommon;

TEST_CASE("Qwen incremental KV continuation equals a single full run",
          "[qwen][realmodel]")
{
    ModelInitConfig cfg;
    cfg.model_path = manifest_query::ConvertedDirFor("qwen3.5-9b");
    auto model = LsxModelFactory::Create(cfg);
    if (!model || !model->Load()) {
        SKIP("Qwen checkpoint not available");
    }

    std::vector<int32_t> input_ids;
    REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                       "The capital of France is",
                                       input_ids));
    REQUIRE_FALSE(input_ids.empty());

    // Baseline: a single 8-token run (full prefill + incremental decode).
    ModelRequest single_req;
    single_req.input_ids = input_ids;
    single_req.max_gen_tokens = 8;
    auto single = model->Infer(single_req);
    REQUIRE(single.ok);
    REQUIRE(single.num_tokens == 8);

    // Session split: 3 + 5 generated tokens across two Infer calls.
    ModelRequest a;
    a.input_ids = input_ids;
    a.max_gen_tokens = 3;
    auto first = model->Infer(a);
    REQUIRE(first.ok);
    REQUIRE(first.num_tokens == 3);

    ModelRequest b;
    b.input_ids = first.gen_tokens;
    b.max_gen_tokens = 5;
    auto second = model->Infer(b);
    REQUIRE(second.ok);
    REQUIRE(second.num_tokens == 5);

    std::vector<int32_t> joined = first.gen_tokens;
    joined.insert(joined.end(), second.gen_tokens.begin(),
                  second.gen_tokens.end());

    // The 8-token single run and the 3+5 incremental continuation must agree
    // on the first 8 generated tokens (same greedy path, bit-exact).
    REQUIRE(joined.size() >= single.gen_tokens.size());
    REQUIRE(std::equal(single.gen_tokens.begin(), single.gen_tokens.end(),
                       joined.begin()));

    // Determinism: a second independent split reproduces the same session.
    ModelRequest a2;
    a2.input_ids = input_ids;
    a2.max_gen_tokens = 3;
    auto first2 = model->Infer(a2);
    REQUIRE(first2.ok);
    REQUIRE(first2.gen_tokens == first.gen_tokens);

    ModelRequest b2;
    b2.input_ids = first2.gen_tokens;
    b2.max_gen_tokens = 5;
    auto second2 = model->Infer(b2);
    REQUIRE(second2.ok);
    REQUIRE(second2.gen_tokens == second.gen_tokens);

    model->Unload();
}
