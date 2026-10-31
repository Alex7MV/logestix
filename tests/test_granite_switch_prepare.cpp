#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/granite_dense_switch_inference.h>

using namespace lsxcommon::granite_switch;

TEST_CASE("granite-switch prefix scan and token exchange",
          "[granite_switch][prepare]")
{
    GraniteSwitchTopology top;
    top.num_adapters = 2;
    top.adapter_token_ids = {100352, 100353};
    top.adapter_substitute_token_ids = {27, 14711};

    auto p = PrepareGraniteSwitchTokens(top, {5, 100352, 6, 100353, 7, 8}, 0);
    CHECK(p.adapter_ids == std::vector<int32_t>({0, 1, 1, 2, 2, 2}));
    CHECK(p.embed_ids == std::vector<int32_t>({5, 27, 6, 14711, 7, 8}));
    CHECK(p.last_adapter == 2);

    // Decode continuation: the next token inherits the adapter.
    auto c = PrepareGraniteSwitchTokens(top, {100353}, 1);
    CHECK(c.adapter_ids == std::vector<int32_t>({2}));
    CHECK(c.embed_ids == std::vector<int32_t>({14711}));
    CHECK(c.last_adapter == 2);

    // A plain token carries the previous adapter.
    auto d = PrepareGraniteSwitchTokens(top, {42}, 2);
    CHECK(d.adapter_ids == std::vector<int32_t>({2}));
    CHECK(d.embed_ids == std::vector<int32_t>({42}));

    // A control token at position 0 routes itself.
    auto e = PrepareGraniteSwitchTokens(top, {100353, 9}, 0);
    CHECK(e.adapter_ids == std::vector<int32_t>({2, 2}));

    CHECK(GraniteSwitchSubstitute(top, 100352) == 27);
    CHECK(GraniteSwitchSubstitute(top, 100353) == 14711);
    CHECK(GraniteSwitchSubstitute(top, 7) == 7);
}
