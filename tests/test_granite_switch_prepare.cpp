#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/granite_dense_switch_8b_constants.h>
#include <lsxcommon/granite_dense_switch_constants.h>
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

TEST_CASE("granite-switch 8B passport accepts only the 8B tree",
          "[granite_switch][prepare]")
{
    // A sibling that shares a base class is exactly where the parent's numbers
    // get reused by accident. The 8B differs from the 30B in four fields, three
    // of which are arithmetic (theta, residual multiplier) or shape that the
    // kernels take from the loaded topology - a shared constant would decode a
    // plausible but wrong model, not crash.
    const auto &f = lsxcommon::granite_switch_8b::Frozen();
    CHECK(f.model_type == std::string("granite-switch-4.1-8b"));
    CHECK(f.num_layers == 40);
    CHECK(f.intermediate_size == 12800);
    CHECK(f.rope_theta == Catch::Approx(10000000.0));
    // The passport stores 0.22f, so the double compares as 0.2199999988... - the
    // checker compares with a relative tolerance and so must the test.
    CHECK(f.residual_multiplier == Catch::Approx(0.22));

    GraniteSwitchTopology own;          // what the 8B checkpoint emits
    own.model_name = "granite-switch-4.1-8b";
    own.num_layers = 40;
    own.hidden_size = 4096;
    own.num_attention_heads = 32;
    own.num_key_value_heads = 8;
    own.head_dim = 128;
    own.intermediate_size = 12800;
    own.vocab_size = 100355;
    own.num_adapters = 2;
    own.adapter_ranks = {16, 64};
    own.adapter_token_ids = {100352, 100353};
    own.adapter_substitute_token_ids = {27, 14711};
    own.rms_norm_eps = 1e-5;
    own.rope_theta = 10000000.0;
    own.attention_multiplier = 0.0078125;
    own.embedding_multiplier = 12.0;
    own.residual_multiplier = 0.22;
    own.logits_scaling = 16.0;
    CHECK(GraniteSwitchGeometryMismatch(f, own).empty());

    // The 30B tree must be rejected by the 8B passport, and the message must
    // name both sides: "geometry mismatch" alone leaves a reader guessing which
    // of 100 manifests produced the tree they just converted.
    GraniteSwitchTopology big = own;
    big.model_name = "granite-switch-4.1-30b";
    big.num_layers = 64;
    big.intermediate_size = 32768;
    big.rope_theta = 50000000.0;
    big.residual_multiplier = 0.175;
    const std::string why = GraniteSwitchGeometryMismatch(f, big);
    CHECK_FALSE(why.empty());
    CHECK(why.find("granite-switch-4.1-30b") != std::string::npos);
    CHECK(why.find("granite-switch-4.1-8b") != std::string::npos);
    CHECK(why.find("num_layers") != std::string::npos);
    CHECK(why.find("rope_theta") != std::string::npos);
    CHECK(why.find("residual_multiplier") != std::string::npos);

    // Each differing field is reported, not just the first: a re-converted tree
    // that is wrong in two ways must not need two runs to discover.
    CHECK(why.find("intermediate_size") != std::string::npos);
    CHECK(why.find("64") != std::string::npos);

    // The identity comparison is case-insensitive about the documented spelling
    // because the 30B tree on disk was written as "Granite-Switch-4.1-30B" by a
    // converter that predates this check, and detect_model_type already folds
    // both spellings to the same kType.
    GraniteSwitchTopology legacy = own;
    legacy.model_name = "Granite-Switch-4.1-8B";
    CHECK(GraniteSwitchGeometryMismatch(f, legacy).empty());
}
