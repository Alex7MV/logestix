// TDD — fastload shared helpers (host-only: AlignUp + suffix lookup; no CUDA
// allocations, no model weights).
#include <catch2/catch_test_macros.hpp>

#include "fast_load_internal.h"
#include "lsxcommon/schema.h"

TEST_CASE("AlignUp rounds up to alignment", "[fastload]")
{
    using lsxcommon::fastload::AlignUp;
    STATIC_REQUIRE(lsxcommon::fastload::kArenaAlign == 256);
    CHECK(AlignUp(0, 256) == 0);
    CHECK(AlignUp(1, 256) == 256);
    CHECK(AlignUp(255, 256) == 256);
    CHECK(AlignUp(256, 256) == 256);
    CHECK(AlignUp(257, 256) == 512);
    CHECK(AlignUp(4096, 256) == 4096);
}

TEST_CASE("FindTensorSuffix matches by name suffix", "[fastload]")
{
    using lsxcommon::fastload::FindTensorSuffix;
    lsxcommon::TensorBatchViews batch{};
    const int64_t shape_a[2] = {3, 4};
    lsxcommon::TensorView a{};
    a.name = "model.layers.3.input_layernorm.weight";
    a.rank = 2;
    a.shape = shape_a;
    const int64_t shape_b[2] = {8, 4};
    lsxcommon::TensorView b{};
    b.name = "model.layers.3.mlp.gate.weight";
    b.rank = 2;
    b.shape = shape_b;
    batch.tensors = {a, b};

    const auto *hit = FindTensorSuffix(batch, "input_layernorm.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "model.layers.3.input_layernorm.weight");

    hit = FindTensorSuffix(batch, "mlp.gate.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "model.layers.3.mlp.gate.weight");

    CHECK(FindTensorSuffix(batch, "missing.weight") == nullptr);
    // A "suffix" longer than the name must not match.
    CHECK(FindTensorSuffix(batch, "x.model.layers.3.mlp.gate.weight") ==
          nullptr);
}

TEST_CASE("FindNvfp4Suffix matches by record name suffix", "[fastload]")
{
    using lsxcommon::fastload::FindNvfp4Suffix;
    lsxcommon::DenseNvfp4BatchViews batch{};
    lsxcommon::DenseNvfp4View r{};
    r.name = "blk.0.attn_q.weight";
    r.rank = 2;
    batch.records = {r};

    const auto *hit = FindNvfp4Suffix(batch, "attn_q.weight");
    REQUIRE(hit != nullptr);
    CHECK(hit->name == "blk.0.attn_q.weight");
    CHECK(FindNvfp4Suffix(batch, "attn_k.weight") == nullptr);
}
