// TML-d24-varloop-research loader test: synthetic converted tree only.
//
// The varloop row reuses the parent's loader, so what this pins is the part
// that differs — the topology the index must carry, and the slab arithmetic
// that follows from it:
//
//   * `varloop` and `loop_passes_max` reach the topology (a tree that loses them
//     looks exactly like a fixed-pass tree and would be run at 2 passes here);
//   * `CacheLayers()` is the MAX slab (72 slots at the real geometry, 24 in this
//     fixture), while `LayerApplications()` is the ACTIVE count — the two are
//     equal for the fixed-pass sibling and must not be conflated, because the
//     slab is addressed by slot index and the trace by application index;
//   * the format gate accepts this tree's tag and REJECTS the sibling's, so the
//     two artifacts can never be loaded as each other.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_loop_convert.h>
#include <lsxcommon/inkling_moe_loop_load.h>
#include <lsxcommon/inkling_moe_varloop_convert.h>
#include "inkling_moe_varloop_constants.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "test_inkling_moe_loop_fixture.h"

namespace fs = std::filesystem;
using namespace inklinglooptest;
using lsxcommon::inkling::InklingLoopTopology;
namespace vf = lsxcommon::inkling::varloop::frozen;

namespace {

constexpr int kVarPassesMax = 5;  // the varloop ceiling (passport)

// The synthetic varloop tree: same fixture, `is_varloop: true`.
fs::path ConvertVarloopFixture(const char *tag)
{
    const fs::path in =
        fs::temp_directory_path() / (std::string("inkling_varloop_load_in_") + tag);
    const fs::path out =
        fs::temp_directory_path() / (std::string("inkling_varloop_load_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/false, /*varloop=*/true);
    auto st = lsxcommon::inkling::ConvertInklingVarloopModel(in.string(),
                                                            out.string());
    if (!st.ok()) FAIL(st.ToString());
    return out;
}

}  // namespace

TEST_CASE("varloop loader topology carries the capability and its ceiling",
          "[inkling][varloop][loader]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path dir = ConvertVarloopFixture("topology");

    InklingLoopTopology top;
    const auto st = lsxcommon::inkling::LoadInklingLoopTopology(
        dir.string(), top, lsxcommon::inkling::VarloopConvertSpec().format_tag);
    INFO(top.error_msg);
    REQUIRE(st.ok());

    CHECK(top.varloop);
    // The checkpoint's declared count (the fixture declares 2) and the trained
    // ceiling the converter recorded from the passport.
    CHECK(top.loop_passes == inklinglooptest::kLoopPasses);
    CHECK(top.loop_passes_max == kVarPassesMax);
    CHECK(top.loop_start == inklinglooptest::kLoopStart);
    CHECK(top.loop_end == inklinglooptest::kLoopEnd);
    CHECK(top.LoopLayers() == inklinglooptest::kLoopLayers);

    // The slab is the MAX configuration; the active stream is the declared one.
    CHECK(top.CacheLayers() ==
          inklinglooptest::kLayers + (kVarPassesMax - 1) * inklinglooptest::kLoopLayers);
    CHECK(top.CacheLayers() == 24);
    CHECK(top.LayerApplications() ==
          inklinglooptest::kLayers +
              (inklinglooptest::kLoopPasses - 1) * inklinglooptest::kLoopLayers);
    CHECK(top.LayerApplications() == 12);
    // Every active slot is inside the slab, and the ceiling fits exactly.
    for (int p = 0; p < top.loop_passes; ++p)
        for (int l = top.loop_start; l < top.loop_end; ++l)
            CHECK(top.CacheSlot(l, p) < top.CacheLayers());
    CHECK(top.CacheSlot(top.loop_end - 1, kVarPassesMax - 1) ==
          top.CacheLayers() - 1);

    fs::remove_all(dir);
}

TEST_CASE("the fixed-pass engine refuses a varloop tree by format",
          "[inkling][varloop][loader]")
{
    // Format before anything else: a Mini engine pointed at this tree must fail
    // loudly instead of loading an artifact whose pass count it cannot express.
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path dir = ConvertVarloopFixture("format");

    InklingLoopTopology top;
    const auto st = lsxcommon::inkling::LoadInklingLoopTopology(dir.string(), top);
    CHECK_FALSE(st.ok());
    CHECK(st.ToString().find("logestix-inkling-2-mini") != std::string::npos);

    // ...and the varloop tag is not accepted by a fixed-pass expectation
    // either: the check names the format it wanted.
    InklingLoopTopology wrong;
    const auto st2 = lsxcommon::inkling::LoadInklingLoopTopology(
        dir.string(), wrong, "logestix-inkling-2-mini");
    CHECK_FALSE(st2.ok());

    fs::remove_all(dir);
}

TEST_CASE("a fixed-pass tree keeps its own single-pass-count shape",
          "[inkling][varloop][loader]")
{
    // The parent's artifact must be untouched by the varloop plumbing: its
    // ceiling equals its declared count, so the slab and the application count
    // are the same number, and `varloop` is false.
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_varloop_fixedload_in";
    const fs::path out = fs::temp_directory_path() / "inkling_varloop_fixedload_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);  // fixed-pass fixture
    auto st = lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string());
    REQUIRE(st.ok());

    InklingLoopTopology top;
    REQUIRE(lsxcommon::inkling::LoadInklingLoopTopology(out.string(), top).ok());
    CHECK_FALSE(top.varloop);
    CHECK(top.loop_passes_max == top.loop_passes);
    CHECK(top.CacheLayers() == top.LayerApplications());
    CHECK(top.CacheLayers() == inklinglooptest::kLayers +
                                   (inklinglooptest::kLoopPasses - 1) *
                                       inklinglooptest::kLoopLayers);

    fs::remove_all(in);
    fs::remove_all(out);
}

TEST_CASE("varloop pinned weights and expert planes load unchanged",
          "[inkling][varloop][loader]")
{
    // The artifact layout is the parent's: the same dense plane, the same
    // six-plane expert blocks. A varloop conversion that changed any of this
    // would be a second implementation of the architecture.
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    using namespace lsxcommon::inkling;
    const fs::path dir = ConvertVarloopFixture("pinned");

    InklingLoopTopology top;
    REQUIRE(LoadInklingLoopTopology(dir.string(), top,
                                    VarloopConvertSpec().format_tag)
                .ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    InklingLoopPinnedWeights pw;
    pw.top = top;
    REQUIRE(LoadInklingLoopPinnedWeights(dir.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(pw.embed != nullptr);
    CHECK(pw.head != nullptr);
    CHECK(pw.embed_rows == inklinglooptest::kVocab);
    CHECK(pw.message_queries != nullptr);
    CHECK(pw.layers.size() == static_cast<size_t>(inklinglooptest::kLayers));
    CHECK(pw.resident_bytes > 0);

    // The routed store is the neutral ExpertStore, geometry-driven: the loop
    // layers are visited once per pass, so the window floor follows the
    // APPLICATION count of the active pass count, not the slab.
    InklingLoopExpertDb db =
        LoadInklingLoopExpertIndex(dir.string(), top.num_layers, top.num_experts,
                                   /*max_open=*/64);
    CHECK(db.num_layers == inklinglooptest::kLayers);
    CHECK(db.experts_per_layer == inklinglooptest::kExperts);
    CHECK(InklingLoopExpertWindowFloor(top) ==
          static_cast<size_t>(top.CacheLayers()) * static_cast<size_t>(top.top_k));

    FreeInklingLoopExpertIndex(db);
    FreeInklingLoopPinnedWeights(pw);
    cudaStreamDestroy(stream);
    fs::remove_all(dir);
}
