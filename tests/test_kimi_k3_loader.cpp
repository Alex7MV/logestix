// Kimi-K3 loader stage: the pinned graph, the mmap-streamed latent/shared
// plane and the neutral routed-expert store, exercised on a synthetic
// converted tree (never the vault).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "kimi_moe_k3_constants.h"
#include "lsxcommon/kimi_moe_k3_convert.h"
#include "lsxcommon/kimi_moe_k3_load.h"
#include "test_kimi_k3_fixture.h"

namespace fs = std::filesystem;
using namespace lsxcommon;

namespace {

bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

fs::path BuildConverted()
{
    auto in = fs::temp_directory_path() / "kimi_k3_loader_in";
    auto out = fs::temp_directory_path() / "kimi_k3_loader_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    kimik3test::WriteCheckpoint(in);
    REQUIRE(kimi_k3::ConvertKimiK3Model(in.string(), out.string()).ok());
    return out;
}

}  // namespace

TEST_CASE("kimi-k3 loader reads the frozen topology", "[kimi-k3][loader]")
{
    const fs::path root = BuildConverted();
    kimi_k3::KimiK3Topology top;
    REQUIRE(kimi_k3::LoadKimiK3Topology(root.string(), top).ok());
    CHECK(top.format == "logestix-kimi-k3-nvfp4");
    CHECK(top.num_layers == kimik3test::kLayers);
    CHECK(top.hidden == kimik3test::kHidden);
    CHECK(top.mla_layers.size() == 2);
    CHECK(top.kda_layers.size() == 2);
    CHECK(top.IsMla(2));
    CHECK(top.IsMla(3));
    CHECK_FALSE(top.IsMla(0));
    CHECK(top.skipped_visual == 1);
    CHECK(top.skipped_projector == 1);
    // Synthetic geometry must NOT pass the frozen cross-check.
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(top));
}

TEST_CASE("kimi-k3 loader uploads the pinned graph and keeps the streamed plane mapped",
          "[kimi-k3][loader]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path root = BuildConverted();

    kimi_k3::KimiK3PinnedWeights pw;
    REQUIRE(kimi_k3::LoadKimiK3PinnedWeights(root.string(), pw, 0));
    REQUIRE(pw.embed.d_packed != nullptr);
    REQUIRE(pw.head.d_packed != nullptr);
    REQUIRE(pw.layers.size() == static_cast<size_t>(kimik3test::kLayers));
    CHECK(pw.layers[0].mla == false);
    CHECK(pw.layers[2].mla == true);
    CHECK(pw.layers[0].kda_q.d_packed != nullptr);
    CHECK(pw.layers[0].kda_conv_q != nullptr);
    CHECK(pw.layers[2].mla_kvb.d_packed != nullptr);
    CHECK(pw.layers[2].mla_qanorm != nullptr);
    CHECK(pw.layers[0].has_moe == false);   // layer 0 is dense
    CHECK(pw.layers[1].has_moe == true);
    // Streamed plane: host views only, no device copy of its own.
    CHECK(pw.streamed[1].loaded);
    CHECK(pw.streamed[1].shared_gate.packed != nullptr);
    CHECK(pw.stream_dev.buffer != nullptr);
    CHECK(pw.stream_dev.latent_down.d_packed == nullptr);  // not staged yet
    REQUIRE(pw.resident_bytes > 0);
    // Staging binds the device pointers for one layer.
    REQUIRE(kimi_k3::StageKimiK3Layer(pw, 1, 0) == cudaSuccess);
    CHECK(pw.stream_dev.latent_down.d_packed != nullptr);
    CHECK(pw.stream_dev.shared_down.d_packed != nullptr);
    CHECK(pw.stream_dev.latent_down.rows == kimik3test::kLatent);
    CHECK(pw.stream_dev.latent_down.cols == kimik3test::kHidden);

    kimi_k3::FreeKimiK3PinnedWeights(pw);
    CHECK(pw.embed.d_packed == nullptr);
    CHECK(pw.stream_dev.buffer == nullptr);
}

TEST_CASE("kimi-k3 expert store maps routed experts lazily", "[kimi-k3][loader]")
{
    const fs::path root = BuildConverted();
    auto db = kimi_k3::LoadKimiK3ExpertIndex(
        root.string(), kimik3test::kLayers, kimik3test::kExperts,
        kimik3test::kDenseLayers, /*max_open=*/4);
    REQUIRE(db.entries.size() ==
            static_cast<size_t>(kimik3test::kLayers * kimik3test::kExperts));
    kimi_k3::EnsureKimiK3ExpertResident(db, 1, 2, 0);
    const size_t idx = 1 * kimik3test::kExperts + 2;
    const auto &e = db.entries[idx];
    REQUIRE(e.host_w1 != nullptr);
    REQUIRE(e.host_w2 != nullptr);
    REQUIRE(e.host_w3 != nullptr);
    // Per-16 scales (2x the source per-32 count).
    CHECK(e.w1_bytes ==
          static_cast<int64_t>(kimik3test::kMoeInter) * kimik3test::kLatent / 2);
    CHECK(e.w1_scales_bytes ==
          static_cast<int64_t>(kimik3test::kMoeInter) * kimik3test::kLatent / 16);
    CHECK(e.w1_global_scale > 0.0f);
    // A dense-prefix layer has no expert files and stays null.
    kimi_k3::EnsureKimiK3ExpertResident(db, 0, 0, 0);
    CHECK(db.entries[0].host_w1 == nullptr);
    kimi_k3::FreeKimiK3ExpertIndex(db);
}

// The routed-expert window is not a free tuning knob: the decode loop visits
// every MoE layer once per token, in ascending layer order, taking top_k
// distinct experts from each. A window smaller than that working set makes the
// access pattern a cyclic scan, so EVERY access misses and the store re-reads
// and re-parses experts it already had. Measured on real weights (both trees,
// 64-token cycles, identical logits_hash): a 512-slot window against a
// 1472-expert working set cost 1.26x on the warm cycles and 1.71x on the cold
// first cycle, i.e. the win is real and the case is not hypothetical.
TEST_CASE("kimi-k3 expert window covers one token's working set",
          "[kimi-k3][loader]")
{
    // Synthetic fixture: 4 layers, 1 dense, top-2 -> (4-1)*2 = 6.
    kimi_k3::KimiK3Topology top;
    top.num_layers = kimik3test::kLayers;
    top.dense_layers = kimik3test::kDenseLayers;
    top.top_k = kimik3test::kTopK;

    const size_t floor = kimi_k3::KimiK3ExpertWindowFloor(top);
    CHECK(floor == 6);

    // The frozen geometry is what the default must actually survive: 93 layers,
    // layer 0 dense, top-16 -> 1472 distinct experts per token.
    kimi_k3::KimiK3Topology frozen;
    frozen.num_layers = kimi_k3::frozen::kLayers;
    frozen.dense_layers = kimi_k3::frozen::kDenseLayers;
    frozen.top_k = kimi_k3::frozen::kTopK;
    CHECK(frozen.num_layers == 93);
    CHECK(kimi_k3::KimiK3ExpertWindowFloor(frozen) == 1472);

    // A request below the floor is clamped up, not honoured: silently keeping a
    // pathological window is a 1.3-1.7x decode regression, which is exactly the
    // defect this pins. A request above it is respected.
    CHECK(kimi_k3::KimiK3ResolveExpertWindow(frozen, 512) == 1472);
    CHECK(kimi_k3::KimiK3ResolveExpertWindow(frozen, 0) == 1472);
    CHECK(kimi_k3::KimiK3ResolveExpertWindow(frozen, 2048) == 2048);
}

// Load() pre-parses max_open experts in PATH order. That is a startup tax on a
// large window and it buys capacity the LRU fills from real accesses within two
// tokens, so this family opts out — but opting out must not disable lazy
// residency, and it must leave the window empty rather than half-warm.
TEST_CASE("kimi-k3 expert store starts cold when warm-on-load is off",
          "[kimi-k3][loader]")
{
    const fs::path root = BuildConverted();

    // Default: the store pre-warms. With a 2-slot window exactly two experts of
    // the first MoE layer are mapped before any decode happens.
    auto warm = kimi_k3::LoadKimiK3ExpertIndex(
        root.string(), kimik3test::kLayers, kimik3test::kExperts,
        kimik3test::kDenseLayers, /*max_open=*/2);
    CHECK(warm.lru.size() == 2);
    CHECK(warm.entries[static_cast<size_t>(1 * kimik3test::kExperts)]
              .host_w1 != nullptr);

    // Opted out: nothing is mapped, and the store is still fully usable.
    auto cold = kimi_k3::LoadKimiK3ExpertIndex(
        root.string(), kimik3test::kLayers, kimik3test::kExperts,
        kimik3test::kDenseLayers, /*max_open=*/2, /*warm_on_load=*/false);
    CHECK(cold.lru.empty());
    for (const auto &e : cold.entries) CHECK(e.host_w1 == nullptr);
    CHECK(cold.entries.size() == warm.entries.size());

    kimi_k3::EnsureKimiK3ExpertResident(cold, 2, 1, 0);
    CHECK(cold.entries[static_cast<size_t>(2 * kimik3test::kExperts + 1)]
              .host_w1 != nullptr);
    CHECK(cold.lru.size() == 1);

    kimi_k3::FreeKimiK3ExpertIndex(warm);
    kimi_k3::FreeKimiK3ExpertIndex(cold);
}
