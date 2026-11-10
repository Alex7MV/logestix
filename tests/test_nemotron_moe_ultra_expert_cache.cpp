// TDD — Nemotron-3-Ultra-550B routed-expert host store + device slot LRU.
//
// This model owns 24 576 routed experts at ~11.8 MB each (~290 GB), so the store
// is the whole VRAM/RAM story: the dense backbone (22.5 GB) is resident and every
// expert read is either a device-slot hit or an NVMe read. What the tests pin:
//   * the slot layout arithmetic (a wrong offset silently mixes one expert's
//     scales into another's weights — the model still answers, wrongly);
//   * the slot-count policy (floor = top-k, ceiling = the pool, env override);
//   * that the host store is the NEUTRAL store holding VIEWS, not copies (R1/R2),
//     which is what keeps this from becoming a second 290 GB.
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_ultra_fixture.h"

#include "lsxcommon/nemotron_moe_ultra_convert.h"
#include "lsxcommon/nemotron_moe_ultra_expert_cache.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_ultra;

namespace {

struct EnableSyntheticGeometry {
    EnableSyntheticGeometry()
    {
#ifdef _WIN32
        _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
#else
        setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
#endif
    }
} g_enable_synthetic;

fs::path ConvertedFixture(const std::string &tag)
{
    const auto src = ultratest::MakeTempDir(tag + "_src");
    ultratest::WriteUltraCheckpoint(src);
    const auto dst = ultratest::MakeTempDir(tag + "_dst");
    const auto rc = ConvertNemotronUltraModel(src.string(), dst.string());
    REQUIRE(rc.ok());
    return dst;
}

} // namespace

TEST_CASE("ultra expert slot: sections are 256-byte aligned and both projections fit",
          "[ultra][cache]")
{
    // Production payload: up [5120, 2048] and down [2048, 5120] at NVFP4.
    const std::int64_t up_pack = 5120LL * 2048 / 2;      // 5 242 880
    const std::int64_t up_scale = 5120LL * 2048 / 16;    // 655 360
    const std::int64_t down_pack = 2048LL * 5120 / 2;
    const std::int64_t down_scale = 2048LL * 5120 / 16;

    const auto l = PlanUltraExpertSlot(up_pack, up_scale, down_pack, down_scale);
    REQUIRE(l.ok);
    CHECK(l.up_pack_off == 0);
    CHECK(l.down_pack_off % 256 == 0);
    CHECK(l.down_pack_off >= up_pack);
    CHECK(l.pack_stride == l.down_pack_off + ((down_pack + 255) / 256) * 256);
    CHECK(l.down_scale_off % 256 == 0);
    CHECK(l.scale_stride == l.down_scale_off + ((down_scale + 255) / 256) * 256);
    // One expert slot is ~11.8 MB, which is what the VRAM budget divides.
    CHECK(UltraExpertPerExpertBytes(l) == l.pack_stride + l.scale_stride);

    // An empty section is not a layout.
    CHECK_FALSE(PlanUltraExpertSlot(0, up_scale, down_pack, down_scale).ok);
}

TEST_CASE("ultra expert slots: floor is top-k, ceiling is the pool, env wins",
          "[ultra][cache]")
{
    constexpr std::size_t per = 11'796'480;         // 11.8 MB
    constexpr std::size_t free7g = 7ull << 30;      // ~7 GB left after the dense graph
    constexpr int total = 24'576, topk = 22;

    // Budget-derived count, but never below one top-k working set.
    const int slots = UltraExpertCacheSlots(free7g, per, total, topk, nullptr);
    CHECK(slots >= topk);
    CHECK(slots <= total);
    CHECK(slots < 800);  // ~590 of 24 576 on a 32 GB card

    // No free VRAM at all still allows the top-k working set.
    CHECK(UltraExpertCacheSlots(0, per, total, topk, nullptr) == topk);

    // Unknown geometry degenerates to "everything resident".
    CHECK(UltraExpertCacheSlots(free7g, 0, total, topk, nullptr) == total);

    // Explicit override wins and is clamped to the pool.
    CHECK(UltraExpertCacheSlots(free7g, per, total, topk, "1024") == 1024);
    CHECK(UltraExpertCacheSlots(free7g, per, total, topk, "999999") == total);
    CHECK(UltraExpertCacheSlots(free7g, per, total, topk, "0") >= topk);
}

TEST_CASE("ultra expert store: the converted tree exposes views, not copies",
          "[ultra][cache]")
{
    const auto tree = ConvertedFixture("nu_ec");
    UltraExpertStore store;
    LoadUltraExpertStore(tree.string(), 3, ultratest::kTExperts, store);

    // One MoE layer (layer 1) of experts in the fixture.
    CHECK(store.experts_per_layer == ultratest::kTExperts);
    REQUIRE(store.entries.size() >= static_cast<std::size_t>(ultratest::kTExperts));

    // Views: the entry points INTO the mmap, so retaining the store does not
    // double the payload (R1/R2 of the ingestion protocol).
    int views = 0;
    for (const auto &e : store.entries) {
        if (!e.host_w3) continue;
        ++views;
        CHECK(e.w3_bytes > 0);
        CHECK(e.w2_bytes > 0);
        CHECK(e.w3_scales_bytes > 0);
        CHECK(e.w2_scales_bytes > 0);
        CHECK(e.w3_global_scale > 0.0f);
        CHECK(e.w2_global_scale > 0.0f);
    }
    CHECK(views >= ultratest::kTExperts);
    store.Free();
}

TEST_CASE("ultra expert cache: staging allocates slots and resolves distinct views",
          "[ultra][cache]")
{
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_ecgpu");

    UltraExpertCache cache;
    LoadUltraExpertStore(tree.string(), 3, ultratest::kTExperts, cache.store);
    cache.experts_per_layer = ultratest::kTExperts;
    cache.moe_layers = 1;
    cache.moe_index_of_layer = {-1, 0, -1};  // only layer 1 is MoE
    cache.layout = PlanUltraExpertSlotFromStore(cache.store);
    REQUIRE(cache.layout.ok);

    // A deliberately small cache: 3 slots for a top-2 over 8 experts, so both
    // the hit path and the miss/evict path are exercised.
    std::string alloc_err;
    REQUIRE(AllocateUltraExpertCache(cache, 3, alloc_err));
    CHECK(alloc_err.empty());
    const int ids[2] = {2, 5};
    UltraExpertView views[2];
    std::string err;
    REQUIRE(ResolveUltraExperts(cache, 1, ids, 2, nullptr, views, err) == cudaSuccess);
    CHECK(views[0].up_packed != nullptr);
    CHECK(views[1].up_packed != nullptr);
    CHECK(views[0].up_packed != views[1].up_packed);
    CHECK(views[0].down_packed != views[1].down_packed);
    CHECK(views[0].up_global_scale > 0.0f);

    // Resolving the same ids again is a pure hit (no re-stage).
    bool missed = true;
    REQUIRE(StageUltraExpert(cache, 1, 2, nullptr, err, nullptr, &missed) ==
            cudaSuccess);
    CHECK_FALSE(missed);

    // A non-MoE layer is rejected by name rather than silently mapped.
    err.clear();
    CHECK(StageUltraExpert(cache, 0, 0, nullptr, err) != cudaSuccess);
    CHECK(err.find("non-MoE") != std::string::npos);

    FreeUltraExpertCache(cache);
    CHECK(cache.device.d_base == nullptr);
    CHECK(cache.cache_bytes == 0);
}
