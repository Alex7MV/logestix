// TDD — gpu::ExpertDeviceCache host-side LRU bookkeeping (no CUDA, no model).
//
// Canonical unified cache (model-unification 2026-08): one implementation
// serving GLM / Qwen-MoE / DeepSeek routed-expert slot caching. Semantics
// pinned here are unchanged from the original per-family copies:
//   - Lookup:  slot index for a flat expert id, or -1 (miss). A hit refreshes
//              recency and counts toward hits.
//   - Insert:  claims a slot for a flat expert id (lowest free slot first;
//              when full, evicts the LRU slot), commits the mapping both ways
//              and counts toward misses.
//   - Touch:   moves a slot to the MRU position of h_lru.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/expert_device_cache.h>

#include <vector>

namespace {

using lsxcommon::gpu::ExpertDeviceCache;

struct TestCache {
    ExpertDeviceCache c{};
    std::vector<int32_t> slot_of;
    std::vector<int32_t> owner;
    std::vector<int32_t> lru;

    TestCache(int slots, int total_experts)
    {
        c.slots = slots;
        c.total_experts = total_experts;
        c.pack_stride = 64;
        c.scale_stride = 16;
        slot_of.assign(static_cast<size_t>(total_experts), -1);
        owner.assign(static_cast<size_t>(slots), -1);
        lru.resize(static_cast<size_t>(slots));
        for (int i = 0; i < slots; ++i) lru[static_cast<size_t>(i)] = slots - 1 - i;
        c.h_slot_of = slot_of.data();
        c.h_slot_owner = owner.data();
        c.h_lru = lru.data();
    }
};

} // namespace

TEST_CASE("glm expert cache: fresh cache reports miss for every expert",
          "[cache]")
{
    TestCache t(4, 16);
    for (int e = 0; e < 16; ++e)
        REQUIRE(t.c.Lookup(e) == -1);
    REQUIRE(t.c.misses == 0);
    REQUIRE(t.c.hits == 0);
}

TEST_CASE("glm expert cache: insert uses lowest free slot and commits mapping",
          "[cache]")
{
    TestCache t(4, 16);
    REQUIRE(t.c.Insert(10) == 0);
    REQUIRE(t.c.Insert(11) == 1);
    REQUIRE(t.c.Lookup(10) == 0);
    REQUIRE(t.c.Lookup(11) == 1);
    REQUIRE(t.owner[static_cast<size_t>(0)] == 10);
    REQUIRE(t.owner[static_cast<size_t>(1)] == 11);
    REQUIRE(t.c.misses == 2);
    REQUIRE(t.c.hits == 2);
}

TEST_CASE("glm expert cache: insert counts misses and commits mapping",
          "[cache]")
{
    TestCache t(4, 16);
    REQUIRE(t.c.Lookup(5) == -1);   // pure query: no stats
    REQUIRE(t.c.misses == 0);
    REQUIRE(t.c.Insert(5) == 0);
    REQUIRE(t.c.misses == 1);
    REQUIRE(t.c.hits == 0);
    REQUIRE(t.c.Lookup(5) == 0);
    REQUIRE(t.c.hits == 1);
}

TEST_CASE("glm expert cache: full insert evicts the LRU slot", "[cache]")
{
    TestCache t(3, 16);
    REQUIRE(t.c.Insert(0) == 0);
    REQUIRE(t.c.Insert(1) == 1);
    REQUIRE(t.c.Insert(2) == 2);

    // Insert into a full cache: LRU tail is slot 0 (owner flat 0 inserted
    // first and never touched since).
    REQUIRE(t.c.Insert(3) == 0);
    REQUIRE(t.c.Lookup(0) == -1);  // victim mapping invalidated
    REQUIRE(t.c.Lookup(3) == 0);
    REQUIRE(t.owner[static_cast<size_t>(0)] == 3);
}

TEST_CASE("glm expert cache: hit refreshes recency and changes eviction order",
          "[cache]")
{
    TestCache t(3, 16);
    REQUIRE(t.c.Insert(0) == 0);
    REQUIRE(t.c.Insert(1) == 1);
    REQUIRE(t.c.Insert(2) == 2);

    // Touch slot 0 (via explicit touch) so slot 1 becomes the LRU tail.
    t.c.Touch(0);
    REQUIRE(t.c.Insert(3) == 1);
    REQUIRE(t.c.Lookup(1) == -1);
    REQUIRE(t.c.Lookup(0) == 0);   // survivor
    REQUIRE(t.c.Lookup(2) == 2);
    REQUIRE(t.c.Lookup(3) == 1);
}

TEST_CASE("glm expert cache: lru front is MRU after touch", "[cache]")
{
    TestCache t(4, 16);
    for (int e = 0; e < 4; ++e) (void)t.c.Insert(e);
    // After inserts 0..3 the tail is slot 0.
    t.c.Touch(2);
    REQUIRE(t.lru[static_cast<size_t>(0)] == 2);
    REQUIRE(t.lru[static_cast<size_t>(3)] != 2);
}

TEST_CASE("glm expert cache: DeviceBytes accounts pack and scale strides",
          "[cache]")
{
    ExpertDeviceCache c{};
    REQUIRE(c.DeviceBytes() == 0);
    c.slots = 8;
    c.pack_stride = 1000;
    c.scale_stride = 125;
    REQUIRE(c.DeviceBytes() == 8 * 1125);
}
