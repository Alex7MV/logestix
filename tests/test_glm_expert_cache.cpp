// TDD — GlmExpertDeviceCache host-side LRU bookkeeping (no CUDA, no model).
//
// The cache mirrors DeepSeek's ExpertDeviceCache but factors the LRU logic
// into named inline helpers so the semantics are unit-testable on the host:
//   - CacheLookup:  slot index for a flat expert id, or -1 (miss). A hit
//                   refreshes recency and counts toward hits.
//   - CacheInsert:  claims a slot for a flat expert id (lowest free slot
//                   first; when full, evicts the LRU slot), commits the
//                   mapping both ways and counts toward misses.
//   - CacheTouch:   moves a slot to the MRU position of h_lru.
#pragma once

#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/glm_expert_cache.h>

#include <vector>

namespace {

using lsxcommon::glm::CacheInsert;
using lsxcommon::glm::CacheLookup;
using lsxcommon::glm::CacheTouch;
using lsxcommon::glm::GlmExpertDeviceCache;

struct TestCache {
    GlmExpertDeviceCache c{};
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
          "[glm][cache]")
{
    TestCache t(4, 16);
    for (int e = 0; e < 16; ++e)
        REQUIRE(CacheLookup(t.c, e) == -1);
    REQUIRE(t.c.misses == 0);
    REQUIRE(t.c.hits == 0);
}

TEST_CASE("glm expert cache: insert uses lowest free slot and commits mapping",
          "[glm][cache]")
{
    TestCache t(4, 16);
    REQUIRE(CacheInsert(t.c, 10) == 0);
    REQUIRE(CacheInsert(t.c, 11) == 1);
    REQUIRE(CacheLookup(t.c, 10) == 0);
    REQUIRE(CacheLookup(t.c, 11) == 1);
    REQUIRE(t.owner[static_cast<size_t>(0)] == 10);
    REQUIRE(t.owner[static_cast<size_t>(1)] == 11);
    REQUIRE(t.c.misses == 2);
    REQUIRE(t.c.hits == 2);
}

TEST_CASE("glm expert cache: insert counts misses and commits mapping",
          "[glm][cache]")
{
    TestCache t(4, 16);
    REQUIRE(CacheLookup(t.c, 5) == -1);   // pure query: no stats
    REQUIRE(t.c.misses == 0);
    REQUIRE(CacheInsert(t.c, 5) == 0);
    REQUIRE(t.c.misses == 1);
    REQUIRE(t.c.hits == 0);
    REQUIRE(CacheLookup(t.c, 5) == 0);
    REQUIRE(t.c.hits == 1);
}

TEST_CASE("glm expert cache: full insert evicts the LRU slot", "[glm][cache]")
{
    TestCache t(3, 16);
    REQUIRE(CacheInsert(t.c, 0) == 0);
    REQUIRE(CacheInsert(t.c, 1) == 1);
    REQUIRE(CacheInsert(t.c, 2) == 2);

    // Insert into a full cache: LRU tail is slot 0 (owner flat 0 inserted
    // first and never touched since).
    REQUIRE(CacheInsert(t.c, 3) == 0);
    REQUIRE(CacheLookup(t.c, 0) == -1);  // victim mapping invalidated
    REQUIRE(CacheLookup(t.c, 3) == 0);
    REQUIRE(t.owner[static_cast<size_t>(0)] == 3);
}

TEST_CASE("glm expert cache: hit refreshes recency and changes eviction order",
          "[glm][cache]")
{
    TestCache t(3, 16);
    REQUIRE(CacheInsert(t.c, 0) == 0);
    REQUIRE(CacheInsert(t.c, 1) == 1);
    REQUIRE(CacheInsert(t.c, 2) == 2);

    // Touch slot 0 (via explicit touch) so slot 1 becomes the LRU tail.
    CacheTouch(t.c, 0);
    REQUIRE(CacheInsert(t.c, 3) == 1);
    REQUIRE(CacheLookup(t.c, 1) == -1);
    REQUIRE(CacheLookup(t.c, 0) == 0);   // survivor
    REQUIRE(CacheLookup(t.c, 2) == 2);
    REQUIRE(CacheLookup(t.c, 3) == 1);
}

TEST_CASE("glm expert cache: lru front is MRU after touch", "[glm][cache]")
{
    TestCache t(4, 16);
    for (int e = 0; e < 4; ++e) (void)CacheInsert(t.c, e);
    // After inserts 0..3 the tail is slot 0.
    CacheTouch(t.c, 2);
    REQUIRE(t.lru[static_cast<size_t>(0)] == 2);
    REQUIRE(t.lru[static_cast<size_t>(3)] != 2);
}

TEST_CASE("glm expert cache: DeviceBytes accounts pack and scale strides",
          "[glm][cache]")
{
    GlmExpertDeviceCache c{};
    REQUIRE(c.DeviceBytes() == 0);
    c.slots = 8;
    c.pack_stride = 1000;
    c.scale_stride = 125;
    REQUIRE(c.DeviceBytes() == 8 * 1125);
}
