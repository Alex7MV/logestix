// Kimi MoE expert device-cache bookkeeping (host-only, zero CUDA calls).
// Spec: docs/superpowers/specs/2026-08-27-kimi-expert-cache-design.md
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "lsxcommon/kimi_moe_expert_cache.h"

using lsxcommon::kimi::KimiMoeCacheBatch;
using lsxcommon::kimi::KimiMoeExpertCachePlan;
using lsxcommon::kimi::KimiMoeExpertCacheSlots;
using lsxcommon::kimi::KimiMoeResolveBatch;

namespace {
constexpr int kEPL = 384;   // Kimi experts per layer
} // namespace

// The shared-expert device arena stores one [w1|w3|w2] pack slab and one
// [s1|s3|s2] scale slab per MoE layer. The SAME layout function drives the
// one-time upload and the inference-time view, so a disagreement between
// them (a silent weight corruption) is impossible by construction.
TEST_CASE("Kimi shared-expert slab layout: 256-aligned, gap-free", "[kimi][cache]")
{
    using lsxcommon::kimi::KimiSharedSlabLayout;
    // Kimi-K2.7-Code real geometry: I*H/2 = 2048*7168/2 packed, /16 scales.
    const int64_t w1 = 7340032, w3 = 7340032, w2 = 7340032;
    const int64_t s1 = 917504, s3 = 917504, s2 = 917504;
    const auto s = KimiSharedSlabLayout(w1, w3, w2, s1, s3, s2);
    REQUIRE(s.p_w1 == 0);
    REQUIRE(s.p_w3 == 7340032);       // already 256-aligned
    REQUIRE(s.p_w2 == 14680064);
    REQUIRE(s.s_s1 == 0);
    REQUIRE(s.s_s3 == 917504);
    REQUIRE(s.s_s2 == 1835008);
    REQUIRE(s.pack_bytes == 3 * 7340032);
    REQUIRE(s.scale_bytes == 3 * 917504);

    // Shapes that are not multiples of 256 round UP (alignment contract).
    const auto r = KimiSharedSlabLayout(10, 20, 30, 5, 6, 7);
    REQUIRE(r.p_w1 == 0);
    REQUIRE(r.p_w3 == 256);
    REQUIRE(r.p_w2 == 512);
    REQUIRE(r.s_s1 == 0);
    REQUIRE(r.s_s3 == 256);
    REQUIRE(r.s_s2 == 512);
    REQUIRE(r.pack_bytes == 768);
    REQUIRE(r.scale_bytes == 768);
}

TEST_CASE("Kimi cache slots: policy wiring and env override", "[kimi][cache]")
{
    const std::size_t per_expert = 26ull << 20;   // ~26 MiB pack+scale

    // Policy path (env absent): usable = free - 2 GiB Kimi reserve.
    const std::size_t free_bytes = (2ull << 30) + 100 * per_expert;
    REQUIRE(KimiMoeExpertCacheSlots(free_bytes, per_expert, 23424, nullptr) == 100);
    // Clamped to total experts (60 GiB usable / 26 MiB > 256).
    REQUIRE(KimiMoeExpertCacheSlots(64ull << 30, per_expert, 256, nullptr) == 256);
    // Free below reserve -> nothing fits.
    REQUIRE(KimiMoeExpertCacheSlots(1ull << 30, per_expert, 23424, nullptr) == 0);

    // Explicit env wins (clamped); "0" disables the cache.
    REQUIRE(KimiMoeExpertCacheSlots(0, per_expert, 23424, "77") == 77);
    REQUIRE(KimiMoeExpertCacheSlots(0, per_expert, 23424, "999999") == 23424);
    REQUIRE(KimiMoeExpertCacheSlots(free_bytes, per_expert, 23424, "0") == 0);
}

TEST_CASE("Kimi cache batch: first touch assigns, second is resident",
          "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(4, 2 * kEPL);

    int ids[8] = {3, 7, 3, 100, 7, 11, 200, 3};
    KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 1, kEPL, ids, 8);

    // Unique experts {3,7,11,100,200} = 5 first-touch uploads, but only 4
    // slots: idx 0(3)->slot0, 1(7)->slot1, 3(100)->slot2; 5(11), 6(200)
    // cannot be placed (all slots hold this-batch residents).
    REQUIRE(b.missing_count == 5);
    std::vector<int> assigned;
    for (int i = 0; i < b.missing_count; ++i)
        if (b.missing_slot[i] >= 0) assigned.push_back(b.missing_slot[i]);
    REQUIRE(assigned.size() == 4);
    std::sort(assigned.begin(), assigned.end());
    REQUIRE(std::adjacent_find(assigned.begin(), assigned.end()) == assigned.end());
    // Duplicates 3/7/3 resolve as resident hits on their second touch.
    REQUIRE(b.resident_slot[7] >= 0);
    REQUIRE(b.resident_slot[2] >= 0);
    REQUIRE(b.resident_slot[4] >= 0);
    REQUIRE(b.resident_slot[0] == -1);   // 3 was itself an upload

    // Same ids again: only 200 is still slot-less.
    KimiMoeCacheBatch b2 = KimiMoeResolveBatch(p, 1, kEPL, ids, 8);
    REQUIRE(b2.missing_count == 1);
    for (int i = 0; i < 8; ++i)
        if (b2.resident_slot[i] < 0) {
            const bool hot = (ids[i] == 200);
            REQUIRE(hot);
        }
}

TEST_CASE("Kimi cache LRU: cold slot evicted before hot one", "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(2, 10 * kEPL);

    int id0[1] = {0}, id1[1] = {1}, id2[1] = {2};
    auto single = [&](int *ids) {
        KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 0, kEPL, ids, 1);
        REQUIRE(b.missing_count == 1);
        REQUIRE(b.resident_slot[0] == -1);
        return b.missing_slot[0];
    };
    REQUIRE(single(id0) == 0);   // empty slot 0 first
    REQUIRE(single(id1) == 1);   // empty slot 1
    REQUIRE(single(id2) == 0);   // evicts LRU victim (expert 0, slot 0)

    // Expert 1 still resident; expert 0 evicted.
    KimiMoeCacheBatch r1 = KimiMoeResolveBatch(p, 0, kEPL, id1, 1);
    REQUIRE(r1.missing_count == 0);
    REQUIRE(r1.resident_slot[0] == 1);
    KimiMoeCacheBatch r0 = KimiMoeResolveBatch(p, 0, kEPL, id0, 1);
    REQUIRE(r0.missing_count == 1);
}

// The cached routed-FFN loop overlaps the per-expert H2D upload of a layer's
// misses with the compute of the experts that already landed. That requires
// the residency decision to say, per batch entry, WHICH missing upload
// delivers its slot: a pre-resident expert needs no wait (its bytes were
// gated by an earlier layer's compute), while a miss and any same-batch
// duplicate of it must wait on that specific missing entry's upload event.
// A single batch-wide fence (the previous design) forces the compute of
// every entry to wait for the LAST upload and re-serializes the pipeline.
TEST_CASE("Kimi cache batch: per-slice upload dependency plan", "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(4, 2 * kEPL);

    // Prime three residents (as an earlier layer would have).
    int pre[3] = {10, 11, 12};
    KimiMoeResolveBatch(p, 1, kEPL, pre, 3);

    // Layer-1 batch: two pre-resident hits, one new miss, a same-batch
    // duplicate of the new miss, a repeated pre-resident, one more miss.
    int ids[6] = {10, 11, 20, 20, 10, 30};
    KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 1, kEPL, ids, 6);

    REQUIRE(b.wait_miss_pos[0] == -1);   // 10: resident before the batch
    REQUIRE(b.wait_miss_pos[1] == -1);   // 11: resident before the batch

    const int pos20 = b.wait_miss_pos[2];   // 20's own upload
    REQUIRE(pos20 >= 0);
    REQUIRE(b.missing_idx[pos20] == 2);
    REQUIRE(b.missing_slot[pos20] >= 0);
    REQUIRE(b.wait_miss_pos[3] == pos20);   // duplicate of 20: same upload

    REQUIRE(b.wait_miss_pos[4] == -1);      // 10 again: still no wait

    const int pos30 = b.wait_miss_pos[5];   // 30's own upload
    REQUIRE(pos30 >= 0);
    REQUIRE(pos30 != pos20);
    REQUIRE(b.missing_idx[pos30] == 5);

    // Every wait position is a real missing entry, and the plan is stable.
    for (int i = 0; i < 6; ++i)
        if (b.wait_miss_pos[i] >= 0)
            REQUIRE(b.wait_miss_pos[i] < b.missing_count);

    // Disabled cache: no slots, so no upload events to wait on.
    KimiMoeExpertCachePlan off;
    off.Init(0, 2 * kEPL);
    KimiMoeCacheBatch bo = KimiMoeResolveBatch(off, 0, kEPL, ids, 6);
    for (int i = 0; i < 6; ++i) REQUIRE(bo.wait_miss_pos[i] == -1);
}

TEST_CASE("Kimi cache rejects out-of-range input deterministically",
          "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(4, 2 * kEPL);

    int ok[1] = {0};
    REQUIRE(KimiMoeResolveBatch(p, -1, kEPL, ok, 1).invalid);
    int over[1] = {kEPL};
    REQUIRE(KimiMoeResolveBatch(p, 0, kEPL, over, 1).invalid);
    int neg[1] = {-1};
    REQUIRE(KimiMoeResolveBatch(p, 0, kEPL, neg, 1).invalid);
}

TEST_CASE("Kimi cache disabled plan is inert", "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(0, 23424);
    REQUIRE_FALSE(p.Active());
    int ids[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 0, kEPL, ids, 8);
    REQUIRE(b.missing_count == 8);
    REQUIRE(b.missing_slot[0] == -1);   // no slots to assign
}