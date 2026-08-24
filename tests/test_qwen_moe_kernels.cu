// Qwen3.5-35B-A3B MoE kernel tests: LRU cache bookkeeping (host-side),
// router top-k, routed expert FFN, shared expert combine. CUDA-dependent
// cases are added by later tasks.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_moe_expert_cache.h"

using lsxcommon::qwen_moe::QwenMoeExpertDeviceCache;

namespace {

QwenMoeExpertDeviceCache MakeCache(int slots, int total_experts,
                                   std::vector<int32_t> &slot_of,
                                   std::vector<int32_t> &owner,
                                   std::vector<int32_t> &lru)
{
    slot_of.assign(static_cast<size_t>(total_experts), -1);
    owner.assign(static_cast<size_t>(slots), -1);
    lru.resize(static_cast<size_t>(slots));
    for (int i = 0; i < slots; ++i) lru[static_cast<size_t>(i)] = i;
    QwenMoeExpertDeviceCache c;
    c.slots = slots;
    c.total_experts = total_experts;
    c.h_slot_of = slot_of.data();
    c.h_slot_owner = owner.data();
    c.h_lru = lru.data();
    return c;
}

} // namespace

TEST_CASE("expert cache LRU bookkeeping", "[qwen-moe-cache]")
{
    std::vector<int32_t> slot_of, owner, lru;
    auto c = MakeCache(4, 8, slot_of, owner, lru);

    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, 3) == -1);
    CHECK(c.misses == 0);  // pure query miss: no stats
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 3) == 0);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 5) == 1);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 1) == 2);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 7) == 3);
    CHECK(c.hits == 0);
    CHECK(c.misses == 4);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, 5) == 1);
    CHECK(c.hits == 1);
    CHECK(lru[0] == 1);  // hit refreshes MRU

    // Insert evicts the true LRU tail (slot holding expert 3, never touched).
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 0) == 0);
    CHECK(slot_of[static_cast<size_t>(3)] == -1);
    CHECK(slot_of[static_cast<size_t>(0)] == 0);
    CHECK(owner[0] == 0);

    // Out-of-range lookups are safe and change nothing.
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, 99) == -1);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, -1) == -1);
}
