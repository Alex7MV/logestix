// TDD — MiMo routed-expert host store + device slot LRU cache. Slot
// layout/sizing are host logic; the staging case asserts the exact bytes land
// in a slot and that the LRU evicts the least-recently-used victim.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mimo_moe_expert_cache.h"

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace lsxcommon::mimo;
using lsxcommon::gpu::ExpertDeviceCache;

TEST_CASE("mimo expert slot layout and sizing", "[mimo][expert-cache]")
{
    const auto l = PlanMimoExpertSlot(256, 256, 256, 256, 256, 256);
    REQUIRE(l.ok);
    CHECK(l.up_pack_off == 256);
    CHECK(l.pack_stride == 256 + 256 + 256);
    CHECK(l.scale_stride == 256 + 256 + 256);
    CHECK(MimoExpertCachePerExpertBytes(l) == 1536);
    CHECK_FALSE(PlanMimoExpertSlot(0, 1, 1, 1, 1, 1).ok);

    const int total = 12032, topk = 8;
    CHECK(MimoExpertCacheSlots(0, 1536, total, topk, "128") == 128);
    CHECK(MimoExpertCacheSlots(0, 1536, total, topk, "999999") == total);
    CHECK(MimoExpertCacheSlots(256, 1536, total, topk, nullptr) == topk);
    CHECK(MimoExpertCacheSlots(1ull << 30, 0, total, topk, nullptr) == total);
}

TEST_CASE("mimo expert staging and LRU eviction", "[mimo][expert-cache]")
{
    const auto l = PlanMimoExpertSlot(256, 256, 256, 256, 256, 256);
    REQUIRE(l.ok);
    constexpr int kExperts = 4;
    MimoExpertCache cache;
    cache.layout = l;
    cache.per_expert_bytes = MimoExpertCachePerExpertBytes(l);
    cache.experts_per_layer = kExperts;
    cache.total_experts = kExperts;
    cache.store.num_layers = 1;
    cache.store.experts_per_layer = kExperts;
    cache.store.entries.assign(kExperts, MimoExpertEntry{});

    std::vector<std::vector<uint8_t>> gate(kExperts), up(kExperts), down(kExperts);
    for (int e = 0; e < kExperts; ++e)
    {
        gate[e].assign(256, static_cast<uint8_t>(0x10 + e));
        up[e].assign(256, static_cast<uint8_t>(0x20 + e));
        down[e].assign(256, static_cast<uint8_t>(0x30 + e));
        MimoExpertEntry &en = cache.store.entries[static_cast<size_t>(e)];
        en.host_w1 = gate[e].data(); en.w1_bytes = 256;
        en.host_w1_scales = gate[e].data(); en.w1_scales_bytes = 256;
        en.host_w3 = up[e].data(); en.w3_bytes = 256;
        en.host_w3_scales = up[e].data(); en.w3_scales_bytes = 256;
        en.host_w2 = down[e].data(); en.w2_bytes = 256;
        en.host_w2_scales = down[e].data(); en.w2_scales_bytes = 256;
    }

    std::string err;
    REQUIRE(AllocateMimoExpertCache(cache, /*slots=*/2, err));
    REQUIRE(err.empty());

    auto check_slot = [&](int slot, int expert) {
        std::vector<uint8_t> pack(static_cast<size_t>(l.pack_stride), 0);
        REQUIRE(cudaMemcpy(pack.data(),
                           cache.device.d_pack + static_cast<int64_t>(slot) * l.pack_stride,
                           pack.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        return pack[static_cast<size_t>(l.gate_pack_off)] == gate[expert][0] &&
               pack[static_cast<size_t>(l.up_pack_off)] == up[expert][0] &&
               pack[static_cast<size_t>(l.down_pack_off)] == down[expert][0];
    };
    auto stage = [&](int expert) {
        int slot = -1;
        REQUIRE(StageMimoExpert(cache, 0, expert, nullptr, err, &slot) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
        return slot;
    };

    CHECK(stage(0) == 0);
    CHECK(check_slot(0, 0));
    CHECK(stage(1) == 1);
    CHECK(check_slot(1, 1));
    CHECK(stage(2) == 0);  // LRU victim slot 0 overwritten
    CHECK(check_slot(0, 2));

    FreeMimoExpertCache(cache);
    CHECK(cache.device.d_base == nullptr);
    CHECK(cache.device.slots == 0);
}
