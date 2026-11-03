// TDD — GigaChat3.5 routed-expert host store + device slot LRU cache. Slot
// layout/sizing/pointer arithmetic are pure host logic; the staging cases run
// on a device and assert the exact bytes that land in a slot.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/gigachat_moe_expert_cache.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace lsxcommon::gigachat;
using lsxcommon::gpu::ExpertDeviceCache;

namespace {
} // namespace

TEST_CASE("gigachat expert slot layout: real geometry", "[gigachat][expert-cache]")
{
    // gate/up [2048,7168] packed 7,340,032 B; scales 2048*448 = 917,504 B.
    const auto l =
        PlanGcExpertSlot(7340032, 917504, 7340032, 917504, 7340032, 917504);
    REQUIRE(l.ok);
    CHECK(l.gate_pack_off == 0);
    CHECK(l.up_pack_off == 7340032);
    CHECK(l.down_pack_off == 14680064);
    CHECK(l.pack_stride == 22020096);
    CHECK(l.gate_scale_off == 0);
    CHECK(l.up_scale_off == 917504);
    CHECK(l.down_scale_off == 1835008);
    CHECK(l.scale_stride == 2752512);
    CHECK(GcExpertCachePerExpertBytes(l) == 24772608);
}

TEST_CASE("gigachat expert slot layout: rounding and empty", "[gigachat][expert-cache]")
{
    const auto l = PlanGcExpertSlot(100, 40, 300, 7, 300, 7);
    REQUIRE(l.ok);
    CHECK(l.up_pack_off == 256);
    CHECK(l.pack_stride == 256 + 512 + 512);
    CHECK(l.up_scale_off == 256);
    CHECK(l.scale_stride == 256 + 256 + 256);
    CHECK_FALSE(PlanGcExpertSlot(0, 40, 300, 7, 300, 7).ok);
    CHECK_FALSE(PlanGcExpertSlot(100, 0, 300, 7, 300, 7).ok);
    CHECK_FALSE(PlanGcExpertSlot(100, 40, 300, 7, 0, 7).ok);
    CHECK(GcExpertCachePerExpertBytes(GcExpertSlotLayout{}) == 0);
}

TEST_CASE("gigachat slot pointers: three sections, per-slot stride",
          "[gigachat][expert-cache]")
{
    const auto l = PlanGcExpertSlot(256, 256, 256, 256, 256, 256);
    REQUIRE(l.ok);
    const int slots = 3;
    std::vector<uint8_t> mem(
        static_cast<size_t>(slots) * (l.pack_stride + l.scale_stride));
    ExpertDeviceCache c{};
    c.slots = slots;
    c.total_experts = 8;
    c.pack_stride = l.pack_stride;
    c.scale_stride = l.scale_stride;
    c.d_base = mem.data();
    c.d_pack = mem.data();
    c.d_scale = mem.data() + static_cast<size_t>(slots) * l.pack_stride;

    const auto v = GcExpertSlotPointers(c, 1, l);
    CHECK(v.gate_packed == mem.data() + l.pack_stride);
    CHECK(v.up_packed == mem.data() + l.pack_stride + l.up_pack_off);
    CHECK(v.down_packed == mem.data() + l.pack_stride + l.down_pack_off);
    CHECK(v.up_scales == c.d_scale + l.scale_stride + l.up_scale_off);
    CHECK(v.down_scales == c.d_scale + l.scale_stride + l.down_scale_off);
}

TEST_CASE("gigachat expert cache sizing", "[gigachat][expert-cache]")
{
    const int total = 9472, topk = 8;
    const std::size_t per = 24772608;
    CHECK(GcExpertCacheSlots(0, per, total, topk, "128") == 128);
    CHECK(GcExpertCacheSlots(0, per, total, topk, "999999") == total);
    const int s13 = GcExpertCacheSlots(13ull << 30, per, total, topk, nullptr);
    CHECK(s13 == static_cast<int>(((13ull << 30) - kGcExpertCacheVramReserveBytes) / per));
    CHECK(s13 > topk);
    CHECK(s13 < total);
    CHECK(GcExpertCacheSlots(240ull << 30, per, total, topk, nullptr) == total);
    CHECK(GcExpertCacheSlots(256, per, total, topk, nullptr) == topk);
    CHECK(GcExpertCacheSlots(0, per, 3, 8, nullptr) == 3);
    CHECK(GcExpertCacheSlots(1ull << 30, 0, total, topk, nullptr) == total);
}

TEST_CASE("gigachat expert staging + LRU eviction", "[gigachat][expert-cache]")
{
    const auto l = PlanGcExpertSlot(256, 256, 256, 256, 256, 256);
    REQUIRE(l.ok);
    constexpr int kExperts = 4;
    GcExpertCache cache;
    cache.layout = l;
    cache.per_expert_bytes = GcExpertCachePerExpertBytes(l);
    cache.experts_per_layer = kExperts;
    cache.total_experts = kExperts;
    cache.store.num_layers = 1;
    cache.store.experts_per_layer = kExperts;
    cache.store.entries.assign(kExperts, GcExpertEntry{});

    std::vector<std::vector<uint8_t>> gate(kExperts), up(kExperts),
        down(kExperts);
    for (int e = 0; e < kExperts; ++e)
    {
        gate[e].assign(256, static_cast<uint8_t>(0x10 + e));
        up[e].assign(256, static_cast<uint8_t>(0x20 + e));
        down[e].assign(256, static_cast<uint8_t>(0x30 + e));
        GcExpertEntry &en = cache.store.entries[static_cast<size_t>(e)];
        en.host_w1 = gate[e].data();
        en.w1_bytes = 256;
        en.host_w1_scales = gate[e].data();
        en.w1_scales_bytes = 256;
        en.host_w3 = up[e].data();
        en.w3_bytes = 256;
        en.host_w3_scales = up[e].data();
        en.w3_scales_bytes = 256;
        en.host_w2 = down[e].data();
        en.w2_bytes = 256;
        en.host_w2_scales = down[e].data();
        en.w2_scales_bytes = 256;
    }

    std::string err;
    REQUIRE(AllocateGcExpertCache(cache, /*slots=*/2, err));
    REQUIRE(err.empty());

    auto check_slot = [&](int slot, int expert) {
        std::vector<uint8_t> pack(static_cast<size_t>(l.pack_stride), 0);
        REQUIRE(cudaMemcpy(pack.data(),
                           cache.device.d_pack +
                               static_cast<int64_t>(slot) * l.pack_stride,
                           pack.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        return pack[static_cast<size_t>(l.gate_pack_off)] ==
                   gate[expert][0] &&
               pack[static_cast<size_t>(l.up_pack_off)] == up[expert][0] &&
               pack[static_cast<size_t>(l.down_pack_off)] == down[expert][0];
    };
    auto stage = [&](int expert) {
        int slot = -1;
        REQUIRE(StageGcExpert(cache, 0, expert, nullptr, err, &slot) ==
                cudaSuccess);
        REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
        return slot;
    };

    CHECK(stage(0) == 0);
    CHECK(check_slot(0, 0));
    CHECK(stage(1) == 1);
    CHECK(check_slot(1, 1));
    CHECK(stage(2) == 0);  // LRU victim slot 0 overwritten
    CHECK(check_slot(0, 2));
    CHECK(cache.device.Lookup(0) == -1);
    CHECK(cache.device.Lookup(2) == 0);

    // Global scales come from the host entry via the resolved view.
    cache.store.entries[0].w1_global_scale = 1.25f;
    cache.store.entries[0].w3_global_scale = 2.5f;
    cache.store.entries[0].w2_global_scale = 3.75f;
    auto v = ResolveGcExpertView(cache, 0, 0, nullptr, err);
    CHECK(v.gate_global_scale == 1.25f);
    CHECK(v.up_global_scale == 2.5f);
    CHECK(v.down_global_scale == 3.75f);

    FreeGcExpertCache(cache);
    CHECK(cache.device.d_base == nullptr);
    CHECK(cache.device.slots == 0);
}
