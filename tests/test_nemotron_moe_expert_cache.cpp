// TDD — Nemotron-3.5 routed-expert host store + device LRU cache
// (Model Ingestion Protocol follow-up: ≤16 GiB VRAM residency). Slot layout,
// slot policy and slot-pointer arithmetic are pure host logic (no CUDA); the
// allocator/staging cases run on a device and assert the exact bytes that land
// in a slot (the bit-exactness contract: the GEMV must read identical packed
// weights whether they came from the resident path or a cache slot).
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/nemotron_moe_expert_cache.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace lsxcommon::nemotron_moe;
using lsxcommon::gpu::ExpertDeviceCache;

namespace {

void FillBytes(std::vector<uint8_t> &v, uint8_t base)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<uint8_t>(base + i);
}

} // namespace

TEST_CASE("nemotron expert slot layout: real geometry sections stay aligned",
          "[nemotron][expert-cache]")
{
    // Frozen geometry: up/down packed 1856*2688/2 = 2,494,464 B, scales
    // 1856*(2688/16) = 311,808 B (already 256-multiples).
    const auto l = PlanNmExpertSlot(2494464, 311808, 2494464, 311808);
    REQUIRE(l.ok);
    CHECK(l.up_pack_off == 0);
    CHECK(l.down_pack_off == 2494464);
    CHECK(l.pack_stride == 4988928);
    CHECK(l.up_scale_off == 0);
    CHECK(l.down_scale_off == 311808);
    CHECK(l.scale_stride == 623616);
    CHECK(NmExpertCachePerExpertBytes(l) == 5612544);
}

TEST_CASE("nemotron expert slot layout: odd sections round up to 256",
          "[nemotron][expert-cache]")
{
    const auto l = PlanNmExpertSlot(100, 40, 300, 7);
    REQUIRE(l.ok);
    CHECK(l.up_pack_off == 0);
    CHECK(l.down_pack_off == 256);
    CHECK(l.pack_stride == 256 + 512);   // 300 -> 512
    CHECK(l.up_scale_off == 0);
    CHECK(l.down_scale_off == 256);      // 40 -> 256
    CHECK(l.scale_stride == 256 + 256);  // 7 -> 256
}

TEST_CASE("nemotron expert slot layout: an empty section is invalid",
          "[nemotron][expert-cache]")
{
    CHECK_FALSE(PlanNmExpertSlot(0, 40, 300, 7).ok);
    CHECK_FALSE(PlanNmExpertSlot(100, 0, 300, 7).ok);
    CHECK_FALSE(PlanNmExpertSlot(100, 40, 0, 7).ok);
    CHECK_FALSE(PlanNmExpertSlot(100, 40, 300, 0).ok);
    CHECK(NmExpertCachePerExpertBytes(NmExpertSlotLayout{}) == 0);
}

TEST_CASE("nemotron slot pointers: pack then scales, per-slot stride",
          "[nemotron][expert-cache]")
{
    const auto l = PlanNmExpertSlot(256, 256, 256, 256);
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

    const auto v = NmExpertSlotPointers(c, 1, l);
    CHECK(v.up_packed == mem.data() + l.pack_stride);
    CHECK(v.down_packed == mem.data() + l.pack_stride + l.down_pack_off);
    CHECK(v.up_scales ==
          c.d_scale + l.scale_stride + l.up_scale_off);
    CHECK(v.down_scales ==
          c.d_scale + l.scale_stride + l.down_scale_off);
}

TEST_CASE("nemotron expert cache sizing: env override, VRAM budget, floor",
          "[nemotron][expert-cache]")
{
    const int total = 2944, topk = 6;
    const std::size_t per = 5612544;  // real geometry

    // Explicit override wins and clamps to the pool size.
    CHECK(NmExpertCacheSlots(0, per, total, topk, "128") == 128);
    CHECK(NmExpertCacheSlots(0, per, total, topk, "999999") == total);

    // 13 GiB free keeps a large fraction of the pool but not all of it.
    const int s16 = NmExpertCacheSlots(13ull << 30, per, total, topk, nullptr);
    CHECK(s16 == static_cast<int>(((13ull << 30) - (2ull << 30)) / per));
    CHECK(s16 > topk);
    CHECK(s16 < total);

    // Full 32 GiB board: everything resident (legacy behavior, no misses).
    CHECK(NmExpertCacheSlots(30ull << 30, per, total, topk, nullptr) == total);

    // A budget below topk floors at one layer's working set (the batch GEMV
    // needs topk distinct slots), never at zero.
    CHECK(NmExpertCacheSlots(256, per, total, topk, nullptr) == topk);
    CHECK(NmExpertCacheSlots(0, per, 3, 6, nullptr) == 3);

    // Unknown geometry degenerates to "everything resident".
    CHECK(NmExpertCacheSlots(1ull << 30, 0, total, topk, nullptr) == total);
}

TEST_CASE("nemotron expert staging writes up/down bytes into the slot",
          "[nemotron][expert-cache]")
{
    const auto l = PlanNmExpertSlot(256, 256, 256, 256);
    REQUIRE(l.ok);

    NmExpertCache cache;
    cache.layout = l;
    cache.per_expert_bytes = NmExpertCachePerExpertBytes(l);
    cache.experts_per_layer = 2;
    cache.total_experts = 4;
    cache.store.num_layers = 2;
    cache.store.experts_per_layer = 2;
    cache.store.entries.assign(4, NmExpertEntry{});

    std::vector<uint8_t> up_pack(256), up_scale(256), dn_pack(256), dn_scale(256);
    FillBytes(up_pack, 0x10);
    FillBytes(up_scale, 0x20);
    FillBytes(dn_pack, 0x30);
    FillBytes(dn_scale, 0x40);
    NmExpertEntry &e = cache.store.entries[2];  // layer 1, expert 0
    e.host_w3 = up_pack.data();
    e.w3_bytes = static_cast<int64_t>(up_pack.size());
    e.w3_scales_bytes = static_cast<int64_t>(up_scale.size());
    e.host_w3_scales = up_scale.data();
    e.w3_global_scale = 1.5f;
    e.host_w2 = dn_pack.data();
    e.w2_bytes = static_cast<int64_t>(dn_pack.size());
    e.w2_scales_bytes = static_cast<int64_t>(dn_scale.size());
    e.host_w2_scales = dn_scale.data();
    e.w2_global_scale = 2.5f;

    std::string err;
    REQUIRE(AllocateNmExpertCache(cache, /*slots=*/2, err));
    REQUIRE(err.empty());

    int slot = -1;
    bool missed = false;
    REQUIRE(StageNmExpert(cache, 1, 0, nullptr, err, &slot, &missed) ==
            cudaSuccess);
    CHECK(err.empty());
    CHECK(missed);
    CHECK(slot == 0);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);

    std::vector<uint8_t> got_pack(static_cast<size_t>(l.pack_stride), 0);
    std::vector<uint8_t> got_scale(static_cast<size_t>(l.scale_stride), 0);
    REQUIRE(cudaMemcpy(got_pack.data(), cache.device.d_pack, got_pack.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(got_scale.data(), cache.device.d_scale,
                       got_scale.size(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(std::memcmp(got_pack.data() + l.up_pack_off, up_pack.data(),
                      up_pack.size()) == 0);
    CHECK(std::memcmp(got_pack.data() + l.down_pack_off, dn_pack.data(),
                      dn_pack.size()) == 0);
    CHECK(std::memcmp(got_scale.data() + l.up_scale_off, up_scale.data(),
                      up_scale.size()) == 0);
    CHECK(std::memcmp(got_scale.data() + l.down_scale_off, dn_scale.data(),
                      dn_scale.size()) == 0);

    // The returned view points at the staged bytes; global scales come from
    // the host entry (not the device slot).
    const auto v = ResolveNmExpertView(cache, 1, 0, nullptr, err);
    CHECK(v.up_packed == cache.device.d_pack + l.up_pack_off);
    CHECK(v.down_scales == cache.device.d_scale + l.down_scale_off);
    CHECK(v.up_global_scale == 1.5f);
    CHECK(v.down_global_scale == 2.5f);

    // A second call hits the same slot without re-staging.
    int slot2 = -1;
    bool missed2 = true;
    REQUIRE(StageNmExpert(cache, 1, 0, nullptr, err, &slot2, &missed2) ==
            cudaSuccess);
    CHECK_FALSE(missed2);
    CHECK(slot2 == 0);
    CHECK(cache.device.hits == 2);  // Resolve view + this repeat both hit
    CHECK(cache.device.misses == 1);

    FreeNmExpertCache(cache);
    CHECK(cache.device.d_base == nullptr);
    CHECK(cache.device.slots == 0);
}

TEST_CASE("nemotron expert cache pre-warm fills slots on the load path",
          "[nemotron][expert-cache]")
{
    const auto l = PlanNmExpertSlot(256, 256, 256, 256);
    NmExpertCache cache;
    cache.layout = l;
    cache.per_expert_bytes = NmExpertCachePerExpertBytes(l);
    cache.experts_per_layer = 2;
    cache.total_experts = 4;
    cache.store.num_layers = 2;
    cache.store.experts_per_layer = 2;
    cache.store.entries.assign(4, NmExpertEntry{});
    std::vector<uint8_t> buf(256, 0x7e);
    for (auto &e : cache.store.entries) {
        e.host_w3 = buf.data();
        e.w3_bytes = 256;
        e.host_w3_scales = buf.data();
        e.w3_scales_bytes = 256;
        e.host_w2 = buf.data();
        e.w2_bytes = 256;
        e.host_w2_scales = buf.data();
        e.w2_scales_bytes = 256;
    }

    std::string err;
    REQUIRE(AllocateNmExpertCache(cache, /*slots=*/2, err));
    REQUIRE(PrewarmNmExpertCache(cache, nullptr, err) == cudaSuccess);
    CHECK(err.empty());
    // Both slots are resident after warm-up: the first decode token is all-hit.
    CHECK(cache.device.misses == 2);
    CHECK(cache.device.Lookup(0) == 0);
    CHECK(cache.device.Lookup(1) == 1);
    CHECK(cache.device.misses == 2);
    FreeNmExpertCache(cache);
}

TEST_CASE("nemotron expert staging evicts the LRU slot and re-stages bytes",
          "[nemotron][expert-cache]")
{
    const auto l = PlanNmExpertSlot(256, 256, 256, 256);
    REQUIRE(l.ok);

    constexpr int kExperts = 4;
    NmExpertCache cache;
    cache.layout = l;
    cache.per_expert_bytes = NmExpertCachePerExpertBytes(l);
    cache.experts_per_layer = kExperts;
    cache.total_experts = kExperts;
    cache.store.num_layers = 1;
    cache.store.experts_per_layer = kExperts;
    cache.store.entries.assign(kExperts, NmExpertEntry{});

    // Distinct payload per expert so a mis-staged (stale) slot is detectable.
    std::vector<std::vector<uint8_t>> up(kExperts), dn(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        up[static_cast<size_t>(e)].assign(256, static_cast<uint8_t>(0x10 + e));
        dn[static_cast<size_t>(e)].assign(256, static_cast<uint8_t>(0xa0 + e));
        NmExpertEntry &en = cache.store.entries[static_cast<size_t>(e)];
        en.host_w3 = up[static_cast<size_t>(e)].data();
        en.w3_bytes = 256;
        en.host_w3_scales = up[static_cast<size_t>(e)].data();
        en.w3_scales_bytes = 256;
        en.host_w2 = dn[static_cast<size_t>(e)].data();
        en.w2_bytes = 256;
        en.host_w2_scales = dn[static_cast<size_t>(e)].data();
        en.w2_scales_bytes = 256;
    }

    std::string err;
    REQUIRE(AllocateNmExpertCache(cache, /*slots=*/2, err));

    auto check_slot = [&](int slot, int expert) {
        std::vector<uint8_t> pack(static_cast<size_t>(l.pack_stride), 0);
        REQUIRE(cudaMemcpy(pack.data(),
                           cache.device.d_pack +
                               static_cast<int64_t>(slot) * l.pack_stride,
                           pack.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        return pack[static_cast<size_t>(l.up_pack_off)] ==
                   up[static_cast<size_t>(expert)][0] &&
               pack[static_cast<size_t>(l.down_pack_off)] ==
                   dn[static_cast<size_t>(expert)][0];
    };
    auto stage = [&](int expert) {
        int slot = -1;
        REQUIRE(StageNmExpert(cache, 0, expert, nullptr, err, &slot) ==
                cudaSuccess);
        REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
        return slot;
    };

    CHECK(stage(0) == 0);
    CHECK(check_slot(0, 0));
    CHECK(stage(1) == 1);
    CHECK(check_slot(1, 1));
    // Slot 0 is the LRU victim and must be overwritten with expert 2's bytes.
    CHECK(stage(2) == 0);
    CHECK(check_slot(0, 2));
    CHECK(cache.device.Lookup(0) == -1);  // expert 0 evicted
    CHECK(cache.device.Lookup(2) == 0);
    // Evict slot 1, then recycle slot 0 again.
    CHECK(stage(3) == 1);
    CHECK(check_slot(1, 3));
    CHECK(stage(0) == 0);
    CHECK(check_slot(0, 0));
    FreeNmExpertCache(cache);
}
