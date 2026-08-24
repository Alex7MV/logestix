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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "lsxcommon/qwen_moe_inference.h"

namespace {

// Host replica of the HF router semantics (fp32 softmax -> top-k -> renorm).
void HostRouter(const std::vector<__nv_bfloat16> &w,
                const std::vector<__half> &x, int seq, int E, int K, int H,
                std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> &out)
{
    out.assign(static_cast<size_t>(seq) * K, {});
    for (int t = 0; t < seq; ++t) {
        std::vector<float> logits(static_cast<size_t>(E));
        float m = -INFINITY;
        for (int e = 0; e < E; ++e) {
            float dot = 0.f;
            for (int d = 0; d < H; ++d)
                dot += __bfloat162float(
                           w[static_cast<size_t>(e) * H + static_cast<size_t>(d)]) *
                       __half2float(x[static_cast<size_t>(t) * H + static_cast<size_t>(d)]);
            logits[static_cast<size_t>(e)] = dot;
            m = fmaxf(m, dot);
        }
        std::vector<float> probs(static_cast<size_t>(E));
        float z = 0.f;
        for (int e = 0; e < E; ++e) {
            probs[static_cast<size_t>(e)] = expf(logits[static_cast<size_t>(e)] - m);
            z += probs[static_cast<size_t>(e)];
        }
        for (auto &p : probs) p /= z;

        int idx[16]; float val[16]; int n = 0;
        for (int e = 0; e < E; ++e) {
            float v = probs[static_cast<size_t>(e)];
            int pos = n;
            while (pos > 0 && (val[pos-1] < v || (val[pos-1] == v && idx[pos-1] > e))) {
                if (pos < K) { val[pos] = val[pos-1]; idx[pos] = idx[pos-1]; }
                --pos;
            }
            if (pos < K) { val[pos] = v; idx[pos] = e; if (n < K) ++n; }
        }
        float s = 0.f;
        for (int i = 0; i < K; ++i) s += val[i];
        for (int i = 0; i < K; ++i) {
            auto &slot = out[static_cast<size_t>(t) * K + static_cast<size_t>(i)];
            slot.eid = idx[i];
            slot.coeff = val[i] / s;
        }
    }
}

} // namespace

TEST_CASE("router top-k matches host replica", "[qwen-moe-router]")
{
    const int seq = 5, E = 8, K = 2, H = 64;
    // Deterministic weight-like inputs in a sane numeric range.
    std::vector<__nv_bfloat16> w(static_cast<size_t>(E) * H);
    std::vector<__half> x(static_cast<size_t>(seq) * H);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2bfloat16(((float)(i * 37 % 251) / 251.0f) - 0.5f);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2half(sinf((float)i) * 0.5f);

    __nv_bfloat16 *d_w;
    __half *d_x;
    lsxcommon::qwen_moe::QwenMoeRoutedSlot *d_slots;
    cudaMalloc(&d_w, w.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_slots, sizeof(lsxcommon::qwen_moe::QwenMoeRoutedSlot) *
                             static_cast<size_t>(seq) * K);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(stream, d_w, d_x, seq, E,
                                                      K, H, d_slots) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> got(
        static_cast<size_t>(seq) * K);
    cudaMemcpy(got.data(), d_slots, got.size() * sizeof(got[0]),
               cudaMemcpyDeviceToHost);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> want;
    HostRouter(w, x, seq, E, K, H, want);

    for (size_t i = 0; i < want.size(); ++i) {
        INFO("slot " << i);
        CHECK(got[i].eid == want[i].eid);
        CHECK(std::fabs(got[i].coeff - want[i].coeff) < 1e-6f);
    }

    // Determinism: repeat run must be bit-identical.
    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> again(
        static_cast<size_t>(seq) * K);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(stream, d_w, d_x, seq, E,
                                                      K, H, d_slots) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaMemcpy(again.data(), d_slots, again.size() * sizeof(again[0]),
               cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(got.data(), again.data(),
                      got.size() * sizeof(got[0])) == 0);

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_slots);
    cudaStreamDestroy(stream);
}
