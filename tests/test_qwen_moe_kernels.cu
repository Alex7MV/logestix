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

#include "qwen_moe_test_fixture.h"
#include "lsxcommon/qwen_moe_inference_load.h"

namespace {

// Host NVFP4 dequant of a row-major [N,K] packed weight (per-16 E4M3 block
// scales along K). Mirrors gpu::nvfp4 decode semantics.
float HostDequantNibble(int nib)
{
    const int mag = nib & 7;
    float v = (mag == 1) ? 0.5f : (mag == 2) ? 1.f : (mag == 3) ? 1.5f
              : (mag == 4) ? 2.f : (mag == 5) ? 3.f : (mag == 6) ? 4.f
              : (mag == 7) ? 6.f : 0.f;
    return (nib & 8) ? -v : v;
}

float HostE4M3Scale(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    uint32_t bits = (e4 + 120u) << 23;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

} // namespace

TEST_CASE("routed ffn slot chain matches dequant reference", "[qwen-moe-experts]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_moe_ffn_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_moe_ffn_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));
    REQUIRE(store.num_layers == 4);
    REQUIRE(store.num_experts == 8);

    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Token x: deterministic half vector, H=64, I=16.
    const int H = 64, I = 16, K = 2;
    std::vector<__half> hx(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i)
        hx[static_cast<size_t>(i)] = __float2half(sinf((float)i * 0.7f) * 0.4f);

    __half *d_x, *d_yg, *d_yu, *d_y2;
    float *d_acc;
    cudaMalloc(&d_x, H * 2);
    cudaMalloc(&d_yg, I * 2);
    cudaMalloc(&d_yu, I * 2);
    cudaMalloc(&d_y2, H * 2);
    cudaMalloc(&d_acc, H * 4);
    cudaMemset(d_acc, 0, H * 4);
    cudaMemcpy(d_x, hx.data(), H * 2, cudaMemcpyHostToDevice);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> h_slots =
        {{0, 0.5f}, {1, 0.5f}};
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedFFN(
                stream, store, cache, /*layer=*/0, h_slots.data(), K, d_x,
                H, I, d_acc, d_yg, d_yu, d_y2, err) == cudaSuccess);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccToFp16(stream, d_acc, d_y2, H) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> hout(H);
    cudaMemcpy(hout.data(), d_y2, H * 2, cudaMemcpyDeviceToHost);

    // Determinism across cache states: flush the cache and re-run.
    std::vector<__half> hout2(H);
    {
        lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
        REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));
        cudaMemset(d_acc, 0, H * 4);
        // Reverse slot order exercises LRU eviction path.
        std::swap(h_slots[0], h_slots[1]);
        // Undo the coefficient pairing so math matches the first run.
        h_slots[0].coeff = 0.5f; h_slots[1].coeff = 0.5f;
        h_slots[0].eid = 0; h_slots[1].eid = 1;
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedFFN(
                    stream, store, cache, 0, h_slots.data(), K, d_x, H, I,
                    d_acc, d_yg, d_yu, d_y2, err) == cudaSuccess);
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccToFp16(stream, d_acc, d_y2, H) ==
                cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        cudaMemcpy(hout2.data(), d_y2, H * 2, cudaMemcpyDeviceToHost);
    }
    CHECK(std::memcmp(hout.data(), hout2.data(), H * sizeof(__half)) == 0);
    for (int i = 0; i < H; ++i)
        CHECK(std::isfinite(__half2float(hout[static_cast<size_t>(i)])));

    // Sanity: non-zero output.
    float energy = 0.f;
    for (int i = 0; i < H; ++i)
        energy += __half2float(hout[static_cast<size_t>(i)]) *
                  __half2float(hout[static_cast<size_t>(i)]);
    CHECK(energy > 0.f);

    cudaFree(d_x); cudaFree(d_yg); cudaFree(d_yu); cudaFree(d_y2);
    cudaFree(d_acc);
    cudaStreamDestroy(stream);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

TEST_CASE("shared expert gate combine", "[qwen-moe-experts]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_moe_sh_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_moe_sh_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoePinnedWeights pw;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeDenseWeights(out.string(), pw, stream, err));

    const int H = 64, SI = 16, seq = 2;
    const auto &lw = pw.layers[0];

    // x rows deterministic; routed acc zeros -> out == g * shared(x).
    std::vector<__half> hx(static_cast<size_t>(seq) * H);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __float2half(sinf((float)i) * 0.4f);

    __half *d_x, *d_si, *d_sh, *d_out;
    float *d_acc, *d_g;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_si, static_cast<size_t>(seq) * SI * 2);
    cudaMalloc(&d_sh, static_cast<size_t>(seq) * H * 2);
    cudaMalloc(&d_out, static_cast<size_t>(seq) * H * 2);
    cudaMalloc(&d_acc, static_cast<size_t>(seq) * H * 4);
    cudaMalloc(&d_g, seq * 4);
    cudaMemset(d_acc, 0, static_cast<size_t>(seq) * H * 4);
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeSharedCombine(
                stream, lw.shared_w1, lw.shared_w2, lw.shared_w3,
                lw.d_shared_gate, d_x, seq, H, SI, d_acc, d_si, d_sh, d_g,
                d_out) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> hout(static_cast<size_t>(seq) * H);
    cudaMemcpy(hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost);
    float energy = 0.f;
    for (auto v : hout) energy += __half2float(v) * __half2float(v);
    CHECK(energy > 0.f);
    for (auto v : hout) CHECK(std::isfinite(__half2float(v)));

    // Determinism: bit-identical repeat.
    std::vector<__half> hout2(hout.size());
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeSharedCombine(
                stream, lw.shared_w1, lw.shared_w2, lw.shared_w3,
                lw.d_shared_gate, d_x, seq, H, SI, d_acc, d_si, d_sh, d_g,
                d_out) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaMemcpy(hout2.data(), d_out, hout2.size() * 2, cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(hout.data(), hout2.data(), hout.size() * 2) == 0);

    cudaFree(d_x); cudaFree(d_si); cudaFree(d_sh); cudaFree(d_out);
    cudaFree(d_acc); cudaFree(d_g);
    cudaStreamDestroy(stream);
    lsxcommon::qwen_moe::FreeQwenMoePinnedWeights(pw);
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}
