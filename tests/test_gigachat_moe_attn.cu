// TDD — GigaChat3.5 MLA decode (Plan 2b Task 2). Bit-exact interleaved YaRN
// RoPE, then full MLA-block parity against an F32 host reference. NVFP4 weights
// are built on the E2M1 grid with unit per-block scales so dequant is exact.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_host_ref.h"
#include "gigachat_moe_device_math.h"
#include "lsxcommon/gigachat_moe_inference_attn.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <lsxcommon/lsx_inference_nvfp4.h>

namespace lsx_nvfp4 = lsxcommon::lsx_nvfp4;

using namespace lsxcommon::gigachat;
using namespace gigachatref;

namespace {

const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float GridVal(int i)
{
    const float m = kE2m1[static_cast<size_t>(i) % 8];
    return ((i / 8) % 2) ? -m : m;
}

uint8_t E2m1Nibble(float v)
{
    const float a = std::fabs(v);
    int code = 0;
    if (a > 5.0f) code = 7;
    else if (a > 3.5f) code = 6;
    else if (a > 2.5f) code = 5;
    else if (a > 1.75f) code = 4;
    else if (a > 1.25f) code = 3;
    else if (a > 0.75f) code = 2;
    else if (a > 0.25f) code = 1;
    return static_cast<uint8_t>(code | (v < 0.0f ? 8 : 0));
}

struct Nvfp4Host {
    std::vector<uint8_t> packed, scales;
};

Nvfp4Host PackNvfp4(const std::vector<float> &w, int N, int K)
{
    Nvfp4Host o;
    o.packed.assign(static_cast<size_t>(N) * K / 2, 0);
    o.scales.assign(static_cast<size_t>(N) * K / 16, 0x38);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; k += 2)
        {
            const uint8_t lo = E2m1Nibble(w[static_cast<size_t>(n) * K + k]);
            const uint8_t hi =
                E2m1Nibble(w[static_cast<size_t>(n) * K + k + 1]);
            o.packed[static_cast<size_t>(n) * K / 2 + k / 2] =
                static_cast<uint8_t>(lo | (hi << 4));
        }
    return o;
}

template <typename T>
T *DevCopy(const std::vector<T> &v)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, v.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), v.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

lsx_nvfp4::LsxNvfp4Weight DevNvfp4(const std::vector<float> &w, int N, int K)
{
    Nvfp4Host p = PackNvfp4(w, N, K);
    lsx_nvfp4::LsxNvfp4Weight out;
    out.rows = N;
    out.cols = K;
    out.global_scale = 1.0f;
    out.packed_bytes = static_cast<int64_t>(p.packed.size());
    out.scale_bytes = static_cast<int64_t>(p.scales.size());
    out.d_packed = DevCopy(p.packed);
    out.d_scales = DevCopy(p.scales);
    return out;
}

uint8_t *DevBf16(const std::vector<float> &v)
{
    std::vector<__nv_bfloat16> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) b[i] = __float2bfloat16_rn(v[i]);
    std::vector<uint8_t> bytes(v.size() * 2);
    std::memcpy(bytes.data(), b.data(), bytes.size());
    return DevCopy(bytes);
}

GcGatedNorm DevNorm(const std::vector<float> &w, const std::vector<float> &up,
                    const std::vector<float> &down, int dim, int rank)
{
    std::vector<__nv_bfloat16> bw(w.size()), bu(up.size()), bd(down.size());
    for (size_t i = 0; i < w.size(); ++i) bw[i] = __float2bfloat16_rn(w[i]);
    for (size_t i = 0; i < up.size(); ++i) bu[i] = __float2bfloat16_rn(up[i]);
    for (size_t i = 0; i < down.size(); ++i) bd[i] = __float2bfloat16_rn(down[i]);
    std::vector<uint8_t> uw(w.size() * 2), uu(up.size() * 2), ud(down.size() * 2);
    std::memcpy(uw.data(), bw.data(), uw.size());
    std::memcpy(uu.data(), bu.data(), uu.size());
    std::memcpy(ud.data(), bd.data(), ud.size());
    GcGatedNorm n;
    n.dim = dim;
    n.rank = rank;
    n.d_weight = DevCopy(uw);
    n.d_gate_up = DevCopy(uu);
    n.d_gate_down = DevCopy(ud);
    return n;
}

} // namespace

TEST_CASE("gigachat MLA interleaved rope is bit-exact", "[gigachat][attn]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const int heads = 3, qk = 16, nope = 8, rope = 8, half = rope / 2;
    std::vector<float> q(static_cast<size_t>(heads) * qk), k(rope);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = 0.125f * static_cast<float>(static_cast<int>(i % 23) - 11);
    for (int i = 0; i < rope; ++i) k[i] = 0.2f * static_cast<float>(i - 3);
    std::vector<float> c(half), s(half);
    for (int j = 0; j < half; ++j)
    {
        c[j] = std::cos(0.37f * j + 0.1f);
        s[j] = std::sin(0.37f * j + 0.1f);
    }
    std::vector<float> qr = q, kr = k;
    RopeRef(qr, heads, qk, nope, kr, c, s);

    float *dq = DevCopy(q);
    float *dk = DevCopy(k);
    float *dc = DevCopy(c);
    float *ds = DevCopy(s);
    REQUIRE(RunGigaChatRopeQk(dq, dk, heads, qk, nope, rope, dc, ds, nullptr) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> hq(q.size()), hk(k.size());
    REQUIRE(cudaMemcpy(hq.data(), dq, hq.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(hk.data(), dk, hk.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    for (size_t i = 0; i < hq.size(); ++i) CHECK(hq[i] == qr[i]);
    for (size_t i = 0; i < hk.size(); ++i) CHECK(hk[i] == kr[i]);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dc);
    cudaFree(ds);
}

TEST_CASE("gigachat MLA decode matches the host reference",
          "[gigachat][attn]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);

    const int hidden = 32, heads = 2, q_lora = 8, kv_lora = 8, nope = 8,
              rope = 8, vh = 8;
    const int qk = nope + rope, kvw = kv_lora + rope;
    const int kvb = heads * (nope + vh), hv = heads * vh;
    const GcMlaDims d{hidden, heads, q_lora, kv_lora, nope, rope, vh};

    auto gen = [](int n, int seed) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) v[i] = GridVal(i * 5 + seed);
        return v;
    };
    // Weights [N, K].
    std::vector<float> w_qa = gen(q_lora * hidden, 1);
    std::vector<float> w_qb = gen(heads * qk * q_lora, 2);
    std::vector<float> w_kva = gen(kvw * hidden, 3);
    std::vector<float> w_kvb = gen(kvb * kv_lora, 4);
    std::vector<float> w_gate = gen(hv * hidden, 5);
    std::vector<float> w_o = gen(hidden * hv, 6);
    // Norm triples: gate_up [rank, dim], gate_down [dim, rank].
    const int rank = 2;
    std::vector<float> qn_w = gen(q_lora, 7);
    std::vector<float> qn_up = gen(rank * q_lora, 8);
    std::vector<float> qn_down = gen(q_lora * rank, 9);
    std::vector<float> kn_w = gen(kv_lora, 10);
    std::vector<float> kn_up = gen(rank * kv_lora, 11);
    std::vector<float> kn_down = gen(kv_lora * rank, 12);

    GcLayerWeights lw;
    lw.q_a = DevNvfp4(w_qa, q_lora, hidden);
    lw.q_b = DevNvfp4(w_qb, heads * qk, q_lora);
    lw.kv_a = DevNvfp4(w_kva, kvw, hidden);
    lw.kv_b = DevNvfp4(w_kvb, kvb, kv_lora);
    lw.attn_gate = DevNvfp4(w_gate, hv, hidden);
    lw.o_proj = DevNvfp4(w_o, hidden, hv);
    lw.q_a_norm = DevNorm(qn_w, qn_up, qn_down, q_lora, rank);
    lw.kv_a_norm = DevNorm(kn_w, kn_up, kn_down, kv_lora, rank);

    const int capacity = 4;
    const int half = rope / 2;
    std::vector<float> rc(static_cast<size_t>(capacity) * half),
        rs(static_cast<size_t>(capacity) * half);
    for (int p = 0; p < capacity; ++p)
        for (int j = 0; j < half; ++j)
        {
            rc[static_cast<size_t>(p) * half + j] = std::cos(0.23f * (p + 1) + j);
            rs[static_cast<size_t>(p) * half + j] = std::sin(0.23f * (p + 1) + j);
        }
    std::vector<float> x0(static_cast<size_t>(hidden)),
        x1(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
    {
        x0[i] = 0.25f * static_cast<float>((i % 7) - 3);
        x1[i] = 0.2f * static_cast<float>((i % 5) - 2);
    }

    const float eps = 1e-6f, scale = 0.55f;
    float *kx = DevCopy(x0), *vx = DevCopy(x1);
    GcMlaScratch sc;
    REQUIRE(sc.Alloc(d, capacity));
    float *k_cache = nullptr, *v_cache = nullptr;
    REQUIRE(cudaMalloc(&k_cache, static_cast<size_t>(capacity) * heads * qk * 4) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&v_cache, static_cast<size_t>(capacity) * heads * vh * 4) ==
            cudaSuccess);
    float *d_rc = DevCopy(rc), *d_rs = DevCopy(rs);
    float *out0 = nullptr, *out1 = nullptr;
    REQUIRE(cudaMalloc(&out0, hidden * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&out1, hidden * 4) == cudaSuccess);
    REQUIRE(RunGigaChatMlaDecode(kx, lw, d, eps, scale, d_rc, d_rs, 0, capacity,
                                 k_cache, v_cache, sc, out0, nullptr) ==
            cudaSuccess);
    REQUIRE(RunGigaChatMlaDecode(vx, lw, d, eps, scale, d_rc, d_rs, 1, capacity,
                                 k_cache, v_cache, sc, out1, nullptr) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> hout(static_cast<size_t>(hidden));
    REQUIRE(cudaMemcpy(hout.data(), out1, hidden * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    // ── Host reference (shared header). Mirrors each F16 rounding. ──
    RefMla rmla;
    rmla.q_a = {q_lora, hidden, w_qa};
    rmla.q_b = {heads * qk, q_lora, w_qb};
    rmla.kv_a = {kvw, hidden, w_kva};
    rmla.kv_b = {kvb, kv_lora, w_kvb};
    rmla.gate = {hv, hidden, w_gate};
    rmla.o_proj = {hidden, hv, w_o};
    rmla.q_a_norm = {q_lora, rank, qn_w, qn_up, qn_down};
    rmla.kv_a_norm = {kv_lora, rank, kn_w, kn_up, kn_down};
    std::vector<float> k_cache_ref(static_cast<size_t>(capacity) * heads * qk,
                                   0.0f);
    std::vector<float> v_cache_ref(static_cast<size_t>(capacity) * heads * vh,
                                   0.0f);
    std::vector<float> ref;
    for (int p = 0; p < 2; ++p)
    {
        std::vector<float> out_ref;
        MlaDecodeRef(rmla, d, eps, scale, rc, rs, p, k_cache_ref, v_cache_ref,
                     p == 0 ? x0 : x1, out_ref);
        ref = out_ref;
    }

    for (int i = 0; i < hidden; ++i)
        CHECK(hout[static_cast<size_t>(i)] ==
              Catch::Approx(ref[static_cast<size_t>(i)]).margin(1e-3f));

    sc.Free();
    cudaFree(k_cache);
    cudaFree(v_cache);
    cudaFree(out0);
    cudaFree(out1);
    cudaFree(kx);
    cudaFree(vx);
    cudaFree(d_rc);
    cudaFree(d_rs);
}

TEST_CASE("gigachat GatedDeltaNet decode matches the host reference",
          "[gigachat][attn]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);

    const int hidden = 32, k_heads = 1, v_heads = 2, hk = 16, hv = 16,
              conv_k = 3;
    const int key_dim = k_heads * hk, value_dim = v_heads * hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int qkvz_dim = 2 * key_dim + 2 * value_dim;
    const int ba_dim = 2 * v_heads, rep = v_heads / k_heads;
    const int space = 2 * hk + 2 * hv * rep;
    const GcGdnDims d{hidden, k_heads, v_heads, hk, hv, conv_k};

    auto gen = [](int n, int seed) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) v[i] = GridVal(i * 5 + seed);
        return v;
    };
    auto small = [](int n, int seed) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            v[i] = 0.25f * static_cast<float>((i * 3 + seed) % 7 - 3);
        return v;
    };
    std::vector<float> w_qkvz = gen(qkvz_dim * hidden, 1);
    std::vector<float> w_ba = gen(ba_dim * hidden, 2);
    std::vector<float> w_out = gen(hidden * value_dim, 3);
    std::vector<float> w_conv = small(conv_dim * conv_k, 4);
    std::vector<float> A_log = small(v_heads, 5);
    std::vector<float> dt_bias = small(v_heads, 6);
    std::vector<float> w_norm = gen(hv, 7);

    GcLayerWeights lw;
    lw.gdn_in_qkvz = DevNvfp4(w_qkvz, qkvz_dim, hidden);
    lw.gdn_out_proj = DevNvfp4(w_out, hidden, value_dim);
    lw.d_gdn_in_ba = DevBf16(w_ba);
    lw.gdn_ba_rows = ba_dim;
    lw.d_conv1d = reinterpret_cast<const uint8_t *>(DevCopy(w_conv));
    lw.conv_dim = conv_dim;
    lw.conv_kernel = conv_k;
    lw.d_A_log = reinterpret_cast<const uint8_t *>(DevCopy(A_log));
    lw.d_dt_bias = reinterpret_cast<const uint8_t *>(DevCopy(dt_bias));
    lw.gdn_value_heads = v_heads;
    lw.d_gdn_norm = DevBf16(w_norm);
    lw.gdn_norm_dim = hv;

    std::vector<float> x0(static_cast<size_t>(hidden)),
        x1(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
    {
        x0[i] = 0.2f * static_cast<float>((i % 6) - 3);
        x1[i] = 0.3f * static_cast<float>((i % 4) - 2);
    }
    float *dx0 = DevCopy(x0), *dx1 = DevCopy(x1);
    float *conv_state = nullptr, *rec_state = nullptr;
    REQUIRE(cudaMalloc(&conv_state,
                       static_cast<size_t>(conv_dim) * (conv_k - 1) * 4) ==
            cudaSuccess);
    REQUIRE(cudaMemset(conv_state, 0,
                       static_cast<size_t>(conv_dim) * (conv_k - 1) * 4) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&rec_state,
                       static_cast<size_t>(v_heads) * hk * hv * 4) ==
            cudaSuccess);
    REQUIRE(cudaMemset(rec_state, 0,
                       static_cast<size_t>(v_heads) * hk * hv * 4) ==
            cudaSuccess);
    GcGdnScratch sc;
    REQUIRE(sc.Alloc(d));
    float *out0 = nullptr, *out1 = nullptr;
    REQUIRE(cudaMalloc(&out0, hidden * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&out1, hidden * 4) == cudaSuccess);
    const float eps = 1e-6f, onorm_scale = 2.0f;
    REQUIRE(RunGigaChatGdnDecode(dx0, lw, d, eps, onorm_scale, conv_state,
                                 rec_state, sc, out0, nullptr) == cudaSuccess);
    REQUIRE(RunGigaChatGdnDecode(dx1, lw, d, eps, onorm_scale, conv_state,
                                 rec_state, sc, out1, nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> hout(static_cast<size_t>(hidden));
    REQUIRE(cudaMemcpy(hout.data(), out1, hidden * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    // ── Host reference (shared header), mirroring each F16 rounding. ──
    RefGdn rg;
    rg.in_qkvz = {qkvz_dim, hidden, w_qkvz};
    rg.out_proj = {hidden, value_dim, w_out};
    rg.in_ba = w_ba;
    rg.conv = w_conv;
    rg.A_log = A_log;
    rg.dt_bias = dt_bias;
    rg.norm = w_norm;
    rg.conv_dim = conv_dim;
    rg.conv_k = conv_k;
    rg.ba_rows = ba_dim;
    rg.v_heads = v_heads;
    std::vector<float> conv_state_ref(static_cast<size_t>(conv_dim) * (conv_k - 1),
                                      0.0f);
    std::vector<float> rec_state_ref(static_cast<size_t>(v_heads) * hk * hv,
                                     0.0f);
    std::vector<float> r1;
    for (int p = 0; p < 2; ++p)
    {
        std::vector<float> out_ref;
        GdnDecodeRef(rg, d, eps, onorm_scale, conv_state_ref, rec_state_ref,
                     p == 0 ? x0 : x1, out_ref);
        r1 = out_ref;
    }
    for (int i = 0; i < hidden; ++i)
        CHECK(hout[static_cast<size_t>(i)] ==
              Catch::Approx(r1[static_cast<size_t>(i)]).margin(1e-3f));

    sc.Free();
    cudaFree(conv_state);
    cudaFree(rec_state);
    cudaFree(dx0);
    cudaFree(dx1);
    cudaFree(out0);
    cudaFree(out1);
}
