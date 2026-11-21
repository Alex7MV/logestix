// Inkling-2-Mini kernel + session tests (CUDA TU, -fmad=false for bit-exact
// parity against the host references).
//
// Task 6 cases pin the session plumbing: per-pass KV slots survive growth
// (the EnsureCapacity over-read trap), Reset zeroes every conv slot, the
// geometry is pos-major with CacheLayers() slots. Task 7 adds the numeric
// kernel cases (MessageGate, message mix, and the inkling2 arithmetic at Mini
// geometry).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "lsxcommon/inkling_moe_kernels.h"
#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/inkling_moe_loop_load.h"

using Catch::Approx;

using lsxcommon::inkling::InklingLoopScratch;
using lsxcommon::inkling::InklingLoopSession;
using lsxcommon::inkling::InklingLoopTopology;

namespace {

// Synthetic topology matching the converter fixture's shape (8 layers, loop
// [2,6) x 2 passes -> 12 cache slots).
InklingLoopTopology MakeTop()
{
    InklingLoopTopology top;
    top.num_layers = 8;
    top.hidden = 64;
    top.vocab = 256;
    top.unpadded_vocab = 200;
    top.num_heads = 4;
    top.num_kv_heads = 2;
    top.head_dim = 16;
    top.d_rel = 4;
    top.rel_extent = 8;
    top.sliding_window = 8;
    top.conv_kernel = 4;
    top.attn_res_block = 2;
    top.num_experts = 4;
    top.top_k = 2;
    top.moe_intermediate = 32;
    top.loop_start = 2;
    top.loop_end = 6;
    top.loop_passes = 2;
    top.loop_gate_rank = 8;
    top.full_layers = {5};
    top.sliding_layers = {0, 1, 2, 3, 4, 6, 7};
    top.expert_nvfp4 = {true, true, true, true, true, true, false, false};
    top.expert_fp8 = {false, false, false, false, false, false, true, true};
    return top;
}

}  // namespace

TEST_CASE("inkling loop session growth keeps every slot's rows intact",
          "[inkling][kernels]")
{
    const InklingLoopTopology top = MakeTop();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    InklingLoopSession session;
    REQUIRE(session.EnsureCapacity(top, 4, stream) == cudaSuccess);
    REQUIRE(session.capacity >= 4);
    REQUIRE(session.geom.layers == top.CacheLayers());
    REQUIRE(session.geom.layers == 12);
    // Pos-major: pos_stride = kv_heads*head_dim, head_stride = head_dim.
    REQUIRE(session.geom.pos_stride == top.num_kv_heads * top.head_dim);
    REQUIRE(session.geom.head_stride == top.head_dim);

    // Write a distinct pattern into every cache slot's first 3 positions.
    // Values stay below 2048 so every integer is exact in FP16 (a larger
    // pattern would round and the equality check would test the codec, not
    // the copy).
    const int64_t rows = 3;
    const int64_t row_elems = session.geom.pos_stride;
    std::vector<__half> pattern(static_cast<size_t>(rows * row_elems));
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        for (int64_t i = 0; i < rows * row_elems; ++i)
            pattern[static_cast<size_t>(i)] =
                __float2half(static_cast<float>(slot * 100 + i + 1));
        REQUIRE(cudaMemcpyAsync(
                    session.d_k + slot * session.geom.LayerStride(),
                    pattern.data(), pattern.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(
                    session.d_v + slot * session.geom.LayerStride(),
                    pattern.data(), pattern.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream) == cudaSuccess);
    }
    session.kv_rows = static_cast<int>(rows);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Grow: the copy must move exactly kv_rows * pos_stride per slot — not
    // kv_rows * LayerStride(), which over-reads the old allocation (the Small
    // defect that surfaced as cudaErrorInvalidValue on the NEXT request).
    REQUIRE(session.EnsureCapacity(top, 9, stream) == cudaSuccess);
    REQUIRE(session.capacity >= 9);
    REQUIRE(session.kv_rows == rows);

    std::vector<__half> back(pattern.size());
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        // Rebuild THIS slot's expected pattern (the write loop above left
        // `pattern` holding the last slot's values).
        for (int64_t i = 0; i < rows * row_elems; ++i)
            pattern[static_cast<size_t>(i)] =
                __float2half(static_cast<float>(slot * 100 + i + 1));
        REQUIRE(cudaMemcpy(back.data(),
                           session.d_k + slot * session.geom.LayerStride(),
                           back.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < pattern.size(); ++i) {
            INFO("slot " << slot << " k elem " << i);
            REQUIRE(__half2float(back[i]) == __half2float(pattern[i]));
        }
        REQUIRE(cudaMemcpy(back.data(),
                           session.d_v + slot * session.geom.LayerStride(),
                           back.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < pattern.size(); ++i) {
            INFO("slot " << slot << " v elem " << i);
            REQUIRE(__half2float(back[i]) == __half2float(pattern[i]));
        }
    }
    session.Free();
    cudaStreamDestroy(stream);
}

TEST_CASE("inkling loop session Reset zeroes every conv slot",
          "[inkling][kernels]")
{
    const InklingLoopTopology top = MakeTop();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    InklingLoopSession session;
    REQUIRE(session.EnsureCapacity(top, 4, stream) == cudaSuccess);
    REQUIRE(session.conv_state != nullptr);
    const int64_t kv_dim = top.num_kv_heads * top.head_dim;
    const int64_t want_floats =
        2 * top.conv_kernel * kv_dim + 2 * top.conv_kernel * top.hidden;
    REQUIRE(session.conv_slot_floats == want_floats);
    REQUIRE(session.conv_state_bytes ==
            top.CacheLayers() * want_floats * static_cast<int64_t>(sizeof(float)));

    // Dirty every slot, then Reset: all 12 rings must come back zero.
    std::vector<float> dirty(static_cast<size_t>(want_floats), 7.0f);
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        REQUIRE(cudaMemcpyAsync(session.ConvSlot(slot), dirty.data(),
                                dirty.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream) == cudaSuccess);
    }
    session.kv_rows = 3;
    session.last_tok = 42;
    session.gen_tail = {1, 2, 3};
    session.Reset(stream);
    REQUIRE(session.kv_rows == 0);
    REQUIRE(session.last_tok == -1);
    REQUIRE(session.gen_tail.empty());
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> back(static_cast<size_t>(want_floats));
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        REQUIRE(cudaMemcpy(back.data(), session.ConvSlot(slot),
                           back.size() * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < back.size(); ++i) {
            INFO("slot " << slot << " float " << i);
            REQUIRE(back[i] == 0.0f);
        }
    }
    session.Free();
    REQUIRE(session.conv_state == nullptr);
    cudaStreamDestroy(stream);
}

// ── numeric kernel cases (Task 7) ──────────────────────────────────────────
// Host-reference parity for the loop deltas (MessageGate, message retrieval)
// and for the shared inkling2 arithmetic at Mini-relevant geometry (GQA 3,
// hidden 2304-class widths, the MXFP8 expert plane). The references mirror
// transformers 5.24 models/inkling2_loop exactly (see the stage-0 trail).

namespace {

bool HaveCuda()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

float F(__half h) { return __half2float(h); }
__half H(float v) { return __float2half(v); }

std::vector<float> FPattern(int64_t n, float scale, int64_t seed)
{
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] =
            static_cast<float>(static_cast<int64_t>((i * 7 + seed * 13) % 23) - 11) *
            scale;
    return v;
}

std::vector<__half> HPattern(int64_t n, float scale, int64_t seed)
{
    const auto f = FPattern(n, scale, seed);
    std::vector<__half> v(static_cast<size_t>(n));
    for (size_t i = 0; i < f.size(); ++i) v[i] = H(f[i]);
    return v;
}

template <typename T>
T *ToDevice(const std::vector<T> &v)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, sizeof(T) * v.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), sizeof(T) * v.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

template <typename T>
void FromDevice(const T *d, std::vector<T> &v)
{
    REQUIRE(cudaMemcpy(v.data(), d, sizeof(T) * v.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
}

void Sync()
{
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

double Softcap(double v, double cap) { return std::tanh(v / cap) * cap; }

// Unweighted RMSNorm (FP32 square mean, rsqrt(mean+eps)).
std::vector<float> RefUnweightedNorm(const std::vector<float> &x, int rows,
                                     int dim, float eps)
{
    std::vector<float> out(x.size());
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) {
            const double v = x[static_cast<size_t>(r) * dim + i];
            ss += v * v;
        }
        const float inv = static_cast<float>(
            1.0 / std::sqrt(ss / dim + static_cast<double>(eps)));
        for (int i = 0; i < dim; ++i)
            out[static_cast<size_t>(r) * dim + i] =
                x[static_cast<size_t>(r) * dim + i] * inv;
    }
    return out;
}

// MessageGate host reference: the release's Inkling2LoopMessageGate.
// down is FP16 [rank, 2*hidden]; the concat is cast to FP16 before the dot
// (the reference casts to the down weight's dtype); gates and blend FP32.
std::vector<float> RefMessageGate(const std::vector<float> &h,
                                  const std::vector<float> &msg,
                                  const std::vector<__half> &down,
                                  const std::vector<float> &forget,
                                  const std::vector<float> &gain,
                                  int hidden, int rank, float eps)
{
    const auto nh = RefUnweightedNorm(h, 1, hidden, eps);
    const auto nm = RefUnweightedNorm(msg, 1, hidden, eps);
    std::vector<float> g(static_cast<size_t>(rank));
    for (int r = 0; r < rank; ++r) {
        double acc = 0.0;
        for (int c = 0; c < 2 * hidden; ++c) {
            const float xv = c < hidden ? nh[static_cast<size_t>(c)]
                                        : nm[static_cast<size_t>(c - hidden)];
            acc += static_cast<double>(
                       F(down[static_cast<size_t>(r) * (2 * hidden) + c])) *
                   static_cast<double>(H(xv));  // cast to the weight dtype
        }
        g[static_cast<size_t>(r)] = static_cast<float>(acc);
    }
    std::vector<float> out(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) {
        double f = 0.0, gn = 0.0;
        for (int r = 0; r < rank; ++r) {
            f += static_cast<double>(forget[static_cast<size_t>(i) * rank + r]) *
                 g[static_cast<size_t>(r)];
            gn += static_cast<double>(gain[static_cast<size_t>(i) * rank + r]) *
                  g[static_cast<size_t>(r)];
        }
        out[static_cast<size_t>(i)] = static_cast<float>(
            (1.0 - std::tanh(f)) * h[static_cast<size_t>(i)] +
            std::tanh(gn) * msg[static_cast<size_t>(i)]);
    }
    return out;
}

// Depth-mix reference (also the message-retrieval contract).
std::vector<float> RefDepthMix(const std::vector<float> &sources, int n_src,
                               const std::vector<float> &queries, int nq,
                               int hidden, float eps)
{
    const auto norm = RefUnweightedNorm(sources, n_src, hidden, eps);
    std::vector<float> out(static_cast<size_t>(nq) * hidden, 0.0f);
    const double qscale = 1.0 / std::sqrt(static_cast<double>(hidden));
    for (int q = 0; q < nq; ++q) {
        std::vector<double> logit(static_cast<size_t>(n_src));
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_src; ++j) {
            double dot = 0.0;
            for (int i = 0; i < hidden; ++i)
                dot += static_cast<double>(queries[static_cast<size_t>(q) * hidden + i]) *
                       norm[static_cast<size_t>(j) * hidden + i];
            logit[static_cast<size_t>(j)] = dot * qscale;
            mx = std::max(mx, logit[static_cast<size_t>(j)]);
        }
        double sum = 0.0;
        for (int j = 0; j < n_src; ++j) {
            logit[static_cast<size_t>(j)] = std::exp(logit[static_cast<size_t>(j)] - mx);
            sum += logit[static_cast<size_t>(j)];
        }
        for (int j = 0; j < n_src; ++j) {
            const double w = logit[static_cast<size_t>(j)] / sum;
            for (int i = 0; i < hidden; ++i)
                out[static_cast<size_t>(q) * hidden + i] +=
                    static_cast<float>(w * norm[static_cast<size_t>(j) * hidden + i]);
        }
    }
    return out;
}

}  // namespace

TEST_CASE("inkling loop MessageGate matches the host reference",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kHidden = 64;
    constexpr int kRank = 8;
    constexpr float kEps = 1e-6f;
    const auto h = FPattern(kHidden, 0.5f, 1);
    const auto msg = FPattern(kHidden, 0.5f, 2);
    const auto down = HPattern(kRank * 2 * kHidden, 0.05f, 3);
    const auto forget = FPattern(kHidden * kRank, 0.1f, 4);
    const auto gain = FPattern(kHidden * kRank, 0.1f, 5);
    const auto ref = RefMessageGate(h, msg, down, forget, gain, kHidden, kRank, kEps);

    float *d_h = ToDevice(h);
    float *d_msg = ToDevice(msg);
    __half *d_down = ToDevice(down);
    float *d_forget = ToDevice(forget);
    float *d_gain = ToDevice(gain);
    float *d_out = nullptr;
    float *d_concat = nullptr;
    float *d_rank = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(float) * kHidden) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_concat, sizeof(float) * 2 * kHidden) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_rank, sizeof(float) * kRank) == cudaSuccess);
    REQUIRE(lsxcommon::inkling::RunInklingLoopMessageGate(
                nullptr, d_h, d_msg, d_down, d_forget, d_gain, d_out, kHidden,
                kRank, kEps, d_concat, d_rank) == cudaSuccess);
    Sync();
    std::vector<float> got(kHidden);
    FromDevice(d_out, got);
    for (int i = 0; i < kHidden; ++i) {
        INFO("i=" << i);
        // The gate dot accumulates in FP32 on the device vs float64 on the
        // host; tanh then amplifies the difference by the h/msg magnitude.
        REQUIRE(got[static_cast<size_t>(i)] ==
                Approx(ref[static_cast<size_t>(i)])
                    .margin(std::fabs(ref[static_cast<size_t>(i)]) * 2e-3 + 2e-3));
    }
    // A zero message must leave (1 - tanh(forget(down(norm(h),0))))*h — NOT h
    // itself (the forget gate still acts). Pins that the gate is not bypassed.
    std::vector<float> zero_msg(kHidden, 0.0f);
    const auto ref0 = RefMessageGate(h, zero_msg, down, forget, gain, kHidden,
                                     kRank, kEps);
    float *d_z = ToDevice(zero_msg);
    REQUIRE(lsxcommon::inkling::RunInklingLoopMessageGate(
                nullptr, d_h, d_z, d_down, d_forget, d_gain, d_out, kHidden,
                kRank, kEps, d_concat, d_rank) == cudaSuccess);
    Sync();
    FromDevice(d_out, got);
    int differing = 0;
    for (int i = 0; i < kHidden; ++i) {
        REQUIRE(got[static_cast<size_t>(i)] ==
                Approx(ref0[static_cast<size_t>(i)])
                    .margin(std::fabs(ref0[static_cast<size_t>(i)]) * 2e-3 + 2e-3));
        if (std::fabs(ref0[static_cast<size_t>(i)] - h[static_cast<size_t>(i)]) >
            1e-3f)
            ++differing;
    }
    REQUIRE(differing > 0);
    cudaFree(d_h);
    cudaFree(d_msg);
    cudaFree(d_z);
    cudaFree(d_down);
    cudaFree(d_forget);
    cudaFree(d_gain);
    cudaFree(d_out);
    cudaFree(d_concat);
    cudaFree(d_rank);
}

TEST_CASE("inkling loop message retrieval matches the depth-mix reference",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kHidden = 64;
    constexpr float kEps = 1e-6f;
    // The message sources are the previous pass's block stack + final hidden:
    // up to 5 rows at Mini geometry (4 blocks + live state).
    for (const int n_src : {1, 3, 5}) {
        const auto src = FPattern(n_src * kHidden, 0.5f, n_src);
        const auto qry = FPattern(4 * kHidden, 0.25f, n_src + 5);
        const auto ref = RefDepthMix(src, n_src, qry, 4, kHidden, kEps);
        float *d_src = ToDevice(src);
        float *d_qry = ToDevice(qry);
        float *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(float) * ref.size()) == cudaSuccess);
        REQUIRE(lsxcommon::inkling::RunInklingLoopMessageMix(
                    nullptr, d_src, n_src, d_qry, d_out, kHidden, kEps) ==
                cudaSuccess);
        Sync();
        std::vector<float> got(ref.size());
        FromDevice(d_out, got);
        for (size_t i = 0; i < ref.size(); ++i) {
            INFO("n_src=" << n_src << " i=" << i);
            REQUIRE(got[i] == Approx(ref[i]).margin(1e-5));
        }
        cudaFree(d_src);
        cudaFree(d_qry);
        cudaFree(d_out);
    }
}

TEST_CASE("inkling loop attention at GQA 3 with sinks, window and bias",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    // Mini geometry: 24 query heads / 8 kv heads (GQA 3), head_dim 128. Scaled
    // down 8x for the fixture but the SAME ratio (3 heads per kv head).
    constexpr int kHeads = 6;
    constexpr int kKvHeads = 2;
    constexpr int kDim = 16;
    constexpr int kCap = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kDim));
    const int q_pos = 11;

    for (const int window : {0, 4}) {
        const auto q = HPattern(kHeads * kDim, 0.6f, 1);
        auto k_hist = HPattern((q_pos + 1) * kKvHeads * kDim, 0.5f, 2);
        auto v_hist = HPattern((q_pos + 1) * kKvHeads * kDim, 0.5f, 3);
        const auto sink = FPattern(kHeads, 0.5f, 4);
        const auto bias = FPattern(kHeads * (q_pos + 1), 0.5f, 5);

        std::vector<float> ref(kHeads * kDim, 0.0f);
        for (int h = 0; h < kHeads; ++h) {
            std::vector<double> logit(q_pos + 1);
            double mx = -std::numeric_limits<double>::infinity();
            for (int j = 0; j <= q_pos; ++j) {
                const int kvh = h / (kHeads / kKvHeads);
                double dot = 0.0;
                for (int i = 0; i < kDim; ++i)
                    dot += static_cast<double>(F(q[static_cast<size_t>(h) * kDim + i])) *
                           F(k_hist[(static_cast<size_t>(j) * kKvHeads + kvh) * kDim + i]);
                double l = dot * scale + bias[static_cast<size_t>(h) * (q_pos + 1) + j];
                const bool visible = (window == 0) || (j > q_pos - window);
                logit[static_cast<size_t>(j)] = visible ? l : -1e30;
                mx = std::max(mx, logit[static_cast<size_t>(j)]);
            }
            mx = std::max(mx, static_cast<double>(sink[static_cast<size_t>(h)]));
            double sum = 0.0;
            for (int j = 0; j <= q_pos; ++j) {
                logit[static_cast<size_t>(j)] = std::exp(logit[static_cast<size_t>(j)] - mx);
                sum += logit[static_cast<size_t>(j)];
            }
            sum += std::exp(static_cast<double>(sink[static_cast<size_t>(h)]) - mx);
            for (int j = 0; j <= q_pos; ++j) {
                const double p = logit[static_cast<size_t>(j)] / sum;
                const int kvh = h / (kHeads / kKvHeads);
                for (int i = 0; i < kDim; ++i)
                    ref[static_cast<size_t>(h) * kDim + i] +=
                        static_cast<float>(p * F(v_hist[(static_cast<size_t>(j) * kKvHeads + kvh) * kDim + i]));
            }
        }

        const lsxcommon::KvCacheGeometry geom = lsxcommon::KvCacheGeometry::Make(
            lsxcommon::KvCacheLayout::kPosMajor, kCap, kKvHeads, kDim, 1);
        __half *d_k = nullptr;
        __half *d_v = nullptr;
        REQUIRE(cudaMalloc(&d_k, sizeof(__half) * geom.Elements()) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_v, sizeof(__half) * geom.Elements()) == cudaSuccess);
        REQUIRE(cudaMemcpy2D(d_k, geom.pos_stride * sizeof(__half), k_hist.data(),
                             kKvHeads * kDim * sizeof(__half),
                             kKvHeads * kDim * sizeof(__half), q_pos + 1,
                             cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy2D(d_v, geom.pos_stride * sizeof(__half), v_hist.data(),
                             kKvHeads * kDim * sizeof(__half),
                             kKvHeads * kDim * sizeof(__half), q_pos + 1,
                             cudaMemcpyHostToDevice) == cudaSuccess);
        __half *d_q = ToDevice(q);
        float *d_sink = ToDevice(sink);
        float *d_bias = ToDevice(bias);
        __half *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kHeads * kDim) == cudaSuccess);
        REQUIRE(lsxcommon::inkling::RunInklingAttention(
                    nullptr, d_q, d_k, d_v, geom, kHeads, q_pos, d_sink, d_bias,
                    window, scale, d_out) == cudaSuccess);
        Sync();
        std::vector<__half> got(static_cast<size_t>(kHeads) * kDim);
        FromDevice(d_out, got);
        for (size_t i = 0; i < got.size(); ++i) {
            INFO("window=" << window << " i=" << i);
            REQUIRE(F(got[i]) == Approx(ref[i]).margin(2e-3));
        }
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_out);
        cudaFree(d_sink);
        cudaFree(d_bias);
    }
}

TEST_CASE("inkling loop MXFP8 expert GEMV at Mini width", "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    // Mini's MXFP8 experts: w13 [2304, 2304], w2 [2304, 1152] per expert. The
    // fixture uses the real w2 width (K=1152, N=64 rows sampled) so the per-16
    // duplicated ue8m0 decode is exercised at production stride.
    constexpr int kCount = 2;
    constexpr int kK = 1152;
    constexpr int kN = 8;
    auto e4m3_val = [](uint8_t b) {
        const int sign = (b >> 7) & 1;
        const int e = (b >> 3) & 0xF;
        const int m = b & 0x7;
        const double v = (e == 0) ? (m * std::ldexp(1.0, -9))
                                  : ((1.0 + m / 8.0) * std::ldexp(1.0, e - 7));
        return sign ? -v : v;
    };
    std::vector<uint8_t> w(static_cast<size_t>(kCount) * kN * kK);
    for (size_t i = 0; i < w.size(); ++i) {
        uint8_t b = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
        if (b == 0x7F || b == 0xFF) b ^= 0x01;
        w[i] = b;
    }
    std::vector<uint8_t> sc(static_cast<size_t>(kCount) * kN * (kK / 16));
    for (int e = 0; e < kCount; ++e)
        for (int n = 0; n < kN; ++n)
            for (int b = 0; b < kK / 16; b += 2) {
                const uint8_t ex = static_cast<uint8_t>(120 + ((n + b) % 5));
                sc[static_cast<size_t>(e) * kN * (kK / 16) + n * (kK / 16) + b] = ex;
                sc[static_cast<size_t>(e) * kN * (kK / 16) + n * (kK / 16) + b + 1] = ex;
            }
    auto x = HPattern(static_cast<int64_t>(kCount) * kK, 0.01f, 3);
    std::vector<float> ref(static_cast<size_t>(kCount) * kN);
    for (int e = 0; e < kCount; ++e)
        for (int n = 0; n < kN; ++n) {
            double acc = 0.0;
            for (int k = 0; k < kK; ++k) {
                const uint8_t sv =
                    sc[static_cast<size_t>(e) * kN * (kK / 16) + n * (kK / 16) + (k >> 4)];
                acc += static_cast<double>(
                           e4m3_val(w[static_cast<size_t>(e) * kN * kK + n * kK + k]) *
                           std::ldexp(1.0, int(sv) - 127)) *
                       F(x[static_cast<size_t>(e) * kK + k]);
            }
            ref[static_cast<size_t>(e) * kN + n] = static_cast<float>(acc);
        }
    uint8_t *d_w = ToDevice(w);
    uint8_t *d_s = ToDevice(sc);
    __half *d_x = ToDevice(x);
    __half *d_y = nullptr;
    REQUIRE(cudaMalloc(&d_y, sizeof(__half) * kCount * kN) == cudaSuccess);
    lsxcommon::inkling::InklingFp8ExpertPtrs ptrs;
    for (int e = 0; e < kCount; ++e) {
        ptrs.packed[e] = d_w + static_cast<size_t>(e) * kN * kK;
        ptrs.scales[e] = d_s + static_cast<size_t>(e) * kN * (kK / 16);
    }
    REQUIRE(lsxcommon::inkling::RunInklingFp8ExpertGemm(
                nullptr, d_x, ptrs, kCount, kK, d_y, kK, kN, false) ==
            cudaSuccess);
    Sync();
    std::vector<__half> got(static_cast<size_t>(kCount) * kN);
    FromDevice(d_y, got);
    for (size_t i = 0; i < got.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(F(got[i]) ==
                Approx(ref[i]).margin(std::fabs(ref[i]) * 2e-3 + 2e-3));
    }
    cudaFree(d_w);
    cudaFree(d_s);
    cudaFree(d_x);
    cudaFree(d_y);
}

TEST_CASE("inkling loop router and softcapped SwiGLU at Mini shape",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    // Router: 128 experts, top-8, hidden 2304 — fixture at 1/32 width.
    constexpr int kE = 128;
    constexpr int kK = 8;
    constexpr int kH = 72;
    const float route_scale = 8.0f;
    const auto w = FPattern(kE * kH, 0.02f, 1);
    const auto x = HPattern(kH, 0.5f, 2);
    std::vector<double> score(kE);
    for (int e = 0; e < kE; ++e) {
        double dot = 0.0;
        for (int i = 0; i < kH; ++i)
            dot += static_cast<double>(w[static_cast<size_t>(e) * kH + i]) *
                   F(x[static_cast<size_t>(i)]);
        score[static_cast<size_t>(e)] = 1.0 / (1.0 + std::exp(-dot));
    }
    // A selection-only bias that flips the choice toward the high ids.
    std::vector<float> bias(kE);
    for (int e = 0; e < kE; ++e) bias[static_cast<size_t>(e)] = (e >= 120) ? 10.0f : -10.0f;
    std::vector<bool> taken(kE, false);
    std::vector<int> want;
    for (int k = 0; k < kK; ++k) {
        int best = -1;
        double best_v = -1e300;
        for (int e = 0; e < kE; ++e) {
            if (taken[static_cast<size_t>(e)]) continue;
            const double v = score[static_cast<size_t>(e)] + bias[static_cast<size_t>(e)];
            if (v > best_v) { best_v = v; best = e; }
        }
        taken[static_cast<size_t>(best)] = true;
        want.push_back(best);
    }
    std::sort(want.begin(), want.end());
    double sum = 0.0;
    for (int id : want) sum += score[static_cast<size_t>(id)];

    float *d_w = ToDevice(w);
    __half *d_x = ToDevice(x);
    float *d_bias = ToDevice(bias);
    lsxcommon::inkling::InklingRoutedSlot *d_slots = nullptr;
    int32_t *d_ids = nullptr;
    REQUIRE(cudaMalloc(&d_slots, sizeof(lsxcommon::inkling::InklingRoutedSlot) * kK) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ids, sizeof(int32_t) * kK) == cudaSuccess);
    REQUIRE(lsxcommon::inkling::RunInklingRouter(nullptr, d_w, d_bias, d_x, kE,
                                                 kK, kH, route_scale, d_slots,
                                                 d_ids) == cudaSuccess);
    Sync();
    std::vector<lsxcommon::inkling::InklingRoutedSlot> slots(kK);
    FromDevice(d_slots, slots);
    std::vector<int> got_ids;
    for (auto s : slots) got_ids.push_back(s.eid);
    std::sort(got_ids.begin(), got_ids.end());
    REQUIRE(got_ids == want);
    double total = 0.0;
    for (auto s : slots) {
        const auto it = std::find(want.begin(), want.end(), s.eid);
        const size_t idx = static_cast<size_t>(it - want.begin());
        REQUIRE(s.coeff ==
                Approx(static_cast<float>(score[static_cast<size_t>(want[idx])] /
                                            sum * route_scale)).margin(1e-5));
        total += s.coeff;
    }
    REQUIRE(total == Approx(route_scale).margin(1e-4));
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_bias);
    cudaFree(d_slots);
    cudaFree(d_ids);

    // Softcapped SwiGLU at the expert intermediate width (1152).
    constexpr int64_t kN = 1152;
    const auto g = HPattern(kN, 3.0f, 1);
    const auto u = HPattern(kN, 9.0f, 2);
    std::vector<float> ref(static_cast<size_t>(kN));
    for (int64_t i = 0; i < kN; ++i) {
        const double gv = F(g[static_cast<size_t>(i)]);
        const double uv = F(u[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = static_cast<float>(
            Softcap(gv, 4.0) * (1.0 / (1.0 + std::exp(-gv))) * Softcap(uv, 25.0));
    }
    __half *d_g = ToDevice(g);
    __half *d_u = ToDevice(u);
    __half *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kN) == cudaSuccess);
    REQUIRE(lsxcommon::inkling::RunInklingSoftcapSwiGLU(nullptr, d_g, d_u, d_out,
                                                        kN, 4.0f, 25.0f) ==
            cudaSuccess);
    Sync();
    std::vector<__half> got(static_cast<size_t>(kN));
    FromDevice(d_out, got);
    for (int64_t i = 0; i < kN; ++i) {
        INFO("i=" << i);
        REQUIRE(F(got[static_cast<size_t>(i)]) ==
                Approx(ref[static_cast<size_t>(i)])
                    .margin(std::fabs(ref[static_cast<size_t>(i)]) * 2e-3 + 2e-3));
    }
    cudaFree(d_g);
    cudaFree(d_u);
    cudaFree(d_out);
}

TEST_CASE("inkling loop scratch allocates the loop machinery",
          "[inkling][kernels]")
{
    const InklingLoopTopology top = MakeTop();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    InklingLoopScratch scratch;
    REQUIRE(scratch.Allocate(top, 16, stream));
    REQUIRE(scratch.arena != nullptr);
    REQUIRE(scratch.arena_bytes > 0);
    // Loop buffers exist and are distinct.
    REQUIRE(scratch.pre_loop_hidden != nullptr);
    REQUIRE(scratch.pre_loop_stack != nullptr);
    REQUIRE(scratch.prev_stack != nullptr);
    REQUIRE(scratch.prev_hidden != nullptr);
    REQUIRE(scratch.msg != nullptr);
    REQUIRE(scratch.gate_h != nullptr);
    REQUIRE(scratch.gate_rank != nullptr);
    REQUIRE(scratch.gate_out != nullptr);
    REQUIRE(scratch.pre_loop_hidden != scratch.prev_hidden);
    REQUIRE(scratch.msg != scratch.gate_out);
    // The expert staging slabs are sized for top_k experts of both codecs.
    REQUIRE(scratch.expert_stage != nullptr);
    REQUIRE(scratch.fp8_stage != nullptr);
    REQUIRE(scratch.expert_stage_bytes > 0);
    REQUIRE(scratch.fp8_stage_bytes > 0);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    scratch.Free();
    REQUIRE(scratch.arena == nullptr);
    REQUIRE(scratch.expert_stage == nullptr);
    REQUIRE(scratch.fp8_stage == nullptr);
    cudaStreamDestroy(stream);
}