// GigaChat3.5 40-layer hybrid decode pipeline (Plan 2b). Composes the MLA and
// GatedDeltaNet launchers, the gated norms, the MoE/MLP and lm_head. Session
// state (caches, tables, scratch) lives in GcPipeline::Impl.
#include "lsxcommon/gigachat_moe_inference_pipeline.h"

#include "gigachat_moe_constants.h"
#include "gigachat_moe_device_math.h"
#include "gigachat_moe_kernels.h"

#include "lsxcommon/lsx_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lsxcommon::gigachat {
namespace {

constexpr int64_t kAlign = 256;
int64_t AlignUp(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

__global__ void embed_gather_kernel(const __nv_bfloat16 *__restrict__ e,
                                    int token, int hidden,
                                    float *__restrict__ h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        h[i] = __bfloat162float(e[static_cast<size_t>(token) * hidden + i]);
}

// Variant for the host-resident embedding table: the selected row is staged
// H2D into a scratch row, then converted here.
__global__ void embed_row_f32_kernel(const __nv_bfloat16 *__restrict__ row,
                                     int hidden, float *__restrict__ h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) h[i] = __bfloat162float(row[i]);
}

__global__ void f32_to_f16_kernel(const float *__restrict__ x,
                                  __half *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void f16_to_f32_kernel(const __half *__restrict__ x,
                                  float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

__global__ void f32_to_bf16_kernel(const float *__restrict__ x,
                                   __nv_bfloat16 *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

__global__ void add_f32_kernel(const float *__restrict__ a,
                               const float *__restrict__ b,
                               float *__restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void swiglu_f16_kernel(const __half *__restrict__ gate,
                                  const __half *__restrict__ up, float limit,
                                  int n, __half *__restrict__ mid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mid[i] = __float2half(dev::SwigluLimitElement(
            __half2float(gate[i]), __half2float(up[i]), limit));
}

__global__ void moe_weighted_sum_kernel(const __half *__restrict__ y,
                                        const float *__restrict__ w, int k,
                                        int hidden, float *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float acc = 0.0f;
    for (int e = 0; e < k; ++e)
        acc += w[e] * __half2float(y[static_cast<size_t>(e) * hidden + i]);
    out[i] = acc;
}

inline cudaError_t PLaunchGemm(const lsx_nvfp4::LsxNvfp4Weight &w,
                               const __half *x, __half *y, cudaStream_t s)
{
    return lsx_nvfp4::RunLsxNvfp4Gemm(s, x, w.d_packed, w.d_scales,
                                      w.global_scale, y, 1, w.cols, w.rows);
}

inline cudaError_t PToF16(const float *x, __half *y, int n, cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

inline cudaError_t PToF32(const __half *x, float *y, int n, cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    f16_to_f32_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

} // namespace

bool GcBuildYarnTables(int capacity, int rope_dim, std::vector<float> &cos_t,
                       std::vector<float> &sin_t)
{
    if (capacity <= 0 || rope_dim <= 0 || rope_dim % 2 != 0) return false;
    const int half = rope_dim / 2;
    const double base = frozen::kRopeTheta;
    const double factor = frozen::kYarnFactor;
    const double orig = frozen::kYarnOrig;
    const double beta_fast = frozen::kYarnBetaFast;
    const double beta_slow = frozen::kYarnBetaSlow;
    std::vector<double> freqs(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j)
        freqs[static_cast<size_t>(j)] =
            1.0 / std::pow(base, static_cast<double>(2 * j) / rope_dim);
    if (orig > 0.0)
    {
        const double ln_base = std::log(base);
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        auto corrected = [&](double rotations) {
            return rope_dim * std::log(orig / (rotations * kTwoPi)) /
                   (2.0 * ln_base);
        };
        const int low = std::max(static_cast<int>(std::floor(corrected(beta_fast))), 0);
        const int high = std::min(static_cast<int>(std::ceil(corrected(beta_slow))),
                                  rope_dim - 1);
        for (int i = 0; i < half; ++i)
        {
            const double span = std::max(high - low, 1);
            const double ramp = std::clamp((i - low) / span, 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            freqs[static_cast<size_t>(i)] =
                freqs[static_cast<size_t>(i)] / factor * (1.0 - smooth) +
                freqs[static_cast<size_t>(i)] * smooth;
        }
    }
    cos_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    sin_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    for (int p = 0; p < capacity; ++p)
        for (int j = 0; j < half; ++j)
        {
            const double a = static_cast<double>(p) * freqs[static_cast<size_t>(j)];
            cos_t[static_cast<size_t>(p) * half + j] = static_cast<float>(std::cos(a));
            sin_t[static_cast<size_t>(p) * half + j] = static_cast<float>(std::sin(a));
        }
    return true;
}

struct GcPipeline::Impl {
    GcPipelineDims d;
    int capacity = 0;
    int dense_inter = 0;
    float attn_scale = 1.0f;
    GcMlaScratch mla;
    GcGdnScratch gdn;
    std::vector<float> rc, rs;
    // Device copies of rc/rs. The YaRN tables are built on the host but read
    // by rope_qk_kernel, so a raw rc.data() would hand a kernel a host
    // pointer; carves out of d_base like every other pipeline buffer.
    float *d_rc = nullptr, *d_rs = nullptr;
    std::vector<float *> kcache, vcache, conv, rec;

    void *d_base = nullptr;
    int64_t bytes = 0;
    float *h = nullptr, *res = nullptr, *buf = nullptr, *attn = nullptr,
          *mlp = nullptr;
    __half *n_h = nullptr;
    int *d_idx = nullptr;
    float *d_w = nullptr;
    std::vector<int> h_idx;
    std::vector<float> h_w;
    std::vector<GcExpertView> views;
    std::vector<const uint8_t *> gate_p, gate_s, up_p, up_s, down_p, down_s;
    std::vector<float> gate_gs, up_gs, down_gs;
    __half *ff0_h = nullptr, *ff1_h = nullptr, *ffmid_h = nullptr;
    __half *sgate_h = nullptr, *sup_h = nullptr, *smid_h = nullptr,
           *sdown_h = nullptr;
    float *sdown_f = nullptr;
    __nv_bfloat16 *xb = nullptr;
    __nv_bfloat16 *d_embed_row = nullptr;

    void FreeAll();
};

void GcPipeline::Impl::FreeAll()
{
    for (float *p : kcache) cudaFree(p);
    for (float *p : vcache) cudaFree(p);
    for (float *p : conv) cudaFree(p);
    for (float *p : rec) cudaFree(p);
    kcache.clear();
    vcache.clear();
    conv.clear();
    rec.clear();
    mla.Free();
    gdn.Free();
    if (d_base)
    {
        cudaFree(d_base);
        d_base = nullptr;
    }
    bytes = 0;
    h = res = buf = attn = mlp = nullptr;
    n_h = nullptr;
    d_idx = nullptr;
    d_w = nullptr;
    ff0_h = ff1_h = ffmid_h = nullptr;
    sgate_h = sup_h = smid_h = sdown_h = nullptr;
    sdown_f = nullptr;
    xb = nullptr;
    d_embed_row = nullptr;
}

GcPipeline::GcPipeline() : impl_(new Impl()) {}
GcPipeline::~GcPipeline()
{
    if (impl_)
    {
        impl_->FreeAll();
        delete impl_;
    }
}
bool GcPipeline::ok() const { return impl_ && impl_->d_base != nullptr; }
void GcPipeline::Free()
{
    if (impl_) impl_->FreeAll();
}

bool GcPipeline::Init(const GcWeights &w, const GcPipelineDims &dims,
                      int capacity, std::string &error)
{
    Impl &p = *impl_;
    p.FreeAll();
    if (capacity <= 0 || w.layers.empty())
    {
        error = "gigachat pipeline: invalid capacity or empty weights";
        return false;
    }
    p.d = dims;
    p.capacity = capacity;
    const int hidden = dims.mla.hidden;
    if (!GcBuildYarnTables(capacity, dims.mla.rope, p.rc, p.rs))
    {
        error = "gigachat pipeline: rope table build failed";
        return false;
    }
    const float mscale = 0.1f * std::log(static_cast<float>(frozen::kYarnFactor)) + 1.0f;
    p.attn_scale =
        mscale * mscale / std::sqrt(static_cast<float>(dims.mla.qk_nope + dims.mla.rope));

    for (const auto &lw : w.layers)
        if (!lw.moe) p.dense_inter = lw.mlp_gate.rows;
    // ff0 holds the gate/up ([top_k, moe_intermediate]), dense ([dense_inter])
    // AND the expert down output ([top_k, hidden]) -- the last is the widest
    // on the real model (top_k*hidden = 8*7168 = 57344 vs
    // top_k*moe_intermediate = 16384 and dense_inter = 18432).
    const int64_t ff0 = std::max<int64_t>(
        std::max<int64_t>(static_cast<int64_t>(dims.top_k) * dims.moe_intermediate,
                          static_cast<int64_t>(dims.top_k) * hidden),
        p.dense_inter);

    if (!p.mla.Alloc(dims.mla, capacity) || !p.gdn.Alloc(dims.gdn))
    {
        error = "gigachat pipeline: attention scratch allocation failed";
        return false;
    }

    const int num_layers = static_cast<int>(w.layers.size());
    const int64_t kv_bytes = static_cast<int64_t>(capacity) * dims.mla.heads *
                             (dims.mla.qk_nope + dims.mla.rope) * 4;
    const int64_t vv_bytes = static_cast<int64_t>(capacity) * dims.mla.heads *
                             dims.mla.v_head * 4;
    const int64_t conv_dim =
        2 * static_cast<int64_t>(dims.gdn.k_heads) * dims.gdn.hk +
        static_cast<int64_t>(dims.gdn.v_heads) * dims.gdn.hv;
    const int64_t conv_bytes =
        conv_dim * std::max(dims.gdn.conv_k - 1, 1) * 4;
    const int64_t rec_bytes = static_cast<int64_t>(dims.gdn.v_heads) *
                              dims.gdn.hk * dims.gdn.hv * 4;
    p.kcache.assign(num_layers, nullptr);
    p.vcache.assign(num_layers, nullptr);
    p.conv.assign(num_layers, nullptr);
    p.rec.assign(num_layers, nullptr);
    for (int l = 0; l < num_layers; ++l)
    {
        const GcLayerWeights &lw = w.layers[static_cast<size_t>(l)];
        if (lw.full_attention)
        {
            if (cudaMalloc(&p.kcache[l], kv_bytes) != cudaSuccess ||
                cudaMemset(p.kcache[l], 0, kv_bytes) != cudaSuccess ||
                cudaMalloc(&p.vcache[l], vv_bytes) != cudaSuccess ||
                cudaMemset(p.vcache[l], 0, vv_bytes) != cudaSuccess)
            {
                error = "gigachat pipeline: KV cache allocation failed";
                return false;
            }
        }
        else
        {
            if (cudaMalloc(&p.conv[l], conv_bytes) != cudaSuccess ||
                cudaMemset(p.conv[l], 0, conv_bytes) != cudaSuccess ||
                cudaMalloc(&p.rec[l], rec_bytes) != cudaSuccess ||
                cudaMemset(p.rec[l], 0, rec_bytes) != cudaSuccess)
            {
                error = "gigachat pipeline: GDN state allocation failed";
                return false;
            }
        }
    }

    int64_t total = 0;
    auto need = [&](int64_t b) { total = AlignUp(total) + b; };
    need(static_cast<int64_t>(hidden) * 4);  // h
    need(static_cast<int64_t>(hidden) * 4);  // res
    need(static_cast<int64_t>(hidden) * 4);  // buf
    need(static_cast<int64_t>(hidden) * 4);  // attn
    need(static_cast<int64_t>(hidden) * 4);  // mlp
    need(static_cast<int64_t>(hidden) * 2);  // n_h
    need(static_cast<int64_t>(dims.top_k) * 4);
    need(static_cast<int64_t>(dims.top_k) * 4);
    need(ff0 * 2);  // ff0_h
    need(ff0 * 2);  // ff1_h
    need(ff0 * 2);  // ffmid_h
    need(static_cast<int64_t>(dims.moe_intermediate) * 2);  // sgate
    need(static_cast<int64_t>(dims.moe_intermediate) * 2);  // sup
    need(static_cast<int64_t>(dims.moe_intermediate) * 2);  // smid
    need(static_cast<int64_t>(hidden) * 2);                 // sdown
    need(static_cast<int64_t>(hidden) * 4);                 // sdown_f
    need(static_cast<int64_t>(hidden) * 2);                 // xb
    need(static_cast<int64_t>(hidden) * 2);                 // d_embed_row
    const int64_t rope_bytes =
        static_cast<int64_t>(capacity) * (dims.mla.rope / 2) * 4;
    need(rope_bytes);  // d_rc
    need(rope_bytes);  // d_rs
    void *base = nullptr;
    if (cudaMalloc(&base, static_cast<size_t>(total)) != cudaSuccess)
    {
        error = "gigachat pipeline: scratch allocation failed";
        return false;
    }
    p.d_base = base;
    p.bytes = total;
    int64_t off = 0;
    auto carve = [&](int64_t b) {
        off = AlignUp(off);
        void *q = static_cast<uint8_t *>(base) + off;
        off += b;
        return q;
    };
    p.h = static_cast<float *>(carve(static_cast<int64_t>(hidden) * 4));
    p.res = static_cast<float *>(carve(static_cast<int64_t>(hidden) * 4));
    p.buf = static_cast<float *>(carve(static_cast<int64_t>(hidden) * 4));
    p.attn = static_cast<float *>(carve(static_cast<int64_t>(hidden) * 4));
    p.mlp = static_cast<float *>(carve(static_cast<int64_t>(hidden) * 4));
    p.n_h = static_cast<__half *>(carve(static_cast<int64_t>(hidden) * 2));
    p.d_idx = static_cast<int *>(carve(static_cast<int64_t>(dims.top_k) * 4));
    p.d_w = static_cast<float *>(carve(static_cast<int64_t>(dims.top_k) * 4));
    p.ff0_h = static_cast<__half *>(carve(ff0 * 2));
    p.ff1_h = static_cast<__half *>(carve(ff0 * 2));
    p.ffmid_h = static_cast<__half *>(carve(ff0 * 2));
    p.sgate_h = static_cast<__half *>(carve(static_cast<int64_t>(dims.moe_intermediate) * 2));
    p.sup_h = static_cast<__half *>(carve(static_cast<int64_t>(dims.moe_intermediate) * 2));
    p.smid_h = static_cast<__half *>(carve(static_cast<int64_t>(dims.moe_intermediate) * 2));
    p.sdown_h = static_cast<__half *>(carve(static_cast<int64_t>(hidden) * 2));
    p.sdown_f = static_cast<float *>(carve(static_cast<int64_t>(hidden) * 4));
    p.xb = static_cast<__nv_bfloat16 *>(carve(static_cast<int64_t>(hidden) * 2));
    p.d_embed_row =
        static_cast<__nv_bfloat16 *>(carve(static_cast<int64_t>(hidden) * 2));
    p.d_rc = static_cast<float *>(carve(rope_bytes));
    p.d_rs = static_cast<float *>(carve(rope_bytes));
    if (cudaMemcpy(p.d_rc, p.rc.data(), static_cast<size_t>(rope_bytes),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(p.d_rs, p.rs.data(), static_cast<size_t>(rope_bytes),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        error = "gigachat pipeline: rope table upload failed";
        return false;
    }

    p.h_idx.assign(static_cast<size_t>(dims.top_k), 0);
    p.h_w.assign(static_cast<size_t>(dims.top_k), 0.0f);
    p.views.assign(static_cast<size_t>(dims.top_k), GcExpertView{});
    const size_t k = static_cast<size_t>(dims.top_k);
    p.gate_p.assign(k, nullptr);
    p.gate_s.assign(k, nullptr);
    p.up_p.assign(k, nullptr);
    p.up_s.assign(k, nullptr);
    p.down_p.assign(k, nullptr);
    p.down_s.assign(k, nullptr);
    p.gate_gs.assign(k, 1.0f);
    p.up_gs.assign(k, 1.0f);
    p.down_gs.assign(k, 1.0f);
    error.clear();
    return true;
}

cudaError_t GcPipeline::Reset(cudaStream_t stream)
{
    Impl &p = *impl_;
    const int dims_qk = p.d.mla.qk_nope + p.d.mla.rope;
    for (size_t l = 0; l < p.kcache.size(); ++l)
    {
        if (p.kcache[l])
        {
            const size_t kbytes = static_cast<size_t>(p.capacity) *
                                  p.d.mla.heads * dims_qk * 4;
            const size_t vbytes = static_cast<size_t>(p.capacity) *
                                  p.d.mla.heads * p.d.mla.v_head * 4;
            cudaError_t e = cudaMemsetAsync(p.kcache[l], 0, kbytes, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemsetAsync(p.vcache[l], 0, vbytes, stream);
            if (e != cudaSuccess) return e;
        }
        if (p.conv[l])
        {
            const size_t cdim = static_cast<size_t>(
                2 * p.d.gdn.k_heads * p.d.gdn.hk +
                p.d.gdn.v_heads * p.d.gdn.hv);
            const size_t cbytes =
                cdim * static_cast<size_t>(std::max(p.d.gdn.conv_k - 1, 1)) * 4;
            const size_t rbytes = static_cast<size_t>(p.d.gdn.v_heads) *
                                  p.d.gdn.hk * p.d.gdn.hv * 4;
            cudaError_t e = cudaMemsetAsync(p.conv[l], 0, cbytes, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemsetAsync(p.rec[l], 0, rbytes, stream);
            if (e != cudaSuccess) return e;
        }
    }
    return cudaSuccess;
}

cudaError_t GcPipeline::Step(const GcWeights &w, GcExpertCache &ec,
                             int token_id, int pos, float *logits,
                             cudaStream_t stream)
{
    Impl &p = *impl_;
    const int hidden = p.d.mla.hidden;
    const int num_layers = static_cast<int>(w.layers.size());
    if (!p.d_base || pos < 0 || pos >= p.capacity ||
        token_id < 0 || token_id >= w.vocab)
        return cudaErrorInvalidValue;

#define GC_TRY(expr)                      \
    do {                                  \
        const cudaError_t _e = (expr);    \
        if (_e != cudaSuccess) return _e; \
    } while (false)

    if (w.h_embed)
    {
        GC_TRY(cudaMemcpyAsync(
            p.d_embed_row,
            w.h_embed + static_cast<size_t>(token_id) * hidden * 2,
            static_cast<size_t>(hidden) * 2, cudaMemcpyHostToDevice, stream));
        embed_row_f32_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
            p.d_embed_row, hidden, p.h);
    }
    else
    {
        embed_gather_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(w.d_embed), token_id, hidden,
            p.h);
    }
    GC_TRY(cudaGetLastError());

    for (int l = 0; l < num_layers; ++l)
    {
        const GcLayerWeights &lw = w.layers[static_cast<size_t>(l)];
        GC_TRY(cudaMemcpyAsync(p.res, p.h, static_cast<size_t>(hidden) * 4,
                               cudaMemcpyDeviceToDevice, stream));
        GC_TRY(RunZeroCenteredGatedNorm(p.h, 1, hidden, lw.input_norm.rank,
                                        lw.input_norm.d_weight,
                                        lw.input_norm.d_gate_up,
                                        lw.input_norm.d_gate_down, p.d.rms_eps,
                                        p.buf, stream));
        GC_TRY(PToF16(p.buf, p.n_h, hidden, stream));
        if (lw.full_attention)
            GC_TRY(RunGigaChatMlaDecode(p.buf, lw, p.d.mla, p.d.rms_eps,
                                        p.attn_scale, p.d_rc, p.d_rs,
                                        pos, p.capacity, p.kcache[l],
                                        p.vcache[l], p.mla, p.attn, stream));
        else
            GC_TRY(RunGigaChatGdnDecode(p.buf, lw, p.d.gdn, p.d.rms_eps,
                                        frozen::kLinearSigmoidGateScale,
                                        p.conv[l], p.rec[l], p.gdn, p.attn,
                                        stream));
        GC_TRY(RunZeroCenteredGatedNorm(
            p.attn, 1, hidden, lw.post_self_attn_norm.rank,
            lw.post_self_attn_norm.d_weight,
            lw.post_self_attn_norm.d_gate_up,
            lw.post_self_attn_norm.d_gate_down, p.d.rms_eps, p.buf, stream));
        add_f32_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(p.res, p.buf,
                                                                 p.h, hidden);
        GC_TRY(cudaGetLastError());

        GC_TRY(cudaMemcpyAsync(p.res, p.h, static_cast<size_t>(hidden) * 4,
                               cudaMemcpyDeviceToDevice, stream));
        GC_TRY(RunZeroCenteredGatedNorm(p.h, 1, hidden, lw.post_attn_norm.rank,
                                        lw.post_attn_norm.d_weight,
                                        lw.post_attn_norm.d_gate_up,
                                        lw.post_attn_norm.d_gate_down,
                                        p.d.rms_eps, p.buf, stream));
        GC_TRY(PToF16(p.buf, p.n_h, hidden, stream));

        if (lw.moe)
        {
            const int E = p.d.num_experts, K = p.d.top_k,
                      I = p.d.moe_intermediate;
            GC_TRY(RunGigaChatRouterTopK(
                p.buf, lw.d_router,
                reinterpret_cast<const float *>(lw.d_router_bias), E, hidden, K,
                p.d.routed_scale, p.d_idx, p.d_w, stream));
            GC_TRY(cudaMemcpyAsync(p.h_idx.data(), p.d_idx,
                                   static_cast<size_t>(K) * 4,
                                   cudaMemcpyDeviceToHost, stream));
            GC_TRY(cudaStreamSynchronize(stream));
            std::string err;
            if (ResolveGcExperts(ec, l, p.h_idx.data(), K, stream, p.views.data(),
                                 err) != cudaSuccess)
                return cudaErrorInvalidValue;
            for (int e = 0; e < K; ++e)
            {
                p.gate_p[e] = p.views[e].gate_packed;
                p.gate_s[e] = p.views[e].gate_scales;
                p.gate_gs[e] = p.views[e].gate_global_scale;
                p.up_p[e] = p.views[e].up_packed;
                p.up_s[e] = p.views[e].up_scales;
                p.up_gs[e] = p.views[e].up_global_scale;
                p.down_p[e] = p.views[e].down_packed;
                p.down_s[e] = p.views[e].down_scales;
                p.down_gs[e] = p.views[e].down_global_scale;
            }
            GC_TRY(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, p.n_h, p.gate_p.data(), p.gate_s.data(),
                p.gate_gs.data(), K, 0, p.ff0_h, hidden, I));
            GC_TRY(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, p.n_h, p.up_p.data(), p.up_s.data(), p.up_gs.data(), K,
                0, p.ff1_h, hidden, I));
            swiglu_f16_kernel<<<(K * I + 255) / 256, 256, 0, stream>>>(
                p.ff0_h, p.ff1_h, p.d.swiglu_limit, K * I, p.ffmid_h);
            GC_TRY(cudaGetLastError());
            GC_TRY(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, p.ffmid_h, p.down_p.data(), p.down_s.data(),
                p.down_gs.data(), K, I, p.ff0_h, I, hidden));
            moe_weighted_sum_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
                p.ff0_h, p.d_w, K, hidden, p.mlp);
            GC_TRY(cudaGetLastError());

            GC_TRY(PLaunchGemm(lw.shared_gate, p.n_h, p.sgate_h, stream));
            GC_TRY(PLaunchGemm(lw.shared_up, p.n_h, p.sup_h, stream));
            swiglu_f16_kernel<<<(I + 255) / 256, 256, 0, stream>>>(
                p.sgate_h, p.sup_h, p.d.swiglu_limit, I, p.smid_h);
            GC_TRY(cudaGetLastError());
            GC_TRY(PLaunchGemm(lw.shared_down, p.smid_h, p.sdown_h, stream));
            GC_TRY(PToF32(p.sdown_h, p.sdown_f, hidden, stream));
            add_f32_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
                p.mlp, p.sdown_f, p.mlp, hidden);
            GC_TRY(cudaGetLastError());
        }
        else
        {
            GC_TRY(PLaunchGemm(lw.mlp_gate, p.n_h, p.ff0_h, stream));
            GC_TRY(PLaunchGemm(lw.mlp_up, p.n_h, p.ff1_h, stream));
            const int di = p.dense_inter;
            swiglu_f16_kernel<<<(di + 255) / 256, 256, 0, stream>>>(
                p.ff0_h, p.ff1_h, p.d.swiglu_limit, di, p.ffmid_h);
            GC_TRY(cudaGetLastError());
            GC_TRY(PLaunchGemm(lw.mlp_down, p.ffmid_h, p.ff0_h, stream));
            GC_TRY(PToF32(p.ff0_h, p.mlp, hidden, stream));
        }

        GC_TRY(RunZeroCenteredGatedNorm(p.mlp, 1, hidden, lw.post_ffw_norm.rank,
                                        lw.post_ffw_norm.d_weight,
                                        lw.post_ffw_norm.d_gate_up,
                                        lw.post_ffw_norm.d_gate_down,
                                        p.d.rms_eps, p.buf, stream));
        add_f32_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(p.res, p.buf,
                                                                 p.h, hidden);
        GC_TRY(cudaGetLastError());
    }

    GC_TRY(RunZeroCenteredGatedNorm(
        p.h, 1, hidden, w.final_norm.rank, w.final_norm.d_weight,
        w.final_norm.d_gate_up,
        w.final_norm.d_gate_down, p.d.rms_eps, p.buf, stream));
    f32_to_bf16_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(p.buf, p.xb,
                                                                hidden);
    GC_TRY(cudaGetLastError());
    GC_TRY(gemv::RunLxGemmVec(stream, w.d_lm_head, p.xb, 1, w.vocab, hidden,
                              gemv::LxDtype::kBF16, gemv::LxDtype::kBF16, 1.0f,
                              logits, gemv::LxDtype::kF32, 0.0f));
#undef GC_TRY
    return cudaSuccess;
}

} // namespace lsxcommon::gigachat
