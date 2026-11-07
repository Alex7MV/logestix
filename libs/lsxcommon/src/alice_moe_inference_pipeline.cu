// AliceAI correctness-first generation pipeline. Each step recomputes the full
// sequence token-by-token (fresh KV/KDA state) so the recurrent + AttnRes state
// is rebuilt deterministically; routed experts stream from the host store.
#include "lsxcommon/alice_moe_inference_pipeline.h"

#include "lsxcommon/alice_moe_inference.h"
#include "lsxcommon/alice_moe_kernels.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#include <cstdio>
#include <chrono>
#include <unordered_map>
#include <cstring>
#include <vector>

namespace lsxcommon::alice {
namespace {

constexpr int kMaxBlocks = 16;
constexpr int kBlock = 256;

__global__ void add_into_kernel(__half *__restrict__ acc,
                                const __half *__restrict__ b, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) acc[i] = __float2half(__half2float(acc[i]) + __half2float(b[i]));
}

__global__ void axpy_kernel(__half *__restrict__ acc,
                            const __half *__restrict__ y, float scale, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        acc[i] = __float2half(__half2float(acc[i]) + scale * __half2float(y[i]));
}

__global__ void mul_sigmoid_kernel(__half *__restrict__ y,
                                   const __half *__restrict__ g, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float s = 1.0f / (1.0f + __expf(-__half2float(g[i])));
        y[i] = __float2half(__half2float(y[i]) * s);
    }
}

__global__ void scale_kernel(__half *__restrict__ y, float s, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(__half2float(y[i]) * s);
}

inline cudaError_t Proj(cudaStream_t s, const lsx_nvfp4::LsxNvfp4Weight &w,
                        const __half *x, __half *y)
{
    return lsx_nvfp4::RunLsxNvfp4Gemm(s, x, w.d_packed, w.d_scales,
                                      w.global_scale, y, 1, w.cols, w.rows,
                                      false);
}

inline cudaError_t ProjRaw(cudaStream_t s, const __half *w, const __half *x,
                           __half *y, int N, int K)
{
    return gemv::RunLxGemmVec(s, w, x, 1, N, K, gemv::LxDtype::kF16,
                              gemv::LxDtype::kF16, 1.0f, y, gemv::LxDtype::kF16,
                              0.0f);
}

// Device-resident expert slot cache (LRU). Each slot holds one expert's
// w1/w3/w2 packed nibbles + E4M3 scales; a miss streams it from the host store
// once, a hit reuses the resident copy (no per-token re-upload).
struct ExpertCache {
    uint8_t *pool = nullptr;
    int slots = 0;
    size_t stride = 0;
    std::vector<std::pair<int32_t, uint64_t>> tbl;  // (key, last-use stamp)
    std::unordered_map<int32_t, int> map;
    uint64_t clk = 0;
    size_t off_w1p = 0, off_w1s = 0, off_w3p = 0, off_w3s = 0, off_w2p = 0,
           off_w2s = 0;
    size_t len_w1p = 0, len_w1s = 0, len_w2p = 0, len_w2s = 0;

    bool Init(cudaStream_t /*s*/, const AliceExpertEntry &ex, int want)
    {
        len_w1p = static_cast<size_t>(ex.w1_bytes);
        len_w1s = static_cast<size_t>(ex.w1_scales_bytes);
        len_w2p = static_cast<size_t>(ex.w2_bytes);
        len_w2s = static_cast<size_t>(ex.w2_scales_bytes);
        if (len_w1p == 0 || len_w2p == 0) return false;
        off_w1p = 0;
        off_w1s = off_w1p + len_w1p;
        off_w3p = off_w1s + len_w1s;
        off_w3s = off_w3p + len_w1p;
        off_w2p = off_w3s + len_w1s;
        off_w2s = off_w2p + len_w2p;
        stride = off_w2s + len_w2s;
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t budget = free_b / 2;  // leave headroom
        int n = static_cast<int>(budget / stride);
        if (n > want) n = want;
        if (n < 1) return false;
        slots = n;
        if (cudaMalloc(&pool, static_cast<size_t>(slots) * stride) !=
            cudaSuccess) {
            pool = nullptr;
            slots = 0;
            return false;
        }
        tbl.assign(static_cast<size_t>(slots), {-1, 0});
        map.reserve(static_cast<size_t>(slots) * 2);
        return true;
    }

    // Returns the slot index for (layer, expert), uploading on a miss.
    int Get(cudaStream_t s, int layer, int expert, const AliceExpertEntry &pe)
    {
        const int32_t key = layer * 100000 + expert;
        auto it = map.find(key);
        if (it != map.end()) {
            tbl[static_cast<size_t>(it->second)].second = ++clk;
            return it->second;
        }
        int slot = -1;
        for (int i = 0; i < slots; ++i)
            if (tbl[static_cast<size_t>(i)].first < 0) { slot = i; break; }
        if (slot < 0) {
            uint64_t best = ~0ull;
            for (int i = 0; i < slots; ++i)
                if (tbl[static_cast<size_t>(i)].second < best) {
                    best = tbl[static_cast<size_t>(i)].second;
                    slot = i;
                }
            map.erase(tbl[static_cast<size_t>(slot)].first);
        }
        uint8_t *base = pool + static_cast<size_t>(slot) * stride;
        cudaMemcpyAsync(base + off_w1p, pe.host_w1, len_w1p,
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(base + off_w1s, pe.host_w1_scales, len_w1s,
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(base + off_w3p, pe.host_w3, len_w1p,
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(base + off_w3s, pe.host_w3_scales, len_w1s,
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(base + off_w2p, pe.host_w2, len_w2p,
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(base + off_w2s, pe.host_w2_scales, len_w2s,
                        cudaMemcpyHostToDevice, s);
        tbl[static_cast<size_t>(slot)].first = key;
        tbl[static_cast<size_t>(slot)].second = ++clk;
        map[key] = slot;
        return slot;
    }

    void Free()
    {
        if (pool) cudaFree(pool);
        pool = nullptr;
        slots = 0;
    }
    ~ExpertCache() { Free(); }
    ExpertCache() = default;
    ExpertCache(const ExpertCache &) = delete;
    ExpertCache &operator=(const ExpertCache &) = delete;
};

// Frees every per-call device allocation on ANY exit path. The decode has
// ~30 CUDA-error returns plus two host abort returns; with a single free
// block at the end each of them leaked the whole workspace, and because CUDA
// errors are sticky one failure cascaded into progressive VRAM exhaustion.
struct WorkspaceGuard {
    std::vector<void *> ptrs;
    void track(void *p) { ptrs.push_back(p); }
    ~WorkspaceGuard()
    {
        for (void *p : ptrs) {
            if (p) cudaFree(p);
        }
    }
    WorkspaceGuard() = default;
    WorkspaceGuard(const WorkspaceGuard &) = delete;
    WorkspaceGuard &operator=(const WorkspaceGuard &) = delete;
};

inline cudaError_t ExpertGemmSlot(cudaStream_t s, const ExpertCache &c,
                                  const uint8_t *base, int which,
                                  const __half *x, __half *y, int rows,
                                  int cols, float gs)
{
    const uint8_t *p = base + (which == 1 ? c.off_w1p
                                          : which == 2 ? c.off_w2p : c.off_w3p);
    const uint8_t *sc = base + (which == 1 ? c.off_w1s
                                           : which == 2 ? c.off_w2s : c.off_w3s);
    return lsx_nvfp4::RunLsxNvfp4Gemm(s, x, p, sc, gs, y, 1, cols, rows, false);
}

inline cudaError_t AddInto(cudaStream_t s, __half *acc, const __half *b, int n)
{
    add_into_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(acc, b, n);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunAliceGeneration(cudaStream_t stream, const AlicePinnedWeights &pw,
                               AliceExpertDb &experts,
                               const std::vector<int32_t> &prompt, int max_new,
                               int eos_id, AliceGenerationResult &result,
                               lsxcommon::sample::ISampler &sampler,
                               const lsxcommon::GenerationStream *gs)
{
    (void)sampler;
    (void)gs;
    const int HID = pw.top.hidden_size;
    const int NH = pw.top.num_attention_heads;
    const int KVH = pw.top.num_key_value_heads;
    const int HD = pw.top.head_dim;
    const int ROT = static_cast<int>(HD * pw.top.partial_rotary_factor);
    const int KDAH = pw.top.linear_num_value_heads;
    const int KD = pw.top.linear_key_head_dim;
    const int VD = pw.top.linear_value_head_dim;
    const int CONVK = pw.top.linear_conv_kernel_dim;
    const int E = pw.top.num_experts;
    const int TOPK = pw.top.num_experts_per_tok;
    const int MI = pw.top.moe_intermediate_size;
    const int SI = pw.top.shared_expert_intermediate_size;
    const int VOCAB = pw.top.vocab_size;
    const int L = pw.top.num_layers;
    const int max_total = static_cast<int>(prompt.size()) + max_new + 2;
    const float eps = static_cast<float>(pw.top.rms_norm_eps);
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(HD));
    const int kdim = KDAH * KD, vdim = KDAH * VD;
    const int attn_len = NH * HD;

    // Device buffers (owned by ws_free: freed on every exit path).
    WorkspaceGuard ws_free;
    __half *x=nullptr,*mixed=nullptr,*partial=nullptr,*src=nullptr,*completed=nullptr;
    __half *qg=nullptr,*attn=nullptr,*attn_out=nullptr,*kcache=nullptr,*vcache=nullptr;
    __half *kq=nullptr,*kk=nullptr,*kv=nullptr,*kfa=nullptr,*kga=nullptr,*kb=nullptr;
    __half *alpha=nullptr,*gate=nullptr,*convq=nullptr,*convk=nullptr,*convv=nullptr,*kout=nullptr;
    __half *mgate=nullptr,*mup=nullptr,*mact=nullptr,*mdown=nullptr;
    __half *sgate=nullptr,*sup=nullptr,*sact=nullptr,*sdown=nullptr,*acc=nullptr;
    __half *cosb=nullptr,*sinb=nullptr,*logits=nullptr;
    float *state=nullptr,*hist=nullptr,*router_logits=nullptr,*wts=nullptr;
    int32_t *idx=nullptr,*posb=nullptr;
    uint8_t *stage_p=nullptr,*stage_s=nullptr;
    cudaError_t e = cudaSuccess;
    // A bare "alloc failed" was unactionable: it named neither the buffer nor the
    // cudaError, so an OOM and a bad API call read the same. Record both.
    const char *failed_buf = nullptr;
    size_t failed_bytes = 0;
    auto fail = [&](const char *why) {
        result.ok = false;
        result.error_msg = why;
        return cudaErrorInvalidValue;
    };
    // `A` yields true when the allocation FAILED, so the `||` chain below reads
    // "any of these failed". The polarity lives in WsAllocFailed(), not in the
    // ternary arms: it was expressed as cudaError_t constants until 397ab361
    // rewrote them as `true`/`false` to silence -Wint-in-bool-context, which
    // inverted the meaning while leaving this chain untouched — the first
    // *successful* allocation then satisfied the `||` and aborted the workspace.
#define A(ptr, n)                                                              \
    ((e = cudaMallocAsync(reinterpret_cast<void **>(&(ptr)),                    \
                          (n) * sizeof(*(ptr)), stream)),                       \
     WsAllocFailed(e) ? (failed_buf = #ptr,                                     \
                         failed_bytes = (n) * sizeof(*(ptr)), true)            \
                      : (ws_free.track(ptr), false))
    const bool ws_ok =
        !(A(x,HID)||A(mixed,HID)||A(partial,HID)||A(attn,attn_len)||A(attn_out,HID)||
          A(src,kMaxBlocks*HID)||A(completed,kMaxBlocks*HID)||A(qg,2*attn_len)||
          A(kq,kdim)||A(kk,kdim)||A(kv,vdim)||A(kfa,VD)||A(kga,VD)||A(kb,KDAH)||
          A(alpha,kdim)||A(gate,vdim)||A(convq,kdim)||A(convk,kdim)||A(convv,vdim)||
          A(kout,vdim)||A(mgate,MI)||A(mup,MI)||A(mact,MI)||A(mdown,HID)||
          A(sgate,SI)||A(sup,SI)||A(sact,SI)||A(sdown,HID)||A(acc,HID)||
          A(cosb,ROT)||A(sinb,ROT)||A(logits,VOCAB)||
          A(kcache,(size_t)(L/4)*max_total*KVH*HD)||A(vcache,(size_t)(L/4)*max_total*KVH*HD)||
          A(state,(size_t)L*KDAH*KD*VD)||A(hist,(size_t)L*3*(2*kdim+vdim))||
          A(router_logits,E)||A(wts,TOPK)||A(idx,TOPK)||A(posb,1)||
          A(stage_p,(size_t)MI*HID/2+16)||A(stage_s,(size_t)MI*HID/16+16));
#undef A
    if (!ws_ok) {
        cudaError_t err = e;
        size_t free_b = 0, total_b = 0;
        (void)cudaMemGetInfo(&free_b, &total_b);
        log::error(absl::StrFormat(
            "[alice] workspace alloc failed: %s needs %u MiB, cudaMallocAsync "
            "-> %s (%d), free %u of %u MiB",
            failed_buf ? failed_buf : "?", static_cast<unsigned>(failed_bytes >> 20),
            cudaGetErrorName(err), static_cast<int>(err),
            static_cast<unsigned>(free_b >> 20), static_cast<unsigned>(total_b >> 20)));
        return fail("alice workspace alloc failed");
    }

    std::string stage = "start";
    ExpertCache ec;
    {
        ensure_alice_expert_resident(experts, 0, 0, stream);
        const AliceExpertEntry *sample = experts.entries.empty()
                                             ? nullptr
                                             : &experts.entries[0];
        if (!sample || !sample->host_w1 || !ec.Init(stream, *sample, E * L)) {
            result.error_msg = "expert device cache init failed";
            return cudaErrorMemoryAllocation;
        }
    }
    std::vector<int32_t> ids = prompt;
    result.first_token = -1;
    const size_t state_bytes = (size_t)L * KDAH * KD * VD * sizeof(float);
    const size_t hist_bytes = (size_t)L * 3 * (2 * kdim + vdim) * sizeof(float);

    auto forward = [&](int32_t tok, int pos, bool want_logits) -> cudaError_t {
        std::vector<__half> emb(HID);
        cudaMemcpy(emb.data(), pw.embed + (size_t)tok * HID, HID * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(x, emb.data(), HID * sizeof(__half),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(completed, x, HID * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
        int nblocks = 1;
        int full_idx = 0;
        bool has_partial = false;

        for (int l = 0; l < L; ++l) {
            const AliceLayerWeights &lw = pw.layers[l];
            if (l > 0 && l % 4 == 0) {
                cudaMemcpyAsync(completed + (size_t)nblocks * HID, partial,
                                HID * sizeof(__half), cudaMemcpyDeviceToDevice,
                                stream);
                ++nblocks;
                has_partial = false;
            }
            if (l == 0) {
                cudaMemcpyAsync(mixed, completed, HID * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
            } else {
                int ns = nblocks;
                cudaMemcpyAsync(src, completed, (size_t)ns * HID * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                if (has_partial) {
                    cudaMemcpyAsync(src + (size_t)ns * HID, partial,
                                    HID * sizeof(__half),
                                    cudaMemcpyDeviceToDevice, stream);
                    ++ns;
                }
                e = RunAliceDepthMix(stream, src, ns, lw.attn_res_norm,
                                     lw.attn_res_proj, mixed, HID, eps);
                if (e) return e;
            }
            e = RunAliceZcRmsNorm(stream, mixed, lw.input_norm, mixed, 1, HID,
                                  eps, true);
            if (e) return e;

            if (lw.full) {
                stage="fa_q";
                e = ProjRaw(stream, lw.fa_q, mixed, qg, 2 * attn_len, HID); if (e) return e;
                __half *kc = kcache + (size_t)full_idx * max_total * KVH * HD;
                __half *vc = vcache + (size_t)full_idx * max_total * KVH * HD;
                __half *krow = kc + (size_t)pos * KVH * HD;
                __half *vrow = vc + (size_t)pos * KVH * HD;
                stage="fa_k"; e = ProjRaw(stream, lw.fa_k, mixed, krow, KVH * HD, HID); if (e) return e;
                e = ProjRaw(stream, lw.fa_v, mixed, vrow, KVH * HD, HID); if (e) return e;
                cudaMemcpyAsync(posb, &pos, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
                e = qwen::RunAttnRopeCosSin(stream, posb, 1, ROT,
                                            pw.top.rope_theta, cosb, sinb);
                if (e) return e;
                stage="fa_core"; e = RunAliceGatedAttentionCore(
                    stream, qg, kc, vc, lw.fa_qnorm, lw.fa_knorm, cosb,
                    sinb, attn, 1, NH, KVH, HD, ROT, attn_scale, pos, nullptr);
                if (e) return e;
                stage="fa_o"; e = ProjRaw(stream, lw.fa_o, attn, attn_out, HID, attn_len); if (e) return e;
                ++full_idx;
            } else {
                stage="kda_proj"; e = ProjRaw(stream, lw.kda_q, mixed, kq, kdim, HID); if (e) return e;
                e = ProjRaw(stream, lw.kda_k, mixed, kk, kdim, HID); if (e) return e;
                e = ProjRaw(stream, lw.kda_v, mixed, kv, vdim, HID); if (e) return e;
                e = ProjRaw(stream, lw.kda_fa, mixed, kfa, VD, HID); if (e) return e;
                e = ProjRaw(stream, lw.kda_fb, kfa, alpha, kdim, VD); if (e) return e;
                e = ProjRaw(stream, lw.kda_ga, mixed, kga, VD, HID); if (e) return e;
                e = ProjRaw(stream, lw.kda_gb, kga, gate, vdim, VD); if (e) return e;
                e = ProjRaw(stream, lw.kda_b, mixed, kb, KDAH, HID); if (e) return e;
                const size_t stride = (size_t)3 * (2 * kdim + vdim);
                float *hq = hist + (size_t)l * stride;
                float *hk = hq + (size_t)kdim * 3;
                float *hv = hk + (size_t)kdim * 3;
                stage="kda_conv"; e = RunAliceKdaConv(stream, kq, lw.kda_qconv, hq, convq, 1, kdim,
                                    CONVK);
                if (e) return e;
                e = RunAliceKdaConv(stream, kk, lw.kda_kconv, hk, convk, 1, kdim,
                                    CONVK);
                if (e) return e;
                e = RunAliceKdaConv(stream, kv, lw.kda_vconv, hv, convv, 1, vdim,
                                    CONVK);
                if (e) return e;
                stage="kda_recur"; e = RunAliceKdaRecurrence(stream, convq, convk, convv, alpha, kb,
                                          lw.kda_alog, lw.kda_dt,
                                          state + (size_t)l * KDAH * KD * VD,
                                          kout, 1, KDAH, KD, VD);
                if (e) return e;
                e = RunAliceZcRmsNorm(stream, kout, lw.kda_onorm, kout, KDAH, VD,
                                      eps, false);
                if (e) return e;
                mul_sigmoid_kernel<<<(vdim + kBlock - 1) / kBlock, kBlock, 0,
                                     stream>>>(kout, gate, vdim);
                e = cudaGetLastError(); if (e) return e;
                e = ProjRaw(stream, lw.kda_o, kout, attn_out, HID, vdim); if (e) return e;
            }

            if (!has_partial) {
                cudaMemcpyAsync(partial, attn_out, HID * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                has_partial = true;
            } else {
                e = AddInto(stream, partial, attn_out, HID); if (e) return e;
            }

            {
                int ns = nblocks;
                cudaMemcpyAsync(src, completed, (size_t)ns * HID * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(src + (size_t)ns * HID, partial,
                                HID * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                ++ns;
                e = RunAliceDepthMix(stream, src, ns, lw.mlp_res_norm,
                                     lw.mlp_res_proj, mixed, HID, eps);
                if (e) return e;
            }
            e = RunAliceZcRmsNorm(stream, mixed, lw.post_norm, mixed, 1, HID, eps,
                                  true);
            if (e) return e;

            // ── MoE ──
            stage="moe_router"; e = RunAliceRouterTopK(stream, mixed, lw.router_gate, lw.router_bias,
                                   HID, E, TOPK, idx, wts, nullptr);
            if (e) return e;
            std::vector<int32_t> hidx(TOPK);
            std::vector<float> hw(TOPK);
            cudaMemcpyAsync(hidx.data(), idx, TOPK * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(hw.data(), wts, TOPK * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaMemsetAsync(acc, 0, HID * sizeof(__half), stream);
            for (int k = 0; k < TOPK; ++k) {
                ensure_alice_expert_resident(experts, l, hidx[k], stream);
                const size_t pe_idx = static_cast<size_t>(l) *
                                          static_cast<size_t>(E) +
                                      static_cast<size_t>(hidx[k]);
                if (pe_idx >= experts.entries.size() ||
                    experts.entries[pe_idx].host_w1 == nullptr) {
                    result.error_msg = "missing routed expert layer=" +
                                       std::to_string(l) + " e=" +
                                       std::to_string(hidx[k]);
                    return cudaErrorInvalidValue;
                }
                const AliceExpertEntry &pe = experts.entries[pe_idx];
                const int slot = ec.Get(stream, l, hidx[k], pe);
                if (slot < 0) {
                    result.error_msg = "expert device cache miss (no slot)";
                    return cudaErrorMemoryAllocation;
                }
                const uint8_t *ebase =
                    ec.pool + static_cast<size_t>(slot) * ec.stride;
                stage="moe_expert_w1"; e = ExpertGemmSlot(stream, ec, ebase, 1, mixed, mgate, MI, HID, pe.w1_global_scale);
                if (e) return e;
                stage="moe_expert_w3"; e = ExpertGemmSlot(stream, ec, ebase, 3, mixed, mup, MI, HID, pe.w3_global_scale);
                if (e) return e;
                e = RunAliceSwiGLU(stream, mgate, mup, mact, MI); if (e) return e;
                stage="moe_expert_w2"; e = ExpertGemmSlot(stream, ec, ebase, 2, mact, mdown, HID, MI, pe.w2_global_scale);
                if (e) return e;
                axpy_kernel<<<(HID + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
                    acc, mdown, hw[(size_t)k], HID);
                e = cudaGetLastError(); if (e) return e;
            }
            stage="moe_shared"; e = Proj(stream, lw.shared_gate_proj, mixed, sgate); if (e) return e;
            e = Proj(stream, lw.shared_up_proj, mixed, sup); if (e) return e;
            e = RunAliceSwiGLU(stream, sgate, sup, sact, SI); if (e) return e;
            e = Proj(stream, lw.shared_down_proj, sact, sdown); if (e) return e;
            e = ProjRaw(stream, lw.shared_gate, mixed, sgate, 1, HID); if (e) return e;
            __half hgate = __float2half(0.0f);
            cudaMemcpyAsync(&hgate, sgate, sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            const float sg = 1.0f / (1.0f + expf(-__half2float(hgate)));
            scale_kernel<<<(HID + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
                sdown, sg, HID);
            e = cudaGetLastError(); if (e) return e;
            e = AddInto(stream, acc, sdown, HID); if (e) return e;
            e = AddInto(stream, partial, acc, HID); if (e) return e;
        }

        if (!want_logits) return cudaSuccess;
        {
            int ns = nblocks;
            cudaMemcpyAsync(src, completed, (size_t)ns * HID * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(src + (size_t)ns * HID, partial,
                            HID * sizeof(__half), cudaMemcpyDeviceToDevice,
                            stream);
            ++ns;
            stage="final_mix"; e = RunAliceDepthMix(stream, src, ns, pw.attnres_final_norm,
                                 pw.attnres_final_proj, mixed, HID, eps);
            if (e) return e;
        }
        e = RunAliceZcRmsNorm(stream, mixed, pw.final_norm, mixed, 1, HID, eps,
                              true);
        if (e) return e;
        stage="head"; e = gemv::RunLxGemmVec(stream, pw.head, mixed, 1, VOCAB, HID,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               logits, gemv::LxDtype::kF16, 0.0f);
        return e;
    };

    const auto t_start = std::chrono::steady_clock::now();
    // Recurrent state (KDA + conv history) is reset once and then carried
    // across the whole prefill + decode (one forward per token, not a full
    // recompute per step): prefill the prompt, then decode one token per step.
    cudaMemsetAsync(state, 0, state_bytes, stream);
    cudaMemsetAsync(hist, 0, hist_bytes, stream);

    int pos = 0;
    for (; pos < static_cast<int>(ids.size()); ++pos) {
        if (gs != nullptr && gs->Aborted()) break;
        const bool last = (pos == static_cast<int>(ids.size()) - 1);
        e = forward(ids[(size_t)pos], pos, last);
        if (e) {
            if (result.error_msg.empty())
                result.error_msg = std::string("alice forward failed [") +
                                   stage + "]: " + cudaGetErrorString(e);
            return e;
        }
    }

    result.ttft_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_start).count();
    for (int step = 0; step < max_new; ++step) {
        if (gs != nullptr && gs->Aborted()) break;
        std::vector<__half> hlog(VOCAB);
        cudaMemcpyAsync(hlog.data(), logits, VOCAB * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int best = 0;
        float bv = -1e30f;
        for (int i = 0; i < VOCAB; ++i) {
            const float v = __half2float(hlog[(size_t)i]);
            if (v > bv) { bv = v; best = i; }
        }
        if (result.first_token < 0) {
            result.first_token = best;
        }
        // Publish the LAST step's fp32 logits row, matching the fleet-wide
        // contract (OPEN_TASKS §E2: every other family publishes the last row,
        // e.g. qwen_inference.cu / granite_dense_inference.cu / nemotron_moe_model.cpp).
        // This used to freeze the FIRST row (`first_token < 0`); overwriting on
        // every step keeps only the final one. Alice previously held the first
        // row, which made any cross-model logits/hash comparison
        // apples-to-oranges.
        result.final_logits.resize(VOCAB);
        for (int i = 0; i < VOCAB; ++i)
            result.final_logits[(size_t)i] = __half2float(hlog[(size_t)i]);
        result.tokens.push_back(best);
        // One chunk per decoded token, then honour a cooperative cancel
        // before the next forward (lsxfabric --serve / TaskMosaic streaming).
        if (gs != nullptr) gs->Emit(static_cast<std::uint32_t>(best),
                                    static_cast<std::uint32_t>(step),
                                    static_cast<float>(bv));
        if (eos_id >= 0 && best == eos_id) { result.eos_terminated = true; break; }
        if (step + 1 >= max_new) break;
        e = forward(best, pos, true);
        ++pos;
        if (e) {
            if (result.error_msg.empty())
                result.error_msg = std::string("alice forward failed [") +
                                   stage + "]: " + cudaGetErrorString(e);
            return e;
        }
    }
    result.total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_start).count();
    result.ok = true;

    // ws_free / ec release the workspace here and on every error return.
    return cudaSuccess;
}

} // namespace lsxcommon::alice
