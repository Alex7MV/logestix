// AliceAI correctness-first generation pipeline. Each step recomputes the full
// sequence token-by-token (fresh KV/KDA state) so the recurrent + AttnRes state
// is rebuilt deterministically; routed experts stream from the host store.
#include "lsxcommon/alice_moe_inference_pipeline.h"

#include "lsxcommon/alice_moe_inference.h"
#include "lsxcommon/alice_moe_kernels.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/qwen_inference.h"

#include <cstdio>
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

inline cudaError_t ExpertGemm(cudaStream_t s, const PackedExpert &pe, int which,
                              const __half *x, __half *y, int rows, int cols,
                              uint8_t *stage_p, uint8_t *stage_s)
{
    const std::vector<uint8_t> *packed;
    const std::vector<uint8_t> *scales;
    float gs;
    if (which == 1) { packed = &pe.w1_packed; scales = &pe.w1_scales; gs = pe.w1_global_scale; }
    else if (which == 2) { packed = &pe.w2_packed; scales = &pe.w2_scales; gs = pe.w2_global_scale; }
    else { packed = &pe.w3_packed; scales = &pe.w3_scales; gs = pe.w3_global_scale; }
    if (packed->empty() || scales->empty()) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemcpyAsync(stage_p, packed->data(), packed->size(),
                                    cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(stage_s, scales->data(), scales->size(),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    return lsx_nvfp4::RunLsxNvfp4Gemm(s, x, stage_p, stage_s, gs, y, 1, cols,
                                      rows, false);
}

inline cudaError_t AddInto(cudaStream_t s, __half *acc, const __half *b, int n)
{
    add_into_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(acc, b, n);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunAliceGeneration(cudaStream_t stream, const AlicePinnedWeights &pw,
                               AliceExpertHost &experts,
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

    // Device buffers.
#define A(ptr, n) cudaMallocAsync(reinterpret_cast<void **>(&(ptr)), (n) * sizeof(*(ptr)), stream)
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
    auto fail = [&](const char *why) {
        result.ok = false;
        result.error_msg = why;
        return cudaErrorInvalidValue;
    };
    if (A(x,HID)||A(mixed,HID)||A(partial,HID)||A(attn,attn_len)||A(attn_out,HID)||
        A(src,kMaxBlocks*HID)||A(completed,kMaxBlocks*HID)||A(qg,2*attn_len)||
        A(kq,kdim)||A(kk,kdim)||A(kv,vdim)||A(kfa,VD)||A(kga,VD)||A(kb,KDAH)||
        A(alpha,kdim)||A(gate,vdim)||A(convq,kdim)||A(convk,kdim)||A(convv,vdim)||
        A(kout,vdim)||A(mgate,MI)||A(mup,MI)||A(mact,MI)||A(mdown,HID)||
        A(sgate,SI)||A(sup,SI)||A(sact,SI)||A(sdown,HID)||A(acc,HID)||
        A(cosb,ROT)||A(sinb,ROT)||A(logits,VOCAB)||
        A(kcache,(size_t)(L/4)*max_total*KVH*HD)||A(vcache,(size_t)(L/4)*max_total*KVH*HD)||
        A(state,(size_t)L*KDAH*KD*VD)||A(hist,(size_t)L*3*(2*kdim+vdim))||
        A(router_logits,E)||A(wts,TOPK)||A(idx,TOPK)||A(posb,1)||
        A(stage_p,(size_t)MI*HID/2+16)||A(stage_s,(size_t)MI*HID/16+16)) {
        return fail("alice workspace alloc failed");
    }
#undef A

    std::string stage = "start";
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
                e = qwen::RunQwenRopeCosSin(stream, posb, 1, ROT,
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
                const PackedExpert *pe = experts.Get(l, hidx[k]);
                if (!pe) {
                    result.error_msg = "missing routed expert layer=" +
                                       std::to_string(l) + " e=" +
                                       std::to_string(hidx[k]);
                    return cudaErrorInvalidValue;
                }
                stage="moe_expert_w1"; e = ExpertGemm(stream, *pe, 1, mixed, mgate, MI, HID, stage_p, stage_s);
                if (e) return e;
                stage="moe_expert_w3"; e = ExpertGemm(stream, *pe, 3, mixed, mup, MI, HID, stage_p, stage_s);
                if (e) return e;
                e = RunAliceSwiGLU(stream, mgate, mup, mact, MI); if (e) return e;
                stage="moe_expert_w2"; e = ExpertGemm(stream, *pe, 2, mact, mdown, HID, MI, stage_p, stage_s);
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

    for (int step = 0; step < max_new; ++step) {
        // Per-step state: the recurrent KDA state and conv history must carry
        // across the tokens of one step (reset once, before the token loop).
        cudaMemsetAsync(state, 0, state_bytes, stream);
        cudaMemsetAsync(hist, 0, hist_bytes, stream);
        for (int t = 0; t < static_cast<int>(ids.size()); ++t) {
            const bool last = (t == static_cast<int>(ids.size()) - 1);
            e = forward(ids[(size_t)t], t, last);
            if (e) {
                if (result.error_msg.empty())
                    result.error_msg = std::string("alice forward failed [") +
                                       stage + "]: " + cudaGetErrorString(e);
                return e;
            }
        }
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
            result.final_logits.resize(VOCAB);
            for (int i = 0; i < VOCAB; ++i)
                result.final_logits[(size_t)i] = __half2float(hlog[(size_t)i]);
        }
        result.tokens.push_back(best);
        ids.push_back(best);
        if (eos_id >= 0 && best == eos_id) { result.eos_terminated = true; break; }
    }
    result.ok = true;

#define F(p) if (p) cudaFree(p)
    F(x);F(mixed);F(partial);F(attn);F(attn_out);F(src);F(completed);F(qg);
    F(kq);F(kk);F(kv);F(kfa);F(kga);F(kb);F(alpha);F(gate);F(convq);F(convk);
    F(convv);F(kout);F(mgate);F(mup);F(mact);F(mdown);F(sgate);F(sup);F(sact);
    F(sdown);F(acc);F(cosb);F(sinb);F(logits);F(kcache);F(vcache);F(state);
    F(hist);F(router_logits);F(wts);F(idx);F(posb);F(stage_p);F(stage_s);
#undef F
    return cudaSuccess;
}

} // namespace lsxcommon::alice
