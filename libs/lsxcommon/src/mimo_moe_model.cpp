#include "mimo_moe_model.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/mimo_moe_kernels.h"
#include "lsxcommon/mimo_tool_format.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>

#include "mimo_moe_constants.h"
#include "mimo_moe_device_math.h"

namespace lsxcommon {
namespace {

const DenseNvfp4View *FindDense(const DenseNvfp4BatchViews &nv,
                                std::string_view suffix)
{
    for (const auto &r : nv.records)
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &r;
    return nullptr;
}

const TensorView *FindTensor(const TensorBatchViews &raw,
                             std::string_view suffix)
{
    for (const auto &t : raw.tensors)
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            return &t;
    return nullptr;
}

float Bf16BitsToFloat(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

bool UploadNvfp4(const DenseNvfp4View &v, lsx_nvfp4::LsxNvfp4Weight &w,
                 cudaStream_t st, std::string &err)
{
    if (v.rank < 2 || v.packed_bytes <= 0 || v.scale_bytes <= 0)
    {
        err = "invalid NVFP4 record";
        return false;
    }
    w.rows = static_cast<int>(v.shape[0]);
    w.cols = static_cast<int>(v.shape[1]);
    w.global_scale = v.global_scale;
    w.packed_bytes = v.packed_bytes;
    w.scale_bytes = v.scale_bytes;
    void *p = nullptr, *s = nullptr;
    if (cudaMalloc(&p, v.packed_bytes) != cudaSuccess ||
        cudaMalloc(&s, v.scale_bytes) != cudaSuccess)
    {
        err = "cudaMalloc NVFP4 failed";
        return false;
    }
    w.d_packed = static_cast<const uint8_t *>(p);
    w.d_scales = static_cast<const uint8_t *>(s);
    // Synchronous: the Arrow view (mmap) is destroyed when this call returns.
    (void)st;
    if (cudaMemcpy(p, v.packed, v.packed_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(s, v.scales, v.scale_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess)
    {
        err = "NVFP4 H2D failed";
        return false;
    }
    return true;
}

bool UploadHalf(const TensorView &t, __half **out, int64_t expect,
                cudaStream_t st, std::string &err)
{
    if (t.dtype != "BF16" || static_cast<int64_t>(t.bytes) != expect * 2)
    {
        err = "raw tensor is not BF16 of the expected size";
        return false;
    }
    std::vector<__half> h(static_cast<size_t>(expect));
    const uint16_t *b = reinterpret_cast<const uint16_t *>(t.data);
    for (int64_t i = 0; i < expect; ++i)
        h[static_cast<size_t>(i)] = __float2half(Bf16BitsToFloat(b[i]));
    void *p = nullptr;
    if (cudaMalloc(&p, static_cast<size_t>(expect) * 2) != cudaSuccess)
    {
        err = "cudaMalloc half failed";
        return false;
    }
    *out = static_cast<__half *>(p);
    (void)st;
    return cudaMemcpy(p, h.data(), static_cast<size_t>(expect) * 2,
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool UploadBf16(const TensorView &t, __nv_bfloat16 **out, int64_t expect,
                cudaStream_t st, std::string &err)
{
    if (t.dtype != "BF16" || static_cast<int64_t>(t.bytes) != expect * 2)
    {
        err = "raw tensor is not BF16 of the expected size";
        return false;
    }
    void *p = nullptr;
    if (cudaMalloc(&p, static_cast<size_t>(expect) * 2) != cudaSuccess)
    {
        err = "cudaMalloc bf16 failed";
        return false;
    }
    *out = static_cast<__nv_bfloat16 *>(p);
    (void)st;
    return cudaMemcpy(p, t.data, static_cast<size_t>(expect) * 2,
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool UploadF32(const TensorView &t, float **out, int64_t expect, cudaStream_t st,
               std::string &err)
{
    if (t.dtype != "F32" || static_cast<int64_t>(t.bytes) != expect * 4)
    {
        err = "raw tensor is not F32 of the expected size";
        return false;
    }
    void *p = nullptr;
    if (cudaMalloc(&p, static_cast<size_t>(expect) * 4) != cudaSuccess)
    {
        err = "cudaMalloc f32 failed";
        return false;
    }
    *out = static_cast<float *>(p);
    (void)st;
    return cudaMemcpy(p, t.data, static_cast<size_t>(expect) * 4,
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool Contains(const std::vector<int> &v, int x)
{
    for (int e : v)
        if (e == x) return true;
    return false;
}

} // namespace

const std::string MiMoV26FlashModel::kType = "mimo-v2.6-flash-rl";

MiMoV26FlashModel::MiMoV26FlashModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

MiMoV26FlashModel::~MiMoV26FlashModel() { Unload(); }

void MiMoV26FlashModel::FreeLayer(MimoLayerWeights &l)
{
    cudaFree(l.qkv.d_w);
    cudaFree(l.dense_gate.d_w);
    cudaFree(l.dense_up.d_w);
    cudaFree(l.dense_down.d_w);
    cudaFree(l.d_input_norm);
    cudaFree(l.d_post_norm);
    cudaFree(l.d_o_proj);
    cudaFree(l.d_sink);
    cudaFree(l.d_router_gate);
    cudaFree(l.d_router_bias);
    l = MimoLayerWeights{};
}

void MiMoV26FlashModel::TeardownWeights()
{
    for (auto &l : layers_) FreeLayer(l);
    layers_.clear();
    for (__half *p : kv_k_) cudaFree(p);
    for (__half *p : kv_v_) cudaFree(p);
    kv_k_.clear();
    kv_v_.clear();
    cudaFree(d_embed); cudaFree(d_head); cudaFree(d_final_norm);
    cudaFree(d_x); cudaFree(d_xn); cudaFree(d_qkv); cudaFree(d_attn);
    cudaFree(d_oproj); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_act);
    cudaFree(d_mlp); cudaFree(d_expert_out); cudaFree(d_router_idx);
    cudaFree(d_router_w); cudaFree(d_logits);
    d_embed = d_head = d_final_norm = nullptr;
    d_x = d_xn = d_qkv = d_attn = d_oproj = nullptr;
    d_gate = d_up = d_act = d_mlp = d_expert_out = nullptr;
    d_router_idx = nullptr; d_router_w = nullptr; d_logits = nullptr;
    mimo::FreeMimoExpertCache(expert_cache_);
    resident_bytes_ = 0;
    TeardownContext();
}

bool MiMoV26FlashModel::UploadLayer(int layer)
{
    MimoLayerWeights &L = layers_[layer];
    const int hidden = topo_.hidden;
    const int o_size = topo_.heads * topo_.v_head_dim;
    L.is_swa = !Contains(topo_.full_attn_layers, layer);
    L.is_moe = Contains(topo_.moe_layers, layer);

    auto raw = LoadTensorBatchViews(files_.layer_raw[layer]);
    if (!raw.ok())
    {
        lsxcommon::log::error("[mimo-model] cannot open layer raw file");
        return false;
    }
    std::string err;

    const TensorView *qkv = FindTensor(*raw, "self_attn.qkv_proj.weight");
    if (!qkv || qkv->rank < 2)
    {
        lsxcommon::log::error("[mimo-model] missing qkv_proj");
        return false;
    }
    L.qkv.rows = static_cast<int>(qkv->shape[0]);
    L.qkv.cols = static_cast<int>(qkv->shape[1]);
    if (!UploadHalf(*qkv, &L.qkv.d_w,
                    static_cast<int64_t>(L.qkv.rows) * L.qkv.cols, stream(), err))
    {
        lsxcommon::log::error(absl::StrCat("[mimo-model] qkv upload: ", err));
        return false;
    }
    const TensorView *onorm = FindTensor(*raw, "input_layernorm.weight");
    const TensorView *pnorm = FindTensor(*raw, "post_attention_layernorm.weight");
    const TensorView *op = FindTensor(*raw, "self_attn.o_proj.weight");
    if (!onorm || !UploadHalf(*onorm, &L.d_input_norm, hidden, stream(), err) ||
        !pnorm || !UploadHalf(*pnorm, &L.d_post_norm, hidden, stream(), err) ||
        !op || !UploadHalf(*op, &L.d_o_proj,
                           static_cast<int64_t>(hidden) * o_size, stream(), err))
    {
        lsxcommon::log::error(absl::StrCat("[mimo-model] norm/o_proj upload: ", err));
        return false;
    }
    L.d_sink = nullptr;
    if (L.is_swa)
    {
        const TensorView *sink = FindTensor(*raw, "attention_sink_bias");
        if (!sink || !UploadHalf(*sink, &L.d_sink, topo_.heads, stream(), err))
        {
            lsxcommon::log::error("[mimo-model] sink upload");
            return false;
        }
    }

    if (L.is_moe)
    {
        const TensorView *g = FindTensor(*raw, "mlp.gate.weight");
        const TensorView *b = FindTensor(*raw, "mlp.gate.e_score_correction_bias");
        if (!g || !UploadBf16(*g, &L.d_router_gate,
                              static_cast<int64_t>(topo_.num_experts) * hidden,
                              stream(), err) ||
            !b || !UploadF32(*b, &L.d_router_bias, topo_.num_experts, stream(), err))
        {
            lsxcommon::log::error("[mimo-model] router upload");
            return false;
        }
    }
    else
    {
        const TensorView *dgate = FindTensor(*raw, "mlp.gate_proj.weight");
        const TensorView *dup = FindTensor(*raw, "mlp.up_proj.weight");
        const TensorView *ddown = FindTensor(*raw, "mlp.down_proj.weight");
        if (!dgate || !dup || !ddown)
        {
            lsxcommon::log::error("[mimo-model] missing dense MLP tensors");
            return false;
        }
        auto up1 = [&](const TensorView *t, MimoLinearF16 &w) {
            w.rows = static_cast<int>(t->shape[0]);
            w.cols = static_cast<int>(t->shape[1]);
            return UploadHalf(*t, &w.d_w,
                              static_cast<int64_t>(w.rows) * w.cols, stream(),
                              err);
        };
        if (!up1(dgate, L.dense_gate) || !up1(dup, L.dense_up) ||
            !up1(ddown, L.dense_down))
        {
            lsxcommon::log::error(absl::StrCat("[mimo-model] dense MLP upload: ", err));
            return false;
        }
    }
    return true;
}

bool MiMoV26FlashModel::Load()
{
    Unload();
    tokenizer_optional_ = false;
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext())
    {
        lsxcommon::log::error("[mimo-model] stream create failed");
        return AbortLoad();
    }

    auto layout = mimo::LoadMiMoLayout(cfg_.model_path);
    if (!layout.ok())
    {
        lsxcommon::log::error(absl::StrCat("[mimo-model] layout: ",
                                           layout.status().ToString()));
        return AbortLoad();
    }
    files_ = *layout;
    topo_ = files_.topo;
    if (!SyntheticGeometryAllowed())
    {
        auto vs = mimo::ValidateMiMoGeometry(topo_);
        if (!vs.ok())
        {
            lsxcommon::log::error(absl::StrCat("[mimo-model] geometry: ",
                                               vs.ToString()));
            return AbortLoad();
        }
    }

    const int hidden = topo_.hidden;
    const int vocab = topo_.vocab;
    const int heads = topo_.heads;
    const int o_size = heads * topo_.v_head_dim;
    const int moe_I = topo_.moe_intermediate;
    const int dense_I = topo_.dense_intermediate;
    const int act_I = std::max(moe_I, dense_I);
    const int q_size = heads * topo_.head_dim;
    const int k_size_max = std::max(topo_.kv_heads_ga, topo_.kv_heads_swa) * topo_.head_dim;
    const int v_size_max = std::max(topo_.kv_heads_ga, topo_.kv_heads_swa) * topo_.v_head_dim;
    const int qkv_rows = q_size + k_size_max + v_size_max;
    o_size_ = o_size;
    kv_heads_ga_ = topo_.kv_heads_ga;
    kv_heads_swa_ = topo_.kv_heads_swa;
    max_seq_ = 8192;

    // Globals.
    auto glob = LoadTensorBatchViews(files_.global_file);
    if (!glob.ok())
    {
        lsxcommon::log::error("[mimo-model] cannot open global pinned file");
        return AbortLoad();
    }
    std::string err;
    const TensorView *emb = FindTensor(*glob, "model.embed_tokens.weight");
    const TensorView *hdf = FindTensor(*glob, "lm_head.weight");
    const TensorView *fn = FindTensor(*glob, "model.norm.weight");
    if (!emb || !UploadHalf(*emb, &d_embed, static_cast<int64_t>(vocab) * hidden,
                            stream(), err) ||
        !hdf || !UploadHalf(*hdf, &d_head, static_cast<int64_t>(vocab) * hidden,
                            stream(), err) ||
        !fn || !UploadHalf(*fn, &d_final_norm, hidden, stream(), err))
    {
        lsxcommon::log::error(absl::StrCat("[mimo-model] global upload: ", err));
        return AbortLoad();
    }

    layers_.assign(static_cast<size_t>(topo_.layers), MimoLayerWeights{});
    for (int l = 0; l < topo_.layers; ++l)
        if (!UploadLayer(l)) return AbortLoad();

    // Scratch.
    auto alloc = [](void **p, size_t bytes) { return cudaMalloc(p, bytes) == cudaSuccess; };
    if (!alloc(reinterpret_cast<void **>(&d_x), hidden * 2) ||
        !alloc(reinterpret_cast<void **>(&d_xn), hidden * 2) ||
        !alloc(reinterpret_cast<void **>(&d_qkv), static_cast<size_t>(qkv_rows) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_attn), static_cast<size_t>(o_size) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_oproj), static_cast<size_t>(hidden) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_gate), static_cast<size_t>(act_I) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_up), static_cast<size_t>(act_I) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_act), static_cast<size_t>(act_I) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_mlp), static_cast<size_t>(hidden) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_expert_out), static_cast<size_t>(hidden) * 2) ||
        !alloc(reinterpret_cast<void **>(&d_router_idx), static_cast<size_t>(topo_.top_k) * 4) ||
        !alloc(reinterpret_cast<void **>(&d_router_w), static_cast<size_t>(topo_.top_k) * 4) ||
        !alloc(reinterpret_cast<void **>(&d_logits), static_cast<size_t>(vocab) * 4))
    {
        lsxcommon::log::error("[mimo-model] scratch alloc failed");
        return AbortLoad();
    }

    // KV caches.
    kv_k_.assign(static_cast<size_t>(topo_.layers), nullptr);
    kv_v_.assign(static_cast<size_t>(topo_.layers), nullptr);
    for (int l = 0; l < topo_.layers; ++l)
    {
        const int kh = Contains(topo_.full_attn_layers, l) ? topo_.kv_heads_ga
                                                           : topo_.kv_heads_swa;
        const size_t kb = static_cast<size_t>(max_seq_) * kh * topo_.head_dim * 2;
        const size_t vb = static_cast<size_t>(max_seq_) * kh * topo_.v_head_dim * 2;
        if (cudaMalloc(reinterpret_cast<void **>(&kv_k_[l]), kb) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&kv_v_[l]), vb) != cudaSuccess)
        {
            lsxcommon::log::error("[mimo-model] KV alloc failed");
            return AbortLoad();
        }
        cudaMemsetAsync(kv_k_[l], 0, kb, stream());
        cudaMemsetAsync(kv_v_[l], 0, vb, stream());
    }

    std::string cerr;
    if (!mimo::InitMiMoExpertCache(cfg_.model_path, topo_.layers, 1,
                                   topo_.num_experts, topo_.top_k, 0, stream(),
                                   expert_cache_, cerr))
    {
        lsxcommon::log::error(absl::StrCat("[mimo-model] expert cache: ", cerr));
        return AbortLoad();
    }

    resident_bytes_ = static_cast<std::uint64_t>(vocab) * hidden * 2 * 2 +
                      static_cast<std::uint64_t>(topo_.layers) * hidden * 2 * 2;
    if (!AcquirePermanentWeights(resident_bytes_)) return AbortLoad();
    FinishLoad();
    return true;
}

bool MiMoV26FlashModel::ForwardOne(int token, int pos, float *logits)
{
    cudaStream_t st = stream();
    const int hidden = topo_.hidden;
    const int heads = topo_.heads;
    const int o_size = heads * topo_.v_head_dim;
    const int moe_I = topo_.moe_intermediate;

    cudaMemcpyAsync(d_x, d_embed + static_cast<int64_t>(token) * hidden,
                    static_cast<size_t>(hidden) * 2, cudaMemcpyDeviceToDevice, st);
    if (std::getenv("LSX_MIMO_TRACE") && pos == 0)
    {
        if (const char *dd = std::getenv("LSX_MIMO_DUMP_DIR"))
        {
            std::vector<__half> hb(static_cast<size_t>(hidden));
            cudaStreamSynchronize(st);
            cudaMemcpy(hb.data(), d_x, static_cast<size_t>(hidden) * 2,
                       cudaMemcpyDeviceToHost);
            FILE *f = std::fopen((std::string(dd) + "/emb.bin").c_str(), "wb");
            if (f) { std::fwrite(hb.data(), 2, static_cast<size_t>(hidden), f); std::fclose(f); }
        }
    }

    for (int l = 0; l < topo_.layers; ++l)
    {
        MimoLayerWeights &L = layers_[l];
        mimo::MimoAttnShape ap;
        ap.heads = heads;
        ap.kv_heads = L.is_swa ? topo_.kv_heads_swa : topo_.kv_heads_ga;
        ap.head_dim = topo_.head_dim;
        ap.v_head_dim = topo_.v_head_dim;
        ap.rope_dim = topo_.rope_dim;
        ap.is_swa = L.is_swa;
        ap.window = topo_.sliding_window;
        ap.value_scale = static_cast<float>(topo_.value_scale);
        ap.scale = 1.0f / std::sqrt(static_cast<float>(topo_.head_dim));
        ap.rope_theta = L.is_swa ? topo_.rope_theta_swa : topo_.rope_theta_ga;
        ap.q_size = heads * topo_.head_dim;
        ap.k_size = ap.kv_heads * topo_.head_dim;
        ap.v_size = ap.kv_heads * topo_.v_head_dim;

        if (mimo::RunMimoRmsNorm(st, d_x, L.d_input_norm, 1, hidden,
                           static_cast<float>(topo_.rms_eps), d_xn) != cudaSuccess)
            return false;
        auto dbg = [&](const char *nm, const __half *p, int n) {
            if (!std::getenv("LSX_MIMO_TRACE") || pos > 1) return;
            const char *dd = std::getenv("LSX_MIMO_DUMP_DIR");
            if (!dd) return;
            std::vector<__half> b(static_cast<size_t>(n));
            cudaStreamSynchronize(st);
            cudaMemcpy(b.data(), p, static_cast<size_t>(n) * 2, cudaMemcpyDeviceToHost);
            FILE *f = std::fopen((std::string(dd) + "/p" + std::to_string(pos) + "_l" + std::to_string(l) + "_" + nm + ".bin").c_str(), "wb");
            if (f) { std::fwrite(b.data(), 2, static_cast<size_t>(n), f); std::fclose(f); }
        };
        dbg("innorm", d_xn, hidden);
        if (gemv::RunLxGemmVec(st, L.qkv.d_w, d_xn, 1, L.qkv.rows, L.qkv.cols,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               d_qkv, gemv::LxDtype::kF16, 0.0f) != cudaSuccess)
            return false;
        dbg("qkv", d_qkv, L.qkv.rows);
        if (mimo::RunMimoAttentionDecode(st, d_qkv, pos, L.d_sink, ap, kv_k_[l],
                                   kv_v_[l], d_attn) != cudaSuccess)
            return false;
        dbg("attn", d_attn, topo_.heads * topo_.v_head_dim);
        if (gemv::RunLxGemmVec(st, L.d_o_proj, d_attn, 1, hidden, o_size,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               d_oproj, gemv::LxDtype::kF16, 0.0f) != cudaSuccess)
            return false;
        if (mimo::RunMimoAddInPlace(st, d_oproj, hidden, d_x) != cudaSuccess)
            return false;
        dbg("postattn", d_x, hidden);

        if (mimo::RunMimoRmsNorm(st, d_x, L.d_post_norm, 1, hidden,
                           static_cast<float>(topo_.rms_eps), d_xn) != cudaSuccess)
            return false;
        dbg("postnorm", d_xn, hidden);

        if (!L.is_moe)
        {
            if (gemv::RunLxGemmVec(st, L.dense_gate.d_w, d_xn, 1,
                                   L.dense_gate.rows, L.dense_gate.cols,
                                   gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.0f, d_gate, gemv::LxDtype::kF16, 0.0f) != cudaSuccess)
                return false;
            if (gemv::RunLxGemmVec(st, L.dense_up.d_w, d_xn, 1,
                                   L.dense_up.rows, L.dense_up.cols,
                                   gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.0f, d_up, gemv::LxDtype::kF16, 0.0f) != cudaSuccess)
                return false;
            if (mimo::RunMimoSwiGLU(st, d_gate, d_up, L.dense_up.rows, d_act) != cudaSuccess)
                return false;
            if (gemv::RunLxGemmVec(st, L.dense_down.d_w, d_act, 1,
                                   L.dense_down.rows, L.dense_down.cols,
                                   gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.0f, d_mlp, gemv::LxDtype::kF16, 0.0f) != cudaSuccess)
                return false;
            dbg("dgate", d_gate, L.dense_up.rows);
            dbg("dup", d_up, L.dense_up.rows);
            dbg("act", d_act, L.dense_up.rows);
            dbg("mlp", d_mlp, hidden);
        }
        else
        {
            cudaMemsetAsync(d_mlp, 0, static_cast<size_t>(hidden) * 2, st);
            if (std::getenv("LSX_MIMO_NO_MOE") == nullptr)
            {
            if (mimo::RunMimoRouter(st, d_xn, L.d_router_gate, L.d_router_bias, 1,
                              hidden, topo_.num_experts, topo_.top_k,
                              d_router_idx, d_router_w) != cudaSuccess)
                return false;
            std::vector<int> idx(topo_.top_k);
            std::vector<float> w(topo_.top_k);
            if (cudaMemcpyAsync(idx.data(), d_router_idx, topo_.top_k * 4,
                                cudaMemcpyDeviceToHost, st) != cudaSuccess ||
                cudaMemcpyAsync(w.data(), d_router_w, topo_.top_k * 4,
                                cudaMemcpyDeviceToHost, st) != cudaSuccess)
                return false;
            cudaStreamSynchronize(st);
            cudaMemsetAsync(d_mlp, 0, static_cast<size_t>(hidden) * 2, st);
            std::string err;
            for (int e = 0; e < topo_.top_k; ++e)
            {
                mimo::MimoExpertView v = mimo::ResolveMimoExpertView(
                    expert_cache_, l, idx[e], st, err);
                if (!v.gate_packed) { lsxcommon::log::error(absl::StrCat("[mimo-model] expert: ", err)); return false; }
                if (lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                        st, d_xn, v.gate_packed, v.gate_scales, v.gate_global_scale,
                        d_gate, hidden, moe_I) != cudaSuccess)
                    return false;
                if (lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                        st, d_xn, v.up_packed, v.up_scales, v.up_global_scale,
                        d_up, hidden, moe_I) != cudaSuccess)
                    return false;
                if (mimo::RunMimoSwiGLU(st, d_gate, d_up, moe_I, d_act) != cudaSuccess)
                    return false;
                if (lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                        st, d_act, v.down_packed, v.down_scales, v.down_global_scale,
                        d_expert_out, moe_I, hidden) != cudaSuccess)
                    return false;
                if (mimo::RunMimoScaleAdd(st, d_expert_out, w[e], hidden, d_mlp) != cudaSuccess)
                    return false;
                if (e == 0 && std::getenv("LSX_MIMO_TRACE_MOE"))
                {
                    auto dump = [&](const std::string &suffix, const __half *p, int n) {
                        const char *dir = std::getenv("LSX_MIMO_DUMP_DIR");
                        if (!dir) return;
                        std::vector<__half> b(static_cast<size_t>(n));
                        cudaStreamSynchronize(st);
                        cudaMemcpy(b.data(), p, static_cast<size_t>(n) * 2,
                                   cudaMemcpyDeviceToHost);
                        FILE *f = std::fopen(
                            (std::string(dir) + "/" + suffix).c_str(), "wb");
                        if (f)
                        {
                            std::fwrite(b.data(), 2, static_cast<size_t>(n), f);
                            std::fclose(f);
                        }
                    };
                    dump(absl::StrCat("layer", l, "_xn.bin"), d_xn, hidden);
                    dump(absl::StrCat("layer", l, "_gate.bin"), d_gate, moe_I);
                    dump(absl::StrCat("layer", l, "_up.bin"), d_up, moe_I);
                    dump(absl::StrCat("layer", l, "_act.bin"), d_act, moe_I);
                    dump(absl::StrCat("layer", l, "_down.bin"), d_expert_out, hidden);
                    FILE *f = std::fopen(
                        (std::string(std::getenv("LSX_MIMO_DUMP_DIR")) +
                         "/layer" + std::to_string(l) + "_meta.txt")
                            .c_str(),
                        "w");
                    if (f)
                    {
                        std::fprintf(f, "expert %d w0 %a\n", idx[0], w[0]);
                        std::fclose(f);
                    }
                }
                if (e == 0 && std::getenv("LSX_MIMO_TRACE_MOE"))
                {
                    auto hnorm = [&](const __half *p, int n) {
                        std::vector<__half> b(static_cast<size_t>(n));
                        cudaStreamSynchronize(st);
                        cudaMemcpy(b.data(), p, static_cast<size_t>(n) * 2,
                                   cudaMemcpyDeviceToHost);
                        double ss = 0.0;
                        for (int i = 0; i < n; ++i)
                        {
                            const float v = __half2float(b[static_cast<size_t>(i)]);
                            ss += static_cast<double>(v) * v;
                        }
                        return std::sqrt(ss);
                    };
                    lsxcommon::log::info(absl::StrCat(
                        "[mimo-moe] layer ", l, " w0=", w[0], " idx0=", idx[0],
                        " gs_gate=", v.gate_global_scale,
                        " |xn|=", hnorm(d_xn, hidden),
                        " |gate|=", hnorm(d_gate, moe_I),
                        " |up|=", hnorm(d_up, moe_I),
                        " |act|=", hnorm(d_act, moe_I),
                        " |down|=", hnorm(d_expert_out, hidden)));
                    // Host router reference.
                    std::vector<__half> xh(static_cast<size_t>(hidden));
                    std::vector<__nv_bfloat16> gh(
                        static_cast<size_t>(topo_.num_experts) * hidden);
                    cudaMemcpy(xh.data(), d_xn,
                               static_cast<size_t>(hidden) * 2,
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(gh.data(), L.d_router_gate,
                               gh.size() * 2, cudaMemcpyDeviceToHost);
                    std::vector<int> hidx; std::vector<float> hw;
                    {
                        std::vector<float> xf(static_cast<size_t>(hidden));
                        for (int i = 0; i < hidden; ++i)
                            xf[static_cast<size_t>(i)] =
                                __half2float(xh[static_cast<size_t>(i)]);
                        std::vector<float> gwf(gh.size());
                        for (size_t i = 0; i < gh.size(); ++i)
                            gwf[i] = __bfloat162float(gh[i]);
                        std::vector<float> bias(topo_.num_experts);
                        cudaMemcpy(bias.data(), L.d_router_bias,
                                   bias.size() * 4, cudaMemcpyDeviceToHost);
                        lsxcommon::mimo::ref::RouterTopK(
                            xf.data(), gwf.data(), bias.data(), hidden,
                            topo_.num_experts, topo_.top_k, hidx, hw);
                    }
                    for (int t = 0; t < topo_.top_k && t < 3; ++t)
                        lsxcommon::log::info(absl::StrCat(
                            "[mimo-moe-host] layer ", l, " t=", t,
                            " host_idx=", hidx[t], " host_w=", hw[t],
                            " dev_idx=", idx[t], " dev_w=", w[t]));
                }
            }
            }  // LSX_MIMO_NO_MOE
        }
        if (mimo::RunMimoAddInPlace(st, d_mlp, hidden, d_x) != cudaSuccess)
            return false;

        if (const char *tr = std::getenv("LSX_MIMO_TRACE"))
        {
            (void)tr;
            if (const char *dd = std::getenv("LSX_MIMO_DUMP_DIR"))
            {
                std::vector<__half> hb(static_cast<size_t>(hidden));
                cudaStreamSynchronize(st);
                cudaMemcpy(hb.data(), d_x, static_cast<size_t>(hidden) * 2,
                           cudaMemcpyDeviceToHost);
                FILE *f = std::fopen(
                    (std::string(dd) + "/dx_p" + std::to_string(pos) + "_l" +
                     std::to_string(l) + ".bin")
                        .c_str(),
                    "wb");
                if (f)
                {
                    std::fwrite(hb.data(), 2, static_cast<size_t>(hidden), f);
                    std::fclose(f);
                }
            }
            std::vector<__half> hbuf(static_cast<size_t>(hidden));
            cudaStreamSynchronize(st);
            cudaMemcpy(hbuf.data(), d_x, static_cast<size_t>(hidden) * 2,
                       cudaMemcpyDeviceToHost);
            double ss = 0.0;
            for (int i = 0; i < hidden; ++i)
            {
                const float v = __half2float(hbuf[static_cast<size_t>(i)]);
                ss += static_cast<double>(v) * v;
            }
            lsxcommon::log::info(absl::StrCat("[mimo-trace] layer ", l, " pos ",
                                              pos, " |x|=", std::sqrt(ss),
                                              " x0=",
                                              __half2float(hbuf[0])));
        }
    }

    if (mimo::RunMimoRmsNorm(st, d_x, d_final_norm, 1, hidden,
                       static_cast<float>(topo_.rms_eps), d_xn) != cudaSuccess)
        return false;
    if (gemv::RunLxGemmVec(st, d_head, d_xn, 1, topo_.vocab, hidden,
                           gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                           logits, gemv::LxDtype::kF32, 0.0f) != cudaSuccess)
        return false;
    return true;
}

gpu::InferenceResult MiMoV26FlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    if (!EnsureContext()) { r.error_msg = "cuda init failed"; return r; }
    cudaStream_t st = stream();

    const int total = static_cast<int>(req.input_ids.size()) + req.max_gen_tokens;
    if (total > max_seq_)
    {
        r.error_msg = "sequence exceeds the KV capacity";
        return r;
    }

    std::vector<int32_t> gen;
    std::vector<float> host_logits(topo_.vocab);
    int pos = 0;
    for (int32_t t : req.input_ids)
    {
        if (!ForwardOne(t, pos, d_logits)) { r.error_msg = "forward failed"; return r; }
        ++pos;
    }
    auto argmax = [&]() {
        if (cudaMemcpyAsync(host_logits.data(), d_logits,
                            static_cast<size_t>(topo_.vocab) * 4,
                            cudaMemcpyDeviceToHost, st) != cudaSuccess)
            return -1;
        cudaStreamSynchronize(st);
        int best = 0;
        for (int i = 1; i < topo_.vocab; ++i)
            if (host_logits[i] > host_logits[best]) best = i;
        return best;
    };
    int next = argmax();
    const int32_t eos = (req.ignore_eos || !tok_db_) ? kIgnoreEosTokenId : tok_db_->eos_id;
    for (int step = 0; step < req.max_gen_tokens; ++step)
    {
        gen.push_back(next);
        if (next == eos) { r.eos_terminated = true; break; }
        if (step + 1 >= req.max_gen_tokens) break;
        if (!ForwardOne(next, pos, d_logits)) { r.error_msg = "forward failed"; return r; }
        ++pos;
        next = argmax();
        if (next < 0) { r.error_msg = "logit copy failed"; return r; }
    }

    runtime::FillInferenceResult(
        r, {.tokens = &gen,
            .final_logits = &host_logits,
            .ttft_ms = 0.0,
            .total_ms = 0.0,
            .error_msg = &r.error_msg,
            .eos_terminated = r.eos_terminated,
            .kv_slots_used = pos,
            .first_maxv = 0.f,
            .resident_bytes = resident_bytes_},
        topo_.layers);
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult MiMoV26FlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req, {.frame_prompt = true});
}

bool MiMoV26FlashModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                            std::string_view raw_prompt,
                                            std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back(
        {"user", std::string(raw_prompt), "", {}});
    return mimo::BuildChatInputIds(tok, conv, /*add_generation_prompt=*/true,
                                   /*enable_thinking=*/false, out_ids);
}

bool MiMoV26FlashModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    return mimo::BuildChatInputIds(tok, conv, true, false, out_ids);
}

bool MiMoV26FlashModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return mimo::ParseChatToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
