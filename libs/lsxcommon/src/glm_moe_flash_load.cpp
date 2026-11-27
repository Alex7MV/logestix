// GLM-5.3-Flash loader implementation: strict dense upload (pinned ring)
// + routed-expert pinned slabs. Mirrors the qwen_moe loader mechanics
// (fastload helpers reused by reference) against the glm5next index format.
#include "lsxcommon/glm_moe_flash_load.h"
#include "lsxcommon/synthetic_geometry.h"

#include "fast_load_internal.h"
#include "glm_moe_flash_constants.h"

#include <absl/strings/str_cat.h>
#include <arrow/api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace lsxcommon {
namespace glm_moe_flash {

using fastload::AlignUp;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::PinnedStagingRing;
using fastload::UploadNvfp4Weight;

namespace {

bool ReadIndexJson(const fs::path &idx_path, FlashTopology &top,
                   std::string &error)
{
    std::ifstream f(idx_path, std::ios::binary);
    if (!f) {
        error = "cannot open " + idx_path.string();
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();
    // Minimal scalar extraction (loader-side: convert is the authority).
    auto find_int = [&](const char *key, int dflt) {
        const std::string pat = std::string("\"") + key + "\": ";
        const size_t p = json.find(pat);
        if (p == std::string::npos) return dflt;
        return std::atoi(json.c_str() + p + pat.size());
    };
    top.hidden = find_int("hidden_size", 0);
    top.layers = find_int("num_layers", 0);
    top.vocab = find_int("vocab_size", 0);
    top.heads = find_int("num_attention_heads", 0);
    top.qk_head_dim = find_int("qk_head_dim", 0);
    top.q_lora = find_int("q_lora_rank", 0);
    top.kv_lora = find_int("kv_lora_rank", 0);
    top.num_experts = find_int("num_experts", 0);
    top.top_k = find_int("num_experts_per_tok", 0);
    top.moe_intermediate = find_int("moe_intermediate_size", 0);
    top.dense_intermediate = find_int("dense_intermediate_size", 0);
    top.hc_count = find_int("hc_count", 0);
    top.conv_kernel = find_int("linear_conv_kernel_dim", 4);
    top.indexer_heads = find_int("indexer_n_heads", 0);
    top.indexer_head_dim = find_int("indexer_head_dim", 0);
    top.indexer_topk = find_int("indexer_topk", 0);
    top.pad_id = find_int("pad_token_id", -1);
    if (json.find("\"format\": \"logestix-glm5next\"") == std::string::npos) {
        error = "unexpected format tag in " + idx_path.string();
        return false;
    }
    // Optional conversion-time VRAM profile (IModel::VramRequirements). A tree
    // converted before the block existed loads exactly as before - nothing in
    // the loader or the planner depends on it, and the session reserves its own
    // scratch dynamically (OPEN_TASKS §U).
    {
        std::string verr;
        if (!lsxcommon::ParseVramBlock(json, top.vram, verr)) top.vram = {};
    }
    if (!SyntheticGeometryAllowed()
        && (top.hidden != frozen::kHidden || top.layers != frozen::kLayers
            || top.num_experts != frozen::kNumExperts)) {
        error = "index topology disagrees with frozen geometry";
        return false;
    }
    return true;
}

// BF16 bits -> F32 upconvert on host, then single raw upload (hc fn
// tensors feed the FP32 run_hc_pre/run_hc_post path).
bool UploadBf16ToF32(PinnedStagingRing &ring, const uint8_t *host_bf16,
                     int64_t numel, float **d_dst, cudaStream_t stream,
                     std::uint64_t &bytes_acc)
{
    std::vector<float> h(static_cast<size_t>(numel));
    const uint16_t *src = reinterpret_cast<const uint16_t *>(host_bf16);
    for (int64_t i = 0; i < numel; ++i) {
        const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&h[static_cast<size_t>(i)], &bits, 4);
    }
    if (cudaMalloc(reinterpret_cast<void **>(d_dst),
                   static_cast<size_t>(numel) * 4) != cudaSuccess)
        return false;
    if (ring.UploadRaw(h.data(), *d_dst, static_cast<size_t>(numel) * 4,
                       stream, bytes_acc) != cudaSuccess) {
        cudaFree(*d_dst);
        return false;
    }
    return true;
}

cudaError_t UploadBf16Raw(PinnedStagingRing &ring, const uint8_t *host_bf16,
                          int64_t bytes, uint8_t **d_dst,
                          cudaStream_t stream, std::uint64_t &bytes_acc)
{
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(d_dst),
                               static_cast<size_t>(bytes));
    if (e != cudaSuccess) return e;
    e = ring.UploadRaw(host_bf16, *d_dst, static_cast<size_t>(bytes), stream,
                       bytes_acc);
    if (e != cudaSuccess) cudaFree(*d_dst);
    return e;
}

// Validates the frozen geometry of the parsed index. Kept explicit so the
// loader fails fast on any mismatched conversion output.
bool ValidateGeometry(const FlashTopology &top, std::string &why)
{
    using namespace frozen;
    if (top.hidden != kHidden) why = "hidden_size";
    else if (top.layers != kLayers) why = "num_layers";
    else if (top.vocab != kVocab) why = "vocab_size";
    else if (top.num_experts != kNumExperts) why = "num_experts";
    else if (top.top_k != kTopK) why = "top_k";
    else if (top.hc_count != kHcCount) why = "hc_count";
    else if (top.q_lora != kQLora) why = "q_lora_rank";
    else if (top.kv_lora != kKvLora) why = "kv_lora_rank";
    else if (top.moe_intermediate != kMoeIntermediate) why = "moe_intermediate";
    else return true;
    return false;
}

} // namespace

void FlashLayerWeights::Free()
{
    cudaFree(d_input_norm);
    cudaFree(d_post_norm);
    cudaFree(d_hc_attn_fn);
    cudaFree(d_hc_attn_base);
    cudaFree(d_hc_attn_scale);
    cudaFree(d_hc_ffn_fn);
    cudaFree(d_hc_ffn_base);
    cudaFree(d_hc_ffn_scale);
    kda_q.Free();
    kda_k.Free();
    kda_v.Free();
    kda_o.Free();
    cudaFree(d_kda_q_conv);
    cudaFree(d_kda_k_conv);
    cudaFree(d_kda_v_conv);
    cudaFree(d_A_log);
    cudaFree(d_dt_bias);
    cudaFree(d_b_proj);
    cudaFree(d_f_a);
    cudaFree(d_f_b);
    cudaFree(d_g_a);
    cudaFree(d_g_b);
    cudaFree(d_o_norm);
    q_a.Free();
    q_b.Free();
    kv_a.Free();
    o_big.Free();
    cudaFree(d_q_a_norm);
    cudaFree(d_kv_a_norm);
    cudaFree(d_kv_b);
    cudaFree(d_idx_wq_b);
    cudaFree(d_idx_wk);
    cudaFree(d_idx_knorm_w);
    cudaFree(d_idx_knorm_b);
    cudaFree(d_idx_wproj);
    cudaFree(d_idx_gate);
    cudaFree(d_idx_ape);
    cudaFree(d_router);
    cudaFree(d_router_bias);
    shared_w1.Free();
    shared_w2.Free();
    shared_w3.Free();
    mlp_gate.Free();
    mlp_up.Free();
    mlp_down.Free();
    *this = {};
}

void FlashPinnedWeights::Free()
{
    cudaFree(d_embed_bf16);
    cudaFree(d_head_bf16);
    cudaFree(d_final_norm);
    for (auto &l : layers) l.Free();
    layers.clear();
    *this = {};
}

bool LoadGlmFlashDenseWeights(const std::string &model_path,
                              FlashPinnedWeights &pw, cudaStream_t stream,
                              std::string &error)
{
    namespace fs = std::filesystem;
    pw.Free();
    error.clear();

    const fs::path base = fs::path(model_path);
    const fs::path pinned_dir = base / "pinned";
    if (!fs::exists(pinned_dir)) {
        error = "no pinned dir under " + base.string();
        return false;
    }

    if (!ReadIndexJson(base / "model.index.json", pw.top, error)) return false;
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!ValidateGeometry(pw.top, why)) {
            error = "frozen geometry mismatch: " + why;
            return false;
        }
    }

    PinnedStagingRing ring;
    if (!ring.ok()) {
        error = "pinned staging ring init failed";
        return false;
    }
    std::uint64_t bytes_acc = 0;

    // ── Globals: embed / lm_head raw BF16 bits + final norm FP16. ──
    {
        auto raw_res = LoadTensorBatchViews(pinned_dir
                                                      / "global.arrow");
        if (!raw_res.ok()) {
            error = "global.arrow read failed: " + raw_res.status().ToString();
            return false;
        }
        for (const char *name :
             {"embed_tokens.weight", "lm_head.weight"}) {
            const TensorView *t =
                FindTensorSuffix(*raw_res, name);
            if (!t || t->rank != 2) {
                error = std::string("missing global: ") + name;
                pw.Free();
                return false;
            }
            uint8_t *dst = nullptr;
            cudaError_t e =
                UploadBf16Raw(ring, t->data, t->bytes, &dst, stream, bytes_acc);
            if (e != cudaSuccess) {
                error = std::string("upload failed: ") + name;
                pw.Free();
                return false;
            }
            if (std::string_view(name).find("embed") != std::string_view::npos)
                pw.d_embed_bf16 = dst;
            else pw.d_head_bf16 = dst;
            pw.embed_bytes = t->bytes;
        }
        const TensorView *fn =
            FindTensorSuffix(*raw_res, "norm.weight");
        if (!fn || fn->rank != 1) {
            error = "final norm missing in global.arrow";
            pw.Free();
            return false;
        }
        pw.final_norm_size = static_cast<int>(fn->shape[0]);
        cudaError_t e =
            cudaMalloc(reinterpret_cast<void **>(&pw.d_final_norm),
                       static_cast<size_t>(fn->shape[0]) * sizeof(__half));
        if (e != cudaSuccess
            || (e = ring.UploadBf16AsFp16(
                    fn->data, fn->shape[0], pw.d_final_norm, stream,
                    bytes_acc)) != cudaSuccess) {
            error = "final norm upload failed";
            pw.Free();
            return false;
        }
    }

    // ── Per-layer upload. ──
    const int num_layers = pw.top.layers;
    pw.layers.resize(static_cast<size_t>(num_layers));

    for (int l = 0; l < num_layers; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        const std::string l3 = absl::StrCat("layer_",
                                            absl::Dec(l, absl::kZeroPad3));
        auto raw = LoadTensorBatchViews(pinned_dir
                                                  / (l3 + ".arrow"));
        if (!raw.ok()) {
            error = l3 + ".arrow read failed: " + raw.status().ToString();
            pw.Free();
            return false;
        }

        lw.is_kda = frozen::IsKdaLayer(l);
        lw.is_moe = frozen::IsMoeLayer(l);
        // Frozen-range layers without tensors flush empty raw batches at
        // conversion; under the synthetic gate they are skipped (loaded
        // stays false). Real conversions never produce them — strict path.
        if (SyntheticGeometryAllowed() && raw->tensors.empty()) continue;
        lw.loaded = true;

        // Norms (FP16).
        if (const TensorView *t =
                FindTensorSuffix(*raw, "input_layernorm.weight")) {
            lw.norm_size = static_cast<int>(t->shape[0]);
            cudaMalloc(reinterpret_cast<void **>(&lw.d_input_norm),
                       static_cast<size_t>(t->shape[0]) * sizeof(__half));
            if (ring.UploadBf16AsFp16(t->data, t->shape[0], lw.d_input_norm,
                                      stream, bytes_acc) != cudaSuccess) {
                error = l3 + " input norm upload failed";
                pw.Free();
                return false;
            }
        }
        if (const TensorView *t =
                FindTensorSuffix(*raw, "post_attention_layernorm.weight")) {
            cudaMalloc(reinterpret_cast<void **>(&lw.d_post_norm),
                       static_cast<size_t>(t->shape[0]) * sizeof(__half));
            if (ring.UploadBf16AsFp16(t->data, t->shape[0], lw.d_post_norm,
                                      stream, bytes_acc) != cudaSuccess) {
                error = l3 + " post norm upload failed";
                pw.Free();
                return false;
            }
        }

        // Hyper-connections (raw bits).
        if (const TensorView *t = FindTensorSuffix(*raw, "hc_attn_fn")) {
            lw.hc_fn_rows = t->shape[0];
            lw.hc_fn_cols = t->rank >= 2 ? t->shape[1] : 0;
            int64_t numel = 1;
            for (int64_t d = 0; d < t->rank; ++d) numel *= t->shape[d];
            if (!UploadBf16ToF32(ring, t->data, numel,
                                 reinterpret_cast<float **>(&lw.d_hc_attn_fn),
                                 stream, bytes_acc)) {
                error = l3 + " hc_attn_fn upload failed";
                pw.Free();
                return false;
            }
        }
        if (const TensorView *t = FindTensorSuffix(*raw, "hc_ffn_fn")) {
            int64_t numel2 = 1;
            for (int64_t d = 0; d < t->rank; ++d) numel2 *= t->shape[d];
            if (!UploadBf16ToF32(ring, t->data, numel2,
                                 reinterpret_cast<float **>(&lw.d_hc_ffn_fn),
                                 stream, bytes_acc)) {
                error = l3 + " hc_ffn_fn upload failed";
                pw.Free();
                return false;
            }
        }
        if (const TensorView *t = FindTensorSuffix(*raw, "hc_attn_base")) {
            cudaMalloc(reinterpret_cast<void **>(&lw.d_hc_attn_base),
                       t->bytes);
            if (ring.UploadRaw(t->data, lw.d_hc_attn_base, t->bytes, stream,
                               bytes_acc) != cudaSuccess) {
                error = l3 + " hc_attn_base upload failed";
                pw.Free();
                return false;
            }
        }
        if (const TensorView *t = FindTensorSuffix(*raw, "hc_ffn_base")) {
            cudaMalloc(reinterpret_cast<void **>(&lw.d_hc_ffn_base),
                       t->bytes);
            if (ring.UploadRaw(t->data, lw.d_hc_ffn_base, t->bytes, stream,
                               bytes_acc) != cudaSuccess) {
                error = l3 + " hc_ffn_base upload failed";
                pw.Free();
                return false;
            }
        }
        if (const TensorView *t = FindTensorSuffix(*raw, "hc_attn_scale")) {
            cudaMalloc(reinterpret_cast<void **>(&lw.d_hc_attn_scale),
                       t->bytes);
            if (ring.UploadRaw(t->data, lw.d_hc_attn_scale, t->bytes, stream,
                               bytes_acc) != cudaSuccess) {
                error = l3 + " hc_attn_scale upload failed";
                pw.Free();
                return false;
            }
        }
        if (const TensorView *t = FindTensorSuffix(*raw, "hc_ffn_scale")) {
            cudaMalloc(reinterpret_cast<void **>(&lw.d_hc_ffn_scale),
                       t->bytes);
            if (ring.UploadRaw(t->data, lw.d_hc_ffn_scale, t->bytes, stream,
                               bytes_acc) != cudaSuccess) {
                error = l3 + " hc_ffn_scale upload failed";
                pw.Free();
                return false;
            }
        }

        if (lw.is_kda) {
            // KDA: q/k/v/o NVFP4 (quantized at conversion time), conv
            // weights FP16, gating/scalars F32.
            {
                auto nv = LoadDenseNvfp4BatchViews(pinned_dir
                                                   / (l3 + "_nvfp4.arrow"));
                if (!nv.ok()) {
                    error = l3 + "_nvfp4.arrow read failed: "
                          + nv.status().ToString();
                    pw.Free();
                    return false;
                }
                struct KdaNv {
                    const char *suffix;
                    lsx_nvfp4::LsxNvfp4Weight *dst;
                };
                const KdaNv kNv[] = {
                    {"self_attn.q_proj.weight", &lw.kda_q},
                    {"self_attn.k_proj.weight", &lw.kda_k},
                    {"self_attn.v_proj.weight", &lw.kda_v},
                    {"self_attn.o_proj.weight", &lw.kda_o},
                };
                for (const auto &kv : kNv) {
                    std::string dummy;
                    const cudaError_t e = UploadNvfp4Weight(
                        FindNvfp4Suffix(*nv, kv.suffix), *kv.dst, stream,
                        ring, bytes_acc, &dummy);
                    if (e != cudaSuccess || !kv.dst->d_packed) {
                        error = std::string(l3) + " kda nvfp4 upload failed ("
                              + cudaGetErrorString(e) + "): " + kv.suffix;
                        pw.Free();
                        return false;
                    }
                }
            }
            // KDA: FP16 weights + F32 scalars.
            struct KdaFp16 {
                const char *suffix;
                __half **dst;
            };
            const KdaFp16 kFp16[] = {
                {"self_attn.q_conv1d.weight", &lw.d_kda_q_conv},
                {"self_attn.k_conv1d.weight", &lw.d_kda_k_conv},
                {"self_attn.v_conv1d.weight", &lw.d_kda_v_conv},
                {"self_attn.b_proj.weight", &lw.d_b_proj},
                {"self_attn.f_a_proj.weight", &lw.d_f_a},
                {"self_attn.f_b_proj.weight", &lw.d_f_b},
                {"self_attn.g_a_proj.weight", &lw.d_g_a},
                {"self_attn.g_b_proj.weight", &lw.d_g_b},
                {"self_attn.o_norm.weight", &lw.d_o_norm},
            };
            for (const auto &kv : kFp16) {
                const TensorView *t = FindTensorSuffix(*raw, kv.suffix);
                if (!t) {
                    error = std::string(l3) + " missing " + kv.suffix;
                    pw.Free();
                    return false;
                }
                int64_t numel = 1;
                for (int64_t d = 0; d < t->rank; ++d) numel *= t->shape[d];
                cudaMalloc(reinterpret_cast<void **>(kv.dst),
                           static_cast<size_t>(numel) * sizeof(__half));
                if (ring.UploadBf16AsFp16(t->data, numel, *kv.dst,
                                          stream, bytes_acc)
                    != cudaSuccess) {
                    error = std::string(l3) + " upload failed: "
                          + kv.suffix;
                    pw.Free();
                    return false;
                }
            }
            for (const auto &kv : {std::pair{"self_attn.A_log", &lw.d_A_log},
                                   {"self_attn.dt_bias", &lw.d_dt_bias}}) {
                const TensorView *t = FindTensorSuffix(*raw, kv.first);
                if (!t) {
                    error = std::string(l3) + " missing " + kv.first;
                    pw.Free();
                    return false;
                }
                cudaMalloc(reinterpret_cast<void **>(kv.second), t->bytes);
                if (ring.UploadRaw(t->data, *kv.second, t->bytes, stream,
                                   bytes_acc) != cudaSuccess) {
                    error = std::string(l3) + " upload failed: "
                          + kv.first;
                    pw.Free();
                    return false;
                }
            }
        } else {
            // DSA: NVFP4 projections + FP16 norms + BF16 raw kv_b + indexer.
            auto nv = LoadDenseNvfp4BatchViews(pinned_dir
                                                         / (l3
                                                            + "_nvfp4.arrow"));
            if (!nv.ok()) {
                error = l3 + "_nvfp4.arrow read failed: "
                      + nv.status().ToString();
                pw.Free();
                return false;
            }
            std::string dummy;
            cudaError_t e;
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "q_a_proj.weight"),
                                  lw.q_a, stream, ring, bytes_acc, &dummy);
            if (e != cudaSuccess || !lw.q_a.d_packed) {
                error = l3 + " q_a upload failed";
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "q_b_proj.weight"),
                                  lw.q_b, stream, ring, bytes_acc, &dummy);
            if (e != cudaSuccess || !lw.q_b.d_packed) {
                error = l3 + " q_b upload failed";
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(
                FindNvfp4Suffix(*nv, "kv_a_proj_with_mqa.weight"), lw.kv_a,
                stream, ring, bytes_acc, &dummy);
            if (e != cudaSuccess || !lw.kv_a.d_packed) {
                error = l3 + " kv_a upload failed";
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(FindNvfp4Suffix(*nv, "o_proj.weight"),
                                  lw.o_big, stream, ring, bytes_acc, &dummy);
            if (e != cudaSuccess || !lw.o_big.d_packed) {
                error = l3 + " o upload failed";
                pw.Free();
                return false;
            }
            for (const auto &kv :
                 {std::pair{"self_attn.q_a_layernorm.weight",
                            &lw.d_q_a_norm},
                  {"self_attn.kv_a_layernorm.weight", &lw.d_kv_a_norm}}) {
                const TensorView *t = FindTensorSuffix(*raw, kv.first);
                if (!t) {
                    error = std::string(l3) + " missing " + kv.first;
                    pw.Free();
                    return false;
                }
                int64_t numel = 1;
                for (int64_t d = 0; d < t->rank; ++d) numel *= t->shape[d];
                cudaMalloc(reinterpret_cast<void **>(kv.second),
                           static_cast<size_t>(numel) * sizeof(__half));
                if (ring.UploadBf16AsFp16(t->data, numel, *kv.second,
                                          stream, bytes_acc)
                    != cudaSuccess) {
                    error = std::string(l3) + " norm upload failed";
                    pw.Free();
                    return false;
                }
            }
            if (const TensorView *t =
                    FindTensorSuffix(*raw, "self_attn.kv_b_proj.weight")) {
                cudaError_t e2 =
                    UploadBf16Raw(ring, t->data, t->bytes,
                                  reinterpret_cast<uint8_t **>(&lw.d_kv_b),
                                  stream, bytes_acc);
                if (e2 != cudaSuccess) {
                    error = l3 + " kv_b upload failed";
                    pw.Free();
                    return false;
                }
            } else {
                error = l3 + " missing kv_b_proj";
                pw.Free();
                return false;
            }
            // Indexer payloads: uploaded as BF16 bits; the indexer kernels
            // read them directly. Byte sizes are validated against the frozen
            // passport, and nothing is retained on the host.
            using namespace frozen;
            struct IdxTensor {
                const char *suffix;
                uint16_t **dst;
                int64_t want_bytes;  // 0 = shape-only validation
            };
            const IdxTensor kIdxTensors[] = {
                {"indexer.wq_b.weight", &lw.d_idx_wq_b,
                 static_cast<int64_t>(kIndexerHeads) * kIndexerHeadDim *
                     kQLora * 2},
                {"indexer.wk.weight", &lw.d_idx_wk,
                 static_cast<int64_t>(kIndexerHeadDim) * kHidden * 2},
                {"indexer.weights_proj.weight", &lw.d_idx_wproj,
                 static_cast<int64_t>(kIndexerHeads) * kHidden * 2},
                {"indexer.k_norm.weight", &lw.d_idx_knorm_w,
                 static_cast<int64_t>(kIndexerHeadDim) * 2},
                {"indexer.k_norm.bias", &lw.d_idx_knorm_b,
                 static_cast<int64_t>(kIndexerHeadDim) * 2},
                {"indexer.index_kpool_compress_ape", &lw.d_idx_ape,
                 static_cast<int64_t>(kIndexerKpool) * kIndexerHeadDim * 2},
                {"indexer.index_kpool_compress_gate", &lw.d_idx_gate,
                 static_cast<int64_t>(kIndexerHeadDim) * kHidden * 2},
            };
            for (const IdxTensor &it : kIdxTensors) {
                const TensorView *t = FindTensorSuffix(*raw, it.suffix);
                if (!t) {
                    error = std::string(l3) + " missing indexer " + it.suffix;
                    pw.Free();
                    return false;
                }
                // A frozen-geometry mismatch here would silently mis-shape
                // every indexer dot product. Under the synthetic gate the
                // fixture writes its own (small) geometry, as everywhere else
                // in this loader.
                if (!SyntheticGeometryAllowed() &&
                    t->bytes != static_cast<std::size_t>(it.want_bytes)) {
                    error = absl::StrCat(l3, " indexer ", it.suffix,
                                         " has ", t->bytes, " bytes, want ",
                                         it.want_bytes);
                    pw.Free();
                    return false;
                }
                if (UploadBf16Raw(ring, t->data, t->bytes,
                                  reinterpret_cast<uint8_t **>(it.dst), stream,
                                  bytes_acc) != cudaSuccess) {
                    error = std::string(l3) + " indexer upload failed: " +
                            it.suffix;
                    pw.Free();
                    return false;
                }
                // The device copy is the only copy: keeping the host
                // payloads here cost ~180 MB of pinned-side RSS across the
                // 12 DSA layers for nothing (the parity tests build their own
                // fixtures).
            }
        }

        if (std::getenv("LSX_FLASH_LOAD_DEBUG"))
            std::fprintf(stderr, "[load] l=%d kda=%d q=%p o=%p A=%p\n", l,
                         (int) lw.is_kda, (void *) lw.kda_q.d_packed,
                         (void *) lw.kda_o.d_packed, (void *) lw.d_A_log);

        // MoE: router + shared expert; Dense MLP on layers 0..2.
        auto nv = LoadDenseNvfp4BatchViews(pinned_dir
                                                     / (l3
                                                        + "_nvfp4.arrow"));
        if (!nv.ok()) {
            error = l3 + "_nvfp4.arrow read failed: "
                  + nv.status().ToString();
            pw.Free();
            return false;
        }
        std::string dummy;
        if (lw.is_moe) {
            if (const TensorView *t =
                    FindTensorSuffix(*raw, "mlp.gate.weight")) {
                cudaError_t e = UploadBf16Raw(
                    ring, t->data, t->bytes, &lw.d_router, stream, bytes_acc);
                if (e != cudaSuccess) {
                    error = l3 + " router upload failed";
                    pw.Free();
                    return false;
                }
                lw.router_bytes = t->bytes;
            } else {
                error = l3 + " missing router gate";
                pw.Free();
                return false;
            }
            if (const TensorView *t = FindTensorSuffix(
                    *raw, "mlp.gate.e_score_correction_bias")) {
                cudaMalloc(reinterpret_cast<void **>(&lw.d_router_bias),
                           t->bytes);
                if (ring.UploadRaw(t->data, lw.d_router_bias, t->bytes,
                                   stream, bytes_acc) != cudaSuccess) {
                    error = l3 + " router bias upload failed";
                    pw.Free();
                    return false;
                }
            } else {
                error = l3 + " missing router bias";
                pw.Free();
                return false;
            }
            if (UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "shared_experts.gate_proj.weight"),
                    lw.shared_w1, stream, ring, bytes_acc, &dummy)
                    != cudaSuccess
                || UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "shared_experts.down_proj.weight"),
                    lw.shared_w2, stream, ring, bytes_acc, &dummy)
                    != cudaSuccess
                || UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "shared_experts.up_proj.weight"),
                    lw.shared_w3, stream, ring, bytes_acc, &dummy)
                    != cudaSuccess
                || !lw.shared_w1.d_packed || !lw.shared_w2.d_packed
                || !lw.shared_w3.d_packed) {
                error = l3 + " shared expert upload failed";
                pw.Free();
                return false;
            }
        }
        if (l < frozen::kMoeFirstLayer) {
            if (UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mlp.gate_proj.weight"),
                                  lw.mlp_gate, stream, ring, bytes_acc,
                                  &dummy)
                    != cudaSuccess
                || UploadNvfp4Weight(FindNvfp4Suffix(*nv, "mlp.up_proj.weight"),
                                     lw.mlp_up, stream, ring, bytes_acc,
                                     &dummy)
                    != cudaSuccess
                || UploadNvfp4Weight(
                    FindNvfp4Suffix(*nv, "mlp.down_proj.weight"), lw.mlp_down,
                    stream, ring, bytes_acc, &dummy)
                    != cudaSuccess
                || !lw.mlp_gate.d_packed) {
                error = l3 + " dense mlp upload failed";
                pw.Free();
                return false;
            }
        }
    }

    pw.resident_bytes = bytes_acc;
    return true;
}

bool FlashExpertLayerRange(int max_layer_seen, int &first, int &last,
                           std::string &why)
{
    why.clear();
    first = frozen::kMoeFirstLayer;
    last = -1;
    if (max_layer_seen < frozen::kMoeFirstLayer) {
        why = absl::StrCat("no MoE layers: highest layer dir is ",
                            max_layer_seen);
        return false;
    }
    if (max_layer_seen > frozen::kMtpLayer) {
        why = absl::StrCat("expert layer beyond the MTP slot: found ", 
                            max_layer_seen, ", the model has layers 0..",
                            frozen::kMtpLayer);
        return false;
    }
    // The MTP layer is stored-not-applied: clamp to the last resident layer.
    last = std::min(max_layer_seen, frozen::kLayers - 1);
    return true;
}

bool LoadGlmFlashExpertStore(const std::string &model_path,
                             lsxcommon::moe::MoEExpertStore &store,
                             std::string &error)
{
    namespace fs = std::filesystem;
    store.Free();
    error.clear();

    const fs::path experts_dir = fs::path(model_path) / "experts";
    if (!fs::exists(experts_dir)) {
        error = "no experts dir under " + experts_dir.string();
        return false;
    }

    int max_layer = -1;
    for (fs::directory_iterator it(experts_dir), end; it != end; ++it) {
        const std::string name = it->path().filename().string();
        if (name.rfind("layer_", 0) == 0) {
            int l = 0;
            if (absl::SimpleAtoi(name.substr(6), &l))
                max_layer = std::max(max_layer, l);
        }
    }
    if (max_layer < 0) {
        error = "no expert layer dirs under " + experts_dir.string();
        return false;
    }
    int first_layer = -1;
    std::string why;
    if (!FlashExpertLayerRange(max_layer, first_layer, max_layer, why)) {
        error = absl::StrCat("expert store: ", why);
        return false;
    }
    (void)first_layer;  // == frozen::kMoeFirstLayer
    // Compact layout: slab index = layer - kMoeFirstLayer (resident MoE layers
    // 3..44 -> store slots 0..41). num_layers counts MoE layers only. The MTP
    // layer is excluded by FlashExpertLayerRange.
    store.num_layers = max_layer + 1 - frozen::kMoeFirstLayer;
    store.records.resize(static_cast<size_t>(store.num_layers));
    store.h_slabs.assign(static_cast<size_t>(store.num_layers), nullptr);
    store.slab_bytes.assign(static_cast<size_t>(store.num_layers), 0);

    for (int l = frozen::kMoeFirstLayer; l <= max_layer; ++l) {
        const int slot = l - frozen::kMoeFirstLayer;
        const fs::path layer_dir =
            experts_dir / absl::StrCat("layer_", absl::Dec(l,
                                                           absl::kZeroPad3));
        int E = store.num_experts;
        if (E <= 0) {
            E = 0;
            for (fs::directory_iterator it(layer_dir), end; it != end; ++it) {
                const std::string name = it->path().filename().string();
                if (name.rfind("exp_", 0) == 0
                    && name.find(".arrow") != std::string::npos) {
                    int e = 0;
                    if (absl::SimpleAtoi(name.substr(4, name.find('.') - 4),
                                         &e))
                        E = std::max(E, e + 1);
                }
            }
            store.num_experts = E;
        }
        if (E <= 0) {
            error = absl::StrCat("no expert files in layer_", l);
            return false;
        }

        struct StagedExpert {
            std::vector<uint8_t> w1, w2, w3, s1, s2, s3;
        };
        std::vector<StagedExpert> staged(static_cast<size_t>(E));
        auto &recs = store.records[static_cast<size_t>(slot)];
        recs.assign(static_cast<size_t>(E), {});
        size_t total = 0;
        for (int e = 0; e < E; ++e) {
            const std::string path =
                (layer_dir
                 / absl::StrCat("exp_", absl::Dec(e, absl::kZeroPad3),
                                ".arrow")).string();
            auto batch_res = ReadRecordBatchFromFile(path);
            if (!batch_res.ok()) {
                error = absl::StrCat("read failed: ", path);
                return false;
            }
            auto pe_res = ExpertSchema::UnpackBatch(*batch_res, kNVFP4BlockSize);
            if (!pe_res.ok()) {
                error = absl::StrCat("unpack failed: ", path);
                return false;
            }
            auto &pe = *pe_res;
            auto &st = staged[static_cast<size_t>(e)];
            st.w1 = std::move(pe.w1_packed);
            st.w2 = std::move(pe.w2_packed);
            st.w3 = std::move(pe.w3_packed);
            st.s1 = std::move(pe.w1_scales);
            st.s2 = std::move(pe.w2_scales);
            st.s3 = std::move(pe.w3_scales);
            auto &r = recs[static_cast<size_t>(e)];
            r.slab_w1_off = total;
            r.w1_bytes = static_cast<int64_t>(st.w1.size());
            total += r.w1_bytes;
            r.slab_w2_off = total;
            r.w2_bytes = static_cast<int64_t>(st.w2.size());
            total += r.w2_bytes;
            r.slab_w3_off = total;
            r.w3_bytes = static_cast<int64_t>(st.w3.size());
            total += r.w3_bytes;
            r.gs1 = pe.w1_global_scale;
            r.gs2 = pe.w2_global_scale;
            r.gs3 = pe.w3_global_scale;
            total = AlignUp(total, 16);
            r.slab_s1_off = total;
            r.s1_bytes = static_cast<int64_t>(st.s1.size());
            total += r.s1_bytes;
            r.slab_s2_off = total;
            r.s2_bytes = static_cast<int64_t>(st.s2.size());
            total += r.s2_bytes;
            r.slab_s3_off = total;
            r.s3_bytes = static_cast<int64_t>(st.s3.size());
            total += r.s3_bytes;
        }
        store.slab_bytes[static_cast<size_t>(slot)] = total;

        uint8_t *slab = nullptr;
        if (cudaMallocHost(reinterpret_cast<void **>(&slab), total)
            != cudaSuccess) {
            error = absl::StrCat("pinned slab alloc failed (layer ", l, ")");
            return false;
        }
        store.h_slabs[static_cast<size_t>(slot)] = slab;
        for (int e = 0; e < E; ++e) {
            const auto &st = staged[static_cast<size_t>(e)];
            const auto &r = recs[static_cast<size_t>(e)];
            if (r.w1_bytes) memcpy(slab + r.slab_w1_off, st.w1.data(),
                                   static_cast<size_t>(r.w1_bytes));
            if (r.w2_bytes) memcpy(slab + r.slab_w2_off, st.w2.data(),
                                   static_cast<size_t>(r.w2_bytes));
            if (r.w3_bytes) memcpy(slab + r.slab_w3_off, st.w3.data(),
                                   static_cast<size_t>(r.w3_bytes));
            if (r.s1_bytes) memcpy(slab + r.slab_s1_off, st.s1.data(),
                                   static_cast<size_t>(r.s1_bytes));
            if (r.s2_bytes) memcpy(slab + r.slab_s2_off, st.s2.data(),
                                   static_cast<size_t>(r.s2_bytes));
            if (r.s3_bytes) memcpy(slab + r.slab_s3_off, st.s3.data(),
                                   static_cast<size_t>(r.s3_bytes));
        }
    }
    return true;
}

} // namespace glm_moe_flash
} // namespace lsxcommon
