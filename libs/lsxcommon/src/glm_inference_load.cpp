#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/glm_inference.h"
#include "lsxcommon/glm_convert.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/log.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace lsxcommon {
namespace glm {

namespace {

// Host-side BF16 → FP16 conversion + upload. Mirrors the DeepSeek loader
// (inference_load.cu): converts on the host to avoid device temp buffers for
// large tensors, then one async H2D copy. Manual bit path (no __bfloat162float
// host/device ambiguity under MSVC):
//   bf16 = top 16 bits of fp32 → reconstruct fp32 → round to fp16.
cudaError_t UploadBf16ToHalf(const uint8_t *src_bytes, int64_t numel,
                             __half **d_out, cudaStream_t stream)
{
    std::vector<__half> h_half(static_cast<size_t>(numel));
    const uint16_t *src_u16 = reinterpret_cast<const uint16_t *>(src_bytes);
    for (int64_t i = 0; i < numel; ++i) {
        uint32_t f32_bits = static_cast<uint32_t>(src_u16[i]) << 16;
        float f32;
        std::memcpy(&f32, &f32_bits, sizeof(f32));
        h_half[static_cast<size_t>(i)] = __float2half(f32);
    }
    cudaError_t e = cudaMalloc(d_out, static_cast<size_t>(numel) * sizeof(__half));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(*d_out, h_half.data(),
                        static_cast<size_t>(numel) * sizeof(__half),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaFree(*d_out);
        *d_out = nullptr;
        return e;
    }
    return cudaSuccess;
}

cudaError_t UploadRaw(const std::vector<uint8_t> &data, float **d_out,
                      cudaStream_t stream)
{
    if (data.empty()) return cudaSuccess;
    float *d = nullptr;
    cudaError_t e = cudaMalloc(&d, data.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d, data.data(), data.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaFree(d);
        return e;
    }
    *d_out = d;
    return cudaSuccess;
}

// Uploads one NVFP4-packed expert tensor (packed E2M1 nibbles + per-block E4M3
// scales) into a GlmExpertDeviceBlock slot. On any failure the slot stays
// inactive (null payload pointers) — the deterministic zero-contribution skip.
cudaError_t UploadExpertTensor(const std::vector<uint8_t> &packed,
                               const std::vector<uint8_t> &scales,
                               float global_scale,
                               const uint8_t **d_packed,
                               const uint8_t **d_scales,
                               float *out_global_scale,
                               int64_t *out_packed_bytes,
                               int64_t *out_scale_bytes,
                               cudaStream_t stream)
{
    void *d1 = nullptr;
    void *d2 = nullptr;
    cudaError_t e = cudaMalloc(&d1, packed.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d1, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d1); return e; }
    e = cudaMalloc(&d2, scales.size());
    if (e != cudaSuccess) { cudaFree(d1); return e; }
    e = cudaMemcpyAsync(d2, scales.data(), scales.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d1); cudaFree(d2); return e; }
    *d_packed         = static_cast<const uint8_t *>(d1);
    *d_scales         = static_cast<const uint8_t *>(d2);
    *out_global_scale = global_scale;
    *out_packed_bytes = static_cast<int64_t>(packed.size());
    *out_scale_bytes  = static_cast<int64_t>(scales.size());
    return cudaSuccess;
}

// Frees the six payload pointers of one NVFP4 expert descriptor.
void FreeExpertBlock(GlmExpertDeviceBlock &b)
{
    cudaFree(const_cast<uint8_t *>(b.d_w1_packed));
    cudaFree(const_cast<uint8_t *>(b.d_w1_scales));
    cudaFree(const_cast<uint8_t *>(b.d_w2_packed));
    cudaFree(const_cast<uint8_t *>(b.d_w2_scales));
    cudaFree(const_cast<uint8_t *>(b.d_w3_packed));
    cudaFree(const_cast<uint8_t *>(b.d_w3_scales));
    b = GlmExpertDeviceBlock{};
}

const TensorRecord *FindTensor(const std::vector<TensorRecord> &tensors,
                               const std::string &suffix)
{
    for (const auto &t : tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(),
                           suffix.size(), suffix) == 0)
            return &t;
    }
    return nullptr;
}

arrow::Result<std::vector<TensorRecord>> LoadArrowFile(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok()) return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    return PinnedSchema::UnpackBatch(*batch_res);
}

void Upload1D(const TensorRecord *t, __half **dst, cudaStream_t stream)
{
    if (!t || t->shape.empty()) return;
    int64_t numel = t->shape[0];
    if (numel <= 0) return;
    if (UploadBf16ToHalf(t->data.data(), numel, dst, stream) != cudaSuccess)
        *dst = nullptr;
}

// Uploads a 2D BF16 weight, storing its shape (when rows/cols are non-null).
// The base is the full suffix within the layer, e.g. "self_attn.q_a_proj.weight".
void Upload2D(const std::vector<TensorRecord> &tensors,
              const std::string &suffix, __half **dst,
              int64_t *rows, int64_t *cols, cudaStream_t stream)
{
    auto *t = FindTensor(tensors, suffix);
    if (!t || t->shape.size() < 2) return;
    int64_t r = t->shape[0];
    int64_t c = t->shape[1];
    if (rows) *rows = r;
    if (cols) *cols = c;
    int64_t numel = r * c;
    if (numel <= 0) return;
    if (UploadBf16ToHalf(t->data.data(), numel, dst, stream) != cudaSuccess)
        *dst = nullptr;
}

cudaError_t LoadGlobal(const std::filesystem::path &pinned_dir,
                       GlmPinnedWeights &pw, cudaStream_t stream)
{
    auto tensors_res = LoadArrowFile((pinned_dir / "global.arrow").string());
    if (!tensors_res.ok()) {
        log::error("[glm_load] ERROR loading global.arrow: "
                   + tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    auto &tensors = *tensors_res;

    auto *embed = FindTensor(tensors, "embed_tokens.weight");
    if (embed && embed->shape.size() >= 2) {
        pw.vocab_size = static_cast<int>(embed->shape[0]);
        pw.hidden = static_cast<int>(embed->shape[1]);
        if (UploadBf16ToHalf(embed->data.data(),
                             pw.vocab_size * pw.hidden, &pw.d_embed,
                             stream) != cudaSuccess)
            log::error("[glm_load] embed upload failed");
    }
    auto *head = FindTensor(tensors, "lm_head.weight");
    if (head && head->shape.size() >= 2 && head != embed) {
        int64_t numel = head->shape[0] * head->shape[1];
        if (UploadBf16ToHalf(head->data.data(), numel, &pw.d_head,
                             stream) != cudaSuccess)
            log::error("[glm_load] head upload failed");
    }
    auto *norm = FindTensor(tensors, "model.norm.weight");
    if (norm && !norm->shape.empty()) {
        if (UploadBf16ToHalf(norm->data.data(), norm->shape[0],
                             &pw.d_final_norm, stream) != cudaSuccess)
            log::error("[glm_load] model.norm upload failed");
    }
    return cudaSuccess;
}

void LoadSingleLayer(const std::vector<TensorRecord> &tensors,
                     GlmLayerWeights &lw, cudaStream_t stream)
{
    // Layer norms.
    Upload1D(FindTensor(tensors, "input_layernorm.weight"),
             &lw.d_input_norm, stream);
    Upload1D(FindTensor(tensors, "post_attention_layernorm.weight"),
             &lw.d_post_attn_norm, stream);

    // MLA attention.
    Upload2D(tensors, "self_attn.q_a_proj.weight", &lw.d_q_a_proj,
             &lw.q_a_rows, &lw.q_a_cols, stream);
    Upload1D(FindTensor(tensors, "self_attn.q_a_layernorm.weight"),
             &lw.d_q_a_norm, stream);
    Upload2D(tensors, "self_attn.q_b_proj.weight", &lw.d_q_b_proj,
             &lw.q_b_rows, &lw.q_b_cols, stream);
    Upload2D(tensors, "self_attn.kv_a_proj_with_mqa.weight", &lw.d_kv_a_proj,
             &lw.kv_a_rows, &lw.kv_a_cols, stream);
    Upload1D(FindTensor(tensors, "self_attn.kv_a_layernorm.weight"),
             &lw.d_kv_a_norm, stream);
    Upload2D(tensors, "self_attn.kv_b_proj.weight", &lw.d_kv_b_proj,
             &lw.kv_b_rows, &lw.kv_b_cols, stream);
    Upload2D(tensors, "self_attn.o_proj.weight", &lw.d_o_proj,
             &lw.o_rows, &lw.o_cols, stream);

    // Router (MoE layers): gate weight BF16 + e_score_correction_bias F32.
    auto *router = FindTensor(tensors, "mlp.gate.weight");
    if (router && router->shape.size() >= 2) {
        int64_t numel = router->shape[0] * router->shape[1];
        if (UploadBf16ToHalf(router->data.data(), numel,
                             &lw.d_router_gate, stream) == cudaSuccess)
            log::debug("[glm_load] router gate uploaded");
    }
    auto *gate_bias = FindTensor(tensors, "mlp.gate.e_score_correction_bias");
    if (gate_bias && gate_bias->shape.size() == 1) {
        int n = static_cast<int>(gate_bias->shape[0]);
        if (cudaMallocHost(&lw.h_gate_bias, static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
            std::memcpy(lw.h_gate_bias, gate_bias->data.data(),
                        static_cast<size_t>(n) * sizeof(float));
            lw.gate_bias_n = n;
        }
    }

    // Dense MLP (layers 0-2).
    Upload2D(tensors, "mlp.gate_proj.weight", &lw.d_mlp_gate,
             nullptr, nullptr, stream);
    Upload2D(tensors, "mlp.down_proj.weight", &lw.d_mlp_down,
             nullptr, nullptr, stream);
    Upload2D(tensors, "mlp.up_proj.weight", &lw.d_mlp_up,
             nullptr, nullptr, stream);

    // Indexer (indexer layers).
    Upload2D(tensors, "self_attn.indexer.wq_b.weight", &lw.d_idx_wq_b,
             nullptr, nullptr, stream);
    Upload2D(tensors, "self_attn.indexer.wk.weight", &lw.d_idx_wk,
             nullptr, nullptr, stream);
    Upload2D(tensors, "self_attn.indexer.weights_proj.weight",
             &lw.d_idx_weights_proj, nullptr, nullptr, stream);
    Upload1D(FindTensor(tensors, "self_attn.indexer.k_norm.weight"),
             &lw.d_idx_k_norm, stream);
    auto *k_bias = FindTensor(tensors, "self_attn.indexer.k_norm.bias");
    if (k_bias && !k_bias->data.empty())
        UploadRaw(k_bias->data, &lw.d_idx_k_norm_bias, stream);

    // MTP block (layer 78).
    Upload2D(tensors, "eh_proj.weight", &lw.d_eh_proj,
             &lw.q_a_rows, &lw.q_a_cols, stream);
    Upload1D(FindTensor(tensors, "enorm.weight"), &lw.d_enorm, stream);
    Upload1D(FindTensor(tensors, "hnorm.weight"), &lw.d_hnorm, stream);
    Upload1D(FindTensor(tensors, "shared_head.norm.weight"),
             &lw.d_shared_head_norm, stream);
}

} // anonymous namespace

void GlmLayerWeights::Free()
{
    if (d_input_norm)          cudaFree(d_input_norm);
    if (d_post_attn_norm)      cudaFree(d_post_attn_norm);
    if (d_q_a_proj)            cudaFree(d_q_a_proj);
    if (d_q_a_norm)            cudaFree(d_q_a_norm);
    if (d_q_b_proj)            cudaFree(d_q_b_proj);
    if (d_kv_a_proj)           cudaFree(d_kv_a_proj);
    if (d_kv_a_norm)           cudaFree(d_kv_a_norm);
    if (d_kv_b_proj)           cudaFree(d_kv_b_proj);
    if (d_o_proj)              cudaFree(d_o_proj);
    if (d_router_gate)         cudaFree(d_router_gate);
    if (h_gate_bias)           cudaFreeHost(h_gate_bias);
    if (d_mlp_gate)            cudaFree(d_mlp_gate);
    if (d_mlp_down)            cudaFree(d_mlp_down);
    if (d_mlp_up)              cudaFree(d_mlp_up);
    if (d_idx_wq_b)            cudaFree(d_idx_wq_b);
    if (d_idx_wk)              cudaFree(d_idx_wk);
    if (d_idx_weights_proj)    cudaFree(d_idx_weights_proj);
    if (d_idx_k_norm)          cudaFree(d_idx_k_norm);
    if (d_idx_k_norm_bias)     cudaFree(d_idx_k_norm_bias);
    if (d_eh_proj)             cudaFree(d_eh_proj);
    if (d_enorm)               cudaFree(d_enorm);
    if (d_hnorm)               cudaFree(d_hnorm);
    if (d_shared_head_norm)    cudaFree(d_shared_head_norm);

    d_input_norm = d_post_attn_norm = d_q_a_proj = d_q_a_norm = nullptr;
    d_q_b_proj = d_kv_a_proj = d_kv_a_norm = d_kv_b_proj = d_o_proj = nullptr;
    d_router_gate = nullptr; h_gate_bias = nullptr; gate_bias_n = 0;
    d_mlp_gate = d_mlp_down = d_mlp_up = nullptr;
    d_idx_wq_b = d_idx_wk = d_idx_weights_proj = nullptr;
    d_idx_k_norm = nullptr; d_idx_k_norm_bias = nullptr;
    d_eh_proj = d_enorm = d_hnorm = d_shared_head_norm = nullptr;
}

void GlmPinnedWeights::Free()
{
    if (d_embed)      cudaFree(d_embed);
    if (d_head)       cudaFree(d_head);
    if (d_final_norm) cudaFree(d_final_norm);
    d_embed = d_head = d_final_norm = nullptr;
    vocab_size = 0;
    hidden = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
}

cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream,
                                 const std::vector<int> *layer_filter)
{
    namespace fs = std::filesystem;

    // Accept both the logestix root itself and its container directory:
    //   <model>/logestix/pinned/...   (model dir, like the DeepSeek loader)
    //   <logestix>/pinned/...         (converter output_root)
    fs::path pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    fs::path index_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(pinned_dir) && fs::exists(fs::path(model_path) / "pinned")) {
        pinned_dir = fs::path(model_path) / "pinned";
        index_path = fs::path(model_path) / "model.index.json";
    }

    // Topology: number of layers sizes the layers vector (heterogeneous grid).
    auto topo = load_topology(index_path.string());
    if (!topo.error_msg.empty()) {
        log::error("[glm_load] topology error: " + topo.error_msg);
        return cudaErrorInvalidValue;
    }
    pw.layers.resize(static_cast<size_t>(topo.num_layers));

    auto err = LoadGlobal(pinned_dir, pw, stream);
    if (err != cudaSuccess) return err;

    for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
        if (layer_filter &&
            std::find(layer_filter->begin(), layer_filter->end(), l) ==
                layer_filter->end())
            continue;
        auto fn = "layer_" + std::to_string(l) + ".arrow";
        auto layer_path = pinned_dir / fn;
        if (!fs::exists(layer_path)) continue;
        auto tensors_res = LoadArrowFile(layer_path.string());
        if (!tensors_res.ok()) {
            log::error("[glm_load] ERROR loading " + fn + ": "
                       + tensors_res.status().ToString());
            continue;
        }
        LoadSingleLayer(*tensors_res, pw.layers[static_cast<size_t>(l)], stream);
    }

    return cudaSuccess;
}

cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream)
{
    return LoadGlmPinnedWeights(model_path, pw, stream, nullptr);
}

void GlmExpertTables::Free()
{
    for (auto &tab : tables)
        for (auto &b : tab) FreeExpertBlock(b);
    tables.clear();
}

cudaError_t LoadGlmExpertTables(const std::string &model_path,
                                GlmExpertTables &experts,
                                cudaStream_t stream,
                                const std::vector<int> *layer_filter,
                                int max_experts)
{
    namespace fs = std::filesystem;

    // Accept both the logestix root and its container directory, mirroring
    // the pinned loader:
    //   <model>/logestix/experts/layer_<l>/exp_<e>.arrow
    //   <model>/experts/layer_<l>/exp_<e>.arrow
    fs::path exp_dir = fs::path(model_path) / "logestix" / "experts";
    if (!fs::exists(exp_dir) && fs::exists(fs::path(model_path) / "experts"))
        exp_dir = fs::path(model_path) / "experts";

    fs::path index_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(index_path) &&
        fs::exists(fs::path(model_path) / "model.index.json"))
        index_path = fs::path(model_path) / "model.index.json";

    auto topo = load_topology(index_path.string());
    if (!topo.error_msg.empty()) {
        log::error("[glm_load] expert topology error: " + topo.error_msg);
        return cudaErrorInvalidValue;
    }

    experts.tables.resize(static_cast<size_t>(topo.num_layers));
    for (auto &tab : experts.tables)
        tab.resize(static_cast<size_t>(kGlmNumRouted));  // all-inactive default

    for (int l = 0; l < topo.num_layers; ++l) {
        if (layer_filter &&
            std::find(layer_filter->begin(), layer_filter->end(), l) ==
                layer_filter->end())
            continue;
        auto layer_dir = exp_dir / ("layer_" + std::to_string(l));
        if (!fs::is_directory(layer_dir)) continue;
        auto &tab = experts.tables[static_cast<size_t>(l)];
        for (int e = 0; e < max_experts; ++e) {
            auto p = layer_dir / ("exp_" + std::to_string(e) + ".arrow");
            if (!fs::exists(p)) continue;
            auto batch_res = ReadRecordBatchFromFile(p);
            if (!batch_res.ok()) continue;
            auto pe_res = ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) continue;
            const PackedExpert &pe = *pe_res;
            if (pe.w1_packed.empty() || pe.w2_packed.empty() ||
                pe.w3_packed.empty() || pe.w1_scales.empty() ||
                pe.w2_scales.empty() || pe.w3_scales.empty())
                continue;

            GlmExpertDeviceBlock b;
            cudaError_t err = UploadExpertTensor(
                pe.w1_packed, pe.w1_scales, pe.w1_global_scale,
                &b.d_w1_packed, &b.d_w1_scales, &b.w1_global_scale,
                &b.w1_packed_bytes, &b.w1_scale_bytes, stream);
            if (err == cudaSuccess)
                err = UploadExpertTensor(
                    pe.w2_packed, pe.w2_scales, pe.w2_global_scale,
                    &b.d_w2_packed, &b.d_w2_scales, &b.w2_global_scale,
                    &b.w2_packed_bytes, &b.w2_scale_bytes, stream);
            if (err == cudaSuccess)
                err = UploadExpertTensor(
                    pe.w3_packed, pe.w3_scales, pe.w3_global_scale,
                    &b.d_w3_packed, &b.d_w3_scales, &b.w3_global_scale,
                    &b.w3_packed_bytes, &b.w3_scale_bytes, stream);
            if (err != cudaSuccess) {
                log::error("[glm_load] expert upload failed: layer " +
                           std::to_string(l) + " exp " + std::to_string(e));
                FreeExpertBlock(b);  // slot stays inactive
                continue;
            }
            tab[static_cast<size_t>(e)] = b;
        }
    }

    return cudaSuccess;
}

} // namespace glm
} // namespace lsxcommon