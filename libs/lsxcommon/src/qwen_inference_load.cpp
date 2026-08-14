#include "lsxcommon/qwen_inference_load.h"
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
#include <fstream>
#include <iterator>

namespace lsxcommon::qwen {

// QwenNvfp4Weight::Free is defined inline in the header.

void QwenLayerWeights::Free() {
    cudaFree(d_input_norm); d_input_norm = nullptr;
    cudaFree(d_post_norm);  d_post_norm = nullptr;
    qkv.Free(); z.Free(); b.Free(); a.Free(); out.Free();
    cudaFree(d_conv1d);    d_conv1d = nullptr;
    cudaFree(d_A_log);     d_A_log = nullptr;
    cudaFree(d_dt_bias);   d_dt_bias = nullptr;
    cudaFree(d_attn_norm); d_attn_norm = nullptr;
    q.Free(); k.Free(); v.Free(); o.Free();
    cudaFree(d_q_norm); d_q_norm = nullptr;
    cudaFree(d_k_norm); d_k_norm = nullptr;
    mlp_gate.Free(); mlp_up.Free(); mlp_down.Free();
    loaded = false;
}

void QwenPinnedWeights::Free() {
    embed.Free();
    head.Free();
    cudaFree(d_final_norm); d_final_norm = nullptr;
    for (auto &l : layers) l.Free();
    layers.clear();
}

std::vector<uint8_t> ReadFileBytes(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    return buf;
}

namespace {

// Host-side BF16 -> FP16 upload (mirror of the GLM path).
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
    if (e != cudaSuccess) { cudaFree(*d_out); *d_out = nullptr; }
    return e;
}

cudaError_t UploadRawPtr(const std::vector<uint8_t> &data, void **d_out,
                         cudaStream_t stream)
{
    if (data.empty()) return cudaSuccess;
    cudaError_t e = cudaMalloc(d_out, data.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(*d_out, data.data(), data.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(*d_out); *d_out = nullptr; }
    return e;
}

// Uploads a 1D BF16 norm tensor.
void Upload1D(const uint8_t *data, int64_t n, __half **dst, cudaStream_t stream)
{
    if (!data || n <= 0) return;
    if (UploadBf16ToHalf(data, n, dst, stream) != cudaSuccess)
        *dst = nullptr;
}

// Uploads a raw F32 tensor (A_log / dt_bias).
void UploadF32(const uint8_t *data, int64_t n, float **dst, cudaStream_t stream)
{
    if (!data || n <= 0) return;
    cudaError_t e = cudaMalloc(dst, static_cast<size_t>(n) * sizeof(float));
    if (e != cudaSuccess) return;
    e = cudaMemcpyAsync(*dst, data, static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(*dst); *dst = nullptr; }
}

// Matches a tensor record whose name ENDS with `suffix`.
const TensorRecord *FindRaw(const std::vector<TensorRecord> &tensors,
                            const std::string &suffix)
{
    for (const auto &t : tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return &t;
    }
    return nullptr;
}

const DenseNvfp4Record *FindNvfp4(const std::vector<DenseNvfp4Record> &recs,
                                  const std::string &suffix)
{
    for (const auto &r : recs) {
        if (r.name.size() >= suffix.size() &&
            r.name.compare(r.name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return &r;
    }
    return nullptr;
}

void UploadNvfp4Weight(const DenseNvfp4Record *r, QwenNvfp4Weight &w,
                       cudaStream_t stream)
{
    if (!r || r->packed.empty()) return;
    w.rows = r->shape.size() >= 2 ? static_cast<int>(r->shape[0]) : 0;
    w.cols = r->shape.size() >= 2 ? static_cast<int>(r->shape[1]) : 0;
    w.global_scale = r->global_scale;
    w.packed_bytes = static_cast<int64_t>(r->packed.size());
    w.scale_bytes = static_cast<int64_t>(r->scales.size());
    void *d1 = nullptr;
    cudaError_t e1 = cudaMalloc(&d1, r->packed.size());
    if (e1 != cudaSuccess) { w.Free(); return; }
    e1 = cudaMemcpyAsync(d1, r->packed.data(),
                         r->packed.size(), cudaMemcpyHostToDevice, stream);
    if (e1 != cudaSuccess) { cudaFree(d1); w.Free(); return; }
    void *d2 = nullptr;
    cudaError_t e2 = cudaMalloc(&d2, r->scales.size());
    if (e2 != cudaSuccess) { cudaFree(d1); w.Free(); return; }
    e2 = cudaMemcpyAsync(d2, r->scales.data(),
                         r->scales.size(), cudaMemcpyHostToDevice, stream);
    if (e2 != cudaSuccess) { cudaFree(d1); cudaFree(d2); w.Free(); return; }
    w.d_packed = static_cast<const uint8_t*>(d1);
    w.d_scales = static_cast<const uint8_t*>(d2);
}

arrow::Result<std::vector<TensorRecord>> LoadRawFile(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok()) return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    return PinnedSchema::UnpackBatch(*batch_res);
}

arrow::Result<std::vector<DenseNvfp4Record>> LoadNvfp4File(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok()) return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    return DenseNvfp4Schema::UnpackBatch(*batch_res);
}

cudaError_t LoadLayer(const std::vector<TensorRecord> &raw,
                      const std::vector<DenseNvfp4Record> &nv,
                      QwenLayerWeights &lw, cudaStream_t stream,
                      bool is_linear)
{
    lw.is_linear = is_linear;

    auto *in_norm = FindRaw(raw, "input_layernorm.weight");
    if (in_norm && !in_norm->shape.empty())
        lw.norm_size = static_cast<int>(in_norm->shape[0]);
    Upload1D(in_norm ? in_norm->data.data() : nullptr, lw.norm_size,
             &lw.d_input_norm, stream);
    auto *post_norm = FindRaw(raw, "post_attention_layernorm.weight");
    Upload1D(post_norm ? post_norm->data.data() : nullptr, lw.norm_size,
             &lw.d_post_norm, stream);

    if (is_linear) {
        UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_qkv.weight"), lw.qkv, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_z.weight"), lw.z, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_b.weight"), lw.b, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.in_proj_a.weight"), lw.a, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "linear_attn.out_proj.weight"), lw.out, stream);

        auto *conv = FindRaw(raw, "linear_attn.conv1d.weight");
        if (conv && conv->shape.size() >= 2) {
            lw.conv_dim = static_cast<int>(conv->shape[0]);
            lw.conv_kernel = static_cast<int>(conv->shape[1]);
            int64_t numel = static_cast<int64_t>(lw.conv_dim) * lw.conv_kernel;
            UploadBf16ToHalf(conv->data.data(), numel, &lw.d_conv1d, stream);
        }
        auto *al = FindRaw(raw, "linear_attn.A_log");
        if (al && !al->shape.empty())
            UploadF32(al->data.data(), al->shape[0], &lw.d_A_log, stream);
        auto *dt = FindRaw(raw, "linear_attn.dt_bias");
        if (dt && !dt->shape.empty())
            UploadF32(dt->data.data(), dt->shape[0], &lw.d_dt_bias, stream);
        auto *an = FindRaw(raw, "linear_attn.norm.weight");
        if (an && !an->shape.empty())
            lw.attn_norm_size = static_cast<int>(an->shape[0]);
        Upload1D(an ? an->data.data() : nullptr, lw.attn_norm_size,
                 &lw.d_attn_norm, stream);
    } else {
        UploadNvfp4Weight(FindNvfp4(nv, "self_attn.q_proj.weight"), lw.q, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "self_attn.k_proj.weight"), lw.k, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "self_attn.v_proj.weight"), lw.v, stream);
        UploadNvfp4Weight(FindNvfp4(nv, "self_attn.o_proj.weight"), lw.o, stream);
        auto *qn = FindRaw(raw, "self_attn.q_norm.weight");
        if (qn && !qn->shape.empty())
            lw.q_norm_size = static_cast<int>(qn->shape[0]);
        Upload1D(qn ? qn->data.data() : nullptr, lw.q_norm_size, &lw.d_q_norm, stream);
        auto *kn = FindRaw(raw, "self_attn.k_norm.weight");
        if (kn && !kn->shape.empty())
            lw.k_norm_size = static_cast<int>(kn->shape[0]);
        Upload1D(kn ? kn->data.data() : nullptr, lw.k_norm_size, &lw.d_k_norm, stream);
    }

    // MLP.
    UploadNvfp4Weight(FindNvfp4(nv, "mlp.gate_proj.weight"), lw.mlp_gate, stream);
    UploadNvfp4Weight(FindNvfp4(nv, "mlp.up_proj.weight"),   lw.mlp_up,   stream);
    UploadNvfp4Weight(FindNvfp4(nv, "mlp.down_proj.weight"), lw.mlp_down, stream);

    lw.loaded = true;
    return cudaSuccess;
}

} // namespace

bool LoadQwenPinnedWeights(const std::string &output_root,
                           QwenPinnedWeights &pw, cudaStream_t stream)
{
    namespace fs = std::filesystem;

    auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir))
        pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[qwen_load] no pinned dir under " + output_root);
        return false;
    }

    auto idx = fs::path(output_root) / "logestix" / "model.index.json";
    if (!fs::exists(idx))
        idx = fs::path(output_root) / "model.index.json";
    if (fs::exists(idx))
        pw.top = load_qwen_topology(idx.string());

    // Global raw (final norm).
    if (fs::exists(pinned_dir / "global.arrow")) {
        auto raw_res = LoadRawFile((pinned_dir / "global.arrow").string());
        if (raw_res.ok()) {
            auto *fn = FindRaw(*raw_res, "norm.weight");
            if (fn && !fn->shape.empty()) {
                pw.final_norm_size = static_cast<int>(fn->shape[0]);
                Upload1D(fn->data.data(), pw.final_norm_size, &pw.d_final_norm, stream);
            }
        }
    }
    // Global NVFP4 (embed + head).
    if (fs::exists(pinned_dir / "global_nvfp4.arrow")) {
        auto nv_res = LoadNvfp4File((pinned_dir / "global_nvfp4.arrow").string());
        if (nv_res.ok()) {
            UploadNvfp4Weight(FindNvfp4(*nv_res, "embed_tokens.weight"), pw.embed, stream);
            UploadNvfp4Weight(FindNvfp4(*nv_res, "lm_head.weight"), pw.head, stream);
        }
    }

    // Per-layer.
    int num_layers = pw.top.num_layers;
    if (num_layers <= 0) {
        // Infer from files present.
        for (auto &entry : fs::directory_iterator(pinned_dir)) {
            auto fn = entry.path().filename().string();
            if (fn.rfind("layer_", 0) == 0 && fn.find("_nvfp4") == std::string::npos) {
                int l = std::atoi(fn.c_str() + 6);
                num_layers = std::max(num_layers, l + 1);
            }
        }
    }

    pw.layers.resize(static_cast<size_t>(num_layers));
    for (int l = 0; l < num_layers; ++l) {
        std::string raw_name = "layer_" + std::to_string(l) + ".arrow";
        std::string nv_name  = "layer_" + std::to_string(l) + "_nvfp4.arrow";
        if (!fs::exists(pinned_dir / raw_name)) continue;

        auto raw = LoadRawFile((pinned_dir / raw_name).string());
        if (!raw.ok()) {
            log::error("[qwen_load] " + raw_name + " read failed");
            continue;
        }
        std::vector<DenseNvfp4Record> nv;
        if (fs::exists(pinned_dir / nv_name)) {
            auto nv_res = LoadNvfp4File((pinned_dir / nv_name).string());
            if (nv_res.ok()) nv = std::move(*nv_res);
        }

        bool is_linear = true;
        if (l < static_cast<int>(pw.top.layer_types.size()))
            is_linear = pw.top.layer_types[static_cast<size_t>(l)] == "linear_attention";

        LoadLayer(*raw, nv, pw.layers[static_cast<size_t>(l)], stream, is_linear);
    }

    return true;
}

} // namespace lsxcommon::qwen