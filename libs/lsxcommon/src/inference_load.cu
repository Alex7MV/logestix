#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/schema.h"
#include "inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

namespace {

// BF16 → FP16 conversion kernel (reads from device memory)
__global__ void bf16_to_half_kernel(const uint16_t *src, __half *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__bfloat162float(src[i]));
}

// Upload BF16 host data → device FP16 via device-side kernel.
// Two-step: cudaMemcpyAsync raw BF16 → device, then convert kernel.
cudaError_t upload_bf16_to_half(const uint8_t *src_bytes, int64_t numel,
                                 __half **d_out, cudaStream_t stream)
{
    // Allocate device buffer (BF16 capacity = __half capacity = 2 bytes per element)
    cudaError_t e = cudaMalloc(d_out, static_cast<size_t>(numel) * sizeof(__half));
    if (e != cudaSuccess) return e;
    // Allocate temp device buffer for raw BF16 data
    uint16_t *d_bf16 = nullptr;
    e = cudaMalloc(&d_bf16, static_cast<size_t>(numel) * sizeof(uint16_t));
    if (e != cudaSuccess) { cudaFree(*d_out); *d_out = nullptr; return e; }
    // Copy host BF16 data → device BF16 buffer
    const uint16_t *src_u16 = reinterpret_cast<const uint16_t *>(src_bytes);
    e = cudaMemcpyAsync(d_bf16, src_u16,
                        static_cast<size_t>(numel) * sizeof(uint16_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d_bf16); cudaFree(*d_out); *d_out = nullptr; return e; }
    // Launch conversion kernel
    int block = 256;
    int grid = (static_cast<int>(numel) + block - 1) / block;
    bf16_to_half_kernel<<<grid, block, 0, stream>>>(d_bf16, *d_out, static_cast<int>(numel));
    e = cudaGetLastError();
    if (e != cudaSuccess) { cudaFree(d_bf16); cudaFree(*d_out); *d_out = nullptr; return e; }
    e = cudaFree(d_bf16);
    return e;
}

const TensorRecord *find_tensor_suffix(
    const std::vector<TensorRecord> &tensors,
    const std::string &suffix)
{
    for (const auto &t : tensors) {
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return &t;
    }
    return nullptr;
}

const TensorRecord *find_tensor_substr(
    const std::vector<TensorRecord> &tensors,
    const std::string &substr)
{
    for (const auto &t : tensors) {
        if (t.name.find(substr) != std::string::npos)
            return &t;
    }
    return nullptr;
}

arrow::Result<std::vector<TensorRecord>> load_arrow_file(const std::string &path)
{
    auto mmap_res = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!mmap_res.ok())
        return mmap_res.status();
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap_res->get());
    if (!reader_res.ok())
        return reader_res.status();
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok())
        return batch_res.status();
    return PinnedSchema::UnpackBatch(*batch_res);
}

} // anonymous namespace

ModelTopology load_topology(const std::string &model_path)
{
    ModelTopology topo;
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(ipath);
    if (!f) {
        topo.error_msg = "Cannot open " + ipath.string();
        return topo;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        topo.error_msg = "JSON parse error in " + ipath.string();
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "Expected JSON object in " + ipath.string();
        return topo;
    }

    int64_t nl = 0, epl = 0, te = 0;
    root["num_layers"].get_int64().get(nl);
    root["experts_per_layer"].get_int64().get(epl);
    root["total_experts"].get_int64().get(te);
    topo.num_layers = static_cast<int>(nl);
    topo.experts_per_layer = static_cast<int>(epl);
    topo.total_experts = te;
    return topo;
}

ExpertDb load_expert_index(const std::string &model_path,
                            int num_layers, int experts_per_layer)
{
    ExpertDb db;
    int64_t total = static_cast<int64_t>(num_layers) * experts_per_layer;
    db.entries.resize(static_cast<size_t>(total));
    db.mmaps.reserve(static_cast<size_t>(total));
    db.batches.reserve(static_cast<size_t>(total));
    namespace fs = std::filesystem;

    for (int layer = 0; layer < num_layers; ++layer) {
        for (int exp = 0; exp < experts_per_layer; ++exp) {
            size_t idx = static_cast<size_t>(layer) * experts_per_layer
                       + static_cast<size_t>(exp);
            auto apath = fs::path(model_path)
                / "logestix" / "experts" / ("layer_" + std::to_string(layer))
                / ("exp_" + std::to_string(exp) + ".arrow");

            auto mmap_res = arrow::io::MemoryMappedFile::Open(
                apath.string(), arrow::io::FileMode::READ);
            if (!mmap_res.ok()) {
                db.mmaps.push_back(nullptr);
                db.batches.push_back(nullptr);
                continue;
            }
            auto mmap = *mmap_res;

            auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
            if (!reader_res.ok() || (*reader_res)->num_record_batches() == 0) {
                db.mmaps.push_back(nullptr);
                db.batches.push_back(nullptr);
                continue;
            }
            auto batch_res = (*reader_res)->ReadRecordBatch(0);
            if (!batch_res.ok()) {
                db.mmaps.push_back(nullptr);
                db.batches.push_back(nullptr);
                continue;
            }
            auto batch = *batch_res;

            auto get_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
                auto arr = batch->column(col);
                auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(arr);
                if (!bin) return {nullptr, 0};
                auto view = bin->GetView(0);
                return {reinterpret_cast<const uint8_t *>(view.data()),
                        static_cast<int64_t>(view.size())};
            };

            auto get_scale_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
                auto lst = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(col));
                if (!lst) return {nullptr, 0};
                auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
                if (!u8) return {nullptr, 0};
                auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
                if (!offs) return {nullptr, 0};
                int64_t begin = offs->Value(0);
                int64_t end   = offs->Value(1);
                return {u8->raw_values() + begin, end - begin};
            };

            auto get_gs = [&](int col) -> float {
                auto arr = batch->column(col);
                auto f = std::dynamic_pointer_cast<arrow::FloatArray>(arr);
                return f ? f->Value(0) : 1.0f;
            };

            auto [w1p, w1s] = get_bin(ExpertSchema::kIdxW1Weight);
            auto [w2p, w2s] = get_bin(ExpertSchema::kIdxW2Weight);
            auto [w3p, w3s] = get_bin(ExpertSchema::kIdxW3Weight);
            auto [w1sp, w1ss] = get_scale_bin(ExpertSchema::kIdxW1Scale);
            auto [w2sp, w2ss] = get_scale_bin(ExpertSchema::kIdxW2Scale);
            auto [w3sp, w3ss] = get_scale_bin(ExpertSchema::kIdxW3Scale);
            float w1gs = get_gs(ExpertSchema::kIdxW1GlobalScale);
            float w2gs = get_gs(ExpertSchema::kIdxW2GlobalScale);
            float w3gs = get_gs(ExpertSchema::kIdxW3GlobalScale);

            db.entries[idx] = {w1p, w2p, w3p, w1sp, w2sp, w3sp,
                               w1gs, w2gs, w3gs,
                               w1s, w2s, w3s, w1ss, w2ss, w3ss};
            db.mmaps.push_back(std::move(mmap));
            db.batches.push_back(std::move(batch));
        }
    }
    return db;
}

void LayerWeights::Free()
{
    if (d_ffn_norm)     cudaFree(d_ffn_norm);
    if (d_gate_weight)  cudaFree(d_gate_weight);
    if (d_shared_w1)    cudaFree(d_shared_w1);
    if (d_shared_w2)    cudaFree(d_shared_w2);
    if (d_shared_w3)    cudaFree(d_shared_w3);
    if (d_shared_w1_scales) cudaFree(d_shared_w1_scales);
    if (d_shared_w2_scales) cudaFree(d_shared_w2_scales);
    if (d_shared_w3_scales) cudaFree(d_shared_w3_scales);
}

void PinnedWeights::Free()
{
    if (d_embed)      cudaFree(d_embed);
    if (d_head)       cudaFree(d_head);
    if (d_final_norm) cudaFree(d_final_norm);
    for (auto &l : layers) l.Free();
}

cudaError_t load_pinned_weights(
    const std::string &model_path,
    PinnedWeights &pw,
    cudaStream_t stream)
{
    namespace fs = std::filesystem;
    auto pinned_dir = fs::path(model_path) / "logestix" / "pinned";

    auto global_path = pinned_dir / "global.arrow";
    auto global_tensors_res = load_arrow_file(global_path.string());
    if (!global_tensors_res.ok()) {
        std::fprintf(stderr, "[load_pinned] ERROR loading global.arrow: %s\n",
                     global_tensors_res.status().ToString().c_str());
        return cudaErrorUnknown;
    }
    auto &global_tensors = *global_tensors_res;

    // Debug: print first 10 tensor names
    for (int di = 0; di < std::min(10, static_cast<int>(global_tensors.size())); ++di) {
        const auto &t = global_tensors[di];
        std::fprintf(stderr, "[load_pinned] global tensor[%d] name='%s' shape=[",
                     di, t.name.c_str());
        for (size_t si = 0; si < t.shape.size(); ++si)
            std::fprintf(stderr, "%s%lld", si > 0 ? "," : "", static_cast<long long>(t.shape[si]));
        std::fprintf(stderr, "] data_sz=%lld\n", static_cast<long long>(t.data.size()));
    }

    // Upload via upload_bf16_to_half (two-step: host→device copy, then kernel)
    auto *embed_t = find_tensor_suffix(global_tensors, "embed.weight");
    auto *head_t = find_tensor_suffix(global_tensors, "head.weight");
    auto *norm_t = find_tensor_suffix(global_tensors, "norm.weight");

    if (embed_t && embed_t->shape.size() >= 2) {
        pw.vocab_size = static_cast<int>(embed_t->shape[0]);
        int64_t numel = embed_t->shape[0] * embed_t->shape[1];
        cudaError_t e = upload_bf16_to_half(embed_t->data.data(), numel,
                                             &pw.d_embed, stream);
        if (e != cudaSuccess) { std::fprintf(stderr, "embed upload failed\n"); return e; }
    }
    if (norm_t && !norm_t->shape.empty()) {
        int64_t numel = norm_t->shape[0];
        cudaError_t e = upload_bf16_to_half(norm_t->data.data(), numel,
                                             &pw.d_final_norm, stream);
        if (e != cudaSuccess) { std::fprintf(stderr, "norm upload failed\n"); return e; }
    }
    if (head_t && head_t->shape.size() >= 2) {
        if (pw.vocab_size == 0)
            pw.vocab_size = static_cast<int>(head_t->shape[0]);
        int64_t numel = head_t->shape[0] * head_t->shape[1];
        cudaError_t e = upload_bf16_to_half(head_t->data.data(), numel,
                                             &pw.d_head, stream);
        if (e != cudaSuccess) { std::fprintf(stderr, "head upload failed\n"); return e; }
    }

    pw.layers.resize(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
        char fn[64];
        std::snprintf(fn, sizeof(fn), "layer_%d.arrow", l);
        auto layer_path = pinned_dir / fn;

        if (!fs::exists(layer_path)) continue;

        auto tensors_res = load_arrow_file(layer_path.string());
        if (!tensors_res.ok()) {
            std::fprintf(stderr, "[load_pinned] ERROR loading %s: %s\n",
                         fn, tensors_res.status().ToString().c_str());
            continue;
        }
        auto &lt = *tensors_res;
        auto &lw = pw.layers[l];
        for (int di = 0; di < std::min(5, static_cast<int>(lt.size())); ++di) {
            const auto &t = lt[di];
            std::fprintf(stderr, "[load_pinned]   layer_%03d tensor[%d] name='%s' shape=[",
                         l, di, t.name.c_str());
            for (size_t si = 0; si < t.shape.size(); ++si)
                std::fprintf(stderr, "%s%lld", si > 0 ? "," : "", static_cast<long long>(t.shape[si]));
            std::fprintf(stderr, "]\n");
        }

        auto *ffn_norm = find_tensor_substr(lt, "ffn_norm.weight");
        if (ffn_norm && !ffn_norm->shape.empty())
            upload_bf16_to_half(ffn_norm->data.data(), ffn_norm->shape[0],
                                &lw.d_ffn_norm, stream);

        auto *gate = find_tensor_substr(lt, "ffn.gate.weight");
        if (gate && gate->shape.size() >= 2) {
            int64_t numel = gate->shape[0] * gate->shape[1];
            // Use upload_bf16_to_half which does proper host→device copy first
            upload_bf16_to_half(gate->data.data(), numel,
                                &lw.d_gate_weight, stream);
        }

        auto load_shared = [&](const std::string &wname,
                               uint8_t *&d_buf, uint8_t *&d_scales,
                               float &gs, int64_t &nbytes)
        {
            auto *wt = find_tensor_substr(lt,
                "ffn.shared_experts." + wname + ".weight");
            if (!wt) return;
            nbytes = static_cast<int64_t>(wt->data.size());
            cudaMalloc(&d_buf, static_cast<size_t>(nbytes));
            cudaMemcpyAsync(d_buf, wt->data.data(),
                           static_cast<size_t>(nbytes),
                           cudaMemcpyHostToDevice, stream);
            d_scales = nullptr;
            gs = 1.0f;
        };

        load_shared("w1", lw.d_shared_w1, lw.d_shared_w1_scales,
                    lw.shared_w1_global_scale, lw.shared_w1_bytes);
        load_shared("w2", lw.d_shared_w2, lw.d_shared_w2_scales,
                    lw.shared_w2_global_scale, lw.shared_w2_bytes);
        load_shared("w3", lw.d_shared_w3, lw.d_shared_w3_scales,
                    lw.shared_w3_global_scale, lw.shared_w3_bytes);
    }

    return cudaSuccess;
}

} // namespace gpu
} // namespace lsxcommon
