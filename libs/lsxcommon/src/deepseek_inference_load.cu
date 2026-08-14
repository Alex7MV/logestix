#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/path_registry.h"
#include "deepseek_inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include "lsxcommon/log.h"
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

namespace {

// BF16 → FP16 conversion kernel (reads from device memory).
// Manual bit path (avoid __bfloat162float host/device ambiguity under MSVC):
//   bf16 = top 16 bits of fp32 → reconstruct fp32 → round to fp16.
__global__ void bf16_to_half_kernel(const uint16_t *src, __half *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t f32 = static_cast<uint32_t>(src[i]) << 16;
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    dst[i] = __float2half(v);
}

// Upload BF16 host data → device FP16 via device-side kernel.
// Convert BF16 → FP16 on host, then upload. Avoids temporary device buffers
// that can cause memory corruption with large tensors (1GB+ embed/head).
cudaError_t upload_bf16_to_half(const uint8_t *src_bytes, int64_t numel,
                                 __half **d_out, cudaStream_t stream)
{
    // Host-side BF16 → FP16 conversion
    std::vector<__half> h_half(static_cast<size_t>(numel));
    const uint16_t *src_u16 = reinterpret_cast<const uint16_t *>(src_bytes);
    for (int64_t i = 0; i < numel; ++i) {
        uint32_t f32_bits = static_cast<uint32_t>(src_u16[i]) << 16;
        float f32;
        std::memcpy(&f32, &f32_bits, sizeof(f32));
        h_half[static_cast<size_t>(i)] = __float2half(f32);
    }
    // Allocate device buffer and copy
    cudaError_t e = cudaMalloc(d_out, static_cast<size_t>(numel) * sizeof(__half));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(*d_out, h_half.data(),
                        static_cast<size_t>(numel) * sizeof(__half),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(*d_out); *d_out = nullptr; return e; }
    return cudaSuccess;
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

// Upload raw F8_E4M3 / F8_E8M0 bytes host → device (no conversion).
static cudaError_t upload_f8_bytes(const std::vector<uint8_t> &data,
                                   const uint8_t **d_out, cudaStream_t stream)
{
    if (data.empty()) return cudaSuccess;
    uint8_t *d = nullptr;
    cudaError_t e = cudaMalloc(&d, data.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d, data.data(), data.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d); return e; }
    *d_out = d;
    return cudaSuccess;
}

// Upload raw FP32 bytes host → device.
static cudaError_t upload_f32_bytes(const std::vector<uint8_t> &data,
                                    float **d_out, cudaStream_t stream)
{
    if (data.empty()) return cudaSuccess;
    float *d = nullptr;
    cudaError_t e = cudaMalloc(&d, data.size());
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d, data.data(), data.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d); return e; }
    *d_out = d;
    return cudaSuccess;
}

// Load one F8 attention weight + its per-tile scale from the arrow batch.
static cudaError_t load_attn_weight(
    const std::vector<TensorRecord> &tensors,
    const std::string &weight_suffix, const std::string &scale_suffix,
    const uint8_t **d_w, const uint8_t **d_scale,
    int64_t &n, int64_t &scale_n, int &rows, int &cols,
    cudaStream_t stream)
{
    const TensorRecord *wt = find_tensor_suffix(tensors, weight_suffix);
    if (!wt) return cudaSuccess;
    if (wt->shape.size() >= 2) {
        rows = static_cast<int>(wt->shape[0]);
        cols = static_cast<int>(wt->shape[1]);
    }
    n = static_cast<int64_t>(wt->data.size());
    cudaError_t e = upload_f8_bytes(wt->data, d_w, stream);
    if (e != cudaSuccess) return e;
    const TensorRecord *st = find_tensor_suffix(tensors, scale_suffix);
    if (st) {
        scale_n = static_cast<int64_t>(st->data.size());
        e = upload_f8_bytes(st->data, d_scale, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// Load one FP32 HC tensor (fn / base / scale) from the arrow batch.
static cudaError_t load_hc_f32(
    const std::vector<TensorRecord> &tensors,
    const std::string &suffix,
    float **d_out, cudaStream_t stream)
{
    const TensorRecord *t = find_tensor_suffix(tensors, suffix);
    if (!t) return cudaSuccess;
    return upload_f32_bytes(t->data, d_out, stream);
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
    const auto vault = std::filesystem::path(
        lsxcommon::lsx_model_metadata("manifests/model.index.json"));
    if (!std::filesystem::exists(ipath) && std::filesystem::exists(vault))
        ipath = vault;
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

std::string detect_model_type(const std::string &model_path)
{
    std::string result;
    auto ipath = std::filesystem::path(model_path) / "logestix" / "model.index.json";
    const auto vault = std::filesystem::path(
        lsxcommon::lsx_model_metadata("manifests/model.index.json"));
    if (!std::filesystem::exists(ipath) && std::filesystem::exists(vault))
        ipath = vault;
    std::ifstream f(ipath);
    if (!f) return result;
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return result;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return result;

    std::string_view name;
    if (root["model"].get_string().get(name) != simdjson::SUCCESS) return result;
    std::string raw(name);
    // Normalize: "DeepSeek-V4-Flash" -> "deepseek-v4-flash"
    for (char &c : raw) {
        if (c == ' ') c = '-';
        else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return raw;
}

// Parse a single expert arrow (given its flat index) into db.entries[idx] and
// keep the mmap + record batch alive in the cache. Returns true on success.
static bool parse_expert_file(ExpertDb &db, size_t idx)
{
    auto &apath = db.paths[idx];
    auto mmap_res = arrow::io::MemoryMappedFile::Open(apath, arrow::io::FileMode::READ);
    if (!mmap_res.ok()) {
        if (idx < 5 || idx % 1000 == 0)
            log::debug("[load_expert_index] mmap fail " + apath
                + ": " + mmap_res.status().ToString());
        return false;
    }
    auto mmap = *mmap_res;

    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
    if (!reader_res.ok() || (*reader_res)->num_record_batches() == 0)
        return false;
    auto batch_res = (*reader_res)->ReadRecordBatch(0);
    if (!batch_res.ok())
        return false;
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

    // ── Diagnostic: log real packed/scale geometry for the first few experts.
    // Reveals the actual NVFP4 scale block size (per-16 / per-32 / 2D tile) that
    // the converter passthrough stored, which the inference dequant must match.
    if (idx == 0 || idx == 1 || idx == static_cast<size_t>(db.experts_per_layer)) {
        auto fmt_scale_bytes = [](const uint8_t *p, int64_t n) -> std::string {
            std::ostringstream os;
            os << "[";
            for (int64_t i = 0; i < n && i < 8; ++i) {
                if (i) os << ",";
                char b[4];
                std::snprintf(b, sizeof(b), "%02x", p[i]);
                os << b;
            }
            os << "]";
            return os.str();
        };
        auto implied_blk = [](int64_t packed_bytes, int64_t scale_bytes) -> std::string {
            int64_t numel = packed_bytes * 2;
            if (scale_bytes <= 0) return "n/a (0 scales)";
            int64_t blk = numel / scale_bytes;
            // Detect a 2D tile layout: numel/blk is a perfect square-ish tile.
            std::ostringstream os;
            os << blk << " elems/scale (numel=" << numel
               << " scales=" << scale_bytes << ")";
            return os.str();
        };
        std::ostringstream os;
        os << "[diag-expert] idx=" << idx
           << " w1: packed=" << w1s << "B scales=" << w1ss << "B gs=" << w1gs
           << " blk=" << implied_blk(w1s, w1ss)
           << " s1[0..7]=" << fmt_scale_bytes(w1sp, w1ss)
           << " | w2: packed=" << w2s << "B scales=" << w2ss << "B gs=" << w2gs
           << " blk=" << implied_blk(w2s, w2ss)
           << " | w3: packed=" << w3s << "B scales=" << w3ss << "B gs=" << w3gs
           << " blk=" << implied_blk(w3s, w3ss);
        log::info(os.str());
    }

    db.entries[idx] = {w1p, w2p, w3p, w1sp, w2sp, w3sp,
                       w1gs, w2gs, w3gs,
                       w1s, w2s, w3s, w1ss, w2ss, w3ss};
    db.mmaps[idx] = std::move(mmap);
    db.batches[idx] = std::move(batch);
    return true;
}

ExpertDb load_expert_index(const std::string &model_path,
                            int num_layers, int experts_per_layer)
{
    ExpertDb db;
    int64_t total = static_cast<int64_t>(num_layers) * experts_per_layer;
    db.entries.resize(static_cast<size_t>(total));
    db.mmaps.resize(static_cast<size_t>(total));
    db.batches.resize(static_cast<size_t>(total));
    db.paths.resize(static_cast<size_t>(total));
    db.lru.reserve(db.max_open);
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    namespace fs = std::filesystem;

    // Build the file path per flat index. Parsing is deferred so we never hold
    // more than max_open CRT file descriptors (Windows cap is 8192).
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int exp = 0; exp < experts_per_layer; ++exp) {
            size_t idx = static_cast<size_t>(layer) * experts_per_layer
                       + static_cast<size_t>(exp);
            db.paths[idx] = (fs::path(model_path)
                / "logestix" / "experts" / ("layer_" + std::to_string(layer))
                / ("exp_" + std::to_string(exp) + ".arrow")).string();
        }
    }

    // Warm the cache up to max_open (keeps old behavior for the common layers).
    for (size_t idx = 0; idx < db.paths.size() && db.lru.size() < db.max_open; ++idx) {
        if (parse_expert_file(db, idx))
            db.lru.push_back(idx);
    }
    return db;
}

void ensure_expert_resident(ExpertDb &db, int layer, int expert)
{
    if (expert < 0 || expert >= db.experts_per_layer) return;
    if (db.paths.empty()) return;
    size_t idx = static_cast<size_t>(layer) * static_cast<size_t>(db.experts_per_layer)
               + static_cast<size_t>(expert);
    if (idx >= db.entries.size()) return;

    if (db.entries[idx].host_w1) {
        // Already resident — touch it (move to front of LRU).
        for (auto it = db.lru.begin(); it != db.lru.end(); ++it) {
            if (*it == idx) {
                db.lru.erase(it);
                db.lru.insert(db.lru.begin(), idx);
                break;
            }
        }
        return;
    }

    // Evict least-recently-used entries until under the cap.
    while (db.lru.size() >= db.max_open) {
        size_t evict = db.lru.back();
        db.lru.pop_back();
        db.mmaps[evict].reset();
        db.batches[evict].reset();
        db.entries[evict] = ExpertEntry();
    }
    if (parse_expert_file(db, idx))
        db.lru.insert(db.lru.begin(), idx);
}

void free_expert_index(ExpertDb &db)
{
    for (auto &m : db.mmaps) m.reset();
    for (auto &b : db.batches) b.reset();
    db.mmaps.clear();
    db.batches.clear();
    db.lru.clear();
    db.entries.clear();
    db.paths.clear();
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
    if (d_wq_a)         cudaFree(const_cast<uint8_t*>(d_wq_a));
    if (d_wq_a_scales)  cudaFree(const_cast<uint8_t*>(d_wq_a_scales));
    if (d_wq_b)         cudaFree(const_cast<uint8_t*>(d_wq_b));
    if (d_wq_b_scales)  cudaFree(const_cast<uint8_t*>(d_wq_b_scales));
    if (d_wkv)          cudaFree(const_cast<uint8_t*>(d_wkv));
    if (d_wkv_scales)   cudaFree(const_cast<uint8_t*>(d_wkv_scales));
    if (d_wo_a)         cudaFree(const_cast<uint8_t*>(d_wo_a));
    if (d_wo_a_scales)  cudaFree(const_cast<uint8_t*>(d_wo_a_scales));
    if (d_wo_b)         cudaFree(const_cast<uint8_t*>(d_wo_b));
    if (d_wo_b_scales)  cudaFree(const_cast<uint8_t*>(d_wo_b_scales));
    if (d_q_norm)       cudaFree(d_q_norm);
    if (d_kv_norm)      cudaFree(d_kv_norm);
    if (d_attn_norm)    cudaFree(d_attn_norm);
    if (d_attn_sink)    cudaFree(d_attn_sink);
    if (d_hc_attn_fn)   cudaFree(d_hc_attn_fn);
    if (d_hc_attn_base) cudaFree(d_hc_attn_base);
    if (d_hc_attn_scale)cudaFree(d_hc_attn_scale);
    if (d_hc_ffn_fn)    cudaFree(d_hc_ffn_fn);
    if (d_hc_ffn_base)  cudaFree(d_hc_ffn_base);
    if (d_hc_ffn_scale) cudaFree(d_hc_ffn_scale);
    if (h_tid2eid)      cudaFreeHost(h_tid2eid);
    if (h_gate_bias)    cudaFreeHost(h_gate_bias);
    // CSA compressor
    if (d_comp_wkv)     cudaFree(d_comp_wkv);
    if (d_comp_wgate)   cudaFree(d_comp_wgate);
    if (d_comp_ape)     cudaFree(d_comp_ape);
    if (d_comp_norm)    cudaFree(d_comp_norm);
    // CSA indexer
    if (d_idx_wq_b)     cudaFree(d_idx_wq_b);
    if (d_idx_wq_b_scales) cudaFree(d_idx_wq_b_scales);
    if (d_idx_weights_proj) cudaFree(d_idx_weights_proj);
    if (d_idx_comp_wkv)     cudaFree(d_idx_comp_wkv);
    if (d_idx_comp_wgate)   cudaFree(d_idx_comp_wgate);
    if (d_idx_comp_ape)     cudaFree(d_idx_comp_ape);
    if (d_idx_comp_norm)    cudaFree(d_idx_comp_norm);
}

void PinnedWeights::Free()
{
    if (d_embed)      cudaFree(d_embed);
    if (d_head)       cudaFree(d_head);
    if (d_final_norm) cudaFree(d_final_norm);
    if (d_hc_head_fn)   cudaFree(d_hc_head_fn);
    if (d_hc_head_base) cudaFree(d_hc_head_base);
    if (d_hc_head_scale)cudaFree(d_hc_head_scale);
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
        log::error("[load_pinned] ERROR loading global.arrow: "
            + global_tensors_res.status().ToString());
        return cudaErrorUnknown;
    }
    auto &global_tensors = *global_tensors_res;

    // Debug: print first 10 tensor names
    for (int di = 0; di < std::min(10, static_cast<int>(global_tensors.size())); ++di) {
        const auto &t = global_tensors[di];
        std::string shape;
        for (size_t si = 0; si < t.shape.size(); ++si)
            shape += (si > 0 ? "," : "") + std::to_string(t.shape[si]);
        std::ostringstream raw;
        for (size_t bi = 0; bi < std::min<size_t>(8, t.data.size() / 2); ++bi) {
            uint16_t u = static_cast<uint16_t>(t.data[bi * 2])
                       | (static_cast<uint16_t>(t.data[bi * 2 + 1]) << 8);
            raw << " " << std::hex << u << std::dec;
        }
        log::info("[load_pinned] global tensor[" + std::to_string(di)
            + "] name='" + t.name + "' dtype=" + t.dtype + " shape=[" + shape + "] data_sz="
            + std::to_string(t.data.size()) + " raw_u16[0..7]=" + raw.str());
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
        if (e != cudaSuccess) { log::error("embed upload failed"); return e; }
    }
    if (norm_t && !norm_t->shape.empty()) {
        int64_t numel = norm_t->shape[0];
        cudaError_t e = upload_bf16_to_half(norm_t->data.data(), numel,
                                             &pw.d_final_norm, stream);
        if (e != cudaSuccess) { log::error("norm upload failed"); return e; }
    }
    if (head_t && head_t->shape.size() >= 2) {
        if (pw.vocab_size == 0)
            pw.vocab_size = static_cast<int>(head_t->shape[0]);
        int64_t numel = head_t->shape[0] * head_t->shape[1];
        cudaError_t e = upload_bf16_to_half(head_t->data.data(), numel,
                                             &pw.d_head, stream);
        if (e != cudaSuccess) { log::error("head upload failed"); return e; }
    }

    // Hyper-Connections head (FP32, in global.arrow)
    auto *hc_head_fn = find_tensor_suffix(global_tensors, "hc_head_fn");
    auto *hc_head_base = find_tensor_suffix(global_tensors, "hc_head_base");
    auto *hc_head_scale = find_tensor_suffix(global_tensors, "hc_head_scale");
    if (hc_head_fn) upload_f32_bytes(hc_head_fn->data, &pw.d_hc_head_fn, stream);
    if (hc_head_base) upload_f32_bytes(hc_head_base->data, &pw.d_hc_head_base, stream);
    if (hc_head_scale) upload_f32_bytes(hc_head_scale->data, &pw.d_hc_head_scale, stream);

    pw.layers.resize(kNumLayers);
    for (int l = 0; l < kNumLayers; ++l) {
        auto fn = "layer_" + std::to_string(l) + ".arrow";
        auto layer_path = pinned_dir / fn;

        if (!fs::exists(layer_path)) continue;

        auto tensors_res = load_arrow_file(layer_path.string());
        if (!tensors_res.ok()) {
            log::error("[load_pinned] ERROR loading " + fn + ": "
                + tensors_res.status().ToString());
            continue;
        }
        auto &lt = *tensors_res;
        auto &lw = pw.layers[l];

        // Set compress_ratio and rope_theta per layer from config.
        // config.compress_ratios has 44 entries; entry 43 is the MTP block.
        // Decoder layers 0..42 map to config indices 0..42 (pure MLA=0, CSA>0).
        static const int compress_ratios[] = {
            0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4
        };
        lw.compress_ratio = (l < 43) ? compress_ratios[l] : 0;
        lw.rope_theta = (lw.compress_ratio > 0) ? 160000.0f : 10000.0f;
        // CSA overlap: ratio==4 uses coff=2 (reference Compressor: coff = 1 + overlap).
        lw.comp_coff = (lw.compress_ratio == 4) ? 2 : 1;
        for (int di = 0; di < std::min(5, static_cast<int>(lt.size())); ++di) {
            const auto &t = lt[di];
            std::string shape;
            for (size_t si = 0; si < t.shape.size(); ++si)
                shape += (si > 0 ? "," : "") + std::to_string(t.shape[si]);
            log::debug(std::string("[load_pinned]   layer_")
                + (l < 10 ? "00" : (l < 100 ? "0" : "")) + std::to_string(l)
                + " tensor[" + std::to_string(di) + "] name='" + t.name
                + "' shape=[" + shape + "]");
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

        // Router selection bias (F32, non-hash layers only) — host copy.
        auto *gate_bias = find_tensor_substr(lt, "ffn.gate.bias");
        if (gate_bias && gate_bias->shape.size() == 1 &&
            gate_bias->dtype == "F32") {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias, static_cast<size_t>(n) * sizeof(float))
                    == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data.data(),
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }

        // Hash-routing table (I64 [vocab, topk], hash layers only) — host copy.
        auto *tid2eid = find_tensor_substr(lt, "ffn.gate.tid2eid");
        if (tid2eid && tid2eid->shape.size() == 2 &&
            tid2eid->dtype == "I64") {
            int64_t numel = tid2eid->shape[0] * tid2eid->shape[1];
            if (cudaMallocHost(&lw.h_tid2eid, static_cast<size_t>(numel) * sizeof(int64_t))
                    == cudaSuccess) {
                std::memcpy(lw.h_tid2eid, tid2eid->data.data(),
                            static_cast<size_t>(numel) * sizeof(int64_t));
                lw.tid2eid_vocab = static_cast<int>(tid2eid->shape[0]);
                lw.tid2eid_topk  = static_cast<int>(tid2eid->shape[1]);
            }
        }

        // MLA attention weights (F8_E4M3 + per-tile F8_E8M0 scales)
        load_attn_weight(lt, "attn.wq_a.weight", "attn.wq_a.scale",
                         &lw.d_wq_a, &lw.d_wq_a_scales, lw.wq_a_n, lw.wq_a_s_n,
                         lw.wq_a_rows, lw.wq_a_cols, stream);
        load_attn_weight(lt, "attn.wq_b.weight", "attn.wq_b.scale",
                         &lw.d_wq_b, &lw.d_wq_b_scales, lw.wq_b_n, lw.wq_b_s_n,
                         lw.wq_b_rows, lw.wq_b_cols, stream);
        load_attn_weight(lt, "attn.wkv.weight", "attn.wkv.scale",
                         &lw.d_wkv, &lw.d_wkv_scales, lw.wkv_n, lw.wkv_s_n,
                         lw.wkv_rows, lw.wkv_cols, stream);
        load_attn_weight(lt, "attn.wo_a.weight", "attn.wo_a.scale",
                         &lw.d_wo_a, &lw.d_wo_a_scales, lw.wo_a_n, lw.wo_a_s_n,
                         lw.wo_a_rows, lw.wo_a_cols, stream);
        load_attn_weight(lt, "attn.wo_b.weight", "attn.wo_b.scale",
                         &lw.d_wo_b, &lw.d_wo_b_scales, lw.wo_b_n, lw.wo_b_s_n,
                         lw.wo_b_rows, lw.wo_b_cols, stream);

        // Attention norms (BF16) + sink (FP32)
        auto *q_norm = find_tensor_suffix(lt, "attn.q_norm.weight");
        if (q_norm && !q_norm->shape.empty())
            upload_bf16_to_half(q_norm->data.data(), q_norm->shape[0],
                                &lw.d_q_norm, stream);
        auto *kv_norm = find_tensor_suffix(lt, "attn.kv_norm.weight");
        if (kv_norm && !kv_norm->shape.empty())
            upload_bf16_to_half(kv_norm->data.data(), kv_norm->shape[0],
                                &lw.d_kv_norm, stream);
        auto *attn_norm = find_tensor_suffix(lt, "attn_norm.weight");
        if (attn_norm && !attn_norm->shape.empty())
            upload_bf16_to_half(attn_norm->data.data(), attn_norm->shape[0],
                                &lw.d_attn_norm, stream);
        auto *attn_sink = find_tensor_suffix(lt, "attn.attn_sink");
        if (attn_sink)
            upload_f32_bytes(attn_sink->data, &lw.d_attn_sink, stream);

        // Hyper-Connections (FP32)
        load_hc_f32(lt, "hc_attn_fn", &lw.d_hc_attn_fn, stream);
        load_hc_f32(lt, "hc_attn_base", &lw.d_hc_attn_base, stream);
        load_hc_f32(lt, "hc_attn_scale", &lw.d_hc_attn_scale, stream);
        load_hc_f32(lt, "hc_ffn_fn", &lw.d_hc_ffn_fn, stream);
        load_hc_f32(lt, "hc_ffn_base", &lw.d_hc_ffn_base, stream);
        load_hc_f32(lt, "hc_ffn_scale", &lw.d_hc_ffn_scale, stream);

        if (l == 0) {
            std::ostringstream oss;
            oss << "[load_pinned] layer0 attn: wq_a=" << (lw.d_wq_a ? "yes" : "no")
                << " (" << lw.wq_a_n << ") wq_b=" << (lw.d_wq_b ? "yes" : "no")
                << " (" << lw.wq_b_n << ") wkv=" << (lw.d_wkv ? "yes" : "no")
                << " (" << lw.wkv_n << ") wo_a=" << (lw.d_wo_a ? "yes" : "no")
                << " (" << lw.wo_a_n << ") wo_b=" << (lw.d_wo_b ? "yes" : "no")
                << " (" << lw.wo_b_n << ")";
            log::info(oss.str());
            std::ostringstream oss2;
            oss2 << "[load_pinned] layer0 hc: attn_fn=" << (lw.d_hc_attn_fn ? "yes" : "no")
                << " ffn_fn=" << (lw.d_hc_ffn_fn ? "yes" : "no")
                << " sink=" << (lw.d_attn_sink ? "yes" : "no")
                << " q_norm=" << (lw.d_q_norm ? "yes" : "no")
                << " kv_norm=" << (lw.d_kv_norm ? "yes" : "no")
                << " attn_norm=" << (lw.d_attn_norm ? "yes" : "no");
            log::info(oss2.str());
        }

        // ── CSA Compressor (layers 2-41) ──────────────────────────────────
        if (lw.compress_ratio > 0) {
            if (l == 2) log::info("[load_pinned] layer2 CSA loading (ratio=" + std::to_string(lw.compress_ratio) + ")");
            int coff = lw.comp_coff;
            int comp_out = coff * kHeadDim;  // 1024 for ratio=4, 512 for ratio=128

            auto *t_comp_wkv = find_tensor_suffix(lt, "attn.compressor.wkv.weight");
            if (t_comp_wkv && !t_comp_wkv->data.empty())
                upload_bf16_to_half(t_comp_wkv->data.data(),
                    static_cast<int64_t>(comp_out) * kHiddenDim,
                    &lw.d_comp_wkv, stream);

            auto *t_comp_wgate = find_tensor_suffix(lt, "attn.compressor.wgate.weight");
            if (t_comp_wgate && !t_comp_wgate->data.empty())
                upload_bf16_to_half(t_comp_wgate->data.data(),
                    static_cast<int64_t>(comp_out) * kHiddenDim,
                    &lw.d_comp_wgate, stream);

            auto *t_comp_ape = find_tensor_suffix(lt, "attn.compressor.ape");
            if (t_comp_ape && !t_comp_ape->data.empty()) {
                std::ostringstream aoss;
                aoss << "[load_pinned] layer CSA ape shape=[";
                for (size_t ai = 0; ai < t_comp_ape->shape.size(); ++ai) {
                    if (ai) aoss << ",";
                    aoss << t_comp_ape->shape[ai];
                }
                aoss << "] dsz=" << t_comp_ape->data.size();
                if (l == 2) log::info(aoss.str());
                upload_f32_bytes(t_comp_ape->data, &lw.d_comp_ape, stream);
            }

            auto *t_comp_norm = find_tensor_suffix(lt, "attn.compressor.norm.weight");
            if (t_comp_norm && !t_comp_norm->data.empty())
                upload_bf16_to_half(t_comp_norm->data.data(),
                    static_cast<int64_t>(kHeadDim), &lw.d_comp_norm, stream);

            // ── CSA Indexer (ratio=4 only) ─────────────────────────────────
            if (lw.compress_ratio == 4) {
                // indexer.wq_b: F8_E4M3 [8192, 1024] + F8_E8M0 scale [64, 8]
                auto *t_idx_wq_b = find_tensor_suffix(lt, "attn.indexer.wq_b.weight");
                auto *t_idx_wq_b_s = find_tensor_suffix(lt, "attn.indexer.wq_b.scale");
                if (t_idx_wq_b && !t_idx_wq_b->data.empty()) {
                    int64_t wq_n = static_cast<int64_t>(t_idx_wq_b->data.size());
                    uint8_t *d_raw = nullptr;
                    cudaMalloc(&d_raw, wq_n);
                    cudaMemcpyAsync(d_raw, t_idx_wq_b->data.data(), wq_n,
                        cudaMemcpyHostToDevice, stream);
                    lw.d_idx_wq_b = d_raw;
                    lw.idx_wq_b_rows = 8192;
                    lw.idx_wq_b_cols = 1024;
                }
                if (t_idx_wq_b_s && !t_idx_wq_b_s->data.empty()) {
                    int64_t sc_n = static_cast<int64_t>(t_idx_wq_b_s->data.size());
                    uint8_t *d_sc = nullptr;
                    cudaMalloc(&d_sc, sc_n);
                    cudaMemcpyAsync(d_sc, t_idx_wq_b_s->data.data(), sc_n,
                        cudaMemcpyHostToDevice, stream);
                    lw.d_idx_wq_b_scales = d_sc;
                }
                // indexer.weights_proj: BF16 [64, 4096]
                auto *t_idx_wp = find_tensor_suffix(lt, "attn.indexer.weights_proj.weight");
                if (t_idx_wp && !t_idx_wp->data.empty())
                    upload_bf16_to_half(t_idx_wp->data.data(),
                        64LL * kHiddenDim, &lw.d_idx_weights_proj, stream);
                // indexer.compressor.* (head_dim=128)
                auto *t_ic_wkv = find_tensor_suffix(lt, "attn.indexer.compressor.wkv.weight");
                if (t_ic_wkv && !t_ic_wkv->data.empty())
                    upload_bf16_to_half(t_ic_wkv->data.data(),
                        256LL * kHiddenDim, &lw.d_idx_comp_wkv, stream);
                auto *t_ic_wgate = find_tensor_suffix(lt, "attn.indexer.compressor.wgate.weight");
                if (t_ic_wgate && !t_ic_wgate->data.empty())
                    upload_bf16_to_half(t_ic_wgate->data.data(),
                        256LL * kHiddenDim, &lw.d_idx_comp_wgate, stream);
                auto *t_ic_ape = find_tensor_suffix(lt, "attn.indexer.compressor.ape");
                if (t_ic_ape && !t_ic_ape->data.empty())
                    upload_f32_bytes(t_ic_ape->data, &lw.d_idx_comp_ape, stream);
                auto *t_ic_norm = find_tensor_suffix(lt, "attn.indexer.compressor.norm.weight");
                if (t_ic_norm && !t_ic_norm->data.empty())
                    upload_bf16_to_half(t_ic_norm->data.data(),
                        128LL, &lw.d_idx_comp_norm, stream);
            }

            if (l == 2)
                log::info("[load_pinned] layer2 CSA: comp_wkv=" + std::string(lw.d_comp_wkv ? "yes" : "no")
                    + " comp_wgate=" + std::string(lw.d_comp_wgate ? "yes" : "no")
                    + " comp_ape=" + std::string(lw.d_comp_ape ? "yes" : "no")
                    + " comp_norm=" + std::string(lw.d_comp_norm ? "yes" : "no")
                    + " idx_wq_b=" + std::string(lw.d_idx_wq_b ? "yes" : "no")
                    + " idx_weights_proj=" + std::string(lw.d_idx_weights_proj ? "yes" : "no"));
        }

    }

    // Shared experts: NVFP4-packed metadata from pinned/shared_experts.arrow.
    // Loaded AFTER the layer loop so the descriptor registry lives independently
    // of the per-layer pinned files.
    {
        auto shared_path = pinned_dir / "shared_experts.arrow";
        if (fs::exists(shared_path)) {
            auto mmap_res = arrow::io::MemoryMappedFile::Open(
                shared_path.string(), arrow::io::FileMode::READ);
            if (mmap_res.ok()) {
                SharedExpertDb db;
                auto st = load_shared_experts_metadata(*mmap_res, db);
                if (!st.ok()) {
                    log::error("[load_pinned] ERROR shared_experts metadata: "
                        + st.ToString());
                } else {
                    log::info("[load_pinned] shared_experts: "
                        + std::to_string(db.num_layers()) + " layers");
                    if (db.num_layers() > 0) {
                        std::ostringstream gs_oss;
                        gs_oss << "[load_pinned] layer0 shared: w1_bytes="
                               << db.w1[0].packed_bytes
                               << " w1_gs=" << db.w1[0].global_scale
                               << " w2_bytes=" << db.w2[0].packed_bytes
                               << " w2_gs=" << db.w2[0].global_scale
                               << " w3_bytes=" << db.w3[0].packed_bytes
                               << " w3_gs=" << db.w3[0].global_scale
                               << " w1_scale_bytes=" << db.w1[0].scale_bytes
                               << " w1_numel=" << db.w1[0].num_elements;
                        log::info(gs_oss.str());
                    }
                    for (int l = 0; l < db.num_layers() &&
                                    l < static_cast<int>(pw.layers.size()); ++l) {
                        auto &lw = pw.layers[l];
                        auto upload = [&](const SharedExpertDescriptor &desc,
                                          uint8_t *&d_buf, uint8_t *&d_scales,
                                          float &gs, int64_t &nbytes)
                        {
                            nbytes = desc.packed_bytes;
                            if (nbytes <= 0) return;
                            cudaError_t e = cudaMalloc(&d_buf,
                                static_cast<size_t>(desc.packed_bytes));
                            if (e != cudaSuccess) return;
                            cudaMemcpyAsync(d_buf, desc.host_ptr,
                                static_cast<size_t>(desc.packed_bytes),
                                cudaMemcpyHostToDevice, stream);
                            if (desc.scale_bytes > 0) {
                                cudaMalloc(&d_scales,
                                    static_cast<size_t>(desc.scale_bytes));
                                cudaMemcpyAsync(d_scales, desc.scale_host_ptr,
                                    static_cast<size_t>(desc.scale_bytes),
                                    cudaMemcpyHostToDevice, stream);
                            }
                            gs = desc.global_scale;
                        };
                        upload(db.w1[l], lw.d_shared_w1, lw.d_shared_w1_scales,
                               lw.shared_w1_global_scale, lw.shared_w1_bytes);
                        upload(db.w2[l], lw.d_shared_w2, lw.d_shared_w2_scales,
                               lw.shared_w2_global_scale, lw.shared_w2_bytes);
                        upload(db.w3[l], lw.d_shared_w3, lw.d_shared_w3_scales,
                               lw.shared_w3_global_scale, lw.shared_w3_bytes);
                    }
                }
            }
        }
    }

    return cudaSuccess;
}

} // namespace gpu
} // namespace lsxcommon
