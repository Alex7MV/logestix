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
                                 __half **d_out, cudaStream_t stream,
                                 std::uint64_t &bytes_acc)
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
    bytes_acc += static_cast<std::uint64_t>(numel) * sizeof(__half);
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
                                   const uint8_t **d_out, cudaStream_t stream,
                                   std::uint64_t &bytes_acc)
{
    if (data.empty()) return cudaSuccess;
    uint8_t *d = nullptr;
    cudaError_t e = cudaMalloc(&d, data.size());
    if (e != cudaSuccess) return e;
    bytes_acc += data.size();
    e = cudaMemcpyAsync(d, data.data(), data.size(),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { cudaFree(d); return e; }
    *d_out = d;
    return cudaSuccess;
}

// Upload raw FP32 bytes host → device.
static cudaError_t upload_f32_bytes(const std::vector<uint8_t> &data,
                                    float **d_out, cudaStream_t stream,
                                    std::uint64_t &bytes_acc)
{
    if (data.empty()) return cudaSuccess;
    float *d = nullptr;
    cudaError_t e = cudaMalloc(&d, data.size());
    if (e != cudaSuccess) return e;
    bytes_acc += data.size();
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
    cudaStream_t stream, std::uint64_t &bytes_acc)
{
    const TensorRecord *wt = find_tensor_suffix(tensors, weight_suffix);
    if (!wt) return cudaSuccess;
    if (wt->shape.size() >= 2) {
        rows = static_cast<int>(wt->shape[0]);
        cols = static_cast<int>(wt->shape[1]);
    }
    n = static_cast<int64_t>(wt->data.size());
    cudaError_t e = upload_f8_bytes(wt->data, d_w, stream, bytes_acc);
    if (e != cudaSuccess) {
        log::error("[deepseek_load] weight upload failed for '"
                   + weight_suffix + "': " + cudaGetErrorString(e));
        return e;
    }
    const TensorRecord *st = find_tensor_suffix(tensors, scale_suffix);
    if (st) {
        scale_n = static_cast<int64_t>(st->data.size());
        e = upload_f8_bytes(st->data, d_scale, stream, bytes_acc);
        if (e != cudaSuccess) {
            // The weight upload above already succeeded and *d_w is a live
            // device pointer, but without its matching scale a dequant
            // kernel would read a null/garbage scale against a valid weight
            // buffer — an inconsistent, not deterministically-zero, state.
            // Roll the weight back too so this attention weight is either
            // fully present or (matching every other partial-load path in
            // this loader) cleanly absent.
            log::error("[deepseek_load] scale upload failed for '"
                       + scale_suffix + "', rolling back weight '"
                       + weight_suffix + "': " + cudaGetErrorString(e));
            cudaFree(const_cast<uint8_t *>(*d_w));
            *d_w = nullptr;
            n = 0;
            return e;
        }
    }
    return cudaSuccess;
}

// Load one FP32 HC tensor (fn / base / scale) from the arrow batch.
static cudaError_t load_hc_f32(
    const std::vector<TensorRecord> &tensors,
    const std::string &suffix,
    float **d_out, cudaStream_t stream,
    std::uint64_t &bytes_acc)
{
    const TensorRecord *t = find_tensor_suffix(tensors, suffix);
    if (!t) return cudaSuccess;
    return upload_f32_bytes(t->data, d_out, stream, bytes_acc);
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

// No per-model parser needed: the shared ExpertStore<TEntry>::DecodeEntry()
// extracts the six ExpertSchema bins + three global scales into TEntry.

ExpertDb load_expert_index(const std::string &model_path,
                            int num_layers, int experts_per_layer)
{
    ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.max_open = 7000;
    db.log_prefix = "[load_expert_index] ";
    db.mmap_fail_verb = "mmap fail ";
    db.emit_diag = true;             // DeepSeek diag block
    db.log_batch_fails = false;      // DeepSeek logs no batch-failure lines
    db.Load(model_path, /*fallback=*/false);
    return db;
}

void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_expert_index(ExpertDb &db) { db.Free(); }

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
    resident_bytes = 0;
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
    std::uint64_t rb = 0;   // device-resident byte total (weights only)

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
                                             &pw.d_embed, stream, rb);
        if (e != cudaSuccess) { log::error("embed upload failed"); return e; }
    }
    if (norm_t && !norm_t->shape.empty()) {
        int64_t numel = norm_t->shape[0];
        cudaError_t e = upload_bf16_to_half(norm_t->data.data(), numel,
                                             &pw.d_final_norm, stream, rb);
        if (e != cudaSuccess) { log::error("norm upload failed"); return e; }
    }
    if (head_t && head_t->shape.size() >= 2) {
        if (pw.vocab_size == 0)
            pw.vocab_size = static_cast<int>(head_t->shape[0]);
        int64_t numel = head_t->shape[0] * head_t->shape[1];
        cudaError_t e = upload_bf16_to_half(head_t->data.data(), numel,
                                             &pw.d_head, stream, rb);
        if (e != cudaSuccess) { log::error("head upload failed"); return e; }
    }

    // Hyper-Connections head (FP32, in global.arrow)
    auto *hc_head_fn = find_tensor_suffix(global_tensors, "hc_head_fn");
    auto *hc_head_base = find_tensor_suffix(global_tensors, "hc_head_base");
    auto *hc_head_scale = find_tensor_suffix(global_tensors, "hc_head_scale");
    if (hc_head_fn) upload_f32_bytes(hc_head_fn->data, &pw.d_hc_head_fn, stream, rb);
    if (hc_head_base) upload_f32_bytes(hc_head_base->data, &pw.d_hc_head_base, stream, rb);
    if (hc_head_scale) upload_f32_bytes(hc_head_scale->data, &pw.d_hc_head_scale, stream, rb);

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
                                &lw.d_ffn_norm, stream, rb);

        auto *gate = find_tensor_substr(lt, "ffn.gate.weight");
        if (gate && gate->shape.size() >= 2) {
            int64_t numel = gate->shape[0] * gate->shape[1];
            // Use upload_bf16_to_half which does proper host→device copy first
            upload_bf16_to_half(gate->data.data(), numel,
                                &lw.d_gate_weight, stream, rb);
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
                         lw.wq_a_rows, lw.wq_a_cols, stream, rb);
        load_attn_weight(lt, "attn.wq_b.weight", "attn.wq_b.scale",
                         &lw.d_wq_b, &lw.d_wq_b_scales, lw.wq_b_n, lw.wq_b_s_n,
                         lw.wq_b_rows, lw.wq_b_cols, stream, rb);
        load_attn_weight(lt, "attn.wkv.weight", "attn.wkv.scale",
                         &lw.d_wkv, &lw.d_wkv_scales, lw.wkv_n, lw.wkv_s_n,
                         lw.wkv_rows, lw.wkv_cols, stream, rb);
        load_attn_weight(lt, "attn.wo_a.weight", "attn.wo_a.scale",
                         &lw.d_wo_a, &lw.d_wo_a_scales, lw.wo_a_n, lw.wo_a_s_n,
                         lw.wo_a_rows, lw.wo_a_cols, stream, rb);
        load_attn_weight(lt, "attn.wo_b.weight", "attn.wo_b.scale",
                         &lw.d_wo_b, &lw.d_wo_b_scales, lw.wo_b_n, lw.wo_b_s_n,
                         lw.wo_b_rows, lw.wo_b_cols, stream, rb);

        // Attention norms (BF16) + sink (FP32)
        auto *q_norm = find_tensor_suffix(lt, "attn.q_norm.weight");
        if (q_norm && !q_norm->shape.empty())
            upload_bf16_to_half(q_norm->data.data(), q_norm->shape[0],
                                &lw.d_q_norm, stream, rb);
        auto *kv_norm = find_tensor_suffix(lt, "attn.kv_norm.weight");
        if (kv_norm && !kv_norm->shape.empty())
            upload_bf16_to_half(kv_norm->data.data(), kv_norm->shape[0],
                                &lw.d_kv_norm, stream, rb);
        auto *attn_norm = find_tensor_suffix(lt, "attn_norm.weight");
        if (attn_norm && !attn_norm->shape.empty())
            upload_bf16_to_half(attn_norm->data.data(), attn_norm->shape[0],
                                &lw.d_attn_norm, stream, rb);
        auto *attn_sink = find_tensor_suffix(lt, "attn.attn_sink");
        if (attn_sink)
            upload_f32_bytes(attn_sink->data, &lw.d_attn_sink, stream, rb);

        // Hyper-Connections (FP32)
        load_hc_f32(lt, "hc_attn_fn", &lw.d_hc_attn_fn, stream, rb);
        load_hc_f32(lt, "hc_attn_base", &lw.d_hc_attn_base, stream, rb);
        load_hc_f32(lt, "hc_attn_scale", &lw.d_hc_attn_scale, stream, rb);
        load_hc_f32(lt, "hc_ffn_fn", &lw.d_hc_ffn_fn, stream, rb);
        load_hc_f32(lt, "hc_ffn_base", &lw.d_hc_ffn_base, stream, rb);
        load_hc_f32(lt, "hc_ffn_scale", &lw.d_hc_ffn_scale, stream, rb);

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
                    &lw.d_comp_wkv, stream, rb);

            auto *t_comp_wgate = find_tensor_suffix(lt, "attn.compressor.wgate.weight");
            if (t_comp_wgate && !t_comp_wgate->data.empty())
                upload_bf16_to_half(t_comp_wgate->data.data(),
                    static_cast<int64_t>(comp_out) * kHiddenDim,
                    &lw.d_comp_wgate, stream, rb);

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
                upload_f32_bytes(t_comp_ape->data, &lw.d_comp_ape, stream, rb);
            }

            auto *t_comp_norm = find_tensor_suffix(lt, "attn.compressor.norm.weight");
            if (t_comp_norm && !t_comp_norm->data.empty())
                upload_bf16_to_half(t_comp_norm->data.data(),
                    static_cast<int64_t>(kHeadDim), &lw.d_comp_norm, stream, rb);

            // ── CSA Indexer (ratio=4 only) ─────────────────────────────────
            if (lw.compress_ratio == 4) {
                // indexer.wq_b: F8_E4M3 [8192, 1024] + F8_E8M0 scale [64, 8]
                auto *t_idx_wq_b = find_tensor_suffix(lt, "attn.indexer.wq_b.weight");
                auto *t_idx_wq_b_s = find_tensor_suffix(lt, "attn.indexer.wq_b.scale");
                if (t_idx_wq_b && !t_idx_wq_b->data.empty()) {
                    int64_t wq_n = static_cast<int64_t>(t_idx_wq_b->data.size());
                    uint8_t *d_raw = nullptr;
                    cudaError_t ce = cudaMalloc(&d_raw, wq_n);
                    if (ce != cudaSuccess) { log::error("indexer wq_b alloc failed"); return ce; }
                    ce = cudaMemcpyAsync(d_raw, t_idx_wq_b->data.data(), wq_n,
                        cudaMemcpyHostToDevice, stream);
                    if (ce != cudaSuccess) {
                        cudaFree(d_raw);
                        log::error("indexer wq_b upload failed");
                        return ce;
                    }
                    rb += static_cast<std::uint64_t>(wq_n);
                    lw.d_idx_wq_b = d_raw;
                    lw.idx_wq_b_rows = 8192;
                    lw.idx_wq_b_cols = 1024;
                }
                if (t_idx_wq_b_s && !t_idx_wq_b_s->data.empty()) {
                    int64_t sc_n = static_cast<int64_t>(t_idx_wq_b_s->data.size());
                    uint8_t *d_sc = nullptr;
                    cudaError_t ce = cudaMalloc(&d_sc, sc_n);
                    if (ce != cudaSuccess) { log::error("indexer wq_b scale alloc failed"); return ce; }
                    ce = cudaMemcpyAsync(d_sc, t_idx_wq_b_s->data.data(), sc_n,
                        cudaMemcpyHostToDevice, stream);
                    if (ce != cudaSuccess) {
                        cudaFree(d_sc);
                        log::error("indexer wq_b scale upload failed");
                        return ce;
                    }
                    rb += static_cast<std::uint64_t>(sc_n);
                    lw.d_idx_wq_b_scales = d_sc;
                }
                // indexer.weights_proj: BF16 [64, 4096]
                auto *t_idx_wp = find_tensor_suffix(lt, "attn.indexer.weights_proj.weight");
                if (t_idx_wp && !t_idx_wp->data.empty())
                    upload_bf16_to_half(t_idx_wp->data.data(),
                        64LL * kHiddenDim, &lw.d_idx_weights_proj, stream, rb);
                // indexer.compressor.* (head_dim=128)
                auto *t_ic_wkv = find_tensor_suffix(lt, "attn.indexer.compressor.wkv.weight");
                if (t_ic_wkv && !t_ic_wkv->data.empty())
                    upload_bf16_to_half(t_ic_wkv->data.data(),
                        256LL * kHiddenDim, &lw.d_idx_comp_wkv, stream, rb);
                auto *t_ic_wgate = find_tensor_suffix(lt, "attn.indexer.compressor.wgate.weight");
                if (t_ic_wgate && !t_ic_wgate->data.empty())
                    upload_bf16_to_half(t_ic_wgate->data.data(),
                        256LL * kHiddenDim, &lw.d_idx_comp_wgate, stream, rb);
                auto *t_ic_ape = find_tensor_suffix(lt, "attn.indexer.compressor.ape");
                if (t_ic_ape && !t_ic_ape->data.empty())
                    upload_f32_bytes(t_ic_ape->data, &lw.d_idx_comp_ape, stream, rb);
                auto *t_ic_norm = find_tensor_suffix(lt, "attn.indexer.compressor.norm.weight");
                if (t_ic_norm && !t_ic_norm->data.empty())
                    upload_bf16_to_half(t_ic_norm->data.data(),
                        128LL, &lw.d_idx_comp_norm, stream, rb);
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
                            rb += static_cast<std::uint64_t>(desc.packed_bytes);
                            cudaMemcpyAsync(d_buf, desc.host_ptr,
                                static_cast<size_t>(desc.packed_bytes),
                                cudaMemcpyHostToDevice, stream);
                            if (desc.scale_bytes > 0) {
                                cudaMalloc(&d_scales,
                                    static_cast<size_t>(desc.scale_bytes));
                                rb += static_cast<std::uint64_t>(desc.scale_bytes);
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

    pw.resident_bytes = rb;
    return cudaSuccess;
}

// ── DeepSeekInferArena: single-allocation per-run footprint ────────────────
// Mirrors GlmMoEArena::AllocateGeneration's page() lambda: one device block +
// one pinned host block, 16-byte-aligned slices, sized exactly as the old
// malloc-per-buffer pipeline (deepseek_inference_pipeline.cu) and
// alloc_csa_state. The CSA score buffers must start at -inf (the overlap
// pooling masks rows [0, ratio) before the first shift), everything else zero.

namespace {

__global__ void ds_fill_neg_inf_kernel(float* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = -INFINITY;
}

} // namespace

cudaError_t DeepSeekInferArena::Allocate(cudaStream_t stream, int num_layers,
                                         int max_seq_len,
                                         const int* compress_ratios, int vocab)
{
    Free();
    constexpr int64_t kIdxHeadDim = 128;

    // Sizing replicated verbatim from the old pipeline (lines 252-268) and
    // alloc_csa_state so the arena is a pure allocation refactor.
    const int64_t dequant_sz = static_cast<int64_t>(kHiddenDim) * kHiddenDim
                             * sizeof(float);
    const int64_t shared_int_sz = static_cast<int64_t>(kHiddenDim)
                                * sizeof(__half);
    const int64_t max_attn_elems = static_cast<int64_t>(kAttnHeads) * kHeadDim
                                 * kQloraRank;
    const int64_t max_attn_elems_wo = 8192LL * 4096;
    const int64_t attn_dequant_max = std::max(max_attn_elems, max_attn_elems_wo);
    // Extra FP32 scratch past the dequant block for the largest single attention
    // GEMM: f8_gemm_1d needs cols for the FP32 activation copy plus rows for the
    // FP32 output (clamped to FP16 afterwards). wq_b [32768,1024] needs
    // kQloraRank + kAttnHeads*kHeadDim; wo_b [4096,8192] needs 8192 + 4096. The
    // wq_b case (33,792) dominates. An undersized scratch here overflows into
    // the adjacent kv cache (corrupting layer-0 window slots on the first step).
    const int64_t act_plus_out = static_cast<int64_t>(kQloraRank)
                               + static_cast<int64_t>(kAttnHeads) * kHeadDim;
    const int64_t attn_dequant_sz = (attn_dequant_max + act_plus_out)
                                  * sizeof(float);
    const int64_t max_compressed = max_seq_len / 4 + 1;
    const int64_t kv_cache_sz = static_cast<int64_t>(kNumLayers)
                              * (kAttnWindow + max_compressed) * kHeadDim
                              * sizeof(__half);
    const int64_t pack_total = 3 * (static_cast<int64_t>(kIntermediateDim)
                                    * kHiddenDim / 2);
    const int64_t scale_total = 3 * (static_cast<int64_t>(kIntermediateDim)
                                     * kHiddenDim / 32);

    int64_t off = 0;
    auto page = [&](int64_t bytes_, int64_t align) -> int64_t {
        off = (off + align - 1) / align * align;
        const int64_t at = off;
        off += bytes_;
        return at;
    };
    const int64_t o_single       = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_normed       = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_ffn_out      = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_hidden       = page(static_cast<int64_t>(kHcDim) * sizeof(float), 16);
    const int64_t o_hidden_next  = page(static_cast<int64_t>(kHcDim) * sizeof(float), 16);
    const int64_t o_shared_int   = page(shared_int_sz, 16);
    const int64_t o_shared_gate  = page(shared_int_sz, 16);
    const int64_t o_routed_buf   = page(shared_int_sz, 16);
    const int64_t o_dequant      = page(dequant_sz, 16);
    const int64_t o_act_quant    = page(std::max(kHiddenDim, kIntermediateDim) * sizeof(float), 16);
    const int64_t o_ffn_accum    = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_swiglu       = page(static_cast<int64_t>(kIntermediateDim) * sizeof(float), 16);
    const int64_t o_act_scales   = page(((std::max(kHiddenDim, kIntermediateDim) + 127) / 128) * sizeof(float), 16);
    const int64_t o_gate_f32     = page(static_cast<int64_t>(kIntermediateDim) * sizeof(float), 16);
    const int64_t o_up_f32       = page(static_cast<int64_t>(kIntermediateDim) * sizeof(float), 16);
    const int64_t o_attn_dequant = page(attn_dequant_sz, 16);
    const int64_t o_kv_cache     = page(kv_cache_sz, 16);
    const int64_t o_q            = page(static_cast<int64_t>(kAttnHeads) * kHeadDim * sizeof(__half), 16);
    const int64_t o_qr           = page(static_cast<int64_t>(kQloraRank) * sizeof(__half), 16);
    const int64_t o_qr2          = page(static_cast<int64_t>(kQloraRank) * sizeof(__half), 16);
    const int64_t o_kv           = page(static_cast<int64_t>(kHeadDim) * sizeof(__half), 16);
    const int64_t o_kv2          = page(static_cast<int64_t>(kHeadDim) * sizeof(__half), 16);
    const int64_t o_o            = page(static_cast<int64_t>(kAttnHeads) * kHeadDim * sizeof(__half), 16);
    const int64_t o_olora        = page(static_cast<int64_t>(kOGroups) * kOloraRank * sizeof(__half), 16);
    const int64_t o_attn_out_h   = page(static_cast<int64_t>(kHiddenDim) * sizeof(__half), 16);
    const int64_t o_attn_out     = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_hc_post      = page(static_cast<int64_t>(kHcMult) * sizeof(float), 16);
    const int64_t o_hc_comb      = page(static_cast<int64_t>(kHcMult) * kHcMult * sizeof(float), 16);
    const int64_t o_hc_pre       = page(static_cast<int64_t>(kHcMult) * sizeof(float), 16);
    const int64_t o_logits       = page(static_cast<int64_t>(vocab) * sizeof(float), 16);
    const int64_t o_norm_scratch = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_norm_gamma   = page(static_cast<int64_t>(kHiddenDim) * sizeof(float), 16);
    const int64_t o_next_token   = page(sizeof(int32_t), 16);
    const int64_t o_next_maxv    = page(sizeof(float), 16);
    const int64_t o_pack_stage   = page(pack_total, 16);
    const int64_t o_scale_stage  = page(scale_total, 16);
    const int64_t o_routing      = page(static_cast<int64_t>(kNumRouted) * sizeof(__half), 16);

    // Per-layer CSA region (exact alloc_csa_state sizes; ratio<=0 skipped).
    const int64_t o_csa = off;
    struct CsaOffsets {
        int64_t kv_state, score_state, compressed_kv;
        int64_t idx_kv_state, idx_score_state, idx_compressed_kv;
    };
    std::vector<CsaOffsets> csa_off(num_layers > 0 ? num_layers : 0);
    for (int l = 0; l < num_layers; ++l) {
        const int ratio = compress_ratios ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        const int coff = (ratio == 4) ? 2 : 1;
        const int state_sz = coff * ratio * coff * kHeadDim;
        const int max_comp = max_seq_len / ratio + 1;
        const int idx_state_sz = 2 * 4 * 2 * static_cast<int>(kIdxHeadDim);
        csa_off[static_cast<size_t>(l)].kv_state =
            page(static_cast<int64_t>(state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].score_state =
            page(static_cast<int64_t>(state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].compressed_kv =
            page(static_cast<int64_t>(max_comp) * kHeadDim * sizeof(__half), 16);
        csa_off[static_cast<size_t>(l)].idx_kv_state =
            page(static_cast<int64_t>(idx_state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].idx_score_state =
            page(static_cast<int64_t>(idx_state_sz) * sizeof(float), 16);
        csa_off[static_cast<size_t>(l)].idx_compressed_kv =
            page(static_cast<int64_t>(max_comp) * kIdxHeadDim * sizeof(__half), 16);
    }
    bytes = static_cast<size_t>(off);

    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_base), bytes);
    if (e != cudaSuccess) { bytes = 0; Free(); return e; }

    // Pinned host block: top-k result slots + route_token logits staging.
    int64_t hoff = 0;
    auto hpage = [&](int64_t bytes_, int64_t align) -> int64_t {
        hoff = (hoff + align - 1) / align * align;
        const int64_t at = hoff;
        hoff += bytes_;
        return at;
    };
    const int64_t ho_topk    = hpage(static_cast<int64_t>(kTopK) * sizeof(int), 16);
    const int64_t ho_topw    = hpage(static_cast<int64_t>(kTopK) * sizeof(float), 16);
    const int64_t ho_routing = hpage(static_cast<int64_t>(kNumRouted) * sizeof(float), 16);
    host_bytes = static_cast<size_t>(hoff);

    e = cudaMallocHost(reinterpret_cast<void**>(&h_base), host_bytes);
    if (e != cudaSuccess) { host_bytes = 0; Free(); return e; }

    InferenceAllocCount().fetch_add(2);

    // Bind device slices.
    d_single       = reinterpret_cast<__half*>(d_base + o_single);
    d_normed       = reinterpret_cast<__half*>(d_base + o_normed);
    d_ffn_out      = reinterpret_cast<__half*>(d_base + o_ffn_out);
    d_hidden       = reinterpret_cast<float*>(d_base + o_hidden);
    d_hidden_next  = reinterpret_cast<float*>(d_base + o_hidden_next);
    d_shared_int   = reinterpret_cast<__half*>(d_base + o_shared_int);
    d_shared_gate  = reinterpret_cast<__half*>(d_base + o_shared_gate);
    d_routed_buf   = reinterpret_cast<__half*>(d_base + o_routed_buf);
    d_dequant_scratch = reinterpret_cast<float*>(d_base + o_dequant);
    d_act_quant    = reinterpret_cast<float*>(d_base + o_act_quant);
    d_ffn_accum    = reinterpret_cast<float*>(d_base + o_ffn_accum);
    d_swiglu_out   = reinterpret_cast<float*>(d_base + o_swiglu);
    d_act_scales   = reinterpret_cast<float*>(d_base + o_act_scales);
    d_gate_f32     = reinterpret_cast<float*>(d_base + o_gate_f32);
    d_up_f32       = reinterpret_cast<float*>(d_base + o_up_f32);
    d_attn_dequant = reinterpret_cast<float*>(d_base + o_attn_dequant);
    d_kv_cache     = reinterpret_cast<__half*>(d_base + o_kv_cache);
    d_q            = reinterpret_cast<__half*>(d_base + o_q);
    d_qr           = reinterpret_cast<__half*>(d_base + o_qr);
    d_qr2          = reinterpret_cast<__half*>(d_base + o_qr2);
    d_kv           = reinterpret_cast<__half*>(d_base + o_kv);
    d_kv2          = reinterpret_cast<__half*>(d_base + o_kv2);
    d_o            = reinterpret_cast<__half*>(d_base + o_o);
    d_olora        = reinterpret_cast<__half*>(d_base + o_olora);
    d_attn_out_h   = reinterpret_cast<__half*>(d_base + o_attn_out_h);
    d_attn_out     = reinterpret_cast<float*>(d_base + o_attn_out);
    d_hc_post      = reinterpret_cast<float*>(d_base + o_hc_post);
    d_hc_comb      = reinterpret_cast<float*>(d_base + o_hc_comb);
    d_hc_pre       = reinterpret_cast<float*>(d_base + o_hc_pre);
    d_logits       = reinterpret_cast<float*>(d_base + o_logits);
    d_norm_scratch = reinterpret_cast<float*>(d_base + o_norm_scratch);
    d_norm_gamma_scratch = reinterpret_cast<float*>(d_base + o_norm_gamma);
    d_next_token   = reinterpret_cast<int32_t*>(d_base + o_next_token);
    d_next_maxv    = reinterpret_cast<float*>(d_base + o_next_maxv);
    d_pack_stage   = d_base + o_pack_stage;
    d_scale_stage  = d_base + o_scale_stage;
    d_routing_scratch = reinterpret_cast<__half*>(d_base + o_routing);

    // Bind pinned slices.
    h_topk          = reinterpret_cast<int*>(h_base + ho_topk);
    h_top_weights   = reinterpret_cast<float*>(h_base + ho_topw);
    h_routing_scratch = reinterpret_cast<float*>(h_base + ho_routing);

    // Bind + init per-layer CSA state (same zero / -inf init as alloc_csa_state).
    csa_states = new CSAState[num_layers];
    for (int l = 0; l < num_layers; ++l) {
        const int ratio = compress_ratios ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        CSAState& s = csa_states[l];
        const CsaOffsets& co = csa_off[static_cast<size_t>(l)];
        s.d_comp_kv_state    = reinterpret_cast<float*>(d_base + co.kv_state);
        s.d_comp_score_state = reinterpret_cast<float*>(d_base + co.score_state);
        s.d_compressed_kv    = reinterpret_cast<__half*>(d_base + co.compressed_kv);
        s.d_idx_kv_state     = reinterpret_cast<float*>(d_base + co.idx_kv_state);
        s.d_idx_score_state  = reinterpret_cast<float*>(d_base + co.idx_score_state);
        s.d_idx_compressed_kv = reinterpret_cast<__half*>(d_base + co.idx_compressed_kv);
    }

    // CSA init (mirror alloc_csa_state): zero KV/compressed, -inf score states.
    for (int l = 0; l < num_layers; ++l) {
        const int ratio = compress_ratios ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        CSAState& s = csa_states[l];
        const int coff = (ratio == 4) ? 2 : 1;
        const int state_sz = coff * ratio * coff * kHeadDim;
        const int max_comp = max_seq_len / ratio + 1;
        const int idx_state_sz = 2 * 4 * 2 * static_cast<int>(kIdxHeadDim);

        if ((e = cudaMemsetAsync(s.d_comp_kv_state, 0,
                 static_cast<size_t>(state_sz) * sizeof(float), stream)) != cudaSuccess) goto fail;
        ds_fill_neg_inf_kernel<<<(state_sz + 255) / 256, 256, 0, stream>>>(
            s.d_comp_score_state, state_sz);
        if ((e = cudaGetLastError()) != cudaSuccess) goto fail;
        if ((e = cudaMemsetAsync(s.d_compressed_kv, 0,
                 static_cast<size_t>(max_comp) * kHeadDim * sizeof(__half), stream)) != cudaSuccess) goto fail;
        if ((e = cudaMemsetAsync(s.d_idx_kv_state, 0,
                 static_cast<size_t>(idx_state_sz) * sizeof(float), stream)) != cudaSuccess) goto fail;
        ds_fill_neg_inf_kernel<<<(idx_state_sz + 255) / 256, 256, 0, stream>>>(
            s.d_idx_score_state, idx_state_sz);
        if ((e = cudaGetLastError()) != cudaSuccess) goto fail;
        if ((e = cudaMemsetAsync(s.d_idx_compressed_kv, 0,
                 static_cast<size_t>(max_comp) * kIdxHeadDim * sizeof(__half), stream)) != cudaSuccess) goto fail;
    }
    return cudaSuccess;

fail:
    Free();
    return e;
}

void DeepSeekInferArena::Free()
{
    if (d_base) cudaFree(d_base);
    if (h_base) cudaFreeHost(h_base);
    delete[] csa_states;
    d_base = nullptr; h_base = nullptr; csa_states = nullptr;
    d_single = d_normed = d_ffn_out = nullptr;
    d_hidden = d_hidden_next = nullptr;
    d_shared_int = d_shared_gate = d_routed_buf = nullptr;
    d_dequant_scratch = d_act_quant = d_ffn_accum = d_swiglu_out = nullptr;
    d_act_scales = d_gate_f32 = d_up_f32 = d_attn_dequant = nullptr;
    d_kv_cache = nullptr;
    d_q = d_qr = d_qr2 = d_kv = d_kv2 = d_o = d_olora = d_attn_out_h = nullptr;
    d_attn_out = nullptr;
    d_hc_post = d_hc_comb = d_hc_pre = nullptr;
    d_logits = d_norm_scratch = d_norm_gamma_scratch = nullptr;
    d_next_token = nullptr;
    d_next_maxv = nullptr;
    d_pack_stage = d_scale_stage = nullptr;
    d_routing_scratch = nullptr;
    h_topk = nullptr;
    h_top_weights = nullptr;
    h_routing_scratch = nullptr;
    bytes = 0; host_bytes = 0;
}

} // namespace gpu
} // namespace lsxcommon
