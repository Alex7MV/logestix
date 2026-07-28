#include "lsxcommon/testmodel_gpu.h"
#include "lsxcommon/schema.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <arrow/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

namespace {

#define CUDA_CHECK(call, result)                         \
    do {                                                 \
        cudaError_t e = call;                            \
        if (e != cudaSuccess) {                          \
            result.error_msg = std::string(#call)         \
                + " failed: " + cudaGetErrorString(e);   \
            result.ok = false;                           \
            goto cleanup;                                \
        }                                                \
    } while (0)

#define CUBLAS_CHECK(call, result)                       \
    do {                                                 \
        cublasStatus_t s = call;                         \
        if (s != CUBLAS_STATUS_SUCCESS) {                \
            result.error_msg = std::string(#call)         \
                + " failed: code="                        \
                + std::to_string(static_cast<int>(s));   \
            result.ok = false;                           \
            goto cleanup;                                \
        }                                                \
    } while (0)

constexpr int kHiddenDim = 7168;
constexpr int kNumPinnedSlots = 2;
constexpr int kDefaultTokenCount = 256;

__global__ void init_hidden_states_kernel(float *d_hidden, int dim, uint64_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    uint64_t h = seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    d_hidden[i] = (static_cast<float>(h & 0xFFFF) / 65536.0f) * 2.0f - 1.0f;
}

inline int moe_route(int token_idx, int layer_idx, int experts_per_layer)
{
    unsigned h = static_cast<unsigned>(token_idx) * 2654435761U
               + static_cast<unsigned>(layer_idx) * 314159265U;
    h ^= h >> 16; h *= 0x85ebca6bU;
    h ^= h >> 13; h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return static_cast<int>(h % static_cast<unsigned>(experts_per_layer));
}

struct ExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;
    const uint8_t *host_w3 = nullptr;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
};

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    std::string error_msg;
};

ModelTopology load_topology(const std::string &model_path)
{
    ModelTopology topo;
    auto ipath = std::filesystem::path(model_path) / "model.safetensors.index.json";
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

struct ExpertDb {
    std::vector<ExpertEntry> entries;
    std::vector<std::shared_ptr<arrow::io::MemoryMappedFile>> mmaps;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::string error_msg;
};

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
                / ("experts/layer_" + std::to_string(layer))
                / ("exp_" + std::to_string(exp) + ".arrow");

            auto mmap_res = arrow::io::MemoryMappedFile::Open(
                apath.string(), arrow::io::FileMode::READ);
            if (!mmap_res.ok()) {
                db.error_msg = "Cannot mmap " + apath.string();
                return db;
            }
            auto mmap = *mmap_res;

            auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
            if (!reader_res.ok() || (*reader_res)->num_record_batches() == 0) {
                db.error_msg = "Cannot read batch " + apath.string();
                return db;
            }
            auto batch_res = (*reader_res)->ReadRecordBatch(0);
            if (!batch_res.ok()) {
                db.error_msg = "Cannot read batch 0 " + apath.string();
                return db;
            }
            auto batch = *batch_res;

            auto get_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
                auto arr = batch->column(col);
                auto bin = std::dynamic_pointer_cast<arrow::BinaryArray>(arr);
                if (!bin) return {nullptr, 0};
                auto view = bin->GetView(0);
                return {reinterpret_cast<const uint8_t *>(view.data()),
                        static_cast<int64_t>(view.size())};
            };

            auto [w1p, w1s] = get_bin(ExpertSchema::kIdxW1Weight);
            auto [w2p, w2s] = get_bin(ExpertSchema::kIdxW2Weight);
            auto [w3p, w3s] = get_bin(ExpertSchema::kIdxW3Weight);

            db.entries[idx] = {w1p, w2p, w3p, w1s, w2s, w3s};
            db.mmaps.push_back(std::move(mmap));
            db.batches.push_back(std::move(batch));
        }
    }
    return db;
}

} // anonymous namespace

TestModelResult RunTestModel(const std::string &model_path, int num_tokens)
{
    TestModelResult result;
    if (num_tokens <= 0) num_tokens = kDefaultTokenCount;

    uint8_t *h_pinned_stage[kNumPinnedSlots] = {nullptr, nullptr};
    float *d_hidden = nullptr;
    float *d_intermediate = nullptr;
    float *d_gate = nullptr;
    uint8_t *d_w1_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w2_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w3_slot[kNumPinnedSlots] = {nullptr, nullptr};
    cudaStream_t stream_A = nullptr, stream_B = nullptr;
    cudaEvent_t dma_done[kNumPinnedSlots]  = {nullptr, nullptr};
    cudaEvent_t comp_done[kNumPinnedSlots] = {nullptr, nullptr};
    cublasHandle_t cublas_h = nullptr;
    cudaEvent_t t_start = nullptr, t_stop = nullptr;

    auto topo = load_topology(model_path);
    if (topo.num_layers == 0 || topo.experts_per_layer == 0) {
        result.error_msg = topo.error_msg.empty()
            ? "Invalid model topology" : topo.error_msg;
        return result;
    }

    auto edb = load_expert_index(model_path, topo.num_layers,
                                  topo.experts_per_layer);
    if (edb.entries.empty()) {
        result.error_msg = edb.error_msg.empty()
            ? "No experts loaded" : edb.error_msg;
        return result;
    }

    int64_t w1_bytes = edb.entries[0].w1_bytes;
    int64_t w2_bytes = edb.entries[0].w2_bytes;
    int64_t w3_bytes = edb.entries[0].w3_bytes;
    int64_t total_exp_bytes = w1_bytes + w2_bytes + w3_bytes;

    int64_t w1_el = w1_bytes * 2;
    int intermediate_dim = static_cast<int>(w1_el / kHiddenDim);
    if (intermediate_dim <= 0) {
        result.error_msg = "Invalid intermediate_dim="
            + std::to_string(intermediate_dim)
            + " from w1_bytes=" + std::to_string(w1_bytes);
        return result;
    }

    size_t stage_sz = static_cast<size_t>(total_exp_bytes);
    size_t w1_dev_sz = static_cast<size_t>(w1_bytes);
    size_t w2_dev_sz = static_cast<size_t>(w2_bytes);
    size_t w3_dev_sz = static_cast<size_t>(w3_bytes);
    std::vector<int> route_table;
    route_table.reserve(static_cast<size_t>(num_tokens) * topo.num_layers);
    for (int tok = 0; tok < num_tokens; ++tok)
        for (int lay = 0; lay < topo.num_layers; ++lay)
            route_table.push_back(moe_route(tok, lay, topo.experts_per_layer));
    int64_t total_dma_bytes = 0;
    int total_layers_processed = 0;
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < kNumPinnedSlots; ++i) {
        CUDA_CHECK(cudaMallocHost(&h_pinned_stage[i], stage_sz), result);
    }

    CUDA_CHECK(cudaMalloc(&d_hidden, static_cast<size_t>(kHiddenDim) * sizeof(float)), result);
    CUDA_CHECK(cudaMalloc(&d_intermediate, static_cast<size_t>(intermediate_dim) * sizeof(float)), result);
    CUDA_CHECK(cudaMalloc(&d_gate, static_cast<size_t>(intermediate_dim) * sizeof(float)), result);

    for (int i = 0; i < kNumPinnedSlots; ++i) {
        CUDA_CHECK(cudaMalloc(&d_w1_slot[i], w1_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w3_slot[i], w3_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w2_slot[i], w2_dev_sz), result);
    }

    init_hidden_states_kernel<<<1, kHiddenDim>>>(d_hidden, kHiddenDim, 0xdeadbeefULL);
    CUDA_CHECK(cudaGetLastError(), result);
    CUDA_CHECK(cudaDeviceSynchronize(), result);

    CUDA_CHECK(cudaStreamCreate(&stream_A), result);
    CUDA_CHECK(cudaStreamCreate(&stream_B), result);
    for (int i = 0; i < kNumPinnedSlots; ++i) {
        CUDA_CHECK(cudaEventCreate(&dma_done[i]), result);
        CUDA_CHECK(cudaEventCreate(&comp_done[i]), result);
    }
    CUBLAS_CHECK(cublasCreate(&cublas_h), result);
    CUDA_CHECK(cudaEventCreate(&t_start), result);
    CUDA_CHECK(cudaEventCreate(&t_stop), result);

    {
        int exp0 = route_table[0];
        size_t eidx = static_cast<size_t>(exp0);
        const auto &e = edb.entries[eidx];
        uint8_t *stg = h_pinned_stage[0];
        std::memcpy(stg,               e.host_w1, static_cast<size_t>(w1_bytes));
        std::memcpy(stg + w1_bytes,    e.host_w3, static_cast<size_t>(w3_bytes));
        std::memcpy(stg + w1_bytes + w3_bytes, e.host_w2, static_cast<size_t>(w2_bytes));
        cudaMemcpyAsync(d_w1_slot[0], stg, w1_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w3_slot[0], stg + w1_bytes, w3_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w2_slot[0], stg + w1_bytes + w3_bytes, w2_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        CUDA_CHECK(cudaEventRecord(dma_done[0], stream_A), result);
    }

    CUDA_CHECK(cudaEventRecord(t_start), result);

    for (int tok = 0; tok < num_tokens; ++tok) {
        for (int lay = 0; lay < topo.num_layers; ++lay) {
            int slot      = lay & 1;
            int next_slot = (lay + 1) & 1;

            cudaStream_t compute_stream = (lay & 1) ? stream_B : stream_A;
            cudaStream_t dma_stream     = (lay & 1) ? stream_A : stream_B;

            int exp_idx = route_table[static_cast<size_t>(tok * topo.num_layers + lay)];
            size_t eidx = static_cast<size_t>(exp_idx);
            const auto &entry = edb.entries[eidx];

            if (lay < topo.num_layers - 1 || tok < num_tokens - 1) {
                int next_tok = (lay == topo.num_layers - 1) ? tok + 1 : tok;
                int next_lay = (lay == topo.num_layers - 1) ? 0 : lay + 1;
                int next_exp = route_table[static_cast<size_t>(
                    next_tok * topo.num_layers + next_lay)];
                size_t next_eidx = static_cast<size_t>(next_exp);
                const auto &ne = edb.entries[next_eidx];
                uint8_t *stg = h_pinned_stage[next_slot];

                std::memcpy(stg,            ne.host_w1, static_cast<size_t>(w1_bytes));
                std::memcpy(stg + w1_bytes, ne.host_w3, static_cast<size_t>(w3_bytes));
                std::memcpy(stg + w1_bytes + w3_bytes, ne.host_w2, static_cast<size_t>(w2_bytes));

                cudaMemcpyAsync(d_w1_slot[next_slot], stg, w1_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w3_slot[next_slot], stg + w1_bytes, w3_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w2_slot[next_slot], stg + w1_bytes + w3_bytes, w2_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                CUDA_CHECK(cudaEventRecord(dma_done[next_slot], dma_stream), result);
                total_dma_bytes += total_exp_bytes;
            }

            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, dma_done[slot], 0), result);

            CUBLAS_CHECK(cublasSetStream(cublas_h, compute_stream), result);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                intermediate_dim, 1, kHiddenDim, &alpha,
                d_w1_slot[slot], CUDA_R_4F_E2M1, kHiddenDim,
                d_hidden, CUDA_R_32F, kHiddenDim, &beta,
                d_intermediate, CUDA_R_32F, intermediate_dim,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), result);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                intermediate_dim, 1, kHiddenDim, &alpha,
                d_w3_slot[slot], CUDA_R_4F_E2M1, kHiddenDim,
                d_hidden, CUDA_R_32F, kHiddenDim, &beta,
                d_gate, CUDA_R_32F, intermediate_dim,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), result);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                kHiddenDim, 1, intermediate_dim, &alpha,
                d_w2_slot[slot], CUDA_R_4F_E2M1, intermediate_dim,
                d_intermediate, CUDA_R_32F, intermediate_dim, &beta,
                d_hidden, CUDA_R_32F, kHiddenDim,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), result);

            CUDA_CHECK(cudaEventRecord(comp_done[slot], compute_stream), result);
            CUDA_CHECK(cudaStreamWaitEvent(dma_stream, comp_done[slot], 0), result);

            ++total_layers_processed;
        }
    }

    CUDA_CHECK(cudaEventRecord(t_stop), result);
    CUDA_CHECK(cudaDeviceSynchronize(), result);

    {
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop), result);
        result.total_time_ms = static_cast<double>(elapsed_ms);
        result.avg_bus_latency_ms = total_layers_processed > 0
            ? elapsed_ms / total_layers_processed : 0.0;
        result.tokens_per_sec = elapsed_ms > 0.0f
            ? (static_cast<double>(num_tokens) / elapsed_ms) * 1000.0 : 0.0;
    }

    result.num_layers = topo.num_layers;
    result.num_tokens = num_tokens;
    result.total_bytes_transferred = total_dma_bytes;
    result.ok = true;

cleanup:
    for (int i = 0; i < kNumPinnedSlots; ++i) {
        if (h_pinned_stage[i]) cudaFreeHost(h_pinned_stage[i]);
        if (d_w1_slot[i])      cudaFree(d_w1_slot[i]);
        if (d_w3_slot[i])      cudaFree(d_w3_slot[i]);
        if (d_w2_slot[i])      cudaFree(d_w2_slot[i]);
        if (dma_done[i])       cudaEventDestroy(dma_done[i]);
        if (comp_done[i])      cudaEventDestroy(comp_done[i]);
    }
    if (d_hidden)        cudaFree(d_hidden);
    if (d_intermediate)  cudaFree(d_intermediate);
    if (d_gate)          cudaFree(d_gate);
    if (stream_A)        cudaStreamDestroy(stream_A);
    if (stream_B)        cudaStreamDestroy(stream_B);
    if (t_start)         cudaEventDestroy(t_start);
    if (t_stop)          cudaEventDestroy(t_stop);
    if (cublas_h)        cublasDestroy(cublas_h);

    return result;
}

} // namespace gpu
} // namespace lsxcommon