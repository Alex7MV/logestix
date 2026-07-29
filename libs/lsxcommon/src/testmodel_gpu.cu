#include "lsxcommon/testmodel_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/converter.h"

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
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

namespace {

inline void safe_copy(void *dst, const void *src, size_t n)
{
    if (src && n > 0)
        std::memcpy(dst, src, n);
    else if (n > 0)
        std::memset(dst, 0, n);
}

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
constexpr int kDefaultTokenCount = 1024;
constexpr int kLogInterval = 128;

__global__ void init_hidden_states_kernel(__half *d_hidden, int dim, uint64_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    uint64_t h = seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    float v = (static_cast<float>(h & 0xFFFF) / 65536.0f) * 2.0f - 1.0f;
    d_hidden[i] = __float2half(v);
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
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;
    const uint8_t *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
};

__device__ float nvfp4_to_float(uint8_t nibble)
{
    int sign = (nibble >> 3) & 1;
    int exp  = (nibble >> 1) & 3;
    int mant = nibble & 1;
    float v = (exp == 0)
        ? static_cast<float>(mant) * 0.5f
        : (1.0f + static_cast<float>(mant) * 0.5f) * static_cast<float>(1 << (exp - 1));
    return sign ? -v : v;
}

__device__ float e4m3_scale_to_float(uint8_t e4m3)
{
    int e4 = (e4m3 >> 3) & 0x0F;
    if (e4 == 0) return 0.0f;
    return __uint_as_float(static_cast<uint32_t>(e4 - 7 + 127) << 23);
}

__global__ void dequant_nvfp4_to_half_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float global_scale,
    __half* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int byte_idx = i >> 1;
    uint8_t nibble = (i & 1)
        ? (packed[byte_idx] >> 4)
        : (packed[byte_idx] & 0x0F);

    float bs = e4m3_scale_to_float(scales[i >> 4]);
    float f32 = nvfp4_to_float(nibble) * bs * global_scale;
    output[i] = __float2half(f32);
}

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    std::string error_msg;
};

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

} // anonymous namespace

TestModelResult RunTestModel(const std::string &model_path, int num_tokens)
{
    TestModelResult result;
    if (num_tokens <= 0) num_tokens = kDefaultTokenCount;

    uint8_t *h_pinned_stage[kNumPinnedSlots] = {nullptr, nullptr};
    __half *d_hidden = nullptr;
    __half *d_intermediate = nullptr;
    __half *d_gate = nullptr;
    uint8_t *d_w1_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w2_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w3_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w1_scale_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w2_scale_slot[kNumPinnedSlots] = {nullptr, nullptr};
    uint8_t *d_w3_scale_slot[kNumPinnedSlots] = {nullptr, nullptr};
    __half *d_w1_fp16[kNumPinnedSlots] = {nullptr, nullptr};
    __half *d_w2_fp16[kNumPinnedSlots] = {nullptr, nullptr};
    __half *d_w3_fp16[kNumPinnedSlots] = {nullptr, nullptr};
    cudaStream_t stream_A = nullptr, stream_B = nullptr;
    cudaEvent_t dma_done[kNumPinnedSlots]  = {nullptr, nullptr};
    cudaEvent_t comp_done[kNumPinnedSlots] = {nullptr, nullptr};
    cublasHandle_t cublas_h = nullptr;
    cudaEvent_t t_start = nullptr, t_stop = nullptr;

    auto topo = load_topology(model_path);
    if (topo.num_layers == 0 || topo.experts_per_layer == 0) {
        bool created = false;
        auto logestix_dir = std::filesystem::path(model_path) / "logestix";
        if (!std::filesystem::exists(logestix_dir)) {
            arrow::Status st = lsxcommon::Converter{}
                .SetInputPath(model_path)
                .SetOutputPath(logestix_dir.string())
                .SetVerbose(false)
                .Run();
            if (!st.ok()) {
                result.error_msg = "Conversion failed: " + st.ToString();
                return result;
            }
            created = true;
        }
        topo = load_topology(model_path);
        if (topo.num_layers == 0 || topo.experts_per_layer == 0) {
            result.error_msg = topo.error_msg.empty()
                ? (created
                    ? "Conversion created logestix/ but topology still empty"
                    : "Invalid model topology")
                : topo.error_msg;
            return result;
        }
    }

    auto edb = load_expert_index(model_path, topo.num_layers,
                                  topo.experts_per_layer);
    if (edb.entries.empty() || !edb.error_msg.empty()) {
        result.error_msg = edb.error_msg.empty()
            ? "No experts loaded" : edb.error_msg;
        return result;
    }

    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    for (auto &e : edb.entries) {
        if (e.w1_bytes > 0) {
            w1_bytes = e.w1_bytes;
            w2_bytes = e.w2_bytes;
            w3_bytes = e.w3_bytes;
            break;
        }
    }
    int64_t total_exp_bytes = w1_bytes + w2_bytes + w3_bytes;
    if (total_exp_bytes == 0) {
        result.error_msg = "All expert entries have zero weight data";
        return result;
    }

    int64_t w1_el = w1_bytes * 2;
    int64_t w1_fp16_bytes = w1_el * static_cast<int64_t>(sizeof(__half));
    int64_t w2_el = w2_bytes * 2;
    int64_t w2_fp16_bytes = w2_el * static_cast<int64_t>(sizeof(__half));
    int64_t w3_el = w3_bytes * 2;
    int64_t w3_fp16_bytes = w3_el * static_cast<int64_t>(sizeof(__half));
    int64_t w1_scale_bytes = w1_el / kNVFP4BlockSize;
    int64_t w2_scale_bytes = w2_el / kNVFP4BlockSize;
    int64_t w3_scale_bytes = w3_el / kNVFP4BlockSize;
    int intermediate_dim = static_cast<int>(w1_el / kHiddenDim);
    if (intermediate_dim <= 0) {
        result.error_msg = "Invalid intermediate_dim="
            + std::to_string(intermediate_dim)
            + " from w1_bytes=" + std::to_string(w1_bytes);
        return result;
    }

    size_t stage_sz = static_cast<size_t>(
        total_exp_bytes + w1_scale_bytes + w2_scale_bytes + w3_scale_bytes);
    size_t w1_dev_sz = static_cast<size_t>(w1_bytes);
    size_t w2_dev_sz = static_cast<size_t>(w2_bytes);
    size_t w3_dev_sz = static_cast<size_t>(w3_bytes);
    size_t w1_scale_dev_sz = static_cast<size_t>(w1_scale_bytes);
    size_t w2_scale_dev_sz = static_cast<size_t>(w2_scale_bytes);
    size_t w3_scale_dev_sz = static_cast<size_t>(w3_scale_bytes);
    std::vector<int> route_table;
    route_table.reserve(static_cast<size_t>(num_tokens) * topo.num_layers);
    for (int tok = 0; tok < num_tokens; ++tok)
        for (int lay = 0; lay < topo.num_layers; ++lay)
            route_table.push_back(moe_route(tok, lay, topo.experts_per_layer));
    int64_t total_dma_bytes = 0;
    int total_layers_processed = 0;
    float alpha = 1.0f, beta = 0.0f;

    int num_checkpoints = num_tokens / kLogInterval;
    std::vector<cudaEvent_t> checkpoints(static_cast<size_t>(num_checkpoints), nullptr);
    std::vector<int> checkpoint_counts(static_cast<size_t>(num_checkpoints), 0);

    for (int i = 0; i < kNumPinnedSlots; ++i) {
        CUDA_CHECK(cudaMallocHost(&h_pinned_stage[i], stage_sz), result);
    }

    CUDA_CHECK(cudaMalloc(&d_hidden, static_cast<size_t>(kHiddenDim) * sizeof(__half)), result);
    CUDA_CHECK(cudaMalloc(&d_intermediate, static_cast<size_t>(intermediate_dim) * sizeof(__half)), result);
    CUDA_CHECK(cudaMalloc(&d_gate, static_cast<size_t>(intermediate_dim) * sizeof(__half)), result);

    for (int i = 0; i < kNumPinnedSlots; ++i) {
        CUDA_CHECK(cudaMalloc(&d_w1_slot[i], w1_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w3_slot[i], w3_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w2_slot[i], w2_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w1_scale_slot[i], w1_scale_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w2_scale_slot[i], w2_scale_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w3_scale_slot[i], w3_scale_dev_sz), result);
        CUDA_CHECK(cudaMalloc(&d_w1_fp16[i], w1_fp16_bytes), result);
        CUDA_CHECK(cudaMalloc(&d_w2_fp16[i], w2_fp16_bytes), result);
        CUDA_CHECK(cudaMalloc(&d_w3_fp16[i], w3_fp16_bytes), result);
    }

    {
        constexpr int kBlockSize = 256;
        int grid = (kHiddenDim + kBlockSize - 1) / kBlockSize;
        init_hidden_states_kernel<<<grid, kBlockSize>>>(d_hidden, kHiddenDim, 0xdeadbeefULL);
    }
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
    for (int ci = 0; ci < num_checkpoints; ++ci)
        CUDA_CHECK(cudaEventCreate(&checkpoints[static_cast<size_t>(ci)]), result);

    {
        int exp0 = route_table[0];
        size_t eidx = static_cast<size_t>(exp0);
        const auto &e = edb.entries[eidx];
        uint8_t *stg = h_pinned_stage[0];
        size_t off_w3 = static_cast<size_t>(w1_bytes);
        size_t off_w2 = off_w3 + static_cast<size_t>(w3_bytes);
        size_t off_s1 = static_cast<size_t>(total_exp_bytes);
        size_t off_s3 = off_s1 + w1_scale_dev_sz;
        size_t off_s2 = off_s3 + w3_scale_dev_sz;
        safe_copy(stg,                  e.host_w1, static_cast<size_t>(w1_bytes));
        safe_copy(stg + off_w3,         e.host_w3, static_cast<size_t>(w3_bytes));
        safe_copy(stg + off_w2,         e.host_w2, static_cast<size_t>(w2_bytes));
        safe_copy(stg + off_s1, e.host_w1_scales, w1_scale_dev_sz);
        safe_copy(stg + off_s3, e.host_w3_scales, w3_scale_dev_sz);
        safe_copy(stg + off_s2, e.host_w2_scales, w2_scale_dev_sz);

        cudaMemcpyAsync(d_w1_slot[0], stg, w1_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w3_slot[0], stg + off_w3, w3_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w2_slot[0], stg + off_w2, w2_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w1_scale_slot[0], stg + off_s1, w1_scale_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w3_scale_slot[0], stg + off_s3, w3_scale_dev_sz,
                        cudaMemcpyHostToDevice, stream_A);
        cudaMemcpyAsync(d_w2_scale_slot[0], stg + off_s2, w2_scale_dev_sz,
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
                size_t off_w3 = static_cast<size_t>(w1_bytes);
                size_t off_w2 = off_w3 + static_cast<size_t>(w3_bytes);
                size_t off_s1 = static_cast<size_t>(total_exp_bytes);
                size_t off_s3 = off_s1 + w1_scale_dev_sz;
                size_t off_s2 = off_s3 + w3_scale_dev_sz;

                safe_copy(stg,                  ne.host_w1, static_cast<size_t>(w1_bytes));
                safe_copy(stg + off_w3,         ne.host_w3, static_cast<size_t>(w3_bytes));
                safe_copy(stg + off_w2,         ne.host_w2, static_cast<size_t>(w2_bytes));
                safe_copy(stg + off_s1, ne.host_w1_scales, w1_scale_dev_sz);
                safe_copy(stg + off_s3, ne.host_w3_scales, w3_scale_dev_sz);
                safe_copy(stg + off_s2, ne.host_w2_scales, w2_scale_dev_sz);

                cudaMemcpyAsync(d_w1_slot[next_slot], stg, w1_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w3_slot[next_slot], stg + off_w3, w3_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w2_slot[next_slot], stg + off_w2, w2_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w1_scale_slot[next_slot], stg + off_s1, w1_scale_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w3_scale_slot[next_slot], stg + off_s3, w3_scale_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                cudaMemcpyAsync(d_w2_scale_slot[next_slot], stg + off_s2, w2_scale_dev_sz,
                                cudaMemcpyHostToDevice, dma_stream);
                CUDA_CHECK(cudaEventRecord(dma_done[next_slot], dma_stream), result);
                total_dma_bytes += total_exp_bytes;
            }

            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, dma_done[slot], 0), result);

            {
                constexpr int kDQBlock = 256;
                const auto &e = edb.entries[eidx];
                int w1_el_i = static_cast<int>(w1_el);
                int w2_el_i = static_cast<int>(w2_el);
                int w3_el_i = static_cast<int>(w3_el);
                int dq_grid_w1 = (w1_el_i + kDQBlock - 1) / kDQBlock;
                int dq_grid_w3 = (w3_el_i + kDQBlock - 1) / kDQBlock;
                int dq_grid_w2 = (w2_el_i + kDQBlock - 1) / kDQBlock;
                if (dq_grid_w1 > 0)
                    dequant_nvfp4_to_half_kernel<<<dq_grid_w1, kDQBlock, 0, compute_stream>>>(
                        d_w1_slot[slot], d_w1_scale_slot[slot], e.w1_global_scale,
                        d_w1_fp16[slot], w1_el_i);
                if (dq_grid_w3 > 0)
                    dequant_nvfp4_to_half_kernel<<<dq_grid_w3, kDQBlock, 0, compute_stream>>>(
                        d_w3_slot[slot], d_w3_scale_slot[slot], e.w3_global_scale,
                        d_w3_fp16[slot], w3_el_i);
                if (dq_grid_w2 > 0)
                    dequant_nvfp4_to_half_kernel<<<dq_grid_w2, kDQBlock, 0, compute_stream>>>(
                        d_w2_slot[slot], d_w2_scale_slot[slot], e.w2_global_scale,
                        d_w2_fp16[slot], w2_el_i);
            }

            CUBLAS_CHECK(cublasSetStream(cublas_h, compute_stream), result);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                intermediate_dim, 1, kHiddenDim, &alpha,
                d_w1_fp16[slot], CUDA_R_16F, kHiddenDim,
                d_hidden, CUDA_R_16F, kHiddenDim, &beta,
                d_intermediate, CUDA_R_16F, intermediate_dim,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), result);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                intermediate_dim, 1, kHiddenDim, &alpha,
                d_w3_fp16[slot], CUDA_R_16F, kHiddenDim,
                d_hidden, CUDA_R_16F, kHiddenDim, &beta,
                d_gate, CUDA_R_16F, intermediate_dim,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), result);

            CUBLAS_CHECK(cublasGemmEx(
                cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                kHiddenDim, 1, intermediate_dim, &alpha,
                d_w2_fp16[slot], CUDA_R_16F, intermediate_dim,
                d_intermediate, CUDA_R_16F, intermediate_dim, &beta,
                d_hidden, CUDA_R_16F, kHiddenDim,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), result);

            CUDA_CHECK(cudaEventRecord(comp_done[slot], compute_stream), result);
            CUDA_CHECK(cudaStreamWaitEvent(dma_stream, comp_done[slot], 0), result);

            ++total_layers_processed;
        }

        int ci = tok / kLogInterval;
        int cm1 = (tok + 1) % kLogInterval;
        if (cm1 == 0 && ci < num_checkpoints) {
            cudaStream_t last_stream = ((topo.num_layers - 1) & 1) ? stream_B : stream_A;
            CUDA_CHECK(cudaEventRecord(checkpoints[static_cast<size_t>(ci)], last_stream), result);
            checkpoint_counts[static_cast<size_t>(ci)] = tok + 1;
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

        std::fprintf(stderr, "\n=== --testmodel Throughput by interval ===\n");
        for (int ci = 0; ci < num_checkpoints; ++ci) {
            float cp_ms = 0.0f;
            cudaEventElapsedTime(&cp_ms, t_start, checkpoints[static_cast<size_t>(ci)]);
            int ct = checkpoint_counts[static_cast<size_t>(ci)];
            double tps = ct / static_cast<double>(cp_ms) * 1000.0;
            std::fprintf(stderr, "  tok=%4d  elapsed=%8.1f ms  %8.1f t/s\n",
                         ct, static_cast<double>(cp_ms), tps);
        }
        std::fprintf(stderr, "  ---\n  total    elapsed=%8.1f ms  %8.1f t/s\n",
                     static_cast<double>(elapsed_ms), result.tokens_per_sec);
    }

    result.num_layers = topo.num_layers;
    result.num_tokens = num_tokens;
    result.total_bytes_transferred = total_dma_bytes;
    result.ok = true;

cleanup:
    for (int i = 0; i < kNumPinnedSlots; ++i) {
        if (h_pinned_stage[i])    cudaFreeHost(h_pinned_stage[i]);
        if (d_w1_slot[i])         cudaFree(d_w1_slot[i]);
        if (d_w3_slot[i])         cudaFree(d_w3_slot[i]);
        if (d_w2_slot[i])         cudaFree(d_w2_slot[i]);
        if (d_w1_scale_slot[i])   cudaFree(d_w1_scale_slot[i]);
        if (d_w3_scale_slot[i])   cudaFree(d_w3_scale_slot[i]);
        if (d_w2_scale_slot[i])   cudaFree(d_w2_scale_slot[i]);
        if (d_w1_fp16[i])         cudaFree(d_w1_fp16[i]);
        if (d_w2_fp16[i])         cudaFree(d_w2_fp16[i]);
        if (d_w3_fp16[i])         cudaFree(d_w3_fp16[i]);
        if (dma_done[i])          cudaEventDestroy(dma_done[i]);
        if (comp_done[i])         cudaEventDestroy(comp_done[i]);
    }
    if (d_hidden)        cudaFree(d_hidden);
    if (d_intermediate)  cudaFree(d_intermediate);
    if (d_gate)          cudaFree(d_gate);
    if (stream_A)        cudaStreamDestroy(stream_A);
    if (stream_B)        cudaStreamDestroy(stream_B);
    if (t_start)         cudaEventDestroy(t_start);
    if (t_stop)          cudaEventDestroy(t_stop);
    for (auto &cp : checkpoints)
        if (cp) cudaEventDestroy(cp);
    if (cublas_h)        cublasDestroy(cublas_h);

    return result;
}

} // namespace gpu
} // namespace lsxcommon