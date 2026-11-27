#include "deepseek_moe_v41_load_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unordered_map>

#include "deepseek_moe_v41_dequant.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace deepseek_v41 {
namespace gpu {

namespace fs = std::filesystem;
using lsxcommon::TensorRecord;

namespace {

using TensorMap = std::unordered_map<std::string, TensorRecord>;

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

const TensorRecord *Get(const TensorMap &m, const std::string &name)
{
    auto it = m.find(name);
    return it == m.end() ? nullptr : &it->second;
}

cudaError_t ReadPinnedFile(const fs::path &path, TensorMap *out)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return cudaErrorInvalidValue;
    auto recs = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!recs.ok()) return cudaErrorInvalidValue;
    for (auto &r : *recs) (*out)[r.name] = std::move(r);
    return cudaSuccess;
}

// Reusable device staging buffer. Host tensors are copied here before any
// conversion kernel touches them: passing a pageable host pointer straight to
// a kernel is only legal where the device can address host memory
// (cudaDevAttrPageableMemoryAccess), and even there it is fragile. All uses are
// on one stream, so the next copy waits for the previous kernel.
struct StageBuf {
    void *buf = nullptr;
    size_t cap = 0;
    ~StageBuf()
    {
        if (buf) cudaFree(buf);
    }
    cudaError_t Ensure(size_t n)
    {
        if (n <= cap) return cudaSuccess;
        if (buf) cudaFree(buf);
        buf = nullptr;
        cap = 0;
        if (cudaMalloc(&buf, n) != cudaSuccess) return cudaGetLastError();
        cap = n;
        return cudaSuccess;
    }
};

// BF16 [n] -> FP16 device allocation.
cudaError_t UploadBf16(const TensorRecord &t, std::vector<void *> &owned,
                       uint64_t &bytes, __half **out, cudaStream_t stream,
                       StageBuf &stage)
{
    const int64_t n = Numel(t.shape);
    if (t.dtype != "BF16" ||
        static_cast<int64_t>(t.data.size()) != n * 2)
        return cudaErrorInvalidValue;
    void *d = nullptr;
    if (cudaMalloc(&d, static_cast<size_t>(n) * 2) != cudaSuccess)
        return cudaGetLastError();
    owned.push_back(d);
    bytes += static_cast<uint64_t>(n) * 2;
    *out = static_cast<__half *>(d);
    const size_t nb = static_cast<size_t>(n) * 2;
    if (stage.Ensure(nb) != cudaSuccess) return cudaGetLastError();
    if (cudaMemcpyAsync(stage.buf, t.data.data(), nb,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return cudaGetLastError();
    return LaunchBf16ToHalf(static_cast<const uint8_t *>(stage.buf),
                            static_cast<uint16_t *>(d), n, stream);
}

// F32 passthrough.
cudaError_t UploadF32(const TensorRecord &t, std::vector<void *> &owned,
                      uint64_t &bytes, float **out, cudaStream_t stream)
{
    const int64_t n = Numel(t.shape);
    if (t.dtype != "F32" ||
        static_cast<int64_t>(t.data.size()) != n * 4)
        return cudaErrorInvalidValue;
    void *d = nullptr;
    if (cudaMalloc(&d, static_cast<size_t>(n) * 4) != cudaSuccess)
        return cudaGetLastError();
    owned.push_back(d);
    bytes += static_cast<uint64_t>(n) * 4;
    *out = static_cast<float *>(d);
    return cudaMemcpyAsync(d, t.data.data(), static_cast<size_t>(n) * 4,
                           cudaMemcpyHostToDevice, stream);
}

// F8_E4M3 weight + UE8M0 per-32 scale -> raw device FP8 bytes + scale bytes.
// No dequant: the GEMV reads the FP8 weight and dequantizes on the fly (halves
// the persistent bytes and the HBM traffic). The final stream sync in
// load_pinned_weights keeps the pageable host source alive until the async
// copy completes.
cudaError_t UploadFp8(const TensorRecord &w, const TensorRecord &sc,
                      std::vector<void *> &owned, uint64_t &bytes,
                      uint8_t **out, uint8_t **scale_out, int *rows_out,
                      int *cols_out, cudaStream_t stream)
{
    if (w.dtype != "F8_E4M3" || w.shape.size() != 2 ||
        sc.dtype != "F8_E8M0" || sc.shape.size() != 2)
        return cudaErrorInvalidValue;
    const int rows = static_cast<int>(w.shape[0]);
    const int cols = static_cast<int>(w.shape[1]);
    if (rows <= 0 || cols <= 0 || rows % 32 != 0 || cols % 32 != 0)
        return cudaErrorInvalidValue;
    if (static_cast<int64_t>(w.data.size()) !=
        static_cast<int64_t>(rows) * cols)
        return cudaErrorInvalidValue;
    const int64_t srows = (rows + 31) / 32, scols = (cols + 31) / 32;
    if (static_cast<int64_t>(sc.data.size()) < srows * scols)
        return cudaErrorInvalidValue;

    const size_t wb = static_cast<size_t>(rows) * cols;
    const size_t sb = static_cast<size_t>(srows) * scols;
    void *d = nullptr, *ds = nullptr;
    if (cudaMalloc(&d, wb) != cudaSuccess) return cudaGetLastError();
    owned.push_back(d);
    if (cudaMalloc(&ds, sb) != cudaSuccess) return cudaGetLastError();
    owned.push_back(ds);
    bytes += static_cast<uint64_t>(wb) + static_cast<uint64_t>(sb);
    if (cudaMemcpyAsync(d, w.data.data(), wb, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(ds, sc.data.data(), sb, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess)
        return cudaGetLastError();
    if (out) *out = static_cast<uint8_t *>(d);
    if (scale_out) *scale_out = static_cast<uint8_t *>(ds);
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    return cudaSuccess;
}

// F8_E4M3 weight + UE8M0 per-32 scale -> FP16 dequant cache. Used for the
// shared experts, which remain FP16 residents (the GEMV consumes FP16).
cudaError_t UploadFp8ToHalf(const TensorRecord &w, const TensorRecord &sc,
                            std::vector<void *> &owned, uint64_t &bytes,
                            __half **out, cudaStream_t stream,
                            StageBuf &stage_w, StageBuf &stage_s)
{
    if (w.dtype != "F8_E4M3" || w.shape.size() != 2 ||
        sc.dtype != "F8_E8M0" || sc.shape.size() != 2)
        return cudaErrorInvalidValue;
    const int rows = static_cast<int>(w.shape[0]);
    const int cols = static_cast<int>(w.shape[1]);
    if (rows <= 0 || cols <= 0 || rows % 32 != 0 || cols % 32 != 0)
        return cudaErrorInvalidValue;
    if (static_cast<int64_t>(w.data.size()) != static_cast<int64_t>(rows) * cols)
        return cudaErrorInvalidValue;
    const int64_t srows = (rows + 31) / 32, scols = (cols + 31) / 32;
    if (static_cast<int64_t>(sc.data.size()) < srows * scols)
        return cudaErrorInvalidValue;

    void *d = nullptr;
    if (cudaMalloc(&d, static_cast<size_t>(rows) * cols * 2) != cudaSuccess)
        return cudaGetLastError();
    owned.push_back(d);
    bytes += static_cast<uint64_t>(rows) * cols * 2;
    *out = static_cast<__half *>(d);
    const size_t wb = static_cast<size_t>(rows) * cols;
    const size_t sb = static_cast<size_t>(srows) * scols;
    if (stage_w.Ensure(wb) != cudaSuccess || stage_s.Ensure(sb) != cudaSuccess)
        return cudaGetLastError();
    if (cudaMemcpyAsync(stage_w.buf, w.data.data(), wb,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(stage_s.buf, sc.data.data(), sb,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return cudaGetLastError();
    return LaunchFp8BlockDequantToHalf(
        static_cast<const uint8_t *>(stage_w.buf),
        static_cast<const uint8_t *>(stage_s.buf), static_cast<uint16_t *>(d),
        rows, cols, stream);
}

#define V41_REQUIRE(expr)                                   \
    do {                                                    \
        if ((expr) != cudaSuccess) return cudaErrorInvalidValue; \
    } while (0)

} // namespace

void LayerWeights::Free()
{
    for (void *p : owned)
        if (p) cudaFree(p);
    owned.clear();
}

void PinnedWeights::Free()
{
    for (void *p : owned)
        if (p) cudaFree(p);
    owned.clear();
    for (auto &l : layers) l.Free();
    layers.clear();
    d_embed = nullptr;
    d_head = nullptr;
    d_final_norm = nullptr;
    resident_bytes = 0;
}

cudaError_t load_pinned_weights(const std::string &model_path,
                                PinnedWeights &pw, cudaStream_t stream)
{
    pw.Free();
    const ModelTopology topo = LoadTopology(model_path);
    if (!topo.error_msg.empty()) return cudaErrorInvalidValue;

    const fs::path pinned = fs::path(model_path) / "pinned";

    TensorMap globals;
    if (ReadPinnedFile(pinned / "global.arrow", &globals) != cudaSuccess)
        return cudaErrorInvalidValue;

    const auto *embed = Get(globals, "embed.weight");
    const auto *head = Get(globals, "head.weight");
    const auto *fnorm = Get(globals, "norm.weight");
    if (!embed || !head || !fnorm) return cudaErrorInvalidValue;
    if (embed->shape.size() != 2) return cudaErrorInvalidValue;
    pw.vocab_size = static_cast<int>(embed->shape[0]);
    pw.hidden = static_cast<int>(embed->shape[1]);

    StageBuf stage_bf16, stage_w, stage_s;
    V41_REQUIRE(UploadBf16(*embed, pw.owned, pw.resident_bytes, &pw.d_embed,
                           stream, stage_bf16));
    V41_REQUIRE(UploadBf16(*head, pw.owned, pw.resident_bytes, &pw.d_head,
                           stream, stage_bf16));
    V41_REQUIRE(UploadBf16(*fnorm, pw.owned, pw.resident_bytes,
                           &pw.d_final_norm, stream, stage_bf16));
    // Drain the stream before `globals` (and its vectors) goes out of scope.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return cudaErrorInvalidValue;

    const int num_layers = topo.num_layers;
    pw.layers.resize(static_cast<size_t>(num_layers));

    for (int l = 0; l < num_layers; ++l) {
        LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        TensorMap m;
        char layer_name[32];
        std::snprintf(layer_name, sizeof(layer_name), "layer_%03d.arrow", l);
        const auto layer_path = pinned / layer_name;
        if (ReadPinnedFile(layer_path, &m) != cudaSuccess)
            return cudaErrorInvalidValue;
        const std::string p = "layers." + std::to_string(l) + ".";
        const LayerMeta &meta = topo.layers[static_cast<size_t>(l)];
        lw.meta = meta;
        auto req = [&](const char *suffix) -> const TensorRecord * {
            return Get(m, p + suffix);
        };

        auto bf16 = [&](const char *suffix, __half **dst, bool required) {
            const auto *t = req(suffix);
            if (!t) {
                if (!required) { *dst = nullptr; return cudaSuccess; }
                return cudaErrorInvalidValue;
            }
            return UploadBf16(*t, lw.owned, pw.resident_bytes, dst, stream,
                              stage_bf16);
        };
        auto f32 = [&](const char *suffix, float **dst, bool required) {
            const auto *t = req(suffix);
            if (!t) {
                if (!required) { *dst = nullptr; return cudaSuccess; }
                return cudaErrorInvalidValue;
            }
            return UploadF32(*t, lw.owned, pw.resident_bytes, dst, stream);
        };
        auto fp8 = [&](const char *w_suffix, uint8_t **dst, uint8_t **scale,
                       int *rows, int *cols) {
            const auto *w = req(w_suffix);
            std::string sc_name(w_suffix);
            sc_name.replace(sc_name.size() - 7, 7, ".scale");
            const auto *sc = req(sc_name.c_str());
            if (!w || !sc) return cudaErrorInvalidValue;
            return UploadFp8(*w, *sc, lw.owned, pw.resident_bytes, dst, scale,
                             rows, cols, stream);
        };
        auto fp8half = [&](const char *w_suffix, __half **dst) {
            const auto *w = req(w_suffix);
            std::string sc_name(w_suffix);
            sc_name.replace(sc_name.size() - 7, 7, ".scale");
            const auto *sc = req(sc_name.c_str());
            if (!w || !sc) return cudaErrorInvalidValue;
            return UploadFp8ToHalf(*w, *sc, lw.owned, pw.resident_bytes, dst,
                                   stream, stage_w, stage_s);
        };

        V41_REQUIRE(bf16("attn_norm.weight", &lw.d_attn_norm, true));
        V41_REQUIRE(bf16("ffn_norm.weight", &lw.d_ffn_norm, true));
        V41_REQUIRE(bf16("attn.q_norm.weight", &lw.d_q_norm, true));
        V41_REQUIRE(bf16("attn.kv_norm.weight", &lw.d_kv_norm, true));
        V41_REQUIRE(f32("attn.attn_sink", &lw.d_attn_sink, true));

        V41_REQUIRE(fp8("attn.wq_a.weight", &lw.d_wq_a, &lw.d_wq_a_scale,
                        &lw.wq_a_rows, &lw.wq_a_cols));
        V41_REQUIRE(fp8("attn.wq_b.weight", &lw.d_wq_b, &lw.d_wq_b_scale,
                        &lw.wq_b_rows, &lw.wq_b_cols));
        V41_REQUIRE(fp8("attn.wkv.weight", &lw.d_wkv, &lw.d_wkv_scale,
                        &lw.wkv_rows, &lw.wkv_cols));
        V41_REQUIRE(fp8("attn.wo_a.weight", &lw.d_wo_a, &lw.d_wo_a_scale,
                        &lw.wo_a_rows, &lw.wo_a_cols));
        V41_REQUIRE(fp8("attn.wo_b.weight", &lw.d_wo_b, &lw.d_wo_b_scale,
                        &lw.wo_b_rows, &lw.wo_b_cols));

        V41_REQUIRE(bf16("ffn.gate.weight", &lw.d_gate_weight, true));
        // The router launcher needs [n_routed, hidden]; UploadBf16 does not
        // record shapes, so read them from the source record.
        {
            const auto *g = req("ffn.gate.weight");
            if (!g || g->shape.size() != 2) return cudaErrorInvalidValue;
            lw.gate_rows = static_cast<int>(g->shape[0]);
            lw.gate_cols = static_cast<int>(g->shape[1]);
        }
        V41_REQUIRE(f32("ffn.gate.bias", &lw.d_gate_bias, true));
        V41_REQUIRE(f32("ffn.gate.bias_vl", &lw.d_gate_bias_vl, false));

        V41_REQUIRE(fp8half("ffn.shared_experts.w1.weight", &lw.d_shared_w1));
        V41_REQUIRE(fp8half("ffn.shared_experts.w2.weight", &lw.d_shared_w2));
        V41_REQUIRE(fp8half("ffn.shared_experts.w3.weight", &lw.d_shared_w3));

        V41_REQUIRE(f32("hc_attn_fn", &lw.d_hc_attn_fn, true));
        V41_REQUIRE(f32("hc_attn_base", &lw.d_hc_attn_base, true));
        V41_REQUIRE(f32("hc_attn_scale", &lw.d_hc_attn_scale, true));
        V41_REQUIRE(f32("hc_ffn_fn", &lw.d_hc_ffn_fn, true));
        V41_REQUIRE(f32("hc_ffn_base", &lw.d_hc_ffn_base, true));
        V41_REQUIRE(f32("hc_ffn_scale", &lw.d_hc_ffn_scale, true));

        if (meta.is_kv_source) {
            V41_REQUIRE(bf16("attn.compressor.wkv.weight", &lw.d_comp_wkv,
                             true));
            V41_REQUIRE(bf16("attn.compressor.norm.weight", &lw.d_comp_norm,
                             true));
            if (meta.compress_ratio > 1)
                V41_REQUIRE(bf16("attn.compressor.wgate.weight",
                                 &lw.d_comp_wgate, true));
        }
        if (meta.is_index_source) {
            V41_REQUIRE(fp8("attn.indexer.wq_b.weight", &lw.d_idx_wq_b,
                            &lw.d_idx_wq_b_scale, nullptr, nullptr));
            V41_REQUIRE(bf16("attn.indexer.weights_proj.weight",
                             &lw.d_idx_weights_proj, true));
            if (meta.is_kv_source) {
                V41_REQUIRE(bf16("attn.indexer.wk.weight", &lw.d_idx_wk,
                                 true));
                V41_REQUIRE(bf16("attn.indexer.k_norm.weight",
                                 &lw.d_idx_k_norm, true));
            }
        }
        // Drain before `m` (the host source vectors) leaves scope: the dequant
        // kernels and H2D copies read it asynchronously.
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return cudaErrorInvalidValue;
    }

    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "[v41-load] pinned dense: %d layers, %llu bytes resident",
                  num_layers,
                  static_cast<unsigned long long>(pw.resident_bytes));
    lsxcommon::log::info(msg);
    return cudaSuccess;
}

ExpertDb load_expert_index(const std::string &model_path, int num_layers,
                           int experts_per_layer)
{
    ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.max_open = 65536;
    db.log_prefix = "[v41-expert] ";
    db.Load(model_path, /*fallback=*/false);
    // Pin the whole routed store into locked host memory (V4-Flash / Kimi /
    // GLM / Gemma parity). The per-expert staging copy then DMAs straight out
    // of pinned memory instead of the driver's pageable staging path, and it
    // can overlap compute once double-buffered. With 768 GiB host RAM the full
    // 285 GiB store fits; the mmap views are released lazily. Disable with
    // LSX_EXPERT_PIN=0 (falls back to mmap views, byte-identical results).
    if (std::getenv("LSX_EXPERT_PIN") == nullptr ||
        std::getenv("LSX_EXPERT_PIN")[0] != '0') {
        if (!db.PinChunked<ExpertDb::PinLayout::kStream>())
            lsxcommon::log::warn(
                "[v41-expert] pinned store unavailable; using mmap views");
    }
    return db;
}

void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_expert_index(ExpertDb &db) { db.Free(); }

} // namespace gpu
} // namespace deepseek_v41
} // namespace lsxcommon
