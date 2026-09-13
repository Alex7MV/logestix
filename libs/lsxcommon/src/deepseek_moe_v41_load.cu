#include "deepseek_moe_v41_load_internal.h"

#include <cstring>
#include <filesystem>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>

#include "deepseek_moe_v41_dequant.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace deepseek_v41 {
namespace gpu {

namespace fs = std::filesystem;
using lsxcommon::TensorRecord;

namespace {

using TensorMap = absl::flat_hash_map<std::string, TensorRecord>;

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

// BF16 [n] -> FP16 device allocation.
cudaError_t UploadBf16(const TensorRecord &t, std::vector<void *> &owned,
                       uint64_t &bytes, __half **out, cudaStream_t stream)
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
    return LaunchBf16ToHalf(t.data.data(), static_cast<uint16_t *>(d), n,
                            stream);
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

// F8_E4M3 weight + UE8M0 per-32 scale -> FP16 dequant cache.
cudaError_t UploadFp8(const TensorRecord &w, const TensorRecord &sc,
                      std::vector<void *> &owned, uint64_t &bytes,
                      __half **out, int *rows_out, int *cols_out,
                      cudaStream_t stream)
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

    void *d = nullptr;
    if (cudaMalloc(&d, static_cast<size_t>(rows) * cols * 2) != cudaSuccess)
        return cudaGetLastError();
    owned.push_back(d);
    bytes += static_cast<uint64_t>(rows) * cols * 2;
    *out = static_cast<__half *>(d);
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    return LaunchFp8BlockDequantToHalf(
        w.data.data(), sc.data.data(), static_cast<uint16_t *>(d), rows, cols,
        stream);
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

    const fs::path pinned = fs::path(model_path) / "logestix" / "pinned";

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

    V41_REQUIRE(UploadBf16(*embed, pw.owned, pw.resident_bytes, &pw.d_embed,
                           stream));
    V41_REQUIRE(UploadBf16(*head, pw.owned, pw.resident_bytes, &pw.d_head,
                           stream));
    V41_REQUIRE(UploadBf16(*fnorm, pw.owned, pw.resident_bytes,
                           &pw.d_final_norm, stream));
    // The upload kernels/copies read the host TensorRecord buffers directly;
    // drain the stream before `globals` (and its vectors) goes out of scope.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return cudaErrorInvalidValue;

    const int num_layers = topo.num_layers;
    pw.layers.resize(static_cast<size_t>(num_layers));

    for (int l = 0; l < num_layers; ++l) {
        LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        TensorMap m;
        const auto layer_path = pinned /
            (absl::StrCat("layer_", absl::StrFormat("%03d", l)) + ".arrow");
        if (ReadPinnedFile(layer_path, &m) != cudaSuccess)
            return cudaErrorInvalidValue;
        const std::string p = "layers." + std::to_string(l) + ".";
        const LayerMeta &meta = topo.layers[static_cast<size_t>(l)];
        auto req = [&](const char *suffix) -> const TensorRecord * {
            return Get(m, p + suffix);
        };

        auto bf16 = [&](const char *suffix, __half **dst, bool required) {
            const auto *t = req(suffix);
            if (!t) {
                if (!required) { *dst = nullptr; return cudaSuccess; }
                return cudaErrorInvalidValue;
            }
            return UploadBf16(*t, lw.owned, pw.resident_bytes, dst, stream);
        };
        auto f32 = [&](const char *suffix, float **dst, bool required) {
            const auto *t = req(suffix);
            if (!t) {
                if (!required) { *dst = nullptr; return cudaSuccess; }
                return cudaErrorInvalidValue;
            }
            return UploadF32(*t, lw.owned, pw.resident_bytes, dst, stream);
        };
        auto fp8 = [&](const char *w_suffix, __half **dst, int *rows,
                       int *cols) {
            const auto *w = req(w_suffix);
            std::string sc_name(w_suffix);
            sc_name.replace(sc_name.size() - 7, 7, ".scale");
            const auto *sc = req(sc_name.c_str());
            if (!w || !sc) return cudaErrorInvalidValue;
            return UploadFp8(*w, *sc, lw.owned, pw.resident_bytes, dst, rows,
                             cols, stream);
        };

        V41_REQUIRE(bf16("attn_norm.weight", &lw.d_attn_norm, true));
        V41_REQUIRE(bf16("ffn_norm.weight", &lw.d_ffn_norm, true));
        V41_REQUIRE(bf16("attn.q_norm.weight", &lw.d_q_norm, true));
        V41_REQUIRE(bf16("attn.kv_norm.weight", &lw.d_kv_norm, true));
        V41_REQUIRE(f32("attn.attn_sink", &lw.d_attn_sink, true));

        V41_REQUIRE(fp8("attn.wq_a.weight", &lw.d_wq_a, &lw.wq_a_rows,
                        &lw.wq_a_cols));
        V41_REQUIRE(fp8("attn.wq_b.weight", &lw.d_wq_b, &lw.wq_b_rows,
                        &lw.wq_b_cols));
        V41_REQUIRE(fp8("attn.wkv.weight", &lw.d_wkv, &lw.wkv_rows,
                        &lw.wkv_cols));
        V41_REQUIRE(fp8("attn.wo_a.weight", &lw.d_wo_a, &lw.wo_a_rows,
                        &lw.wo_a_cols));
        V41_REQUIRE(fp8("attn.wo_b.weight", &lw.d_wo_b, &lw.wo_b_rows,
                        &lw.wo_b_cols));

        V41_REQUIRE(bf16("ffn.gate.weight", &lw.d_gate_weight, true));
        V41_REQUIRE(f32("ffn.gate.bias", &lw.d_gate_bias, true));
        V41_REQUIRE(f32("ffn.gate.bias_vl", &lw.d_gate_bias_vl, false));

        V41_REQUIRE(fp8("ffn.shared_experts.w1.weight", &lw.d_shared_w1,
                        nullptr, nullptr));
        V41_REQUIRE(fp8("ffn.shared_experts.w2.weight", &lw.d_shared_w2,
                        nullptr, nullptr));
        V41_REQUIRE(fp8("ffn.shared_experts.w3.weight", &lw.d_shared_w3,
                        nullptr, nullptr));

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
                            nullptr, nullptr));
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

    lsxcommon::log::info(absl::StrFormat(
        "[v41-load] pinned dense: %d layers, %llu bytes resident",
        num_layers, static_cast<unsigned long long>(pw.resident_bytes)));
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
