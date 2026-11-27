#include "lsxcommon/dory_dense_load.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include <arrow/api.h>

#include <absl/strings/str_cat.h>

#include "fast_load_internal.h"
#include "lsxcommon/dory_dense_convert.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace dory {

namespace fs = std::filesystem;

void DoryLayerWeights::Free()
{
    qkv.Free();
    g.Free();
    o.Free();
    qkv_res.Free();
    g_res.Free();
    o_res.Free();
    gate_up.Free();
    down.Free();
    gate_up_res.Free();
    down_res.Free();
    cudaFree(d_sqk2);
    cudaFree(d_sgate2);
    cudaFree(d_suv_gate);
    cudaFree(d_suv_up);
    cudaFree(d_alpha);
    d_sqk2 = d_sgate2 = d_suv_gate = d_suv_up = d_alpha = nullptr;
    is_attention = false;
    loaded = false;
}

void DoryPinnedWeights::Free()
{
    cudaFree(d_embed_f16);
    d_embed_f16 = nullptr;
    cudaFree(d_head_f16);
    d_head_f16 = nullptr;
    cudaFree(d_logit_scale);
    d_logit_scale = nullptr;
    embed_rows = embed_cols = head_rows = head_cols = 0;
    logit_scale_size = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
    slot_of_layer.clear();
    kv_slots = 0;
    resident_bytes = 0;
}

namespace {

// Raw BF16 bits of a host view -> float. The nGPT vectors are 1-D and tiny.
std::vector<float> ToF32(const TensorView *t)
{
    if (t == nullptr || t->rank <= 0) return {};
    int64_t n = t->shape[0];
    for (int64_t d = 1; d < t->rank; ++d) n *= t->shape[d];
    std::vector<float> out(static_cast<size_t>(n), 0.f);
    if (t->dtype == "F32") {
        std::memcpy(out.data(), t->data, static_cast<size_t>(n) * 4);
        return out;
    }
    const uint16_t *src = reinterpret_cast<const uint16_t *>(t->data);
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(bits));
    }
    return out;
}

// Uploads one pre-scaled F32 vector to a freshly allocated device buffer.
cudaError_t UploadF32(const std::vector<float> &host, float **dst,
                      cudaStream_t stream, std::uint64_t &bytes_acc)
{
    *dst = nullptr;
    if (host.empty()) return cudaErrorInvalidValue;
    const size_t bytes = host.size() * sizeof(float);
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(dst), bytes);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(*dst, host.data(), bytes, cudaMemcpyHostToDevice,
                        stream);
    if (e != cudaSuccess) {
        cudaFree(*dst);
        *dst = nullptr;
        return e;
    }
    bytes_acc += bytes;
    return cudaSuccess;
}

cudaError_t UploadResidual(const DenseNvfp4BatchViews &nv, const char *suffix,
                           lsx_nvfp4::LsxNvfp4Weight &w, cudaStream_t stream,
                           fastload::PinnedStagingRing &ring,
                           std::uint64_t &bytes_acc)
{
    const auto *view = fastload::FindNvfp4Suffix(nv, suffix);
    if (view == nullptr) return cudaSuccess;   // legacy 4-bit-only tree
    return fastload::UploadNvfp4Weight(view, w, stream, ring, bytes_acc,
                                       nullptr);
}

cudaError_t LoadAttentionLayer(const TensorBatchViews &raw,
                               const DenseNvfp4BatchViews &nv,
                               const DoryTopology &top, DoryLayerWeights &lw,
                               cudaStream_t stream,
                               fastload::PinnedStagingRing &ring,
                               std::uint64_t &bytes_acc)
{
    const int Q = top.num_attention_heads * top.head_dim;
    const int H = top.hidden_size;

    cudaError_t e = fastload::UploadNvfp4Weight(
        fastload::FindNvfp4Suffix(nv, "mixer.qkv_proj.weight"), lw.qkv, stream,
        ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = fastload::UploadNvfp4Weight(
            fastload::FindNvfp4Suffix(nv, "mixer.g_proj.weight"), lw.g, stream,
            ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = fastload::UploadNvfp4Weight(
            fastload::FindNvfp4Suffix(nv, "mixer.o_proj.weight"), lw.o, stream,
            ring, bytes_acc, nullptr);
    if (e != cudaSuccess) return e;
    if ((e = UploadResidual(nv, "mixer.qkv_proj.weight.res", lw.qkv_res, stream,
                            ring, bytes_acc)) != cudaSuccess)
        return e;
    if ((e = UploadResidual(nv, "mixer.g_proj.weight.res", lw.g_res, stream,
                            ring, bytes_acc)) != cudaSuccess)
        return e;
    if ((e = UploadResidual(nv, "mixer.o_proj.weight.res", lw.o_res, stream,
                            ring, bytes_acc)) != cudaSuccess)
        return e;

    const auto *sqk = fastload::FindTensorSuffix(raw, "mixer.sqk");
    const auto *sgate = fastload::FindTensorSuffix(raw, "mixer.s_gate");
    const auto *alpha = fastload::FindTensorSuffix(raw, "attn_alpha");
    if (sqk == nullptr || sgate == nullptr || alpha == nullptr)
        return cudaErrorInvalidValue;
    std::vector<float> v_sqk = ToF32(sqk);
    std::vector<float> v_sgate = ToF32(sgate);
    std::vector<float> v_alpha = ToF32(alpha);
    if (static_cast<int>(v_sqk.size()) != Q ||
        static_cast<int>(v_sgate.size()) != Q ||
        static_cast<int>(v_alpha.size()) != H)
        return cudaErrorInvalidValue;
    const double sqk_factor =
        std::sqrt(static_cast<double>(top.head_dim)) / top.initializer_range;
    const double sgate_factor =
        std::sqrt(std::sqrt(static_cast<double>(top.hidden_size))) /
        top.initializer_range;
    const double alpha_scale = top.ngpt_alpha_init / top.initializer_range;
    for (auto &v : v_sqk) {
        const double s = static_cast<double>(v) * sqk_factor;
        v = static_cast<float>(s * s);
    }
    for (auto &v : v_sgate) {
        const double s = static_cast<double>(v) * sgate_factor;
        v = static_cast<float>(s * s);
    }
    for (auto &v : v_alpha)
        v = static_cast<float>(std::fabs(static_cast<double>(v) * alpha_scale));

    if ((e = UploadF32(v_sqk, &lw.d_sqk2, stream, bytes_acc)) != cudaSuccess)
        return e;
    if ((e = UploadF32(v_sgate, &lw.d_sgate2, stream, bytes_acc)) != cudaSuccess)
        return e;
    if ((e = UploadF32(v_alpha, &lw.d_alpha, stream, bytes_acc)) != cudaSuccess)
        return e;

    lw.is_attention = true;
    lw.loaded = lw.qkv.d_packed != nullptr && lw.g.d_packed != nullptr &&
                lw.o.d_packed != nullptr;
    if (!lw.loaded)
        log::error("[dory_load] an attention mixer is incomplete");
    return cudaSuccess;
}

cudaError_t LoadMlpLayer(const TensorBatchViews &raw,
                         const DenseNvfp4BatchViews &nv,
                         const DoryTopology &top, DoryLayerWeights &lw,
                         cudaStream_t stream, fastload::PinnedStagingRing &ring,
                         std::uint64_t &bytes_acc)
{
    const int I = top.intermediate_size;
    const int H = top.hidden_size;

    cudaError_t e = fastload::UploadNvfp4Weight(
        fastload::FindNvfp4Suffix(nv, "mixer.gate_up_proj.weight"), lw.gate_up,
        stream, ring, bytes_acc, nullptr);
    if (e == cudaSuccess)
        e = fastload::UploadNvfp4Weight(
            fastload::FindNvfp4Suffix(nv, "mixer.down_proj.weight"), lw.down,
            stream, ring, bytes_acc, nullptr);
    if (e != cudaSuccess) return e;
    if ((e = UploadResidual(nv, "mixer.gate_up_proj.weight.res", lw.gate_up_res,
                            stream, ring, bytes_acc)) != cudaSuccess)
        return e;
    if ((e = UploadResidual(nv, "mixer.down_proj.weight.res", lw.down_res,
                            stream, ring, bytes_acc)) != cudaSuccess)
        return e;

    const auto *suv_gate = fastload::FindTensorSuffix(raw, "mixer.suv_gate");
    const auto *suv_up = fastload::FindTensorSuffix(raw, "mixer.suv_up");
    const auto *alpha = fastload::FindTensorSuffix(raw, "mlp_alpha");
    if (suv_gate == nullptr || suv_up == nullptr || alpha == nullptr)
        return cudaErrorInvalidValue;
    std::vector<float> v_gate = ToF32(suv_gate);
    std::vector<float> v_up = ToF32(suv_up);
    std::vector<float> v_alpha = ToF32(alpha);
    if (static_cast<int>(v_gate.size()) != I ||
        static_cast<int>(v_up.size()) != I ||
        static_cast<int>(v_alpha.size()) != H)
        return cudaErrorInvalidValue;
    const double suv_factor = std::sqrt(static_cast<double>(top.hidden_size));
    const double alpha_scale = top.ngpt_alpha_init / top.initializer_range;
    for (auto &v : v_gate) v = static_cast<float>(v * suv_factor);
    for (auto &v : v_up) v = static_cast<float>(v * suv_factor);
    for (auto &v : v_alpha)
        v = static_cast<float>(std::fabs(static_cast<double>(v) * alpha_scale));

    if ((e = UploadF32(v_gate, &lw.d_suv_gate, stream, bytes_acc)) !=
        cudaSuccess)
        return e;
    if ((e = UploadF32(v_up, &lw.d_suv_up, stream, bytes_acc)) != cudaSuccess)
        return e;
    if ((e = UploadF32(v_alpha, &lw.d_alpha, stream, bytes_acc)) != cudaSuccess)
        return e;

    lw.is_attention = false;
    lw.loaded = lw.gate_up.d_packed != nullptr && lw.down.d_packed != nullptr;
    if (!lw.loaded) log::error("[dory_load] an mlp mixer is incomplete");
    return cudaSuccess;
}

} // namespace

bool LoadDoryPinnedWeights(const std::string &output_root,
                           DoryPinnedWeights &pw, cudaStream_t stream)
{
    pw.Free();
    const auto t0 = std::chrono::steady_clock::now();

    fs::path pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[dory_load] no pinned dir under " + output_root);
        return false;
    }

    const auto idx = fs::path(output_root) / "model.index.json";
    if (fs::exists(idx)) {
        pw.top = load_dory_topology(idx.string());
        if (!pw.top.error_msg.empty()) {
            log::error("[dory_load] topology error in " + idx.string() + ": " +
                       pw.top.error_msg);
            return false;
        }
    }

    fastload::StreamDrain drain{stream};
    fastload::PinnedStagingRing ring;
    if (!ring.ok()) return false;
    std::uint64_t &bytes_acc = pw.resident_bytes;
    bytes_acc = 0;

    // ── Global raw: embed (FP16 edge), lm head (FP16 edge), logit scale ──
    const auto global_path = pinned_dir / "global.arrow";
    if (!fs::exists(global_path)) {
        log::error("[dory_load] global.arrow missing");
        return false;
    }
    auto raw_global = LoadTensorBatchViews(global_path);
    if (!raw_global.ok()) {
        log::error("[dory_load] global.arrow read failed: " +
                   raw_global.status().ToString());
        pw.Free();
        return false;
    }
    auto upload_edge = [&](const char *suffix, __half **dst, int &rows,
                           int &cols) -> cudaError_t {
        const auto *t = fastload::FindTensorSuffix(*raw_global, suffix);
        if (t == nullptr || t->rank < 2) return cudaErrorNotSupported;
        rows = static_cast<int>(t->shape[0]);
        cols = static_cast<int>(t->shape[1]);
        const int64_t n = static_cast<int64_t>(rows) * cols;
        cudaError_t e = cudaMalloc(reinterpret_cast<void **>(dst),
                                   static_cast<size_t>(n) * sizeof(__half));
        if (e != cudaSuccess) return e;
        if (t->dtype == "F32") {
            std::vector<__half> h(static_cast<size_t>(n));
            const float *src = reinterpret_cast<const float *>(t->data);
            for (int64_t i = 0; i < n; ++i)
                h[static_cast<size_t>(i)] = __float2half(src[i]);
            return cudaMemcpyAsync(*dst, h.data(),
                                   static_cast<size_t>(n) * sizeof(__half),
                                   cudaMemcpyHostToDevice, stream);
        }
        return ring.UploadBf16AsFp16(t->data, n, *dst, stream, bytes_acc);
    };
    cudaError_t e = upload_edge("backbone.embeddings.weight", &pw.d_embed_f16,
                               pw.embed_rows, pw.embed_cols);
    if (e == cudaSuccess)
        e = upload_edge("lm_head.weight", &pw.d_head_f16, pw.head_rows,
                        pw.head_cols);
    if (e != cudaSuccess) {
        log::error("[dory_load] edge upload failed");
        pw.Free();
        return false;
    }

    const auto *logit = fastload::FindTensorSuffix(*raw_global,
                                                   "lm_head.logit_scale");
    if (logit == nullptr) {
        log::error("[dory_load] lm_head.logit_scale missing");
        pw.Free();
        return false;
    }
    std::vector<float> scale = ToF32(logit);
    if (scale.empty() || scale.size() != static_cast<size_t>(pw.top.vocab_size)) {
        log::error("[dory_load] lm_head.logit_scale size mismatch");
        pw.Free();
        return false;
    }
    for (auto &v : scale)
        v = static_cast<float>(v / pw.top.initializer_range);
    pw.logit_scale_size = static_cast<int>(scale.size());
    if ((e = UploadF32(scale, &pw.d_logit_scale, stream, bytes_acc)) !=
        cudaSuccess) {
        pw.Free();
        return false;
    }

    // ── Per-mixer layers + KV slot map ──
    int num_layers = pw.top.num_layers;
    if (num_layers <= 0) {
        for (const auto &entry : fs::directory_iterator(pinned_dir)) {
            const auto fn = entry.path().filename().string();
            if (fn.rfind("layer_", 0) != 0) continue;
            if (fn.find("_nvfp4") != std::string::npos) continue;
            const std::string digits = fn.substr(6, fn.find('.') - 6);
            int l = 0;
            if (digits.empty()) continue;
            try {
                l = std::stoi(digits);
            } catch (...) {
                continue;
            }
            num_layers = std::max(num_layers, l + 1);
        }
    }
    if (num_layers <= 0) {
        log::error("[dory_load] no layer files found");
        pw.Free();
        return false;
    }

    pw.layers.resize(static_cast<size_t>(num_layers));
    pw.slot_of_layer.assign(static_cast<size_t>(num_layers), -1);
    int slot = 0;
    for (int l = 0; l < num_layers; ++l) {
        const bool attention = frozen::LayerIsAttention(l);
        const bool recurrent =
            l >= pw.top.input_layers &&
            l < pw.top.input_layers + pw.top.recurrent_layers;
        if (attention) {
            pw.slot_of_layer[static_cast<size_t>(l)] = slot;
            slot += recurrent ? pw.top.recurrent_loops : 1;
        }

        const auto raw_name = pinned_dir / absl::StrCat("layer_", l, ".arrow");
        const auto nv_name =
            pinned_dir / absl::StrCat("layer_", l, "_nvfp4.arrow");
        if (!fs::exists(raw_name) || !fs::exists(nv_name)) {
            log::error(absl::StrCat("[dory_load] missing layer files for ", l));
            pw.Free();
            return false;
        }
        auto raw = LoadTensorBatchViews(raw_name);
        if (!raw.ok()) {
            log::error("[dory_load] " + raw_name.string() + " read failed: " +
                       raw.status().ToString());
            pw.Free();
            return false;
        }
        auto nv = LoadDenseNvfp4BatchViews(nv_name);
        if (!nv.ok()) {
            log::error("[dory_load] " + nv_name.string() + " read failed: " +
                       nv.status().ToString());
            pw.Free();
            return false;
        }
        auto &lw = pw.layers[static_cast<size_t>(l)];
        const cudaError_t lerr =
            attention ? LoadAttentionLayer(*raw, *nv, pw.top, lw, stream, ring,
                                           bytes_acc)
                      : LoadMlpLayer(*raw, *nv, pw.top, lw, stream, ring,
                                     bytes_acc);
        if (lerr != cudaSuccess) {
            log::error(absl::StrCat("[dory_load] layer ", l,
                                    " upload failed: ",
                                    cudaGetErrorString(lerr)));
            pw.Free();
            return false;
        }
        if (!lw.loaded) {
            pw.Free();
            return false;
        }
        // The two batch views are mmap-backed and die with the scope; the
        // uploaded copies are what the decode reads.
    }
    pw.kv_slots = slot;
    pw.resident_bytes = bytes_acc;

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    log::info(absl::StrCat("[dory_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms, ", num_layers,
                           " mixers, ", pw.kv_slots, " kv slots"));
    return true;
}

} // namespace dory
} // namespace lsxcommon
