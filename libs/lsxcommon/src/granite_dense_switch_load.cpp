#include "lsxcommon/granite_dense_switch_load.h"

#include "lsxcommon/granite_dense_switch_convert.h"
#include "lsxcommon/granite_dense_switch_inference.h"
#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

#include <arrow/api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include "fast_load_internal.h"

namespace lsxcommon {

namespace {

// detect_model_type (deepseek_moe_load.cu) folds "Granite-Switch-4.1-30B" to
// "granite-switch-4.1-30b" before it becomes a factory key, so the comparison
// here must use the same fold or the check would disagree with the factory about
// which model a tree is.
std::string FoldModelId(const std::string &raw)
{
    std::string s = raw;
    for (char &c : s) {
        if (c == ' ') c = '-';
        else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

std::string Num(double v)
{
    // 1e7 / 0.22 / 1e-5 must not print as 10000000.000000 and bury the delta.
    return absl::StrFormat("%g", v);
}

} // namespace

namespace granite_switch {

std::string GraniteSwitchGeometryMismatch(const GraniteSwitchFrozen &f,
                                          const GraniteSwitchTopology &top)
{
    std::string why;
    auto add = [&why](const std::string &line) {
        if (!why.empty()) why += "\n";
        why += "  " + line;
    };
    const std::string want = f.model_type;
    const std::string got = top.model_name.empty() ? "<absent>"
                                                   : FoldModelId(top.model_name);
    if (got != want)
        add("model identity: this class is '" + want + "', the converted tree is '" +
            got + "'");

    const double eps = 1e-9;
    auto close = [eps](double a, double b) {
        return std::abs(a - b) <= 1e-6 * std::max(1.0, std::abs(b)) + eps;
    };
    auto ints = [&add](const char *name, long long want_v, long long got_v) {
        if (want_v != got_v)
            add(absl::StrCat(name, ": expected ", want_v, ", got ", got_v));
    };
    ints("num_layers", f.num_layers, top.num_layers);
    ints("hidden_size", f.hidden_size, top.hidden_size);
    ints("num_attention_heads", f.num_attention_heads, top.num_attention_heads);
    ints("num_key_value_heads", f.num_key_value_heads, top.num_key_value_heads);
    ints("head_dim", f.head_dim, top.head_dim);
    ints("intermediate_size", f.intermediate_size, top.intermediate_size);
    ints("vocab_size", f.vocab_size, top.vocab_size);
    ints("num_adapters", f.num_adapters, top.num_adapters);
    auto dbl = [&add, &close](const char *name, double want_v, double got_v) {
        if (!close(want_v, got_v))
            add(absl::StrCat(name, ": expected ", Num(want_v), ", got ",
                             Num(got_v)));
    };
    dbl("rms_norm_eps", f.rms_norm_eps, top.rms_norm_eps);
    dbl("rope_theta", f.rope_theta, top.rope_theta);
    dbl("attention_multiplier", f.attention_multiplier,
        top.attention_multiplier);
    dbl("embedding_multiplier", f.embedding_multiplier,
        top.embedding_multiplier);
    dbl("residual_multiplier", f.residual_multiplier, top.residual_multiplier);
    dbl("logits_scaling", f.logits_scaling, top.logits_scaling);

    // Adapter metadata is per-adapter, and a mismatch here is silent at runtime:
    // the wrong rank reads a valid but different slice of the LoRA plane.
    const int n = std::min<int>(f.num_adapters,
                                static_cast<int>(top.adapter_ranks.size()));
    for (int a = 0; a < n; ++a) {
        ints("adapter_ranks", f.adapter_ranks[a], top.adapter_ranks[a]);
        ints("adapter_token_ids", f.control_token_ids[a], top.adapter_token_ids[a]);
        ints("adapter_substitute_token_ids", f.substitute_token_ids[a],
             top.adapter_substitute_token_ids[a]);
    }
    return why;
}

namespace {

using fastload::CvKind;
using fastload::FindNvfp4Suffix;
using fastload::FindTensorSuffix;
using fastload::PinnedStagingRing;
using fastload::RunUploadJobs;
using fastload::UploadJob;
using fastload::UploadNvfp4Weight;

CvKind NormKind(const TensorView *t)
{
    return (t && t->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
}

void ClearLoraAliases(GraniteSwitchProjection &p)
{
    for (auto &per_adapter : p.lora)
        for (auto &a : per_adapter) {
            a.d_A = nullptr;
            a.d_B = nullptr;
            a.rank = 0;
            a.in = 0;
            a.out = 0;
        }
}

} // namespace

void GraniteSwitchLayerWeights::Free()
{
    qkv.base.Free();
    qkv.res.Free();
    o.base.Free();
    o.res.Free();
    gate_up.base.Free();
    gate_up.res.Free();
    down.base.Free();
    down.res.Free();
    if (d_arena) {
        cudaFree(d_arena);
        d_arena = nullptr;
        arena_bytes = 0;
        d_input_norm = nullptr;
        d_post_norm = nullptr;
        for (auto *p : {&qkv, &o, &gate_up, &down}) ClearLoraAliases(*p);
        norm_size = 0;
        loaded = false;
        return;
    }
    cudaFree(d_input_norm);
    d_input_norm = nullptr;
    cudaFree(d_post_norm);
    d_post_norm = nullptr;
    for (auto *p : {&qkv, &o, &gate_up, &down}) ClearLoraAliases(*p);
    norm_size = 0;
    loaded = false;
}

void GraniteSwitchPinnedWeights::Free()
{
    head.Free();  // non-owning view
    embed.Free();
    cudaFree(d_final_norm);
    d_final_norm = nullptr;
    final_norm_size = 0;
    for (auto &l : layers) l.Free();
    layers.clear();
    resident_bytes = 0;
}

namespace {

bool LoadLayer(const TensorBatchViews &raw, const TensorBatchViews *adapters,
               const DenseNvfp4BatchViews &nv,
               const GraniteSwitchTopology &top, GraniteSwitchLayerWeights &lw,
               cudaStream_t stream, PinnedStagingRing &ring,
               std::uint64_t &bytes_acc)
{
    auto *in_norm = FindTensorSuffix(raw, "input_layernorm.weight");
    auto *post_norm = FindTensorSuffix(raw, "post_attention_layernorm.weight");
    if (in_norm && in_norm->rank > 0)
        lw.norm_size = static_cast<int>(in_norm->shape[0]);

    struct Site {
        GraniteSwitchProjection *proj;
        const char *a_suffix;
        const char *b_suffix;
        int slices;
    };
    Site sites[] = {
        {&lw.qkv, "qkv_proj.lora_A_slices.", "qkv_proj.lora_B_slices.", 3},
        {&lw.o, "o_proj.lora_A", "o_proj.lora_B", 1},
        {&lw.gate_up, "input_linear.lora_A_slices.",
         "input_linear.lora_B_slices.", 2},
        {&lw.down, "output_linear.lora_A", "output_linear.lora_B", 1},
    };
    for (auto &site : sites) {
        site.proj->lora.assign(static_cast<size_t>(top.num_adapters),
                               std::vector<GraniteSwitchLora>(
                                   static_cast<size_t>(site.slices)));
    }

    std::vector<UploadJob> jobs = {
        {in_norm, reinterpret_cast<void **>(&lw.d_input_norm),
         NormKind(in_norm)},
        {post_norm, reinterpret_cast<void **>(&lw.d_post_norm),
         NormKind(post_norm)},
    };

    if (adapters) {
        for (auto &site : sites) {
            for (int a = 0; a < top.num_adapters; ++a) {
                for (int s = 0; s < site.slices; ++s) {
                    const std::string slice =
                        site.slices > 1 ? std::to_string(s) : std::string();
                    const std::string a_name = site.a_suffix + slice +
                                               ".adapter" +
                                               std::to_string(a);
                    const std::string b_name = site.b_suffix + slice +
                                               ".adapter" +
                                               std::to_string(a);
                    const TensorView *av = FindTensorSuffix(*adapters, a_name);
                    const TensorView *bv = FindTensorSuffix(*adapters, b_name);
                    if (!av && !bv) continue;
                    if (!av || !bv || av->rank != 2 || bv->rank != 2) {
                        log::error(
                            "[granite_switch_load] bad LoRA pair: " + a_name);
                        return false;
                    }
                    GraniteSwitchLora &lora =
                        site.proj->lora[static_cast<size_t>(a)]
                                       [static_cast<size_t>(s)];
                    lora.rank = static_cast<int>(av->shape[0]);
                    lora.in = static_cast<int>(av->shape[1]);
                    lora.out = static_cast<int>(bv->shape[0]);
                    if (static_cast<int>(bv->shape[1]) != lora.rank) {
                        log::error("[granite_switch_load] LoRA rank mismatch: " +
                                   a_name);
                        return false;
                    }
                    jobs.push_back({av, reinterpret_cast<void **>(&lora.d_A),
                                    CvKind::kBf16ToHalf});
                    jobs.push_back({bv, reinterpret_cast<void **>(&lora.d_B),
                                    CvKind::kBf16ToHalf});
                }
            }
        }
    }

    cudaError_t e = RunUploadJobs(std::move(jobs), &lw.d_arena,
                                  &lw.arena_bytes, stream, ring, bytes_acc,
                                  nullptr);
    if (e != cudaSuccess) return false;

    struct Plane {
        const char *suffix;
        GraniteSwitchProjection *proj;
    };
    const Plane planes[] = {
        {"self_attn.qkv_proj.weight", &lw.qkv},
        {"self_attn.o_proj.weight", &lw.o},
        {"shared_mlp.gate_up_proj.weight", &lw.gate_up},
        {"shared_mlp.down_proj.weight", &lw.down},
    };
    for (const Plane &pl : planes) {
        e = UploadNvfp4Weight(FindNvfp4Suffix(nv, pl.suffix), pl.proj->base,
                              stream, ring, bytes_acc, nullptr);
        if (e != cudaSuccess) return false;
        // FindNvfp4Suffix matches by SUFFIX, and "<name>.res" ends with
        // "<name>" only if the search string omits nothing — so the residual is
        // looked up with its own suffix rather than reusing the base's.
        const std::string res_suffix = std::string(pl.suffix) + ".res";
        const auto *res = FindNvfp4Suffix(nv, res_suffix);
        if (res != nullptr) {
            e = UploadNvfp4Weight(res, pl.proj->res, stream, ring, bytes_acc,
                                  nullptr);
            if (e != cudaSuccess) return false;
        }
    }

    lw.loaded = lw.d_input_norm && lw.d_post_norm && lw.qkv.base.d_packed &&
                lw.o.base.d_packed && lw.gate_up.base.d_packed &&
                lw.down.base.d_packed;
    if (!lw.loaded)
        log::error("[granite_switch_load] layer weights incomplete");
    return lw.loaded;
}

} // namespace

bool LoadGraniteSwitchPinnedWeights(const std::string &output_root,
                                    GraniteSwitchPinnedWeights &pw,
                                    cudaStream_t stream)
{
    namespace fs = std::filesystem;

    pw.Free();
    const auto t0 = std::chrono::steady_clock::now();

    auto pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) {
        log::error("[granite_switch_load] no pinned dir under " + output_root);
        return false;
    }

    auto idx = fs::path(output_root) / "model.index.json";
    if (!fs::exists(idx)) {
        log::error("[granite_switch_load] no model.index.json");
        return false;
    }
    pw.top = load_granite_switch_topology(idx.string());
    if (!pw.top.error_msg.empty() || pw.top.num_layers <= 0) {
        log::error("[granite_switch_load] topology error: " + pw.top.error_msg);
        return false;
    }

    fastload::StreamDrain drain{stream};
    PinnedStagingRing ring;
    if (!ring.ok()) return false;
    std::uint64_t &bytes_acc = pw.resident_bytes;
    bytes_acc = 0;

    if (fs::exists(pinned_dir / "global.arrow")) {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw_res.ok()) {
            log::error("[granite_switch_load] global.arrow read failed");
            pw.Free();
            return false;
        }
        auto *fn = FindTensorSuffix(*raw_res, "model.norm.weight");
        if (fn && fn->rank > 0) {
            const int64_t n = fn->shape[0];
            pw.final_norm_size = static_cast<int>(n);
            cudaError_t e = cudaMalloc(
                &pw.d_final_norm, static_cast<size_t>(n) * sizeof(__half));
            if (e != cudaSuccess) {
                pw.Free();
                return false;
            }
            if (fn->dtype == "F32") {
                std::vector<__half> h(static_cast<size_t>(n));
                const float *src = reinterpret_cast<const float *>(fn->data);
                for (int64_t i = 0; i < n; ++i)
                    h[static_cast<size_t>(i)] = __float2half(src[i]);
                e = cudaMemcpyAsync(pw.d_final_norm, h.data(),
                                    static_cast<size_t>(n) * sizeof(__half),
                                    cudaMemcpyHostToDevice, stream);
            } else {
                e = ring.UploadBf16AsFp16(fn->data, n, pw.d_final_norm, stream,
                                          bytes_acc);
            }
            if (e != cudaSuccess) {
                pw.Free();
                return false;
            }
        }
    }

    if (fs::exists(pinned_dir / "global_nvfp4.arrow")) {
        auto nv_res =
            LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
        if (!nv_res.ok()) {
            pw.Free();
            return false;
        }
        cudaError_t e = UploadNvfp4Weight(
            FindNvfp4Suffix(*nv_res, "embed_tokens.weight"), pw.embed, stream,
            ring, bytes_acc, nullptr);
        if (e != cudaSuccess) {
            pw.Free();
            return false;
        }
    }
    if (!pw.embed.d_packed) {
        log::error("[granite_switch_load] embed missing");
        pw.Free();
        return false;
    }
    // Tied LM head: same matrix, scaled by 1/logits_scaling. The scale is a
    // power of two, so folding it into global_scale is exact.
    pw.head = pw.embed;
    pw.head.global_scale =
        pw.embed.global_scale / static_cast<float>(pw.top.logits_scaling);
    pw.head.view_only = true;

    pw.layers.resize(static_cast<size_t>(pw.top.num_layers));
    for (int l = 0; l < pw.top.num_layers; ++l) {
        auto raw_path = pinned_dir / absl::StrCat("layer_", l, ".arrow");
        auto nv_path = pinned_dir / absl::StrCat("layer_", l, "_nvfp4.arrow");
        auto ad_path = pinned_dir / absl::StrCat("layer_", l, "_adapters.arrow");
        if (!fs::exists(raw_path) || !fs::exists(nv_path)) {
            log::error("[granite_switch_load] missing layer " + std::to_string(l));
            pw.Free();
            return false;
        }
        auto raw = LoadTensorBatchViews(raw_path);
        auto nv = LoadDenseNvfp4BatchViews(nv_path);
        if (!raw.ok() || !nv.ok()) {
            pw.Free();
            return false;
        }
        TensorBatchViews ad;
        if (fs::exists(ad_path)) {
            auto ad_res = LoadTensorBatchViews(ad_path);
            if (!ad_res.ok()) {
                pw.Free();
                return false;
            }
            ad = std::move(*ad_res);
        }
        if (!LoadLayer(*raw, fs::exists(ad_path) ? &ad : nullptr, *nv, pw.top,
                       pw.layers[static_cast<size_t>(l)], stream, ring,
                       bytes_acc)) {
            pw.Free();
            return false;
        }
    }

    pw.resident_bytes = bytes_acc;
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    log::info(absl::StrCat("[granite_switch_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms"));
    return true;
}

int32_t GraniteSwitchSubstitute(const GraniteSwitchTopology &top, int32_t id)
{
    for (int a = 0; a < top.num_adapters; ++a) {
        if (id == top.adapter_token_ids[static_cast<size_t>(a)])
            return static_cast<int32_t>(
                top.adapter_substitute_token_ids[static_cast<size_t>(a)]);
    }
    return id;
}

GraniteSwitchPrepared PrepareGraniteSwitchTokens(
    const GraniteSwitchTopology &top, const std::vector<int32_t> &ids,
    int carry_adapter)
{
    GraniteSwitchPrepared out;
    out.embed_ids.reserve(ids.size());
    out.adapter_ids.reserve(ids.size());
    int carry = carry_adapter;
    if (carry < 0 || carry > top.num_adapters) carry = 0;
    for (int32_t id : ids) {
        int next = carry;
        int32_t subst = id;
        for (int a = 0; a < top.num_adapters; ++a) {
            if (id == top.adapter_token_ids[static_cast<size_t>(a)]) {
                next = a + 1;
                subst = static_cast<int32_t>(
                    top.adapter_substitute_token_ids[static_cast<size_t>(a)]);
                break;
            }
        }
        out.embed_ids.push_back(subst);
        out.adapter_ids.push_back(next);
        carry = next;
    }
    out.last_adapter = carry;
    return out;
}

} // namespace granite_switch
} // namespace lsxcommon
