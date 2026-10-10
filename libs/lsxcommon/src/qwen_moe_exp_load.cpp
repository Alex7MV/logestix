#include "lsxcommon/qwen_moe_exp_load.h"
#include "qwen_moe_exp_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/schema.h"
#include "fast_load_internal.h"

#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace lsxcommon::qwen_moe_exp {
namespace {

using fastload::CvKind;
using fastload::OutElemSize;
using fastload::UploadJob;
using fastload::kArenaAlign;
using fastload::AlignUp;
using fastload::StreamDrain;
using fastload::RunUploadJobs;
using fastload::UploadNvfp4Weight;

CvKind NormKind(const TensorView *t)
{
    return (t && t->dtype == "F32") ? CvKind::kF32ToHalf : CvKind::kBf16ToHalf;
}

// Dtype-aware kind for F32-destined tensors: honor BF16-on-disk records.
CvKind CvKindFromDtype(const TensorView *t, CvKind f32_kind)
{
    return (t && t->dtype == "BF16") ? CvKind::kBf16ToF32 : f32_kind;
}

// Parses the superset index (format logestix-qwen38flashnext) into both
// topology views. Returns false with `error` filled on structural problems.
bool ParseExpIndex(const std::filesystem::path &idx_path,
                   Qwen4ExpTopology &exp, qwen::QwenTopology &dense,
                   std::string &error)
{
    std::ifstream f(idx_path, std::ios::binary);
    if (!f) {
        error = "cannot open " + idx_path.string();
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        error = "index parse error: " + idx_path.string();
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        error = "index is not an object";
        return false;
    }
    auto gi = [&](const char *k, int dflt) {
        int64_t v;
        if (root[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto gd = [&](const char *k, double dflt) {
        double v;
        if (root[k].get_double().get(v) != simdjson::SUCCESS) return dflt;
        return v;
    };

    exp.num_layers = gi("num_layers", 0);
    exp.hidden_size = gi("hidden_size", 0);
    exp.vocab_size = gi("vocab_size", 0);
    exp.num_attention_heads = gi("num_attention_heads", 0);
    exp.num_key_value_heads = gi("num_key_value_heads", 0);
    exp.head_dim = gi("head_dim", 0);
    exp.linear_num_key_heads = gi("linear_num_key_heads", 0);
    exp.linear_num_value_heads = gi("linear_num_value_heads", 0);
    exp.linear_key_head_dim = gi("linear_key_head_dim", 0);
    exp.linear_value_head_dim = gi("linear_value_head_dim", 0);
    exp.linear_conv_kernel_dim = gi("linear_conv_kernel_dim", 4);
    exp.rms_norm_eps = gd("rms_norm_eps", 1e-6);
    exp.partial_rotary_factor = gd("partial_rotary_factor", 0.25);
    exp.rope_theta = gd("rope_theta", 1e7);
    exp.num_experts = gi("num_experts", 0);
    exp.num_experts_per_tok = gi("num_experts_per_tok", 0);
    exp.moe_intermediate_size = gi("moe_intermediate_size", 0);
    exp.shared_expert_intermediate_size =
        gi("shared_expert_intermediate_size", 0);
    exp.eos_token_id = gi("eos_token_id", -1);
    exp.hc_count = gi("hc_count", 4);
    exp.hc_lowrank = gi("hc_lowrank", 320);
    exp.indexer_n_heads = gi("indexer_n_heads", 4);
    exp.indexer_kv_heads = gi("indexer_kv_heads", 1);
    exp.indexer_head_dim = gi("indexer_head_dim", 128);
    exp.indexer_budget = gi("indexer_budget", 2048);
    exp.indexer_compress_ratio = gi("indexer_compress_ratio", 4);
    exp.ngram_size = gi("ngram_size", 3);
    exp.heads_per_ngram = gi("heads_per_ngram", 8);
    exp.ple_embed_dim = gi("ple_embed_dim", 0);
    exp.ple_conv_kernel_size = gi("ple_conv_kernel_size", 4);
    exp.split_ngram_parts = gi("split_ngram_parts", 128);
    if (auto v = root["ngram_vocab_size_base"].get_int64();
        v.error() == simdjson::SUCCESS)
        exp.ngram_vocab_size_base = v.value();

    if (auto lt = root["layer_types"].get_array();
        lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                exp.layer_types.emplace_back(sv);
        }
    }
    if (auto pl = root["ple_layer_ids"].get_array();
        pl.error() == simdjson::SUCCESS) {
        for (auto e : pl.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                exp.ple_layer_ids.push_back(static_cast<int>(v));
        }
    }

    // Dense mirror for the reused qwen kernels.
    dense.num_layers = exp.num_layers;
    dense.layer_types = exp.layer_types;
    dense.hidden_size = exp.hidden_size;
    dense.vocab_size = exp.vocab_size;
    dense.num_attention_heads = exp.num_attention_heads;
    dense.num_key_value_heads = exp.num_key_value_heads;
    dense.head_dim = exp.head_dim;
    dense.linear_num_key_heads = exp.linear_num_key_heads;
    dense.linear_num_value_heads = exp.linear_num_value_heads;
    dense.linear_key_head_dim = exp.linear_key_head_dim;
    dense.linear_value_head_dim = exp.linear_value_head_dim;
    dense.linear_conv_kernel_dim = exp.linear_conv_kernel_dim;
    dense.rms_norm_eps = exp.rms_norm_eps;
    dense.partial_rotary_factor = exp.partial_rotary_factor;
    dense.rope_theta = exp.rope_theta;

    if (exp.num_layers <= 0 || exp.num_experts <= 0 ||
        static_cast<int>(exp.layer_types.size()) != exp.num_layers) {
        error = "index missing layers/experts fields";
        return false;
    }
    return true;
}

// Computes the PLE hash meta from config + checkpoint cross-check. Prefers
// the stored meta.arrow values; hard-verifies both against recomputation.
bool ComputePleMeta(const Qwen4ExpTopology &exp,
                    const std::vector<TensorRecord> &stored_meta,
                    ExpPleMeta &meta, std::string &error)
{
    const int heads = (exp.ngram_size - 1) * exp.heads_per_ngram;
    if (heads <= 0 || heads > ExpPleMeta::kMaxHeads) {
        error = absl::StrCat("PLE head count out of range: ", heads);
        return false;
    }
    meta.num_heads = heads;
    meta.head_dim = exp.ple_embed_dim > 0
        ? exp.ple_embed_dim / heads : 0;
    meta.context_len = exp.ngram_size - 1;
    Splitmix64Multipliers(frozen::kPleSeed, 0, exp.vocab_size,
                          exp.ngram_size, meta.multipliers);
    int64_t total = 0;
    for (int h = 0; h < heads; ++h) {
        meta.head_vocab_sizes[h] =
            FindNthPrimeAfter(exp.ngram_vocab_size_base - 1, h + 1);
        meta.head_offsets[h] = total;
        total += meta.head_vocab_sizes[h];
    }
    const int64_t divisor = 128;
    meta.padded_vocab = (total + divisor - 1) / divisor * divisor;

    // Cross-check stored buffers when present (bit-exact requirement).
    for (const auto &rec : stored_meta) {
        const size_t n64 = rec.data.size() / 8;
        if (rec.name.ends_with("layer_multipliers")) {
            if (n64 != 3) {
                error = "PLE layer_multipliers wrong size";
                return false;
            }
            for (int i = 0; i < 3; ++i) {
                int64_t got;
                std::memcpy(&got, rec.data.data() + i * 8, 8);
                if (got != meta.multipliers[i]) {
                    error = absl::StrCat("PLE multiplier ", i,
                                         " mismatch: stored ", got,
                                         " computed ", meta.multipliers[i]);
                    return false;
                }
            }
        } else if (rec.name.ends_with("ngram_heads_vocab_sizes")) {
            if (static_cast<int>(n64) != heads) {
                error = "PLE vocab sizes wrong count";
                return false;
            }
            for (int h = 0; h < heads; ++h) {
                int64_t got;
                std::memcpy(&got, rec.data.data() + h * 8, 8);
                if (got != meta.head_vocab_sizes[h]) {
                    error = absl::StrCat("PLE prime size ", h, " mismatch");
                    return false;
                }
            }
        } else if (rec.name.ends_with("ngram_heads_offsets")) {
            if (static_cast<int>(n64) != heads) {
                error = "PLE offsets wrong count";
                return false;
            }
            for (int h = 0; h < heads; ++h) {
                int64_t got;
                std::memcpy(&got, rec.data.data() + h * 8, 8);
                if (got != meta.head_offsets[h]) {
                    error = absl::StrCat("PLE offset ", h, " mismatch");
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

void ExpLayerWeightsFree(ExpLayerWeights &lw)
{
    lw.attn.Free();
    if (lw.d_attn_hc_norm) { cudaFree(lw.d_attn_hc_norm); lw.d_attn_hc_norm = nullptr; }
    if (lw.d_mlp_hc_norm) { cudaFree(lw.d_mlp_hc_norm); lw.d_mlp_hc_norm = nullptr; }
    lw.hc_norm_bytes = 0;
    if (lw.attn_hc_cache) { lw.attn_hc_cache->Free(); delete lw.attn_hc_cache; lw.attn_hc_cache = nullptr; }
    if (lw.mlp_hc_cache) { lw.mlp_hc_cache->Free(); delete lw.mlp_hc_cache; lw.mlp_hc_cache = nullptr; }
    lw.attn_hc_down.Free();
    lw.attn_hc_up.Free();
    lw.attn_hc_inject.Free();
    lw.mlp_hc_down.Free();
    lw.mlp_hc_up.Free();
    lw.mlp_hc_inject.Free();
    if (lw.d_router) { cudaFree(lw.d_router); lw.d_router = nullptr; }
    lw.router_bytes = 0;
    lw.shared_w1.Free();
    lw.shared_w2.Free();
    lw.shared_w3.Free();
    if (lw.d_shared_gate) { cudaFree(lw.d_shared_gate); lw.d_shared_gate = nullptr; }
    lw.ple_key.Free();
    lw.ple_value.Free();
    if (lw.d_ple_conv_bf16) { cudaFree(lw.d_ple_conv_bf16); lw.d_ple_conv_bf16 = nullptr; }
    lw.ple_conv_bytes = 0;
    if (lw.d_ple_norm_key) { cudaFree(lw.d_ple_norm_key); lw.d_ple_norm_key = nullptr; }
    if (lw.d_ple_norm_query) { cudaFree(lw.d_ple_norm_query); lw.d_ple_norm_query = nullptr; }
    if (lw.d_ple_norm_conv) { cudaFree(lw.d_ple_norm_conv); lw.d_ple_norm_conv = nullptr; }
    lw.ple_conv_state_len = 0;
    lw.is_ple_layer = false;
    lw.moe_loaded = false;
}

void ExpPinnedWeights::Free()
{
    embed.Free();
    head.Free();
    if (d_head_f16) { cudaFree(d_head_f16); d_head_f16 = nullptr; }
    head_elems = 0;
    if (d_mixer_norm) { cudaFree(d_mixer_norm); d_mixer_norm = nullptr; }
    mixer_norm_bytes = 0;
    if (mixer_hc_cache) { mixer_hc_cache->Free(); delete mixer_hc_cache; mixer_hc_cache = nullptr; }
    mixer_down.Free();
    mixer_up.Free();
    if (d_final_norm) { cudaFree(d_final_norm); d_final_norm = nullptr; }
    final_norm_size = 0;
    for (auto &l : layers) ExpLayerWeightsFree(l);
    layers.clear();
    ple_meta = {};
    resident_bytes = 0;
}

void FreeQwen4ExpPinnedWeights(ExpPinnedWeights &pw) { pw.Free(); }

bool LoadQwen4ExpDenseWeights(const std::string &model_path,
                              ExpPinnedWeights &pw, cudaStream_t stream,
                              std::string &error)
{
    namespace fs = std::filesystem;
    pw.Free();
    error.clear();

    fs::path base = fs::path(model_path) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    if (!fs::exists(pinned_dir)) {
        error = "no pinned dir under " + base.string();
        return false;
    }

    StreamDrain drain{stream};

    // Topology from the superset index.
    {
        const fs::path idx = base / "model.index.json";
        std::string idx_error;
        if (!ParseExpIndex(idx, pw.exp_top, pw.top, idx_error)) {
            error = idx_error;
            return false;
        }
    }

    fastload::PinnedStagingRing ring;
    if (!ring.ok()) {
        error = "pinned staging ring init failed";
        return false;
    }
    std::uint64_t bytes_acc = 0;

    // Globals: final norm (BF16->FP16), embed (NVFP4) + lm_head (BF16 raw).
    {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw_res.ok()) {
            error = "global.arrow read failed: " + raw_res.status().ToString();
            return false;
        }
        // Final norm: the qwen4_exp inventory has no separate final norm —
        // the global mixer replaces it. Accept either shape here.
        if (auto *fn = fastload::FindTensorSuffix(*raw_res, "norm.weight");
            fn && fn->rank > 0) {
            const int64_t n = fn->shape[0];
            pw.final_norm_size = static_cast<int>(n);
            cudaError_t e = cudaMalloc(&pw.d_final_norm,
                                       static_cast<size_t>(n) * sizeof(__half));
            if (e != cudaSuccess) {
                error = absl::StrCat("final norm alloc failed: ",
                                     cudaGetErrorString(e));
                pw.Free();
                return false;
            }
            e = ring.UploadBf16AsFp16(fn->data, n, pw.d_final_norm, stream,
                                      bytes_acc);
            if (e != cudaSuccess) {
                error = "final norm upload failed";
                pw.Free();
                return false;
            }
        }
        // LM head: BF16 bits, converted to a persistent FP16 table for the
        // FP16 head GEMV (NVFP4 logit noise exceeds the top-1/top-2 margin
        // of this model).
        if (auto *hd = fastload::FindTensorSuffix(*raw_res, "lm_head.weight");
            hd && hd->rank >= 2) {
            pw.head_elems = static_cast<int64_t>(hd->shape[0]) *
                            static_cast<int64_t>(hd->shape[1]);
            cudaError_t e = cudaMalloc(
                reinterpret_cast<void **>(&pw.d_head_f16),
                static_cast<size_t>(pw.head_elems) * sizeof(__half));
            if (e != cudaSuccess) {
                error = "lm_head fp16 alloc failed";
                pw.Free();
                return false;
            }
            e = ring.UploadBf16AsFp16(hd->data, pw.head_elems, pw.d_head_f16,
                                      stream, bytes_acc);
            if (e != cudaSuccess) {
                error = "lm_head upload failed";
                pw.Free();
                return false;
            }
        } else {
            error = "lm_head missing from global.arrow (BF16 raw expected)";
            pw.Free();
            return false;
        }
    }
    {
        auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
        if (!nv_res.ok()) {
            error = "global_nvfp4.arrow read failed: " +
                    nv_res.status().ToString();
            pw.Free();
            return false;
        }
        std::string dummy;
        cudaError_t e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv_res, "embed_tokens.weight"),
                                          pw.embed, stream, ring, bytes_acc, &dummy);
        if (e != cudaSuccess || !pw.embed.d_packed) {
            error = "embed missing or upload failed";
            pw.Free();
            return false;
        }
    }

    // Global mixer bundle (norm in global.arrow under hc suffix; down/up in
    // global_nvfp4.arrow).
    {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (raw_res.ok()) {
            auto *mn = fastload::FindTensorSuffix(*raw_res,
                "hyper_connection_mixer.hc_norm.weight");
            if (mn && mn->bytes > 0) {
                pw.mixer_norm_bytes = static_cast<int64_t>(mn->bytes);
                cudaError_t e = cudaMalloc(
                    reinterpret_cast<void **>(&pw.d_mixer_norm), mn->bytes);
                if (e != cudaSuccess) {
                    error = "global mixer norm alloc failed";
                    pw.Free();
                    return false;
                }
                e = ring.UploadRaw(mn->data, pw.d_mixer_norm, mn->bytes,
                                   stream, bytes_acc);
                if (e != cudaSuccess) {
                    error = "global mixer norm upload failed";
                    pw.Free();
                    return false;
                }
            }
        }
        auto nv_res = LoadDenseNvfp4BatchViews(pinned_dir / "global_nvfp4.arrow");
        if (nv_res.ok()) {
            std::string job_error;
            cudaError_t e = UploadNvfp4Weight(
                fastload::FindNvfp4Suffix(*nv_res, "hyper_connection_mixer.input_mix_weight_down.weight"),
                pw.mixer_down, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess)
                e = UploadNvfp4Weight(
                    fastload::FindNvfp4Suffix(*nv_res, "hyper_connection_mixer.input_mix_weight_up.weight"),
                    pw.mixer_up, stream, ring, bytes_acc, &job_error);
            if (e != cudaSuccess || !pw.mixer_down.d_packed ||
                !pw.mixer_up.d_packed) {
                error = "global hc mixer weights incomplete";
                pw.Free();
                return false;
            }
            pw.mixer_hc_cache = new (std::nothrow) ExpHcDequantCache();
            cudaError_t mhe = pw.mixer_hc_cache
                                  ? BuildExpHcDequantCache(
                                        stream, pw.mixer_down, pw.mixer_up,
                                        lsx_nvfp4::LsxNvfp4Weight{},
                                        *pw.mixer_hc_cache)
                                  : cudaErrorMemoryAllocation;
            if (mhe != cudaSuccess) {
                error = "global hc mixer dequant cache build failed";
                pw.Free();
                return false;
            }
        }
    }

    // PLE meta (recompute + cross-check against meta.arrow buffers).
    {
        std::vector<TensorRecord> stored_meta;
        const fs::path meta_path = base / "ple" / "meta.arrow";
        if (fs::exists(meta_path)) {
            auto meta_res = LoadTensorBatchViews(meta_path);
            if (!meta_res.ok()) {
                error = "ple/meta.arrow read failed: " +
                        meta_res.status().ToString();
                pw.Free();
                return false;
            }
            for (const auto &tv : meta_res->tensors) {
                TensorRecord r;
                r.name = std::string(tv.name);
                r.dtype = std::string(tv.dtype);
                r.shape.assign(tv.shape, tv.shape + tv.rank);
                r.data.assign(tv.data, tv.data + tv.bytes);
                stored_meta.push_back(std::move(r));
            }
        }
        std::string meta_error;
        if (!ComputePleMeta(pw.exp_top, stored_meta, pw.ple_meta, meta_error)) {
            error = meta_error;
            pw.Free();
            return false;
        }
    }

    // Per-layer dense parts.
    const int num_layers = pw.top.num_layers;
    if (num_layers <= 0) {
        error = "topology num_layers unknown";
        pw.Free();
        return false;
    }
    pw.layers.resize(static_cast<size_t>(num_layers));

    // PLE layer index (tensor-space: layers.<N>.ple where N = layer idx).
    int ple_layer = -1;
    if (!pw.exp_top.ple_layer_ids.empty()) {
        // HF: ple_layer_ids are 1-based "position after layer"; tensor prefix
        // is layer_idx = id - 1.
        ple_layer = pw.exp_top.ple_layer_ids[0] - 1;
    }

    for (int l = 0; l < num_layers; ++l) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        const std::string raw_name =
            (pinned_dir / absl::StrCat("layer_", l, ".arrow")).string();
        const std::string nv_name =
            (pinned_dir / absl::StrCat("layer_", l, "_nvfp4.arrow")).string();

        auto raw = LoadTensorBatchViews(raw_name);
        if (!raw.ok()) {
            error = absl::StrCat("layer_", l, ".arrow read failed");
            pw.Free();
            return false;
        }
        auto nv = LoadDenseNvfp4BatchViews(nv_name);
        if (!nv.ok()) {
            error = absl::StrCat("layer_", l, "_nvfp4.arrow read failed");
            pw.Free();
            return false;
        }

        bool is_linear = true;
        if (l < static_cast<int>(pw.top.layer_types.size()))
            is_linear = pw.top.layer_types[static_cast<size_t>(l)] ==
                        "linear_attention";

        // Dense attention block (adapted from qwen LoadLayer minus MLP).
        lw.attn.is_linear = is_linear;
        auto *in_norm = fastload::FindTensorSuffix(*raw, "input_layernorm.weight");
        if (in_norm && in_norm->rank > 0)
            lw.attn.norm_size = static_cast<int>(in_norm->shape[0]);
        auto *post_norm = fastload::FindTensorSuffix(*raw, "post_attention_layernorm.weight");

        std::string job_error;
        std::vector<UploadJob> jobs = {
            {in_norm,
             reinterpret_cast<void **>(&lw.attn.d_input_norm), NormKind(in_norm)},
            {post_norm,
             reinterpret_cast<void **>(&lw.attn.d_post_norm), NormKind(post_norm)},
        };
        if (is_linear) {
            auto *conv = fastload::FindTensorSuffix(*raw, "linear_attn.conv1d.weight");
            if (conv && conv->rank > 0) {
                lw.attn.conv_dim = static_cast<int>(conv->shape[0]);
                lw.attn.conv_kernel = conv->rank >= 3
                    ? static_cast<int>(conv->shape[2])
                    : static_cast<int>(conv->shape[conv->rank - 1]);
            }
            jobs.push_back({conv, reinterpret_cast<void **>(&lw.attn.d_conv1d),
                            CvKind::kBf16ToHalf});
            jobs.push_back({fastload::FindTensorSuffix(*raw, "linear_attn.A_log"),
                            reinterpret_cast<void **>(&lw.attn.d_A_log),
                            CvKindFromDtype(fastload::FindTensorSuffix(*raw, "linear_attn.A_log"),
                                            CvKind::kF32Raw)});
            jobs.push_back({fastload::FindTensorSuffix(*raw, "linear_attn.dt_bias"),
                            reinterpret_cast<void **>(&lw.attn.d_dt_bias),
                            CvKind::kBf16ToF32});
            auto *an = fastload::FindTensorSuffix(*raw, "linear_attn.norm.weight");
            if (an && an->rank > 0)
                lw.attn.attn_norm_size = static_cast<int>(an->shape[0]);
            jobs.push_back({an, reinterpret_cast<void **>(&lw.attn.d_attn_norm),
                            NormKind(an)});
        } else {
            auto *qn = fastload::FindTensorSuffix(*raw, "self_attn.q_norm.weight");
            auto *kn = fastload::FindTensorSuffix(*raw, "self_attn.k_norm.weight");
            if (qn && qn->rank > 0)
                lw.attn.q_norm_size = static_cast<int>(qn->shape[0]);
            if (kn && kn->rank > 0)
                lw.attn.k_norm_size = static_cast<int>(kn->shape[0]);
            jobs.push_back({qn, reinterpret_cast<void **>(&lw.attn.d_q_norm),
                            NormKind(qn)});
            jobs.push_back({kn, reinterpret_cast<void **>(&lw.attn.d_k_norm),
                            NormKind(kn)});
        }

        cudaError_t e = RunUploadJobs(std::move(jobs), &lw.attn.d_arena,
                                      &lw.attn.arena_bytes, stream, ring,
                                      bytes_acc, &job_error);
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, ": ", job_error);
            pw.Free();
            return false;
        }

        if (is_linear) {
            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "linear_attn.in_proj_qkv.weight"), lw.attn.qkv, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "linear_attn.in_proj_z.weight"), lw.attn.z, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "linear_attn.in_proj_b.weight"), lw.attn.b, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "linear_attn.in_proj_a.weight"), lw.attn.a, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "linear_attn.out_proj.weight"), lw.attn.out, stream, ring, bytes_acc, &job_error);
        } else {
            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "self_attn.q_proj.weight"), lw.attn.q, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "self_attn.k_proj.weight"), lw.attn.k, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "self_attn.v_proj.weight"), lw.attn.v, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "self_attn.o_proj.weight"), lw.attn.o, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "self_attn.indexer.index_qk_proj.weight"), lw.indexer_qk, stream, ring, bytes_acc, &job_error);
        }
        if (e != cudaSuccess) {
            error = absl::StrCat("layer_", l, ": ", job_error);
            pw.Free();
            return false;
        }

        const bool attn_ok = is_linear
            ? (lw.attn.qkv.d_packed && lw.attn.out.d_packed)
            : (lw.attn.q.d_packed && lw.attn.k.d_packed &&
               lw.attn.v.d_packed && lw.attn.o.d_packed);
        // qwen4_exp has NO input/post layernorms — residual normalization is
        // the hyper-connection group-RMSNorm. Require only the projections.
        if (!attn_ok) {
            error = absl::StrCat("layer_", l, ": attention weights incomplete");
            pw.Free();
            return false;
        }
        lw.attn.loaded = true;

        // Hyper-connections: norms (BF16 raw) + down/up/inject (NVFP4).
        {
            auto *an = fastload::FindTensorSuffix(*raw, "attn_hyper_connection.hc_norm.weight");
            auto *mn = fastload::FindTensorSuffix(*raw, "mlp_hyper_connection.hc_norm.weight");
            if (!an || !mn) {
                error = absl::StrCat("layer_", l, ": hc norms missing");
                pw.Free();
                return false;
            }
            lw.hc_norm_bytes = static_cast<int64_t>(an->bytes);
            cudaError_t he = cudaMalloc(
                reinterpret_cast<void **>(&lw.d_attn_hc_norm), an->bytes);
            if (he == cudaSuccess)
                he = cudaMalloc(
                    reinterpret_cast<void **>(&lw.d_mlp_hc_norm), mn->bytes);
            if (he != cudaSuccess) {
                error = absl::StrCat("layer_", l, ": hc norm alloc failed");
                pw.Free();
                return false;
            }
            he = ring.UploadRaw(an->data, lw.d_attn_hc_norm, an->bytes, stream, bytes_acc);
            if (he == cudaSuccess)
                he = ring.UploadRaw(mn->data, lw.d_mlp_hc_norm, mn->bytes, stream, bytes_acc);
            if (he != cudaSuccess) {
                error = absl::StrCat("layer_", l, ": hc norm upload failed");
                pw.Free();
                return false;
            }

            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "attn_hyper_connection.input_mix_weight_down.weight"), lw.attn_hc_down, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "attn_hyper_connection.input_mix_weight_up.weight"), lw.attn_hc_up, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "attn_hyper_connection.block_inject_weight.weight"), lw.attn_hc_inject, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "mlp_hyper_connection.input_mix_weight_down.weight"), lw.mlp_hc_down, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "mlp_hyper_connection.input_mix_weight_up.weight"), lw.mlp_hc_up, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv, "mlp_hyper_connection.block_inject_weight.weight"), lw.mlp_hc_inject, stream, ring, bytes_acc, &job_error);
            if (e != cudaSuccess || !lw.attn_hc_down.d_packed ||
                !lw.attn_hc_up.d_packed || !lw.attn_hc_inject.d_packed ||
                !lw.mlp_hc_down.d_packed || !lw.mlp_hc_up.d_packed ||
                !lw.mlp_hc_inject.d_packed) {
                error = absl::StrCat("layer_", l, ": hc weights incomplete");
                pw.Free();
                return false;
            }

            // One-time dequant of the six hc matrices into per-layer fp16
            // caches: removes the per-token per-layer dequant (the largest
            // single decode cost of this architecture). Bit-identical to the
            // legacy path (same dequant kernel; fp16->fp32 upcast lossless).
            auto build_hc = [&](ExpHcDequantCache **slot,
                                const lsx_nvfp4::LsxNvfp4Weight &dn,
                                const lsx_nvfp4::LsxNvfp4Weight &up,
                                const lsx_nvfp4::LsxNvfp4Weight &inj) -> cudaError_t {
                *slot = new (std::nothrow) ExpHcDequantCache();
                if (!*slot) return cudaErrorMemoryAllocation;
                cudaError_t bhe =
                    BuildExpHcDequantCache(stream, dn, up, inj, **slot);
                if (bhe != cudaSuccess) {
                    (*slot)->Free();
                    delete *slot;
                    *slot = nullptr;
                }
                return bhe;
            };
            e = build_hc(&lw.attn_hc_cache, lw.attn_hc_down, lw.attn_hc_up,
                         lw.attn_hc_inject);
            if (e == cudaSuccess)
                e = build_hc(&lw.mlp_hc_cache, lw.mlp_hc_down, lw.mlp_hc_up,
                             lw.mlp_hc_inject);
            if (e != cudaSuccess) {
                error = absl::StrCat("layer_", l, ": hc dequant cache build failed");
                pw.Free();
                return false;
            }
        }

        // MoE extras: router (BF16 raw), shared expert trio + scalar gate.
        {
            const std::string router_path =
                (base / "router" / absl::StrCat("layer_", l, ".arrow")).string();
            auto rr = LoadTensorBatchViews(router_path);
            if (!rr.ok()) {
                error = absl::StrCat("router/layer_", l, ".arrow read failed");
                pw.Free();
                return false;
            }
            auto *gate = fastload::FindTensorSuffix(*rr, "mlp.gate.weight");
            if (!gate || gate->rank != 2) {
                error = absl::StrCat("router/layer_", l, ": mlp.gate.weight missing");
                pw.Free();
                return false;
            }
            lw.router_bytes = gate->bytes;
            cudaError_t re = cudaMalloc(reinterpret_cast<void **>(&lw.d_router),
                                        static_cast<size_t>(gate->bytes));
            if (re != cudaSuccess) {
                error = absl::StrCat("router/layer_", l, ": alloc failed");
                pw.Free();
                return false;
            }
            re = ring.UploadRaw(gate->data, lw.d_router,
                                static_cast<size_t>(gate->bytes), stream,
                                bytes_acc);
            if (re != cudaSuccess) {
                error = absl::StrCat("router/layer_", l, ": upload failed");
                pw.Free();
                return false;
            }

            const std::string shared_nv =
                (base / "shared" / absl::StrCat("layer_", l, "_nvfp4.arrow")).string();
            auto sv = LoadDenseNvfp4BatchViews(shared_nv);
            if (!sv.ok()) {
                error = absl::StrCat("shared/layer_", l, "_nvfp4.arrow read failed");
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*sv, "shared_expert.gate_proj.weight"), lw.shared_w1, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*sv, "shared_expert.down_proj.weight"), lw.shared_w2, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*sv, "shared_expert.up_proj.weight"), lw.shared_w3, stream, ring, bytes_acc, &job_error);
            if (e != cudaSuccess || !lw.shared_w1.d_packed ||
                !lw.shared_w2.d_packed || !lw.shared_w3.d_packed) {
                error = absl::StrCat("shared/layer_", l, ": weights incomplete");
                pw.Free();
                return false;
            }

            const std::string gate_path =
                (base / "shared" / absl::StrCat("layer_", l, "_gate.arrow")).string();
            auto gr = LoadTensorBatchViews(gate_path);
            if (!gr.ok()) {
                error = absl::StrCat("shared/layer_", l, "_gate.arrow read failed");
                pw.Free();
                return false;
            }
            auto *sg = fastload::FindTensorSuffix(*gr, "shared_expert_gate.weight");
            if (!sg || sg->rank < 1) {
                error = absl::StrCat("shared/layer_", l, ": gate row missing");
                pw.Free();
                return false;
            }
            const int64_t gate_elems = sg->shape[sg->rank - 1];
            __half *dgate = nullptr;
            cudaError_t ge = cudaMalloc(reinterpret_cast<void **>(&dgate),
                                        static_cast<size_t>(gate_elems) *
                                            sizeof(__half));
            if (ge != cudaSuccess) {
                error = absl::StrCat("shared/layer_", l, ": gate alloc failed");
                pw.Free();
                return false;
            }
            ge = ring.UploadBf16AsFp16(sg->data, gate_elems, dgate, stream,
                                       bytes_acc);
            if (ge != cudaSuccess) {
                error = absl::StrCat("shared/layer_", l, ": gate upload failed");
                pw.Free();
                return false;
            }
            lw.d_shared_gate = dgate;
        }

        // PLE projections (only on the PLE layer).
        if (l == ple_layer) {
            const std::string ple_nv =
                (base / "ple" / absl::StrCat("layer_", l, "_nvfp4.arrow")).string();
            auto pv = LoadDenseNvfp4BatchViews(ple_nv);
            if (!pv.ok()) {
                error = absl::StrCat("ple/layer_", l, "_nvfp4.arrow read failed");
                pw.Free();
                return false;
            }
            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*pv, "ple.key_proj.weight"), lw.ple_key, stream, ring, bytes_acc, &job_error);
            if (e == cudaSuccess) e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*pv, "ple.value_proj.weight"), lw.ple_value, stream, ring, bytes_acc, &job_error);
            if (e != cudaSuccess || !lw.ple_key.d_packed ||
                !lw.ple_value.d_packed) {
                error = absl::StrCat("ple/layer_", l, ": projections incomplete");
                pw.Free();
                return false;
            }
            auto pr = LoadTensorBatchViews(
                (base / "ple" / absl::StrCat("layer_", l, ".arrow")).string());
            if (!pr.ok()) {
                error = absl::StrCat("ple/layer_", l, ".arrow read failed");
                pw.Free();
                return false;
            }
            auto *nk = fastload::FindTensorSuffix(*pr, "ple.norm_key.weight");
            auto *nq = fastload::FindTensorSuffix(*pr, "ple.norm_query.weight");
            auto *nc = fastload::FindTensorSuffix(*pr, "ple.norm_conv.weight");
            // conv1d is stored as raw BF16 (not NVFP4) in the same raw file.
            auto *cv = fastload::FindTensorSuffix(*pr, "ple.conv1d.weight");
            if (!nk || !nq || !nc || !cv) {
                error = absl::StrCat("ple/layer_", l, ": norms/conv missing");
                pw.Free();
                return false;
            }
            cudaError_t pe = cudaMalloc(
                reinterpret_cast<void **>(&lw.d_ple_norm_key), nk->bytes);
            if (pe == cudaSuccess) pe = cudaMalloc(
                reinterpret_cast<void **>(&lw.d_ple_norm_query), nq->bytes);
            if (pe == cudaSuccess) pe = cudaMalloc(
                reinterpret_cast<void **>(&lw.d_ple_norm_conv), nc->bytes);
            if (pe == cudaSuccess) pe = cudaMalloc(
                reinterpret_cast<void **>(&lw.d_ple_conv_bf16), cv->bytes);
            if (pe != cudaSuccess) {
                error = absl::StrCat("ple/layer_", l, ": norm alloc failed");
                pw.Free();
                return false;
            }
            pe = ring.UploadRaw(nk->data, lw.d_ple_norm_key, nk->bytes, stream, bytes_acc);
            if (pe == cudaSuccess) pe = ring.UploadRaw(nq->data, lw.d_ple_norm_query, nq->bytes, stream, bytes_acc);
            if (pe == cudaSuccess) pe = ring.UploadRaw(nc->data, lw.d_ple_norm_conv, nc->bytes, stream, bytes_acc);
            if (pe == cudaSuccess) pe = ring.UploadRaw(cv->data, lw.d_ple_conv_bf16, cv->bytes, stream, bytes_acc);
            if (pe != cudaSuccess) {
                error = absl::StrCat("ple/layer_", l, ": norm upload failed");
                pw.Free();
                return false;
            }
            lw.ple_conv_bytes = static_cast<int64_t>(cv->bytes);
            lw.ple_conv_state_len =
                (pw.exp_top.ple_conv_kernel_size - 1) * pw.exp_top.ngram_size;
            lw.is_ple_layer = true;
        }

        lw.moe_loaded = true;
    }

    pw.resident_bytes = bytes_acc;
    return true;
}

// ── Routed-expert host store ──────────────────────────────────────────────

void ExpExpertStore::Free()
{
    // All layer slabs share one pinned arena; release it once.
    if (arena_base) {
        cudaFreeHost(arena_base);
        arena_base = nullptr;
    }
    arena_bytes = 0;
    for (auto &p : h_slabs)
        p = nullptr;
    slab_bytes.clear();
    records.clear();
    if (d_seg_wgs) cudaFree(d_seg_wgs);
    d_seg_wgs = nullptr;
    if (d_gemv_meta) cudaFree(d_gemv_meta);
    d_gemv_meta = nullptr;
    h_gemv_meta.clear();
    gemv_ready = false;
    num_layers = 0;
    num_experts = 0;
}

bool LoadQwen4ExpExpertStore(const std::string &model_path,
                             ExpExpertStore &store, std::string &error)
{
    namespace fs = std::filesystem;
    store.Free();
    error.clear();

    fs::path base = fs::path(model_path) / "logestix";
    const fs::path experts_dir = base / "experts";
    if (!fs::exists(experts_dir)) {
        error = "no experts dir under " + base.string();
        return false;
    }

    // Discover layer count from directory names.
    int max_layer = -1;
    for (fs::directory_iterator it(experts_dir), end; it != end; ++it) {
        const std::string name = it->path().filename().string();
        if (name.rfind("layer_", 0) == 0) {
            int l = 0;
            if (absl::SimpleAtoi(name.substr(6), &l))
                max_layer = std::max(max_layer, l);
        }
    }
    if (max_layer < 0) {
        error = "no expert layer dirs under " + experts_dir.string();
        return false;
    }
    store.num_layers = max_layer + 1;
    store.records.resize(static_cast<size_t>(store.num_layers));
    store.h_slabs.assign(static_cast<size_t>(store.num_layers), nullptr);
    store.slab_bytes.assign(static_cast<size_t>(store.num_layers), 0);

    // First pass: build all records with GLOBAL arena offsets and measure
    // per-layer sizes so a single pinned arena can back every slab.
    // Per-layer cudaMallocHost of ~1.4GB slabs costs ~660ms page-locking
    // each (31.8s over 48 layers); one big allocation pays it exactly once.
    size_t arena_bytes = 0;
    std::vector<size_t> layer_totals(static_cast<size_t>(store.num_layers), 0);
    for (int l = 0; l < store.num_layers; ++l) {
        const fs::path layer_dir = experts_dir / absl::StrCat("layer_", l);

        // Expert count from files present in layer 0; strict afterwards.
        int E = store.num_experts;
        if (E <= 0) {
            E = 0;
            for (fs::directory_iterator it(layer_dir), end; it != end; ++it) {
                const std::string name = it->path().filename().string();
                if (name.rfind("exp_", 0) == 0 && name.find(".arrow") != std::string::npos) {
                    int e = 0;
                    if (absl::SimpleAtoi(name.substr(4, name.find('.') - 4), &e))
                        E = std::max(E, e + 1);
                }
            }
            store.num_experts = E;
        }
        if (E <= 0) {
            error = absl::StrCat("no expert files in layer_", l);
            return false;
        }
        if (store.num_experts != E) {
            error = absl::StrCat("layer_", l, ": expert count ", E,
                                 " != ", store.num_experts);
            return false;
        }

        auto &recs = store.records[static_cast<size_t>(l)];
        recs.assign(static_cast<size_t>(E), {});
        size_t total = 0;
        for (int e = 0; e < E; ++e) {
            const std::string path =
                (layer_dir / absl::StrCat("exp_", e, ".arrow")).string();
            auto batch_res = ReadRecordBatchFromFile(path);
            if (!batch_res.ok()) {
                error = absl::StrCat("read failed: ", path);
                return false;
            }
            auto pe_res = ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) {
                error = absl::StrCat("unpack failed: ", path);
                return false;
            }
            auto &pe = *pe_res;
            auto &r = recs[static_cast<size_t>(e)];
            // Arena layout mirrors the device slot layout EXACTLY
            // ([w13 = w1|w3][w2] packed, [s13 = s1|s3][s2] scales): expert
            // miss staging then needs just 2 memcpyAsync (pack + scales)
            // instead of 6 segment copies — the single largest host-side
            // decode cost at high miss rates.
            r.w1_off = static_cast<int64_t>(arena_bytes + total);
            r.w1_bytes = static_cast<int64_t>(pe.w1_packed.size());
            total += r.w1_bytes;
            r.w3_off = static_cast<int64_t>(arena_bytes + total);
            r.w3_bytes = static_cast<int64_t>(pe.w3_packed.size());
            total += r.w3_bytes;
            r.w2_off = static_cast<int64_t>(arena_bytes + total);
            r.w2_bytes = static_cast<int64_t>(pe.w2_packed.size());
            total += r.w2_bytes;
            r.gs1 = pe.w1_global_scale;
            r.gs2 = pe.w2_global_scale;
            r.gs3 = pe.w3_global_scale;
            // Scales stored after the packed region with 16B alignment.
            total = AlignUp(total, 16);
            r.s1_off = static_cast<int64_t>(arena_bytes + total);
            r.s1_bytes = static_cast<int64_t>(pe.w1_scales.size());
            total += r.s1_bytes;
            r.s3_off = static_cast<int64_t>(arena_bytes + total);
            r.s3_bytes = static_cast<int64_t>(pe.w3_scales.size());
            total += r.s3_bytes;
            r.s2_off = static_cast<int64_t>(arena_bytes + total);
            r.s2_bytes = static_cast<int64_t>(pe.w2_scales.size());
            total += r.s2_bytes;
        }
        layer_totals[static_cast<size_t>(l)] = total;
        store.slab_bytes[static_cast<size_t>(l)] = total;
        arena_bytes += total;
    }

    // Single pinned arena backing all layer slabs; every slab points at the
    // arena base and record offsets are already global.
    uint8_t *arena = nullptr;
    if (cudaMallocHost(reinterpret_cast<void **>(&arena), arena_bytes) !=
        cudaSuccess) {
        error = absl::StrCat("pinned expert arena alloc failed (",
                             arena_bytes, " bytes)");
        return false;
    }
    store.arena_base = arena;
    store.arena_bytes = arena_bytes;

    // Second pass: stream expert payloads straight into the arena.
    for (int l = 0; l < store.num_layers; ++l) {
        const fs::path layer_dir = experts_dir / absl::StrCat("layer_", l);
        const int E = store.num_experts;
        auto &recs = store.records[static_cast<size_t>(l)];
        for (int e = 0; e < E; ++e) {
            const std::string path =
                (layer_dir / absl::StrCat("exp_", e, ".arrow")).string();
            auto batch_res = ReadRecordBatchFromFile(path);
            if (!batch_res.ok()) {
                error = absl::StrCat("read failed: ", path);
                return false;
            }
            auto pe_res = ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) {
                error = absl::StrCat("unpack failed: ", path);
                return false;
            }
            auto &pe = *pe_res;
            const auto &r = recs[static_cast<size_t>(e)];
            std::memcpy(arena + r.w1_off, pe.w1_packed.data(), r.w1_bytes);
            std::memcpy(arena + r.w3_off, pe.w3_packed.data(), r.w3_bytes);
            std::memcpy(arena + r.w2_off, pe.w2_packed.data(), r.w2_bytes);
            std::memcpy(arena + r.s1_off, pe.w1_scales.data(), r.s1_bytes);
            std::memcpy(arena + r.s3_off, pe.w3_scales.data(), r.s3_bytes);
            std::memcpy(arena + r.s2_off, pe.w2_scales.data(), r.s2_bytes);
        }
        store.h_slabs[static_cast<size_t>(l)] = arena;
        log::info(absl::StrCat("[qwen_moe_exp_load] layer_", l, ": ", E,
                               " experts, ", layer_totals[static_cast<size_t>(l)] >> 10,
                               " KiB pinned (shared arena)"));
    }

    // Device-side per-expert {gs1, gs3} pairs for the fused w13 GEMM
    // epilogue: the kernel treats a non-null seg_wgs with null bounds as a
    // two-segment split at N/2 (gate | up).
    const int flats = store.num_layers * store.num_experts;
    if (flats > 0) {
        std::vector<float> hw(static_cast<size_t>(flats) * 2);
        for (int l = 0; l < store.num_layers; ++l)
            for (int e = 0; e < store.num_experts; ++e) {
                const auto &r =
                    store.records[static_cast<size_t>(l)]
                                 [static_cast<size_t>(e)];
                hw[(static_cast<size_t>(l) * store.num_experts + e) * 2] =
                    static_cast<float>(r.gs1);
                hw[(static_cast<size_t>(l) * store.num_experts + e) * 2 + 1] =
                    static_cast<float>(r.gs3);
            }
        if (cudaMalloc(&store.d_seg_wgs, hw.size() * sizeof(float)) !=
            cudaSuccess)
            return false;
        if (cudaMemcpy(store.d_seg_wgs, hw.data(), hw.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }
    return true;
}

bool AllocateQwen4ExpExpertCache(ExpExpertStore &store, int slots,
                                 gpu::ExpertDeviceCache &cache,
                                 std::string &error)
{
    if (slots <= 0 || store.num_layers <= 0 || store.num_experts <= 0 ||
        store.h_slabs.empty() || !store.h_slabs[0]) {
        error = "expert store not loaded";
        return false;
    }
    // Slot strides sized to the largest expert payload across layers.
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < store.num_layers; ++l) {
        for (const auto &r : store.records[static_cast<size_t>(l)]) {
            const size_t pack = static_cast<size_t>(r.w1_bytes + r.w2_bytes +
                                                    r.w3_bytes);
            const size_t scale = static_cast<size_t>(r.s1_bytes + r.s2_bytes +
                                                     r.s3_bytes);
            max_pack = std::max(max_pack, pack);
            max_scale = std::max(max_scale, scale);
        }
    }
    cache = gpu::ExpertDeviceCache{};
    cache.slots = slots;
    cache.total_experts = store.num_layers * store.num_experts;
    cache.pack_stride = AlignUp(max_pack, kArenaAlign);
    cache.scale_stride = AlignUp(max_scale, kArenaAlign);
    const size_t bytes = cache.DeviceBytes();

    void *base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        error = absl::StrCat("expert device block alloc failed: ",
                             cudaGetErrorString(cudaGetLastError()),
                             " (", (bytes >> 20), " MiB requested)");
        return false;
    }
    cache.d_base = static_cast<uint8_t *>(base);
    cache.d_pack = cache.d_base;
    cache.d_scale = cache.d_base +
                    static_cast<int64_t>(slots) * cache.pack_stride;
    // Host LRU bookkeeping owned by the cache allocation.
    cache.h_slot_of = new int32_t[static_cast<size_t>(cache.total_experts)];
    cache.h_slot_owner = new int32_t[static_cast<size_t>(slots)];
    cache.h_lru = new int32_t[static_cast<size_t>(slots)];
    // O(1) recency chain + free-slot stack (the legacy h_lru array is kept
    // for diagnostics; per-insert linear scans over 8k+ slots dominated the
    // host loop at high miss rates).
    cache.h_prev = new int32_t[static_cast<size_t>(slots)];
    cache.h_next = new int32_t[static_cast<size_t>(slots)];
    cache.h_free = new int32_t[static_cast<size_t>(slots)];
    for (int i = 0; i < cache.total_experts; ++i)
        cache.h_slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) {
        cache.h_slot_owner[i] = -1;
        cache.h_lru[i] = i;
        // All slots start free: stack in descending order pops 0 first
        // (lowest-free-slot-first semantics preserved).
        cache.h_free[i] = slots - 1 - i;
        cache.h_prev[i] = -1;
        cache.h_next[i] = -1;
    }
    cache.h_free_top = slots;
    cache.h_lru_front = -1;
    cache.h_lru_back = -1;
    cache.hits = 0;
    cache.misses = 0;
    return true;
}

void FreeQwen4ExpExpertCache(gpu::ExpertDeviceCache &cache)
{
    if (cache.d_base) {
        cudaFree(cache.d_base);
        cache.d_base = nullptr;
    }
    delete[] cache.h_slot_of;
    delete[] cache.h_slot_owner;
    delete[] cache.h_lru;
    delete[] cache.h_prev;
    delete[] cache.h_next;
    delete[] cache.h_free;
    cache.h_slot_of = nullptr;
    cache.h_slot_owner = nullptr;
    cache.h_lru = nullptr;
    cache.h_prev = nullptr;
    cache.h_next = nullptr;
    cache.h_free = nullptr;
    cache.h_free_top = 0;
    cache.h_lru_front = -1;
    cache.h_lru_back = -1;
    cache.d_pack = nullptr;
    cache.d_scale = nullptr;
    cache.slots = 0;
}

int EnsureQwen4ExpExpertResident(ExpExpertStore &store, int layer, int eid,
                                 gpu::ExpertDeviceCache &cache,
                                 cudaStream_t stream, std::string &error,
                                 cudaStream_t h2d_stream, bool *missed)
{
    if (missed) *missed = false;
    if (layer < 0 || layer >= store.num_layers || eid < 0 ||
        eid >= store.num_experts) {
        error = absl::StrCat("expert out of range: layer ", layer, " eid ", eid);
        return -1;
    }
    const int flat = layer * store.num_experts + eid;
    int slot = cache.Lookup(flat);
    if (slot >= 0) return slot;
    slot = cache.Insert(flat);
    if (missed) *missed = true;

    // No-handshake decode bookkeeping: mirror residency + scales so the
    // batched GEMV path can compose its table entirely on device.
    if (static_cast<int>(store.h_gemv_meta.size()) <
        store.num_layers * store.num_experts)
        store.h_gemv_meta.assign(
            static_cast<size_t>(store.num_layers) * store.num_experts,
            ExpExpertMetaDev{-1, 0.f, 0.f, 0.f});
    {
        const auto &rec =
            store.records[static_cast<size_t>(layer)]
                         [static_cast<size_t>(eid)];
        store.h_gemv_meta[static_cast<size_t>(flat)] =
            ExpExpertMetaDev{slot, rec.gs1, rec.gs2, rec.gs3};
        store.gemv_ready = false;  // invalidate until re-uploaded
    }

    const uint8_t *slab = store.h_slabs[static_cast<size_t>(layer)];
    if (!slab) {
        error = "expert slab not allocated";
        return -1;
    }
    const auto &r = store.records[static_cast<size_t>(layer)]
                                  [static_cast<size_t>(eid)];
    uint8_t *dst_pack =
        cache.d_pack + static_cast<int64_t>(slot) * cache.pack_stride;
    uint8_t *dst_scale =
        cache.d_scale + static_cast<int64_t>(slot) * cache.scale_stride;
    // Slot layout is FUSED for inference and MIRRORS the pinned arena layout
    // exactly: [w13 = w1|w3][w2] packed and [s13 = s1|s3][s2] scales are each
    // ONE contiguous blob on the host, so miss staging is 2 memcpyAsync total
    // (pack + scales) instead of 6 segment copies. The gate/up projections
    // then run as ONE GEMM with per-segment global scales (RunMoERoutedFFN).
    cudaStream_t copy_stream = h2d_stream ? h2d_stream : stream;
    const int64_t pack_blob = r.w1_bytes + r.w3_bytes + r.w2_bytes;
    const int64_t scale_blob = r.s1_bytes + r.s3_bytes + r.s2_bytes;
    if (cudaMemcpyAsync(dst_pack, slab + r.w1_off,
                        static_cast<size_t>(pack_blob),
                        cudaMemcpyHostToDevice, copy_stream) != cudaSuccess) {
        error = "expert H2D pack staging failed";
        return -1;
    }
    if (cudaMemcpyAsync(dst_scale, slab + r.s1_off,
                        static_cast<size_t>(scale_blob),
                        cudaMemcpyHostToDevice, copy_stream) != cudaSuccess) {
        error = "expert H2D scale staging failed";
        return -1;
    }
    return slot;
}

// ── PLE shard mmap ────────────────────────────────────────────────────────

bool LoadQwen4ExpPleShards(const std::string &model_path, ExpPinnedWeights &pw,
                           ExpPleShardView *shards, std::string &error)
{
    namespace fs = std::filesystem;
    error.clear();
    const int parts = pw.exp_top.split_ngram_parts;
    if (parts <= 0 || !shards) {
        error = "PLE shard count unknown or output array null";
        return false;
    }
    const fs::path ple_dir = fs::path(model_path) / "logestix" / "ple";

    for (int s = 0; s < parts; ++s) {
        const fs::path path = ple_dir / absl::StrCat("shard_", s, ".arrow");
        if (!fs::exists(path)) {
            error = "PLE shard missing: " + path.string();
            return false;
        }
#ifdef _WIN32
        // Windows path: read via Arrow mmap-backed loader (no raw Win32
        // MapViewOfFile in this path; Arrow's MemoryMappedFile covers it).
        auto res = LoadTensorBatchViews(path);
        if (!res.ok()) {
            error = "PLE shard open failed: " + path.string();
            return false;
        }
        // Keep the whole batch views alive; expose the first tensor's bytes.
        auto keep = res->keepalive;
        const auto &tv = res->tensors[0];
        shards[s].data = tv.data;
        shards[s].bytes = tv.bytes;
        shards[s].shard = s;
        shards[s].keepalive = keep;
#else
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            error = "PLE shard open failed: " + path.string();
            return false;
        }
        const off_t size = ::lseek(fd, 0, SEEK_END);
        if (size <= 0) {
            ::close(fd);
            error = "PLE shard empty: " + path.string();
            return false;
        }
        void *map = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ,
                           MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) {
            error = "PLE shard mmap failed: " + path.string();
            return false;
        }
        // Locate the first tensor payload: skip the safetensors-style arrow
        // container? No — these files are Arrow IPC, so parse the batch and
        // point at the data buffer inside the mapping.
        auto res = LoadTensorBatchViews(path);
        if (!res.ok()) {
            ::munmap(map, static_cast<size_t>(size));
            error = "PLE shard parse failed: " + path.string();
            return false;
        }
        // NOTE: LoadTensorBatchViews mmaps the file itself via Arrow; the
        // raw ::mmap above is redundant. Release it and rely on the Arrow
        // keepalive (zero-copy view loading, etalon commit 51c42d0).
        ::munmap(map, static_cast<size_t>(size));
        const auto &tv = res->tensors[0];
        shards[s].data = tv.data;
        shards[s].bytes = tv.bytes;
        shards[s].shard = s;
        shards[s].keepalive = res->keepalive;
#endif
    }
    return true;
}

std::size_t GenContextBytes(int max_seq)
{
    using namespace frozen;
    if (max_seq <= 0) return 0;

    const int H = kHidden, HCH = kHcHidden, NH = kHeads, HD = kHeadDim;
    const int NKV = kKVHeads, rot = kRotaryDim;
    const int KD = kLinearNumKeyHeads * kLinearKeyHeadDim;
    const int VD = kLinearNumValueHeads * kLinearValueHeadDim;
    const int S = KD * 2 + VD;
    const int nv = kLinearNumValueHeads;

    // Arena carving replica (qwen_moe_exp_inference.cu RunExpGeneration):
    // the __half and float sections of the persistent scratch.
    std::size_t half_per_tok =
        2 * (std::size_t)H + 3 * (std::size_t)HCH +
        3 * (std::size_t)(NH * HD) + 2 * (std::size_t)(NH * HD * 2) +
        2 * (std::size_t)(NKV * HD) + 2 * (std::size_t)rot +
        2 * (std::size_t)S + 3 * (std::size_t)VD + 2 * (std::size_t)nv;
    std::size_t half_fixed =
        (std::size_t)kTopK * 2 * kMoeIntermediate +
        (std::size_t)kTopK * kMoeIntermediate + (std::size_t)kTopK * H +
        2 * (std::size_t)kMoeIntermediate + 2 * (std::size_t)H +
        kSharedIntermediate;
    std::size_t float_per_tok = 2 * (std::size_t)kVocab +
                                2 * (std::size_t)H + 16 + 2560 +
                                2 * (std::size_t)nv;
    const std::size_t half_bytes =
        ((half_per_tok * (std::size_t)max_seq + half_fixed) * sizeof(__half) +
         255) &
        ~(std::size_t)255;
    const std::size_t float_bytes =
        ((float_per_tok * (std::size_t)max_seq + 1024) * sizeof(float) + 255) &
        ~(std::size_t)255;

    // KV history (QwenKvCache::Allocate): full-attention k/v + linear qkv
    // history scale with max_seq; the linear GDN column state does not.
    constexpr int kFull = kLayers / kFullAttnInterval;
    constexpr int kLin = kLayers - kFull;
    const std::size_t kv_bytes =
        2 * (std::size_t)kFull * (std::size_t)max_seq * NKV * HD *
            sizeof(__half) +
        (std::size_t)kLin * (std::size_t)max_seq * S * sizeof(__half) +
        (std::size_t)kLin * nv * kLinearValueHeadDim * kLinearKeyHeadDim *
            kLinearNumKeyHeads * sizeof(float);

    return half_bytes + float_bytes + 4096 + kv_bytes;
}

int SizeExpertCacheSlots(std::size_t free_bytes, std::size_t per_expert_bytes,
                         int total_experts, int max_seq, int slots_pct)
{
    // max_seq-independent post-cache floor: persistent fp16 embed table
    // (V x H x 2B ~ 1.2 GiB), per-layer TC fused builds (~1.6 GiB), driver
    // margin. Conservative: under-reserving repeats the max-gen-2048 OOM.
    constexpr std::size_t kPostCacheFloorBytes = 4ull << 30;
    const std::size_t reserve =
        GenContextBytes(max_seq) + kPostCacheFloorBytes;
    return lifecycle::ExpertCacheSlots(free_bytes, per_expert_bytes,
                                       total_experts, slots_pct, reserve);
}

} // namespace lsxcommon::qwen_moe_exp
