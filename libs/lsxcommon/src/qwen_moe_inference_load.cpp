#include "lsxcommon/qwen_moe_inference_load.h"

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "fast_load_internal.h"

#include <absl/strings/str_cat.h>
#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace lsxcommon::qwen_moe {
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

// Dtype-aware kind for F32-destined tensors: honor BF16-on-disk records
// (e.g. A_log in real Qwen checkpoints) instead of reinterpreting bits.
CvKind CvKindFromDtype(const TensorView *t, CvKind f32_kind)
{
    return (t && t->dtype == "BF16") ? CvKind::kBf16ToF32 : f32_kind;
}

// Parses the superset index (format logestix-qwen35moe) into both topology
// views. Returns false with `error` filled on structural problems.
bool ParseMoeIndex(const std::filesystem::path &idx_path,
                   QwenMoeTopology &moe, qwen::QwenTopology &dense,
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

    moe.num_layers = gi("num_layers", 0);
    moe.hidden_size = gi("hidden_size", 0);
    moe.vocab_size = gi("vocab_size", 0);
    moe.num_attention_heads = gi("num_attention_heads", 0);
    moe.num_key_value_heads = gi("num_key_value_heads", 0);
    moe.head_dim = gi("head_dim", 0);
    moe.linear_num_key_heads = gi("linear_num_key_heads", 0);
    moe.linear_num_value_heads = gi("linear_num_value_heads", 0);
    moe.linear_key_head_dim = gi("linear_key_head_dim", 0);
    moe.linear_value_head_dim = gi("linear_value_head_dim", 0);
    moe.linear_conv_kernel_dim = gi("linear_conv_kernel_dim", 4);
    moe.rms_norm_eps = gd("rms_norm_eps", 1e-6);
    moe.partial_rotary_factor = gd("partial_rotary_factor", 0.25);
    moe.rope_theta = gd("rope_theta", 1e7);
    moe.num_experts = gi("num_experts", 0);
    moe.num_experts_per_tok = gi("num_experts_per_tok", 0);
    moe.moe_intermediate_size = gi("moe_intermediate_size", 0);
    moe.shared_expert_intermediate_size =
        gi("shared_expert_intermediate_size", 0);
    moe.eos_token_id = gi("eos_token_id", -1);

    if (auto lt = root["layer_types"].get_array();
        lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                moe.layer_types.emplace_back(sv);
        }
    }

    // Dense mirror for the reused qwen kernels.
    dense.num_layers = moe.num_layers;
    dense.layer_types = moe.layer_types;
    dense.hidden_size = moe.hidden_size;
    dense.vocab_size = moe.vocab_size;
    dense.num_attention_heads = moe.num_attention_heads;
    dense.num_key_value_heads = moe.num_key_value_heads;
    dense.head_dim = moe.head_dim;
    dense.linear_num_key_heads = moe.linear_num_key_heads;
    dense.linear_num_value_heads = moe.linear_num_value_heads;
    dense.linear_key_head_dim = moe.linear_key_head_dim;
    dense.linear_value_head_dim = moe.linear_value_head_dim;
    dense.linear_conv_kernel_dim = moe.linear_conv_kernel_dim;
    dense.rms_norm_eps = moe.rms_norm_eps;
    dense.partial_rotary_factor = moe.partial_rotary_factor;
    dense.rope_theta = moe.rope_theta;

    if (moe.num_layers <= 0 || moe.num_experts <= 0 ||
        static_cast<int>(moe.layer_types.size()) != moe.num_layers) {
        error = "index missing layers/experts fields";
        return false;
    }
    return true;
}

} // namespace

void QwenMoeLayerWeightsFree(QwenMoeLayerWeights &lw)
{
    lw.attn.Free();
    if (lw.d_router) { cudaFree(lw.d_router); lw.d_router = nullptr; }
    lw.router_bytes = 0;
    lw.shared_w1.Free();
    lw.shared_w2.Free();
    lw.shared_w3.Free();
    if (lw.d_shared_gate) { cudaFree(lw.d_shared_gate); lw.d_shared_gate = nullptr; }
    lw.moe_loaded = false;
}

void QwenMoePinnedWeights::Free()
{
    embed.Free();
    head.Free();
    if (d_final_norm) { cudaFree(d_final_norm); d_final_norm = nullptr; }
    final_norm_size = 0;
    for (auto &l : layers) QwenMoeLayerWeightsFree(l);
    layers.clear();
    resident_bytes = 0;
}

void FreeQwenMoePinnedWeights(QwenMoePinnedWeights &pw) { pw.Free(); }

bool LoadQwenMoeDenseWeights(const std::string &model_path,
                             QwenMoePinnedWeights &pw, cudaStream_t stream,
                             std::string &error)
{
    namespace fs = std::filesystem;
    pw.Free();
    error.clear();

    fs::path base = fs::path(model_path) / "logestix";
    if (!fs::exists(base))
        base = fs::path(model_path) / "logestix-qwen35moe";  // legacy layout
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
        if (!ParseMoeIndex(idx, pw.moe_top, pw.top, idx_error)) {
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

    // Globals.
    {
        auto raw_res = LoadTensorBatchViews(pinned_dir / "global.arrow");
        if (!raw_res.ok()) {
            error = "global.arrow read failed: " + raw_res.status().ToString();
            return false;
        }
        auto *fn = fastload::FindTensorSuffix(*raw_res, "norm.weight");
        if (fn && fn->rank > 0) {
            int64_t n = fn->shape[0];
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
        } else {
            error = "final norm missing in global.arrow";
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
        if (e == cudaSuccess)
            e = UploadNvfp4Weight(fastload::FindNvfp4Suffix(*nv_res, "lm_head.weight"),
                                  pw.head, stream, ring, bytes_acc, &dummy);
        if (e != cudaSuccess || !pw.embed.d_packed || !pw.head.d_packed) {
            error = "embed/lm_head missing or upload failed";
            pw.Free();
            return false;
        }
    }

    // Per-layer dense parts. Record names are identical to the qwen layout.
    const int num_layers = pw.top.num_layers;
    if (num_layers <= 0) {
        error = "topology num_layers unknown";
        pw.Free();
        return false;
    }
    pw.layers.resize(static_cast<size_t>(num_layers));

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
        if (!attn_ok || !lw.attn.d_input_norm || !lw.attn.d_post_norm) {
            error = absl::StrCat("layer_", l, ": attention weights incomplete");
            pw.Free();
            return false;
        }
        lw.attn.loaded = true;

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
        lw.moe_loaded = true;
    }

    pw.resident_bytes = bytes_acc;
    return true;
}

// ── Routed-expert host store ──────────────────────────────────────────────

void QwenMoeExpertStore::Free()
{
    for (auto &p : h_slabs)
        if (p) cudaFreeHost(p);
    h_slabs.clear();
    slab_bytes.clear();
    records.clear();
    if (d_seg_wgs) cudaFree(d_seg_wgs);
    if (d_seg_bounds) cudaFree(d_seg_bounds);
    d_seg_wgs = nullptr;
    d_seg_bounds = nullptr;
    num_layers = 0;
    num_experts = 0;
}

bool LoadQwenMoeExpertStore(const std::string &model_path,
                            QwenMoeExpertStore &store, std::string &error)
{
    namespace fs = std::filesystem;
    store.Free();
    error.clear();

    fs::path base = fs::path(model_path) / "logestix";
    if (!fs::exists(base))
        base = fs::path(model_path) / "logestix-qwen35moe";  // legacy layout
    const fs::path experts_dir = base / "experts";

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

        // Build all records, staging payloads once (single disk pass), then
        // one pinned slab per layer.
        struct StagedExpert {
            std::vector<uint8_t> w1, w2, w3, s1, s2, s3;
        };
        std::vector<StagedExpert> staged(static_cast<size_t>(E));
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
            auto &st = staged[static_cast<size_t>(e)];
            st.w1 = std::move(pe.w1_packed);
            st.w2 = std::move(pe.w2_packed);
            st.w3 = std::move(pe.w3_packed);
            st.s1 = std::move(pe.w1_scales);
            st.s2 = std::move(pe.w2_scales);
            st.s3 = std::move(pe.w3_scales);
            auto &r = recs[static_cast<size_t>(e)];
            r.w1_off = total; r.w1_bytes = static_cast<int64_t>(st.w1.size());
            total += r.w1_bytes;
            r.w2_off = total; r.w2_bytes = static_cast<int64_t>(st.w2.size());
            total += r.w2_bytes;
            r.w3_off = total; r.w3_bytes = static_cast<int64_t>(st.w3.size());
            total += r.w3_bytes;
            r.gs1 = pe.w1_global_scale;
            r.gs2 = pe.w2_global_scale;
            r.gs3 = pe.w3_global_scale;
            // Scales stored after the packed region with 16B alignment.
            total = AlignUp(total, 16);
            r.s1_off = total; r.s1_bytes = static_cast<int64_t>(st.s1.size()); total += r.s1_bytes;
            r.s2_off = total; r.s2_bytes = static_cast<int64_t>(st.s2.size()); total += r.s2_bytes;
            r.s3_off = total; r.s3_bytes = static_cast<int64_t>(st.s3.size()); total += r.s3_bytes;
        }
        store.slab_bytes[static_cast<size_t>(l)] = total;

        uint8_t *slab = nullptr;
        if (cudaMallocHost(reinterpret_cast<void **>(&slab), total) !=
            cudaSuccess) {
            error = absl::StrCat("pinned slab alloc failed (layer ", l, ")");
            return false;
        }
        store.h_slabs[static_cast<size_t>(l)] = slab;

        for (int e = 0; e < E; ++e) {
            const auto &st = staged[static_cast<size_t>(e)];
            const auto &r = recs[static_cast<size_t>(e)];
            std::memcpy(slab + r.w1_off, st.w1.data(), r.w1_bytes);
            std::memcpy(slab + r.w2_off, st.w2.data(), r.w2_bytes);
            std::memcpy(slab + r.w3_off, st.w3.data(), r.w3_bytes);
            std::memcpy(slab + r.s1_off, st.s1.data(), r.s1_bytes);
            std::memcpy(slab + r.s2_off, st.s2.data(), r.s2_bytes);
            std::memcpy(slab + r.s3_off, st.s3.data(), r.s3_bytes);
        }
        staged.clear();
        staged.shrink_to_fit();
        log::info(absl::StrCat("[qwen_moe_load] layer_", l, ": ", E,
                               " experts, ", total >> 10, " KiB pinned"));
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

bool AllocateQwenMoeExpertCache(QwenMoeExpertStore &store, int slots,
                                QwenMoeExpertDeviceCache &cache,
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
    for (int i = 0; i < cache.total_experts; ++i)
        cache.h_slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) {
        cache.h_slot_owner[i] = -1;
        cache.h_lru[i] = i;
    }
    cache.hits = 0;
    cache.misses = 0;
    return true;
}

void FreeQwenMoeExpertCache(QwenMoeExpertDeviceCache &cache)
{
    if (cache.d_base) {
        cudaFree(cache.d_base);
        cache.d_base = nullptr;
    }
    delete[] cache.h_slot_of;
    delete[] cache.h_slot_owner;
    delete[] cache.h_lru;
    cache.h_slot_of = nullptr;
    cache.h_slot_owner = nullptr;
    cache.h_lru = nullptr;
    cache.d_pack = nullptr;
    cache.d_scale = nullptr;
    cache.slots = 0;
}

int EnsureQwenMoeExpertResident(QwenMoeExpertStore &store, int layer, int eid,
                                QwenMoeExpertDeviceCache &cache,
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
    int slot = QwenMoeExpertCacheLookup(cache, flat);
    if (slot >= 0) return slot;
    slot = QwenMoeExpertCacheInsert(cache, flat);
    if (missed) *missed = true;

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
    // Slot layout is FUSED for inference: [w13 = w1|w3][w2] and
    // [s13 = s1|s3][s2] so the gate/up projections run as ONE GEMM with
    // per-segment global scales (RunQwenMoeRoutedFFN). The pinned slab keeps
    // the on-disk order; only this staging reorder differs.
    cudaStream_t copy_stream = h2d_stream ? h2d_stream : stream;
    uint8_t *pk = dst_pack;
    struct PackSeg { int64_t off, bytes; };
    const PackSeg psegs[3] = {{r.w1_off, r.w1_bytes},
                              {r.w3_off, r.w3_bytes},
                              {r.w2_off, r.w2_bytes}};
    for (const auto &seg : psegs) {
        if (cudaMemcpyAsync(pk, slab + seg.off,
                            static_cast<size_t>(seg.bytes),
                            cudaMemcpyHostToDevice,
                            copy_stream) != cudaSuccess) {
            error = "expert H2D pack staging failed";
            return -1;
        }
        pk += seg.bytes;
    }
    uint8_t *sc = dst_scale;
    const PackSeg ssegs[3] = {{r.s1_off, r.s1_bytes},
                              {r.s3_off, r.s3_bytes},
                              {r.s2_off, r.s2_bytes}};
    for (const auto &seg : ssegs) {
        if (cudaMemcpyAsync(sc, slab + seg.off,
                            static_cast<size_t>(seg.bytes),
                            cudaMemcpyHostToDevice,
                            copy_stream) != cudaSuccess) {
            error = "expert H2D scale staging failed";
            return -1;
        }
        sc += seg.bytes;
    }
    return slot;
}

} // namespace lsxcommon::qwen_moe
