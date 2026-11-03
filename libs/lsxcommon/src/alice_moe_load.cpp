#include "lsxcommon/alice_moe_load.h"

#include "alice_moe_constants.h"

#include <simdjson.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

namespace lsxcommon::alice {
namespace {

namespace fs = std::filesystem;

bool get_i(simdjson::ondemand::object &o, const char *k, int &out)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return false;
    out = static_cast<int>(v);
    return true;
}

bool get_d(simdjson::ondemand::object &o, const char *k, double &out)
{
    simdjson::ondemand::value x;
    if (o[k].get(x) != simdjson::SUCCESS) return false;
    if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) {
        out = d.value();
        return true;
    }
    if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS) {
        out = static_cast<double>(i.value());
        return true;
    }
    return false;
}

} // namespace

arrow::Status LoadAliceTopology(const std::string &model_path,
                                AliceMoeTopology &out)
{
    out = AliceMoeTopology{};
    const fs::path idx = fs::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(idx);
    if (!f)
        return arrow::Status::IOError("cannot open ", idx.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        out.error_msg = "index JSON parse error";
        return arrow::Status::Invalid("alice index: parse error");
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("alice index: root is not an object");

    std::string_view s;
    if (root["format"].get_string().get(s) == simdjson::SUCCESS)
        out.format.assign(s.data(), s.size());
    if (out.format != "logestix-alice-nvfp4") {
        out.error_msg = "unexpected format: " + out.format;
        return arrow::Status::Invalid("alice loader: expected format "
                                      "logestix-alice-nvfp4, got ",
                                      out.format);
    }
    if (root["model"].get_string().get(s) == simdjson::SUCCESS)
        out.model.assign(s.data(), s.size());

    get_i(root, "num_layers", out.num_layers);
    get_i(root, "hidden_size", out.hidden_size);
    get_i(root, "vocab_size", out.vocab_size);
    get_i(root, "num_attention_heads", out.num_attention_heads);
    get_i(root, "num_key_value_heads", out.num_key_value_heads);
    get_i(root, "head_dim", out.head_dim);
    get_i(root, "linear_num_key_heads", out.linear_num_key_heads);
    get_i(root, "linear_num_value_heads", out.linear_num_value_heads);
    get_i(root, "linear_key_head_dim", out.linear_key_head_dim);
    get_i(root, "linear_value_head_dim", out.linear_value_head_dim);
    get_i(root, "linear_conv_kernel_dim", out.linear_conv_kernel_dim);
    get_i(root, "num_experts", out.num_experts);
    get_i(root, "num_experts_per_tok", out.num_experts_per_tok);
    get_i(root, "moe_intermediate_size", out.moe_intermediate_size);
    get_i(root, "shared_expert_intermediate_size",
          out.shared_expert_intermediate_size);
    get_i(root, "block_attn_res_block_size", out.block_attn_res_block_size);
    get_i(root, "mtp_num_hidden_layers", out.mtp_num_hidden_layers);
    get_i(root, "bos_id", out.bos_id);
    get_i(root, "eos_id", out.eos_id);
    get_i(root, "pad_id", out.pad_id);
    get_d(root, "rms_norm_eps", out.rms_norm_eps);
    get_d(root, "partial_rotary_factor", out.partial_rotary_factor);
    get_d(root, "rope_theta", out.rope_theta);

    if (auto lt = root["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                out.layer_types.emplace_back(sv);
        }
    }
    if (out.num_layers <= 0 ||
        static_cast<int>(out.layer_types.size()) != out.num_layers)
        return arrow::Status::Invalid(
            "alice index: layer_types size mismatch");
    return arrow::Status::OK();
}

namespace {

namespace fs = std::filesystem;

std::vector<__half> Bf16ToHalf(const std::vector<uint8_t> &data)
{
    const size_t n = data.size() / 2;
    std::vector<__half> out(n);
    for (size_t i = 0; i < n; ++i) {
        const uint16_t b =
            static_cast<uint16_t>(data[2 * i] | (data[2 * i + 1] << 8));
        const uint32_t f = static_cast<uint32_t>(b) << 16;
        float v;
        std::memcpy(&v, &f, 4);
        out[i] = __float2half(v);
    }
    return out;
}

__half *UploadHalf(const std::vector<uint8_t> &data, cudaStream_t s)
{
    if (data.empty()) return nullptr;
    const auto h = Bf16ToHalf(data);
    __half *d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(__half)) != cudaSuccess) return nullptr;
    cudaMemcpyAsync(d, h.data(), h.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, s);
    return d;
}

bool UploadNvfp4(const DenseNvfp4Record &r, lsx_nvfp4::LsxNvfp4Weight &w,
                 cudaStream_t s)
{
    w = lsx_nvfp4::LsxNvfp4Weight{};
    if (r.shape.size() != 2) return false;
    w.rows = static_cast<int>(r.shape[0]);
    w.cols = static_cast<int>(r.shape[1]);
    w.global_scale = r.global_scale;
    void *p = nullptr, *sc = nullptr;
    if (cudaMalloc(&p, r.packed.size()) != cudaSuccess) return false;
    if (cudaMalloc(&sc, r.scales.size()) != cudaSuccess) {
        cudaFree(p);
        return false;
    }
    cudaMemcpyAsync(p, r.packed.data(), r.packed.size(),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(sc, r.scales.data(), r.scales.size(),
                    cudaMemcpyHostToDevice, s);
    w.d_packed = static_cast<const uint8_t *>(p);
    w.d_scales = static_cast<const uint8_t *>(sc);
    w.packed_bytes = static_cast<int64_t>(r.packed.size());
    w.scale_bytes = static_cast<int64_t>(r.scales.size());
    return true;
}

const TensorRecord *FindRaw(const std::vector<TensorRecord> &v,
                            std::string_view name)
{
    for (const auto &r : v)
        if (r.name == name) return &r;
    return nullptr;
}

const DenseNvfp4Record *FindNvfp4(const std::vector<DenseNvfp4Record> &v,
                                  std::string_view name)
{
    for (const auto &r : v)
        if (r.name == name) return &r;
    return nullptr;
}

bool ends_with(std::string_view s, std::string_view suf)
{
    return s.size() >= suf.size() && s.substr(s.size() - suf.size()) == suf;
}

} // namespace

void FreeAlicePinnedWeights(AlicePinnedWeights &pw)
{
    if (pw.embed) cudaFree(pw.embed);
    if (pw.head) cudaFree(pw.head);
    if (pw.final_norm) cudaFree(pw.final_norm);
    if (pw.attnres_final_proj) cudaFree(pw.attnres_final_proj);
    if (pw.attnres_final_norm) cudaFree(pw.attnres_final_norm);
    for (auto &lw : pw.layers) {
        auto free_h = [](__half *p) { if (p) cudaFree(p); };
        free_h(lw.input_norm); free_h(lw.post_norm); free_h(lw.attn_res_norm);
        free_h(lw.mlp_res_norm); free_h(lw.attn_res_proj); free_h(lw.mlp_res_proj);
        free_h(lw.kda_q); free_h(lw.kda_k); free_h(lw.kda_v); free_h(lw.kda_o);
        free_h(lw.kda_fa); free_h(lw.kda_fb); free_h(lw.kda_ga); free_h(lw.kda_gb); free_h(lw.kda_b);
        free_h(lw.fa_q); free_h(lw.fa_k); free_h(lw.fa_v); free_h(lw.fa_o);
        free_h(lw.kda_qconv); free_h(lw.kda_kconv); free_h(lw.kda_vconv);
        free_h(lw.kda_alog); free_h(lw.kda_dt); free_h(lw.kda_onorm);
        free_h(lw.fa_qnorm); free_h(lw.fa_knorm);
        free_h(lw.router_gate); free_h(lw.router_bias); free_h(lw.shared_gate);
        lw.shared_gate_proj.Free(); lw.shared_up_proj.Free();
        lw.shared_down_proj.Free();
    }
    pw = AlicePinnedWeights{};
}

bool LoadAlicePinnedWeights(const std::string &root, AlicePinnedWeights &pw,
                            cudaStream_t stream)
{
    const fs::path base = fs::path(root) / "logestix";
    fs::path pinned = base / "pinned";
    if (!fs::exists(pinned)) pinned = fs::path(root) / "pinned";
    if (!fs::exists(pinned)) return false;

    auto st = LoadAliceTopology(root, pw.top);
    if (!st.ok()) return false;

    auto load_raw_file = [&](const fs::path &p, std::vector<TensorRecord> &out) {
        if (!fs::exists(p)) return true;
        auto batch = ReadRecordBatchFromFile(p);
        if (!batch.ok()) return false;
        auto rows = PinnedSchema::UnpackBatch(*batch);
        if (!rows.ok()) return false;
        out = std::move(*rows);
        return true;
    };
    auto load_nvfp4_file = [&](const fs::path &p,
                               std::vector<DenseNvfp4Record> &out) {
        if (!fs::exists(p)) return true;
        auto batch = ReadRecordBatchFromFile(p);
        if (!batch.ok()) return false;
        auto rows = DenseNvfp4Schema::UnpackBatch(*batch);
        if (!rows.ok()) return false;
        out = std::move(*rows);
        return true;
    };

    // Count at the allocation sites (never a re-derived formula): resident
    // bytes drive the governor lease and the `resident:` line, so an omitted
    // term is a silent under-report, not a cosmetic number.
    std::uint64_t resident = 0;
    auto upload_global = [&](const std::vector<TensorRecord> &rows,
                             std::string_view name) -> __half * {
        const TensorRecord *r = FindRaw(rows, name);
        if (!r) return nullptr;
        __half *p = UploadHalf(r->data, stream);
        if (p) resident += static_cast<std::uint64_t>(r->data.size());
        return p;
    };

    std::vector<TensorRecord> g_raw;
    if (!load_raw_file(pinned / "global.arrow", g_raw)) return false;
    pw.embed = upload_global(g_raw, "model.embed_tokens.weight");
    pw.head = upload_global(g_raw, "lm_head.weight");
    pw.final_norm = upload_global(g_raw, "model.norm.weight");
    pw.attnres_final_proj =
        upload_global(g_raw, "model.attnres_final.res_proj.weight");
    pw.attnres_final_norm =
        upload_global(g_raw, "model.attnres_final.res_norm_weight");
    if (!pw.embed || !pw.head || !pw.final_norm || !pw.attnres_final_proj ||
        !pw.attnres_final_norm)
        return false;

    const int L = pw.top.num_layers;
    pw.layers.resize(L);
    for (int l = 0; l < L; ++l) {
        std::vector<TensorRecord> raw;
        std::vector<DenseNvfp4Record> nv;
        const std::string tag = "model.layers." + std::to_string(l) + ".";
        if (!load_raw_file(pinned / ("layer_" + std::to_string(l) + ".arrow"),
                           raw))
            return false;
        if (!load_nvfp4_file(
                pinned / ("layer_" + std::to_string(l) + "_nvfp4.arrow"), nv))
            return false;

        AliceLayerWeights &lw = pw.layers[l];
        lw.full = (l % 4 == 3);
        auto H = [&](std::string_view suffix) -> __half * {
            const auto *r = FindRaw(raw, tag + std::string(suffix));
            if (!r) return nullptr;
            __half *p = UploadHalf(r->data, stream);
            if (p) resident += static_cast<std::uint64_t>(r->data.size());
            return p;
        };
        auto W = [&](std::string_view suffix,
                     lsx_nvfp4::LsxNvfp4Weight &w) -> bool {
            const auto *r = FindNvfp4(nv, tag + std::string(suffix));
            if (!r || !UploadNvfp4(*r, w, stream)) return false;
            resident += static_cast<std::uint64_t>(w.packed_bytes +
                                                   w.scale_bytes);
            return true;
        };
        lw.input_norm = H("input_layernorm.weight");
        lw.post_norm = H("post_attention_layernorm.weight");
        lw.mlp_res_norm = H("mlp_res_norm_weight");
        lw.mlp_res_proj = H("mlp_res_proj.weight");
        if (l != 0) {
            lw.attn_res_norm = H("attn_res_norm_weight");
            lw.attn_res_proj = H("attn_res_proj.weight");
        }
        if (lw.full) {
            lw.fa_q = H("self_attn.q_proj.weight");
            lw.fa_k = H("self_attn.k_proj.weight");
            lw.fa_v = H("self_attn.v_proj.weight");
            lw.fa_o = H("self_attn.o_proj.weight");
            lw.fa_qnorm = H("self_attn.q_norm.weight");
            lw.fa_knorm = H("self_attn.k_norm.weight");
        } else {
            lw.kda_q = H("linear_attn.q_proj.weight");
            lw.kda_k = H("linear_attn.k_proj.weight");
            lw.kda_v = H("linear_attn.v_proj.weight");
            lw.kda_o = H("linear_attn.o_proj.weight");
            lw.kda_fb = H("linear_attn.f_b_proj.weight");
            lw.kda_gb = H("linear_attn.g_b_proj.weight");
            lw.kda_fa = H("linear_attn.f_a_proj.weight");
            lw.kda_ga = H("linear_attn.g_a_proj.weight");
            lw.kda_b = H("linear_attn.b_proj.weight");
            lw.kda_qconv = H("linear_attn.q_conv1d.weight");
            lw.kda_kconv = H("linear_attn.k_conv1d.weight");
            lw.kda_vconv = H("linear_attn.v_conv1d.weight");
            lw.kda_alog = H("linear_attn.a_log_bias");
            lw.kda_dt = H("linear_attn.dt_bias");
            lw.kda_onorm = H("linear_attn.o_norm.weight");
        }
        lw.router_gate = H("mlp.gate.weight");
        lw.router_bias = H("mlp.gate.e_score_correction_bias");
        lw.shared_gate = H("mlp.shared_expert_gate.weight");
        if (!W("mlp.shared_expert.gate_proj.weight", lw.shared_gate_proj))
            return false;
        if (!W("mlp.shared_expert.up_proj.weight", lw.shared_up_proj))
            return false;
        if (!W("mlp.shared_expert.down_proj.weight", lw.shared_down_proj))
            return false;
        lw.loaded = true;
    }

    pw.resident_bytes = resident;
    return true;
}

AliceExpertDb load_alice_expert_index(const std::string &model_path,
                                      int num_layers, int experts_per_layer,
                                      size_t max_open)
{
    AliceExpertDb db;
    // No per-model parser: the shared ExpertStore<TEntry>::DecodeEntry()
    // extracts the nine ExpertSchema columns into AliceExpertEntry.
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.first_moe_layer = 0;   // every layer routes (block-4 full-attn too)
    db.log_prefix = "[alice_load] ";
    db.mmap_fail_verb = "expert mmap fail ";
    db.emit_diag = false;
    db.log_batch_fails = true;
    // Bound the mapped set. Alice carries 48x512 experts (~38 GiB of NVFP4),
    // far more than any other family, so the store default (65536 = map
    // everything) is not appropriate here: an explicit cap keeps the host
    // mapping bounded and lets the LRU retire the tail. Each entry is a file
    // mmap, so what the cap bounds is mapped file, not heap.
    db.max_open = max_open ? max_open : kAliceExpertMaxOpen;
    db.Load(model_path, /*fallback=*/true);
    return db;
}

void ensure_alice_expert_resident(AliceExpertDb &db, int layer, int expert,
                                  cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_alice_expert_index(AliceExpertDb &db) { db.Free(); }

bool AliceTopologyMatchesFrozen(const AliceMoeTopology &top)
{
    using namespace frozen;
    if (top.hidden_size != kHidden || top.num_layers != kLayers ||
        top.vocab_size != kVocab || top.num_attention_heads != kHeads ||
        top.num_key_value_heads != kKVHeads || top.head_dim != kHeadDim ||
        top.linear_num_key_heads != kLinearKeyHeads ||
        top.linear_num_value_heads != kLinearValueHeads ||
        top.linear_key_head_dim != kLinearKeyHeadDim ||
        top.linear_value_head_dim != kLinearValueHeadDim ||
        top.linear_conv_kernel_dim != kLinearConvKernel ||
        top.num_experts != kNumExperts ||
        top.num_experts_per_tok != kTopK ||
        top.moe_intermediate_size != kMoeIntermediate ||
        top.shared_expert_intermediate_size != kSharedIntermediate ||
        top.block_attn_res_block_size != kBlockAttnRes)
        return false;
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want =
            (i % kFullAttnInterval == kFullAttnInterval - 1)
                ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want) return false;
    }
    return true;
}

} // namespace lsxcommon::alice
