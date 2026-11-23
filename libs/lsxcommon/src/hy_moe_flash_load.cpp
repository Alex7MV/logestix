// hy_moe_flash_load.cpp — Hy4-flash topology gates and device upload (Stage 3).
//
// Every read is a zero-copy Arrow view over an mmap-backed IPC file (R2): the
// pinned tensors are uploaded straight from the page cache and the routed
// experts stay in the neutral ExpertStore's LRU, so a 90 GB artifact never lands
// in host RAM in one piece.
#include "hy_moe_flash_load.h"

#include "fast_load_internal.h"

#include <absl/strings/str_format.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <simdjson.h>

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

#include <cctype>

namespace lsxcommon {
namespace hy4 {
namespace gpu {

namespace fs = std::filesystem;

namespace {

// ── Upload helpers ─────────────────────────────────────────────────────────

// BF16 source -> FP16 device. BF16's 8-bit mantissa fits inside FP16's 11-bit
// mantissa and every in-range BF16 value is exactly representable, so this is a
// lossless widening; the shared in-place kernel does the rounding.
cudaError_t UploadBf16AsHalf(const uint8_t *src, int64_t elems, __half **dst,
                             cudaStream_t stream, std::uint64_t &bytes)
{
    if (elems <= 0) return cudaSuccess;
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, static_cast<size_t>(elems) * 2);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p, src, static_cast<size_t>(elems) * 2,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaFree(p);
        return e;
    }
    e = fastload::Bf16ToHalfInPlace(static_cast<uint16_t *>(p), elems, stream);
    if (e != cudaSuccess) {
        cudaFree(p);
        return e;
    }
    *dst = static_cast<__half *>(p);
    bytes += static_cast<std::uint64_t>(elems) * 2;
    return cudaSuccess;
}

// F32 source -> F32 device (copy).
cudaError_t UploadF32(const uint8_t *src, int64_t elems, float **dst,
                      cudaStream_t stream, std::uint64_t &bytes)
{
    if (elems <= 0) return cudaSuccess;
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, static_cast<size_t>(elems) * 4);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p, src, static_cast<size_t>(elems) * 4,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaFree(p);
        return e;
    }
    *dst = static_cast<float *>(p);
    bytes += static_cast<std::uint64_t>(elems) * 4;
    return cudaSuccess;
}

const TensorView *Find(const TensorBatchViews &views, std::string_view name)
{
    for (const auto &v : views.tensors)
        if (v.name == name) return &v;
    return nullptr;
}

const DenseNvfp4View *FindNvfp4(const DenseNvfp4BatchViews &views,
                                std::string_view name)
{
    for (const auto &v : views.records)
        if (v.name == name) return &v;
    return nullptr;
}

int64_t Numel(const TensorView &v)
{
    int64_t n = 1;
    for (int64_t i = 0; i < v.rank; ++i) n *= v.shape[i];
    return n;
}

// Uploads a named raw tensor as FP16, requiring the exact element count.
cudaError_t UploadHalfNamed(const TensorBatchViews &views,
                            std::string_view name,
                            int64_t elems, __half **dst, cudaStream_t stream,
                            std::uint64_t &bytes, std::string &error)
{
    const TensorView *v = Find(views, name);
    if (!v) {
        error = "missing tensor: " + std::string(name);
        return cudaErrorInvalidValue;
    }
    if (Numel(*v) != elems) {
        error = "element count mismatch: " + std::string(name);
        return cudaErrorInvalidValue;
    }
    if (v->dtype != "BF16") {
        error = "expected BF16 source for " + std::string(name);
        return cudaErrorInvalidValue;
    }
    return UploadBf16AsHalf(v->data, elems, dst, stream, bytes);
}

cudaError_t UploadF32Named(const TensorBatchViews &views,
                           std::string_view name,
                           int64_t elems, float **dst, cudaStream_t stream,
                           std::uint64_t &bytes, std::string &error)
{
    const TensorView *v = Find(views, name);
    if (!v) {
        error = "missing tensor: " + std::string(name);
        return cudaErrorInvalidValue;
    }
    if (Numel(*v) != elems) {
        error = "element count mismatch: " + std::string(name);
        return cudaErrorInvalidValue;
    }
    if (v->dtype != "F32") {
        error = "expected F32 source for " + std::string(name);
        return cudaErrorInvalidValue;
    }
    return UploadF32(v->data, elems, dst, stream, bytes);
}

cudaError_t UploadNvfp4Named(const DenseNvfp4BatchViews &views,
                             std::string_view name, int rows, int cols,
                             lsx_nvfp4::LsxNvfp4Weight &w, cudaStream_t stream,
                             std::uint64_t &bytes, std::string &error)
{
    const DenseNvfp4View *r = FindNvfp4(views, name);
    if (!r) {
        error = "missing NVFP4 record: " + std::string(name);
        return cudaErrorInvalidValue;
    }
    if (r->rank != 2 || r->shape[0] != rows || r->shape[1] != cols) {
        error = "NVFP4 shape mismatch: " + std::string(name);
        return cudaErrorInvalidValue;
    }
    void *p1 = nullptr;
    void *p2 = nullptr;
    cudaError_t e = cudaMalloc(&p1, static_cast<size_t>(r->packed_bytes));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(&p2, static_cast<size_t>(r->scale_bytes));
    if (e != cudaSuccess) {
        cudaFree(p1);
        return e;
    }
    e = cudaMemcpyAsync(p1, r->packed, static_cast<size_t>(r->packed_bytes),
                        cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(p2, r->scales, static_cast<size_t>(r->scale_bytes),
                            cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaFree(p1);
        cudaFree(p2);
        return e;
    }
    w.d_packed = static_cast<const uint8_t *>(p1);
    w.d_scales = static_cast<const uint8_t *>(p2);
    w.global_scale = r->global_scale;
    w.rows = rows;
    w.cols = cols;
    w.packed_bytes = r->packed_bytes;
    w.scale_bytes = r->scale_bytes;
    bytes += static_cast<std::uint64_t>(r->packed_bytes + r->scale_bytes);
    return cudaSuccess;
}

// simdjson readers for the index document.
template <typename Obj>
int GetInt(Obj &o, const char *k, int dflt)
{
    int64_t v = 0;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
}

} // namespace

// ── Topology ───────────────────────────────────────────────────────────────

Topology LoadTopology(const std::string &model_path)
{
    Topology top;
    const fs::path idx_path =
        fs::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(idx_path, std::ios::binary);
    if (!f) {
        top.error_msg = "model.index.json not found: " + idx_path.string();
        return top;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        top.error_msg = "model.index.json parse error";
        return top;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        top.error_msg = "model.index.json root is not an object";
        return top;
    }
    std::string_view s;
    if (root["format"].get_string().get(s) == simdjson::SUCCESS)
        top.format = std::string(s);
    if (root["model"].get_string().get(s) == simdjson::SUCCESS)
        top.model = std::string(s);
    top.num_layers = GetInt(root, "num_layers", 0);
    top.experts_per_layer = GetInt(root, "experts_per_layer", 0);
    top.hidden_size = GetInt(root, "hidden_size", 0);
    top.vocab_size = GetInt(root, "vocab_size", 0);
    top.num_experts_per_tok = GetInt(root, "num_experts_per_tok", 0);
    top.moe_intermediate_size = GetInt(root, "moe_intermediate_size", 0);
    top.latent_size = GetInt(root, "latent_size", 0);
    top.shared_intermediate_size = GetInt(root, "shared_intermediate_size", 0);
    top.dense_intermediate_size = GetInt(root, "dense_intermediate_size", 0);
    top.num_attention_heads = GetInt(root, "num_attention_heads", 0);
    top.swa_num_attention_heads = GetInt(root, "swa_num_attention_heads", 0);
    top.q_lora_rank = GetInt(root, "q_lora_rank", 0);
    top.kv_lora_rank = GetInt(root, "kv_lora_rank", 0);
    top.qk_nope_head_dim = GetInt(root, "qk_nope_head_dim", 0);
    top.qk_rope_head_dim = GetInt(root, "qk_rope_head_dim", 0);
    top.v_head_dim = GetInt(root, "v_head_dim", 0);
    top.sliding_window = GetInt(root, "sliding_window", 0);
    top.hc_mult = GetInt(root, "hc_mult", 0);
    top.index_n_heads = GetInt(root, "index_n_heads", 0);
    top.index_head_dim = GetInt(root, "index_head_dim", 0);
    top.index_topk = GetInt(root, "index_topk", 0);
    top.bos_token_id = GetInt(root, "bos_token_id", -1);
    top.eos_token_id = GetInt(root, "eos_token_id", -1);
    top.pad_token_id = GetInt(root, "pad_token_id", -1);
    int64_t v64 = 0;
    if (root["total_experts"].get_int64().get(v64) == simdjson::SUCCESS)
        top.total_experts = v64;
    if (root["total_pinned_tensors"].get_int64().get(v64) == simdjson::SUCCESS)
        top.total_pinned_tensors = v64;

    simdjson::ondemand::object deferred;
    if (root["deferred"].get_object().get(deferred) == simdjson::SUCCESS) {
        top.deferred_engram = GetInt(deferred, "engram", 0);
        top.deferred_vision = GetInt(deferred, "vision", 0);
        top.deferred_dfly = GetInt(deferred, "dfly", 0);
    }

    simdjson::ondemand::array layers;
    if (root["layers"].get_array().get(layers) == simdjson::SUCCESS) {
        for (auto e : layers) {
            simdjson::ondemand::object o;
            if (e.get_object().get(o) != simdjson::SUCCESS) continue;
            LayerShape ls;
            bool b = false;
            if (o["is_dsa"].get_bool().get(b) == simdjson::SUCCESS)
                ls.is_dsa = b;
            if (o["is_moe"].get_bool().get(b) == simdjson::SUCCESS)
                ls.is_moe = b;
            if (o["has_engram"].get_bool().get(b) == simdjson::SUCCESS)
                ls.has_engram = b;
            ls.heads = GetInt(o, "heads", 0);
            top.layers.push_back(ls);
        }
    }

    std::string vram_err;
    if (!ParseVramBlock(json, top.vram, vram_err)) {
        top.error_msg = "index vram block: " + vram_err;
        return top;
    }
    if (top.num_layers <= 0 || top.experts_per_layer < 0 || top.hidden_size <= 0)
        top.error_msg = "model.index.json is missing its geometry";
    return top;
}

std::string NormalizeCardName(std::string_view card)
{
    std::string out;
    out.reserve(card.size());
    for (char c : card) {
        if (c == ' ')
            out.push_back('-');
        else
            out.push_back(static_cast<char>(std::tolower(
                static_cast<unsigned char>(c))));
    }
    return out;
}

bool MatchesModelType(const Topology &topo, std::string_view model_type)
{
    if (topo.model.empty()) return true;
    return NormalizeCardName(topo.model) == model_type;
}

bool ValidateGeometry(const Topology &topo, std::string *error)
{
    auto bad = [error](const char *field, long long got, long long want) {
        if (error)
            *error = absl::StrFormat("%s: index says %lld, frozen passport %lld",
                                     field, got, want);
        return false;
    };
    if (topo.num_layers != kLayers)
        return bad("num_layers", topo.num_layers, kLayers);
    if (topo.hidden_size != kHidden)
        return bad("hidden_size", topo.hidden_size, kHidden);
    if (topo.vocab_size != kVocab)
        return bad("vocab_size", topo.vocab_size, kVocab);
    if (topo.experts_per_layer != kNumRouted)
        return bad("experts_per_layer", topo.experts_per_layer, kNumRouted);
    if (topo.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", topo.num_experts_per_tok, kTopK);
    if (topo.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", topo.moe_intermediate_size,
                   kMoeIntermediate);
    if (topo.latent_size != kLatent)
        return bad("latent_size", topo.latent_size, kLatent);
    if (topo.shared_intermediate_size != kSharedIntermediate)
        return bad("shared_intermediate_size", topo.shared_intermediate_size,
                   kSharedIntermediate);
    if (topo.dense_intermediate_size != kDenseIntermediate)
        return bad("dense_intermediate_size", topo.dense_intermediate_size,
                   kDenseIntermediate);
    if (topo.num_attention_heads != kHeadsDsa)
        return bad("num_attention_heads", topo.num_attention_heads, kHeadsDsa);
    if (topo.swa_num_attention_heads != kHeadsSwa)
        return bad("swa_num_attention_heads", topo.swa_num_attention_heads,
                   kHeadsSwa);
    if (topo.q_lora_rank != kQLora)
        return bad("q_lora_rank", topo.q_lora_rank, kQLora);
    if (topo.kv_lora_rank != kKvLora)
        return bad("kv_lora_rank", topo.kv_lora_rank, kKvLora);
    if (topo.qk_nope_head_dim != kQkNope)
        return bad("qk_nope_head_dim", topo.qk_nope_head_dim, kQkNope);
    if (topo.qk_rope_head_dim != kQkRope)
        return bad("qk_rope_head_dim", topo.qk_rope_head_dim, kQkRope);
    if (topo.v_head_dim != kVHead)
        return bad("v_head_dim", topo.v_head_dim, kVHead);
    if (topo.sliding_window != kSwaWindow)
        return bad("sliding_window", topo.sliding_window, kSwaWindow);
    if (topo.hc_mult != kHcMult)
        return bad("hc_mult", topo.hc_mult, kHcMult);
    if (topo.index_n_heads != kIndexNHeads)
        return bad("index_n_heads", topo.index_n_heads, kIndexNHeads);
    if (topo.index_head_dim != kIndexHeadDim)
        return bad("index_head_dim", topo.index_head_dim, kIndexHeadDim);
    if (topo.index_topk != kIndexTopk)
        return bad("index_topk", topo.index_topk, kIndexTopk);
    if (static_cast<int>(topo.layers.size()) != kLayers) {
        if (error) *error = "index layer table is short";
        return false;
    }
    for (int l = 0; l < kLayers; ++l) {
        const LayerShape &ls = topo.layers[static_cast<size_t>(l)];
        if (ls.is_dsa != IsDsa(l)) {
            if (error)
                *error = absl::StrFormat("layer %d: index is_dsa=%d, passport "
                                         "IsDsa=%d", l, ls.is_dsa ? 1 : 0,
                                         IsDsa(l) ? 1 : 0);
            return false;
        }
        if (ls.is_moe != IsMoeLayer(l)) {
            if (error)
                *error = absl::StrFormat("layer %d: index is_moe=%d, passport "
                                         "IsMoeLayer=%d", l, ls.is_moe ? 1 : 0,
                                         IsMoeLayer(l) ? 1 : 0);
            return false;
        }
        if (ls.heads != Heads(l)) {
            if (error)
                *error = absl::StrFormat("layer %d: index heads=%d, passport "
                                         "%d", l, ls.heads, Heads(l));
            return false;
        }
    }
    return true;
}

KvCacheGeometry KvGeometry(int64_t capacity)
{
    // One MLA latent per position: the latent (kv_lora) plus the shared rope
    // key, one "head" wide. The 64-way expansion into per-head K/V happens
    // inside attention, never in the cache.
    return KvCacheGeometry::Make(KvCacheLayout::kPosMajor, capacity, 1,
                                 kKvLora + kQkRope, kLayers);
}

// ── Upload ─────────────────────────────────────────────────────────────────

namespace {

void FreeLayer(LayerWeights &lw)
{
    lw.attn.q_a.Free();
    lw.attn.q_b.Free();
    lw.attn.kv_a.Free();
    lw.attn.kv_b.Free();
    lw.attn.kv_b_value.Free();
    lw.attn.o.Free();
    lw.attn.gate.Free();
    lw.attn.indexer.q.Free();
    lw.attn.indexer.tok.Free();
    lw.attn.indexer.hw.Free();
    cudaFree(lw.attn.d_q_norm);
    cudaFree(lw.attn.d_kv_norm);
    cudaFree(lw.attn.d_post_proj_norm);
    cudaFree(lw.attn.d_sink);
    cudaFree(lw.attn.indexer.d_key_value);
    cudaFree(lw.attn.indexer.d_key_gate);
    cudaFree(lw.attn.indexer.d_token_key_norm);
    cudaFree(lw.attn.indexer.d_key_norm);
    cudaFree(lw.attn.indexer.d_ape);
    cudaFree(lw.hc_attn.d_fn);
    cudaFree(lw.hc_attn.d_base);
    cudaFree(lw.hc_attn.d_scale);
    cudaFree(lw.hc_mlp.d_fn);
    cudaFree(lw.hc_mlp.d_base);
    cudaFree(lw.hc_mlp.d_scale);
    cudaFree(lw.d_input_norm);
    cudaFree(lw.d_post_attn_norm);
    lw.d_gate.Free();
    lw.d_up.Free();
    lw.d_down.Free();
    cudaFree(lw.d_post_fc2_norm);
    cudaFree(lw.d_router);
    cudaFree(lw.d_router_bias);
    lw.s_gate.Free();
    lw.s_up.Free();
    lw.s_down.Free();
    lw.r_down.Free();
    lw.r_up.Free();
    cudaFree(lw.d_combine_norm);
    lw = LayerWeights{};
}

} // namespace

void FreeHy4PinnedWeights(PinnedWeights &pw)
{
    cudaFree(pw.d_embed);
    cudaFree(pw.d_lm_head);
    cudaFree(pw.d_final_norm);
    cudaFree(pw.head.d_fn);
    cudaFree(pw.head.d_base);
    cudaFree(pw.head.d_scale);
    pw.d_embed = nullptr;
    pw.d_lm_head = nullptr;
    pw.d_final_norm = nullptr;
    pw.head = HeadWeights{};
    for (auto &lw : pw.layers) FreeLayer(lw);
    pw.layers.clear();
    pw.resident_bytes = 0;
}

void PinnedWeights::Free() { FreeHy4PinnedWeights(*this); }

bool LoadHy4Weights(const std::string &model_path, const Topology &topo,
                    PinnedWeights &pw, cudaStream_t stream, std::string &error)
{
    error.clear();
    FreeHy4PinnedWeights(pw);
    pw.num_layers = topo.num_layers;
    pw.experts_per_layer = topo.experts_per_layer;
    pw.vocab_size = topo.vocab_size;

    const fs::path log_root = fs::path(model_path) / "logestix";
    cudaError_t e = cudaSuccess;
    std::uint64_t bytes = 0;

    // Globals.
    {
        auto views = LoadTensorBatchViews(log_root / "pinned" / "global.arrow");
        if (!views.ok()) {
            error = "global.arrow: " + views.status().ToString();
            return false;
        }
        e = UploadHalfNamed(*views, "model.embed_tokens.weight",
                            static_cast<int64_t>(kVocab) * kHidden, &pw.d_embed,
                            stream, bytes, error);
        if (e == cudaSuccess)
            e = UploadHalfNamed(*views, "lm_head.weight",
                                static_cast<int64_t>(kVocab) * kHidden,
                                &pw.d_lm_head, stream, bytes, error);
        if (e == cudaSuccess)
            e = UploadHalfNamed(*views, "model.norm.weight", kHidden,
                                &pw.d_final_norm, stream, bytes, error);
        if (e == cudaSuccess)
            e = UploadF32Named(*views, "model.hc_head.hc_head_fn", kHcMult * kHcDim,
                               &pw.head.d_fn, stream, bytes, error);
        if (e == cudaSuccess)
            e = UploadF32Named(*views, "model.hc_head.hc_head_base", kHcMult,
                               &pw.head.d_base, stream, bytes, error);
        if (e == cudaSuccess)
            e = UploadF32Named(*views, "model.hc_head.hc_head_scale", 1,
                               &pw.head.d_scale, stream, bytes, error);
        if (e != cudaSuccess) return false;
    }

    pw.layers.resize(static_cast<size_t>(topo.num_layers));
    for (int l = 0; l < topo.num_layers; ++l) {
        LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        lw.shape = topo.layers[static_cast<size_t>(l)];
        const int heads = lw.shape.heads;
        const std::string lname = absl::StrFormat("layer_%03d", l);

        auto raw = LoadTensorBatchViews(log_root / "pinned" / (lname + ".arrow"));
        auto dense =
            LoadDenseNvfp4BatchViews(log_root / "pinned" / (lname + "_nvfp4.arrow"));
        if (!raw.ok() || !dense.ok()) {
            error = absl::StrFormat("layer %d: pinned Arrow missing (%s / %s)", l,
                                    raw.ok() ? "ok" : raw.status().ToString(),
                                    dense.ok() ? "ok" : dense.status().ToString());
            return false;
        }

        // The artifact keeps the SOURCE tensor names, so a lookup is
        // "model.layers.<l>." + leaf; `up`/`upf`/`upw` take the leaf only.
        const std::string prefix = absl::StrFormat("model.layers.%d.", l);
        auto leaf_name = [&prefix](std::string_view leaf) {
            return prefix + std::string(leaf);
        };
        auto up = [&](std::string_view leaf, int64_t elems, __half **dst) {
            return UploadHalfNamed(*raw, leaf_name(leaf), elems, dst, stream,
                                   bytes, error);
        };
        auto upf = [&](std::string_view leaf, int64_t elems, float **dst) {
            return UploadF32Named(*raw, leaf_name(leaf), elems, dst, stream,
                                  bytes, error);
        };
        auto upw = [&](std::string_view leaf, int rows, int cols,
                       lsx_nvfp4::LsxNvfp4Weight &w) {
            return UploadNvfp4Named(*dense, leaf_name(leaf), rows, cols, w,
                                    stream, bytes, error);
        };

        e = up("input_layernorm.weight", kHidden,
               &lw.d_input_norm);
        if (e == cudaSuccess)
            e = up("post_attention_layernorm.weight", kHidden,
                   &lw.d_post_attn_norm);
        if (e == cudaSuccess)
            e = upf("hc_attn_layer.hc_pre.hc_fn", kHcRows * kHcDim,
                    &lw.hc_attn.d_fn);
        if (e == cudaSuccess)
            e = upf("hc_attn_layer.hc_pre.hc_base", kHcRows,
                    &lw.hc_attn.d_base);
        if (e == cudaSuccess)
            e = upf("hc_attn_layer.hc_pre.hc_scale", 2,
                    &lw.hc_attn.d_scale);
        if (e == cudaSuccess)
            e = upf("hc_mlp_layer.hc_pre.hc_fn", kHcRows * kHcDim,
                    &lw.hc_mlp.d_fn);
        if (e == cudaSuccess)
            e = upf("hc_mlp_layer.hc_pre.hc_base", kHcRows,
                    &lw.hc_mlp.d_base);
        if (e == cudaSuccess)
            e = upf("hc_mlp_layer.hc_pre.hc_scale", 2,
                    &lw.hc_mlp.d_scale);

        // Attention.
        if (e == cudaSuccess)
            e = upw("self_attn.q_a_proj.weight", kQLora, kHidden,
                    lw.attn.q_a);
        if (e == cudaSuccess)
            e = upw("self_attn.q_b_proj.weight", heads * kQkHead,
                    kQLora, lw.attn.q_b);
        if (e == cudaSuccess)
            e = upw("self_attn.kv_a_proj_with_mqa.weight",
                    kKvLora + kQkRope, kHidden, lw.attn.kv_a);
        if (e == cudaSuccess)
            e = upw("self_attn.kv_b_proj.weight", heads * kQkNope, kKvLora,
                    lw.attn.kv_b);
        if (e == cudaSuccess)
            e = upw("self_attn.kv_b_proj.weight.value", heads * kVHead, kKvLora,
                    lw.attn.kv_b_value);
        if (e == cudaSuccess)
            e = upw("self_attn.o_proj.weight", kHidden,
                    heads * kVHead, lw.attn.o);
        if (e == cudaSuccess)
            e = upw("self_attn.linear_gate.weight",
                    heads * kVHead, kHidden, lw.attn.gate);
        if (e == cudaSuccess)
            e = up("self_attn.q_a_layernorm.weight", kQLora,
                   &lw.attn.d_q_norm);
        if (e == cudaSuccess)
            e = up("self_attn.kv_a_layernorm.weight", kKvLora,
                   &lw.attn.d_kv_norm);
        if (e == cudaSuccess)
            e = up("self_attn.post_proj_norm.weight", kHidden,
                   &lw.attn.d_post_proj_norm);
        if (e == cudaSuccess)
            e = upf("self_attn.learnable_sink_param", heads,
                    &lw.attn.d_sink);
        lw.attn.heads = heads;

        if (e == cudaSuccess && lw.shape.is_dsa) {
            IndexerWeights &iw = lw.attn.indexer;
            e = upw("self_attn.compressed_indexer.query_projection.weight",
                    kIndexNHeads * kIndexHeadDim, kQLora, iw.q);
            if (e == cudaSuccess)
                e = upw("self_attn.compressed_indexer.token_key_projection.weight",
                        kIndexHeadDim, kHidden, iw.tok);
            if (e == cudaSuccess)
                e = upw("self_attn.compressed_indexer.head_weight_projection.weight",
                        kIndexNHeads, kHidden, iw.hw);
            if (e == cudaSuccess)
                e = upf("self_attn.compressed_indexer.key_value_projection.weight",
                        kIndexHeadDim * kIndexHeadDim, &iw.d_key_value);
            if (e == cudaSuccess)
                e = upf("self_attn.compressed_indexer.key_gate_projection.weight",
                        kIndexHeadDim * kIndexHeadDim, &iw.d_key_gate);
            if (e == cudaSuccess)
                e = up("self_attn.compressed_indexer.token_key_norm.weight",
                       kIndexHeadDim, &iw.d_token_key_norm);
            if (e == cudaSuccess)
                e = upf("self_attn.compressed_indexer.key_norm.weight",
                        kIndexHeadDim, &iw.d_key_norm);
            if (e == cudaSuccess)
                e = upf("self_attn.compressed_indexer.intra_group_ape",
                        kPoolRatio * kIndexHeadDim, &iw.d_ape);
            iw.loaded = (e == cudaSuccess);
        }

        // FFN: dense (layer 0) or LatentMoE.
        if (e == cudaSuccess && !lw.shape.is_moe) {
            e = upw("mlp.gate_proj.weight", kDenseIntermediate,
                    kHidden, lw.d_gate);
            if (e == cudaSuccess)
                e = upw("mlp.up_proj.weight", kDenseIntermediate,
                        kHidden, lw.d_up);
            if (e == cudaSuccess)
                e = upw("mlp.down_proj.weight", kHidden,
                        kDenseIntermediate, lw.d_down);
            if (e == cudaSuccess)
                e = up("mlp.post_fc2_norm.weight", kHidden,
                       &lw.d_post_fc2_norm);
        }
        if (e == cudaSuccess && lw.shape.is_moe) {
            e = up("mlp.gate.weight",
                   static_cast<int64_t>(kNumRouted) * kHidden, &lw.d_router);
            lw.router_bytes = static_cast<int64_t>(kNumRouted) * kHidden * 2;
            if (e == cudaSuccess)
                e = upf("mlp.gate.e_score_correction_bias",
                        kNumRouted, &lw.d_router_bias);
            if (e == cudaSuccess)
                e = upw("mlp.shared_experts.gate_proj.weight",
                        kSharedIntermediate, kHidden, lw.s_gate);
            if (e == cudaSuccess)
                e = upw("mlp.shared_experts.up_proj.weight",
                        kSharedIntermediate, kHidden, lw.s_up);
            if (e == cudaSuccess)
                e = upw("mlp.shared_experts.down_proj.weight",
                        kHidden, kSharedIntermediate, lw.s_down);
            if (e == cudaSuccess)
                e = upw("mlp.routed_expert_down_proj.weight",
                        kLatent, kHidden, lw.r_down);
            if (e == cudaSuccess)
                e = upw("mlp.routed_expert_up_proj.weight",
                        kHidden, kLatent, lw.r_up);
            if (e == cudaSuccess)
                e = up("mlp.combine_output_norm.weight", kHidden,
                       &lw.d_combine_norm);
        }
        if (e != cudaSuccess) {
            if (error.empty())
                error = absl::StrFormat("layer %d upload failed: %s", l,
                                        cudaGetErrorString(e));
            return false;
        }
        lw.loaded = true;
        // The mapping views die here: every byte is on the device.
    }

    pw.resident_bytes = bytes;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

// ── Expert store ───────────────────────────────────────────────────────────

ExpertDb LoadExpertIndex(const std::string &model_path, int num_layers,
                         int experts_per_layer, int max_open)
{
    ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.first_moe_layer = 1;  // layer 0 is the dense FFN; flat math stays full
    db.max_open = static_cast<size_t>(max_open > 0 ? max_open : 4096);
    // Warm lazily: 20 992 experts at ~2.4 ms each is 50 s of startup tax for a
    // window that the first tokens fill on their own.
    db.warm_on_load = false;
    db.mmap_fail_verb = "hy4 expert mmap fail ";
    if (!db.Load(model_path, /*fallback=*/false))
        log::error("[hy4-load] expert store enumeration failed");
    return db;
}

void EnsureExpertResident(ExpertDb &db, int layer, int expert,
                          cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void FreeExpertIndex(ExpertDb &db) { db.Free(); }

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
