// deepseek_moe_v41_inference_pipeline.cu — DeepSeek-V4.1-Flash autoregressive
// driver (text backbone, milestone 1).
//
// Token-by-token decode matching inference/model.py: embed -> hc_mult copies ->
// 40 blocks (Hyper-Connections + CED attention + MoE) -> collapse -> head. The
// prompt is consumed one token per step (like the V4 pipeline), so every step
// is a seqlen==1 forward and the CED state (window ring, compressor partials,
// compressed/index caches) simply advances with start_pos.
#include "deepseek_moe_v41_load_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "deepseek_moe_v41_device_math.h"
#include "deepseek_moe_v41_inference.h"
#include "lsxcommon/expert_device_cache.h"
#include "deepseek_moe_v41_inference_csa.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace gpu {

namespace {

namespace dv41 = lsxcommon::deepseek_v41;
using V41PinnedWeights = dv41::gpu::PinnedWeights;
using V41ExpertDb = dv41::gpu::ExpertDb;
using V41ExpertEntry = dv41::gpu::ExpertEntry;

constexpr int kHidden = dv41::kHidden;
constexpr int kLayers = dv41::kLayers;
constexpr int kHeads = dv41::kHeads;
constexpr int kHeadDim = dv41::kHeadDim;
constexpr int kRopeDim = dv41::kRopeHeadDim;
constexpr int kQLora = dv41::kQLoraRank;
constexpr int kOLora = dv41::kOLoraRank;
constexpr int kOGroups = dv41::kOGroups;
constexpr int kHc = dv41::kHcMult;
constexpr int kWindow = dv41::kWindow;
constexpr int kIdxHeads = dv41::kIndexNHeads;
constexpr int kIdxDim = dv41::kIndexHeadDim;
constexpr int kIdxTopk = dv41::kIndexTopk;
constexpr int kCompressCap = 4096;

// Frees every device allocation registered through add().
struct EventSet {
    cudaEvent_t start = nullptr, ttft = nullptr, stop = nullptr;
    EventSet()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&ttft);
        cudaEventCreate(&stop);
    }
    ~EventSet()
    {
        if (start) cudaEventDestroy(start);
        if (ttft) cudaEventDestroy(ttft);
        if (stop) cudaEventDestroy(stop);
    }
};

struct DevSet {
    std::vector<void *> ptrs;
    ~DevSet()
    {
        for (void *p : ptrs)
            if (p) cudaFree(p);
    }
    void *add(size_t bytes)
    {
        void *p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        // Deterministic baseline: every scratch/persistent buffer starts at
        // zero, so a read-before-full-write (e.g. an invalid attention index
        // that must never be dereferenced) cannot leak stale memory.
        if (cudaMemset(p, 0, bytes) != cudaSuccess) {
            cudaFree(p);
            return nullptr;
        }
        ptrs.push_back(p);
        return p;
    }
    template <typename T>
    T *typed(size_t n)
    {
        return reinterpret_cast<T *>(add(n * sizeof(T)));
    }
};

// Double-buffered expert-weight prefetch. A dedicated non-blocking copy stream
// DMAs the next expert's pinned NVFP4 slabs while the current expert's FFN runs
// on the compute stream. ev_copy[b] fires when staging buffer b is filled;
// ev_free[b] fires when the compute stream is done reading buffer b.
struct CopyPipeline {
    cudaStream_t stream = nullptr;
    cudaEvent_t ev_copy[2] = {nullptr, nullptr};
    cudaEvent_t ev_free[2] = {nullptr, nullptr};
    cudaEvent_t ev_ready[8] = {};

    bool Init()
    {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess)
            return false;
        for (int b = 0; b < 2; ++b) {
            if (cudaEventCreateWithFlags(&ev_copy[b], cudaEventDisableTiming) !=
                    cudaSuccess ||
                cudaEventCreateWithFlags(&ev_free[b], cudaEventDisableTiming) !=
                    cudaSuccess)
                return false;
        }
        for (int b = 0; b < 8; ++b)
            if (cudaEventCreateWithFlags(&ev_ready[b], cudaEventDisableTiming) !=
                cudaSuccess)
                return false;
        return true;
    }
    ~CopyPipeline()
    {
        for (int b = 0; b < 2; ++b) {
            if (ev_copy[b]) cudaEventDestroy(ev_copy[b]);
            if (ev_free[b]) cudaEventDestroy(ev_free[b]);
        }
        for (int b = 0; b < 8; ++b)
            if (ev_ready[b]) cudaEventDestroy(ev_ready[b]);
        if (stream) cudaStreamDestroy(stream);
    }
};

} // namespace

gpu::InferenceResult RunDeepSeekV41Inference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, ::lsxcommon::ArrowTokenizerDb *tok_db,
    const V41PinnedWeights &pw, V41ExpertDb &expert_db,
    lsxcommon::sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos,
    const lsxcommon::GenerationStream *gs)
{
    (void)model_path;
    gpu::InferenceResult result;
    result.num_layers = static_cast<int>(pw.layers.size());
    const int num_layers = static_cast<int>(pw.layers.size());

    const int prompt_len = static_cast<int>(input_ids.size());
    int max_gen = std::min(max_gen_tokens, 4096);
    const int total_steps = prompt_len + max_gen;
    const int max_seq = std::min(kCompressCap, total_steps + 2);

    DevSet dev;

    // ── Global decode buffers ─────────────────────────────────────────────
    int32_t *d_tok = dev.typed<int32_t>(1);
    uint16_t *d_single = dev.typed<uint16_t>(kHidden);
    float *d_h = dev.typed<float>(kHc * kHidden);
    float *d_h_next = dev.typed<float>(kHc * kHidden);
    float *d_xf = dev.typed<float>(kHidden);
    uint16_t *d_xh = dev.typed<uint16_t>(kHidden);
    uint16_t *d_attn_h = dev.typed<uint16_t>(kHidden);
    float *d_attn_f = dev.typed<float>(kHidden);
    uint16_t *d_shared_h = dev.typed<uint16_t>(kHidden);
    float *d_acc = dev.typed<float>(kHidden);
    float *d_pre_mix = dev.typed<float>(kHc);
    float *d_hc_mix = dev.typed<float>(dv41::kHcMix);
    float *d_hc_rsqrt = dev.typed<float>(1);
    float *d_apre = dev.typed<float>(kHc);
    float *d_apost = dev.typed<float>(kHc);
    float *d_acomb = dev.typed<float>(kHc * kHc);
    float *d_fpre = dev.typed<float>(kHc);
    float *d_fpost = dev.typed<float>(kHc);
    float *d_fcomb = dev.typed<float>(kHc * kHc);
    float *d_logits = dev.typed<float>(pw.vocab_size);
    int32_t *d_ridx = dev.typed<int32_t>(dv41::kTopK);
    float *d_rw = dev.typed<float>(dv41::kTopK);
    int32_t *d_next = dev.typed<int32_t>(1);
    float *d_maxv = dev.typed<float>(1);

    // Expert staging (largest projection), double-buffered so the next
    // expert's slabs can be prefetched while the current one computes. The
    // pinned kStream layout packs an expert as [w1|w3|w2|s1|s3|s2] with
    // 256-byte-aligned spans, so a whole expert is one contiguous transfer.
    const int inter = dv41::kMoeIntermediate;
    auto aln256 = [](int64_t v) { return (v + 255) / 256 * 256; };
    const int64_t packed_bytes = static_cast<int64_t>(inter) * kHidden / 2;
    const int64_t scales_bytes = static_cast<int64_t>(inter) * kHidden / 16;
    const size_t expert_block = static_cast<size_t>(
        3 * aln256(packed_bytes) + 3 * aln256(scales_bytes));
    uint8_t *d_stage[2] = {};
    for (int b = 0; b < 2; ++b) d_stage[b] = dev.typed<uint8_t>(expert_block);
    // Expert-FFN activation scratch: per-expert cudaMalloc/cudaFree inside
    // LaunchExpertFfn serializes the device and would defeat the overlap.
    const size_t ffn_scratch_bytes =
        dv41::ExpertFfnScratchBytes(inter, kHidden);
    uint8_t *d_ffn_scratch = dev.typed<uint8_t>(ffn_scratch_bytes);
    // Shared-expert and router scratch, allocated once: allocating per layer
    // (cudaMalloc/cudaFree) serializes the device and kills copy/compute
    // overlap (204 alloc+free pairs per token otherwise).
    const size_t shared_scratch_bytes =
        dv41::SharedExpertScratchBytes(dv41::kSharedIntermediate);
    uint8_t *d_shared_scratch = dev.typed<uint8_t>(shared_scratch_bytes);
    float *d_router_scores = dev.typed<float>(dv41::kNumRouted);

    // Device span offsets for one staged expert (mirrors the pinned layout).
    struct StagePtrs {
        uint8_t *w1 = nullptr, *w3 = nullptr, *w2 = nullptr;
        uint8_t *s1 = nullptr, *s3 = nullptr, *s2 = nullptr;
        size_t block = 0;
    };
    auto stage_ptrs = [&](uint8_t *base, const V41ExpertEntry &en) {
        StagePtrs p;
        p.w1 = base;
        p.w3 = p.w1 + aln256(en.w1_bytes);
        p.w2 = p.w3 + aln256(en.w3_bytes);
        p.s1 = p.w2 + aln256(en.w2_bytes);
        p.s3 = p.s1 + aln256(en.w1_scales_bytes);
        p.s2 = p.s3 + aln256(en.w3_scales_bytes);
        p.block = static_cast<size_t>(p.s2 + aln256(en.w2_scales_bytes) - p.w1);
        return p;
    };

    // Per-layer CED persistent state.
    uint16_t *d_window_kv = dev.typed<uint16_t>(
        static_cast<size_t>(num_layers) * kWindow * kHeadDim);
    // One compress/index cache per source layer (rounded up generously).
    std::vector<uint16_t *> compress_kv(static_cast<size_t>(num_layers), nullptr);
    std::vector<uint16_t *> index_k(static_cast<size_t>(num_layers), nullptr);
    std::vector<float *> kv_state(static_cast<size_t>(num_layers), nullptr);
    std::vector<float *> score_state(static_cast<size_t>(num_layers), nullptr);
    for (int l = 0; l < num_layers; ++l) {
        const auto meta = pw.layers[l].meta;
        if (meta.is_kv_source)
            compress_kv[l] = dev.typed<uint16_t>(
                static_cast<size_t>(kCompressCap) * kHeadDim);
        if (meta.is_kv_source && meta.compress_ratio > 1) {
            kv_state[l] = dev.typed<float>(static_cast<size_t>(
                meta.compress_ratio) * kHeadDim);
            score_state[l] = dev.typed<float>(static_cast<size_t>(
                meta.compress_ratio) * kHeadDim);
        }
        if (meta.is_kv_source)
            index_k[l] = dev.typed<uint16_t>(
                static_cast<size_t>(kCompressCap) * kIdxDim);
    }

    // Shared CED scratch (reused across layers; layers run sequentially).
    dv41::CedLayerBuffers sb;
    sb.qr = dev.typed<uint16_t>(kQLora);
    sb.q = dev.typed<uint16_t>(kHeads * kHeadDim);
    sb.kv = dev.typed<uint16_t>(kHeadDim);
    sb.latent = dev.typed<uint16_t>(kHeadDim);
    sb.o = dev.typed<uint16_t>(kHeads * kHeadDim);
    sb.olora = dev.typed<uint16_t>(kOGroups * kOLora);
    sb.full_kv = dev.typed<uint16_t>(
        static_cast<size_t>(kWindow + kCompressCap) * kHeadDim);
    sb.idx_q = dev.typed<uint16_t>(kIdxHeads * kIdxDim);
    sb.idx_knew = dev.typed<uint16_t>(kIdxDim);
    sb.idx_score = dev.typed<float>(kCompressCap);
    sb.idx_w = dev.typed<float>(kIdxHeads);
    sb.latent_f = dev.typed<float>(kHeadDim);
    sb.win_idx = dev.typed<int32_t>(kWindow);
    sb.comp_idx = dev.typed<int32_t>(kIdxTopk);
    sb.full_idx = dev.typed<int32_t>(kWindow + kIdxTopk);
    sb.candidates = dev.typed<uint8_t>(kCompressCap);

    // RoPE tables (ratio 0 uses rope_theta; ratio > 0 uses compressed theta
    // with YaRN over original_seq_len).
    const int half = kRopeDim / 2;
    std::vector<float> cos0, sin0, cos1, sin1;
    dv41::YarnFreqsRef(kRopeDim, max_seq, 0, dv41::kRopeTheta, dv41::kRopeFactor,
                       dv41::kBetaFast, dv41::kBetaSlow, &cos0, &sin0);
    dv41::YarnFreqsRef(kRopeDim, max_seq, dv41::kOriginalSeqLen,
                       dv41::kCompressRopeTheta, dv41::kRopeFactor,
                       dv41::kBetaFast, dv41::kBetaSlow, &cos1, &sin1);
    float *d_cos0 = dev.typed<float>(static_cast<size_t>(max_seq) * half);
    float *d_sin0 = dev.typed<float>(static_cast<size_t>(max_seq) * half);
    float *d_cos1 = dev.typed<float>(static_cast<size_t>(max_seq) * half);
    float *d_sin1 = dev.typed<float>(static_cast<size_t>(max_seq) * half);

    // Validate every allocation up front.
    for (void *p : dev.ptrs)
        if (!p) return result;

    if (cudaMemcpyAsync(d_cos0, cos0.data(), cos0.size() * 4,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_sin0, sin0.data(), sin0.size() * 4,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_cos1, cos1.data(), cos1.size() * 4,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_sin1, sin1.data(), sin1.size() * 4,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        result.error_msg = "rope table upload failed";
        return result;
    }
    // Reset compressor states: kv_state 0, score_state -inf.
    for (int l = 0; l < num_layers; ++l) {
        const auto meta = pw.layers[l].meta;
        if (!(meta.is_kv_source && meta.compress_ratio > 1)) continue;
        const int n = meta.compress_ratio * kHeadDim;
        if (lsxcommon::deepseek_v41::LaunchFillF32(kv_state[l], n, 0.0f,
                                                   stream) != cudaSuccess ||
            lsxcommon::deepseek_v41::LaunchFillF32(
                score_state[l], n, -INFINITY, stream) != cudaSuccess) {
            result.error_msg = "compressor state reset failed";
            return result;
        }
    }

    dv41::CsaRuntime csa;
    csa.Reset();

    int topk = dv41::kTopK;
    for (const auto &lw : pw.layers)
        if (lw.gate_rows > 0) topk = std::min(topk, lw.gate_rows);
    std::vector<int32_t> h_ridx(static_cast<size_t>(topk));
    std::vector<float> h_rw(static_cast<size_t>(topk));
    std::vector<int32_t> gen;
    gen.reserve(static_cast<size_t>(total_steps));

    // Host-only routed-expert locality probe (LSX_V41_RLOC=<slots>): replays
    // the exact router ids through the shared device-cache LRU bookkeeping to
    // report the hit rate a real device cache of that size would achieve.
    const char *rloc_env = std::getenv("LSX_V41_RLOC");
    const bool rloc = rloc_env != nullptr;
    const int total_experts =
        num_layers * expert_db.experts_per_layer;
    gpu::ExpertDeviceCache sim;
    std::vector<int32_t> sim_slot_of, sim_owner, sim_lru, sim_prev, sim_next,
        sim_free;
    if (rloc) {
        const int S = std::max(1, std::atoi(rloc_env) == 0
                                      ? 800
                                      : std::atoi(rloc_env));
        sim_slot_of.assign(static_cast<size_t>(total_experts), -1);
        sim_owner.assign(static_cast<size_t>(S), -1);
        sim_lru.assign(static_cast<size_t>(S), 0);
        sim_prev.assign(static_cast<size_t>(S), -1);
        sim_next.assign(static_cast<size_t>(S), -1);
        sim_free.resize(static_cast<size_t>(S));
        for (int i = 0; i < S; ++i) sim_free[static_cast<size_t>(i)] = i;
        sim.slots = S;
        sim.total_experts = total_experts;
        sim.h_slot_of = sim_slot_of.data();
        sim.h_slot_owner = sim_owner.data();
        sim.h_lru = sim_lru.data();
        sim.h_prev = sim_prev.data();
        sim.h_next = sim_next.data();
        sim.h_free = sim_free.data();
        sim.h_free_top = S;
        sim.h_lru_front = -1;
        sim.h_lru_back = -1;
    }

    // LayerWeights stores __half*, the kernel API takes uint16_t bit patterns.
#define U(x) reinterpret_cast<const uint16_t *>(x)
    EventSet ev;
    // Expert staging overlap is only safe when the store is pinned (locked
    // host memory): the mmap path can evict/munmap an entry whose async DMA is
    // still in flight on the copy stream. Pinned stores never evict.
    const bool overlapped = !expert_db.pinned_pools.empty();
    CopyPipeline cp;
    if (overlapped && !cp.Init()) {
        result.error_msg = "expert copy pipeline init failed";
        return result;
    }
    if (overlapped)
        for (int b = 0; b < 2; ++b) cudaEventRecord(cp.ev_free[b], stream);

    // ── Device routed-expert cache ────────────────────────────────────────
    // 64.5% of the topk expert requests repeat within a 800-slot window
    // (LSX_V41_RLOC), so a VRAM residency cache removes most of the per-token
    // 4.8 GB host->device expert traffic. Slots hold the pinned kStream block
    // layout; eviction is safe because the per-layer router readback drains
    // the compute stream before the next layer stages its experts.
    const char *cache_env = std::getenv("LSX_V41_EXPERT_CACHE");
    const bool cache_enabled = cache_env == nullptr || cache_env[0] != '0';
    void *cache_mem = nullptr;
    int cache_slots = 0;
    if (overlapped && cache_enabled) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve = static_cast<size_t>(2) << 30;
        if (free_b > reserve) {
            size_t n = (free_b - reserve) / expert_block;
            if (n > static_cast<size_t>(total_experts))
                n = static_cast<size_t>(total_experts);
            cache_slots = static_cast<int>(n);
        }
        if (cache_slots > 0 &&
            cudaMalloc(&cache_mem,
                       static_cast<size_t>(cache_slots) * expert_block) !=
                cudaSuccess) {
            cache_mem = nullptr;
            cache_slots = 0;
        }
        if (cache_mem) dev.ptrs.push_back(cache_mem);
    }
    gpu::ExpertDeviceCache cache;
    std::vector<int32_t> c_slot_of, c_owner, c_lru, c_prev, c_next, c_free;
    std::vector<float> slot_gs;
    if (cache_slots > 0) {
        c_slot_of.assign(static_cast<size_t>(total_experts), -1);
        c_owner.assign(static_cast<size_t>(cache_slots), -1);
        c_lru.assign(static_cast<size_t>(cache_slots), 0);
        c_prev.assign(static_cast<size_t>(cache_slots), -1);
        c_next.assign(static_cast<size_t>(cache_slots), -1);
        c_free.resize(static_cast<size_t>(cache_slots));
        for (int i = 0; i < cache_slots; ++i) c_free[i] = i;
        slot_gs.assign(static_cast<size_t>(cache_slots) * 3, 1.0f);
        cache.slots = cache_slots;
        cache.total_experts = total_experts;
        cache.h_slot_of = c_slot_of.data();
        cache.h_slot_owner = c_owner.data();
        cache.h_lru = c_lru.data();
        cache.h_prev = c_prev.data();
        cache.h_next = c_next.data();
        cache.h_free = c_free.data();
        cache.h_free_top = cache_slots;
        cache.h_lru_front = -1;
        cache.h_lru_back = -1;
        log::info("[v41-cache] device expert cache: " +
                  std::to_string(cache_slots) + " slots x " +
                  std::to_string(expert_block) + " B");
    }
    const bool use_cache = cache_slots > 0;
    bool first_gen_captured = false;
    cudaEventRecord(ev.start, stream);

    for (int step = 0; step < total_steps; ++step) {
        const int cur_tok = (step < prompt_len) ? input_ids[step] : gen.back();
        if (cur_tok < 0 || cur_tok >= pw.vocab_size) {
            result.error_msg =
                "token id " + std::to_string(cur_tok) + " out of vocab range (" +
                std::to_string(pw.vocab_size) + ") at step " +
                std::to_string(step);
            return result;
        }
        if (cudaMemcpyAsync(d_tok, &cur_tok, 4, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            result.error_msg = "token upload failed";
            return result;
        }
        if (dv41::LaunchEmbedLookup(U(pw.d_embed), pw.vocab_size, kHidden, d_tok, 1,
                                    d_single, stream) != cudaSuccess ||
            dv41::LaunchHcExpand(d_single, kHidden, kHc, d_h, stream) !=
                cudaSuccess) {
            result.error_msg = "embed/hc expand failed";
            return result;
        }

        const float one_hc[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        if (cudaMemcpyAsync(d_pre_mix, one_hc, kHc * 4, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            result.error_msg = "pre_mix init failed";
            return result;
        }

        for (int lay = 0; lay < num_layers; ++lay) {
            const dv41::gpu::LayerWeights &lw = pw.layers[lay];
            const auto meta = lw.meta;

            // ── Attention sub-block ──────────────────────────────────────
            if (dv41::LaunchHcMixes(d_h, lw.d_hc_attn_fn, lw.d_hc_attn_scale,
                                    lw.d_hc_attn_base, kHc, kHidden,
                                    dv41::kSinkhornIters, dv41::kHcEps, d_apre,
                                    d_apost, d_acomb, d_hc_mix, d_hc_rsqrt,
                                    stream) != cudaSuccess ||
                dv41::LaunchHcPre(d_h, d_pre_mix, d_xf, kHc, kHidden, stream) !=
                    cudaSuccess ||
                dv41::LaunchCastF32ToHalf(d_xf, d_xh, kHidden, stream) !=
                    cudaSuccess ||
                dv41::LaunchRmsNormHalf(d_xh, U(lw.d_attn_norm), d_xh, kHidden,
                                        dv41::kRmsEps, stream) != cudaSuccess) {
                result.error_msg = "attention prelude failed";
                return result;
            }
            cudaError_t ae;
            if (meta.compress_ratio == 0) {
                ae = dv41::AttentionForwardDecode(
                    stream, d_xh, kHidden, lw.d_wq_a, lw.d_wq_a_scale, kQLora,
                    U(lw.d_q_norm), lw.d_wq_b, lw.d_wq_b_scale, kHeads, kHeadDim,
                    kRopeDim, lw.d_wkv, lw.d_wkv_scale, U(lw.d_kv_norm),
                    lw.d_attn_sink, lw.d_wo_a, lw.d_wo_a_scale, kOGroups, kOLora,
                    lw.d_wo_b, lw.d_wo_b_scale, d_cos0, d_sin0,
                    d_window_kv + static_cast<size_t>(lay) * kWindow * kHeadDim,
                    kWindow, step, d_attn_h);
            } else {
                dv41::CedLayerParams p;
                p.compress_ratio = meta.compress_ratio;
                p.is_kv_source = meta.is_kv_source;
                p.is_index_source = meta.is_index_source;
                p.is_candidate_source = meta.is_candidate_source;
                p.owns_k = meta.is_kv_source;
                p.uses_candidates = dv41::UsesCandidates(lay);
                p.wq_a = lw.d_wq_a;
                p.wq_a_scale = lw.d_wq_a_scale;
                p.q_lora = kQLora;
                p.q_norm = U(lw.d_q_norm);
                p.wq_b = lw.d_wq_b;
                p.wq_b_scale = lw.d_wq_b_scale;
                p.heads = kHeads;
                p.head_dim = kHeadDim;
                p.rope_dim = kRopeDim;
                p.wkv = lw.d_wkv;
                p.wkv_scale = lw.d_wkv_scale;
                p.kv_norm = U(lw.d_kv_norm);
                p.attn_sink = lw.d_attn_sink;
                p.wo_a = lw.d_wo_a;
                p.wo_a_scale = lw.d_wo_a_scale;
                p.o_groups = kOGroups;
                p.o_lora_rank = kOLora;
                p.wo_b = lw.d_wo_b;
                p.wo_b_scale = lw.d_wo_b_scale;
                p.comp_wkv = U(lw.d_comp_wkv);
                p.comp_wgate = U(lw.d_comp_wgate);
                p.comp_norm = U(lw.d_comp_norm);
                p.idx_wq_b = lw.d_idx_wq_b;
                p.idx_wq_b_scale = lw.d_idx_wq_b_scale;
                p.idx_wk = U(lw.d_idx_wk);
                p.idx_k_norm = U(lw.d_idx_k_norm);
                p.idx_wproj = U(lw.d_idx_weights_proj);
                p.idx_n_heads = kIdxHeads;
                p.idx_head_dim = kIdxDim;
                p.idx_topk = kIdxTopk;
                p.candidate_topk_blocks = dv41::kCandidateTopkBlocks;
                p.candidate_block_size = dv41::kCandidateBlockSize;
                sb.window_kv =
                    d_window_kv + static_cast<size_t>(lay) * kWindow * kHeadDim;
                sb.compress_kv = compress_kv[lay];
                sb.index_k = index_k[lay];
                sb.kv_state = kv_state[lay];
                sb.score_state = score_state[lay];
                sb.compress_cap = kCompressCap;
                ae = dv41::CedAttentionForwardDecode(
                    stream, d_xh, kHidden, kWindow, lay, p, sb, d_cos1,
                    d_sin1, step, csa, d_attn_h);
            }
            if (ae != cudaSuccess) {
                result.error_msg = "attention forward failed";
                return result;
            }
            if (dv41::LaunchCastHalfToF32(d_attn_h, d_attn_f, kHidden,
                                          stream) != cudaSuccess) {
                result.error_msg = "attention output cast failed";
                return result;
            }
            if (dv41::LaunchHcPost(d_attn_f, d_h, d_apost, d_acomb, d_h_next,
                                   kHc, kHidden, stream) != cudaSuccess) {
                result.error_msg = "attention hc_post failed";
                return result;
            }
            std::swap(d_h, d_h_next);

            // ── FFN sub-block ────────────────────────────────────────────
            if (dv41::LaunchHcMixes(d_h, lw.d_hc_ffn_fn, lw.d_hc_ffn_scale,
                                    lw.d_hc_ffn_base, kHc, kHidden,
                                    dv41::kSinkhornIters, dv41::kHcEps, d_fpre,
                                    d_fpost, d_fcomb, d_hc_mix, d_hc_rsqrt,
                                    stream) != cudaSuccess ||
                dv41::LaunchHcPre(d_h, d_apre, d_xf, kHc, kHidden, stream) !=
                    cudaSuccess ||
                dv41::LaunchCastF32ToHalf(d_xf, d_xh, kHidden, stream) !=
                    cudaSuccess ||
                dv41::LaunchRmsNormHalf(d_xh, U(lw.d_ffn_norm), d_xh, kHidden,
                                        dv41::kRmsEps, stream) != cudaSuccess) {
                result.error_msg = "ffn prelude failed";
                return result;
            }

            // Router + routed experts + shared expert.
            if (dv41::LaunchRouterTopKScratch(
                    stream, d_xh, U(lw.d_gate_weight), lw.d_gate_bias,
                    lw.gate_rows, kHidden, topk, dv41::kGateTemp,
                    dv41::kRouteScale, true, d_ridx, d_rw,
                    d_router_scores) != cudaSuccess) {
                result.error_msg = "router failed";
                return result;
            }
            if (cudaMemcpyAsync(h_ridx.data(), d_ridx, topk * 4,
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaMemcpyAsync(h_rw.data(), d_rw, topk * 4,
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess) {
                result.error_msg = "router readback failed";
                return result;
            }
            if (rloc) {
                for (int k = 0; k < topk; ++k) {
                    const int flat =
                        lay * expert_db.experts_per_layer +
                        h_ridx[static_cast<size_t>(k)];
                    if (sim.Lookup(flat) < 0) sim.Insert(flat);
                }
            }
            if (cudaMemsetAsync(d_acc, 0, kHidden * 4, stream) != cudaSuccess) {
                result.error_msg = "accum reset failed";
                return result;
            }
            // Stage one expert block into double-buffer `buf` on the copy
            // stream. The pinned kStream layout is one contiguous
            // [w1|w3|w2|s1|s3|s2] span, so the locked path is a single ~20 MB
            // transfer. en_stg[buf]/sp_stg[buf] feed the compute step.
            V41ExpertEntry en_stg[2] = {};
            StagePtrs sp_stg[2];
            auto stage_expert = [&](int kk, int buf) -> bool {
                const int e = h_ridx[static_cast<size_t>(kk)];
                dv41::gpu::ensure_expert_resident(expert_db, lay, e, stream);
                const size_t idx =
                    static_cast<size_t>(lay) *
                        static_cast<size_t>(expert_db.experts_per_layer) +
                    static_cast<size_t>(e);
                if (idx >= expert_db.entries.size()) return false;
                const V41ExpertEntry &en = expert_db.entries[idx];
                if (!en.host_w1 || !en.host_w2 || !en.host_w3) return false;
                StagePtrs sp = stage_ptrs(d_stage[buf], en);
                if (sp.block > expert_block) return false;
                if (cudaStreamWaitEvent(cp.stream, cp.ev_free[buf], 0) !=
                    cudaSuccess)
                    return false;
                if (cudaMemcpyAsync(d_stage[buf], en.host_w1, sp.block,
                                    cudaMemcpyHostToDevice,
                                    cp.stream) != cudaSuccess)
                    return false;
                if (cudaEventRecord(cp.ev_copy[buf], cp.stream) != cudaSuccess)
                    return false;
                en_stg[buf] = en;
                sp_stg[buf] = sp;
                return true;
            };
            if (use_cache) {
                // Phase 1: resolve each topk expert to a device cache slot; on
                // a miss copy its pinned block on the copy stream. Phase 2:
                // wait for the miss copies, then run each FFN off its slot.
                int cslot[8] = {};
                bool cact[8] = {}, cwait[8] = {};
                for (int k = 0; k < topk; ++k) {
                    const int e = h_ridx[static_cast<size_t>(k)];
                    dv41::gpu::ensure_expert_resident(expert_db, lay, e, stream);
                    const size_t idx =
                        static_cast<size_t>(lay) *
                            static_cast<size_t>(expert_db.experts_per_layer) +
                        static_cast<size_t>(e);
                    if (idx >= expert_db.entries.size()) continue;
                    const V41ExpertEntry &en = expert_db.entries[idx];
                    if (!en.host_w1 || !en.host_w2 || !en.host_w3) continue;
                    const int flat = lay * expert_db.experts_per_layer + e;
                    int slot = cache.Lookup(flat);
                    if (slot < 0) {
                        int evicted = -1;
                        slot = cache.Insert(flat, &evicted);
                        StagePtrs sp = stage_ptrs(
                            static_cast<uint8_t *>(cache_mem) +
                                static_cast<size_t>(slot) * expert_block,
                            en);
                        if (sp.block > expert_block) continue;
                        if (cudaMemcpyAsync(
                                static_cast<uint8_t *>(cache_mem) +
                                    static_cast<size_t>(slot) * expert_block,
                                en.host_w1, sp.block, cudaMemcpyHostToDevice,
                                cp.stream) != cudaSuccess) {
                            result.error_msg = "expert cache fill failed";
                            return result;
                        }
                        slot_gs[static_cast<size_t>(slot) * 3 + 0] =
                            en.w1_global_scale;
                        slot_gs[static_cast<size_t>(slot) * 3 + 1] =
                            en.w2_global_scale;
                        slot_gs[static_cast<size_t>(slot) * 3 + 2] =
                            en.w3_global_scale;
                        if (cudaEventRecord(cp.ev_ready[k], cp.stream) !=
                            cudaSuccess) {
                            result.error_msg = "expert cache event failed";
                            return result;
                        }
                        cwait[k] = true;
                    }
                    cslot[k] = slot;
                    cact[k] = true;
                }
                for (int k = 0; k < topk; ++k) {
                    if (!cact[k]) continue;
                    const int slot = cslot[k];
                    const V41ExpertEntry &en =
                        expert_db.entries[static_cast<size_t>(lay) *
                                              static_cast<size_t>(
                                                  expert_db
                                                      .experts_per_layer) +
                                          static_cast<size_t>(
                                              h_ridx[static_cast<size_t>(k)])];
                    StagePtrs sp = stage_ptrs(
                        static_cast<uint8_t *>(cache_mem) +
                            static_cast<size_t>(slot) * expert_block,
                        en);
                    if (cwait[k] &&
                        cudaStreamWaitEvent(stream, cp.ev_ready[k], 0) !=
                            cudaSuccess) {
                        result.error_msg = "expert cache wait failed";
                        return result;
                    }
                    dv41::DeviceExpertWeight ew1{sp.w1, sp.s1,
                                                 slot_gs[slot * 3 + 0]};
                    dv41::DeviceExpertWeight ew3{sp.w3, sp.s3,
                                                 slot_gs[slot * 3 + 2]};
                    dv41::DeviceExpertWeight ew2{sp.w2, sp.s2,
                                                 slot_gs[slot * 3 + 1]};
                    if (dv41::LaunchExpertFfnScratch(
                            stream, d_xh, kHidden, ew1, ew3, ew2, inter,
                            dv41::kSwigluLimit,
                            h_rw[static_cast<size_t>(k)], d_acc, d_ffn_scratch,
                            ffn_scratch_bytes) != cudaSuccess) {
                        result.error_msg = "routed expert failed";
                        return result;
                    }
                }
            } else if (overlapped) {
                bool act[2] = {false, false};
                act[0] = stage_expert(0, 0);
                for (int k = 0; k < topk; ++k) {
                    const int buf = k & 1;
                    if (k + 1 < topk)
                        act[(k + 1) & 1] = stage_expert(k + 1, (k + 1) & 1);
                    if (!act[buf]) continue;
                    if (cudaStreamWaitEvent(stream, cp.ev_copy[buf], 0) !=
                        cudaSuccess) {
                        result.error_msg = "expert event wait failed";
                        return result;
                    }
                    const StagePtrs &sp = sp_stg[buf];
                    dv41::DeviceExpertWeight ew1{sp.w1, sp.s1,
                                                 en_stg[buf].w1_global_scale};
                    dv41::DeviceExpertWeight ew3{sp.w3, sp.s3,
                                                 en_stg[buf].w3_global_scale};
                    dv41::DeviceExpertWeight ew2{sp.w2, sp.s2,
                                                 en_stg[buf].w2_global_scale};
                    if (dv41::LaunchExpertFfnScratch(
                            stream, d_xh, kHidden, ew1, ew3, ew2, inter,
                            dv41::kSwigluLimit,
                            h_rw[static_cast<size_t>(k)], d_acc, d_ffn_scratch,
                            ffn_scratch_bytes) != cudaSuccess) {
                        result.error_msg = "routed expert failed";
                        return result;
                    }
                    if (cudaEventRecord(cp.ev_free[buf], stream) !=
                        cudaSuccess) {
                        result.error_msg = "expert event record failed";
                        return result;
                    }
                }
            } else {
                uint8_t *base = d_stage[0];
                for (int k = 0; k < topk; ++k) {
                    const int e = h_ridx[static_cast<size_t>(k)];
                    dv41::gpu::ensure_expert_resident(expert_db, lay, e, stream);
                    const size_t idx =
                        static_cast<size_t>(lay) *
                            static_cast<size_t>(expert_db.experts_per_layer) +
                        static_cast<size_t>(e);
                    if (idx >= expert_db.entries.size()) continue;
                    const V41ExpertEntry &en = expert_db.entries[idx];
                    if (!en.host_w1 || !en.host_w2 || !en.host_w3) continue;
                    StagePtrs sp = stage_ptrs(base, en);
                    if (sp.block > expert_block) continue;
                    auto cpy = [&](void *dst, const void *src, size_t n) {
                        return n == 0 ? cudaSuccess
                                      : cudaMemcpyAsync(dst, src, n,
                                                        cudaMemcpyHostToDevice,
                                                        stream);
                    };
                    if (cpy(sp.w1, en.host_w1, en.w1_bytes) != cudaSuccess ||
                        cpy(sp.w3, en.host_w3, en.w3_bytes) != cudaSuccess ||
                        cpy(sp.w2, en.host_w2, en.w2_bytes) != cudaSuccess ||
                        cpy(sp.s1, en.host_w1_scales, en.w1_scales_bytes) !=
                            cudaSuccess ||
                        cpy(sp.s3, en.host_w3_scales, en.w3_scales_bytes) !=
                            cudaSuccess ||
                        cpy(sp.s2, en.host_w2_scales, en.w2_scales_bytes) !=
                            cudaSuccess) {
                        result.error_msg = "expert staging failed";
                        return result;
                    }
                    dv41::DeviceExpertWeight ew1{sp.w1, sp.s1,
                                                 en.w1_global_scale};
                    dv41::DeviceExpertWeight ew3{sp.w3, sp.s3,
                                                 en.w3_global_scale};
                    dv41::DeviceExpertWeight ew2{sp.w2, sp.s2,
                                                 en.w2_global_scale};
                    if (dv41::LaunchExpertFfnScratch(
                            stream, d_xh, kHidden, ew1, ew3, ew2, inter,
                            dv41::kSwigluLimit,
                            h_rw[static_cast<size_t>(k)], d_acc, d_ffn_scratch,
                            ffn_scratch_bytes) != cudaSuccess) {
                        result.error_msg = "routed expert failed";
                        return result;
                    }
                }
            }
            if (dv41::SharedExpertForwardScratch(
                    stream, d_xh, kHidden, U(lw.d_shared_w1),
                    U(lw.d_shared_w3), U(lw.d_shared_w2), dv41::kSharedIntermediate,
                    dv41::kSwigluLimit, d_shared_h, d_shared_scratch,
                    shared_scratch_bytes) != cudaSuccess ||
                dv41::LaunchAccumulateHalfToF32(d_shared_h, d_acc, kHidden,
                                                stream) != cudaSuccess) {
                result.error_msg = "shared expert failed";
                return result;
            }
            // d_acc holds the FP32 MoE output (routed + shared); hc_post takes
            // it directly in the F32 hc stream.
            if (dv41::LaunchHcPost(d_acc, d_h, d_fpost, d_fcomb, d_h_next, kHc,
                                   kHidden, stream) != cudaSuccess) {
                result.error_msg = "ffn hc_post failed";
                return result;
            }
            std::swap(d_h, d_h_next);
            if (cudaMemcpyAsync(d_pre_mix, d_fpre, kHc * 4,
                                cudaMemcpyDeviceToDevice, stream) !=
                cudaSuccess) {
                result.error_msg = "pre_mix carry failed";
                return result;
            }
        }

        // ── Final collapse + head ─────────────────────────────────────────
        if (dv41::LaunchHcPre(d_h, d_pre_mix, d_xf, kHc, kHidden, stream) !=
                cudaSuccess ||
            dv41::LaunchCastF32ToHalf(d_xf, d_xh, kHidden, stream) !=
                cudaSuccess ||
            dv41::LaunchRmsNormHalf(d_xh, U(pw.d_final_norm), d_xh, kHidden,
                                    dv41::kRmsEps, stream) != cudaSuccess ||
            dv41::LaunchHeadLogits(stream, U(pw.d_head), d_xh, d_logits,
                                   pw.vocab_size, kHidden) != cudaSuccess) {
            result.error_msg = "final head failed";
            return result;
        }
        if (sampler.Sample(stream, d_logits, pw.vocab_size,
                           lsxcommon::sample::LogitType::kF32, d_next,
                           d_maxv) != cudaSuccess) {
            result.error_msg = "sampling failed";
            return result;
        }
        int32_t next_tok = 0;
        float maxv = 0.0f;
        if (cudaMemcpyAsync(&next_tok, d_next, 4, cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaMemcpyAsync(&maxv, d_maxv, 4, cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            result.error_msg = "sampler readback failed";
            return result;
        }
        if (std::getenv("LSX_V41_DEBUG")) {
            log::info("[v41-debug] step=" + std::to_string(step) +
                      " tok=" + std::to_string(next_tok) +
                      " maxv=" + std::to_string(maxv));
        }
        gen.push_back(next_tok);
        if (gs) {
            const int idx = static_cast<int>(gen.size()) - 1;
            const int first_gen = prompt_len - 1;
            if (idx >= first_gen)
                gs->Emit(static_cast<std::uint32_t>(next_tok),
                         static_cast<std::uint32_t>(idx - first_gen));
            if (gs->Aborted()) break;
        }
        if (!first_gen_captured && step == prompt_len - 1) {
            result.first_gen_tok_id = next_tok;
            result.first_gen_maxv = maxv;
            result.first_gen_text =
                tok_db ? lsxcommon::runtime::DecodeGeneratedText(*tok_db, {next_tok}) : std::string();
            cudaEventRecord(ev.ttft, stream);
            first_gen_captured = true;
        }
        if (!ignore_eos && tok_db && next_tok == tok_db->eos_id &&
            step >= prompt_len) {
            result.eos_terminated = true;
            break;
        }
    }
    cudaEventRecord(ev.stop, stream);

    {
        result.final_logits.resize(static_cast<size_t>(pw.vocab_size));
        if (cudaMemcpyAsync(result.final_logits.data(), d_logits,
                            static_cast<size_t>(pw.vocab_size) * 4,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            result.final_logits.clear();
    }

    const int first_gen_idx = std::max(prompt_len - 1, 0);
    std::vector<int32_t> gen_tokens;
    for (size_t i = static_cast<size_t>(first_gen_idx); i < gen.size(); ++i)
        gen_tokens.push_back(gen[i]);
    result.gen_tokens = gen_tokens;
    result.num_tokens = static_cast<int>(gen_tokens.size());
    result.output_text = tok_db ? lsxcommon::runtime::DecodeGeneratedText(*tok_db,
                                                                gen_tokens)
                               : std::string();
    result.weight_resident_bytes = pw.resident_bytes;
    float total_ms = 0.0f, ttft_ms = 0.0f;
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&total_ms, ev.start, ev.stop);
    cudaEventElapsedTime(&ttft_ms, ev.start, ev.ttft);
    result.total_time_ms = static_cast<double>(total_ms);
    result.ttft_ms = static_cast<double>(ttft_ms);
    result.tokens_per_sec = (total_ms > 0.0f)
        ? (static_cast<double>(result.num_tokens) / (total_ms / 1000.0))
        : 0.0;
    result.ok = true;
    if (rloc) {
        const double total = static_cast<double>(sim.hits + sim.misses);
        log::info("[v41-rloc] slots=" + std::to_string(sim.slots) +
                  " hits=" + std::to_string(sim.hits) +
                  " misses=" + std::to_string(sim.misses) + " hit_rate=" +
                  std::to_string(total > 0 ? sim.hits / total : 0.0));
    }
    if (use_cache) {
        const double total = static_cast<double>(cache.hits + cache.misses);
        log::info("[v41-cache] hits=" + std::to_string(cache.hits) +
                  " misses=" + std::to_string(cache.misses) + " hit_rate=" +
                  std::to_string(total > 0 ? cache.hits / total : 0.0));
    }
    return result;
}

#undef U

} // namespace gpu
} // namespace lsxcommon
