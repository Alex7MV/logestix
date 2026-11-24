// hy_moe_flash_inference_pipeline.cu — Hy4-flash autoregressive driver.
//
// One token per step (prefill and decode share the path): embed -> 4 iHC
// streams -> 42 blocks {attention + FFN, each wrapped in iHC} -> head collapse
// -> final norm -> fp32 head. Per block:
//
//   attn:  iHC mix -> RMSNorm -> MLA q_a/norm/q_b + kv_a/norm, RoPE (two
//          regimes) -> latent cache write -> visible set (sliding window, or
//          IndexPool beyond index_topk) -> transposed qk projection -> sink
//          softmax -> value projection -> sigmoid gate -> o_proj ->
//          post_proj_norm -> iHC post
//   ffn:   iHC mix -> RMSNorm -> dense SwiGLU (layer 0) or {router -> 16 routed
//          experts in the latent space -> routed_expert_up_proj ->
//          combine_output_norm + shared expert} -> iHC post
//
// The routed experts are streamed from the bounded host store through a device
// slot cache: 41 x 512 experts at NVFP4 is 83 GB, so the decode reads the ~656
// experts a token routes to and nothing else.
#include "lsxcommon/hy_moe_flash_inference_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "hy_moe_flash_constants.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/hy_moe_flash_kernels.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include <absl/strings/str_format.h>

#include "lsxcommon/log.h"

namespace lsxcommon {
namespace hy4 {
namespace gpu {

namespace {

constexpr int kMaxVisible = kIndexTopk + kPoolRatio;

size_t Align256(size_t v) { return (v + 255) / 256 * 256; }

// One staged expert's device layout: [w1|w3|w2|s1|s3|s2], 256-byte aligned.
struct SlotLayout {
    size_t w1 = 0, w3 = 0, w2 = 0, s1 = 0, s3 = 0, s2 = 0, block = 0;
};

SlotLayout ExpertSlotLayout()
{
    const int I = kMoeIntermediate;
    const int L = kLatent;
    const size_t packed = static_cast<size_t>(I) * L / 2;
    const size_t scales = static_cast<size_t>(I) * L / 16;
    SlotLayout s;
    s.w1 = 0;
    s.w3 = s.w1 + Align256(packed);
    s.w2 = s.w3 + Align256(packed);
    s.s1 = s.w2 + Align256(packed);
    s.s3 = s.s1 + Align256(scales);
    s.s2 = s.s3 + Align256(scales);
    s.block = s.s2 + Align256(scales);
    return s;
}

// RoPE tables for one theta: [max_seq][rope/2] cos / sin.
void BuildRopeTables(int max_seq, int rope_dim, float theta,
                     std::vector<float> &cos_t, std::vector<float> &sin_t)
{
    const int half = rope_dim / 2;
    cos_t.resize(static_cast<size_t>(max_seq) * half);
    sin_t.resize(static_cast<size_t>(max_seq) * half);
    for (int p = 0; p < max_seq; ++p) {
        for (int j = 0; j < half; ++j) {
            const double inv = std::pow(static_cast<double>(theta),
                                        -2.0 * static_cast<double>(j) /
                                            static_cast<double>(rope_dim));
            const double ang = static_cast<double>(p) * inv;
            cos_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::cos(ang));
            sin_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::sin(ang));
        }
    }
}

} // namespace

// ── Session ────────────────────────────────────────────────────────────────

std::int64_t Session::Bytes() const
{
    std::int64_t b = geom.Elements() * 2;
    for (const auto *p : pooled) b += p ? static_cast<std::int64_t>(capacity / kPoolRatio)
                                              * kIndexHeadDim * 2
                                        : 0;
    b += static_cast<std::int64_t>(group_keys.size()) * kPoolRatio *
         kIndexHeadDim * 2;
    return b;
}

void Session::Free()
{
    cudaFree(d_latent);
    d_latent = nullptr;
    for (auto *p : pooled) cudaFree(p);
    pooled.clear();
    for (auto *p : group_keys) cudaFree(p);
    group_keys.clear();
    group_filled.clear();
    kv_rows = 0;
    capacity = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t Session::EnsureCapacity(int64_t want, cudaStream_t stream)
{
    (void)stream;
    if (want <= capacity && d_latent) return cudaSuccess;
    Free();
    const int64_t cap = std::max<int64_t>(want, 2048);
    geom = KvGeometry(cap);
    const size_t bytes = static_cast<size_t>(geom.Elements()) * 2;
    cudaError_t e = cudaMalloc(&d_latent, bytes);
    if (e != cudaSuccess) return e;
    e = cudaMemsetAsync(d_latent, 0, bytes, stream);
    if (e != cudaSuccess) return e;
    capacity = cap;
    group_filled.assign(static_cast<size_t>(kLayers), 0);
    for (int l = 0; l < kLayers; ++l) {
        if (!IsDsa(l)) {
            pooled.push_back(nullptr);
            group_keys.push_back(nullptr);
            continue;
        }
        __half *p = nullptr;
        if (cudaMalloc(&p, static_cast<size_t>(cap / kPoolRatio) * kIndexHeadDim * 2) !=
            cudaSuccess)
            return cudaGetLastError();
        pooled.push_back(p);
        __half *gk = nullptr;
        if (cudaMalloc(&gk, static_cast<size_t>(kPoolRatio) * kIndexHeadDim * 2) !=
            cudaSuccess)
            return cudaGetLastError();
        group_keys.push_back(gk);
    }
    return cudaSuccess;
}

void Session::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
    if (!group_filled.empty())
        std::fill(group_filled.begin(), group_filled.end(), 0);
}

void Scratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    // The expert device-slot cache is a SEPARATE allocation (see EnsureScratch:
    // it is sized from the VRAM left after the arena) and it is by far the
    // biggest thing this family holds — ~25 GB of a 32 GiB card. Nulling the
    // pointer without freeing leaked the whole card on every teardown, so a
    // process that loads a second model (a test binary with two cases, a server
    // swapping models) then failed with "scratch allocation failed" and 965 MiB
    // free. Free it before the field is cleared.
    if (d_slots) cudaFree(d_slots);
    d_slots = nullptr;
    streams = agg = pre = post = mix = rsqrt = nullptr;
    x = q = qr = kv = latent = nullptr;
    qk = scores = acc = nullptr;
    o = gate = attn_out = ffn_out = nullptr;
    moe_acc = nullptr;
    lat = gu = act = s_gu = s_act = nullptr;
    d_gu = d_act = nullptr;
    pos = nullptr;
    idx_q = nullptr;
    idx_key = nullptr;
    idx_scores = nullptr;
    idx_used = nullptr;
    idx_wh = nullptr;
    router_scores = nullptr;
    router_idx = nullptr;
    router_w = nullptr;
    logits = nullptr;
    d_tok = d_argmax = nullptr;
    d_maxv = nullptr;
    d_slots = nullptr;
    slot_bytes = 0;
    slot_count = 0;
}

namespace {

// Bump allocator inside the scratch arena.
struct Arena {
    uint8_t *base = nullptr;
    size_t used = 0;
    template <typename T>
    T *Take(size_t n)
    {
        used = Align256(used);
        T *p = reinterpret_cast<T *>(base + used);
        used += Align256(n * sizeof(T));
        return p;
    }
};

// Bytes of the per-call arena. The expert device slots are a SEPARATE
// allocation (they are sized from the VRAM left after this arena), so counting
// them here would double-book the card and fail the allocation on a 32 GiB GPU.
size_t ScratchBytes(const PinnedWeights &pw)
{
    const int heads = kHeadsSwa;
    size_t b = 0;
    auto add = [&b](size_t n) { b += Align256(n); };
    add(static_cast<size_t>(kHcMult) * kHidden * 4);
    add(static_cast<size_t>(kHidden) * 4);
    add(static_cast<size_t>(2 * kHcMult) * 4);
    add(4);
    add(static_cast<size_t>(2 * kHcMult) * 4);
    add(4);
    add(static_cast<size_t>(kHidden) * 2);
    add(static_cast<size_t>(heads) * kQkHead * 2);
    add(static_cast<size_t>(kQLora) * 2);
    add(static_cast<size_t>(kKvLora + kQkRope) * 2);
    add(static_cast<size_t>(kKvLora + kQkRope) * 2);
    add(static_cast<size_t>(heads) * kKvLora * 4);
    add(static_cast<size_t>(heads) * kMaxVisible * 4);
    add(static_cast<size_t>(heads) * kKvLora * 4);
    add(static_cast<size_t>(heads) * kVHead * 2);
    add(static_cast<size_t>(heads) * kVHead * 2);
    add(static_cast<size_t>(kHidden) * 2);
    add(static_cast<size_t>(kHidden) * 2);
    add(static_cast<size_t>(kLatent) * 4);
    add(static_cast<size_t>(kLatent) * 2);
    add(static_cast<size_t>(2 * kMoeIntermediate) * 2);
    add(static_cast<size_t>(kMoeIntermediate) * 2);
    add(static_cast<size_t>(2 * kSharedIntermediate) * 2);
    add(static_cast<size_t>(kSharedIntermediate) * 2);
    add(static_cast<size_t>(2 * kDenseIntermediate) * 2);
    add(static_cast<size_t>(kDenseIntermediate) * 2);
    add(static_cast<size_t>(kMaxVisible) * 4);
    add(static_cast<size_t>(kIndexNHeads) * kIndexHeadDim * 4);
    add(static_cast<size_t>(kIndexHeadDim) * 2);
    add(static_cast<size_t>(kMaxVisible / kPoolRatio + 8) * 4);
    add(static_cast<size_t>(kMaxVisible / kPoolRatio + 8));
    add(static_cast<size_t>(kIndexNHeads) * 4);
    add(static_cast<size_t>(kNumRouted) * 4);
    add(static_cast<size_t>(kTopK) * 4);
    add(static_cast<size_t>(kTopK) * 4);
    add(static_cast<size_t>(pw.vocab_size) * 4);   // logits
    add(4);   // d_tok
    add(4);   // d_argmax
    add(4);   // d_maxv
    return b;
}

} // namespace

namespace {

// Allocates the scratch arena plus the expert device slots.
cudaError_t EnsureScratch(const PinnedWeights &pw, Scratch &sc, int want_slots,
                          int *free_mib)
{
    const SlotLayout sl = ExpertSlotLayout();
    if (free_mib) *free_mib = 0;
    const size_t want = ScratchBytes(pw);
    if (sc.arena && sc.arena_bytes >= want && sc.slot_count > 0)
        return cudaSuccess;
    sc.Free();
    void *base = nullptr;
    if (cudaMalloc(&base, want) != cudaSuccess) return cudaGetLastError();
    Arena a;
    a.base = static_cast<uint8_t *>(base);
    const int heads = kHeadsSwa;
    sc.streams = a.Take<float>(static_cast<size_t>(kHcMult) * kHidden);
    sc.agg = a.Take<float>(kHidden);
    sc.pre = a.Take<float>(2 * kHcMult);
    sc.post = a.Take<float>(2 * kHcMult);
    sc.mix = a.Take<float>(2 * kHcMult);
    sc.rsqrt = a.Take<float>(1);
    sc.x = a.Take<__half>(kHidden);
    sc.q = a.Take<__half>(static_cast<size_t>(heads) * kQkHead);
    sc.qr = a.Take<__half>(kQLora);
    sc.kv = a.Take<__half>(kKvLora + kQkRope);
    sc.latent = a.Take<__half>(kKvLora + kQkRope);
    sc.qk = a.Take<float>(static_cast<size_t>(heads) * kKvLora);
    sc.scores = a.Take<float>(static_cast<size_t>(heads) * kMaxVisible);
    sc.acc = a.Take<float>(static_cast<size_t>(heads) * kKvLora);
    sc.o = a.Take<__half>(static_cast<size_t>(heads) * kVHead);
    sc.gate = a.Take<__half>(static_cast<size_t>(heads) * kVHead);
    sc.attn_out = a.Take<__half>(kHidden);
    sc.ffn_out = a.Take<__half>(kHidden);
    sc.moe_acc = a.Take<float>(kLatent);
    sc.lat = a.Take<__half>(kLatent);
    sc.gu = a.Take<__half>(2 * kMoeIntermediate);
    sc.act = a.Take<__half>(kMoeIntermediate);
    sc.s_gu = a.Take<__half>(2 * kSharedIntermediate);
    sc.s_act = a.Take<__half>(kSharedIntermediate);
    sc.d_gu = a.Take<__half>(2 * kDenseIntermediate);
    sc.d_act = a.Take<__half>(kDenseIntermediate);
    sc.pos = a.Take<int32_t>(kMaxVisible);
    sc.idx_q = a.Take<float>(static_cast<size_t>(kIndexNHeads) * kIndexHeadDim);
    sc.idx_key = a.Take<__half>(kIndexHeadDim);
    sc.idx_scores = a.Take<float>(kMaxVisible / kPoolRatio + 8);
    sc.idx_used = a.Take<unsigned char>(kMaxVisible / kPoolRatio + 8);
    sc.idx_wh = a.Take<float>(kIndexNHeads);
    sc.router_scores = a.Take<float>(kNumRouted);
    sc.router_idx = a.Take<int32_t>(kTopK);
    sc.router_w = a.Take<float>(kTopK);
    sc.logits = a.Take<float>(pw.vocab_size);
    sc.d_tok = a.Take<int32_t>(1);
    sc.d_argmax = a.Take<int32_t>(1);
    sc.d_maxv = a.Take<float>(1);
    // Invariant, checked AFTER the last slot: the bump allocator must stay inside
    // the arena. A short ScratchBytes() hands out device pointers past the
    // allocation, and the driver does not validate them on a copy — it faults
    // inside libcuda, one stage later, far from the cause.
    if (a.used > want) {
        lsxcommon::log::error(absl::StrFormat(
            "[hy4] scratch arena overflow: ScratchBytes=%zu, bump allocator "
            "used %zu", want, a.used));
        sc.Free();
        return cudaErrorInvalidValue;
    }
    sc.arena = base;
    sc.arena_bytes = want;

    // Expert slots come out of what is left, keeping a fixed reserve for the KV
    // session, the rope tables and driver overhead. Fewer slots is a
    // throughput number, not a correctness one, so a small cache is fine.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        sc.Free();
        return cudaGetLastError();
    }
    const size_t reserve = 1024ull << 20;
    int slots = static_cast<int>(free_b > reserve
                                     ? std::min<size_t>((free_b - reserve) / sl.block,
                                                        static_cast<size_t>(want_slots))
                                     : 0);
    if (slots < 8) {
        // Say WHY. "scratch allocation failed" sent a whole debugging session
        // looking for a leak when the answer was "the previous model's arena was
        // still resident" or "the expert slot cache took every free byte".
        lsxcommon::log::error(absl::StrFormat(
            "[hy4] expert slot cache does not fit: free=%zu MiB, reserve=%zu MiB, "
            "one slot=%zu KiB -> %d slots (need >= 8)",
            free_b >> 20, reserve >> 20, sl.block >> 10, slots));
        sc.Free();
        return cudaErrorMemoryAllocation;
    }
    void *sl_buf = nullptr;
    if (cudaMalloc(&sl_buf, static_cast<size_t>(slots) * sl.block) != cudaSuccess) {
        lsxcommon::log::error(absl::StrFormat(
            "[hy4] expert slot cache alloc failed: %d slots x %zu KiB = %zu MiB "
            "against %zu MiB free",
            slots, sl.block >> 10, (static_cast<size_t>(slots) * sl.block) >> 20,
            free_b >> 20));
        sc.Free();
        return cudaGetLastError();
    }
    sc.d_slots = static_cast<uint8_t *>(sl_buf);
    sc.slot_bytes = static_cast<size_t>(slots) * sl.block;
    sc.slot_count = slots;
    if (free_mib) *free_mib = static_cast<int>(free_b >> 20);
    return cudaSuccess;
}

} // namespace

// ── Debug instrument ──────────────────────────────────────────────────────
//
// LSX_HY4_DUMP_DIR=<dir> writes every layer boundary as raw little-endian fp32
// (last position only, because the decode loop is one token per step). It is
// how a divergence is localised to a component instead of guessed at: the
// reference (tools/hy4_flash_reference.py --dump) writes the same names from
// its own hooks. Off unless the env var is set, so a normal run pays one
// getenv. See docs/open-tasks/live/models/hy4-flash.md.
namespace {

struct Dumper {
    const char *dir = nullptr;
    bool active = false;

    void Init() { dir = std::getenv("LSX_HY4_DUMP_DIR"); }
    // Only the step that consumed the LAST prompt token is dumped: that is the
    // teacher-forced prefix the reference's single forward computes, so the two
    // sides are the same computation rather than the same shape.
    void AtPromptEnd(bool is_prompt_end) { active = is_prompt_end; }
    bool On() const { return dir != nullptr && *dir != '\0' && active; }

    void F32(const char *name, const float *d_src, int64_t n,
             cudaStream_t stream) const
    {
        if (!On()) return;
        std::vector<float> host(static_cast<size_t>(n));
        if (cudaMemcpyAsync(host.data(), d_src, static_cast<size_t>(n) * 4,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return;
        Write(name, host.data(), n);
    }

    // Half-precision buffers (the activations between kernels) are widened on
    // the host so every dumped boundary is the same fp32 format.
    void H(const char *name, const __half *d_src, int64_t n,
           cudaStream_t stream) const
    {
        if (!On()) return;
        std::vector<__half> raw(static_cast<size_t>(n));
        if (cudaMemcpyAsync(raw.data(), d_src, static_cast<size_t>(n) * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return;
        std::vector<float> host(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            host[static_cast<size_t>(i)] =
                __half2float(raw[static_cast<size_t>(i)]);
        Write(name, host.data(), n);
    }

    void Write(const char *name, const float *host, int64_t n) const
    {
        const std::string path = absl::StrFormat("%s/%s.f32", dir, name);
        if (std::FILE *f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(host, 4, static_cast<size_t>(n), f);
            std::fclose(f);
        }
    }
};

} // namespace

// ── Decode ─────────────────────────────────────────────────────────────────

::lsxcommon::gpu::InferenceResult RunHy4Inference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    const PinnedWeights &pw, ExpertDb &expert_db, Session &session,
    Scratch &scratch, sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos, const GenerationStream *gs)
{
    (void)model_path;
    ::lsxcommon::gpu::InferenceResult result;
    result.num_layers = static_cast<int>(pw.layers.size());
    const int num_layers = static_cast<int>(pw.layers.size());
    if (num_layers <= 0 || pw.vocab_size <= 0) {
        result.error_msg = "model has no layers";
        return result;
    }

    const int prompt_len = static_cast<int>(input_ids.size());
    const int max_gen = std::min(max_gen_tokens, 4096);
    const int total_steps = prompt_len - 1 + max_gen;
    if (total_steps <= 0) {
        result.error_msg = "empty request";
        return result;
    }

    // Expert device slots: EnsureScratch sizes them from the VRAM left after
    // the arena; the request here is only the cap.
    if (EnsureScratch(pw, scratch, /*want_slots=*/8192, nullptr) != cudaSuccess) {
        result.error_msg = "scratch allocation failed";
        return result;
    }
    const int slots = scratch.slot_count;
    if (session.EnsureCapacity(total_steps + 2, stream) != cudaSuccess) {
        result.error_msg = "kv session allocation failed";
        return result;
    }

    Dumper dumper;
    dumper.Init();

    // RoPE tables for both regimes (the token positions are absolute).
    const int table_seq = total_steps + 2;
    std::vector<float> cos_dsa, sin_dsa, cos_swa, sin_swa;
    BuildRopeTables(table_seq, kQkRope, kRopeThetaDsa, cos_dsa, sin_dsa);
    BuildRopeTables(table_seq, kQkRope, kRopeThetaSwa, cos_swa, sin_swa);
    float *d_cos_dsa = nullptr, *d_sin_dsa = nullptr;
    float *d_cos_swa = nullptr, *d_sin_swa = nullptr;
    const size_t rowf = static_cast<size_t>(kQkRope / 2) * sizeof(float);
    auto upload = [&](const std::vector<float> &v, float **dst) {
        if (cudaMalloc(dst, v.size() * sizeof(float)) != cudaSuccess) return false;
        return cudaMemcpyAsync(*dst, v.data(), v.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    if (!upload(cos_dsa, &d_cos_dsa) || !upload(sin_dsa, &d_sin_dsa) ||
        !upload(cos_swa, &d_cos_swa) || !upload(sin_swa, &d_sin_swa)) {
        result.error_msg = "rope table upload failed";
        return result;
    }
    auto release_rope = [&] {
        cudaFree(d_cos_dsa);
        cudaFree(d_sin_dsa);
        cudaFree(d_cos_swa);
        cudaFree(d_sin_swa);
    };

    // Host-side expert device cache bookkeeping.
    const int total_experts = num_layers * expert_db.experts_per_layer;
    const SlotLayout slot_layout = ExpertSlotLayout();
    ::lsxcommon::gpu::ExpertDeviceCache cache;
    std::vector<int32_t> c_slot_of(static_cast<size_t>(total_experts), -1);
    std::vector<int32_t> c_owner(static_cast<size_t>(slots), -1);
    std::vector<int32_t> c_lru(static_cast<size_t>(slots), -1);
    std::vector<int32_t> c_prev(static_cast<size_t>(slots), -1);
    std::vector<int32_t> c_next(static_cast<size_t>(slots), -1);
    std::vector<int32_t> c_free(static_cast<size_t>(slots), -1);
    cache.slots = slots;
    cache.total_experts = total_experts;
    cache.h_slot_of = c_slot_of.data();
    cache.h_slot_owner = c_owner.data();
    cache.h_lru = c_lru.data();
    cache.h_prev = c_prev.data();
    cache.h_next = c_next.data();
    cache.h_free = c_free.data();
    cache.h_free_top = slots;
    for (int s = 0; s < slots; ++s) c_free[static_cast<size_t>(s)] = s;
    cache.h_lru_front = -1;
    cache.h_lru_back = -1;

    std::vector<int32_t> gen;
    gen.reserve(static_cast<size_t>(total_steps));
    const bool use_cache = slots > 0;

    for (int step = 0; step < total_steps; ++step) {
        const int cur_tok = (step < prompt_len) ? input_ids[step] : gen.back();
        if (cur_tok < 0 || cur_tok >= pw.vocab_size) {
            result.error_msg = "token id out of vocab range at step " +
                               std::to_string(step);
            release_rope();
            return result;
        }
        dumper.AtPromptEnd(step + 1 == prompt_len);
        if (cudaMemcpyAsync(scratch.d_tok, &cur_tok, 4, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            result.error_msg = "token upload failed";
            release_rope();
            return result;
        }
        if (RunEmbedLookup(pw.d_embed, pw.vocab_size, kHidden, scratch.d_tok, 1,
                           scratch.x, stream) != cudaSuccess ||
            RunCastHalfToF32(scratch.x, scratch.streams, kHidden, stream) !=
                cudaSuccess) {
            result.error_msg = "embed failed";
            release_rope();
            return result;
        }
        for (int j = 1; j < kHcMult; ++j)
            if (cudaMemcpyAsync(scratch.streams + static_cast<size_t>(j) * kHidden,
                                scratch.streams, kHidden * 4,
                                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                result.error_msg = "stream expand failed";
                release_rope();
                return result;
            }
        dumper.F32("embed", scratch.streams, kHcMult * kHidden, stream);

        for (int lay = 0; lay < num_layers; ++lay) {
            const LayerWeights &lw = pw.layers[static_cast<size_t>(lay)];
            const int heads = lw.shape.heads;
            const int q_dim = heads * kVHead;

            // ── attention sub-block ────────────────────────────────────────
            cudaError_t e = RunIhcMix(
                scratch.streams, lw.hc_attn.d_fn, lw.hc_attn.d_base,
                lw.hc_attn.d_scale, kHcMult, kHidden, kHcEps, kHcMagnitude,
                scratch.agg, scratch.pre, scratch.post, scratch.mix,
                scratch.rsqrt, stream);
            if (e == cudaSuccess)
                e = RunCastF32ToHalf(scratch.agg, scratch.x, kHidden, stream);
            if (e == cudaSuccess) {
                dumper.F32(absl::StrFormat("l%d_attn_hc", lay).c_str(),
                           scratch.agg, kHidden, stream);
                dumper.F32(absl::StrFormat("l%d_attn_pre", lay).c_str(),
                           scratch.pre, 2 * kHcMult, stream);
                dumper.F32(absl::StrFormat("l%d_attn_post", lay).c_str(),
                           scratch.post, 2 * kHcMult, stream);
            }
            if (e == cudaSuccess)
                e = RunRmsNormHalf(scratch.x, lw.d_input_norm, kHidden, kRmsEps,
                                   stream);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_attn_in", lay).c_str(),
                         scratch.x, kHidden, stream);
            if (e == cudaSuccess)
                e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                               lw.attn.q_a.d_packed,
                                               lw.attn.q_a.d_scales,
                                               lw.attn.q_a.global_scale, scratch.qr,
                                               1, kHidden, kQLora);
            if (e == cudaSuccess)
                e = RunRmsNormHalf(scratch.qr, lw.attn.d_q_norm, kQLora, kRmsEps,
                                   stream);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_q_resid", lay).c_str(),
                         scratch.qr, kQLora, stream);
            if (e == cudaSuccess)
                e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.qr,
                                               lw.attn.q_b.d_packed,
                                               lw.attn.q_b.d_scales,
                                               lw.attn.q_b.global_scale, scratch.q,
                                               1, kQLora, q_dim);
            if (e == cudaSuccess)
                e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                               lw.attn.kv_a.d_packed,
                                               lw.attn.kv_a.d_scales,
                                               lw.attn.kv_a.global_scale,
                                               scratch.kv, 1, kHidden,
                                               kKvLora + kQkRope);
            if (e == cudaSuccess)
                e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                               lw.attn.gate.d_packed,
                                               lw.attn.gate.d_scales,
                                               lw.attn.gate.global_scale,
                                               scratch.gate, 1, kHidden, q_dim);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_gate", lay).c_str(),
                         scratch.gate, q_dim, stream);
            // RoPE: two regimes, rotate-half over the last 64 lanes of each row.
            const float *cos_t = lw.shape.is_dsa ? d_cos_dsa + step * (kQkRope / 2)
                                                 : d_cos_swa + step * (kQkRope / 2);
            const float *sin_t = lw.shape.is_dsa ? d_sin_dsa + step * (kQkRope / 2)
                                                 : d_sin_swa + step * (kQkRope / 2);
            if (e == cudaSuccess)
                e = RunRopeHalf(scratch.q, heads, kQkHead, kQkRope, cos_t, sin_t,
                                false, stream);
            // The latent cache row is [normed kv_lora | rope key].
            if (e == cudaSuccess)
                e = cudaMemcpyAsync(scratch.latent, scratch.kv,
                                    static_cast<size_t>(kKvLora + kQkRope) * 2,
                                    cudaMemcpyDeviceToDevice, stream) ==
                            cudaSuccess
                        ? cudaSuccess
                        : cudaGetLastError();
            if (e == cudaSuccess)
                e = RunRmsNormHalf(scratch.latent, lw.attn.d_kv_norm, kKvLora,
                                   kRmsEps, stream);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_kv_pass", lay).c_str(),
                         scratch.latent, kKvLora, stream);
            if (e == cudaSuccess)
                e = RunRopeHalf(scratch.latent, 1, kKvLora + kQkRope, kQkRope,
                                cos_t, sin_t, false, stream);
            __half *cache_row = session.d_latent +
                static_cast<int64_t>(lay) * session.geom.LayerStride() + step * (kKvLora + kQkRope);
            if (e == cudaSuccess)
                e = cudaMemcpyAsync(cache_row, scratch.latent,
                                    static_cast<size_t>(kKvLora + kQkRope) * 2,
                                    cudaMemcpyDeviceToDevice, stream) == cudaSuccess
                        ? cudaSuccess
                        : cudaGetLastError();
            if (e != cudaSuccess) {
                result.error_msg = "attention prelude failed";
                release_rope();
                return result;
            }

            // ── visible positions ────────────────────────────────────────
            // Three regimes, and they are not interchangeable:
            //   * sliding layers: the 512-token causal ring;
            //   * sparse layers with T <= index_topk: DENSE causal attention,
            //     which is what the indexer degenerates to (it can select every
            //     position, so the softmax is the same);
            //   * sparse layers beyond that: IndexPool — pooled keys per complete
            //     4-token group, top-512 groups by pooled score, expanded to
            //     tokens, plus the current partial group.
            int npos = 0;
            if (!lw.shape.is_dsa) {
                const int start = std::max(0, step - kSwaWindow + 1);
                npos = step - start + 1;
                if (RunWindowPositions(scratch.pos, kSwaWindow, step, stream) !=
                    cudaSuccess) {
                    result.error_msg = "window positions failed";
                    release_rope();
                    return result;
                }
                if (cudaMemcpyAsync(scratch.pos,
                                    scratch.pos + (kSwaWindow - npos),
                                    static_cast<size_t>(npos) * 4,
                                    cudaMemcpyDeviceToDevice, stream) !=
                    cudaSuccess) {
                    result.error_msg = "window shift failed";
                    release_rope();
                    return result;
                }
            } else if (step + 1 <= kIndexTopk) {
                npos = step + 1;
                std::vector<int32_t> host_pos(static_cast<size_t>(npos));
                for (int i = 0; i < npos; ++i)
                    host_pos[static_cast<size_t>(i)] = i;
                if (cudaMemcpyAsync(scratch.pos, host_pos.data(),
                                    static_cast<size_t>(npos) * 4,
                                    cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                    result.error_msg = "dense index upload failed";
                    release_rope();
                    return result;
                }
            } else {
                const IndexerWeights &iw = lw.attn.indexer;
                __half *group_keys = session.group_keys[static_cast<size_t>(lay)];
                __half *pooled_keys = session.pooled[static_cast<size_t>(lay)];
                const int slot = step % kPoolRatio;

                // 1. This token's key: token_key_projection then its norm.
                cudaError_t ie = lsx_nvfp4::RunLsxNvfp4Gemm(
                    stream, scratch.x, iw.tok.d_packed, iw.tok.d_scales,
                    iw.tok.global_scale, group_keys + slot * kIndexHeadDim, 1,
                    kHidden, kIndexHeadDim);
                if (ie == cudaSuccess)
                    ie = RunRmsNormHalf(group_keys + slot * kIndexHeadDim,
                                        iw.d_token_key_norm, kIndexHeadDim,
                                        kRmsEps, stream);
                if (ie != cudaSuccess) {
                    result.error_msg = "indexer token key failed";
                    release_rope();
                    return result;
                }
                session.group_filled[static_cast<size_t>(lay)] = slot + 1;

                // 2. A completed group produces its pooled key.
                const int n_complete = (step + 1) / kPoolRatio;
                if ((step + 1) % kPoolRatio == 0 && n_complete > 0) {
                    const int g = n_complete - 1;
                    if (RunIndexPoolGroup(group_keys, iw.d_ape, iw.d_key_value,
                                          iw.d_key_gate, iw.d_key_norm,
                                          kPoolRatio, kIndexHeadDim,
                                          pooled_keys + g * kIndexHeadDim,
                                          stream) != cudaSuccess) {
                        result.error_msg = "indexer pooling failed";
                        release_rope();
                        return result;
                    }
                }

                // 3. Score the complete groups and keep the best groups.
                //    qk[h][d] = query_projection[h*dim + d] . q_resid   (fp32)
                //    w[h]     = head_weight_projection[h] . x          (fp32)
                const int n_tail = (step + 1) % kPoolRatio;
                if (n_complete > 0) {
                    if (lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(
                            stream, scratch.qr, iw.q.d_packed, iw.q.d_scales,
                            iw.q.global_scale, scratch.idx_q, kQLora,
                            kIndexNHeads * kIndexHeadDim) != cudaSuccess ||
                        lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(
                            stream, scratch.x, iw.hw.d_packed, iw.hw.d_scales,
                            iw.hw.global_scale, scratch.idx_wh, kHidden,
                            kIndexNHeads) != cudaSuccess ||
                        RunIndexGroupScores(scratch.idx_q, scratch.idx_wh,
                                            pooled_keys, n_complete, kIndexNHeads,
                                            kIndexHeadDim,
                                            1.0f /
                                                std::sqrt(static_cast<float>(
                                                    kIndexNHeads)) /
                                                std::sqrt(static_cast<float>(
                                                    kIndexHeadDim)),
                                            scratch.idx_scores, stream) !=
                            cudaSuccess ||
                        RunIndexSelectGroups(scratch.idx_scores, n_complete,
                                             kPoolRatio, kGroupTopk,
                                             n_complete * kPoolRatio, n_tail,
                                             scratch.pos, kMaxVisible,
                                             scratch.idx_used, stream) !=
                            cudaSuccess) {
                        result.error_msg = "indexer selection failed";
                        release_rope();
                        return result;
                    }
                    // The kernel pads with -1: count the visible positions.
                    std::vector<int32_t> host_pos(kMaxVisible);
                    if (cudaMemcpyAsync(host_pos.data(), scratch.pos,
                                        static_cast<size_t>(kMaxVisible) * 4,
                                        cudaMemcpyDeviceToHost, stream) !=
                            cudaSuccess ||
                        cudaStreamSynchronize(stream) != cudaSuccess) {
                        result.error_msg = "indexer readback failed";
                        release_rope();
                        return result;
                    }
                    npos = 0;
                    while (npos < kMaxVisible &&
                           host_pos[static_cast<size_t>(npos)] >= 0)
                        ++npos;
                } else {
                    // Fewer than one complete group: only the tail is visible.
                    npos = n_tail;
                    std::vector<int32_t> host_pos(static_cast<size_t>(npos));
                    for (int i = 0; i < npos; ++i)
                        host_pos[static_cast<size_t>(i)] = i;
                    if (cudaMemcpyAsync(scratch.pos, host_pos.data(),
                                        static_cast<size_t>(npos) * 4,
                                        cudaMemcpyHostToDevice, stream) !=
                        cudaSuccess) {
                        result.error_msg = "indexer tail upload failed";
                        release_rope();
                        return result;
                    }
                }
            }
            if (npos <= 0 || npos > kMaxVisible) {
                result.error_msg = "empty visible set";
                release_rope();
                return result;
            }

            // Scores and the softmax-weighted latent sum.
            e = RunMlaQkProject(scratch.q, lw.attn.kv_b, heads, kQkNope,
                                kQkHead, scratch.qk, stream);
            if (e == cudaSuccess)
                e = RunMlaSinkSoftmax(
                    scratch.qk, scratch.q, kQkHead, kQkNope, session.d_latent +
                        static_cast<int64_t>(lay) * session.geom.LayerStride(),
                    kKvLora + kQkRope, scratch.pos, npos, heads, kKvLora,
                    kQkRope, lw.attn.d_sink,
                    1.0f / std::sqrt(static_cast<float>(kQkHead)), scratch.scores,
                    scratch.acc, stream);
            if (e == cudaSuccess) {
                dumper.F32(absl::StrFormat("l%d_probs", lay).c_str(),
                           scratch.scores, static_cast<int64_t>(heads) * npos,
                           stream);
                dumper.F32(absl::StrFormat("l%d_acc", lay).c_str(),
                           scratch.acc, static_cast<int64_t>(heads) * kKvLora,
                           stream);
            }
            if (e == cudaSuccess)
                e = RunMlaValueProject(scratch.acc, lw.attn.kv_b_value, heads,
                                       kVHead, kKvLora, scratch.o, stream);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_o", lay).c_str(), scratch.o,
                         static_cast<int64_t>(heads) * kVHead, stream);
            if (e == cudaSuccess)
                e = RunGatedScale(scratch.o, scratch.gate, q_dim, stream);
            if (e == cudaSuccess)
                e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.o,
                                               lw.attn.o.d_packed,
                                               lw.attn.o.d_scales,
                                               lw.attn.o.global_scale,
                                               scratch.attn_out, 1, q_dim,
                                               kHidden);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_o_proj", lay).c_str(),
                         scratch.attn_out, kHidden, stream);
            if (e == cudaSuccess)
                e = RunRmsNormHalf(scratch.attn_out, lw.attn.d_post_proj_norm,
                                   kHidden, kRmsEps, stream);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_attn_out", lay).c_str(),
                         scratch.attn_out, kHidden, stream);
            if (e == cudaSuccess)
                e = RunCastHalfToF32(scratch.attn_out, scratch.agg, kHidden,
                                     stream);
            if (e == cudaSuccess)
                e = RunIhcPost(scratch.agg, scratch.post, kHcMult, kHidden,
                               scratch.streams, stream);
            if (e != cudaSuccess) {
                result.error_msg = std::string("attention failed at layer ") +
                                   std::to_string(lay);
                release_rope();
                return result;
            }

            // ── FFN sub-block ─────────────────────────────────────────────
            e = RunIhcMix(scratch.streams, lw.hc_mlp.d_fn, lw.hc_mlp.d_base,
                          lw.hc_mlp.d_scale, kHcMult, kHidden, kHcEps,
                          kHcMagnitude, scratch.agg, scratch.pre, scratch.post,
                          scratch.mix, scratch.rsqrt, stream);
            if (e == cudaSuccess) {
                dumper.F32(absl::StrFormat("l%d_ffn_hc", lay).c_str(),
                           scratch.agg, kHidden, stream);
                dumper.F32(absl::StrFormat("l%d_ffn_pre", lay).c_str(),
                           scratch.pre, 2 * kHcMult, stream);
                dumper.F32(absl::StrFormat("l%d_ffn_post", lay).c_str(),
                           scratch.post, 2 * kHcMult, stream);
            }
            if (e == cudaSuccess)
                e = RunCastF32ToHalf(scratch.agg, scratch.x, kHidden, stream);
            if (e == cudaSuccess)
                e = RunRmsNormHalf(scratch.x, lw.d_post_attn_norm, kHidden,
                                   kRmsEps, stream);
            if (e == cudaSuccess)
                dumper.H(absl::StrFormat("l%d_ffn_in", lay).c_str(),
                         scratch.x, kHidden, stream);
            if (e != cudaSuccess) {
                result.error_msg = "ffn prelude failed";
                release_rope();
                return result;
            }

            if (!lw.shape.is_moe) {
                e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                               lw.d_gate.d_packed,
                                               lw.d_gate.d_scales,
                                               lw.d_gate.global_scale, scratch.d_gu,
                                               1, kHidden, kDenseIntermediate);
                if (e == cudaSuccess)
                    dumper.H(absl::StrFormat("l%d_d_gate", lay).c_str(),
                             scratch.d_gu, kDenseIntermediate, stream);
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                                   lw.d_up.d_packed,
                                                   lw.d_up.d_scales,
                                                   lw.d_up.global_scale,
                                                   scratch.d_gu + kDenseIntermediate,
                                                   1, kHidden, kDenseIntermediate);
                if (e == cudaSuccess)
                    dumper.H(absl::StrFormat("l%d_d_up", lay).c_str(),
                             scratch.d_gu + kDenseIntermediate,
                             kDenseIntermediate, stream);
                if (e == cudaSuccess)
                    e = RunSwigluClamp(scratch.d_gu, scratch.d_gu + kDenseIntermediate,
                                       scratch.d_act, kDenseIntermediate,
                                       kSwigluLimit, stream);
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.d_act,
                                                   lw.d_down.d_packed,
                                                   lw.d_down.d_scales,
                                                   lw.d_down.global_scale,
                                                   scratch.ffn_out, 1,
                                                   kDenseIntermediate, kHidden);
                if (e == cudaSuccess)
                    dumper.H(absl::StrFormat("l%d_mlp_raw", lay).c_str(),
                             scratch.ffn_out, kHidden, stream);
                if (e == cudaSuccess)
                    e = RunRmsNormHalf(scratch.ffn_out, lw.d_post_fc2_norm, kHidden,
                                       kRmsEps, stream);
                if (e == cudaSuccess)
                    dumper.H(absl::StrFormat("l%d_mlp_out", lay).c_str(),
                             scratch.ffn_out, kHidden, stream);
                if (e != cudaSuccess) {
                    result.error_msg = "dense ffn failed";
                    release_rope();
                    return result;
                }
            } else {
                // Router (fp32 sigmoid scores, selection-only bias).
                e = RunRouterLogits(scratch.x, lw.d_router, kNumRouted, kHidden,
                                    scratch.router_scores, stream);
                if (e == cudaSuccess)
                    e = RunRouterTopK(scratch.router_scores, lw.d_router_bias,
                                      kNumRouted, kTopK, kRouteScale,
                                      scratch.router_idx, scratch.router_w,
                                      stream);
                // LatentMoE: the experts read and write the 1536-wide latent.
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                                   lw.r_down.d_packed,
                                                   lw.r_down.d_scales,
                                                   lw.r_down.global_scale,
                                                   scratch.lat, 1, kHidden,
                                                   kLatent);
                if (e == cudaSuccess)
                    e = cudaMemsetAsync(scratch.moe_acc, 0, kLatent * 4, stream) ==
                                cudaSuccess
                            ? cudaSuccess
                            : cudaGetLastError();
                if (e != cudaSuccess) {
                    result.error_msg = "router failed";
                    release_rope();
                    return result;
                }
                std::vector<int32_t> h_idx(kTopK);
                std::vector<float> h_w(kTopK);
                if (cudaMemcpyAsync(h_idx.data(), scratch.router_idx, kTopK * 4,
                                    cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                    cudaMemcpyAsync(h_w.data(), scratch.router_w, kTopK * 4,
                                    cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess) {
                    result.error_msg = "router readback failed";
                    release_rope();
                    return result;
                }
                // `gu` doubles as the per-expert down-projection output once the
                // SwiGLU has consumed the gate/up halves.
                __half *down_out = scratch.gu;
                for (int k = 0; k < kTopK; ++k) {
                    const int eid = h_idx[static_cast<size_t>(k)];
                    const int flat = lay * expert_db.experts_per_layer + eid;
                    EnsureExpertResident(expert_db, lay, eid, stream);
                    const size_t idx = static_cast<size_t>(flat);
                    if (idx >= expert_db.entries.size()) continue;
                    const ExpertEntry &en = expert_db.entries[idx];
                    if (!en.host_w1 || !en.host_w2 || !en.host_w3) continue;
                    int slot = cache.Lookup(flat);
                    if (slot < 0) {
                        int evicted = -1;
                        slot = cache.Insert(flat, &evicted);
                        uint8_t *dst = scratch.d_slots +
                            static_cast<size_t>(slot) * slot_layout.block;
                        auto cp = [&](size_t off, const uint8_t *src,
                                      int64_t bytes) {
                            if (bytes <= 0) return cudaSuccess;
                            return cudaMemcpyAsync(dst + off, src,
                                                   static_cast<size_t>(bytes),
                                                   cudaMemcpyHostToDevice,
                                                   stream);
                        };
                        cudaError_t ce = cp(slot_layout.w1, en.host_w1, en.w1_bytes);
                        if (ce == cudaSuccess)
                            ce = cp(slot_layout.w3, en.host_w3, en.w3_bytes);
                        if (ce == cudaSuccess)
                            ce = cp(slot_layout.w2, en.host_w2, en.w2_bytes);
                        if (ce == cudaSuccess)
                            ce = cp(slot_layout.s1, en.host_w1_scales,
                                    en.w1_scales_bytes);
                        if (ce == cudaSuccess)
                            ce = cp(slot_layout.s3, en.host_w3_scales,
                                    en.w3_scales_bytes);
                        if (ce == cudaSuccess)
                            ce = cp(slot_layout.s2, en.host_w2_scales,
                                    en.w2_scales_bytes);
                        if (ce != cudaSuccess) {
                            result.error_msg = "expert staging failed";
                            release_rope();
                            return result;
                        }
                    } else {
                        cache.Touch(slot);
                    }
                    const uint8_t *blk = scratch.d_slots +
                        static_cast<size_t>(slot) * slot_layout.block;
                    cudaError_t ce = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, scratch.lat, blk + slot_layout.w1,
                        blk + slot_layout.s1, en.w1_global_scale, scratch.gu, 1,
                        kLatent, kMoeIntermediate);
                    if (ce == cudaSuccess)
                        ce = lsx_nvfp4::RunLsxNvfp4Gemm(
                            stream, scratch.lat, blk + slot_layout.w3,
                            blk + slot_layout.s3, en.w3_global_scale,
                            scratch.gu + kMoeIntermediate, 1, kLatent,
                            kMoeIntermediate);
                    if (ce == cudaSuccess)
                        ce = RunSwigluClamp(scratch.gu,
                                            scratch.gu + kMoeIntermediate,
                                            scratch.act, kMoeIntermediate,
                                            kSwigluLimit, stream);
                    if (ce == cudaSuccess)
                        ce = lsx_nvfp4::RunLsxNvfp4Gemm(
                            stream, scratch.act, blk + slot_layout.w2,
                            blk + slot_layout.s2, en.w2_global_scale, down_out,
                            1, kMoeIntermediate, kLatent);
                    if (ce == cudaSuccess)
                        ce = RunAccumulateScaled(down_out, scratch.moe_acc,
                                                 kLatent,
                                                 h_w[static_cast<size_t>(k)],
                                                 stream);
                    if (ce != cudaSuccess) {
                        result.error_msg = "expert ffn failed";
                        release_rope();
                        return result;
                    }
                }
                // routed_expert_up_proj(router-weighted latent sum) -> hidden.
                e = RunCastF32ToHalf(scratch.moe_acc, scratch.lat, kLatent,
                                     stream);
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.lat,
                                                   lw.r_up.d_packed,
                                                   lw.r_up.d_scales,
                                                   lw.r_up.global_scale,
                                                   scratch.ffn_out, 1, kLatent,
                                                   kHidden);
                // Shared expert reads the FULL-width block input.
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.x,
                                                   lw.s_gate.d_packed,
                                                   lw.s_gate.d_scales,
                                                   lw.s_gate.global_scale,
                                                   scratch.s_gu, 1, kHidden,
                                                   kSharedIntermediate);
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, scratch.x, lw.s_up.d_packed, lw.s_up.d_scales,
                        lw.s_up.global_scale, scratch.s_gu + kSharedIntermediate,
                        1, kHidden, kSharedIntermediate);
                if (e == cudaSuccess)
                    e = RunSwigluClamp(scratch.s_gu,
                                       scratch.s_gu + kSharedIntermediate,
                                       scratch.s_act, kSharedIntermediate,
                                       kSwigluLimit, stream);
                // routed + shared, then the sandwich norm on the total.
                if (e == cudaSuccess)
                    e = RunCastHalfToF32(scratch.ffn_out, scratch.agg, kHidden,
                                         stream);
                if (e == cudaSuccess)
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, scratch.s_act,
                                                   lw.s_down.d_packed,
                                                   lw.s_down.d_scales,
                                                   lw.s_down.global_scale,
                                                   scratch.ffn_out, 1,
                                                   kSharedIntermediate, kHidden);
                if (e == cudaSuccess)
                    e = RunAccumulateScaled(scratch.ffn_out, scratch.agg, kHidden,
                                            1.0f, stream);
                if (e == cudaSuccess)
                    e = RunCastF32ToHalf(scratch.agg, scratch.ffn_out, kHidden,
                                         stream);
                if (e == cudaSuccess)
                    dumper.H(absl::StrFormat("l%d_mlp_raw", lay).c_str(),
                             scratch.ffn_out, kHidden, stream);
                if (e == cudaSuccess)
                    e = RunRmsNormHalf(scratch.ffn_out, lw.d_combine_norm, kHidden,
                                       kRmsEps, stream);
                if (e == cudaSuccess)
                    dumper.H(absl::StrFormat("l%d_mlp_out", lay).c_str(),
                             scratch.ffn_out, kHidden, stream);
                if (e != cudaSuccess) {
                    result.error_msg = "moe failed";
                    release_rope();
                    return result;
                }
            }
            if (e != cudaSuccess) {
                result.error_msg = "ffn failed";
                release_rope();
                return result;
            }
            // The iHC post term multiplies the SUBLAYER OUTPUT AFTER ITS SANDWICH
            // NORM — that is what the norm is for ("sandwich"), and it is what the
            // reference does (`post * post_mlp_layernorm(mlp(x)) + residual`).
            //
            // `agg` is the buffer `RunIhcPost` reads, and the two branches leave
            // the normed value in DIFFERENT buffers:
            //   * dense layer: `ffn_out` was normed, then cast down into `agg`;
            //   * MoE layer:   `agg` still holds the routed+shared sum from
            //                  `RunAccumulateScaled`, i.e. the PRE-norm value,
            //                  while the normed output sits in `ffn_out`.
            // The cast below used to be guarded by `!is_moe`, so on all 41 MoE
            // layers `post` multiplied the pre-norm sum — whose rms is ~240x
            // below the normed output (0.0027 vs 0.66 at layer 0) — and the MLP
            // contribution to the residual was effectively dropped. The residual
            // reconstruction invariant (out == prev + post*attn + post*mlp) failed
            // by 27% at layer 1 while the reference held it to 0.2%; see
            // docs/open-tasks/live/models/hy4-flash.md H1.
            if (e == cudaSuccess)
                e = RunCastHalfToF32(scratch.ffn_out, scratch.agg, kHidden,
                                     stream);
            if (e == cudaSuccess)
                e = RunIhcPost(scratch.agg, scratch.post, kHcMult, kHidden,
                               scratch.streams, stream);
            if (e != cudaSuccess) {
                result.error_msg = std::string("ffn post failed at layer ") +
                                   std::to_string(lay);
                release_rope();
                return result;
            }
            dumper.F32(absl::StrFormat("l%d_out", lay).c_str(),
                       scratch.streams, kHcMult * kHidden, stream);
        }

        // ── head ───────────────────────────────────────────────────────────
        cudaError_t e = RunIhcHead(scratch.streams, pw.head.d_fn, pw.head.d_base,
                                   pw.head.d_scale, kHcMult, kHidden, kHcEps,
                                   scratch.agg, scratch.pre, scratch.mix,
                                   scratch.rsqrt, stream);
        if (e == cudaSuccess)
            dumper.F32("hc_head", scratch.agg, kHidden, stream);
        if (e == cudaSuccess)
            e = RunCastF32ToHalf(scratch.agg, scratch.x, kHidden, stream);
        if (e == cudaSuccess)
            e = RunRmsNormHalf(scratch.x, pw.d_final_norm, kHidden, kRmsEps,
                               stream);
        if (e == cudaSuccess && dumper.On())
            e = RunCastHalfToF32(scratch.x, scratch.agg, kHidden, stream);
        if (e == cudaSuccess)
            dumper.F32("norm", scratch.agg, kHidden, stream);
        if (e == cudaSuccess)
            e = RunHeadLogits(pw.d_lm_head, scratch.x, pw.vocab_size, kHidden,
                              scratch.logits, stream);
        if (e == cudaSuccess)
            dumper.F32("logits", scratch.logits, pw.vocab_size, stream);
        if (e != cudaSuccess) {
            result.error_msg = "head failed";
            release_rope();
            return result;
        }

        int32_t next = 0;
        float maxv = 0.0f;
        if (sampler.Sample(stream, scratch.logits, pw.vocab_size,
                           sample::LogitType::kF32, scratch.d_argmax,
                           scratch.d_maxv) != cudaSuccess ||
            cudaMemcpyAsync(&next, scratch.d_argmax, 4, cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaMemcpyAsync(&maxv, scratch.d_maxv, 4, cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            result.error_msg = "sample failed";
            release_rope();
            return result;
        }

        if (step + 1 >= prompt_len) {
            gen.push_back(next);
            if (step + 1 == total_steps) {
                // The LAST decode step's logits are the determinism artifact
                // the self-check hashes.
                result.final_logits.resize(static_cast<size_t>(pw.vocab_size));
                if (cudaMemcpyAsync(result.final_logits.data(), scratch.logits,
                                    static_cast<size_t>(pw.vocab_size) * 4,
                                    cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess)
                    result.final_logits.clear();
            }
            if (result.first_gen_tok_id < 0) {
                result.first_gen_tok_id = next;
                result.first_gen_maxv = maxv;
            }
            if (gs) {
                gs->Emit(static_cast<std::uint32_t>(next),
                         static_cast<std::uint32_t>(gen.size() - 1), maxv);
                if (gs->Aborted()) {
                    release_rope();
                    result.ok = true;
                    result.gen_tokens = gen;
                    return result;
                }
            }
            bool stop = false;
            if (!ignore_eos)
                for (int id : eos_ids)
                    if (id == next) stop = true;
            if (stop) break;
        }
    }

    release_rope();
    result.ok = true;
    result.gen_tokens = gen;
    session.kv_rows = prompt_len + static_cast<int64_t>(gen.size()) - 1;
    session.last_tok = gen.empty() ? -1 : gen.back();
    session.gen_tail = gen;
    return result;
}

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
