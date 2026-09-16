// ── TDD Phase B Task 4: DeepSeekInferArena + zero-alloc route_token ─────────
// Asserts that the per-run inference arena performs exactly ONE device
// cudaMalloc + ONE pinned cudaMallocHost (spec item 12) and that route_token
// no longer touches any CUDA allocation API (malloc-per-layer kill). The
// allocation-count assertions use the global InferenceAllocCount() atomic that
// the arena and route_token share with the production pipeline.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "deepseek_moe_load.cuh"

using lsxcommon::gpu::CSAState;
using lsxcommon::gpu::DeepSeekInferArena;
using lsxcommon::gpu::InferenceAllocCount;
using lsxcommon::gpu::route_token;

namespace {

// Deterministic synthetic router fixture: num_routed=16, hidden=8, identity-ish
// gate so routing is reproducible. Mirrors the plan's Step-1 test exactly.
void build_synthetic_router(cudaStream_t &stream,
                            __half *&d_gate, __half *&d_hidden,
                            float *&h_bias, int *&h_top, float *&h_weights,
                            __half *&d_logits_scratch,
                            float *&h_logits_scratch)
{
    constexpr int kHidden = 8;
    constexpr int kRouted = 16;
    constexpr int kTopK = 2;

    REQUIRE(cudaMalloc(&d_gate, kRouted * kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_hidden, kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_bias, kRouted * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_top, kTopK * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_weights, kTopK * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_logits_scratch, kRouted * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_logits_scratch, kRouted * sizeof(float)) == cudaSuccess);

    // identity-ish gate so routing is deterministic
    std::vector<__half> gate(static_cast<size_t>(kRouted) * kHidden);
    for (int i = 0; i < kRouted * kHidden; ++i) gate[i] = __float2half(0.01f);
    for (int i = 0; i < kRouted; ++i)
        gate[static_cast<size_t>(i) * kHidden] = __float2half(1.0f);
    std::vector<__half> hidden(static_cast<size_t>(kHidden), __float2half(1.0f));
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 0; i < kRouted; ++i) h_bias[i] = static_cast<float>(i);

    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
}

} // namespace

TEST_CASE("route_token performs zero allocations after arena refactor",
          "[deepseek][arena]")
{
    cudaFree(0);
    constexpr int kHidden = 8;
    constexpr int kRouted = 16;
    constexpr int kTopK = 2;

    cudaStream_t stream = nullptr;
    __half *d_gate = nullptr;
    __half *d_hidden = nullptr;
    float *h_bias = nullptr;
    int *h_top = nullptr;
    float *h_weights = nullptr;
    __half *d_logits_scratch = nullptr;
    float *h_logits_scratch = nullptr;
    build_synthetic_router(stream, d_gate, d_hidden, h_bias, h_top,
                           h_weights, d_logits_scratch, h_logits_scratch);

    auto &count = InferenceAllocCount();
    const auto before = count.load();

    cudaError_t e = route_token(stream, d_gate, d_hidden,
                                kHidden, kRouted,
                                nullptr, 0, 3 /*token_id*/,
                                h_bias, kTopK,
                                d_logits_scratch, h_logits_scratch,
                                h_top, h_weights);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto after = count.load();
    REQUIRE(after == before);            // route_token added ZERO allocations

    REQUIRE(h_top[0] >= 0);
    REQUIRE(h_top[0] < kRouted);
    REQUIRE(h_top[1] >= 0);
    REQUIRE(h_top[1] < kRouted);
    REQUIRE(h_top[0] != h_top[1]);

    cudaStreamDestroy(stream);
    cudaFreeHost(h_logits_scratch);
    cudaFree(d_logits_scratch);
    cudaFreeHost(h_weights);
    cudaFreeHost(h_top);
    cudaFreeHost(h_bias);
    cudaFree(d_hidden);
    cudaFree(d_gate);
}

TEST_CASE("DeepSeekInferArena allocates one device + one pinned block per run",
          "[deepseek][arena]")
{
    cudaFree(0);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // Synthetic geometry: 3 layers, ratios {4, 0, 128}, tiny sequence + vocab.
    constexpr int kLayers = 3;
    constexpr int kSeq = 8;
    constexpr int kVocab = 100;
    int compress_ratios[kLayers] = {4, 0, 128};

    auto &count = InferenceAllocCount();
    const auto before = count.load();

    DeepSeekInferArena arena;
    cudaError_t e = arena.Allocate(stream, kLayers, kSeq, compress_ratios, kVocab);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Exactly +2 (device + pinned host block).
    REQUIRE(count.load() == before + 2);

    // Geometry sanity: every host-facing buffer must be bound into the block.
    REQUIRE(arena.d_base != nullptr);
    REQUIRE(arena.h_base != nullptr);
    REQUIRE(arena.bytes > 0);
    REQUIRE(arena.host_bytes > 0);
    REQUIRE(arena.d_hidden != nullptr);
    REQUIRE(arena.d_single != nullptr);
    REQUIRE(arena.d_logits != nullptr);
    REQUIRE(arena.d_kv_cache != nullptr);
    REQUIRE(arena.d_next_token != nullptr);
    REQUIRE(arena.d_pack_stage != nullptr);
    REQUIRE(arena.h_topk != nullptr);
    REQUIRE(arena.h_top_weights != nullptr);
    REQUIRE(arena.d_routing_scratch != nullptr);
    REQUIRE(arena.h_routing_scratch != nullptr);
    REQUIRE(arena.csa_states != nullptr);

    // CSA binding: ratio=0 layer must have null state; ratio>0 layers bound.
    REQUIRE(arena.csa_states[1].d_comp_kv_state == nullptr);
    REQUIRE(arena.csa_states[1].d_comp_score_state == nullptr);
    REQUIRE(arena.csa_states[0].d_comp_kv_state != nullptr);
    REQUIRE(arena.csa_states[0].d_comp_score_state != nullptr);
    REQUIRE(arena.csa_states[0].d_compressed_kv != nullptr);
    REQUIRE(arena.csa_states[2].d_comp_kv_state != nullptr);
    REQUIRE(arena.csa_states[2].d_idx_compressed_kv != nullptr);

    // All CSA pointers must fall inside the single device block.
    for (int l = 0; l < kLayers; ++l) {
        CSAState &s = arena.csa_states[l];
        if (s.d_comp_kv_state) {
            REQUIRE(reinterpret_cast<uint8_t*>(s.d_comp_kv_state) >= arena.d_base);
            REQUIRE(reinterpret_cast<uint8_t*>(s.d_comp_kv_state) <
                    arena.d_base + arena.bytes);
        }
    }

    arena.Free();
    REQUIRE(arena.d_base == nullptr);
    REQUIRE(arena.h_base == nullptr);
    REQUIRE(arena.csa_states == nullptr);
    // Free must not touch the allocation counter.
    REQUIRE(count.load() == before + 2);

    cudaStreamDestroy(stream);
}