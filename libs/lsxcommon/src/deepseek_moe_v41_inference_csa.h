// deepseek_moe_v41_inference_csa.h — CED shared-attention runtime
// (model.py:1166). Source layers publish one slot each; consumers read the
// most recent value during the in-order layer walk, so a single slot per
// resource is enough within a forward. Reset() at the start of every sequence.
#pragma once

#include <cstdint>

#include "deepseek_moe_v41_constants.h"

namespace lsxcommon {
namespace deepseek_v41 {

struct CsaRuntime {
    // Published by kv_source layers (own a Compressor).
    const float *compress_kv = nullptr;
    int compress_len = 0;
    // Published by kv_source layers whose Indexer owns index keys.
    const uint16_t *index_k = nullptr;
    int index_k_positions = 0;
    // Published by index_source layers.
    const int32_t *topk_idxs = nullptr;
    int topk = 0;
    // Published by the candidate-source layer; consumed downstream.
    const uint8_t *candidates = nullptr;
    int candidate_positions = 0;

    void Reset()
    {
        compress_kv = nullptr;
        compress_len = 0;
        index_k = nullptr;
        index_k_positions = 0;
        topk_idxs = nullptr;
        topk = 0;
        candidates = nullptr;
        candidate_positions = 0;
    }

    void PublishCompressKv(const float *kv, int len)
    {
        compress_kv = kv;
        compress_len = len;
    }
    void PublishIndexK(const uint16_t *k, int positions)
    {
        index_k = k;
        index_k_positions = positions;
    }
    void PublishTopk(const int32_t *idxs, int k)
    {
        topk_idxs = idxs;
        topk = k;
    }
    void PublishCandidates(const uint8_t *mask, int n)
    {
        candidates = mask;
        candidate_positions = n;
    }
};

// A layer runs its own indexer only at index_source layers; the rest reuse the
// source's topk_idxs. index keys (wk) exist only where the layer compresses.
constexpr bool UsesSharedTopk(int layer) { return !IsIndexSource(layer); }
constexpr bool OwnsIndexK(int layer) { return IsKvSource(layer); }

// Two-level indexer: a layer downstream of the candidate source filters its own
// scores through the source's block mask.
constexpr bool UsesCandidates(int layer)
{
    return kCandidateSourceLayer >= 0 && kCandidateSourceLayer < layer;
}

} // namespace deepseek_v41
} // namespace lsxcommon
