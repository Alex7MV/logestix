#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

// Host reference: sigmoid scoring, noaux_tc top-k selection by
// (s_e + bias_e) descending with lower-id tie-break, norm_topk_prob
// renormalization, then routed-scaling.
void RouterReference(const std::vector<float> &scores,
                     const std::vector<float> &bias, int n, int top_k,
                     std::vector<int> &ids, std::vector<float> &weights)
{
    ids.assign(top_k, -1);
    weights.assign(top_k, 0.0f);
    std::vector<char> chosen(static_cast<size_t>(n), 0);
    for (int t = 0; t < top_k; ++t) {
        int best = -1;
        float best_key = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < n; ++e) {
            if (chosen[e]) continue;
            float s = 1.0f / (1.0f + std::exp(-scores[e]));
            float key = s + bias[e];
            // Lower id wins ties (strict > keeps the earliest).
            if (key > best_key) {
                best_key = key;
                best = e;
            }
        }
        ids[t] = best;
        chosen[best] = 1;
    }
    float sum = 0.0f;
    for (int t = 0; t < top_k; ++t)
        sum += 1.0f / (1.0f + std::exp(-scores[ids[t]]));
    for (int t = 0; t < top_k; ++t)
        weights[t] = (1.0f / (1.0f + std::exp(-scores[ids[t]])) / sum) *
                     lsxcommon::kimi::kKimiRoutedScalingFactor;
}

} // namespace

using lsxcommon::kimi::KimiRouteBegin;
using lsxcommon::kimi::KimiRouteFinish;
using lsxcommon::kimi::KimiRouterTopK;
using lsxcommon::kimi::KimiRouterTopKHost;

TEST_CASE("Kimi router host reference is deterministic and normalized",
          "[kimi][router]")
{
    const int n = 384, top_k = 8;
    std::vector<float> scores(n), bias(n);
    for (int i = 0; i < n; ++i) {
        scores[i] = -3.0f + 0.02f * i;
        bias[i] = 0.001f * ((i * 37) % 11);
    }
    std::vector<int> ids1, ids2;
    std::vector<float> w1, w2;
    RouterReference(scores, bias, n, top_k, ids1, w1);
    RouterReference(scores, bias, n, top_k, ids2, w2);
    REQUIRE(ids1 == ids2);

    // Weights renormalize to the routing scale factor.
    float sum = 0.0f;
    for (float w : w1) sum += w;
    REQUIRE(std::abs(sum - lsxcommon::kimi::kKimiRoutedScalingFactor) <
            1e-5f * lsxcommon::kimi::kKimiRoutedScalingFactor);

    // All ids distinct.
    std::vector<int> sorted_ids = ids1;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    REQUIRE(std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) ==
            sorted_ids.end());
}

TEST_CASE("Kimi router GPU matches host reference", "[kimi][router]")
{
    const int n = 384, top_k = 8;
    std::vector<float> scores(n), bias(n);
    for (int i = 0; i < n; ++i) {
        scores[i] = -3.0f + 0.02f * ((i * 13) % n);
        bias[i] = 0.001f * ((i * 37) % 11);
    }

    std::vector<int> ref_ids;
    std::vector<float> ref_w;
    RouterReference(scores, bias, n, top_k, ref_ids, ref_w);

    float *d_scores = nullptr, *d_bias = nullptr, *d_weights = nullptr;
    int *d_ids = nullptr;
    cudaMalloc(&d_scores, n * sizeof(float));
    cudaMalloc(&d_bias, n * sizeof(float));
    cudaMalloc(&d_ids, top_k * sizeof(int));
    cudaMalloc(&d_weights, top_k * sizeof(float));
    cudaMemcpy(d_scores, scores.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    REQUIRE(KimiRouterTopK(d_scores, d_bias, n, top_k, d_ids, d_weights,
                           stream) == cudaSuccess);
    cudaStreamSynchronize(stream);

    std::vector<int> out_ids(top_k);
    std::vector<float> out_w(top_k);
    cudaMemcpy(out_ids.data(), d_ids, top_k * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_w.data(), d_weights, top_k * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Same experts selected (order-insensitive comparison of sets).
    std::vector<int> sorted_out = out_ids, sorted_ref = ref_ids;
    std::sort(sorted_out.begin(), sorted_out.end());
    std::sort(sorted_ref.begin(), sorted_ref.end());
    REQUIRE(sorted_out == sorted_ref);

    // Per-expert weight parity (tolerance for host/device exp difference).
    for (int t = 0; t < top_k; ++t) {
        auto ref_it = std::find(ref_ids.begin(), ref_ids.end(), out_ids[t]);
        REQUIRE(ref_it != ref_ids.end());
        float expected = ref_w[ref_it - ref_ids.begin()];
        INFO("expert " << out_ids[t]);
        REQUIRE(std::abs(out_w[t] - expected) <= 2e-6f * expected);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_scores); cudaFree(d_bias);
    cudaFree(d_ids); cudaFree(d_weights);
}

// ── Split-phase router handshake (F-11): bitwise equivalence vs fused ──────
TEST_CASE("Kimi split-phase route matches fused path bitwise",
          "[kimi][router][split]")
{
    const int n = 384, top_k = 8;
    std::vector<float> scores(n), bias(n);
    // Adversarial: duplicate score values exercise the tie-break rule; bias
    // perturbs ordering.
    for (int i = 0; i < n; ++i) {
        scores[i] = -3.0f + 0.05f * ((i * 7) % 40);   // many duplicates
        bias[i] = 0.001f * ((i * 37) % 11);
    }

    // Fused reference: device top-k + blocking sync.
    float *d_scores = nullptr, *d_bias = nullptr, *d_w_ref = nullptr;
    int *d_ids_ref = nullptr;
    cudaMalloc(&d_scores, n * sizeof(float));
    cudaMalloc(&d_bias, n * sizeof(float));
    cudaMalloc(&d_ids_ref, top_k * sizeof(int));
    cudaMalloc(&d_w_ref, top_k * sizeof(float));
    cudaMemcpy(d_scores, scores.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreate(&st);
    REQUIRE(KimiRouterTopK(d_scores, d_bias, n, top_k, d_ids_ref, d_w_ref,
                           st) == cudaSuccess);
    std::vector<int> ref_ids(top_k);
    std::vector<float> ref_w(top_k);
    cudaMemcpy(ref_ids.data(), d_ids_ref, top_k * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_w.data(), d_w_ref, top_k * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(st);

    // Split path: pinned staging + event + busy-poll.
    int *h_ids = nullptr;
    float *h_w = nullptr;
    REQUIRE(cudaHostAlloc(&h_ids, top_k * sizeof(int),
                          cudaHostAllocDefault) == cudaSuccess);
    REQUIRE(cudaHostAlloc(&h_w, top_k * sizeof(float),
                          cudaHostAllocDefault) == cudaSuccess);
    int *d_ids2 = nullptr;
    float *d_w2 = nullptr;
    cudaMalloc(&d_ids2, top_k * sizeof(int));
    cudaMalloc(&d_w2, top_k * sizeof(float));
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) ==
            cudaSuccess);
    // Prime the event so Begin->Finish re-records over a known state.
    cudaEventRecord(ev, st);

    REQUIRE(KimiRouteBegin(d_scores, d_bias, n, top_k, d_ids2, d_w2, h_ids,
                           h_w, ev, st) == cudaSuccess);
    // Production enqueues unrelated work here (batch resolve/upload); the
    // event fence must not depend on anything else on the stream.
    REQUIRE(KimiRouteFinish(ev) == cudaSuccess);

    for (int k = 0; k < top_k; ++k) {
        INFO("k=" << k << " ref=" << ref_ids[k] << "/" << ref_w[k]
                  << " split=" << h_ids[k] << "/" << h_w[k]);
        REQUIRE(h_ids[k] == ref_ids[k]);
        REQUIRE(h_w[k] == ref_w[k]);   // bitwise float compare
    }

    cudaEventDestroy(ev);
    cudaFreeHost(h_ids);
    cudaFreeHost(h_w);
    cudaFree(d_ids2);
    cudaFree(d_w2);
    cudaFree(d_scores);
    cudaFree(d_bias);
    cudaFree(d_ids_ref);
    cudaFree(d_w_ref);
    cudaStreamDestroy(st);
}
