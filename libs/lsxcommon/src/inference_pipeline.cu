#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/tokenizer.h"
#include "inference_load.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>

namespace lsxcommon {
namespace gpu {

// Forward declarations from other .cu files
extern __global__ void dequant_nvfp4_to_half_kernel(
    const uint8_t* __restrict__, const uint8_t* __restrict__, float,
    __half* __restrict__, int);

// From inference_norm.cu
cudaError_t run_rmsnorm(
    cudaStream_t stream, const __half *d_input, const __half *d_weight,
    __half *d_output, int dim, float eps);

// From inference_moe.cu
cudaError_t run_router(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    int *h_topk_indices, int topk);

void run_shared_experts_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float w1_gs, float w2_gs, float w3_gs,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    __half *d_intermediate, __half *d_gate, __half *d_dequant, __half *d_output);

void run_routed_expert_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const ExpertEntry &entry,
    __half *d_intermediate_buf, __half *d_gate_buf,
    __half *d_dequant,
    __half *d_output,
    int hidden_dim);

// From inference_head.cu
cudaError_t run_lm_head(
    cublasHandle_t cublas_h,
    cudaStream_t stream,
    const __half *d_head_weight,
    const __half *d_hidden,
    int vocab_size,
    int hidden_dim,
    float *d_logits,
    int32_t *d_next_token);

InferenceResult RunInference(const InferenceConfig &cfg)
{
    InferenceResult result;
    auto mp = cfg.model_path;

    // 1. Load tokenizer
    ::lsxcommon::BpeTokenizer tok;
    {
        auto tpath = std::filesystem::path(mp) / "tokenizer.json";
        if (!tok.Load(tpath.string())) {
            result.error_msg = "Failed to load tokenizer from " + tpath.string();
            return result;
        }
    }

    // 2. Tokenize hardcoded prompt
    const char *prompt = "The capital of France is";
    std::string prompt_str(prompt);
    auto input_ids = tok.Encode(prompt_str);
    if (input_ids.empty()) {
        result.error_msg = "Tokenizer returned empty ids";
        return result;
    }
    std::fprintf(stderr, "[inference] prompt=%lld tokens: ", static_cast<long long>(input_ids.size()));
    for (size_t i = 0; i < input_ids.size(); ++i)
        std::fprintf(stderr, "%d ", static_cast<int>(input_ids[i]));
    std::fprintf(stderr, "\n");

    // 3. Load topology
    auto topo = load_topology(mp);
    if (topo.num_layers == 0) {
        result.error_msg = topo.error_msg.empty()
            ? "Empty topology" : topo.error_msg;
        return result;
    }
    std::fprintf(stderr, "[inference] layers=%d experts_per_layer=%d total=%lld\n",
                 topo.num_layers, topo.experts_per_layer,
                 static_cast<long long>(topo.total_experts));

    // 4. Load expert index
    auto edb = load_expert_index(mp, topo.num_layers, topo.experts_per_layer);
    if (edb.entries.empty()) {
        result.error_msg = edb.error_msg.empty()
            ? "No experts loaded" : edb.error_msg;
        return result;
    }
    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    for (auto &e : edb.entries) {
        if (e.w1_bytes > 0) {
            w1_bytes = e.w1_bytes;
            w2_bytes = e.w2_bytes;
            w3_bytes = e.w3_bytes;
            break;
        }
    }
    int intermediate_dim = w1_bytes > 0
        ? static_cast<int>((w1_bytes * 2) / kHiddenDim) : kIntermediateDim;
    intermediate_dim = std::max(intermediate_dim, 1024);

    // Execution block — all INFER_CHECK gotos jump to cleanup: below
    {

#define INFER_CHECK(call)                                              \
    do {                                                               \
        cudaError_t _e = call;                                         \
        if (_e != cudaSuccess) {                                       \
            result.error_msg = std::string(#call)                       \
                + " failed: " + cudaGetErrorString(_e);                \
            goto cleanup;                                              \
        }                                                              \
    } while (0)

#define CUBLAS_INFER_CHECK(call)                                       \
    do {                                                               \
        cublasStatus_t _s = call;                                      \
        if (_s != CUBLAS_STATUS_SUCCESS) {                             \
            result.error_msg = std::string(#call)                       \
                + " failed: code="                                     \
                + std::to_string(static_cast<int>(_s));                \
            goto cleanup;                                              \
        }                                                              \
    } while (0)

    // 5. Allocate device buffers
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_h = nullptr;
    __half *d_hidden = nullptr;
    __half *d_hidden_next = nullptr;
    __half *d_shared_int = nullptr;
    __half *d_shared_gate = nullptr;
    __half *d_routed_buf = nullptr;
    __half *d_dequant_scratch = nullptr;
    float *d_logits = nullptr;
    int32_t *d_next_token = nullptr;
    PinnedWeights pw;

    // Host-side buffers
    float *h_logits = nullptr;
    int *h_topk = nullptr;
    int max_gen = std::min(cfg.max_gen_tokens, 4096);
    int total_tokens = static_cast<int>(input_ids.size()) + max_gen;
    std::vector<int> gen_ids;
    cudaEvent_t t_start = nullptr, t_stop = nullptr;
    float rms_eps = 1e-6f;
    int total_steps = 0;
    const int kLocalTopK = 8;
    size_t shared_int_sz = static_cast<size_t>(intermediate_dim) * sizeof(__half);
    size_t h_logits_sz = static_cast<size_t>(150000) * sizeof(float);
    size_t dequant_sz = static_cast<size_t>(kHiddenDim * kIntermediateDim) * sizeof(__half);

    gen_ids.reserve(static_cast<size_t>(total_tokens));

    INFER_CHECK(cudaStreamCreate(&stream));
    CUBLAS_INFER_CHECK(cublasCreate(&cublas_h));
    CUBLAS_INFER_CHECK(cublasSetStream(cublas_h, stream));

    // Embedding buffer: store all hidden states
    INFER_CHECK(cudaMalloc(&d_hidden, static_cast<size_t>(kHiddenDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_hidden_next, static_cast<size_t>(kHiddenDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_shared_int, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_shared_gate, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_routed_buf, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_dequant_scratch, dequant_sz));
    INFER_CHECK(cudaMalloc(&d_logits, static_cast<size_t>(150000) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_next_token, sizeof(int32_t)));

    INFER_CHECK(cudaMallocHost(&h_logits, h_logits_sz));

    // 6. Load pinned weights
    INFER_CHECK(load_pinned_weights(mp, pw, stream));
    // Ensure vocab size
    if (pw.vocab_size == 0) pw.vocab_size = 128000;
    std::fprintf(stderr, "[inference] vocab_size=%d\n", pw.vocab_size);

    // 7. Route table for top-8
    INFER_CHECK(cudaMallocHost(&h_topk,
        static_cast<size_t>(kLocalTopK) * sizeof(int)));

    // 8. Embedding: first token
    {
        int first_tok = input_ids[0];
        size_t embed_offset = static_cast<size_t>(first_tok) * kHiddenDim;
        INFER_CHECK(cudaMemcpyAsync(
            d_hidden, pw.d_embed + embed_offset,
            static_cast<size_t>(kHiddenDim) * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
    }

    total_steps = static_cast<int>(input_ids.size()) + max_gen;

    // 9. Timing events
    INFER_CHECK(cudaEventCreate(&t_start));
    INFER_CHECK(cudaEventCreate(&t_stop));
    INFER_CHECK(cudaEventRecord(t_start, stream));

    // 10. Autoregressive loop
    for (int step = 0; step < total_steps; ++step) {
        int cur_tok = (step < static_cast<int>(input_ids.size()))
            ? input_ids[step] : gen_ids[static_cast<size_t>(step - input_ids.size())];

        // Embed (if not first token)
        if (step > 0) {
            size_t embed_offset = static_cast<size_t>(cur_tok) * kHiddenDim;
            INFER_CHECK(cudaMemcpyAsync(
                d_hidden, pw.d_embed + embed_offset,
                static_cast<size_t>(kHiddenDim) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream));
        }

        // Per-layer FFN
        for (int lay = 0; lay < topo.num_layers; ++lay) {
            auto &lw = pw.layers[lay];

            // RMSNorm
            if (lw.d_ffn_norm) {
                INFER_CHECK(run_rmsnorm(stream, d_hidden, lw.d_ffn_norm,
                                         d_hidden_next, kHiddenDim, rms_eps));
            } else {
                INFER_CHECK(cudaMemcpyAsync(d_hidden_next, d_hidden,
                    static_cast<size_t>(kHiddenDim) * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream));
            }

            // Router: gate_weight @ d_hidden_next
            if (lw.d_gate_weight) {
                int num_routed = std::min(kNumRouted, topo.experts_per_layer);
                INFER_CHECK(run_router(cublas_h, stream, lw.d_gate_weight,
                                        d_hidden_next, kHiddenDim, num_routed,
                                        h_topk, kLocalTopK));
            } else {
                std::memset(h_topk, 0, static_cast<size_t>(kLocalTopK) * sizeof(int));
            }

            // Shared experts
            if (lw.d_shared_w1 && lw.shared_w1_bytes > 0) {
                run_shared_experts_ffn(
                    cublas_h, stream,
                    d_hidden_next,
                    lw.d_shared_w1, lw.d_shared_w2, lw.d_shared_w3,
                    lw.d_shared_w1_scales, lw.d_shared_w2_scales, lw.d_shared_w3_scales,
                    lw.shared_w1_global_scale, lw.shared_w2_global_scale, lw.shared_w3_global_scale,
                    lw.shared_w1_bytes, lw.shared_w2_bytes, lw.shared_w3_bytes,
                    d_shared_int, d_shared_gate, d_dequant_scratch, d_hidden_next);
            }

            // Routed experts (top-k)
            if (lw.d_gate_weight) {
                for (int ki = 0; ki < kLocalTopK; ++ki) {
                    int exp_idx = h_topk[ki];
                    if (exp_idx < 0) break;
                    size_t eidx = static_cast<size_t>(lay * topo.experts_per_layer + exp_idx);
                    if (eidx < edb.entries.size() && edb.entries[eidx].w1_bytes > 0) {
                        run_routed_expert_ffn(
                            cublas_h, stream, d_hidden_next,
                            edb.entries[eidx],
                            d_shared_int, d_shared_gate, d_dequant_scratch,
                            d_routed_buf,
                            kHiddenDim);
                        // Accumulate: hidden_next = hidden_next + routed_out
                        const float acc_alpha = 1.0f, acc_beta = 1.0f;
                        CUBLAS_INFER_CHECK(cublasGemmEx(
                            cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                            kHiddenDim, 1, 1, &acc_alpha,
                            d_routed_buf, CUDA_R_16F, kHiddenDim,
                            d_hidden_next, CUDA_R_16F, kHiddenDim, &acc_beta,
                            d_hidden, CUDA_R_16F, kHiddenDim,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                        __half *tmp = d_hidden;
                        d_hidden = d_hidden_next;
                        d_hidden_next = tmp;
                    }
                }
            } else {
                // No gate: treat as identity
            }

            // hidden = hidden_next for next layer
            if (lay < topo.num_layers - 1) {
                __half *tmp = d_hidden;
                d_hidden = d_hidden_next;
                d_hidden_next = tmp;
            }
        } // layers

        // Final RMSNorm
        if (pw.d_final_norm) {
            INFER_CHECK(run_rmsnorm(stream, d_hidden, pw.d_final_norm,
                                     d_hidden_next, kHiddenDim, rms_eps));
            __half *tmp = d_hidden;
            d_hidden = d_hidden_next;
            d_hidden_next = tmp;
        }

        // LM Head
        INFER_CHECK(cudaMemsetAsync(d_logits, 0,
            static_cast<size_t>(pw.vocab_size) * sizeof(float), stream));
        INFER_CHECK(run_lm_head(cublas_h, stream,
                                 pw.d_head, d_hidden,
                                 pw.vocab_size, kHiddenDim,
                                 d_logits, d_next_token));

        // Copy token back to host
        INFER_CHECK(cudaMemcpyAsync(h_logits, d_logits,
            static_cast<size_t>(pw.vocab_size) * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        INFER_CHECK(cudaDeviceSynchronize());

        int next_tok = 0;
        float max_val = -1e10f;
        for (int v = 0; v < pw.vocab_size; ++v) {
            if (h_logits[v] > max_val) {
                max_val = h_logits[v];
                next_tok = v;
            }
        }

        // Store generated token
        if (step >= static_cast<int>(input_ids.size()))
            gen_ids.push_back(next_tok);

        std::fprintf(stderr, "[inference] step=%d tok=%d", step, next_tok);
        std::vector<int32_t> single_tok = {static_cast<int32_t>(next_tok)};
        std::string tstr = tok.Decode(single_tok);
        if (!tstr.empty())
            std::fprintf(stderr, " '%s'", tstr.c_str());
        std::fprintf(stderr, " maxv=%.4f\n", max_val);

        if (next_tok == tok.eos_id() && step >= static_cast<int>(input_ids.size()))
            break;
    }

    INFER_CHECK(cudaEventRecord(t_stop, stream));
    INFER_CHECK(cudaDeviceSynchronize());

    // Compute metrics
    {
        float elapsed_ms = 0.0f;
        INFER_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));

        int gen_count = static_cast<int>(gen_ids.size());
        std::vector<int32_t> gen_ids_i32(gen_ids.begin(), gen_ids.end());
        std::string output = tok.Decode(gen_ids_i32);
        result.ttft_ms = elapsed_ms / static_cast<double>(total_steps);
        result.tokens_per_sec = elapsed_ms > 0.0f
            ? (static_cast<double>(gen_count) / elapsed_ms) * 1000.0
            : 0.0;
        result.total_time_ms = static_cast<double>(elapsed_ms);
        result.num_tokens = gen_count;
        result.num_layers = topo.num_layers;
        result.output_text = output;
        result.ok = true;

        std::fprintf(stderr, "\n=== --testprompt Results ===\n");
        std::fprintf(stderr, "  prompt:        \"%s\"\n", prompt);
        std::fprintf(stderr, "  output:        \"%s\"\n", output.c_str());
        std::fprintf(stderr, "  elapsed:       %.1f ms\n", static_cast<double>(elapsed_ms));
        std::fprintf(stderr, "  gen_tokens:    %d\n", gen_count);
        std::fprintf(stderr, "  TTFT:          %.2f ms\n", result.ttft_ms);
        std::fprintf(stderr, "  tokens/s:      %.1f\n", result.tokens_per_sec);
    }

cleanup:
    if (stream)        cudaStreamDestroy(stream);
    if (cublas_h)      cublasDestroy(cublas_h);
    if (d_hidden)      cudaFree(d_hidden);
    if (d_hidden_next) cudaFree(d_hidden_next);
    if (d_shared_int)  cudaFree(d_shared_int);
    if (d_shared_gate) cudaFree(d_shared_gate);
    if (d_routed_buf)  cudaFree(d_routed_buf);
    if (d_dequant_scratch) cudaFree(d_dequant_scratch);
    if (d_logits)      cudaFree(d_logits);
    if (d_next_token)  cudaFree(d_next_token);
    if (h_logits)      cudaFreeHost(h_logits);
    if (h_topk)        cudaFreeHost(h_topk);
    if (t_start)       cudaEventDestroy(t_start);
    if (t_stop)        cudaEventDestroy(t_stop);
    pw.Free();

    return result;
    } // end execution block
} // end RunInference

} // namespace gpu
} // namespace lsxcommon
