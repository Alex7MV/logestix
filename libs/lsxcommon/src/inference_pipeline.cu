#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/norm_kernels.h"
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

//------------------------------------------------------------------------------
// Half ↔ float helpers — bridge between the half-native pipeline and the
// float-based rmsnorm_float kernel.
//------------------------------------------------------------------------------
static __global__ void half_to_float_kernel(
    const __half* __restrict__ src, float* __restrict__ dst, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static __global__ void float_to_half_kernel(
    const float* __restrict__ src, __half* __restrict__ dst, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static __host__ cudaError_t run_rmsnorm(
    cudaStream_t stream, const __half* d_input, const __half* d_weight,
    __half* d_output, float* d_scratch, float* d_gamma_scratch, int dim, float eps)
{
    int n = dim;
    int block = 256;
    int grid = (n + block - 1) / block;

    half_to_float_kernel<<<grid, block, 0, stream>>>(d_input, d_scratch, n);
    half_to_float_kernel<<<grid, block, 0, stream>>>(d_weight, d_gamma_scratch, n);

    cudaError_t err = rmsnorm_float(stream, 1, dim, d_scratch,
                                     d_gamma_scratch, eps, d_scratch);
    if (err != cudaSuccess) return err;

    float_to_half_kernel<<<grid, block, 0, stream>>>(d_scratch, d_output, n);
    return cudaGetLastError();
}

// From inference_moe.cu
cudaError_t run_router(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    int *h_topk_indices, int topk);

cudaError_t run_shared_experts_ffn(
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

    // Load pinned weights FIRST (before expert index, to avoid fd exhaustion)
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

    // 4. Allocate device buffers
    int max_gen = std::min(cfg.max_gen_tokens, 4096);
    int total_tokens = static_cast<int>(input_ids.size()) + max_gen;
    std::vector<int> gen_ids;
    gen_ids.reserve(static_cast<size_t>(total_tokens));

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_h = nullptr;
    __half *d_hidden = nullptr;
    __half *d_hidden_next = nullptr;
    __half *d_shared_int = nullptr;
    __half *d_shared_gate = nullptr;
    __half *d_routed_buf = nullptr;
    __half *d_dequant_scratch = nullptr;
    float *d_logits = nullptr;
    float *d_norm_scratch = nullptr;
    float *d_norm_gamma_scratch = nullptr;
    int32_t *d_next_token = nullptr;
    PinnedWeights pw;
    float *h_logits = nullptr;
    int *h_topk = nullptr;
    cudaEvent_t t_start = nullptr, t_stop = nullptr;
    float rms_eps = 1e-6f;
    int total_steps = 0;
    const int kLocalTopK = 8;
    size_t h_logits_sz = static_cast<size_t>(150000) * sizeof(float);
    // Shared experts may have intermediate_dim = hidden_dim; allocate full size.
    size_t dequant_sz = static_cast<size_t>(kHiddenDim * kHiddenDim) * sizeof(__half);
    // Shared experts may have intermediate_dim up to hidden_dim
    size_t shared_int_sz = static_cast<size_t>(kHiddenDim) * sizeof(__half);

    INFER_CHECK(cudaStreamCreate(&stream));
    CUBLAS_INFER_CHECK(cublasCreate(&cublas_h));
    CUBLAS_INFER_CHECK(cublasSetStream(cublas_h, stream));

    INFER_CHECK(cudaMalloc(&d_hidden, static_cast<size_t>(kHiddenDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_hidden_next, static_cast<size_t>(kHiddenDim) * sizeof(__half)));

    INFER_CHECK(cudaMalloc(&d_shared_int, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_shared_gate, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_routed_buf, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_dequant_scratch, dequant_sz));
    INFER_CHECK(cudaMalloc(&d_next_token, sizeof(int32_t)));
    INFER_CHECK(cudaMalloc(&d_norm_scratch,
        static_cast<size_t>(kHiddenDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_norm_gamma_scratch,
        static_cast<size_t>(kHiddenDim) * sizeof(float)));

    // 5. Load pinned weights
    {
        auto perr = load_pinned_weights(mp, pw, stream);
        if (perr != cudaSuccess) {
            result.error_msg = "load_pinned_weights failed";
            goto cleanup;
        }
        INFER_CHECK(cudaDeviceSynchronize());
    }

    // 6. Expert index: skipped in Phase 1 (routed experts stubbed).
    // Avoids opening 11008 mmap files (fd exhaustion).
    // Hardcoded intermediate_dim = kIntermediateDim = 2048.
    // Ensure vocab size
    if (pw.vocab_size == 0) pw.vocab_size = 128000;
    std::fprintf(stderr, "[inference] vocab_size=%d\n", pw.vocab_size);

    // Allocate logits AFTER load (pre-alloc buffers can be corrupted by loading)
    INFER_CHECK(cudaMalloc(&d_logits, static_cast<size_t>(150000) * sizeof(float)));
    INFER_CHECK(cudaMallocHost(&h_logits, h_logits_sz));

    // 7. Route table for top-8
    INFER_CHECK(cudaMallocHost(&h_topk,
        static_cast<size_t>(kLocalTopK) * sizeof(int)));

    // 8. Embedding: first token
    {
        int first_tok = input_ids[0];
        size_t embed_offset = static_cast<size_t>(first_tok) * kHiddenDim;
        cudaError_t embed_err = cudaMemcpyAsync(
            d_hidden, pw.d_embed + embed_offset,
            static_cast<size_t>(kHiddenDim) * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
        if (embed_err != cudaSuccess) {
            std::fprintf(stderr, "[dbg] embed copy failed: %s\n",
                         cudaGetErrorString(embed_err));
        }
        INFER_CHECK(cudaGetLastError());  // catch any pending
        // Sync after embed copy to catch async errors
        INFER_CHECK(cudaStreamSynchronize(stream));
        std::fprintf(stderr, "[dbg] embed sync OK\n");
        (void)embed_err;
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
            if (lay == 0) {
                std::fprintf(stderr, "[dbg] layer 0: ffn_norm=%p gate=%p sh_w1=%p (bytes=%lld)\n",
                             (void*)lw.d_ffn_norm, (void*)lw.d_gate_weight,
                             (void*)lw.d_shared_w1, static_cast<long long>(lw.shared_w1_bytes));
            }

            // RMSNorm
            if (lw.d_ffn_norm) {
                INFER_CHECK(run_rmsnorm(stream, d_hidden, lw.d_ffn_norm,
                                         d_hidden_next, d_norm_scratch,
                                         d_norm_gamma_scratch,
                                         kHiddenDim, rms_eps));
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

            // Shared experts FFN: NVFP4-packed from shared_experts.arrow
            if (lw.d_shared_w1 && lw.shared_w1_bytes > 0) {
                INFER_CHECK(run_shared_experts_ffn(
                    cublas_h, stream, d_hidden_next,
                    lw.d_shared_w1, lw.d_shared_w2, lw.d_shared_w3,
                    lw.d_shared_w1_scales, lw.d_shared_w2_scales, lw.d_shared_w3_scales,
                    lw.shared_w1_global_scale, lw.shared_w2_global_scale,
                    lw.shared_w3_global_scale,
                    lw.shared_w1_bytes, lw.shared_w2_bytes, lw.shared_w3_bytes,
                    d_shared_int, d_shared_gate, d_dequant_scratch, d_hidden_next));
            }

            // Routed experts: skipped in Phase 1 (stub)
            // Phase 2 will load from ExpertDb mmap + DMA ping-pong

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
                                     d_hidden_next, d_norm_scratch,
                                     d_norm_gamma_scratch,
                                     kHiddenDim, rms_eps));
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

        // Store generated token (always — gen_ids[0] used as first gen-step input)
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

        int prompt_len = static_cast<int>(input_ids.size());
        int gen_count = static_cast<int>(gen_ids.size()) - prompt_len;
        // Skip prompt tokens for output decoding
        auto gen_start = gen_ids.begin() + std::min(prompt_len, static_cast<int>(gen_ids.size()));
        std::vector<int32_t> gen_ids_i32(gen_start, gen_ids.end());
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
    if (d_norm_scratch) cudaFree(d_norm_scratch);
    if (d_norm_gamma_scratch) cudaFree(d_norm_gamma_scratch);
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
