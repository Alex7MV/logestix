#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/qwen_inference.h"  // QwenNvfp4Weight

namespace lsxcommon::qwen_moe_exp {

// GatedResidual hyper-connection: mixed[H] + injection weights[hc] per token.
// d_hyper_input: [seq, hc*H] __half. d_hc_norm: BF16 bit-cast weight [hc*H]
// (1 + w centering applied on device). d_normed_scratch: [seq, hc*H] __half
// workspace (caller-owned). Math: fp32 intermediates, -fmad=false contract.
cudaError_t RunExpHcGatedResidual(
    cudaStream_t stream, const __half *d_hyper_input,
    const qwen::QwenNvfp4Weight &down, const qwen::QwenNvfp4Weight &up,
    const uint8_t *d_hc_norm, int seq, int H, int hc, int lowrank,
    float *d_mixed, float *d_inject_w, __half *d_normed_scratch);

// Test-only raw-float twin (identical op sequence, no NVFP4 decode, no BF16
// norm cast) for bit-parity against the host replica in
// qwen_moe_exp_hostref.h. down/up/inject passed as raw fp32 row-major.
cudaError_t RunExpHcGatedResidualF32(
    cudaStream_t stream, const float *d_hyper_input, const float *d_down,
    const float *d_up, const float *d_inject_w, int seq, int H, int hc,
    int lowrank, float *d_mixed, float *d_inject_w_out);

// hyper_input[c*H+i] += branch_out[i] * inject_w[c] for each stream c.
cudaError_t RunExpHcInjectScatter(cudaStream_t stream,
                                  const __half *d_hyper_input,
                                  const __half *d_branch_out,
                                  const float *d_inject_w, int seq, int H,
                                  int hc, __half *d_out);

// Final global mixer (GatedResidual without block_inject) — out [seq, H].
cudaError_t RunExpHcMixer(cudaStream_t stream, const __half *d_hidden,
                          const qwen::QwenNvfp4Weight &down,
                          const qwen::QwenNvfp4Weight &up,
                          const uint8_t *d_mixer_norm, int seq, int H, int hc,
                          int lowrank, __half *d_out, __half *d_normed_scratch);

} // namespace lsxcommon::qwen_moe_exp
