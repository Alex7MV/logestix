#include "lsxcommon/lsx_weight_gemm.h"

namespace lsxcommon {

cudaError_t RunWeightGemm(cudaStream_t stream, const __half *x,
                          const lsx_nvfp4::LsxNvfp4Weight &nv,
                          const gemv::DenseWeight &dense, __half *y,
                          const WeightGemmShape &shape, bool acc)
{
    if (x == nullptr || y == nullptr || shape.seq <= 0 || shape.k <= 0 ||
        shape.n <= 0)
        return cudaErrorInvalidValue;

    // NVFP4 first: a tree holds one encoding, and a tree that holds NVFP4 must
    // keep the exact call it made before this function existed (the existing
    // artifacts' gate results are the proof of that path).
    if (nv.d_packed != nullptr && nv.d_scales != nullptr) {
        // The record's own shape is authoritative when it disagrees with the
        // caller: a plane uploaded for [N,K] cannot be run as something else, and
        // silently trusting `shape` is how a transposed read reaches the kernel.
        const int k = nv.cols > 0 ? nv.cols : shape.k;
        const int n = nv.rows > 0 ? nv.rows : shape.n;
        if (k != shape.k || n != shape.n) return cudaErrorInvalidValue;
        return lsx_nvfp4::RunLsxNvfp4Gemm(stream, x, nv.d_packed,
                                          nv.d_scales, nv.global_scale, y,
                                          shape.seq, k, n, acc);
    }

    if (dense.valid()) {
        if (dense.cols != shape.k || dense.rows != shape.n)
            return cudaErrorInvalidValue;
        // gemv computes y = alpha*sum + beta*y, so acc folds into beta. alpha=1
        // and beta∈{0,1} reproduce the NVFP4 contract's two modes exactly.
        return gemv::RunLxGemmVec(stream, dense.d_data, x, shape.seq, dense.rows,
                                  dense.cols, dense.dtype, gemv::LxDtype::kF16,
                                  1.0f, y, gemv::LxDtype::kF16, acc ? 1.0f : 0.0f);
    }

    // Neither encoding: the caller's layer is incomplete. NotSupported is what
    // the NVFP4 dispatcher already answers for a shape it cannot serve, and every
    // caller in the tree treats it as "fall back / fail", never as success.
    return cudaErrorNotSupported;
}

} // namespace lsxcommon
