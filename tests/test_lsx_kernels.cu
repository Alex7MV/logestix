// TDD harness for the shared lsx_kernels_*.cuh library (Phase A unification).
// Each cluster task adds red/green equivalence tests here: a verbatim
// reference copy of the kernel being replaced vs. the unified template.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

__host__ inline void RequireCudaOk(cudaError_t e) {
    REQUIRE(e == cudaSuccess);
}

template <typename T>
std::vector<T> ToHost(const T* dev, size_t n) {
    std::vector<T> h(n);
    RequireCudaOk(
        cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

template <typename T>
void RequireDeviceEqual(const T* a, const T* b, size_t n) {
    auto ha = ToHost(a, n);
    auto hb = ToHost(b, n);
    REQUIRE(std::memcmp(ha.data(), hb.data(), n * sizeof(T)) == 0);
}

// Deterministic LCG fill so failures are reproducible.
inline uint32_t LcgNext(uint32_t& s) { return s = s * 1664525u + 1013904223u; }

} // namespace

TEST_CASE("harness smoke", "[lsxk]") {
    int* d = nullptr;
    RequireCudaOk(cudaMalloc(&d, 4));
    RequireCudaOk(cudaMemset(d, 0, 4));
    cudaFree(d);
    REQUIRE(true);
}
