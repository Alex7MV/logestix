#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <thread>

#include "lsxcommon/cuda_event_wait.h"

TEST_CASE("WaitEventChecked returns on a recorded event", "[fast]")
{
    cudaStream_t s = nullptr;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreate(&ev) == cudaSuccess);
    REQUIRE(cudaEventRecord(ev, s) == cudaSuccess);
    CHECK(lsxcommon::cuda::WaitEventChecked(ev, 5000) == cudaSuccess);
    cudaEventDestroy(ev);
    cudaStreamDestroy(s);
}

namespace
{
void SleepOnStream(void*)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
}  // namespace

TEST_CASE("WaitEventChecked times out while the recorded work is pending",
          "[fast]")
{
    cudaStream_t s = nullptr;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreate(&ev) == cudaSuccess);
    // The host callback blocks the stream, so the event recorded behind it
    // stays incomplete past the deadline.
    REQUIRE(cudaLaunchHostFunc(s, &SleepOnStream, nullptr) == cudaSuccess);
    REQUIRE(cudaEventRecord(ev, s) == cudaSuccess);
    CHECK(lsxcommon::cuda::WaitEventChecked(ev, 5) ==
          cudaErrorLaunchTimeout);
    CHECK(lsxcommon::cuda::WaitEventChecked(ev, 5000) == cudaSuccess);
    cudaEventDestroy(ev);
    cudaStreamDestroy(s);
}

TEST_CASE("WaitEventChecked treats a never-recorded event as complete", "[fast]")
{
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreate(&ev) == cudaSuccess);
    CHECK(lsxcommon::cuda::WaitEventChecked(ev, 5) == cudaSuccess);
    cudaEventDestroy(ev);
}
