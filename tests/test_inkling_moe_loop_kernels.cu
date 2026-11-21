// Inkling-2-Mini kernel + session tests (CUDA TU, -fmad=false for bit-exact
// parity against the host references).
//
// Task 6 cases pin the session plumbing: per-pass KV slots survive growth
// (the EnsureCapacity over-read trap), Reset zeroes every conv slot, the
// geometry is pos-major with CacheLayers() slots. Task 7 adds the numeric
// kernel cases (MessageGate, message mix, and the inkling2 arithmetic at Mini
// geometry).
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/inkling_moe_loop_load.h"

using lsxcommon::inkling::InklingLoopScratch;
using lsxcommon::inkling::InklingLoopSession;
using lsxcommon::inkling::InklingLoopTopology;

namespace {

// Synthetic topology matching the converter fixture's shape (8 layers, loop
// [2,6) x 2 passes -> 12 cache slots).
InklingLoopTopology MakeTop()
{
    InklingLoopTopology top;
    top.num_layers = 8;
    top.hidden = 64;
    top.vocab = 256;
    top.unpadded_vocab = 200;
    top.num_heads = 4;
    top.num_kv_heads = 2;
    top.head_dim = 16;
    top.d_rel = 4;
    top.rel_extent = 8;
    top.sliding_window = 8;
    top.conv_kernel = 4;
    top.attn_res_block = 2;
    top.num_experts = 4;
    top.top_k = 2;
    top.moe_intermediate = 32;
    top.loop_start = 2;
    top.loop_end = 6;
    top.loop_passes = 2;
    top.loop_gate_rank = 8;
    top.full_layers = {5};
    top.sliding_layers = {0, 1, 2, 3, 4, 6, 7};
    top.expert_nvfp4 = {true, true, true, true, true, true, false, false};
    top.expert_fp8 = {false, false, false, false, false, false, true, true};
    return top;
}

}  // namespace

TEST_CASE("inkling loop session growth keeps every slot's rows intact",
          "[inkling][kernels]")
{
    const InklingLoopTopology top = MakeTop();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    InklingLoopSession session;
    REQUIRE(session.EnsureCapacity(top, 4, stream) == cudaSuccess);
    REQUIRE(session.capacity >= 4);
    REQUIRE(session.geom.layers == top.CacheLayers());
    REQUIRE(session.geom.layers == 12);
    // Pos-major: pos_stride = kv_heads*head_dim, head_stride = head_dim.
    REQUIRE(session.geom.pos_stride == top.num_kv_heads * top.head_dim);
    REQUIRE(session.geom.head_stride == top.head_dim);

    // Write a distinct pattern into every cache slot's first 3 positions.
    // Values stay below 2048 so every integer is exact in FP16 (a larger
    // pattern would round and the equality check would test the codec, not
    // the copy).
    const int64_t rows = 3;
    const int64_t row_elems = session.geom.pos_stride;
    std::vector<__half> pattern(static_cast<size_t>(rows * row_elems));
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        for (int64_t i = 0; i < rows * row_elems; ++i)
            pattern[static_cast<size_t>(i)] =
                __float2half(static_cast<float>(slot * 100 + i + 1));
        REQUIRE(cudaMemcpyAsync(
                    session.d_k + slot * session.geom.LayerStride(),
                    pattern.data(), pattern.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(
                    session.d_v + slot * session.geom.LayerStride(),
                    pattern.data(), pattern.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream) == cudaSuccess);
    }
    session.kv_rows = static_cast<int>(rows);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Grow: the copy must move exactly kv_rows * pos_stride per slot — not
    // kv_rows * LayerStride(), which over-reads the old allocation (the Small
    // defect that surfaced as cudaErrorInvalidValue on the NEXT request).
    REQUIRE(session.EnsureCapacity(top, 9, stream) == cudaSuccess);
    REQUIRE(session.capacity >= 9);
    REQUIRE(session.kv_rows == rows);

    std::vector<__half> back(pattern.size());
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        // Rebuild THIS slot's expected pattern (the write loop above left
        // `pattern` holding the last slot's values).
        for (int64_t i = 0; i < rows * row_elems; ++i)
            pattern[static_cast<size_t>(i)] =
                __float2half(static_cast<float>(slot * 100 + i + 1));
        REQUIRE(cudaMemcpy(back.data(),
                           session.d_k + slot * session.geom.LayerStride(),
                           back.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < pattern.size(); ++i) {
            INFO("slot " << slot << " k elem " << i);
            REQUIRE(__half2float(back[i]) == __half2float(pattern[i]));
        }
        REQUIRE(cudaMemcpy(back.data(),
                           session.d_v + slot * session.geom.LayerStride(),
                           back.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < pattern.size(); ++i) {
            INFO("slot " << slot << " v elem " << i);
            REQUIRE(__half2float(back[i]) == __half2float(pattern[i]));
        }
    }
    session.Free();
    cudaStreamDestroy(stream);
}

TEST_CASE("inkling loop session Reset zeroes every conv slot",
          "[inkling][kernels]")
{
    const InklingLoopTopology top = MakeTop();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    InklingLoopSession session;
    REQUIRE(session.EnsureCapacity(top, 4, stream) == cudaSuccess);
    REQUIRE(session.conv_state != nullptr);
    const int64_t kv_dim = top.num_kv_heads * top.head_dim;
    const int64_t want_floats =
        2 * top.conv_kernel * kv_dim + 2 * top.conv_kernel * top.hidden;
    REQUIRE(session.conv_slot_floats == want_floats);
    REQUIRE(session.conv_state_bytes ==
            top.CacheLayers() * want_floats * static_cast<int64_t>(sizeof(float)));

    // Dirty every slot, then Reset: all 12 rings must come back zero.
    std::vector<float> dirty(static_cast<size_t>(want_floats), 7.0f);
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        REQUIRE(cudaMemcpyAsync(session.ConvSlot(slot), dirty.data(),
                                dirty.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream) == cudaSuccess);
    }
    session.kv_rows = 3;
    session.last_tok = 42;
    session.gen_tail = {1, 2, 3};
    session.Reset(stream);
    REQUIRE(session.kv_rows == 0);
    REQUIRE(session.last_tok == -1);
    REQUIRE(session.gen_tail.empty());
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> back(static_cast<size_t>(want_floats));
    for (int slot = 0; slot < top.CacheLayers(); ++slot) {
        REQUIRE(cudaMemcpy(back.data(), session.ConvSlot(slot),
                           back.size() * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < back.size(); ++i) {
            INFO("slot " << slot << " float " << i);
            REQUIRE(back[i] == 0.0f);
        }
    }
    session.Free();
    REQUIRE(session.conv_state == nullptr);
    cudaStreamDestroy(stream);
}

TEST_CASE("inkling loop scratch allocates the loop machinery",
          "[inkling][kernels]")
{
    const InklingLoopTopology top = MakeTop();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    InklingLoopScratch scratch;
    REQUIRE(scratch.Allocate(top, 16, stream));
    REQUIRE(scratch.arena != nullptr);
    REQUIRE(scratch.arena_bytes > 0);
    // Loop buffers exist and are distinct.
    REQUIRE(scratch.pre_loop_hidden != nullptr);
    REQUIRE(scratch.pre_loop_stack != nullptr);
    REQUIRE(scratch.prev_stack != nullptr);
    REQUIRE(scratch.prev_hidden != nullptr);
    REQUIRE(scratch.msg != nullptr);
    REQUIRE(scratch.gate_h != nullptr);
    REQUIRE(scratch.gate_rank != nullptr);
    REQUIRE(scratch.gate_out != nullptr);
    REQUIRE(scratch.pre_loop_hidden != scratch.prev_hidden);
    REQUIRE(scratch.msg != scratch.gate_out);
    // The expert staging slabs are sized for top_k experts of both codecs.
    REQUIRE(scratch.expert_stage != nullptr);
    REQUIRE(scratch.fp8_stage != nullptr);
    REQUIRE(scratch.expert_stage_bytes > 0);
    REQUIRE(scratch.fp8_stage_bytes > 0);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    scratch.Free();
    REQUIRE(scratch.arena == nullptr);
    REQUIRE(scratch.expert_stage == nullptr);
    REQUIRE(scratch.fp8_stage == nullptr);
    cudaStreamDestroy(stream);
}