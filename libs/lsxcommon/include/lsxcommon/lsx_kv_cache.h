// Model-neutral KV cache geometry.
//
// The index is an affine 3-stride form, deliberately NOT an enum-tagged union:
//
//   offset = layer*LayerStride() + head*head_stride + pos*pos_stride + dim
//
// So every permutation of (head, pos) is a choice of two numbers rather than a
// branch or a template instantiation. Both layouts this repo cares about are the
// same three values permuted:
//
//   pos-major   pos_stride = kv_heads*head_dim   head_stride = head_dim
//   head-major  pos_stride = head_dim            head_stride = capacity*head_dim
//
// Two invariants make switching layout size-neutral, and are pinned by
// tests/test_kv_cache_layout.cpp:
//
//   LayerStride() == kv_heads * capacity * head_dim   (layout-independent)
//   Elements()    == layers * LayerStride()          (layout-independent)
//
// Only the decomposition of a layer's block into head and position changes, never
// its size - so the VRAM planner and the resource governor are untouched by a
// layout switch.
//
// All arithmetic is int64_t: kv_heads*capacity*head_dim reaches 8.05G elements on
// Xing-4.0 at capacity 32768, twice past int32, and a silently wrapped index
// yields plausible-looking data.
//
// Includes only <cstdint> so it is safe to include from .cu/.cuh.
#pragma once

#include <cstdint>

// The geometry helpers below are __host__ __device__ under nvcc so ONE function
// serves both execution spaces; the guard keeps this header includable from
// plain host C++ (it is pulled in by gemma_dense_load.h / model.h too).
#ifdef __CUDACC__
#define LSX_KV_CACHE_HD __host__ __device__
#else
#define LSX_KV_CACHE_HD
#endif

namespace lsxcommon
{

enum class KvCacheLayout
{
    kHeadMajor,  // [head][pos][dim] - the per-head working set is contiguous
    kPosMajor,   // [pos][head][dim]
};

struct KvCacheGeometry
{
    int64_t capacity = 0;   // max cached positions per layer
    int64_t kv_heads = 0;   // heads storing K/V (== q heads for MHA)
    int64_t head_dim = 0;   // stored elements per head (MLA: kv_lora + rope)
    int64_t layers = 0;
    int64_t pos_stride = 0;
    int64_t head_stride = 0;

    LSX_KV_CACHE_HD static constexpr KvCacheGeometry Make(
        KvCacheLayout layout, int64_t capacity, int64_t kv_heads,
        int64_t head_dim, int64_t layers)
    {
        KvCacheGeometry g;
        g.capacity = capacity;
        g.kv_heads = kv_heads;
        g.head_dim = head_dim;
        g.layers = layers;
        if (layout == KvCacheLayout::kHeadMajor)
        {
            g.pos_stride = head_dim;
            g.head_stride = capacity * head_dim;
        }
        else
        {
            g.pos_stride = kv_heads * head_dim;
            g.head_stride = head_dim;
        }
        return g;
    }

    // __host__ __device__ constexpr: ONE function compiled for both host and
    // device, so the two cannot drift apart. The annotation must be explicit:
    // a host-only constexpr leans on --expt-relaxed-constexpr for device
    // callers, which device-debug nvcc miscompiles ("Invalid instruction with
    // no BB"), and dropping the flag instead breaks the call outright.
    LSX_KV_CACHE_HD constexpr int64_t LayerStride() const
    {
        return kv_heads * capacity * head_dim;
    }

    LSX_KV_CACHE_HD constexpr int64_t Elements() const
    {
        return layers * LayerStride();
    }

    LSX_KV_CACHE_HD constexpr int64_t Offset(int64_t layer, int64_t head,
                                             int64_t pos, int64_t dim = 0) const
    {
        return layer * LayerStride() + head * head_stride + pos * pos_stride +
               dim;
    }
};

}  // namespace lsxcommon

#undef LSX_KV_CACHE_HD
