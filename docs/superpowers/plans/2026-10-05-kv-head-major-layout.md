# KV Head-Major Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-neutral KV-cache geometry layer and switch Xing4.0's decode KV cache from `[pos][head][dim]` to `[head][pos][dim]`, proven bit-exact and measured faster.

**Architecture:** The layer stores three strides, not a tagged layout, so the index is one add and any permutation is a choice of numbers. Both layouts therefore coexist in one binary with no template duplication, which is what makes a byte-identical parity test possible. `LayerStride()` and `Elements()` are layout-invariant, so switching is size-neutral and the VRAM planner is untouched.

**Tech Stack:** C++20, CUDA 13.4 (sm_120), Catch2 v3.16.0, CMake presets (`linux-debug`, `linux-release-znver4`).

## Global Constraints

- Target architecture is Blackwell `cc12x` ONLY. Forbidden: sm_90, sm_80, sm_89.
- CUDA `.cu` files compile in their own OBJECT libraries; host flags must not leak into NVCC.
- Headers reachable from `.cu`/`.cuh` must NOT include `<absl/container/*>` (`test_absl_container_guard`, label `fast`).
- Logging only via spdlog. No `std::cout`/`printf`.
- All index arithmetic is `int64_t`. `kv_heads * capacity * head_dim` reaches 8.05G elements on Xing at capacity 32768 — twice past int32.
- NO comments in code unless the surrounding code carries an explanatory comment for a non-obvious invariant (this codebase does; match it).
- New files not family-prefixed (`lsx_*`) are exempt from `test_code_naming_guard`.
- GPU exclusivity: only one GPU workload at a time; run perf measurements last.
- Model root: `/mnt/llm_models/models/Xing4.0-29B-A4B` (the CLI takes the ROOT).
- Every task's regression gate is `ctest --preset linux-debug -L fast` green (87/87) unless stated otherwise.

## File Structure

| File | Responsibility |
|---|---|
| `libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h` (create) | `KvCacheLayout` enum + `KvCacheGeometry` (strides, `Offset`, `LayerStride`, `Elements`, `Make`). Header-only, includes only `<cstdint>`. |
| `libs/lsxcommon/include/lsxcommon/model.h` (modify) | `ModelInitConfig::kv_cache_layout` field; `IModel::KvLayout()` accessor. |
| `libs/lsxcommon/src/xing_moe_inference.h` (modify) | `XingSession` gains `kgeom`/`vgeom` members; `AllocateXingSession` takes the layout. |
| `libs/lsxcommon/src/xing_moe_inference.cu` (modify) | Build geometry, pass to launches, write path, readback path, debug dump. |
| `libs/lsxcommon/src/xing_moe_kernels.h` (modify) | Launch signatures take geometry. |
| `libs/lsxcommon/src/xing_moe_kernels.cu` (modify) | 6 read/write sites use `Offset`. |
| `libs/lsxcommon/src/xing_moe_model.cpp` (modify) | Pass `cfg.kv_cache_layout` through; implement `KvLayout()`. |
| `libs/lsxcommon/src/nemotron_moe_model.cpp` (modify) | `KvLayout()` returns `kHeadMajor`. |
| `tests/test_kv_cache_layout.cpp` (create) | L1: shared-layer unit tests. |
| `tests/test_xing_kv_layout_parity.cpp` (create) | L2: byte-identical parity across layouts. |
| `tests/test_xing_kv_layout_perf.cu` (create) | L3: isolated KV-read bandwidth budget. |
| `tests/test_model_independence_guard.cpp` (modify) | Guard: families index KV via `KvCacheGeometry`; 2 exemptions. |
| `.opencode/skills/ingesting-a-model/SKILL.md` (modify) | Step 5 KV-layout rule. |
| `tests/perf/kernel_budgets.json` (modify) | New `xing.kv_read.decode_gbps` min budget. |
| `tests/perf/perf_baselines.json` (modify) | E2E 2048 t/s after the change. |

---

### Task 1: The shared geometry layer

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h`
- Create: `tests/test_kv_cache_layout.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  ```cpp
  namespace lsxcommon {
  enum class KvCacheLayout { kHeadMajor, kPosMajor };
  struct KvCacheGeometry {
      int64_t capacity, kv_heads, head_dim, layers, pos_stride, head_stride;
      static constexpr KvCacheGeometry Make(KvCacheLayout, int64_t capacity,
                                            int64_t kv_heads, int64_t head_dim,
                                            int64_t layers);
      __host__ __device__ int64_t LayerStride() const;
      __host__ __device__ int64_t Elements() const;
      __host__ __device__ int64_t Offset(int64_t layer, int64_t head,
                                         int64_t pos, int64_t dim = 0) const;
  };
  }
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_cache_layout.cpp`:

```cpp
// L1: the shared KV-cache geometry layer.
#include <catch2/catch_test_macros.hpp>
#include <set>

#include "lsxcommon/lsx_kv_cache.h"

using lsxcommon::KvCacheGeometry;
using lsxcommon::KvCacheLayout;

TEST_CASE("Make produces the documented strides", "[kv_layout][fast]")
{
    // Xing's real shape: 32 heads, K row 192 (nope 128 + rope 64), V row 128.
    const auto hp =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 192, 40);
    CHECK(hp.pos_stride == 192);
    CHECK(hp.head_stride == 64 * 192);

    const auto pp =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, 64, 32, 192, 40);
    CHECK(pp.pos_stride == 32 * 192);
    CHECK(pp.head_stride == 192);
}

TEST_CASE("LayerStride and Elements are layout-invariant", "[kv_layout][fast]")
{
    // The invariant that makes the swap size-neutral: only the decomposition of a
    // layer's block into head and position changes, never its size.
    const auto hp =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 128, 32, 192, 40);
    const auto pp =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, 128, 32, 192, 40);
    CHECK(hp.LayerStride() == pp.LayerStride());
    CHECK(hp.Elements() == pp.Elements());
    CHECK(hp.LayerStride() == 32 * 128 * 192);
    CHECK(hp.Elements() == 40 * 32 * 128 * 192);
}

TEST_CASE("layouts are address bijections over head x position",
          "[kv_layout][fast]")
{
    // The strongest available property: both layouts cover the same set of
    // offsets. No position is aliased or skipped - which is what licenses the
    // bit-exact parity claim.
    const int64_t cap = 16, heads = 4, dim = 3, layers = 2;
    const auto hp =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, cap, heads, dim, layers);
    const auto pp =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, cap, heads, dim, layers);

    std::set<int64_t> hs, ps;
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t p = 0; p < cap; ++p)
                for (int64_t d = 0; d < dim; ++d) {
                    hs.insert(hp.Offset(l, h, p, d));
                    ps.insert(pp.Offset(l, h, p, d));
                }
    CHECK(hs.size() == static_cast<size_t>(layers * heads * cap * dim));
    CHECK(hs == ps);
}

TEST_CASE("K and V of different widths do not overlap", "[kv_layout][fast]")
{
    // Xing: kQkHeadDim = 192, kVHeadDim = 128. Two independent geometries.
    const auto k =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 192, 40);
    const auto v =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 128, 40);
    CHECK(k.Elements() != v.Elements());
    CHECK(k.Offset(0, 0, 0) == 0);
    CHECK(v.Offset(0, 0, 0) == 0);
    // The last K element must stay inside K, not reach into V's range.
    CHECK(k.Offset(39, 31, 63, 191) == k.Elements() - 1);
    CHECK(v.Offset(39, 31, 63, 127) == v.Elements() - 1);
}

TEST_CASE("MLA latent stays contiguous within a head", "[kv_layout][fast]")
{
    // kv_lora 512 + rope 64 in one stored row: no interleaving.
    const int64_t lora = 512, rope = 64;
    const auto g = KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 32, 8,
                                         lora + rope, 4);
    const int64_t base = g.Offset(0, 3, 7, 0);
    for (int64_t d = 0; d < lora + rope; ++d)
        CHECK(g.Offset(0, 3, 7, d) == base + d);
    CHECK(g.Offset(0, 3, 8, 0) == base + lora + rope);
}

TEST_CASE("degenerate geometry yields zero elements", "[kv_layout][fast]")
{
    const auto z = KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 0, 32, 192, 40);
    CHECK(z.Elements() == 0);
    const auto h = KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 0, 192, 40);
    CHECK(h.Elements() == 0);
}

TEST_CASE("offsets are int64-wide", "[kv_layout][fast]")
{
    // 32 * 32768 * 192 * 40 = 8.05G elements: twice past int32. A wrapped index
    // under head-major yields plausible-looking data, so the width is pinned.
    const auto g =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 32768, 32, 192, 40);
    CHECK(g.Elements() == 8053063680LL);
    CHECK(g.Offset(39, 31, 32767, 191) == g.Elements() - 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --preset linux-debug && cmake --build --preset linux-debug --target test_kv_cache_layout
```
Expected: configure/build error — `lsx_kv_cache.h` does not exist.

- [ ] **Step 3: Write the header**

Create `libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h`:

```cpp
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

    static constexpr KvCacheGeometry Make(KvCacheLayout layout, int64_t capacity,
                                          int64_t kv_heads, int64_t head_dim,
                                          int64_t layers)
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

    __host__ __device__ int64_t LayerStride() const
    {
        return kv_heads * capacity * head_dim;
    }

    __host__ __device__ int64_t Elements() const { return layers * LayerStride(); }

    __host__ __device__ int64_t Offset(int64_t layer, int64_t head, int64_t pos,
                                       int64_t dim = 0) const
    {
        return layer * LayerStride() + head * head_stride + pos * pos_stride +
               dim;
    }
};

}  // namespace lsxcommon
```

- [ ] **Step 4: Register the test in CMake**

In `tests/CMakeLists.txt`, add near the other host-only `fast` tests (after the `test_tokenizer` block at ~line 2210):

```cmake
add_executable(test_kv_cache_layout test_kv_cache_layout.cpp)
target_link_libraries(test_kv_cache_layout PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_kv_cache_layout COMMAND test_kv_cache_layout)
set_tests_properties(test_kv_cache_layout PROPERTIES LABELS fast)
```

- [ ] **Step 5: Build and run**

Run:
```bash
cmake --build --preset linux-debug --target test_kv_cache_layout && \
  LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-debug/tests/test_kv_cache_layout
```
Expected: `All tests passed` (7 test cases).

- [ ] **Step 6: Add the device/host agreement test**

Append to `tests/test_kv_cache_layout.cpp`:

```cpp
#include <cuda_runtime.h>

__global__ void KvOffsetProbe(lsxcommon::KvCacheGeometry g, int64_t layer,
                              int64_t head, int64_t pos, int64_t dim,
                              int64_t *out)
{
    *out = g.Offset(layer, head, pos, dim);
}
```

Then add a Catch2 case that launches the probe on a 1-thread block and compares
against the host value. Guard it with a CUDA-availability `SKIP_RETURN_CODE 4`
style guard consistent with other CUDA tests in this repo, run under
`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`.

- [ ] **Step 7: Run the full fast suite and commit**

Run:
```bash
ctest --preset linux-debug -L fast
```
Expected: `100% tests passed, 0 tests failed out of 88`.

```bash
git add libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h tests/test_kv_cache_layout.cpp tests/CMakeLists.txt
git commit -m "feat: model-neutral KV cache geometry as three strides

The index is an affine 3-stride form rather than an enum-tagged layout, so any
permutation of (head, pos) is a choice of two numbers - no branch, no template
instantiation, and both layouts coexist in one binary. That last property is
what will make a byte-identical parity test possible across layouts.

LayerStride() and Elements() are layout-invariant, so switching layout is
size-neutral: only the decomposition of a layer's block into head and position
changes, never its size. The VRAM planner and resource governor are therefore
untouched by a layout change.

All arithmetic is int64_t - kv_heads*capacity*head_dim reaches 8.05G elements on
Xing at capacity 32768, and a wrapped index under head-major would yield
plausible-looking data rather than an obvious fault."
```

---

### Task 2: The universal knob

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model.h`
- Modify: `libs/lsxcommon/src/xing_moe_model.cpp`
- Modify: `libs/lsxcommon/src/nemotron_moe_model.cpp`
- Test: `tests/test_kv_cache_layout.cpp` (extend)

**Interfaces:**
- Consumes: `KvCacheLayout` from Task 1.
- Produces:
  - `ModelInitConfig::kv_cache_layout` (default `KvCacheLayout::kHeadMajor`).
  - `IModel::KvLayout() const -> KvCacheLayout` (virtual, default `kHeadMajor`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_cache_layout.cpp`:

```cpp
#include "lsxcommon/model.h"

TEST_CASE("default config is maximum acceleration", "[kv_layout][fast]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = "unused";
    CHECK(cfg.kv_cache_layout == lsxcommon::KvCacheLayout::kHeadMajor);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build --preset linux-debug --target test_kv_cache_layout
```
Expected: compile error — `ModelInitConfig` has no member `kv_cache_layout`.

- [ ] **Step 3: Add the field and the accessor**

In `libs/lsxcommon/include/lsxcommon/model.h`, add `#include "lsx_kv_cache.h"`
next to the other includes, and inside `struct ModelInitConfig` (after
`fast_math_mode`, ~line 36) add:

```cpp
    // KV cache layout for families that store one: [head][pos][dim] (default,
    // contiguous per-head working set) or [pos][head][dim]. Read once at init
    // and immutable afterwards - the strides are captured into the CUDA graph as
    // kernel arguments, so mutating this after capture would replay with the
    // stale value silently. Hybrid families whose recurrence state is not a KV
    // cache may ignore it; IModel::KvLayout() makes "ignored" observable.
    KvCacheLayout kv_cache_layout = KvCacheLayout::kHeadMajor;
```

In the `IModel` interface, next to `Type()` (~line 87), add:

```cpp
    // The layout this instance actually uses. Neutral knob, so a family that
    // ignores cfg.kv_cache_layout still has to answer - silently claiming the
    // requested layout would be a lie a test cannot catch.
    virtual KvCacheLayout KvLayout() const { return KvCacheLayout::kHeadMajor; }
```

- [ ] **Step 4: Make Xing report the real value**

In `libs/lsxcommon/src/xing_moe_model.cpp`, add:

```cpp
    KvCacheLayout KvLayout() const override { return kv_layout_; }
```

and store `cfg.kv_cache_layout` into a `KvCacheLayout kv_layout_` member during
init. Task 3 reads that member.

- [ ] **Step 5: Nemotron already is head-major**

In `libs/lsxcommon/src/nemotron_moe_model.cpp`, add the same override returning
`KvCacheLayout::kHeadMajor` with a comment that its index math is already
`(hk * cache_stride + t) * head_dim`.

- [ ] **Step 6: Run the suite and commit**

```bash
ctest --preset linux-debug -L fast
```
Expected: 88/88 (89 after the added case).

```bash
git add libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/src/xing_moe_model.cpp libs/lsxcommon/src/nemotron_moe_model.cpp tests/test_kv_cache_layout.cpp
git commit -m "feat: universal KV layout knob on ModelInitConfig, defaulting to head-major

The layout is chosen once per family, when its attention kernel is written, and
every sibling variant inherits it from there - so the init config is the only
place a default can be set. No env var, no CLI flag: default is maximum
acceleration and tests set what they need explicitly.

KvLayout() exists because Create(cfg) hands back a unique_ptr<IModel>: the
parity test must be able to observe that the requested layout was honoured, and
a hybrid family may legitimately ignore the field. 'Ignored' has to be visible
rather than a silent no-op."
```

---

### Task 3: Migrate Xing to head-major

**Files:**
- Modify: `libs/lsxcommon/src/xing_moe_inference.h`
- Modify: `libs/lsxcommon/src/xing_moe_inference.cu` (lines 285-300, 613-650, 297-298, 920-940, 1236-1247)
- Modify: `libs/lsxcommon/src/xing_moe_kernels.h`
- Modify: `libs/lsxcommon/src/xing_moe_kernels.cu` (lines 285-300, 513, 527, 573, 640, 682, 749-760)

**Interfaces:**
- Consumes: `KvCacheGeometry::Make(...)`, `KvCacheLayout` (Tasks 1-2).
- Produces:
  - `XingSession::kgeom`, `XingSession::vgeom` (`KvCacheGeometry`).
  - `AllocateXingSession(XingSession&, const XingDims&, int capacity, KvCacheLayout, cudaStream_t)`.
  - `LaunchXingAttention(..., KvCacheGeometry kgeom, KvCacheGeometry vgeom, ...)`.

- [ ] **Step 1: Add the geometry to the session**

In `libs/lsxcommon/src/xing_moe_inference.h`, in `XingSession`, add:

```cpp
    KvCacheGeometry kgeom{};
    KvCacheGeometry vgeom{};
```

and include `lsxcommon/lsx_kv_cache.h`.

- [ ] **Step 2: Build the geometry at allocation, keep the size identical**

In `AllocateXingSession` (`xing_moe_inference.cu:613`), change the signature to
take `KvCacheLayout layout`, and replace the stride computation at lines 627-630:

```cpp
    const int kv_layers = std::max(1, dims.layers);
    s.kv_layers = kv_layers;
    s.kgeom = KvCacheGeometry::Make(layout, capacity, frozen::kHeads,
                                   frozen::kQkNopeHeadDim + frozen::kQkRopeHeadDim,
                                   kv_layers);
    s.vgeom = KvCacheGeometry::Make(layout, capacity, frozen::kHeads,
                                   frozen::kVHeadDim, kv_layers);
    // Layout-invariant: identical to the old capacity*heads*dim product, so the
    // allocation size and the VRAM plan do not move.
    s.kv_layer_k = s.kgeom.LayerStride();
    s.kv_layer_v = s.vgeom.LayerStride();
```

- [ ] **Step 3: Change the write path**

`xing_write_cache_kernel` (`xing_moe_kernels.cu:287`) takes `kgeom`/`vgeom`
instead of `heads`; lines 297-298 become:

```cpp
    float *kb = kc + kgeom.Offset(0, h, pos);
    float *vb = vc + vgeom.Offset(0, h, pos);
```

`nope`, `rope`, `vh` stay as parameters (they describe the source row width).

- [ ] **Step 4: Change the six read sites**

In `xing_attn_kernel` and `xing_attn_split_kernel`, replace each
`(static_cast<size_t>(t) * heads + h) * qk` with `kgeom.Offset(0, h, t)` and
each `... * vh + d` with `vgeom.Offset(0, h, t, d)`:

- `xing_moe_kernels.cu:513` → `const float *kt0 = kc + kgeom.Offset(0, h, t);`
- `xing_moe_kernels.cu:527` → `const float *kt = kc + kgeom.Offset(0, h, t);`
- `xing_moe_kernels.cu:573` → `vc[vgeom.Offset(0, h, t, d)]`
- `xing_moe_kernels.cu:640` → `const float *kt = kc + kgeom.Offset(0, h, begin + t);`
- `xing_moe_kernels.cu:682` → `vc[vgeom.Offset(0, h, begin + t, d)]`

Inside these kernels `layer == 0`; the caller already advanced the base pointer
per layer.

- [ ] **Step 5: Update the launch signatures**

`LaunchXingAttention` (`xing_moe_kernels.cu:749`) replaces `int capacity` with
`KvCacheGeometry kgeom, KvCacheGeometry vgeom`; validate
`kgeom.capacity >= n_pos` instead of `capacity < n_pos`. Do the same for the
split-K launcher. Mirror both in `libs/lsxcommon/src/xing_moe_kernels.h`.

- [ ] **Step 6: Update the callers**

In the decode loop (`xing_moe_inference.cu:929`), the per-layer slices are
unchanged in spirit but now derive their length from geometry, and
`LaunchXingAttention` / `LaunchXingWriteCache` / the split launcher receive
`s.kgeom` and `s.vgeom`.

- [ ] **Step 7: Convert the host readback to a 2D copy**

At `xing_moe_inference.cu:1236-1247` one position's row is no longer contiguous
under head-major: `frozen::kHeads` runs of `head_dim` separated by
`head_stride`. Replace the two flat `cudaMemcpyAsync` calls with:

```cpp
    const size_t k_row = static_cast<size_t>(frozen::kHeads) * frozen::kQkHeadDim;
    const size_t v_row = static_cast<size_t>(frozen::kHeads) * frozen::kVHeadDim;
    if (host_k != nullptr && pos < s.capacity &&
        cudaMemcpy2DAsync(host_k, k_row * sizeof(float),
                          s.d_k + s.kgeom.Offset(0, 0, pos),
                          s.kgeom.head_stride * sizeof(float),
                          frozen::kQkHeadDim * sizeof(float), frozen::kHeads,
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return cudaErrorInvalidValue;
```

and the same for `host_v` with `vgeom` and `kVHeadDim`. The last row ends exactly
at the allocation end - no slack, no loop.

- [ ] **Step 8: Move the debug dump onto the geometry**

`XingStageDump("kcache", ...)` at `xing_moe_inference.cu:1013` reads one position;
route it through `s.kgeom` / `s.vgeom` like the readback does.

- [ ] **Step 9: Build both configurations**

```bash
cmake --build --preset linux-debug 2>&1 | grep -c " error:"
cmake --build --preset linux-release-znver4 2>&1 | grep -c " error:"
```
Expected: `0` for both.

- [ ] **Step 10: Run the fast suite**

```bash
ctest --preset linux-debug -L fast
```
Expected: 100% green.

- [ ] **Step 11: Commit**

```bash
git add libs/lsxcommon/src/xing_moe_inference.h libs/lsxcommon/src/xing_moe_inference.cu libs/lsxcommon/src/xing_moe_kernels.h libs/lsxcommon/src/xing_moe_kernels.cu
git commit -m "perf: Xing decode KV cache to [head][pos][dim]

One block per head plus split-K over positions means a single head walks the
whole context, so with [pos][head][dim] each step jumps kv_heads*head_dim - 24
KiB for Xing. Isolated, attention hit ~1364 GB/s while the context fitted L2 and
~788/820 GB/s once it did not; it is the only kernel whose time grows with
context (0.81 -> 2.28 ms/token between 128 and 2048 tokens).

The arithmetic is untouched - same scalar order, same __fadd_rn/__fmul_rn
intrinsics, only addresses move - so results stay bit-identical, which the parity
test checks rather than assumes.

The host KV readback becomes one cudaMemcpy2DAsync (dpitch = head_stride): under
head-major a position's row is kHeads runs of head_dim, and the 2D copy covers
all of them without a loop."
```

---

### Task 4: Parity across layouts (L2)

**Files:**
- Create: `tests/test_xing_kv_layout_parity.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `ModelInitConfig::kv_cache_layout`, `IModel::KvLayout()` (Task 2).
- Produces: the byte-identity guarantee.

- [ ] **Step 1: Write the test**

Create `tests/test_xing_kv_layout_parity.cpp` following the shape of
`tests/test_xing_moe_selfcheck.cpp` (same `ModelInitConfig` construction, same
prompt source):

```cpp
// Bit-exact parity across KV layouts. Characterization test: before the migration
// both runs are pos-major and agree trivially, so this has NO red phase. It is a
// safety net that guards every future change to the addressing, not a driver.
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

#include "lsxcommon/model_factory.h"
#include "lsxcommon/model.h"

namespace
{
constexpr const char *kXingRoot = "/mnt/llm_models/models/Xing4.0-29B-A4B";
constexpr const char *kPrompt = "The capital of France is";
}

TEST_CASE("KV layout does not change a single bit", "[xing][inference]")
{
    if (!std::filesystem::exists(kXingRoot)) SKIP("Xing-4.0 not on this host");

    std::vector<int32_t> head, pos;
    for (auto [layout, out] : {std::pair{lsxcommon::KvCacheLayout::kHeadMajor, &head},
                               std::pair{lsxcommon::KvCacheLayout::kPosMajor, &pos}}) {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = kXingRoot;
        cfg.max_gen_tokens = 32;
        cfg.kv_cache_layout = layout;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        REQUIRE(model);
        CHECK(model->KvLayout() == layout);

        lsxcommon::ModelRequest req;
        req.input_ids.clear();
        const std::string p = kPrompt;
        REQUIRE(model->BuildPromptInputIds(*lsxcommon::ActiveTokenizer(),
                                           std::string_view(p), req.input_ids));
        req.max_gen_tokens = 32;
        req.temperature = 0.0f;
        auto r = model->Infer(req);
        REQUIRE(r.ok);
        out->assign(r.gen_tokens.begin(), r.gen_tokens.end());
    }
    REQUIRE(head.size() == pos.size());
    CHECK(std::memcmp(head.data(), pos.data(),
                      head.size() * sizeof(int32_t)) == 0);
}
```

Adapt the tokenizer access and request field names to the real
`tests/test_xing_moe_selfcheck.cpp` (it already compiles against the real API);
copy its setup verbatim rather than inventing one.

- [ ] **Step 2: Register and run**

```cmake
add_executable(test_xing_kv_layout_parity test_xing_kv_layout_parity.cpp)
target_link_libraries(test_xing_kv_layout_parity PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_xing_kv_layout_parity COMMAND test_xing_kv_layout_parity)
set_tests_properties(test_xing_kv_layout_parity PROPERTIES LABELS inference)
```

```bash
cmake --preset linux-release-znver4 && \
  LSX_TEST_XING_ROOT=/mnt/llm_models/models/Xing4.0-29B-A4B \
  ./build/linux-release-znver4/tests/test_xing_kv_layout_parity
```
Expected: `All tests passed`. If it fails, the arithmetic assumption in the spec
is wrong - go to the Risks section before changing anything.

- [ ] **Step 3: Commit**

```bash
git add tests/test_xing_kv_layout_parity.cpp tests/CMakeLists.txt
git commit -m "test: Xing decode is bit-identical across KV layouts

Runs the same prompts through both layouts in one process and requires
byte-identical gen_tokens. This is a characterization test, not a driver: before
the migration both runs were pos-major and agreed trivially, so there is no red
phase and faking one would be theatre.

It is sound because for each (t, h) the scalar operation order, the values and
the explicitly-rounded intrinsics are unchanged - only addresses move, and the
split-K combine depends solely on partials computed identically."
```

---

### Task 5: Bandwidth budget (L3)

**Files:**
- Create: `tests/test_xing_kv_layout_perf.cu`
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/perf/kernel_budgets.json`

**Interfaces:**
- Consumes: `KvCacheGeometry` (Task 1), the Xing attention kernels (Task 3).
- Produces: budget `xing.kv_read.decode_gbps` (`kind: "min"`).

- [ ] **Step 1: Measure pos-major first**

Create `tests/test_xing_kv_layout_perf.cu`: allocate a real-shaped cache
(`kHeads=32`, `kQkHeadDim=192`, `kVHeadDim=128`, capacity 8191), fill it, run
`LaunchXingAttention` in a loop at a long context, time with CUDA events, and
compute achieved GB/s from the geometry:

```cpp
    const int64_t bytes = geom.Elements() * 2 * sizeof(float);  // K and V read once
    const double gbps = static_cast<double>(bytes) * iters / (ms * 1e6);
```

Sweep both layouts and `INFO` the numbers. Run it BEFORE flipping the default to
record the pos-major baseline. Expected: pos-major well below the ~1364 GB/s the
context-fitting case reaches.

- [ ] **Step 2: Write the budget as a failing test**

Add to `tests/perf/kernel_budgets.json` in the `budgets` array:

```json
    {"name": "xing.kv_read.decode_gbps", "gpu_key": "rtx 5090", "kind": "min",
     "budget": <head_major_measured_gbps * 0.95>, "observed": <pos_major_gbps>,
     "note": "head-major KV read at ctx 8191; the pos_major value is the floor this replaced"}
```

Use the measured numbers from Step 1 - do not invent a threshold. The budget must
be one the head-major layout clears and pos-major does not; that gap is the
deliverable.

- [ ] **Step 3: Register the test with the `perf` label**

```cmake
add_executable(test_xing_kv_layout_perf test_xing_kv_layout_perf.cu)
set_target_properties(test_xing_kv_layout_perf PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_xing_kv_layout_perf PRIVATE test_xing_cu lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_xing_kv_layout_perf COMMAND test_xing_kv_layout_perf)
set_tests_properties(test_xing_kv_layout_perf PROPERTIES LABELS perf)
```

- [ ] **Step 4: Run it, last, with the GPU otherwise idle**

```bash
./build/linux-release-znver4/tests/test_xing_kv_layout_perf
```
Expected: PASS with the head-major number above the budget.

- [ ] **Step 5: Re-baseline the Xing real-weight gates and E2E**

```bash
M=/mnt/llm_models/models/Xing4.0-29B-A4B
LSX_TEST_XING_ROOT=$M ./build/linux-release-znver4/tests/test_xing_moe_selfcheck
./build/linux-release-znver4/apps/lsxbenchmark/lsxbenchmark --bench \
  --model "$M" --max-gen 2048
```
Expect SelfCheck Paris x3 bit-identical and an E2E number above 119.1 t/s. Record
the new E2E figure in `tests/perf/perf_baselines.json`, and tighten
`xing.attention.decode` (currently `max 78.4966`) to the measured value with ~5%
headroom.

- [ ] **Step 6: Full regression, both builds**

```bash
ctest --preset linux-debug -L fast
cd build/linux-release-znver4 && ctest -L fast
```
Expected: 100% green in both.

- [ ] **Step 7: Commit**

```bash
git add tests/test_xing_kv_layout_perf.cu tests/CMakeLists.txt tests/perf/kernel_budgets.json tests/perf/perf_baselines.json
git commit -m "test: budget the head-major KV read, and re-baseline Xing

xing.kv_read.decode_gbps is a min budget set from the measured head-major
number with 5% margin, and observed from the pos-major floor it replaces - so
the budget encodes the gap rather than asserting a hoped-for figure. Bytes moved
come from the geometry, not from a hand-computed constant.

Also tightens xing.attention.decode and records the new E2E 2048 t/s."
```

---

### Task 6: Make the default stick

**Files:**
- Modify: `.opencode/skills/ingesting-a-model/SKILL.md` (step 5, ~line 245)
- Modify: `tests/test_model_independence_guard.cpp`
- Modify: `tests/test_model_suite_guard.cpp` if a new ctest target needs a stage

**Interfaces:**
- Consumes: `KvCacheGeometry` (Task 1).
- Produces: a machine-checked exemption list.

- [ ] **Step 1: Write the failing guard**

Add to `tests/test_model_independence_guard.cpp`:

```cpp
// Families whose attention kernels must index the KV cache through
// KvCacheGeometry rather than hand-rolled (t*heads + h)*dim arithmetic.
TEST_CASE("KV cache layout goes through the shared geometry", "[guard][fast]")
{
    static const std::vector<std::string> kKvLayoutExempt = {
        "nemotron_moe_inference_attn.cu",  // already head-major by construction
        "glm_moe_inference.cu",             // one thread per head: different
                                            // coalescing trade-off, analysed apart
    };
    // Xing migrated in this change; the rest adopt per family.
    ...
}
```

The scan reads every `*_inference*.cu` under `kSrcDir`, flags any file matching
`\*\s*(kv_)?heads\s*\+\s*(h|head|hh)\)\s*\*` (the pos-major idiom) that is not in
the exemption list and does not reference `KvCacheGeometry`, and fails with the
offending file and line. Start with only Xing expected to pass; each later
migration removes a file from the exemption list, so the list is the migration's
honest state at any point.

- [ ] **Step 2: Run it to verify it fails**

```bash
cmake --build --preset linux-debug --target test_model_independence_guard && \
  ./build/linux-debug/tests/test_model_independence_guard
```
Expected: FAIL naming the families not yet migrated. This is correct - the guard
is a migration ledger, and only the exempt files pass today.

**If this fails on files outside the KV path** (a false positive from the regex),
tighten the pattern before weakening the guard. Do not add files to the exemption
list to silence it.

- [ ] **Step 3: Update the skill, step 5**

In `.opencode/skills/ingesting-a-model/SKILL.md`, beside the existing *stride
invariant* bullet in step 5, add:

```markdown
- **KV cache layout:** default `kHeadMajor` via `KvCacheGeometry` from
  `libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h`. Do not hand-roll index
  arithmetic. If your model's attention blocking is not per-head (as in glm,
  one thread per head), do not copy head-major silently: choose explicitly and
  record the reason in the passport. `test_model_independence_guard` enforces
  this; its exemption list names the families exempt and why.
```

This stays step 5 of 8: the protocol is cited in code as "Model Ingestion
Protocol step N", so the step count is fixed. It is a structural choice baked into
kernel addressing, not a phase-4 tuning knob - do not ask for a bandwidth
measurement at ingestion.

- [ ] **Step 4: Run the full fast suite in both builds**

```bash
ctest --preset linux-debug -L fast
cd build/linux-release-znver4 && ctest -L fast
```

- [ ] **Step 5: Commit**

```bash
git add .opencode/skills/ingesting-a-model/SKILL.md tests/test_model_independence_guard.cpp
git commit -m "test: guard the KV layout default, and write it into the protocol

The layout is decided once per family, when its attention kernel is written, and
sibling variants inherit it from there - so step 5 of the ingestion protocol is
the only place the default can be set. Without it the next family copies
whichever neighbour it sits closest to, which is how pos-major spread to nine
files in the first place.

The guard's exemption list doubles as the machine-readable answer to which
families are per-head and therefore free to switch: nemotron is already
head-major, glm is one thread per head. Families migrate off the list over time,
so at any point it reads as the honest state of the migration."
```

---

## Self-Review

**Spec coverage:** Section "Strides, not layouts" -> Task 1. "The shared layer" ->
Task 1. "Runtime knob" -> Task 2 (field, accessor, set-once, nemotron/xing
reporting). "Xing migration" 7 touch points -> Task 3 Steps 1-8 (geometry, write,
5 read sites, signatures, callers, 2D readback, debug dump); prefill limitation is
recorded in the spec and needs no code. "Invariants and edge cases" -> Task 1
tests (int64 width, degenerate geometry, bijection) plus the preserved `pos <
capacity` guards in Task 3. "Testing" L1 -> Task 1, L2 -> Task 4, L3 -> Task 5.
"Order of work" -> the task order, with Task 6 last as specified. "Making the
default stick" -> Task 6. "Risks" -> Task 4 Step 2 names the falsification path.

**Placeholder scan:** No TBD/TODO. Thresholds that cannot be known in advance are
explicitly *measured* in Task 5 Step 1 rather than invented, with the procedure
given. Task 4 Step 1 carries an explicit instruction to copy the real API usage
from `test_xing_moe_selfcheck.cpp` instead of inventing one - the one place where
writing code blind would produce something that does not compile.

**Type consistency:** `KvCacheGeometry::Make(KvCacheLayout, int64_t, int64_t,
int64_t, int64_t)` is defined in Task 1 and called with exactly that arity in
Tasks 2-5. `Offset(layer, head, pos, dim = 0)` is used with 3 args in the write
path and 4 in the read paths, consistent with the default. `KvLayout()` returns
`KvCacheLayout` everywhere. `s.kgeom`/`s.vgeom` are named identically in Task 3's
declaration, allocation and call sites.
