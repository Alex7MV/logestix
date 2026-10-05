# KV cache layout `[head][pos][dim]` — shared layer + Xing migration

Date: 2026-10-05
Status: approved (brainstorming), pending implementation plan
Origin: `docs/OPEN_TASKS.md` §D — "Deferred to its own task (2026-10-05): the KV cache layout"

## Problem

Decode attention reads its K/V from a cache laid out `[pos][head][dim]`. One head's
dot product therefore walks a stride of `kv_heads * head_dim` elements per position —
for Xing, 32 heads x 192 floats = 24 KiB per step. Measured on `main` (`54412ad`):
attention is the ONLY kernel whose time grows with context, 0.81 -> 2.28 ms/token
between 128 and 2048 tokens, and isolated it achieves ~1364 GB/s while the context
still fits L2 (pos 1024) but only ~788/820 GB/s once it cannot (pos 4096/8191). So
it is L2-bandwidth-bound at short context and DRAM-bound at long context, and the
layout is why.

A head-major `[head][pos][dim]` layout makes the per-head working set contiguous.

The eight families whose decode attention is already "one block per head" + split-K
over positions get this for free from an index change:

| family | decode attention head mapping | file |
|---|---|---|
| xing | `h = blockIdx.x`, `s = blockIdx.y` | `xing_moe_kernels.cu:607` |
| qwen (+ornith) | `h = blockIdx.x` | `qwen_inference.cu:310` |
| gemma | `h = blockIdx.x` | `gemma_moe_inference.cu:432` |
| deepseek | `h = blockIdx.x` | `deepseek_moe_inference_attn.cu:448` |
| mimo | `h = blockIdx.x` | `mimo_moe_inference_attn.cu:161` |
| gigachat | `h = blockIdx.x` | `gigachat_moe_inference_attn.cu:66` |
| phi | `h = blockIdx.x % nh` | `phi_dense_inference.cu:183` |
| granite | `h = blockIdx.x % nh` | `granite_dense_inference.cu:183` |

Two families are different and are **out of scope**:

- **nemotron** is already head-major by construction —
  `(hk * cache_stride + t) * head_dim` at `nemotron_moe_inference_attn.cu:52`. It only
  has to report its layout through the new accessor.
- **glm** maps one *thread* per head (`glm_moe_inference.cu:160`). Head-major helps a
  per-head walk but a warp-of-heads has a different coalescing trade-off entirely.
  Deciding glm needs its own analysis, not an index swap.

## Scope

In: a model-neutral shared layer, plus its first adopter (Xing) as the proof and the
measured win. Out: adopting it in the other families — each is a later, separate
commit that reuses the same layer. The knob is neutral from day one so that those
commits need no new concept.

## Design

### Strides, not layouts

The layer stores no layout enumeration in its index math. A cache geometry is three
strides and one add:

```
offset = layer * layer_stride + head * head_stride + pos * pos_stride + dim
```

Both layouts we care about are the same three numbers, permuted:

| layout | `pos_stride` | `head_stride` |
|---|---|---|
| pos-major | `kv_heads * head_dim` | `head_dim` |
| head-major | `head_dim` | `capacity * head_dim` |

So any permutation — including ones no family needs yet — is a choice of strides, not
a separate code path. `KvCacheLayout` survives only as a *factory* for those stride
pairs, which is what makes it a cheap runtime choice (see "Runtime knob" below).

### The shared layer

One new header-only file, `libs/lsxcommon/include/lsxcommon/lsx_kv_cache.h`.
Model-neutral, includes only `<cstdint>`, therefore safe to include from `.cu`
(no absl container headers — `test_absl_container_guard` stays green). No CMake
change is needed, exactly as for `src/lsx_reduce.cuh`. It is not family-prefixed, so
`test_code_naming_guard` does not apply.

```cpp
namespace lsxcommon {

enum class KvCacheLayout { kHeadMajor, kPosMajor };

// Affine 3-stride geometry of one KV cache. Deliberately NOT an enum-tagged
// union: the index is a sum of strides, so every permutation is a choice of
// numbers rather than a branch or a template instantiation.
struct KvCacheGeometry {
    int64_t capacity;   // max cached positions per layer
    int64_t kv_heads;   // heads storing K/V (== q heads for MHA, kv_heads for GQA)
    int64_t head_dim;   // stored elements per head (MLA: kv_lora + rope)
    int64_t layers;
    int64_t pos_stride;
    int64_t head_stride;

    static constexpr KvCacheGeometry Make(KvCacheLayout layout, int64_t capacity,
                                          int64_t kv_heads, int64_t head_dim,
                                          int64_t layers);

    __host__ __device__ int64_t LayerStride() const;
    __host__ __device__ int64_t Elements() const;
    __host__ __device__ int64_t Offset(int64_t layer, int64_t head, int64_t pos,
                                       int64_t dim = 0) const;
};

}  // namespace lsxcommon
```

K and V are **two independent geometries**: Xing's K row is `kQkHeadDim = 192`
(`nope 128 + rope 64`) and its V row is `kVHeadDim = 128`
(`xing_moe_constants.h:52-53`). Both are passed to kernels by value — 8 `int64` = 64
bytes of kernel parameter — leaving two multiplies in the hot loop and no branch.

Two invariants make the swap size-neutral and are pinned by tests:

- `LayerStride() == kv_heads * capacity * head_dim` — **independent of layout**.
- `Elements() == layers * LayerStride()` — likewise.

Only the decomposition of a layer's block into head and position changes, never its
size. The VRAM planner and the resource governor are therefore untouched.

### Runtime knob

Per operator decision: the layout is a **universal model property, defaulting to
maximum acceleration**. No env var, no CLI flag.

- `KvCacheLayout` is declared in the shared layer, so it is a neutral concept.
- `ModelInitConfig` (`model.h:23`) gains one field next to the existing semantic
  init choices `fast_math` / `fast_math_mode`:
  `KvCacheLayout kv_cache_layout = KvCacheLayout::kHeadMajor;`
  `model.h` includes `lsx_kv_cache.h` for the enumeration. The dependency is
  one-way and cycle-free: the layer includes only `<cstdint>` and never sees a model.
- `ModelBase` gains a read accessor `KvCacheLayout KvLayout() const`. This is not
  decoration: `LsxModelFactory::Create(cfg)` hands back a `std::unique_ptr<IModel>`,
  so the parity test needs to *observe* that the requested layout was honoured, and a
  future hybrid family (Nemotron's Mamba-2, Alice's KDA) may legitimately ignore the
  field. "Ignored" must be visible, not a silent no-op.

Only Xing reads the field in this task. Nemotron is head-major already and only
reports `kHeadMajor`. Tests set `cfg.kv_cache_layout` explicitly per test.

Because the strides are plain runtime values, both layouts exist in one binary with no
template duplication — a direct dividend of the strides-not-layouts decision, and what
makes the byte-identical parity test possible at all.

**Set-once.** The layout is fixed for the lifetime of a loaded model. Strides are
captured into the CUDA graph as kernel arguments, so mutating the layout after capture
would replay with the stale value silently. The property is written at init and only
read afterwards.

### Xing migration

Seven touch points. Memory footprint does not change.

1. **Geometry**, built once at allocation:
   `kgeom = Make(kHeadMajor, capacity, kKvHeads, kQkHeadDim, layers)` and the same
   for `vgeom` with `kVHeadDim`. Existing `s.kv_layer_k` / `s.kv_layer_v`
   (`xing_moe_inference.cu:627`) become `kgeom.LayerStride()` / `vgeom.LayerStride()`
   — numerically identical to today.
2. **Position write**, `xing_moe_inference.cu:297-298`:
   `kc + kgeom.Offset(0, h, pos)`, `vc + vgeom.Offset(0, h, pos)`.
3. **Attention reads**, five sites — `xing_moe_kernels.cu:513,527,573,640,682`, all
   currently `(t * heads + h) * dim` — become `kgeom.Offset(0, h, t)` /
   `vgeom.Offset(0, h, t, d)`. Inside the kernels `layer == 0`; the base pointer is
   already advanced per layer by the caller.
4. **Launch signatures**: `LaunchXingAttention` and `LaunchXingAttnSplit` take
   `kgeom, vgeom` by value in place of `capacity`.
5. **Host KV row readback**, `xing_moe_inference.cu:1236-1247`. Under head-major one
   position's row is no longer contiguous: `kKvHeads` runs of `head_dim` separated by
   `head_stride`. `width = head_dim * elem`, `height = kv_heads`,
   `dpitch = head_stride * elem` turns this into **one** `cudaMemcpy2DAsync` instead
   of a loop. The last row ends exactly at the allocation end — no slack.
6. **Debug dump** `XingStageDump("kcache", …)` (`xing_moe_inference.cu:1013`) reads one
   position; it moves to the same geometry.
7. **Prefill**: Xing prefills token-by-token and has no separate prefill kernel, so
   the migration is confined to the decode path. This is a deliberate limitation:
   families with tiled prefill (phi, granite) take a write-pattern hit from the layout
   and must decide separately. That decision is not made here.

### Invariants and edge cases

- **int64 is mandatory, not defensive.** `kv_heads * capacity * head_dim` at capacity
  32768 over 40 layers is 8.05G elements on K alone — twice past int32. A silently
  wrapped index under head-major yields plausible-looking data. Width is pinned by a
  test, not a comment.
- **`pos` bounds do not weaken.** At `pos >= capacity` a pos-major write corrupts a
  foreign position and a head-major write corrupts a foreign head; both are wrong, so
  the existing `pos < capacity` guards stay and gain a test: a write at
  `pos == capacity` must not touch head 0's data.
- **Split-K chunk boundaries are unaffected** — they partition positions, not addresses.
- **Degenerate geometry** (`capacity == 0` or `kv_heads == 0`) yields `Elements() == 0`;
  tested so no family divides by zero.
- **Error style preserved**: readback failures keep returning `cudaErrorInvalidValue`.

## Testing

Three levels, with an honest split between what *drives* the change and what only
*guards* it.

### L1 — shared-layer unit tests (label `fast`, host + one tiny device test)

The real driver: written first, red because the header does not exist, then made
green by the implementation.

- `Make` produces the documented strides for both layouts.
- `LayerStride()` and `Elements()` are equal across layouts (the size-neutrality
  invariant).
- **Bijection**: over the whole `kv_heads x capacity` domain, both layouts yield the
  *same set* of offsets. This is the strongest available property — it says directly
  that no position is aliased or skipped — and it is what licenses the bit-exact
  parity claim below.
- MLA shape: `head_dim = kv_lora + rope` stays contiguous within a head.
- K/V of different widths (192 / 128) produce geometries that do not overlap.
- The `__device__` index agrees with the host one, guarding host/device drift.

### L2 — bit-exact parity (label `inference`, real weights)

Because the strides are runtime values, the test runs the model twice in one process —
`kHeadMajor` and `kPosMajor` — over the same prompts and requires **byte-identical**
logits and `gen_tokens`.

This is sound, not hopeful: for each `(t, h)` the scalar operation order, the values
and the explicitly-rounded intrinsics (`__fadd_rn`, `__fmul_rn`) are unchanged; only
addresses move. The split-K combine depends solely on partials computed identically.
So bit-identity is the expectation, and this test checks it.

**L2 has no red phase and is not a driver.** Before the migration both runs are
pos-major and agree trivially. It is a characterization test — a safety net that turns
green once and then guards every future change. The drivers are L1 and L3. Faking a
red phase for it would be theatre.

Consequence of the default: existing parity gates (SelfCheck Paris, FSI, CODE, DET)
take the new head-major default without being asked. That is intended, and it is why
L2 exists.

### L3 — speed budget (label `perf`)

The honest TDD driver for the layout itself: the feature *is* the speed.

- New budget `xing.kv_read.decode_gbps`, `kind: "min"`, measured by an isolated sweep
  over the real cache shape, bytes moved computed from the geometry rather than
  guessed. The budget is set **before** the change and is expected to fail on
  pos-major. Threshold is not invented: measure pos-major first, then fix the
  head-major number with margin.
- Tighten `xing.attention.decode` (currently `max 78.4966 us`) once measured.
- Record E2E 2048 in `tests/perf/perf_baselines.json`.

### Order of work

1. Branch.
2. L1 red -> green.
3. L3 budget red.
4. Migrate Xing (7 touch points).
5. L2 + L3 green.
6. Full regression: fast 87/87 in **both** builds, Xing real-weight gates, E2E,
   baseline update.

## Risks

- **A family whose attention blocking is not per-head regresses.** Mitigated by
  keeping `kPosMajor` reachable at runtime and by glm/nemotron being explicitly out of
  scope.
- **Tiled-prefill write pattern.** Out of scope here, recorded as a known
  consequence for phi/granite.
- **A non-bit-identical result would falsify the core premise.** If L2 fails, the
  first suspects are FP operation order under coalesced vs strided loads and any smem
  tile reuse that pos-major allowed; there is none in Xing today (`kt0`/`kt` are
  re-read per `t`), but that is the first place to look.
