# GigaChat3.5 Whole-Stack Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **No subagents** (AGENTS.md). Do NOT use subagent-driven-development.

**Goal:** Add an independent host-F32 whole-stack reference and a GPU parity test that compares `GcPipeline::Step` logits against it on the synthetic 2-layer GigaChat3.5 model.

**Architecture:** A pure-host reference header (`tests/gigachat_host_ref.h`) dequantizes the converted Arrow weights (NVFP4 + BF16/F32 raw) and reimplements every stage in F32, mirroring each F16/BF16 rounding the device applies. A new `.cu` test drives the pipeline and the reference over a 2-token sequence and compares logits.

**Tech Stack:** CUDA 13.4 (sm_120a/sm_121a), Catch2 v3.16.0, `lsxcommon` (Arrow, `gigachat_moe_*`), C++20.

## Global Constraints

- CC 12.x: 48 warps/SM, 100 KB smem/SM; no tcgen05; warp-level `mma.sync...mxf4nvf4.block_scale` only.
- One stream, no per-token `cudaDeviceSynchronize`; shared-memory RMW thread-0-only behind `__syncthreads()`; no `atomicAdd` in reduce paths.
- NVFP4 only via `lsx_inference_nvfp4.h`; norms/biases never quantized. No NVFP4 codec change.
- `#include`-reachable from `.cu`: no `<absl/container/*>` (host-only headers included by `.cu` stay `std::`).
- Numeric class C-exact: fixed accumulation order, device-independent launch shapes (`numeric-contract.md` §1.1). `-fmad=false` on device; the reference translation unit is built with `-Xcompiler=-ffp-contract=off`.
- Every new ctest target gets its label in the same commit; SKIP-capable targets declare `SKIP_RETURN_CODE 4`.
- Tests use the synthetic converted fixture only; the real 416 GB checkpoint is never read here.
- Label `inference` for the new target; no new model-suite stage.

### Exact converted layout (read-only facts)

- `logestix/pinned/global.arrow` (TensorBatchViews): `model.embed_tokens.weight` BF16 `[vocab,hidden]`, `lm_head.weight` BF16 `[vocab,hidden]`, and the `model.norm.{weight,gate_up_projection.weight,gate_down_projection.weight}` triple (BF16).
- `logestix/pinned/layer_%03d.arrow` (TensorBatchViews): the four layer norms `layers.N.{input_layernorm,post_attention_layernorm,post_self_attn_layernorm,post_feedforward_layernorm}`; MLA `layers.N.self_attn.{q_a_layernorm,kv_a_layernorm}`; GDN `layers.N.self_attn.{conv1d.weight (BF16,[conv_dim,1,K]), A_log (BF16,[v_heads]), dt_bias (BF16,[v_heads]), in_proj_ba.weight (BF16,[ba_rows,hidden]), norm.weight (BF16,[hv])}`; MoE `layers.N.mlp.gate.weight` BF16 `[E,hidden]` and `layers.N.mlp.gate.e_score_correction_bias` F32 `[E]`.
- `logestix/pinned/layer_%03d_nvfp4.arrow` (DenseNvfp4BatchViews): names end with `layers.N.self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj,gate_proj}.weight`, `layers.N.self_attn.{in_proj_qkvz,out_proj}.weight`, `layers.N.mlp.{gate_proj,up_proj,down_proj}.weight` (dense), `layers.N.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`.
- Routed experts: `logestix/experts/layer_N/exp_E.arrow`, decoded by `GcExpertStore::DecodeEntry` into `GcExpertEntry` (`host_w1`=gate, `host_w3`=up, `host_w2`=down).
- NVFP4 decode (device + host, shared codec): element `e` → nibble of `packed[e>>1]` (low if even, high if odd), scale `E4M3ScaleToFloat(scales[e>>4]) * global_scale`; value `E2M1Value(nib) * scale`. `E2M1Value`/`E4M3ScaleToFloat` live in `libs/lsxcommon/src/nvfp4_codec.h` (`lsxcommon::gpu::nvfp4`).

### Synthetic fixture geometry (from `tests/test_gigachat_moe_fixture.h`)

`hidden=128, vocab=128, layers=2, full_attention_layers={1}, experts=2, top_k=1, moe_intermediate=256, dense_intermediate=256, gate_rank=16, gdn conv_dim=512, conv_k=4, v_heads=2, k_heads=1, hk=hv=128, MLA heads=2, q_lora=kv_lora=128, nope=128, rope=64, v_head=128, rms_eps=1e-6, routed_scale=2.5, swiglu_limit=10`.

---

### Task 1: Host-reference header primitives + Arrow readers; refactor the attn test

**Files:**
- Create: `tests/gigachat_host_ref.h`
- Modify: `tests/test_gigachat_moe_attn.cu:21-192` (delete moved helpers, add include + `using`)

**Interfaces:**
- Consumes: `lsxcommon::TensorBatchViews`, `lsxcommon::DenseNvfp4BatchViews`, `lsxcommon::TensorView`, `lsxcommon::DenseNvfp4View` (`lsxcommon/schema.h`); `lsxcommon::gpu::nvfp4::{E2M1Value,E4M3ScaleToFloat}` (`nvfp4_codec.h`).
- Produces (namespace `gigachatref`): `R16(float)`, `RBf16(float)`, `GemvRef(W,x,N,K)`, `NormRef(x,w,up,down,dim,rank,eps)`, `RopeRef(q,heads,qk,nope,k,c,s)`, `FindTv(batch,suffix)`, `FindNv(batch,suffix)`, `Bf16ToF32(data,n)`, `F32Copy(data,n)`, `DequantNvfp4(packed,scales,gs,n)`, `struct RefNorm`, `struct RefNvfp4`, `ReadNorm(raw,prefix)`, `ReadNv(nv,suffix)`.

- [ ] **Step 1: Refactor the attn test to consume the shared header (makes the build fail)**

In `tests/test_gigachat_moe_attn.cu`:
- Add after the existing includes: `#include "gigachat_host_ref.h"` and `using namespace gigachatref;`.
- Delete the moved definitions from the anonymous namespace: `R16` (line 23), `RBf16` (line 24), `GemvRef` (lines 122-134), `NormRef` (lines 136-167), `RopeRef` (lines 169-190). Keep `kE2m1`, `GridVal`, `E2m1Nibble`, `Nvfp4Host`, `PackNvfp4`, `DevCopy`, `DevNvfp4`, `DevBf16`, `DevNorm`.

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_attn 2>&1 | tail -20`
Expected: FAIL — `gigachat_host_ref.h: No such file or directory`.

- [ ] **Step 3: Create `tests/gigachat_host_ref.h` with primitives and readers**

```cpp
#pragma once

// Host F32 reference for the GigaChat3.5 whole-stack synthetic parity test.
// Pure host (std:: only; included from .cu translation units, so no abseil
// containers). Shares the NVFP4 codec with the device so dequant is exact; the
// arithmetic is independent of the device kernels and mirrors every F16/BF16
// rounding the pipeline applies.
#include "gigachat_moe_inference_attn.h"
#include "gigachat_moe_device_math.h"
#include "nvfp4_codec.h"
#include "lsxcommon/schema.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace gigachatref {

using lsxcommon::DenseNvfp4BatchViews;
using lsxcommon::DenseNvfp4View;
using lsxcommon::TensorBatchViews;
using lsxcommon::TensorView;
using GcMlaDims = lsxcommon::gigachat::GcMlaDims;
using GcGdnDims = lsxcommon::gigachat::GcGdnDims;

// ── F16 / BF16 rounding (mirrors the device conversions) ───────────────────
inline float R16(float v) { return __half2float(__float2half(v)); }
inline float RBf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// ── F32 primitives ─────────────────────────────────────────────────────────
inline std::vector<float> GemvRef(const std::vector<float> &W,
                                  const std::vector<float> &x, int N, int K)
{
    std::vector<float> y(static_cast<size_t>(N), 0.0f);
    for (int n = 0; n < N; ++n)
    {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += W[static_cast<size_t>(n) * K + k] * x[static_cast<size_t>(k)];
        y[static_cast<size_t>(n)] = acc;
    }
    return y;
}

// ZeroCenteredGatedNorm: xn = x * rsqrt(mean(x^2)+eps);
// gate = gate_down @ silu(gate_up @ xn); out = xn*(1+w)*2*sigmoid(gate).
inline std::vector<float> NormRef(const std::vector<float> &x,
                                  const std::vector<float> &w,
                                  const std::vector<float> &up,
                                  const std::vector<float> &down, int dim,
                                  int rank, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    std::vector<float> xn(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) xn[i] = x[i] * inv;
    std::vector<float> gu(static_cast<size_t>(rank), 0.0f);
    for (int r = 0; r < rank; ++r)
    {
        float a = 0.0f;
        for (int i = 0; i < dim; ++i)
            a += up[static_cast<size_t>(r) * dim + i] * xn[i];
        gu[r] = a / (1.0f + std::exp(-a));
    }
    std::vector<float> g(static_cast<size_t>(dim), 0.0f);
    for (int i = 0; i < dim; ++i)
    {
        float a = 0.0f;
        for (int r = 0; r < rank; ++r)
            a += down[static_cast<size_t>(i) * rank + r] * gu[r];
        g[i] = a;
    }
    std::vector<float> out(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
        out[i] = xn[i] * (1.0f + w[i]) * 2.0f / (1.0f + std::exp(-g[i]));
    return out;
}

// Interleaved RoPE over adjacent pairs: q [heads, qk] at offset `nope`, k [rope].
inline void RopeRef(std::vector<float> &q, int heads, int qk, int nope,
                    std::vector<float> &k, const std::vector<float> &c,
                    const std::vector<float> &s)
{
    const int half = static_cast<int>(c.size());
    for (int h = 0; h < heads; ++h)
        for (int j = 0; j < half; ++j)
        {
            const int i0 = h * qk + nope + 2 * j;
            const float a = q[i0];
            const float b = q[i0 + 1];
            q[i0] = a * c[j] - b * s[j];
            q[i0 + 1] = a * s[j] + b * c[j];
        }
    for (int j = 0; j < half; ++j)
    {
        const float a = k[2 * j];
        const float b = k[2 * j + 1];
        k[2 * j] = a * c[j] - b * s[j];
        k[2 * j + 1] = a * s[j] + b * c[j];
    }
}

// ── Arrow readers ──────────────────────────────────────────────────────────
inline const TensorView *FindTv(const TensorBatchViews &b,
                                std::string_view suffix)
{
    const TensorView *r = nullptr;
    for (const auto &t : b.tensors)
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            r = &t;
    return r;
}

inline const DenseNvfp4View *FindNv(const DenseNvfp4BatchViews &b,
                                    std::string_view suffix)
{
    const DenseNvfp4View *r = nullptr;
    for (const auto &t : b.records)
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            r = &t;
    return r;
}

inline std::vector<float> Bf16ToF32(const uint8_t *p, int64_t n)
{
    std::vector<float> o(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        __nv_bfloat16 b;
        std::memcpy(&b, p + 2 * i, sizeof(b));
        o[static_cast<size_t>(i)] = __bfloat162float(b);
    }
    return o;
}

inline std::vector<float> F32Copy(const uint8_t *p, int64_t n)
{
    std::vector<float> o(static_cast<size_t>(n));
    std::memcpy(o.data(), p, static_cast<size_t>(n) * sizeof(float));
    return o;
}

inline std::vector<float> DequantNvfp4(const uint8_t *packed,
                                       const uint8_t *scales, float gs,
                                       int64_t n)
{
    std::vector<float> w(static_cast<size_t>(n));
    for (int64_t e = 0; e < n; ++e)
    {
        const uint8_t byte = packed[e >> 1];
        const uint8_t nib = (e & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0x0F);
        const float scl =
            lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(scales[e >> 4]) * gs;
        w[static_cast<size_t>(e)] =
            lsxcommon::gpu::nvfp4::E2M1Value(nib) * scl;
    }
    return w;
}

struct RefNorm {
    int dim = 0;
    int rank = 0;
    std::vector<float> w, up, down;
};

struct RefNvfp4 {
    int rows = 0;
    int cols = 0;
    std::vector<float> w;  // [rows*cols] dequantized F32
};

inline RefNorm ReadNorm(const TensorBatchViews &raw, const std::string &prefix)
{
    const TensorView *w = FindTv(raw, prefix + ".weight");
    const TensorView *gu = FindTv(raw, prefix + ".gate_up_projection.weight");
    const TensorView *gd = FindTv(raw, prefix + ".gate_down_projection.weight");
    RefNorm n;
    n.dim = static_cast<int>(w->shape[0]);
    n.rank = static_cast<int>(gu->shape[0]);
    n.w = Bf16ToF32(w->data, n.dim);
    n.up = Bf16ToF32(gu->data, static_cast<int64_t>(n.rank) * n.dim);
    n.down = Bf16ToF32(gd->data, static_cast<int64_t>(n.dim) * n.rank);
    return n;
}

inline RefNvfp4 ReadNv(const DenseNvfp4BatchViews &nv, std::string_view suffix)
{
    const DenseNvfp4View *r = FindNv(nv, suffix);
    RefNvfp4 o;
    o.rows = static_cast<int>(r->shape[0]);
    o.cols = static_cast<int>(r->shape[1]);
    o.w = DequantNvfp4(r->packed, r->scales, r->global_scale,
                       static_cast<int64_t>(o.rows) * o.cols);
    return o;
}

} // namespace gigachatref
```

- [ ] **Step 4: Build and run the attn test**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_attn && ./build/linux-debug/tests/test_gigachat_moe_attn`
Expected: PASS (all 3 test cases).

- [ ] **Step 5: Commit**

```bash
git add tests/gigachat_host_ref.h tests/test_gigachat_moe_attn.cu
git commit -m "test: extract GigaChat host reference primitives + Arrow readers"
```

---

### Task 2: MLA and GatedDeltaNet reference functions

**Files:**
- Modify: `tests/gigachat_host_ref.h` (append structs + functions)
- Modify: `tests/test_gigachat_moe_attn.cu` (replace the inline MLA/GDN reference lambdas with the shared functions)

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `struct RefMla { RefNvfp4 q_a,q_b,kv_a,kv_b,gate,o_proj; RefNorm q_a_norm,kv_a_norm; }`; `struct RefGdn { RefNvfp4 in_qkvz,out_proj; std::vector<float> in_ba,conv,A_log,dt_bias,norm; int conv_dim,conv_k,ba_rows,v_heads; }`; `MlaDecodeRef(const RefMla&, const GcMlaDims&, float eps, float scale, const std::vector<float>& rc, const std::vector<float>& rs, int pos, std::vector<float>& k_cache, std::vector<float>& v_cache, const std::vector<float>& x, std::vector<float>& out)`; `GdnDecodeRef(const RefGdn&, const GcGdnDims&, float eps, float onorm_scale, std::vector<float>& conv_state, std::vector<float>& rec_state, const std::vector<float>& x, std::vector<float>& out)`.

- [ ] **Step 1: Replace the attn test's inline MLA reference with `MlaDecodeRef`**

In the MLA test (`tests/test_gigachat_moe_attn.cu`, currently lines ~250-403):
- Replace the local weight vectors used by the reference with a `RefMla` built from the same `w_*` vectors:
```cpp
RefMla rmla;
rmla.q_a = {q_lora, hidden, w_qa};
rmla.q_b = {heads * qk, q_lora, w_qb};
rmla.kv_a = {kvw, hidden, w_kva};
rmla.kv_b = {kvb, kv_lora, w_kvb};
rmla.gate = {hv, hidden, w_gate};
rmla.o_proj = {hidden, hv, w_o};
rmla.q_a_norm = {q_lora, rank, qn_w, qn_up, qn_down};
rmla.kv_a_norm = {kv_lora, rank, kn_w, kn_up, kn_down};
std::vector<float> k_cache_ref((size_t)capacity * heads * qk, 0.0f);
std::vector<float> v_cache_ref((size_t)capacity * heads * vh, 0.0f);
```
- Replace the two `token(...)` calls and the manual attention block with:
```cpp
std::vector<float> ref;
for (int p = 0; p < 2; ++p)
{
    std::vector<float> out_ref;
    MlaDecodeRef(rmla, d, eps, scale, rc, rs, p, k_cache_ref, v_cache_ref,
                 p == 0 ? x0 : x1, out_ref);
    ref = out_ref;
}
```
- Keep the tolerance assertion unchanged (`margin(1e-3f)`) and the `hout` from device token `pos=1`.

- [ ] **Step 2: Replace the attn test's inline GDN reference with `GdnDecodeRef`**

In the GDN test (`tests/test_gigachat_moe_attn.cu`, currently lines ~436-639):
```cpp
RefGdn rg;
rg.in_qkvz = {qkvz_dim, hidden, w_qkvz};
rg.out_proj = {hidden, value_dim, w_out};
rg.in_ba = w_ba;
rg.conv = w_conv;
rg.A_log = A_log;
rg.dt_bias = dt_bias;
rg.norm = w_norm;
rg.conv_dim = conv_dim;
rg.conv_k = conv_k;
rg.ba_rows = ba_dim;
rg.v_heads = v_heads;
std::vector<float> conv_state_ref((size_t)conv_dim * (conv_k - 1), 0.0f);
std::vector<float> rec_state_ref((size_t)v_heads * hk * hv, 0.0f);
std::vector<float> r1;
for (int p = 0; p < 2; ++p)
{
    std::vector<float> out_ref;
    GdnDecodeRef(rg, d, eps, onorm_scale, conv_state_ref, rec_state_ref,
                 p == 0 ? x0 : x1, out_ref);
    r1 = out_ref;
}
```
Keep the tolerance assertion `margin(1e-3f)` on `hout` vs `r1`.

- [ ] **Step 3: Run the build to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_attn 2>&1 | tail -20`
Expected: FAIL — `MlaDecodeRef` / `GdnDecodeRef` not declared.

- [ ] **Step 4: Append the reference functions to `tests/gigachat_host_ref.h`**

Insert before the closing `} // namespace gigachatref`, replacing the existing `} // namespace gigachatref`:

```cpp
struct RefMla {
    RefNvfp4 q_a, q_b, kv_a, kv_b, gate, o_proj;
    RefNorm q_a_norm, kv_a_norm;
};

inline void MlaDecodeRef(const RefMla &m, const GcMlaDims &d, float eps,
                         float scale, const std::vector<float> &rc,
                         const std::vector<float> &rs, int pos,
                         std::vector<float> &k_cache,
                         std::vector<float> &v_cache,
                         const std::vector<float> &x,
                         std::vector<float> &out)
{
    const int hidden = d.hidden, heads = d.heads, q_lora = d.q_lora,
              kv_lora = d.kv_lora, nope = d.qk_nope, rope = d.rope,
              vh = d.v_head;
    const int qk = nope + rope, kvw = kv_lora + rope, hv = heads * vh,
              kvb = heads * (nope + vh), half = rope / 2;
    std::vector<float> x16(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) x16[i] = R16(x[i]);

    std::vector<float> qa = GemvRef(m.q_a.w, x16, q_lora, hidden);
    for (float &z : qa) z = R16(z);
    std::vector<float> qan =
        NormRef(qa, m.q_a_norm.w, m.q_a_norm.up, m.q_a_norm.down, q_lora,
                m.q_a_norm.rank, eps);
    const float aq = std::sqrt(static_cast<float>(hidden) / q_lora);
    std::vector<float> qas(static_cast<size_t>(q_lora));
    for (int i = 0; i < q_lora; ++i) qas[i] = R16(qan[i] * aq);
    std::vector<float> qb = GemvRef(m.q_b.w, qas, heads * qk, q_lora);
    for (float &z : qb) z = R16(z);

    std::vector<float> kv = GemvRef(m.kv_a.w, x16, kvw, hidden);
    for (float &z : kv) z = R16(z);
    std::vector<float> kp(kv.begin(), kv.begin() + kv_lora);
    std::vector<float> krot(kv.begin() + kv_lora, kv.end());
    std::vector<float> kpn =
        NormRef(kp, m.kv_a_norm.w, m.kv_a_norm.up, m.kv_a_norm.down, kv_lora,
                m.kv_a_norm.rank, eps);
    const float akv = std::sqrt(static_cast<float>(hidden) / kv_lora);
    std::vector<float> kps(static_cast<size_t>(kv_lora));
    for (int i = 0; i < kv_lora; ++i) kps[i] = R16(kpn[i] * akv);
    std::vector<float> kvb_w = GemvRef(m.kv_b.w, kps, kvb, kv_lora);
    for (float &z : kvb_w) z = R16(z);

    std::vector<float> c(static_cast<size_t>(half)), s(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j)
    {
        c[static_cast<size_t>(j)] = rc[static_cast<size_t>(pos) * half + j];
        s[static_cast<size_t>(j)] = rs[static_cast<size_t>(pos) * half + j];
    }
    RopeRef(qb, heads, qk, nope, krot, c, s);

    for (int h = 0; h < heads; ++h)
    {
        for (int i = 0; i < nope; ++i)
        {
            k_cache[(static_cast<size_t>(pos) * heads + h) * qk + i] =
                kvb_w[static_cast<size_t>(h) * (nope + vh) + i];
            v_cache[(static_cast<size_t>(pos) * heads + h) * vh + i] =
                kvb_w[static_cast<size_t>(h) * (nope + vh) + nope + i];
        }
        for (int i = 0; i < rope; ++i)
            k_cache[(static_cast<size_t>(pos) * heads + h) * qk + nope + i] =
                krot[static_cast<size_t>(i)];
    }

    std::vector<float> gate = GemvRef(m.gate.w, x16, hv, hidden);
    for (float &z : gate) z = R16(z);

    std::vector<float> attn(static_cast<size_t>(hv), 0.0f);
    for (int h = 0; h < heads; ++h)
    {
        std::vector<float> sc(static_cast<size_t>(pos) + 1);
        for (int t = 0; t <= pos; ++t)
        {
            float a = 0.0f;
            for (int i = 0; i < qk; ++i)
                a += qb[static_cast<size_t>(h) * qk + i] *
                     k_cache[(static_cast<size_t>(t) * heads + h) * qk + i];
            sc[static_cast<size_t>(t)] = a * scale;
        }
        float mx = sc[0];
        for (int t = 1; t <= pos; ++t) mx = std::fmax(mx, sc[static_cast<size_t>(t)]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t)
        {
            sc[static_cast<size_t>(t)] = std::exp(sc[static_cast<size_t>(t)] - mx);
            sum += sc[static_cast<size_t>(t)];
        }
        for (int i = 0; i < vh; ++i)
        {
            float o = 0.0f;
            for (int t = 0; t <= pos; ++t)
                o += sc[static_cast<size_t>(t)] / sum *
                     v_cache[(static_cast<size_t>(t) * heads + h) * vh + i];
            const float g = gate[static_cast<size_t>(h) * vh + i];
            attn[static_cast<size_t>(h) * vh + i] =
                R16(o / (1.0f + std::exp(-g)));
        }
    }
    out = GemvRef(m.o_proj.w, attn, hidden, hv);
    for (float &z : out) z = R16(z);
}

struct RefGdn {
    RefNvfp4 in_qkvz, out_proj;
    std::vector<float> in_ba, conv, A_log, dt_bias, norm;
    int conv_dim = 0, conv_k = 0, ba_rows = 0, v_heads = 0;
};

inline void GdnDecodeRef(const RefGdn &g, const GcGdnDims &d, float eps,
                         float onorm_scale, std::vector<float> &conv_state,
                         std::vector<float> &rec_state,
                         const std::vector<float> &x, std::vector<float> &out)
{
    const int hidden = d.hidden, k_heads = d.k_heads, v_heads = d.v_heads,
              hk = d.hk, hv = d.hv, conv_k = d.conv_k;
    const int key_dim = k_heads * hk, value_dim = v_heads * hv;
    const int conv_dim = 2 * key_dim + value_dim, ba_dim = 2 * v_heads;
    const int qkvz_dim = 2 * key_dim + 2 * value_dim, rep = v_heads / k_heads;
    const int space = 2 * hk + 2 * hv * rep;

    std::vector<float> x16(static_cast<size_t>(hidden)),
        xb(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
    {
        x16[i] = R16(x[i]);
        xb[i] = RBf16(x[i]);
    }
    std::vector<float> qkvz = GemvRef(g.in_qkvz.w, x16, qkvz_dim, hidden);
    for (float &f : qkvz) f = R16(f);
    std::vector<float> ba = GemvRef(g.in_ba, xb, ba_dim, hidden);

    std::vector<float> mixed(static_cast<size_t>(conv_dim), 0.0f);
    std::vector<float> z(static_cast<size_t>(value_dim), 0.0f);
    for (int gh = 0; gh < k_heads; ++gh)
    {
        const float *base = &qkvz[static_cast<size_t>(gh) * space];
        for (int i = 0; i < hk; ++i)
        {
            mixed[gh * hk + i] = base[i];
            mixed[key_dim + gh * hk + i] = base[hk + i];
        }
        for (int j = 0; j < hv * rep; ++j)
        {
            mixed[2 * key_dim + gh * hv * rep + j] = base[2 * hk + j];
            z[gh * hv * rep + j] = base[2 * hk + hv * rep + j];
        }
    }

    std::vector<float> beta(static_cast<size_t>(v_heads));
    std::vector<float> gv(static_cast<size_t>(v_heads));
    for (int h = 0; h < v_heads; ++h)
    {
        const int kh = h / rep;
        const float b = ba[kh * 2 * rep + (h % rep)];
        const float a = ba[kh * 2 * rep + rep + (h % rep)];
        beta[static_cast<size_t>(h)] = 1.0f / (1.0f + std::exp(-b));
        const float t = a + g.dt_bias[static_cast<size_t>(h)];
        const float sp = t > 20.0f ? t : std::log1p(std::exp(t));
        gv[static_cast<size_t>(h)] = -std::exp(g.A_log[static_cast<size_t>(h)]) * sp;
    }

    std::vector<float> conv_out(static_cast<size_t>(conv_dim));
    for (int dd = 0; dd < conv_dim; ++dd)
    {
        float acc = g.conv[static_cast<size_t>(dd) * conv_k + (conv_k - 1)] *
                    mixed[static_cast<size_t>(dd)];
        for (int j = 0; j < conv_k - 1; ++j)
            acc += g.conv[static_cast<size_t>(dd) * conv_k + j] *
                   conv_state[static_cast<size_t>(dd) * (conv_k - 1) + j];
        for (int j = 0; j < conv_k - 2; ++j)
            conv_state[static_cast<size_t>(dd) * (conv_k - 1) + j] =
                conv_state[static_cast<size_t>(dd) * (conv_k - 1) + j + 1];
        conv_state[static_cast<size_t>(dd) * (conv_k - 1) + (conv_k - 2)] =
            mixed[static_cast<size_t>(dd)];
        conv_out[static_cast<size_t>(dd)] = acc / (1.0f + std::exp(-acc));
    }

    std::vector<float> q(static_cast<size_t>(v_heads) * hk),
        kk(static_cast<size_t>(v_heads) * hk),
        vv(static_cast<size_t>(value_dim));
    const float qscale = 1.0f / std::sqrt(static_cast<float>(hk));
    for (int h = 0; h < v_heads; ++h)
    {
        const int kh = h / rep;
        double sq = 0.0, sk = 0.0;
        for (int i = 0; i < hk; ++i)
        {
            const float qv = conv_out[kh * hk + i];
            const float kv = conv_out[key_dim + kh * hk + i];
            sq += static_cast<double>(qv) * qv;
            sk += static_cast<double>(kv) * kv;
        }
        const float iq = 1.0f / std::sqrt(static_cast<float>(sq) + 1e-6f);
        const float ik = 1.0f / std::sqrt(static_cast<float>(sk) + 1e-6f);
        for (int i = 0; i < hk; ++i)
        {
            q[static_cast<size_t>(h) * hk + i] =
                conv_out[kh * hk + i] * iq * qscale;
            kk[static_cast<size_t>(h) * hk + i] =
                conv_out[key_dim + kh * hk + i] * ik;
        }
        for (int j = 0; j < hv; ++j)
            vv[static_cast<size_t>(h) * hv + j] =
                conv_out[2 * key_dim + h * hv + j];
    }

    std::vector<float> o(static_cast<size_t>(value_dim));
    for (int h = 0; h < v_heads; ++h)
    {
        const size_t sb = static_cast<size_t>(h) * hk * hv;
        const float decay = std::exp(gv[static_cast<size_t>(h)]);
        for (int i = 0; i < hk; ++i)
            for (int j = 0; j < hv; ++j)
                rec_state[sb + static_cast<size_t>(i) * hv + j] *= decay;
        for (int j = 0; j < hv; ++j)
        {
            float kvmem = 0.0f;
            for (int i = 0; i < hk; ++i)
                kvmem += rec_state[sb + static_cast<size_t>(i) * hv + j] *
                         kk[static_cast<size_t>(h) * hk + i];
            const float delta =
                (vv[static_cast<size_t>(h) * hv + j] - kvmem) *
                beta[static_cast<size_t>(h)];
            for (int i = 0; i < hk; ++i)
                rec_state[sb + static_cast<size_t>(i) * hv + j] +=
                    kk[static_cast<size_t>(h) * hk + i] * delta;
            float acc = 0.0f;
            for (int i = 0; i < hk; ++i)
                acc += rec_state[sb + static_cast<size_t>(i) * hv + j] *
                       q[static_cast<size_t>(h) * hk + i];
            o[static_cast<size_t>(h) * hv + j] = acc;
        }
    }

    std::vector<float> onorm(static_cast<size_t>(value_dim));
    for (int h = 0; h < v_heads; ++h)
    {
        double ss = 0.0;
        for (int j = 0; j < hv; ++j)
            ss += static_cast<double>(o[static_cast<size_t>(h) * hv + j]) *
                  o[static_cast<size_t>(h) * hv + j];
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / hv) + eps);
        for (int j = 0; j < hv; ++j)
        {
            const float xn = o[static_cast<size_t>(h) * hv + j] * inv;
            const float zz = z[static_cast<size_t>(h) * hv + j];
            onorm[static_cast<size_t>(h) * hv + j] =
                xn * (1.0f + g.norm[static_cast<size_t>(j)]) * onorm_scale /
                (1.0f + std::exp(-zz));
        }
    }
    for (float &f : onorm) f = R16(f);
    out = GemvRef(g.out_proj.w, onorm, hidden, value_dim);
    for (float &f : out) f = R16(f);
}

} // namespace gigachatref
```

- [ ] **Step 5: Build and run the attn test**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_attn && ./build/linux-debug/tests/test_gigachat_moe_attn`
Expected: PASS (bit-exact rope; MLA parity 1e-3; GDN parity 1e-3).

- [ ] **Step 6: Commit**

```bash
git add tests/gigachat_host_ref.h tests/test_gigachat_moe_attn.cu
git commit -m "test: extract MLA/GDN host references into the shared header"
```

---

### Task 3: Stack reference + whole-stack parity test

**Files:**
- Modify: `tests/gigachat_host_ref.h` (append router / MoE / dense / stack)
- Create: `tests/test_gigachat_moe_parity.cu`
- Modify: `tests/CMakeLists.txt` (new standalone `.cu` object + executable + `inference` label)

**Interfaces:**
- Consumes: Task 1+2 header; `lsxcommon::gigachat::{LoadGigaChatIndex,LoadGigaChatWeights,InitGigaChatExpertCache,LoadGcExpertStore,GcExpertCache,GcWeights,GcPipeline,GcPipelineDims,GcBuildYarnTables}`; `lsxcommon::LoadTensorBatchViews`, `lsxcommon::LoadDenseNvfp4BatchViews`; `gigachattest::WriteGigaChatCheckpoint`, `gigachattest::MakeTempDir`, `lsxcommon::gigachat::ConvertGigaChatModel`.
- Produces: `struct RefExpert`, `struct RefMoe`, `struct RefDense`, `struct RefRoute`, `struct RefLayer`, `struct RefModel`, `RouterTopKRef(...)`, `MoeRef(...)`, `DenseMlpRef(...)`, `GatedNormRef(...)`, `RunModelRef(...)`.
- Note on the spec's helper list: `SiluRef`/`SwigluLimitRef` are intentionally not duplicated — `MoeRef`/`DenseMlpRef` call `lsxcommon::gigachat::dev::SwigluLimitElement` (the single-source-of-truth reference shared with the kernels, matching the per-layer tests). `LmHeadRef` is the final `GemvRef(m.lm_head, xb, vocab, hidden)` inside `RunModelRef`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_gigachat_moe_parity.cu`:

```cpp
// TDD — GigaChat3.5 whole-stack parity (Plan 3 pre-work). Converts the
// synthetic checkpoint and compares GcPipeline::Step logits against an
// independent host-F32 reference (`gigachat_host_ref.h`) over a 2-token
// sequence. GPU-gated (CTest SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"

#include "gigachat_host_ref.h"

#include "gigachat_moe_constants.h"
#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_pipeline.h"
#include "lsxcommon/gigachat_moe_load.h"
#include "lsxcommon/schema.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat;
using namespace gigachatref;

namespace {

void ForceSyntheticGeometry()
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
#endif
}

std::string LayerFile(int l, const char *suffix)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "layer_%03d%s", l, suffix);
    return std::string(buf);
}

int ArgMax(const std::vector<float> &v)
{
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

float MaxAbs(const std::vector<float> &v)
{
    float m = 0.0f;
    for (float f : v) m = std::fmax(m, std::fabs(f));
    return m;
}

RefModel LoadRefModel(const std::filesystem::path &root)
{
    const auto base = root / "logestix";
    RefModel m;
    m.hidden = gigachattest::kTH;
    m.vocab = gigachattest::kTVocab;
    m.capacity = 2;
    m.eps = 1e-6f;
    m.num_experts = gigachattest::kTExperts;
    m.top_k = 1;
    m.moe_intermediate = gigachattest::kTMoeI;
    m.first_k_dense = 1;
    m.routed_scale = 2.5f;
    m.swiglu_limit = 10.0f;
    // {hidden, heads, q_lora, kv_lora, qk_nope, rope, v_head}
    m.mla_d = {gigachattest::kTH, gigachattest::kTHeads, 128, 128, 128, 64, 128};
    // {hidden, k_heads, v_heads, hk, hv, conv_k}
    m.gdn_d = {gigachattest::kTH, 1, 2, 128, 128, 4};

    auto gl = lsxcommon::LoadTensorBatchViews(base / "pinned" / "global.arrow");
    REQUIRE(gl.ok());
    m.embed = Bf16ToF32(FindTv(*gl, "model.embed_tokens.weight")->data,
                        static_cast<int64_t>(m.vocab) * m.hidden);
    m.lm_head = Bf16ToF32(FindTv(*gl, "lm_head.weight")->data,
                          static_cast<int64_t>(m.vocab) * m.hidden);
    m.final_norm = ReadNorm(*gl, "model.norm");

    for (int l = 0; l < gigachattest::kTLayers; ++l)
    {
        auto raw = lsxcommon::LoadTensorBatchViews(
            base / "pinned" / LayerFile(l, ".arrow"));
        auto nv = lsxcommon::LoadDenseNvfp4BatchViews(
            base / "pinned" / LayerFile(l, "_nvfp4.arrow"));
        REQUIRE(raw.ok());
        REQUIRE(nv.ok());
        const std::string lp = "layers." + std::to_string(l) + ".";
        RefLayer L;
        L.full_attention = (l == 1);
        L.moe = (l >= 1);
        L.input_norm = ReadNorm(*raw, lp + "input_layernorm");
        L.post_attn_norm = ReadNorm(*raw, lp + "post_attention_layernorm");
        L.post_self_attn_norm = ReadNorm(*raw, lp + "post_self_attn_layernorm");
        L.post_ffw_norm = ReadNorm(*raw, lp + "post_feedforward_layernorm");
        if (L.full_attention)
        {
            L.mla.q_a_norm = ReadNorm(*raw, lp + "self_attn.q_a_layernorm");
            L.mla.kv_a_norm = ReadNorm(*raw, lp + "self_attn.kv_a_layernorm");
            L.mla.q_a = ReadNv(*nv, lp + "self_attn.q_a_proj.weight");
            L.mla.q_b = ReadNv(*nv, lp + "self_attn.q_b_proj.weight");
            L.mla.kv_a = ReadNv(*nv, lp + "self_attn.kv_a_proj_with_mqa.weight");
            L.mla.kv_b = ReadNv(*nv, lp + "self_attn.kv_b_proj.weight");
            L.mla.o_proj = ReadNv(*nv, lp + "self_attn.o_proj.weight");
            L.mla.gate = ReadNv(*nv, lp + "self_attn.gate_proj.weight");
            L.k_cache.assign(static_cast<size_t>(m.capacity) * m.mla_d.heads *
                                 (m.mla_d.qk_nope + m.mla_d.rope),
                             0.0f);
            L.v_cache.assign(static_cast<size_t>(m.capacity) * m.mla_d.heads *
                                 m.mla_d.v_head,
                             0.0f);
        }
        else
        {
            const TensorView *conv = FindTv(*raw, lp + "self_attn.conv1d.weight");
            const TensorView *alog = FindTv(*raw, lp + "self_attn.A_log");
            const TensorView *dtb = FindTv(*raw, lp + "self_attn.dt_bias");
            const TensorView *ba = FindTv(*raw, lp + "self_attn.in_proj_ba.weight");
            const TensorView *gn = FindTv(*raw, lp + "self_attn.norm.weight");
            L.gdn.in_qkvz = ReadNv(*nv, lp + "self_attn.in_proj_qkvz.weight");
            L.gdn.out_proj = ReadNv(*nv, lp + "self_attn.out_proj.weight");
            L.gdn.conv = Bf16ToF32(conv->data, conv->shape[0] * conv->shape[2]);
            L.gdn.A_log = Bf16ToF32(alog->data, alog->shape[0]);
            L.gdn.dt_bias = Bf16ToF32(dtb->data, dtb->shape[0]);
            L.gdn.in_ba = Bf16ToF32(ba->data, ba->shape[0] * ba->shape[1]);
            L.gdn.norm = Bf16ToF32(gn->data, gn->shape[0]);
            L.gdn.conv_dim = static_cast<int>(conv->shape[0]);
            L.gdn.conv_k = static_cast<int>(conv->shape[2]);
            L.gdn.ba_rows = static_cast<int>(ba->shape[0]);
            L.gdn.v_heads = static_cast<int>(alog->shape[0]);
            L.conv_state.assign(
                static_cast<size_t>(L.gdn.conv_dim) * (L.gdn.conv_k - 1), 0.0f);
            L.rec_state.assign(
                static_cast<size_t>(L.gdn.v_heads) * m.gdn_d.hk * m.gdn_d.hv,
                0.0f);
        }
        if (L.moe)
        {
            const TensorView *router = FindTv(*raw, lp + "mlp.gate.weight");
            const TensorView *rb = FindTv(
                *raw, lp + "mlp.gate.e_score_correction_bias");
            L.moe_w.router =
                Bf16ToF32(router->data, router->shape[0] * router->shape[1]);
            L.moe_w.router_bias = F32Copy(rb->data, rb->shape[0]);
            L.moe_w.num_experts = static_cast<int>(router->shape[0]);
            L.moe_w.top_k = 1;
            L.moe_w.intermediate = gigachattest::kTMoeI;
            L.moe_w.shared_gate =
                ReadNv(*nv, lp + "mlp.shared_experts.gate_proj.weight");
            L.moe_w.shared_up =
                ReadNv(*nv, lp + "mlp.shared_experts.up_proj.weight");
            L.moe_w.shared_down =
                ReadNv(*nv, lp + "mlp.shared_experts.down_proj.weight");
        }
        else
        {
            L.dense.gate = ReadNv(*nv, lp + "mlp.gate_proj.weight");
            L.dense.up = ReadNv(*nv, lp + "mlp.up_proj.weight");
            L.dense.down = ReadNv(*nv, lp + "mlp.down_proj.weight");
        }
        m.layers.push_back(std::move(L));
    }
    const float mscale =
        0.1f * std::log(static_cast<float>(frozen::kYarnFactor)) + 1.0f;
    m.attn_scale = mscale * mscale /
                   std::sqrt(static_cast<float>(m.mla_d.qk_nope + m.mla_d.rope));
    return m;
}

void AttachRefExperts(RefModel &m, GcExpertStore &store)
{
    for (int l = 0; l < gigachattest::kTLayers; ++l)
    {
        RefLayer &L = m.layers[static_cast<size_t>(l)];
        if (!L.moe) continue;
        const int E = L.moe_w.num_experts;
        const int I = L.moe_w.intermediate;
        const int H = m.hidden;
        L.moe_w.experts.assign(static_cast<size_t>(E), RefExpert{});
        for (int e = 0; e < E; ++e)
        {
            const GcExpertEntry &ent =
                store.entries[static_cast<size_t>(l) * E + e];
            RefExpert ex;
            ex.gate = {I, H, DequantNvfp4(ent.host_w1, ent.host_w1_scales,
                                          ent.w1_global_scale,
                                          static_cast<int64_t>(I) * H)};
            ex.up = {I, H, DequantNvfp4(ent.host_w3, ent.host_w3_scales,
                                        ent.w3_global_scale,
                                        static_cast<int64_t>(I) * H)};
            ex.down = {H, I, DequantNvfp4(ent.host_w2, ent.host_w2_scales,
                                          ent.w2_global_scale,
                                          static_cast<int64_t>(H) * I)};
            L.moe_w.experts[static_cast<size_t>(e)] = std::move(ex);
        }
    }
}

} // namespace

TEST_CASE("gigachat whole-stack step matches the host reference",
          "[gigachat][inference]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);
    ForceSyntheticGeometry();

    const auto src = gigachattest::MakeTempDir("gc_par_src");
    const auto root = gigachattest::MakeTempDir("gc_par_dst");
    gigachattest::WriteGigaChatCheckpoint(src);
    const auto st = ConvertGigaChatModel(src.string(), root.string());
    INFO("convert: " << st.ToString());
    REQUIRE(st.ok());

    RefModel ref = LoadRefModel(root);

    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(root.string(), idx, err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(GcBuildYarnTables(2, ref.mla_d.rope, ref.rc, ref.rs));

    GcWeights w;
    REQUIRE(LoadGigaChatWeights(root.string(), idx, w, stream, err));
    GcExpertCache cache;
    REQUIRE(InitGigaChatExpertCache(root.string(), idx.num_layers,
                                    idx.first_k_dense_replace, idx.num_experts,
                                    idx.num_experts_per_tok, 2, stream, cache,
                                    err));
    AttachRefExperts(ref, cache.store);

    GcPipelineDims dims;
    dims.mla = ref.mla_d;
    dims.gdn = ref.gdn_d;
    dims.vocab = ref.vocab;
    dims.num_experts = ref.num_experts;
    dims.top_k = ref.top_k;
    dims.moe_intermediate = ref.moe_intermediate;
    dims.first_k_dense = ref.first_k_dense;
    dims.rms_eps = ref.eps;
    dims.routed_scale = ref.routed_scale;
    dims.swiglu_limit = ref.swiglu_limit;

    GcPipeline pipe;
    REQUIRE(pipe.Init(w, dims, 2, err));
    REQUIRE(pipe.Reset(stream) == cudaSuccess);

    float *d_logits = nullptr;
    REQUIRE(cudaMalloc(&d_logits, static_cast<size_t>(ref.vocab) * 4) ==
            cudaSuccess);

    const std::vector<int> tokens = {1, 2};
    for (int pos = 0; pos < 2; ++pos)
    {
        std::vector<float> gpu(static_cast<size_t>(ref.vocab));
        REQUIRE(pipe.Step(w, cache, tokens[static_cast<size_t>(pos)], pos,
                          d_logits, stream) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        REQUIRE(cudaMemcpy(gpu.data(), d_logits,
                           static_cast<size_t>(ref.vocab) * 4,
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        std::vector<float> host =
            RunModelRef(ref, std::vector<int>(tokens.begin(),
                                              tokens.begin() + pos + 1));
        REQUIRE(host.size() == gpu.size());
        // Error budget: the GPU GEMMs read F16 activations and store F16
        // outputs (FP32 accumulate, lsx_inference_nvfp4.cu:112), while the
        // reference computes the same dequantized weights in F32. 5e-2
        // relative is the F16-rounding envelope over two layers; the narrow
        // gates stay with the per-layer attn tests (bit-exact rope, 1e-3).
        const float tol = 5e-2f * std::fmax(1.0f, MaxAbs(host));
        float worst = 0.0f;
        for (size_t i = 0; i < host.size(); ++i)
            worst = std::fmax(worst, std::fabs(host[i] - gpu[i]));
        INFO("pos=" << pos << " argmax gpu=" << ArgMax(gpu)
                    << " ref=" << ArgMax(host) << " worst=" << worst
                    << " tol=" << tol);
        CHECK(ArgMax(gpu) == ArgMax(host));
        CHECK(worst <= tol);
    }

    cudaFree(d_logits);
    pipe.Free();
    FreeGcExpertCache(cache);
    w.Free();
    cudaStreamDestroy(stream);
}
```

- [ ] **Step 2: Add the new target to CMake and label it**

In `tests/CMakeLists.txt`, immediately after the `test_gigachat_moe_attn` block (ends at `set_tests_properties(test_gigachat_moe_attn PROPERTIES SKIP_RETURN_CODE 4)`, currently line 3160), insert:

```cmake
# test_gigachat_moe_parity — whole-stack synthetic parity vs a host-F32
# reference. Same standalone .cu object pattern.
add_library(test_gigachat_moe_parity_cu OBJECT test_gigachat_moe_parity.cu)
set_target_properties(test_gigachat_moe_parity_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_gigachat_moe_parity_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    -fmad=false;-Xcompiler=-ffp-contract=off;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_gigachat_moe_parity_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include"
    "${CMAKE_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${CMAKE_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${CMAKE_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_gigachat_moe_parity_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_gigachat_moe_parity_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_gigachat_moe_parity)
set_target_properties(test_gigachat_moe_parity PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_gigachat_moe_parity PRIVATE
    test_gigachat_moe_parity_cu lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_gigachat_moe_parity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_gigachat_moe_parity>")
endif()
add_test(NAME test_gigachat_moe_parity COMMAND test_gigachat_moe_parity)
set_tests_properties(test_gigachat_moe_parity PROPERTIES SKIP_RETURN_CODE 4)
```

Then add `test_gigachat_moe_parity` to the `inference` label list (the `set_tests_properties(... PROPERTIES LABELS inference)` block that currently contains `test_gigachat_moe_attn` at line 3644), immediately after `test_gigachat_moe_attn,`:

```cmake
    test_gigachat_moe_parity
```

- [ ] **Step 3: Build to verify it fails on the missing stack reference**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_parity 2>&1 | tail -25`
Expected: FAIL — `RefModel` / `RunModelRef` / `DequantNvfp4` member types not declared (Task 3 header not yet appended).

- [ ] **Step 4: Append the stack reference to `tests/gigachat_host_ref.h`**

Before the closing `} // namespace gigachatref`, add (and move the closing brace after it):

```cpp
struct RefExpert {
    RefNvfp4 gate, up, down;
};

struct RefMoe {
    std::vector<float> router;       // [E, hidden]
    std::vector<float> router_bias;  // [E]
    int num_experts = 0, top_k = 0, intermediate = 0;
    RefNvfp4 shared_gate, shared_up, shared_down;
    std::vector<RefExpert> experts;  // [E]
};

struct RefDense {
    RefNvfp4 gate, up, down;
};

struct RefRoute {
    std::vector<int> idx;
    std::vector<float> w;
};

// sigmoid router: logits = x.W^T; scores = sigmoid(logits);
// choice = scores + bias; top-k (lowest index wins ties);
// w = scores[idx] / (sum + 1e-20) * routed_scaling_factor.
inline RefRoute RouterTopKRef(const std::vector<float> &x,
                              const std::vector<float> &W,
                              const std::vector<float> &bias, int E, int hidden,
                              int k, float routed_scale)
{
    std::vector<float> scores(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e)
    {
        float a = 0.0f;
        for (int i = 0; i < hidden; ++i)
            a += W[static_cast<size_t>(e) * hidden + i] * x[static_cast<size_t>(i)];
        scores[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-a));
    }
    std::vector<int> order(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) order[static_cast<size_t>(e)] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float ca = scores[static_cast<size_t>(a)] + bias[static_cast<size_t>(a)];
        const float cb = scores[static_cast<size_t>(b)] + bias[static_cast<size_t>(b)];
        if (ca != cb) return ca > cb;
        return a < b;
    });
    RefRoute r;
    r.idx.assign(order.begin(), order.begin() + k);
    r.w.assign(static_cast<size_t>(k), 0.0f);
    float sum = 0.0f;
    for (int j = 0; j < k; ++j)
    {
        r.w[static_cast<size_t>(j)] = scores[static_cast<size_t>(r.idx[static_cast<size_t>(j)])];
        sum += r.w[static_cast<size_t>(j)];
    }
    for (int j = 0; j < k; ++j)
        r.w[static_cast<size_t>(j)] =
            r.w[static_cast<size_t>(j)] / (sum + 1e-20f) * routed_scale;
    return r;
}

inline std::vector<float> MoeRef(const RefMoe &m,
                                 const std::vector<float> &x_norm, int hidden,
                                 float routed_scale, float swiglu_limit)
{
    std::vector<float> x16(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) x16[i] = R16(x_norm[i]);
    const RefRoute r = RouterTopKRef(x_norm, m.router, m.router_bias,
                                     m.num_experts, hidden, m.top_k,
                                     routed_scale);
    const int I = m.intermediate;
    std::vector<float> mlp(static_cast<size_t>(hidden), 0.0f);
    for (int j = 0; j < m.top_k; ++j)
    {
        const RefExpert &e = m.experts[static_cast<size_t>(r.idx[static_cast<size_t>(j)])];
        std::vector<float> gt = GemvRef(e.gate.w, x16, I, hidden);
        for (float &f : gt) f = R16(f);
        std::vector<float> up = GemvRef(e.up.w, x16, I, hidden);
        for (float &f : up) f = R16(f);
        std::vector<float> mid(static_cast<size_t>(I));
        for (int i = 0; i < I; ++i)
            mid[static_cast<size_t>(i)] =
                R16(lsxcommon::gigachat::dev::SwigluLimitElement(
                    gt[static_cast<size_t>(i)], up[static_cast<size_t>(i)],
                    swiglu_limit));
        std::vector<float> dn = GemvRef(e.down.w, mid, hidden, I);
        for (float &f : dn) f = R16(f);
        for (int i = 0; i < hidden; ++i)
            mlp[static_cast<size_t>(i)] += r.w[static_cast<size_t>(j)] * dn[static_cast<size_t>(i)];
    }
    std::vector<float> sg = GemvRef(m.shared_gate.w, x16, I, hidden);
    for (float &f : sg) f = R16(f);
    std::vector<float> su = GemvRef(m.shared_up.w, x16, I, hidden);
    for (float &f : su) f = R16(f);
    std::vector<float> sm(static_cast<size_t>(I));
    for (int i = 0; i < I; ++i)
        sm[static_cast<size_t>(i)] = R16(lsxcommon::gigachat::dev::SwigluLimitElement(
            sg[static_cast<size_t>(i)], su[static_cast<size_t>(i)],
            swiglu_limit));
    std::vector<float> sd = GemvRef(m.shared_down.w, sm, hidden, I);
    for (float &f : sd) f = R16(f);
    for (int i = 0; i < hidden; ++i) mlp[static_cast<size_t>(i)] += sd[static_cast<size_t>(i)];
    return mlp;
}

inline std::vector<float> DenseMlpRef(const RefDense &dm,
                                      const std::vector<float> &x_norm,
                                      int hidden, int inter, float swiglu_limit)
{
    std::vector<float> x16(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) x16[i] = R16(x_norm[i]);
    std::vector<float> gt = GemvRef(dm.gate.w, x16, inter, hidden);
    for (float &f : gt) f = R16(f);
    std::vector<float> up = GemvRef(dm.up.w, x16, inter, hidden);
    for (float &f : up) f = R16(f);
    std::vector<float> mid(static_cast<size_t>(inter));
    for (int i = 0; i < inter; ++i)
        mid[static_cast<size_t>(i)] = R16(lsxcommon::gigachat::dev::SwigluLimitElement(
            gt[static_cast<size_t>(i)], up[static_cast<size_t>(i)],
            swiglu_limit));
    std::vector<float> dn = GemvRef(dm.down.w, mid, hidden, inter);
    for (float &f : dn) f = R16(f);
    return dn;
}

struct RefLayer {
    bool full_attention = false, moe = false;
    RefNorm input_norm, post_attn_norm, post_self_attn_norm, post_ffw_norm;
    RefMla mla;
    RefGdn gdn;
    RefMoe moe_w;
    RefDense dense;
    std::vector<float> k_cache, v_cache, conv_state, rec_state;
};

struct RefModel {
    int hidden = 0, vocab = 0, capacity = 2;
    std::vector<float> embed, lm_head;
    RefNorm final_norm;
    std::vector<RefLayer> layers;
    GcMlaDims mla_d;
    GcGdnDims gdn_d;
    float eps = 1e-6f;
    float attn_scale = 0.0f;
    float routed_scale = 2.5f;
    float swiglu_limit = 10.0f;
    int num_experts = 0, top_k = 0, moe_intermediate = 0, first_k_dense = 0;
    std::vector<float> rc, rs;  // YaRN tables [capacity, rope/2]
};

inline std::vector<float> GatedNormRef(const std::vector<float> &x,
                                       const RefNorm &n, float eps)
{
    return NormRef(x, n.w, n.up, n.down, n.dim, n.rank, eps);
}

inline std::vector<float> RunModelRef(RefModel &m,
                                      const std::vector<int> &tokens)
{
    for (RefLayer &l : m.layers)
    {
        std::fill(l.k_cache.begin(), l.k_cache.end(), 0.0f);
        std::fill(l.v_cache.begin(), l.v_cache.end(), 0.0f);
        std::fill(l.conv_state.begin(), l.conv_state.end(), 0.0f);
        std::fill(l.rec_state.begin(), l.rec_state.end(), 0.0f);
    }
    std::vector<float> logits;
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos)
    {
        std::vector<float> h(static_cast<size_t>(m.hidden));
        for (int i = 0; i < m.hidden; ++i)
            h[static_cast<size_t>(i)] = RBf16(
                m.embed[static_cast<size_t>(tokens[static_cast<size_t>(pos)]) * m.hidden + i]);
        for (RefLayer &L : m.layers)
        {
            const std::vector<float> res = h;
            const std::vector<float> ni = GatedNormRef(h, L.input_norm, m.eps);
            std::vector<float> attn;
            if (L.full_attention)
                MlaDecodeRef(L.mla, m.mla_d, m.eps, m.attn_scale, m.rc, m.rs,
                             pos, L.k_cache, L.v_cache, ni, attn);
            else
                GdnDecodeRef(L.gdn, m.gdn_d, m.eps,
                             lsxcommon::gigachat::frozen::kLinearSigmoidGateScale,
                             L.conv_state, L.rec_state, ni, attn);
            const std::vector<float> g1 =
                GatedNormRef(attn, L.post_self_attn_norm, m.eps);
            h.resize(static_cast<size_t>(m.hidden));
            for (int i = 0; i < m.hidden; ++i)
                h[static_cast<size_t>(i)] = res[static_cast<size_t>(i)] + g1[static_cast<size_t>(i)];

            const std::vector<float> res2 = h;
            const std::vector<float> np = GatedNormRef(h, L.post_attn_norm, m.eps);
            const std::vector<float> mlp =
                L.moe ? MoeRef(L.moe_w, np, m.hidden, m.routed_scale,
                               m.swiglu_limit)
                      : DenseMlpRef(L.dense, np, m.hidden, L.dense.gate.rows,
                                    m.swiglu_limit);
            const std::vector<float> g2 =
                GatedNormRef(mlp, L.post_ffw_norm, m.eps);
            for (int i = 0; i < m.hidden; ++i)
                h[static_cast<size_t>(i)] = res2[static_cast<size_t>(i)] + g2[static_cast<size_t>(i)];
        }
        const std::vector<float> fn = GatedNormRef(h, m.final_norm, m.eps);
        std::vector<float> xb(static_cast<size_t>(m.hidden));
        for (int i = 0; i < m.hidden; ++i) xb[static_cast<size_t>(i)] = RBf16(fn[static_cast<size_t>(i)]);
        logits = GemvRef(m.lm_head, xb, m.vocab, m.hidden);
    }
    return logits;
}
```

Add `#include "gigachat_moe_constants.h"` at the top of the header (after `gigachat_moe_device_math.h`) so `lsxcommon::gigachat::frozen::kLinearSigmoidGateScale` resolves.

- [ ] **Step 5: Build and run the parity test**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_parity && ./build/linux-debug/tests/test_gigachat_moe_parity`
Expected: PASS (both positions; `argmax` equal and `worst <= tol`).

If it fails, first suspect a reference/loader wiring mismatch, not the kernel: dump `INFO("pos=... worst=...")` and compare per-layer outputs by temporarily truncating `RunModelRef` to layer 0 only. Do not loosen the tolerance before the wiring is confirmed.

- [ ] **Step 6: Run the guard tests**

Run: `ctest --test-dir build/linux-debug -R "gigachat|guard" --output-on-failure`
Expected: all pass (including `test_ctest_labels_guard` and `test_model_suite_guard`).

- [ ] **Step 7: Commit**

```bash
git add tests/gigachat_host_ref.h tests/test_gigachat_moe_parity.cu tests/CMakeLists.txt
git commit -m "test: whole-stack GigaChat3.5 parity vs host-F32 reference"
```

---

## Verification (all tasks)

```bash
cmake --build --preset linux-debug --target test_gigachat_moe_attn test_gigachat_moe_parity
./build/linux-debug/tests/test_gigachat_moe_attn
./build/linux-debug/tests/test_gigachat_moe_parity
ctest --test-dir build/linux-debug -R "gigachat|guard" --output-on-failure
```

## Out of scope

- Real-weight SelfCheck and perf baseline (Plan 3 proper; gated on the operator's 416 GB `lsxforge` conversion — see `docs/superpowers/specs/2026-09-20-gigachat35-validation-design.md` Part B).
- Any production code change: `GcPipeline` and the kernels are not touched.
