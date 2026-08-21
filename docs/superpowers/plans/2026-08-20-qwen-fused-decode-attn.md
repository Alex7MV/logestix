# Qwen Fused Decoder Attention Layer — Implementation Plan (3-stage)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 9-launch seq=1 full-attention chain with **3 fused kernels on one stream** (`fused_qkv` → `fused_attn` → `fused_o_proj`), removing ~8 intermediate scratch round-trips while matching the split path within tolerance.

**Architecture:** Three per-head (grid=NH, block=HD) kernels replace the split launches for the single-token decode. Stage 1 computes q/k/v+gate inline (NVFP4 dequant to registers), per-head RMSNorm, inline RoPE, and appends k/v to the persistent KV. Stage 2 runs block-wide online-softmax GQA attention over the causal KV and applies the gate. Stage 3 accumulates each head's o_proj into `d_out` via cross-block `atomicAdd` on a float accumulator.

**Tech Stack:** C++20, CUDA, NVFP4 codec (`nvfp4_codec.h`), Catch2. Numeric: FP32, tolerance `1e-2` vs the split path.

---

## File map

- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare `RunQwenFusedDecodeAttnKernel`.
- Modify: `libs/lsxcommon/src/qwen_inference.cu` — three fused kernels + the public composition entry (external linkage).
- Create: `tests/test_qwen_fused_decode_attn.cu` — per-stage parity + full-layer parity vs the split path.
- Modify: `tests/CMakeLists.txt` — add `.cu` to `test_qwen_cu` OBJECT + foreach list.
- No codec/other-CMake change.

## Grounding (read first)

- Split reference `RunQwenFullAttention` (`qwen_inference.cu:740-821`): launches in §1. Its math is the reference each stage must match within tolerance.
- **NVFP4 weight layout** (`qwen_inference_load.h:16-30`): `QwenNvfp4Weight{rows=K, cols=N, d_packed nibble-E2M1 row-major [K,N], d_scales per-16 E4M3, global_scale}`. Element `e = n*K + k` → `byte=e>>1`, `nib=(e&1)?byte>>4:byte&0xF`, scale `E4M3ScaleToFloat(scales[e>>4])*global_scale`. `kE2M1Table`/`E4M3ScaleToFloat` aliased in the anonymous namespace (`qwen_inference.cu:340`).
- **Column layout** (verified from `qwen_inference_load.h:57-60` + `qwen_inference.cu:524-528`):
  - `q` weight `cols = NH*HD*2`: query = first `NH*HD` cols, **gate = second `NH*HD` cols** (i.e. gate colidx `= NH*HD + h*HD + d`).
  - `k` weight `cols = NKV*HD`: per kvh colidx `kvh*HD+d`.
  - `v` weight `cols = NKV*HD`: same.
  - `o` weight: `rows = NH*HD` (input K), `cols = H` (output N). `o[ (h*HD+d)*H + n ]` contributes to `d_out[n]`.
- **Per-head RMSNorm** (offset form, `qwen_inference.cu:531-544` for q, :548+ for k): `v * inv * (1 + norm[d])`, `inv = rsqrtf(rms2/HD + eps)`.
- **RoPE** (`:100-116` cos/sin, `:118+` partial_rope): rotate-half pairs `(j, j+rot/2)` for `j in [0, rotary_dim/2)`; passes rest through. `ang = pos * theta^-(2j/rotary_dim)`.
- **Attention** — reuse the step-2 validated `full_attn_softmax_kernel` math (block-reduce `sreduce[512]` static + dynamic `slogits`), applied over attended KV rows `[0..kv_base]`.
- **Gate**: `y = x * sigmoid(gate)` (`:371`).
- **External-entry pattern**: `RunQwenNvfp4Gemm`/`RunQwenFullAttnKernel` — declared in header, defined in `lsxcommon::qwen` (outside anonymous ns).
- **Test registration**: add `.cu` to `test_qwen_cu` OBJECT + foreach list; Arrow include dirs already present.

---

## Task 1: Public entry + split-reference harness (RED→GREEN)

**Files:**
- Create: `tests/test_qwen_fused_decode_attn.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h`
- Modify: `libs/lsxcommon/src/qwen_inference.cu`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Register the test in CMake**

Add `test_qwen_fused_decode_attn.cu` to the `test_qwen_cu` OBJECT lib and to the `foreach(qt ...)` list next to the other Qwen CUDA tests.

- [ ] **Step 2: Write the failing test**

Create `tests/test_qwen_fused_decode_attn.cu`. It builds a tiny synthetic full-attention layer, runs the **split** path as the reference, and calls `RunQwenFusedDecodeAttnKernel` (not yet declared → RED). The committed test:

```cpp
// Fused decoder (seq=1) full-attention 3-stage parity vs the split path.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "qwen_test_fixture.h"

namespace {

int64_t nvfp4_packed_bytes(int K, int N) { return (int64_t)K * N / 2; }
int64_t nvfp4_scale_bytes(int K, int N)  { return (int64_t)K * N / 16; }
int64_t nvfp4_elems(int K, int N)        { return (int64_t)K * N; }

// Deterministic NVFP4 fill (must mirror production decode: E4M3 scales).
void FillNvfp4(QwenNvfp4Weight &w, int K, int N, uint8_t seed_pack, uint8_t seed_scale)
{
    w.rows = K; w.cols = N;
    w.packed_bytes = nvfp4_packed_bytes(K, N);
    w.scale_bytes  = nvfp4_scale_bytes(K, N);
    w.global_scale = 0.5f;
    cudaMalloc(&w.d_packed, (size_t)w.packed_bytes);
    cudaMalloc(&w.d_scales, (size_t)w.scale_bytes);
    std::vector<uint8_t> pk((size_t)w.packed_bytes), sc((size_t)w.scale_bytes);
    for (int64_t i = 0; i < w.packed_bytes; ++i) pk[i] = (uint8_t)((i * 7 + seed_pack) & 0xFF);
    for (int64_t i = 0; i < w.scale_bytes; ++i)  sc[i] = (uint8_t)(128 + ((i + seed_scale) % 32));
    cudaMemcpy((void*)w.d_packed, pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)w.d_scales, sc.data(), sc.size(), cudaMemcpyHostToDevice);
}

// Build a tiny full-attention layer: NH=2, NKV=1, HD=8, H=16, rotary_dim=4.
QwenLayerWeights MakeTinyFullLayer(int NH, int NKV, int HD, int H, int rotary_dim,
                                   const __half *h_input_norm)
{
    QwenLayerWeights lw;
    lw.is_linear = false;
    lw.loaded = true;
    lw.norm_size = H;
    FillNvfp4(lw.q, H, NH*HD*2, 1, 2);
    FillNvfp4(lw.k, H, NKV*HD, 3, 4);
    FillNvfp4(lw.v, H, NKV*HD, 5, 6);
    FillNvfp4(lw.o, NH*HD, H, 7, 8);
    lw.q_norm_size = HD; lw.k_norm_size = HD;
    cudaMalloc(&lw.d_q_norm, (size_t)HD*2);
    cudaMalloc(&lw.d_k_norm, (size_t)HD*2);
    cudaMalloc(&lw.d_input_norm, (size_t)H*2);
    std::vector<__half> qn(HD), kn(HD);
    for (int i = 0; i < HD; ++i) { qn[i] = __float2half(0.1f * ((i % 3))); kn[i] = __float2half(0.05f*((i%5))); }
    cudaMemcpy(lw.d_q_norm, qn.data(), HD*2, cudaMemcpyHostToDevice);
    cudaMemcpy(lw.d_k_norm, kn.data(), HD*2, cudaMemcpyHostToDevice);
    cudaMemcpy(lw.d_input_norm, h_input_norm, H*2, cudaMemcpyHostToDevice);
    return lw;
}

void FreeTinyLayer(QwenLayerWeights &lw)
{
    lw.q.Free(); lw.k.Free(); lw.v.Free(); lw.o.Free();
    cudaFree(lw.d_q_norm); cudaFree(lw.d_k_norm); cudaFree(lw.d_input_norm);
}

}  // namespace
```

Plus the TEST_CASE (fill x/pos, run split vs fused, assert tolerance) — the exact body is in Task 4 (full parity). For Task 1 the test calls `RunQwenFusedDecodeAttnKernel` and asserts the symbol exists; the full parity executes once all three stages exist (Tasks 2-4). **Committed Task-1 test** asserts `RunQwenFusedDecodeAttnKernel(...) == cudaSuccess` on a valid tiny setup.

- [ ] **Step 3: Header declaration**

```cpp
// Fused decoder (seq=1) full-attention layer: 3 fused kernels (qkv -> online
// softmax -> o_proj) replace the ~9 split launches for a SINGLE token. x is the
// embedded token [H]; kv_base appends its k/v and bounds attended rows [0..kv_base].
// kv_k/kv_v are the persistent cache ([max_total, NKV, HD] rows). out [H].
cudaError_t RunQwenFusedDecodeAttnKernel(
    cudaStream_t stream, const __half *x, const QwenLayerWeights &lw,
    const QwenTopology &top, int kv_base,
    const __half *kv_k, const __half *kv_v, __half *out);
```

- [ ] **Step 4: Initial implementation (delegate to split path)** — so the harness symbol-et exists and can be incrementally replaced. The committed Task-1 body calls `RunQwenFullAttention` with seq=1 into a scratch arena, matching the split path exactly (proves the harness wiring; Tasks 2-4 swap in the fused kernels). **Note:** this requires the arena the test provides; the test must allocate the same scratch the generation path uses. Because that is heavy, the committed Task-1 body instead implements **stage 1 only as a calling shell that invokes the existing public `RunQwenNvfp4Gemm`/`RunQwenFullAttnKernel` split launches** for q/k/v/attention/o — i.e., a reference-composition, then Tasks 2-4 replace with the true fused kernels. This is still a real entry compiling and matching the split (parity holds through all tasks).

**Authoritative Task-1 body** (composition via existing public kernels; replaced stage-by-stage in Tasks 2-4):

```cpp
cudaError_t RunQwenFusedDecodeAttnKernel(
    cudaStream_t stream, const __half *x, const QwenLayerWeights &lw,
    const QwenTopology &top, int kv_base,
    const __half *kv_k, const __half *kv_v, __half *out)
{
    if (!x || !out || !kv_k || !kv_v || kv_base < 0 || !lw.loaded)
        return cudaErrorInvalidValue;
    const int NH = top.num_attention_heads, HD = top.head_dim,
              NKV = top.num_key_value_heads, H = top.hidden_size;
    if (NH<=0 || HD<=0 || NKV<=0 || H<=0) return cudaErrorInvalidValue;

    // scratch (half buffers) — sized like the generation arena for seq=1.
    __half *d_normed, *d_qc, *d_q, *d_gate, *d_k, *d_v, *d_attn, *d_cos, *d_sin;
    int32_t *d_pos;
    auto fail = [&](cudaError_t e) { return e; };
    cudaMalloc(&d_normed,H*2);
    cudaMalloc(&d_qc,NH*HD*2*2);
    cudaMalloc(&d_q,NH*HD*2);
    cudaMalloc(&d_gate,NH*HD*2);
    cudaMalloc(&d_k,(kv_base+1)*NKV*HD*2);
    cudaMalloc(&d_v,(kv_base+1)*NKV*HD*2);
    cudaMalloc(&d_attn,NH*HD*2);
    cudaMalloc(&d_cos, (size_t)(int)(HD*top.partial_rotary_factor/2)*2);
    cudaMalloc(&d_sin, (size_t)(int)(HD*top.partial_rotary_factor/2)*2);
    cudaMalloc(&d_pos, sizeof(int32_t));

    // input RMSNorm
    const int th = std::min(H,256);
    rmsnorm_offset_kernel<<<1, th, th*sizeof(float), stream>>>(
        x, lw.d_input_norm ? lw.d_input_norm : __half(0.0f), d_normed, 1, H, (float)top.rms_norm_eps);
    cudaError_t e = cudaGetLastError(); if (e) return e;
    // q/k/v projections (into split buffer layout)
    e = RunQwenNvfp4Gemm(stream, d_normed, lw.q.d_packed, lw.q.d_scales, lw.q.global_scale,
                         d_qc, 1, H, NH*HD*2); if (e) return e;
    e = RunQwenNvfp4Gemm(stream, d_normed, lw.k.d_packed, lw.k.d_scales, lw.k.global_scale,
                         d_k, 1, H, NKV*HD); if (e) return e;
    e = RunQwenNvfp4Gemm(stream, d_normed, lw.v.d_packed, lw.v.d_scales, lw.v.global_scale,
                         d_v, 1, H, NKV*HD); if (e) return e;
    // split query + gate + q head-norm
    {
        const int threads = std::min(HD,512);
        qwen_split_query_kernel<<<NH, threads, HD*sizeof(float), stream>>>(
            d_qc, d_gate, d_q, nullptr, 1, NH, HD, lw.d_q_norm, lw.d_k_norm, (float)top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }
    // k head-norm in place
    {
        const int threads = std::min(HD,512);
        qwen_head_norm_kernel<<<NKV, threads, HD*sizeof(float), stream>>>(
            d_k, lw.d_k_norm, d_k, 1, NKV, HD, (float)top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }
    // RoPE cos/sin + partial rope (position from kv_base)
    {
        const int32_t pos = kv_base;
        cudaMemcpyAsync(d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        const int rot = (int)(HD*top.partial_rotary_factor);
        e = RunQwenRopeCosSin(stream, d_pos, 1, rot, top.rope_theta, d_cos, d_sin); if (e) return e;
        e = RunQwenPartialRope(stream, d_q, d_k, d_cos, d_sin, d_q, d_k, 1, NH, NKV, HD, rot); if (e) return e;
    }
    // attention (step-2 online softmax kernel)
    {
        const float scale = 1.0f / std::sqrt((float)HD);
        const dim3 grid(1, NH);
        const int threads = std::min(HD,512);
        const size_t smem = ((size_t)kv_base + 1 + 1) * sizeof(float);
        full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
            d_q, d_k, d_v, d_attn, 1, NH, HD, NKV, scale, kv_base);
        e = cudaGetLastError(); if (e) return e;
    }
    // gate
    {
        const int64_t n = (int64_t)NH*HD;
        qwen_out_gate_kernel<<<(int)((n+255)/256), 256, 0, stream>>>(d_attn, d_gate, d_attn, 1, n);
        e = cudaGetLastError(); if (e) return e;
    }
    // o_proj
    e = RunQwenNvfp4Gemm(stream, d_attn, lw.o.d_packed, lw.o.d_scales, lw.o.global_scale,
                         out, 1, NH*HD, H);

    cudaFree(d_normed); cudaFree(d_qc); cudaFree(d_q); cudaFree(d_gate);
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_attn); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_pos);
    return e;
}
```

This Task-1 composition uses existing public kernels + the in-file kernels, so it compiles and matches the split exactly (bit-identical, in fact). It is the reference the fused stages will replace in Tasks 2-4; the harness parity actually starts GREEN here (fused==split). The `rmsnorm_offset_kernel`/`qwen_split_query_kernel`/`qwen_head_norm_kernel`/`full_attn_softmax_kernel`/`qwen_out_gate_kernel` are all in the file scope (external or anonymous-visible) so this compiles from `lsxcommon::qwen`.

- [ ] **Step 5: Build + run**

`cmake --build build --config Debug --target test_qwen_fused_decode_attn` → builds; run `tests/test_qwen_fused_decode_attn.exe` → PASS (parity holds since it's a composition of the same split kernels).

- [ ] **Step 6: Commit**

`git add tests/test_qwen_fused_decode_attn.cu include/lsxcommon/qwen_inference.h src/qwen_inference.cu tests/CMakeLists.txt`
`git commit -m "test: RunQwenFusedDecodeAttnKernel public composition entry (split parity base)"`

---

## Task 2: Fused q/k/v stage (inline NVFP4 + head-norm + RoPE + KV append)

**Files:**
- Modify: `src/qwen_inference.cu`

- [ ] **Step 1: Add `fused_qkv_kernel`** (grid=NH, block=HD). Per head `h`:

```cpp
__global__ void fused_qkv_kernel(
    const __half* __restrict__ x,          // input token [H] (already input-normed)
    const QwenNvfp4Weight qw, const QwenNvfp4Weight kw, const QwenNvfp4Weight vw,
    const __half* __restrict__ q_norm, const __half* __restrict__ k_norm,
    double theta, int rotary_dim, int kv_base,
    int NH, int NKV, int HD, int H, float eps,
    __half* __restrict__ kv_k, __half* __restrict__ kv_v,
    __half* __restrict__ q_gl, __half* __restrict__ gate_gl)
{
    __shared__ float sq[512];  // pair staging for RoPE q (rotary_dim<=512 assumed)
    __shared__ float sk[512];
    __shared__ float red[512];
    const int h = blockIdx.x;
    const int kvh = h / (NH / NKV);
    const int d = threadIdx.x;

    // 1. q/k/v/gate projections for head h (inline NVFP4 per element).
    const int qcol = h*HD + d;             // query column
    const int gcol = NH*HD + h*HD + d;     // gate column (second NH*HD half)
    const int kcol = kvh*HD + d;           // key/value column
    float qv = 0.0f, gv = 0.0f, kv_ = 0.0f, vv = 0.0f;
    for (int k = 0; k < H; ++k) {
        const float xv = __half2float(x[k]);
        {
            const int64_t e = (int64_t)qcol * H + k;
            const uint8_t b = qw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            qv += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(qw.d_scales[e>>4]) * qw.global_scale);
        }
        {
            const int64_t e = (int64_t)gcol * H + k;
            const uint8_t b = qw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            gv += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(qw.d_scales[e>>4]) * qw.global_scale);
        }
        {
            const int64_t e = (int64_t)kcol * H + k;
            const uint8_t b = kw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            kv_ += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(kw.d_scales[e>>4]) * kw.global_scale);
        }
        {
            const int64_t e = (int64_t)kcol * H + k;
            const uint8_t b = vw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            vv += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(vw.d_scales[e>>4]) * vw.global_scale);
        }
    }
    // 2. per-head RMSNorm(q) and RMSNorm(k) — offset form.
    red[d] = qv*qv; __syncthreads();
    for (int st = HD/2; st>0; st>>=1){ if(d<st) red[d]+=red[d+st]; __syncthreads(); }
    const float qinv = rsqrtf(red[0]/(float)HD + eps);
    qv *= qinv * (1.0f + __half2float(q_norm[d]));
    red[d] = kv_*kv_; __syncthreads();
    for (int st = HD/2; st>0; st>>=1){ if(d<st) red[d]+=red[d+st]; __syncthreads(); }
    const float kinv = rsqrtf(red[0]/(float)HD + eps);
    kv_ *= kinv * (1.0f + __half2float(k_norm[d]));

    // 3. inline RoPE (rotate-half pairing) on first rotary_dim dims.
    //    Each thread computes ONLY its own dim `d`, reading the partner from
    //    the synced sq/sk staging (mirrors partial_rope_kernel :140-141,168-169).
    sq[d] = qv; sk[d] = kv_; __syncthreads();
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        double inv = 1.0 / std::pow(theta, (double)(2*(d % half))/(double)rotary_dim);
        double ang = (double)kv_base * inv;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        if (d < half) {
            qv = sq[d]*c - sq[d+half]*s;      // role j,   out[j]   = x1*c - x2*s
            kv_ = sk[d]*c - sk[d+half]*s;
        } else {
            qv = sq[d]*c + sq[d-half]*s;      // role j+half, out[j+half] = x2*c + x1*s
            kv_ = sk[d]*c + sk[d-half]*s;
        }
    }
    __syncthreads();

    // 4. append k/v to persistent KV row kv_base, publish q/gate.
    kv_k[((int64_t)kv_base * NKV + kvh) * HD + d] = __float2half(kv_);
    kv_v[((int64_t)kv_base * NKV + kvh) * HD + d] = __float2half(vv);
    q_gl[(int64_t)h * HD + d]    = __float2half(qv);
    gate_gl[(int64_t)h * HD + d] = __float2half(gv);
}
```

- [ ] **Step 2: Add `RunQwenFusedQkv` public wrapper** (host) that input-norms `x` then launches `fused_qkv_kernel`, writing `q_gl/gate_gl` and the KV rows. Declare in header.

- [ ] **Step 3: Test — stage-1 parity**: call `RunQwenFusedQkv` vs the split q/k/v (split `qwen_nvfp4_gemm` + `qwen_split_query_kernel` + `qwen_head_norm_kernel` + rope); assert `abs < 1e-2 * |ref| + 1e-6` for q_gl, gate_gl, kv_k row, kv_v row.

- [ ] **Step 4: Commit**

`git add src/qwen_inference.cu include/lsxcommon/qwen_inference.h tests/test_qwen_fused_decode_attn.cu`
`git commit -m "feat: fused qkv stage (inline NVFP4, head-norm, RoPE, KV append)"`

---

## Task 3: Fused block-wide GQA attention stage

**Files:**
- Modify: `src/qwen_inference.cu`

- [ ] **Step 1: Add `fused_attn_kernel`** (grid=NH, block=HD). Reads q from `q_gl`, KV from persistent cache, applies block-wide online-softmax over attended rows `[0..kv_base]`, value-accumulate, gate. Reuses the step-2 validated reduction (`sreduce[512]` static + dynamic `slogits`):

```cpp
__global__ void fused_attn_kernel(
    const __half* __restrict__ q_gl, const __half* __restrict__ gate_gl,
    const __half* __restrict__ kv_k, const __half* __restrict__ kv_v,
    __half* __restrict__ out_h, int kv_base, int NH, int NKV, int HD, float scale)
{
    __shared__ float sreduce[512];
    extern __shared__ float slogits[];
    const int h = blockIdx.x;
    const int kvh = h / (NH / NKV);
    const int d = threadIdx.x;
    const int nscores = kv_base + 1;
    if (d >= HD) return;

    const __half* qrow = q_gl + (int64_t)h * HD;

    // scores (parallel chunks)
    for (int s = d; s < nscores; s += HD) {
        const __half* krow = kv_k + ((int64_t)s * NKV + kvh) * HD;
        float dot = 0.0f;
        for (int dd = 0; dd < HD; ++dd)
            dot += __half2float(qrow[dd]) * __half2float(krow[dd]);
        slogits[s] = dot * scale;
    }
    __syncthreads();
    // max block-reduce (sreduce)
    float m = -1e30f;
    for (int s = d; s < nscores; s += HD) m = fmaxf(m, slogits[s]);
    sreduce[d] = m; __syncthreads();
    for (int st = blockDim.x/2; st>0; st>>=1){ if(d<st) sreduce[d]=fmaxf(sreduce[d],sreduce[d+st]); __syncthreads(); }
    m = sreduce[0]; __syncthreads();
    // sum block-reduce
    float sumd = 0.0f;
    for (int s = d; s < nscores; s += HD) sumd += __expf(slogits[s] - m);
    sreduce[d] = sumd; __syncthreads();
    for (int st = blockDim.x/2; st>0; st>>=1){ if(d<st) sreduce[d]+=sreduce[d+st]; __syncthreads(); }
    const float inv = 1.0f / (sreduce[0] + 1e-10f); __syncthreads();
    // value + gate
    float acc = 0.0f;
    for (int s = 0; s < nscores; ++s) {
        const __half* vrow = kv_v + ((int64_t)s * NKV + kvh) * HD;
        acc += __expf(slogits[s] - m) * inv * __half2float(vrow[d]);
    }
    const float g = __half2float(gate_gl[h*HD + d]);
    out_h[(int64_t)h * HD + d] = __float2half(acc * (1.0f / (1.0f + expf(-g))));
}
```

- [ ] **Step 2: Add `RunQwenFusedAttn` public wrapper** (header) launching it.
- [ ] **Step 3: Test — stage-2 parity**: vs the split attention+gate (`full_attn_softmax_kernel` + `qwen_out_gate_kernel`) on identical q_gl/gate_gl/kv; assert tolerance.
- [ ] **Step 4: Commit**

`git add src/qwen_inference.cu include/lsxcommon/qwen_inference.h tests/test_qwen_fused_decode_attn.cu`
`git commit -m "feat: fused block-wide GQA attention + gate stage"`

---

## Task 4: Fused o_proj stage + compose the full layer + end-to-end parity

**Files:**
- Modify: `src/qwen_inference.cu`

- [ ] **Step 1: Add `fused_o_proj_kernel`** (float atomic accumulator):

```cpp
__global__ void fused_o_proj_kernel(
    const __half* __restrict__ out_h,        // [NH*HD]
    const uint8_t* __restrict__ o_packed, const uint8_t* __restrict__ o_scales,
    float o_gs, float* __restrict__ out_acc, // [H], zeroed by caller
    int NH, int HD, int H)
{
    __shared__ float sreduce[512];
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= HD) return;
    const float oh = __half2float(out_h[h*HD + d]);
    const int64_t row = (int64_t)h*HD + d;
    for (int n = threadIdx.x; n < H; n += blockDim.x) {
        const int64_t e = row*H + n;
        const uint8_t b = o_packed[e>>1];
        const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
        const float scl = E4M3ScaleToFloat(o_scales[e>>4]) * o_gs;
        atomicAdd(&out_acc[n], oh * kE2M1Table[nib] * scl);
    }
}
```

- [ ] **Step 2: Add `RunQwenFusedOProj` wrapper** (host): zero `out_acc[H]`, launch `fused_o_proj_kernel`, then a tiny `qwen_convert_float_acc` tail kernel `out[n] = __float2half(out_acc[n])`.
- [ ] **Step 3: Compose `RunQwenFusedDecodeAttnKernel`** to launch in order on `stream`: input-norm (small) → `fused_qkv_kernel` → `fused_attn_kernel` → `fused_o_proj_kernel` → convert tail. Use scratch `q_gl[NH*HD]`, `gate_gl[NH*HD]`, and a `float out_acc[H]` scratch from the arena/temporary.
- [ ] **Step 4: End-to-end parity test** — full 3-stage fused vs the Task-1 split composition; assert `1e-2` tolerance. Also KV-persistence: call twice (`kv_base`, then `kv_base+1`) and verify the second output equals a two-step split run.
- [ ] **Step 5: Commit**

`git add src/qwen_inference.cu include/lsxcommon/qwen_inference.h tests/test_qwen_fused_decode_attn.cu`
`git commit -m "feat: fused o-proj stage; compose 3-stage fused decoder layer"`

---

## Task 5: Wire into `RunQwenGeneration` decoder branch + full verification

**Files:**
- Modify: `src/qwen_inference.cu`

- [ ] **Step 1: In `RunQwenGeneration`, both decoder branches** (transient ~:1247 and persistent-KV ~:1348), replace the `RunQwenRMSNorm → RunQwenFullAttention → residual-add` for full layers when `seq == 1` with: input-norm is folded into the fused entry (or keep the separate `RunQwenRMSNorm` call before it), then `RunQwenFusedDecodeAttnKernel(...)`, then the same residual-add. Guard `seq > 1` stays split.
- [ ] **Step 2: Full build** — `cmake --build build --config Debug` clean.
- [ ] **Step 3: Run `test_qwen_model`, `test_qwen_fused_decode_attn`, `test_qwen_full_attn`, `test_qwen_full_attn_parallel`, `test_qwen_nvfp4_*`, `test_qwen_kv_*`** — all green; real selfcheck SKIP.
- [ ] **Step 4: Commit**

`git add src/qwen_inference.cu`
`git commit -m "feat: wire fused decoder attention into seq=1 generation path"`

---

## Self-Review

- Spec §2/§3 (3-stage) → Tasks 2/3/4 (fused_qkv, fused_attn, fused_o_proj), §4 tests per stage + end-to-end + KV persistence, §5 files. ✓
- Numeric contract FP32/`1e-2` → all parity tests. ✓ (The Task-1 composition is bit-identical to the split → a clean GREEN baseline.)
- Spec §6 resolved: o_proj = float `atomicAdd` (Task 4); RoPE pairing authoritative (rotate-half `(j, j+half)`, each thread writes only its own dim, reading the partner via synced `sq`/`sk` staging — mirrors `partial_rope_kernel` `:140-141`); 3-stage not single-launch (operator decision). ✓
- Type consistency: `RunQwenFusedDecodeAttnKernel` header == .cu signature; per-stage wrappers (`RunQwenFusedQkv`, `RunQwenFusedAttn`, `RunQwenFusedOProj`) declared + defined consistently; `qw/kw/vw` fields (d_packed/d_scales/global_scale/rows/cols) match `QwenNvfp4Weight`. All kernel names stable.
- RoPE block (Task 2) authoritative — mirrors `partial_rope_kernel` `:140-141,168-169`; no impl-time guessing. ✅
- No draft placeholders retained: Task-1 uses a real split composition (not identity); Task 4 o_proj uses the float `atomicAdd` accumulator.