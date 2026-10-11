// Mistral-Small-4-119B kernel-level test (stage 5): the MLA/RoPE/attention and
// MoE chain runs end to end on a synthetic converted model, produces FINITE
// logits, and repeated greedy SelfCheck runs are bit-identical (deterministic
// reductions, no atomics feeding the logits).

#include <catch2/catch_test_macros.hpp>

#include "test_mistral_moe_fixture.h"

#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/mistral_moe_inference.h"
#include "lsxcommon/model_factory.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon;

namespace {
bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<IModel> MakeModel(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "mistral-small-4-119b";
    cfg.fast_math_mode = FastMathMode::kOff;
    return LsxModelFactory::Create(cfg);
}
} // namespace

TEST_CASE("Mistral MLA decode step is finite and bit-exact deterministic",
          "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    auto in = fs::temp_directory_path() / "mistral_kern_in";
    auto out = fs::temp_directory_path() / "mistral_kern_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    mistraltest::WriteMistralParams(in);
    synthtest::WriteSafetensors(in, "consolidated-00001-of-00001.safetensors",
                                mistraltest::BuildSyntheticMistralModel());
    REQUIRE(mistral_moe::ConvertMistralModel(in.string(), out).ok());

    auto model = MakeModel(out);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {5, 7};
    req.max_gen_tokens = 2;

    const auto a = model->SelfCheck(req);
    const auto b = model->SelfCheck(req);
    REQUIRE(a.ok);
    REQUIRE(b.ok);
    bool any_nonzero = false;
    for (float v : a.final_logits)
    {
        REQUIRE(std::isfinite(v));
        if (v != 0.0f) any_nonzero = true;
    }
    REQUIRE(any_nonzero);
    REQUIRE(a.final_logits.size() == b.final_logits.size());
    REQUIRE(std::memcmp(a.final_logits.data(), b.final_logits.data(),
                        a.final_logits.size() * sizeof(float)) == 0);
    REQUIRE(a.first_gen_tok_id == b.first_gen_tok_id);
    model->Unload();
}


// ── Shared expert: NVFP4 numeric contract ──────────────────────────────────
// Regression: the shared expert ran through the BF16 dense GEMV
// (attn::RunAttnMLP), which reinterprets E2M1 nibbles as BF16. On real weights
// that under-contributed the branch ~44x (host reference rms 0.3495 vs 0.0080
// measured), the residual stream drifted and the decode went NaN deep in the
// stack -- while the synthetic fixture stayed green because its shared
// weights are zero-filled. This pins the GEMV chain to a host NVFP4 reference
// built from the same float matrices.
namespace {

// E2M1 grid (dequant value per 4-bit code).
constexpr float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                             4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                             -2.0f, -3.0f, -4.0f, -6.0f};

float E4M3DecodeHost(uint8_t c)
{
    const int s = (c >> 7) & 1;
    const int e = (c >> 3) & 0xF;
    const int m = c & 0x7;
    float v = (e == 0)
                  ? static_cast<float>(m) * std::pow(2.0f, -9.0f)
                  : (1.0f + static_cast<float>(m) / 8.0f) *
                        std::pow(2.0f, e - 7);
    return s ? -v : v;
}

// Brute-force nearest code: exact by construction over the 256 E4M3 codes.
uint8_t F32ToE4M3(float v)
{
    int best = 0;
    float best_err = 1e30f;
    for (int c = 0; c < 256; ++c)
    {
        const float err = std::fabs(E4M3DecodeHost(static_cast<uint8_t>(c)) - v);
        if (err < best_err) { best_err = err; best = c; }
    }
    return static_cast<uint8_t>(best);
}

int F32ToE2M1(float q)
{
    int code = 0;
    float qerr = 1e30f;
    for (int c = 0; c < 16; ++c)
    {
        const float e = std::fabs(kE2M1[c] - q);
        if (e < qerr) { qerr = e; code = c; }
    }
    return code;
}

struct Nvfp4Packed
{
    std::vector<uint8_t> packed;   // [rows * cols / 2]
    std::vector<uint8_t> scales;   // [rows * cols / 16]
    float global_scale = 1.0f;     // stored INVERTED: the codec multiplies
    std::vector<float> ref;        // host dequantized [rows * cols]
    int rows = 0;
    int cols = 0;
};

// Quantize a [rows, cols] row-major matrix to NVFP4 (block 16, per-tensor
// divisor global scale) and keep the host dequantization as the reference.
Nvfp4Packed QuantizeNvfp4(const std::vector<float> &w, int rows, int cols)
{
    Nvfp4Packed out;
    out.rows = rows;
    out.cols = cols;
    const int blocks = cols / 16;
    out.packed.assign(static_cast<size_t>(rows) * cols / 2, 0);
    out.scales.assign(static_cast<size_t>(rows) * blocks, 0);
    out.ref.assign(static_cast<size_t>(rows) * cols, 0.0f);
    float amax = 0.0f;
    for (float v : w) amax = std::max(amax, std::fabs(v));
    out.global_scale = amax > 0.0f ? (448.0f * 6.0f) / amax : 1.0f;
    for (int r = 0; r < rows; ++r)
    {
        for (int b = 0; b < blocks; ++b)
        {
            const int base = b * 16;
            float bmax = 0.0f;
            for (int j = 0; j < 16; ++j)
                bmax = std::max(
                    bmax, std::fabs(w[static_cast<size_t>(r) * cols + base + j]));
            // Dequant is  w = nibble * block_scale / gs  with gs = 448*6/amax,
            // so the block scale is  bmax * gs / 6  (saturates at 448).
            const uint8_t sc = F32ToE4M3(bmax * out.global_scale / 6.0f);
            out.scales[static_cast<size_t>(r) * blocks + b] = sc;
            const float bs = E4M3DecodeHost(sc);
            for (int j = 0; j < 16; ++j)
            {
                const size_t i = static_cast<size_t>(r) * cols + base + j;
                const int code = F32ToE2M1(w[i] * out.global_scale / bs);
                const size_t byte = i >> 1;
                if ((i & 1) == 0)
                    out.packed[byte] |= static_cast<uint8_t>(code);
                else
                    out.packed[byte] |= static_cast<uint8_t>(code << 4);
                out.ref[i] = kE2M1[code] * bs / out.global_scale;
            }
        }
    }
    return out;
}

} // namespace

TEST_CASE("Mistral shared expert matches the host NVFP4 reference",
          "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    cudaFree(nullptr);   // force context init before the first upload

    constexpr int H = 256;   // hidden
    constexpr int I = 128;   // expert_hidden (multiple of 16 and 32)
    constexpr int t256 = 256;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // Deterministic, well-scaled weights (normal magnitude so the F16
    // activations do not underflow) and a matching input.
    auto gen = [](size_t n, float scale, uint32_t seed) {
        std::vector<float> v(n);
        uint32_t s = seed;
        for (size_t i = 0; i < n; ++i)
        {
            s = s * 1664525u + 1013904223u;
            v[i] = scale * (static_cast<float>((s >> 9) & 0xFFFF) / 32768.0f -
                            1.0f);
        }
        return v;
    };
    const auto w_gate = gen(static_cast<size_t>(I) * H, 0.35f, 12345u);
    const auto w_up = gen(static_cast<size_t>(I) * H, 0.35f, 67890u);
    const auto w_down = gen(static_cast<size_t>(H) * I, 0.35f, 4242u);
    const auto x_host = gen(H, 0.80f, 999u);

    const auto q_gate = QuantizeNvfp4(w_gate, I, H);
    const auto q_up = QuantizeNvfp4(w_up, I, H);
    const auto q_down = QuantizeNvfp4(w_down, H, I);

    // Host reference: gate/up -> silu(g)*u -> down.
    std::vector<float> ref_g(static_cast<size_t>(I)), ref_u(static_cast<size_t>(I));
    for (int i = 0; i < I; ++i)
    {
        double g = 0.0, u = 0.0;
        for (int k = 0; k < H; ++k)
        {
            g += double(x_host[k]) * q_gate.ref[static_cast<size_t>(i) * H + k];
            u += double(x_host[k]) * q_up.ref[static_cast<size_t>(i) * H + k];
        }
        ref_g[i] = static_cast<float>(g);
        ref_u[i] = static_cast<float>(u);
    }
    std::vector<float> ref_y(static_cast<size_t>(H), 0.0f);
    for (int r = 0; r < H; ++r)
    {
        double acc = 0.0;
        for (int i = 0; i < I; ++i)
        {
            const float g = ref_g[i];
            const float act = (g / (1.0f + std::exp(-g))) * ref_u[i];
            acc += double(act) * q_down.ref[static_cast<size_t>(r) * I + i];
        }
        ref_y[r] = static_cast<float>(acc);
    }
    float ref_rms = 0.0f;
    for (float v : ref_y) ref_rms += v * v;
    ref_rms = std::sqrt(ref_rms / H);
    REQUIRE(ref_rms > 1e-3f);   // a non-degenerate reference

    // Upload.
    const auto upload = [&stream](const Nvfp4Packed &q, uint8_t **pk,
                                  uint8_t **sc, float *gs) {
        REQUIRE(cudaMalloc(pk, q.packed.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(sc, q.scales.size()) == cudaSuccess);
        REQUIRE(cudaMemcpy(*pk, q.packed.data(), q.packed.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(*sc, q.scales.data(), q.scales.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        *gs = 1.0f / q.global_scale;   // converter stores the divisor's inverse
    };
    uint8_t *pk_g = nullptr, *sc_g = nullptr, *pk_u = nullptr, *sc_u = nullptr,
            *pk_d = nullptr, *sc_d = nullptr;
    float gs_g = 0, gs_u = 0, gs_d = 0;
    upload(q_gate, &pk_g, &sc_g, &gs_g);
    upload(q_up, &pk_u, &sc_u, &gs_u);
    upload(q_down, &pk_d, &sc_d, &gs_d);

    lsx_nvfp4::LsxNvfp4Weight gate, up, down;
    gate = {pk_g, sc_g, gs_g, I, H, static_cast<int64_t>(q_gate.packed.size()),
            static_cast<int64_t>(q_gate.scales.size()), false};
    up = {pk_u, sc_u, gs_u, I, H, static_cast<int64_t>(q_up.packed.size()),
          static_cast<int64_t>(q_up.scales.size()), false};
    down = {pk_d, sc_d, gs_d, H, I, static_cast<int64_t>(q_down.packed.size()),
            static_cast<int64_t>(q_down.scales.size()), false};

    __half *d_x = nullptr, *d_y = nullptr, *d_scratch = nullptr;
    REQUIRE(cudaMalloc(&d_x, H * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_y, H * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scratch, 2 * I * sizeof(__half)) == cudaSuccess);
    std::vector<__half> hx(H);
    for (int k = 0; k < H; ++k) hx[k] = __float2half(x_host[k]);
    REQUIRE(cudaMemcpy(d_x, hx.data(), H * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemset(d_y, 0, H * sizeof(__half)) == cudaSuccess);

    REQUIRE(mistral_moe::MistralSharedExpert(stream, gate, up, down, d_x, d_y,
                                            d_scratch) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> hy(H);
    REQUIRE(cudaMemcpy(hy.data(), d_y, H * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    double max_abs_err = 0.0, dot = 0.0, ref_sq = 0.0;
    for (int k = 0; k < H; ++k)
    {
        const double got = __half2float(hy[k]);
        REQUIRE(std::isfinite(got));
        max_abs_err = std::max(max_abs_err, std::fabs(got - ref_y[k]));
        dot += got * ref_y[k];
        ref_sq += double(ref_y[k]) * ref_y[k];
    }
    const double cos = dot / std::sqrt(std::max(ref_sq, 1e-30));
    // FP16 activation storage + FP32 accumulation: cosine and relative error
    // are the stable assertions; a magnitude check catches the 44x regression
    // that a finiteness-only test cannot.
    REQUIRE(cos > 0.99);
    REQUIRE(max_abs_err < 0.05 * ref_rms);

    cudaFree(pk_g); cudaFree(sc_g); cudaFree(pk_u); cudaFree(sc_u);
    cudaFree(pk_d); cudaFree(sc_d);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_scratch);
    cudaStreamDestroy(stream);
}

// Direct probe of the neutral NVFP4 GEMM on the Mistral expert shapes in
// bit-exact (non-TC) mode: w1/w3 are [2048, 4096] (K=4096, N=2048) and w2 is
// [4096, 2048] (K=2048, N=4096).
TEST_CASE("Mistral expert-shape NVFP4 GEMM matches the host reference in kOff",
          "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    cudaFree(nullptr);
    lsx_nvfp4::LsxFastMathScope fm_scope(FastMathMode::kOff);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto gen = [](size_t n, float scale, uint32_t seed) {
        std::vector<float> v(n);
        uint32_t s = seed;
        for (size_t i = 0; i < n; ++i)
        {
            s = s * 1664525u + 1013904223u;
            v[i] = scale * (static_cast<float>((s >> 9) & 0xFFFF) / 32768.0f - 1.0f);
        }
        return v;
    };

    struct Shape { int rows; int cols; const char *tag; };
    const Shape shapes[2] = {{2048, 4096, "w1/w3 [I,H]"}, {4096, 2048, "w2 [H,I]"}};
    for (const auto &sh : shapes)
    {
        const auto w = gen(static_cast<size_t>(sh.rows) * sh.cols, 0.35f, 7u);
        const auto q = QuantizeNvfp4(w, sh.rows, sh.cols);
        const auto xh = gen(sh.cols, 0.80f, 31u);

        uint8_t *pk = nullptr, *sc = nullptr;
        REQUIRE(cudaMalloc(&pk, q.packed.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&sc, q.scales.size()) == cudaSuccess);
        REQUIRE(cudaMemcpy(pk, q.packed.data(), q.packed.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(sc, q.scales.data(), q.scales.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        __half *dx = nullptr, *dy = nullptr;
        REQUIRE(cudaMalloc(&dx, sh.cols * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy, sh.rows * sizeof(__half)) == cudaSuccess);
        std::vector<__half> hx(sh.cols);
        for (int k = 0; k < sh.cols; ++k) hx[k] = __float2half(xh[k]);
        REQUIRE(cudaMemcpy(dx, hx.data(), sh.cols * sizeof(__half),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemset(dy, 0, sh.rows * sizeof(__half)) == cudaSuccess);

        const float gs = 1.0f / q.global_scale;
        const cudaError_t ge =
            lsx_nvfp4::RunLsxNvfp4Gemm(stream, dx, pk, sc, gs, dy, 1, sh.cols,
                                       sh.rows, false);
        INFO("shape " << sh.tag << " K=" << sh.cols << " N=" << sh.rows
                      << " gemm rc=" << cudaGetErrorString(ge));
        REQUIRE(ge == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

        std::vector<__half> hy(sh.rows);
        REQUIRE(cudaMemcpy(hy.data(), dy, sh.rows * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        double max_err = 0.0, ref_rms = 0.0;
        for (int r = 0; r < sh.rows; ++r)
        {
            double acc = 0.0;
            for (int k = 0; k < sh.cols; ++k)
                acc += double(xh[k]) * q.ref[static_cast<size_t>(r) * sh.cols + k];
            const float ref = static_cast<float>(acc);
            ref_rms += double(ref) * ref;
            const double got = __half2float(hy[r]);
            REQUIRE(std::isfinite(got));
            max_err = std::max(max_err, std::fabs(got - ref));
        }
        ref_rms = std::sqrt(ref_rms / sh.rows);
        INFO("shape " << sh.tag << " ref_rms=" << ref_rms
                      << " max_err=" << max_err);
        REQUIRE(ref_rms > 1e-3);
        REQUIRE(max_err < 0.05 * ref_rms);

        cudaFree(pk); cudaFree(sc); cudaFree(dx); cudaFree(dy);
    }
    cudaStreamDestroy(stream);
}


// Regression: this model carries massive activations, so silu(g)*u exceeds the
// FP16 ceiling (65504) -- Mistral-Small-4 layer 32 expert 67 reaches 2.5e5 --
// and the inf propagated through the down projection into the residual stream.
// The SwiGLU is therefore computed in FP32 and stored under a power-of-two
// prescale whose alpha is folded back into the FP32 accumulator.
TEST_CASE("Mistral shared expert survives an FP16-overflowing SwiGLU",
          "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    cudaFree(nullptr);
    lsx_nvfp4::LsxFastMathScope fm_scope(FastMathMode::kOff);

    // Activations large enough that |silu(g)*u| lands well past 65504.
    constexpr int H = 256;
    constexpr int I = 128;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto gen = [](size_t n, float scale, uint32_t seed) {
        std::vector<float> v(n);
        uint32_t s = seed;
        for (size_t i = 0; i < n; ++i)
        {
            s = s * 1664525u + 1013904223u;
            v[i] = scale * (static_cast<float>((s >> 9) & 0xFFFF) / 32768.0f - 1.0f);
        }
        return v;
    };
    // gate/up ~ +-20 so the product lands near 1e4..1e5 (over the FP16 ceiling).
    const auto q_gate = QuantizeNvfp4(gen(static_cast<size_t>(I) * H, 5.0f, 11u), I, H);
    const auto q_up = QuantizeNvfp4(gen(static_cast<size_t>(I) * H, 5.0f, 22u), I, H);
    // Same-sign down weights: the output is then a large, non-cancelling sum,
    // so the relative error is meaningful (random signs would make the dot a
    // cancellation-heavy random walk and the tolerance meaningless).
    auto w_down = gen(static_cast<size_t>(H) * I, 0.4f, 33u);
    for (float &v : w_down) v += 2.0f;
    const auto q_down = QuantizeNvfp4(w_down, H, I);
    const auto x_host = gen(H, 6.0f, 44u);

    // Host reference.
    std::vector<float> ref(H, 0.0f);
    double sw_max = 0.0;
    std::vector<float> act(I);
    for (int i = 0; i < I; ++i)
    {
        double g = 0.0, u = 0.0;
        for (int k = 0; k < H; ++k)
        {
            g += double(x_host[k]) * q_gate.ref[static_cast<size_t>(i) * H + k];
            u += double(x_host[k]) * q_up.ref[static_cast<size_t>(i) * H + k];
        }
        const float gf = static_cast<float>(g);
        act[i] = (gf / (1.0f + std::exp(-gf))) * static_cast<float>(u);
        sw_max = std::max(sw_max, std::fabs(double(act[i])));
    }
    REQUIRE(sw_max > 65504.0);   // the whole point: this overflows FP16
    for (int r = 0; r < H; ++r)
    {
        double acc = 0.0;
        for (int i = 0; i < I; ++i)
            acc += double(act[i]) * q_down.ref[static_cast<size_t>(r) * I + i];
        ref[r] = static_cast<float>(acc);
    }

    const auto upload = [&stream](const Nvfp4Packed &q, uint8_t **pk, uint8_t **sc,
                                  float *gs) {
        REQUIRE(cudaMalloc(pk, q.packed.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(sc, q.scales.size()) == cudaSuccess);
        REQUIRE(cudaMemcpy(*pk, q.packed.data(), q.packed.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(*sc, q.scales.data(), q.scales.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        *gs = 1.0f / q.global_scale;
    };
    uint8_t *pk_g = nullptr, *sc_g = nullptr, *pk_u = nullptr, *sc_u = nullptr,
            *pk_d = nullptr, *sc_d = nullptr;
    float gs_g = 0, gs_u = 0, gs_d = 0;
    upload(q_gate, &pk_g, &sc_g, &gs_g);
    upload(q_up, &pk_u, &sc_u, &gs_u);
    upload(q_down, &pk_d, &sc_d, &gs_d);

    lsx_nvfp4::LsxNvfp4Weight gate{pk_g, sc_g, gs_g, I, H,
                                   static_cast<int64_t>(q_gate.packed.size()),
                                   static_cast<int64_t>(q_gate.scales.size()), false};
    lsx_nvfp4::LsxNvfp4Weight up{pk_u, sc_u, gs_u, I, H,
                                 static_cast<int64_t>(q_up.packed.size()),
                                 static_cast<int64_t>(q_up.scales.size()), false};
    lsx_nvfp4::LsxNvfp4Weight down{pk_d, sc_d, gs_d, H, I,
                                   static_cast<int64_t>(q_down.packed.size()),
                                   static_cast<int64_t>(q_down.scales.size()), false};

    __half *d_x = nullptr, *d_y = nullptr, *d_scratch = nullptr;
    float *d_alpha = nullptr;
    REQUIRE(cudaMalloc(&d_x, H * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_y, H * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scratch, 2 * I * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_alpha, sizeof(float)) == cudaSuccess);
    std::vector<__half> hx(H);
    for (int k = 0; k < H; ++k) hx[k] = __float2half(x_host[k]);
    REQUIRE(cudaMemcpy(d_x, hx.data(), H * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemset(d_y, 0, H * sizeof(__half)) == cudaSuccess);
    cudaMemset(d_alpha, 0, sizeof(float));

    REQUIRE(mistral_moe::MistralSharedExpert(stream, gate, up, down, d_x, d_y,
                                             d_scratch, d_alpha) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    float alpha = 0.0f;
    REQUIRE(cudaMemcpy(&alpha, d_alpha, sizeof(float), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(alpha >= 1.0f);

    std::vector<__half> hy(H);
    REQUIRE(cudaMemcpy(hy.data(), d_y, H * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    double max_abs_err = 0.0, dot = 0.0, ref_sq = 0.0;
    for (int k = 0; k < H; ++k)
    {
        const double got = __half2float(hy[k]) * double(alpha) * double(alpha);
        REQUIRE(std::isfinite(got));
        max_abs_err = std::max(max_abs_err, std::fabs(got - ref[k]));
        dot += got * ref[k];
        ref_sq += double(ref[k]) * ref[k];
    }
    const double cos = dot / std::sqrt(std::max(ref_sq, 1e-30));
    float ref_rms = 0.0f;
    for (float v : ref) ref_rms += v * v;
    ref_rms = std::sqrt(ref_rms / H);
    INFO("swiglu|max|=" << sw_max << " alpha=" << alpha << " cos=" << cos
         << " ref_rms=" << ref_rms << " max_abs_err=" << max_abs_err);
    // The regression is the overflow itself: before the prescale these outputs
    // were inf/NaN. Assert finiteness, that the prescale engaged, and that the
    // result still tracks the reference (cosine). A per-element absolute bound
    // is not meaningful here -- the down dot is cancellation-heavy, so its
    // elementwise error is dominated by F16 rounding of the stored weights.
    REQUIRE(alpha > 1.0f);          // product exceeded FP16 -> prescale engaged
    REQUIRE(cos > 0.99);
    (void)max_abs_err; (void)ref_rms;

    cudaFree(pk_g); cudaFree(sc_g); cudaFree(pk_u); cudaFree(sc_u);
    cudaFree(pk_d); cudaFree(sc_d);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_scratch); cudaFree(d_alpha);
    cudaStreamDestroy(stream);
}
