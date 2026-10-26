// Real-weight gate for the pinned MLA projections (the seq==1 streaming GEMV
// over the converted NVFP4 records). The synthetic attention test hand-rolls
// POW2-only block scales, so it cannot see an E4M3 decode or a K-slice offset
// defect; this case reads the vault tree's own bytes and compares the kernel
// against a host decoder written from the codec contract.
//
// SKIPs when the converted Kimi tree is absent (SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "nvfp4_scale.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/schema.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace kimi = lsxcommon::kimi;

std::filesystem::path ModelDir()
{
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Kimi-K2.7-Code";
}

// E2M1 magnitudes by nibble code (bit 3 = sign).
float E2m1(int nib)
{
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = kMag[nib & 7];
    return (nib & 8) ? -m : m;
}

// Host dot product of one output row: w[i] = E2M1(nib) * E4M3(scale) * gs.
double HostRowDot(const uint8_t *packed, const uint8_t *scales, int row, int K,
                  const std::vector<float> &x, float gs)
{
    const uint8_t *prow = packed + static_cast<size_t>(row) * (K / 2);
    const uint8_t *srow = scales + static_cast<size_t>(row) * (K / 16);
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        const uint8_t byte = prow[k >> 1];
        const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
        const float s =
            lsxcommon::nvfp4::E4m3ToFloat(srow[k >> 4]) * gs;
        acc += static_cast<double>(x[k]) * E2m1(nib) * s;
    }
    return acc;
}

struct Proj {
    std::vector<uint8_t> packed, scales;
    int rows = 0, cols = 0;
    float gs = 1.0f;
};

Proj LoadProj(const std::filesystem::path &log_root, int layer,
              const std::string &base)
{
    const auto views = lsxcommon::LoadTensorBatchViews(
        log_root / "pinned" / ("layer_" + std::to_string(layer) + ".arrow"));
    REQUIRE(views.ok());
    Proj out;
    std::string pk_name, sc_name;
    for (const auto &t : views->tensors) {
        const std::string suffix = base + ".weight_packed";
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0) {
            pk_name = t.name.substr(t.name.size() - suffix.size());
            out.packed.assign(t.data, t.data + t.bytes);
        }
        const std::string ssuffix = base + ".weight_scale";
        if (t.name.size() >= ssuffix.size() &&
            t.name.compare(t.name.size() - ssuffix.size(), ssuffix.size(),
                           ssuffix) == 0) {
            sc_name = t.name.substr(t.name.size() - ssuffix.size());
            out.scales.assign(t.data, t.data + t.bytes);
        }
    }
    REQUIRE_FALSE(out.packed.empty());
    REQUIRE_FALSE(out.scales.empty());

    // Logical [rows, cols] from the source index is the frozen geometry.
    if (base == "q_a_proj") { out.rows = kimi::kKimiQaRows; out.cols = kimi::kKimiHidden; }
    else if (base == "q_b_proj") { out.rows = kimi::kKimiQbRows; out.cols = kimi::kKimiQLoraRank; }
    else if (base == "kv_a_proj_with_mqa") { out.rows = kimi::kKimiKvaRows; out.cols = kimi::kKimiHidden; }
    else if (base == "kv_b_proj") { out.rows = kimi::kKimiKvbRows; out.cols = kimi::kKimiKvLoraRank; }
    else { out.rows = kimi::kKimiORows; out.cols = kimi::kKimiNumHeads * kimi::kKimiHeadV; }

    // Global scale from the model index (mla_nvfp4[layer][base]).
    std::ifstream f(log_root / "model.index.json");
    REQUIRE(f.good());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    const std::string key = "\"" + base + "\": ";
    size_t pos = json.find("\"" + std::to_string(layer) + "\": {");
    REQUIRE(pos != std::string::npos);
    pos = json.find(key, pos);
    REQUIRE(pos != std::string::npos);
    out.gs = std::strtof(json.c_str() + pos + key.size(), nullptr);
    return out;
}

float Frand(int seed, int i)
{
    uint32_t v = static_cast<uint32_t>(seed * 2654435761u + i * 40503u);
    v ^= v >> 13;
    v *= 1274126177u;
    v ^= v >> 16;
    return (static_cast<float>(v % 20001) - 10000.0f) / 10000.0f;
}

// Runs the pinned-projection streaming GEMV for one row block and compares
// against the host decoder for the first rows.
void CheckProjection(const std::filesystem::path &log_root, const char *base)
{
    const Proj p = LoadProj(log_root, 0, base);
    std::vector<float> x(static_cast<size_t>(p.cols));
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.5f * Frand(11, static_cast<int>(i));
    // Diagnostic input override: compare the kernel on the ENGINE's own
    // intermediate (e.g. /tmp/kimi_mla_attn_0.bin, row 0).
    if (const char *inp = std::getenv("KIMI_PROJ_INPUT")) {
        std::ifstream f(inp, std::ios::binary);
        REQUIRE(f.good());
        int hdr[2] = {0, 0};
        f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
        REQUIRE(hdr[1] == p.cols);
        std::vector<__nv_bfloat16> row(static_cast<size_t>(hdr[1]));
        f.read(reinterpret_cast<char *>(row.data()),
               static_cast<std::streamsize>(row.size() * 2));
        for (size_t i = 0; i < row.size(); ++i) x[i] = __bfloat162float(row[i]);
    }

    std::vector<__half> hx(x.size()), hy(static_cast<size_t>(p.rows));
    for (size_t i = 0; i < x.size(); ++i) hx[i] = __float2half(x[i]);

    __half *d_x = nullptr, *d_y = nullptr;
    uint8_t *d_pk = nullptr, *d_sc = nullptr;
    REQUIRE(cudaMalloc(&d_x, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_y, hy.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_pk, p.packed.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sc, p.scales.size()) == cudaSuccess);
    cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, p.packed.data(), p.packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, p.scales.data(), p.scales.size(), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const cudaError_t e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, d_x, d_pk, d_sc, p.gs, d_y, p.cols, p.rows, false);
    INFO(base << " launch: " << cudaGetErrorString(e));
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaMemcpy(hy.data(), d_y, hy.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Compare a spread of rows (the full row set costs seconds per row).
    const int kRows = 24;
    double worst = 0.0;
    for (int r = 0; r < kRows; ++r) {
        const double ref = HostRowDot(p.packed.data(), p.scales.data(), r,
                                      p.cols, x, p.gs);
        const double got = static_cast<double>(__half2float(hy[r]));
        worst = std::max(worst, std::fabs(got - ref));
        INFO(base << " row " << r << " engine=" << got << " host=" << ref);
        CHECK(std::fabs(got - ref) <= 1e-3 + 1e-3 * std::fabs(ref));
    }
    INFO(base << " worst abs error: " << worst);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_pk);
    cudaFree(d_sc);
    cudaStreamDestroy(stream);
}

// One real routed expert through the engine's staging FFN vs a host reference
// built from the same record. This is the MoE analogue of the projection
// check: the record is the converted tree's own bytes.
void CheckExpertFfn(const std::filesystem::path &log_root, int layer, int expert)
{
    const std::string path = (log_root / "experts" /
                              ("layer_" + std::to_string(layer)) /
                              ("exp_" + std::to_string(expert) + ".arrow"))
                                 .string();
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    REQUIRE(batch.ok());
    auto rec = lsxcommon::ExpertSchema::UnpackBatch(*batch);
    REQUIRE(rec.ok());
    const lsxcommon::PackedExpert &e = *rec;

    constexpr int Hd = kimi::kKimiHidden;
    constexpr int I = kimi::kKimiMoeIntermediate;

    // Host reference input and a deterministic row.
    std::vector<float> xf(static_cast<size_t>(Hd));
    for (int i = 0; i < Hd; ++i) xf[i] = 0.5f * Frand(23, i);

    auto matvec_ref = [](const std::vector<uint8_t> &packed,
                         const std::vector<uint8_t> &scales, int rows, int cols,
                         float gs, const std::vector<float> &x) {
        std::vector<double> y(static_cast<size_t>(rows), 0.0);
        for (int r = 0; r < rows; ++r) {
            const uint8_t *prow = packed.data() + static_cast<size_t>(r) * (cols / 2);
            const uint8_t *srow = scales.data() + static_cast<size_t>(r) * (cols / 16);
            double acc = 0.0;
            for (int k = 0; k < cols; ++k) {
                const uint8_t byte = prow[k >> 1];
                const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                acc += static_cast<double>(x[k]) * E2m1(nib) *
                       lsxcommon::nvfp4::E4m3ToFloat(srow[k >> 4]) * gs;
            }
            y[r] = acc;
        }
        return y;
    };

    const auto g = matvec_ref(e.w1_packed, e.w1_scales, I, Hd,
                              e.w1_global_scale, xf);
    const auto u = matvec_ref(e.w3_packed, e.w3_scales, I, Hd,
                              e.w3_global_scale, xf);
    std::vector<float> act(static_cast<size_t>(I));
    for (int i = 0; i < I; ++i) {
        const double s = static_cast<double>(g[i]) /
                         (1.0 + std::exp(-static_cast<double>(g[i])));
        act[i] = static_cast<float>(s * static_cast<double>(u[i]));
    }
    const auto yref = matvec_ref(e.w2_packed, e.w2_scales, Hd, I,
                                 e.w2_global_scale, act);

    // Engine path (same entry the decode loop builds).
    __nv_bfloat16 *d_x = nullptr, *d_out = nullptr;
    void *d_staging = nullptr;
    REQUIRE(cudaMalloc(&d_x, Hd * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, Hd * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_staging, kimi::KimiExpertFfnStagingBytes()) ==
            cudaSuccess);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(Hd));
    std::vector<__nv_bfloat16> zero(static_cast<size_t>(Hd), __float2bfloat16(0.f));
    for (int i = 0; i < Hd; ++i) hx[i] = __float2bfloat16(xf[i]);
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, zero.data(), zero.size() * 2, cudaMemcpyHostToDevice);

    kimi::KimiExpertEntry entry{};
    entry.host_w1 = e.w1_packed.data();
    entry.host_w2 = e.w2_packed.data();
    entry.host_w3 = e.w3_packed.data();
    entry.host_w1_scales = e.w1_scales.data();
    entry.host_w2_scales = e.w2_scales.data();
    entry.host_w3_scales = e.w3_scales.data();
    entry.w1_global_scale = e.w1_global_scale;
    entry.w2_global_scale = e.w2_global_scale;
    entry.w3_global_scale = e.w3_global_scale;
    entry.w1_bytes = static_cast<int64_t>(e.w1_packed.size());
    entry.w2_bytes = static_cast<int64_t>(e.w2_packed.size());
    entry.w3_bytes = static_cast<int64_t>(e.w3_packed.size());
    entry.w1_scales_bytes = static_cast<int64_t>(e.w1_scales.size());
    entry.w2_scales_bytes = static_cast<int64_t>(e.w2_scales.size());
    entry.w3_scales_bytes = static_cast<int64_t>(e.w3_scales.size());

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const cudaError_t err = kimi::RunKimiExpertFfn(
        stream, entry, 1.0f, d_x, d_out, d_staging,
        kimi::KimiExpertFfnStagingBytes());
    INFO("expert ffn launch: " << cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<__nv_bfloat16> out(static_cast<size_t>(Hd));
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    double num = 0.0, den = 0.0, dot = 0.0;
    for (int i = 0; i < Hd; ++i) {
        const double a = static_cast<double>(__bfloat162float(out[i]));
        const double r = yref[i];
        num += (a - r) * (a - r);
        den += r * r;
    }
    const double rel = std::sqrt(num / den);
    INFO("expert " << expert << " relL2=" << rel);
    CHECK(rel <= 0.02);

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_staging);
    cudaStreamDestroy(stream);
}

// Router (mlp.gate.weight) matvec through the engine's BF16 GEMV vs a host
// dot. The router selects the expert set, so a wrong score silently changes
// the whole MoE layer even when every expert kernel is correct.
void CheckRouter(const std::filesystem::path &log_root, int layer)
{
    const auto views = lsxcommon::LoadTensorBatchViews(
        log_root / "pinned" / ("layer_" + std::to_string(layer) + ".arrow"));
    REQUIRE(views.ok());
    const lsxcommon::TensorView *gate = nullptr;
    for (const auto &t : views->tensors) {
        const std::string suffix = "mlp.gate.weight";
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            gate = &t;
    }
    REQUIRE(gate != nullptr);
    REQUIRE(gate->dtype == "BF16");
    const int N = kimi::kKimiNumRouted;
    const int K = kimi::kKimiHidden;

    const uint16_t *w = reinterpret_cast<const uint16_t *>(gate->data);
    std::vector<float> xf(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) xf[i] = 0.5f * Frand(31, i);

    __nv_bfloat16 *d_w = nullptr, *d_x = nullptr;
    float *d_y = nullptr;
    REQUIRE(cudaMalloc(&d_w, static_cast<size_t>(N) * K * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, static_cast<size_t>(K) * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_y, static_cast<size_t>(N) * 4) == cudaSuccess);
    cudaMemcpy(d_w, w, static_cast<size_t>(N) * K * 2, cudaMemcpyHostToDevice);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) hx[i] = __float2bfloat16(xf[i]);
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const cudaError_t e = lsxcommon::gemv::RunLxGemmVec(
        stream, d_w, d_x, 1, N, K, lsxcommon::gemv::LxDtype::kBF16,
        lsxcommon::gemv::LxDtype::kBF16, 1.0f, d_y,
        lsxcommon::gemv::LxDtype::kF32, 0.0f);
    INFO("router gemv: " << cudaGetErrorString(e));
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> y(static_cast<size_t>(N));
    cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost);

    double worst = 0.0;
    for (int r = 0; r < N; ++r) {
        double ref = 0.0;
        for (int k = 0; k < K; ++k)
            ref += static_cast<double>(
                       __bfloat162float(
                           *reinterpret_cast<const __nv_bfloat16 *>(&w[r * K + k]))) *
                   static_cast<double>(__bfloat162float(hx[k]));
        worst = std::max(worst, std::fabs(y[r] - ref));
        if (r < 8) INFO("router row " << r << " gpu=" << y[r] << " host=" << ref);
    }
    INFO("router worst abs error: " << worst);
    CHECK(worst <= 1e-2);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaStreamDestroy(stream);
}

} // namespace

TEST_CASE("pinned MLA projection GEMV matches the host NVFP4 decoder on real "
          "weights", "[kimi][mla][real-model]")
{
    const std::filesystem::path model = ModelDir();
    const std::filesystem::path log_root = model / "logestix";
    if (!std::filesystem::exists(log_root / "pinned" / "layer_0.arrow"))
        SKIP("converted Kimi model directory not present");

    SECTION("o_proj (K=8192, N=7168, kp=8)") { CheckProjection(log_root, "o_proj"); }
    SECTION("kv_a (K=7168, N=576, kp=4)") { CheckProjection(log_root, "kv_a_proj_with_mqa"); }
    SECTION("q_a (K=7168, N=1536, kp=4)") { CheckProjection(log_root, "q_a_proj"); }
    SECTION("q_b (K=1536, N=12288, kp=1)") { CheckProjection(log_root, "q_b_proj"); }

    SECTION("routed expert FFN (layer 1, expert 0)")
    {
        CheckExpertFfn(log_root, 1, 0);
    }
    SECTION("router BF16 matvec (layer 1)") { CheckRouter(log_root, 1); }
}
