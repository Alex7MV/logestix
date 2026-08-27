// Verifies INT4->NVFP4 conversion fidelity against the source checkpoint:
// decodes the SOURCE INT4 gate_proj of layer_1/exp_0 in both nibble orders
// and correlates each against the CONVERTED NVFP4 payload. Real-model test.
#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/path_registry.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <sstream>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {


struct SrcTensor {
    int64_t begin = 0;
    int64_t end = 0;
    std::string shard;
};

// Locates a tensor's data range via model.safetensors.index.json + shard
// headers (manual parse; mirrors the fixture writer's container format).
std::map<std::string, SrcTensor> LocateTensors(
    const std::filesystem::path &model_dir,
    const std::vector<std::string> &names)
{
    std::ifstream f(model_dir / "model.safetensors.index.json");
    REQUIRE(f.good());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    std::map<std::string, SrcTensor> out;
    for (const auto &name : names) {
        const std::string needle = '"' + name + "\": \"";
        size_t p = json.find(needle);
        if (p == std::string::npos) continue;
        p += needle.size();
        size_t q = json.find('"', p);
        SrcTensor st;
        st.shard = json.substr(p, q - p);

        // Parse that shard header for data_offsets.
        std::ifstream sf(model_dir / st.shard, std::ios::binary);
        uint64_t hlen = 0;
        sf.read(reinterpret_cast<char *>(&hlen), 8);
        std::string hj(hlen, 0);
        sf.read(hj.data(), static_cast<std::streamsize>(hlen));
        const std::string name_needle = "\"" + name + "\":{";
        const size_t np = hj.find(name_needle);
        REQUIRE(np != std::string::npos);
        const std::string off_needle = "\"data_offsets\":[";
        size_t o = hj.find(off_needle, np);
        REQUIRE(o != std::string::npos);
        o += off_needle.size();
        const size_t comma = hj.find(',', o);
        const size_t close = hj.find(']', comma);
        REQUIRE(comma != std::string::npos);
        REQUIRE(close != std::string::npos);
        st.begin = 8 + static_cast<int64_t>(hlen) +
                   std::stoll(hj.substr(o, comma - o));
        st.end = 8 + static_cast<int64_t>(hlen) +
                 std::stoll(hj.substr(comma + 1, close - comma - 1));
        out[name] = st;
    }
    return out;
}

template <typename T>
std::vector<T> ReadRange(const std::filesystem::path &model_dir,
                         const SrcTensor &t)
{
    std::ifstream f(model_dir / t.shard, std::ios::binary);
    f.seekg(t.begin);
    std::vector<T> data((t.end - t.begin) / sizeof(T));
    f.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(t.end - t.begin));
    return data;
}

} // namespace

using lsxcommon::kimi::KimiInt4DequantToBf16;
using lsxcommon::kimi::load_kimi_expert_index;
using lsxcommon::kimi::ensure_kimi_expert_resident;
using lsxcommon::kimi::free_kimi_expert_index;

TEST_CASE("Kimi INT4->NVFP4 conversion matches source nibble order",
          "[kimi][convert][real-model]")
{
    namespace fs = std::filesystem;
    const fs::path model = [] {
        const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
        return env ? fs::path(env)
                   : fs::path(lsxcommon::lsx_model_models_dir()) / "Kimi-K2.7-Code";
    }();
    if (!fs::exists(model / "logestix" / "experts"))
        SKIP("converted Kimi experts not present");

    constexpr int rows = lsxcommon::kimi::kKimiMoeIntermediate;
    constexpr int cols = lsxcommon::kimi::kKimiHidden;
    const size_t numel = static_cast<size_t>(rows) * cols;

    auto located = LocateTensors(
        model,
        {"language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed",
         "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale"});
    auto packed_ints =
        ReadRange<int>(model, located.at(
            "language_model.model.layers.1.mlp.experts.0.gate_proj."
            "weight_packed"));
    auto scales_u16 =
        ReadRange<uint16_t>(model, located.at(
            "language_model.model.layers.1.mlp.experts.0.gate_proj."
            "weight_scale"));
    REQUIRE(packed_ints.size() == numel / 8);
    REQUIRE(scales_u16.size() == numel / 32);

    // GPU-dequant both nibble orders.
    auto dequant = [&](bool high_first) {
        std::vector<int> hp = packed_ints;
        if (high_first) {
            for (auto &w : hp) {
                uint32_t u = static_cast<uint32_t>(w);
                uint32_t r = 0;
                for (int k = 0; k < 8; ++k)
                    r |= ((u >> (4 * k)) & 0xF) << (4 * (7 - k));
                w = static_cast<int>(r);
            }
        }
        int *d_p;
        uint16_t *d_s;
        uint16_t *d_o;
        cudaMalloc(&d_p, numel / 2);
        cudaMalloc(&d_s, numel / 32 * 2);
        cudaMalloc(&d_o, numel * 2);
        cudaMemcpy(d_p, hp.data(), numel / 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, scales_u16.data(), numel / 32 * 2,
                   cudaMemcpyHostToDevice);
        cudaStream_t st;
        cudaStreamCreate(&st);
        REQUIRE(KimiInt4DequantToBf16(d_p, d_s, rows, cols, d_o, st) ==
                cudaSuccess);
        cudaStreamSynchronize(st);
        std::vector<uint16_t> out(numel);
        cudaMemcpy(out.data(), d_o, numel * 2, cudaMemcpyDeviceToHost);
        cudaStreamDestroy(st);
        cudaFree(d_p); cudaFree(d_s); cudaFree(d_o);
        return out;
    };
    const auto lo = dequant(false);
    const auto hi = dequant(true);

    // Converted NVFP4 payload of the same expert.
    auto db = load_kimi_expert_index(model.string(), /*num_layers=*/3,
                                     /*experts_per_layer=*/2);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ensure_kimi_expert_resident(db, /*layer=*/1, /*expert=*/0, stream);
    cudaStreamSynchronize(stream);
    const lsxcommon::kimi::KimiExpertEntry &e = db.entries[1 * 2 + 0];
    REQUIRE(e.host_w1 != nullptr);
    const float gs = e.w1_global_scale;
    INFO("w1_global_scale=" << gs);

    static const float tbl[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                  -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f,
                                  -6.f};
    auto bf16bits_to_float = [](uint16_t b) -> float {
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    };
    auto e4m3 = [](uint8_t b) -> float {
        const uint8_t e4 = (b >> 3) & 0x0F;
        if (!e4) return 0.f;
        return std::ldexp(1.0f, static_cast<int>(e4) - 7);
    };
    const size_t sample_stride = 97;
    std::vector<double> nv, src_lo, src_hi;
    for (size_t i = 0; i < numel && i < e.w1_bytes * 2;
         i += sample_stride) {
        const uint8_t byte = e.host_w1[i >> 1];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        nv.push_back(tbl[nib] *
                     e4m3(static_cast<uint8_t>(e.host_w1_scales[i >> 4])) *
                     gs);
        src_lo.push_back(bf16bits_to_float(lo[i]));
        src_hi.push_back(bf16bits_to_float(hi[i]));
    }
    REQUIRE(nv.size() > 100);

    auto corr = [](const std::vector<double> &a,
                   const std::vector<double> &b) {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        const double n = static_cast<double>(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            sa += a[i]; sb += b[i];
            saa += a[i] * a[i]; sbb += b[i] * b[i]; sab += a[i] * b[i];
        }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n);
        double vb = sbb / n - (sb / n) * (sb / n);
        return cov / std::sqrt(va * vb + 1e-30);
    };
    for (int k = 0; k < 6; ++k)
        WARN("elem" << k << ": nv=" << nv[k] << " src_lo=" << src_lo[k]
             << " scale_byte=" << static_cast<int>(e.host_w1_scales[k >> 4])
             << " packed=" << static_cast<int>(e.host_w1[k >> 1]));
    const double c_lo = corr(nv, src_lo);
    const double c_hi = corr(nv, src_hi);
    INFO("corr low-first=" << c_lo << " corr high-first=" << c_hi);
    CHECK(std::max(c_lo, c_hi) > 0.85);

    free_kimi_expert_index(db);
    cudaStreamDestroy(stream);
}
