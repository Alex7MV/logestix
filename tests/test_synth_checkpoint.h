#pragma once

// Model-neutral synthetic-checkpoint primitives shared by every model test
// suite (test-suite autonomy invariant: family fixtures never borrow from one
// another — they all include THIS family-neutral header instead). Deterministic
// byte patterns + a minimal safetensors writer + Arrow round-trip helpers.

#include <lsxcommon/schema.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace synthtest {

inline std::vector<uint8_t> MakeBf16(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t bits = static_cast<uint16_t>(i % 65536);
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(bits & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(bits >> 8);
    }
    return out;
}

// Single source of truth with MakeBf16: element i holds these BF16 bits.
inline uint16_t MakeBf16BitsAt(int64_t i)
{
    return static_cast<uint16_t>(i % 65536);
}

// Round-to-nearest-even float → bf16 bits (host-side, no CUDA dependency).
inline uint16_t FloatToBf16(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000U) == 0x7f800000U)  // NaN/Inf: keep as-is (top 16 bits)
        return static_cast<uint16_t>(u >> 16);
    uint32_t rounding_bias = 0x7fffU + ((u >> 16) & 1U);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

// Verified-value BF16 fixture: element i holds FloatToBf16(pattern(i)).
inline std::vector<uint8_t> MakeBf16Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t b = FloatToBf16(pattern(i));
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(b & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(b >> 8);
    }
    return out;
}

// Verified-value FP32 fixture: element i holds pattern(i) as a float.
inline std::vector<uint8_t> MakeF32Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 4);
    for (int64_t i = 0; i < n; ++i) {
        float v = pattern(i);
        std::memcpy(out.data() + static_cast<size_t>(i) * 4, &v, 4);
    }
    return out;
}

inline std::vector<uint8_t> MakeF32(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 4);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(i * 1000);
        std::memcpy(out.data() + static_cast<size_t>(i) * 4, &bits, 4);
    }
    return out;
}

struct FakeTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

// Builds a single safetensors shard file on disk from in-memory tensors.
// File layout: [8-byte BE header_len][header JSON padded to 8][raw data].
inline void WriteSafetensors(const std::filesystem::path &dir,
                             const std::string &fname,
                             const std::vector<FakeTensor> &tensors)
{
    std::string header = R"({"__metadata__":{})";
    int64_t data_pos = 0;
    for (const auto &t : tensors) {
        std::string shape_s = absl::StrCat("[", absl::StrJoin(t.shape, ","), "]");
        const int64_t begin = data_pos;
        const int64_t end = data_pos + static_cast<int64_t>(t.data.size());
        absl::StrAppend(&header, ",\"", t.name, "\":{\"dtype\":\"", t.dtype,
                        "\",\"shape\":", shape_s, ",\"data_offsets\":[", begin,
                        ",", end, "]}");
        data_pos = end;
    }
    header += "}";
    while ((header.size() + 8) % 8 != 0)
        header += " ";

    std::ofstream f(dir / fname, std::ios::binary);
    uint64_t hlen = static_cast<uint64_t>(header.size());
    uint8_t lenbuf[8];
    for (int i = 0; i < 8; ++i)
        lenbuf[i] = static_cast<uint8_t>((hlen >> (8 * i)) & 0xff);
    f.write(reinterpret_cast<char *>(lenbuf), 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto &t : tensors)
        f.write(reinterpret_cast<const char *>(t.data.data()),
                static_cast<std::streamsize>(t.data.size()));
}

// Reads the logged arrow file and returns the tensor names it stores.
inline std::vector<std::string> TensorNamesIn(const std::filesystem::path &arrow_path)
{
    auto batch_res = lsxcommon::ReadRecordBatchFromFile(arrow_path);
    if (!batch_res.ok()) return {};
    auto t_res = lsxcommon::PinnedSchema::UnpackBatch(*batch_res);
    if (!t_res.ok()) return {};
    std::vector<std::string> names;
    for (auto &t : *t_res) names.push_back(t.name);
    return names;
}

} // namespace synthtest
