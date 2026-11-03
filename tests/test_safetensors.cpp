// Safetensors descriptor layer: TensorInfo initialization/validation and the
// centralized validated tensor reader. The reader replaced 26 private copies
// (one per converter), which disagreed on validation: half never checked the
// read count, several never checked the file bounds, and a malformed header
// produced a descriptor with uninitialized offsets (UB).
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model_integrity.h>
#include <lsxcommon/safetensors.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using lsxcommon::ParseTensorInfos;
using lsxcommon::ReadTensorData;
using lsxcommon::ReadTensorDataChecked;
using lsxcommon::TensorDtypeBytes;
using lsxcommon::TensorInfo;
using lsxcommon::TensorInfoError;
using lsxcommon::TensorInfoErrorMessage;
using lsxcommon::TensorReadError;
using lsxcommon::TensorReadErrorMessage;
using lsxcommon::ValidateTensorInfo;

namespace {

TensorInfo MakeInfo(std::string name, std::string dtype,
                    std::vector<int64_t> shape, int64_t begin, int64_t end)
{
    TensorInfo info;
    info.name = std::move(name);
    info.dtype = std::move(dtype);
    info.shape = std::move(shape);
    info.data_begin = begin;
    info.data_end = end;
    return info;
}

} // namespace

TEST_CASE("TensorInfo: a default-constructed descriptor is a rank-0 scalar",
          "[safetensors]")
{
    // Safetensors stores a single value as shape [] — the parser must default
    // the offsets so a JSON entry without them can never carry garbage.
    const TensorInfo info;
    CHECK(info.shape.empty());
    CHECK(info.numel() == 1);
    CHECK(info.data_begin == 0);
    CHECK(info.data_end == 0);
    CHECK(info.byte_size() == 0);  // unknown (empty) dtype
}

TEST_CASE("TensorDtypeBytes: every safetensors dtype has a width",
          "[safetensors]")
{
    CHECK(TensorDtypeBytes("BOOL") == 1);
    CHECK(TensorDtypeBytes("U8") == 1);
    CHECK(TensorDtypeBytes("I8") == 1);
    CHECK(TensorDtypeBytes("F8_E5M2") == 1);
    CHECK(TensorDtypeBytes("F8_E4M3") == 1);
    CHECK(TensorDtypeBytes("F8_E8M0") == 1);
    CHECK(TensorDtypeBytes("I16") == 2);
    CHECK(TensorDtypeBytes("U16") == 2);
    CHECK(TensorDtypeBytes("F16") == 2);
    CHECK(TensorDtypeBytes("BF16") == 2);
    CHECK(TensorDtypeBytes("I32") == 4);
    CHECK(TensorDtypeBytes("U32") == 4);
    CHECK(TensorDtypeBytes("F32") == 4);
    CHECK(TensorDtypeBytes("I64") == 8);
    CHECK(TensorDtypeBytes("U64") == 8);
    CHECK(TensorDtypeBytes("F64") == 8);
    CHECK(TensorDtypeBytes("") == 0);
    CHECK(TensorDtypeBytes("BF32") == 0);
    CHECK(TensorDtypeBytes("f32") == 0);  // the spec is case-sensitive
}

TEST_CASE("ValidateTensorInfo: accepts spec-conforming descriptors",
          "[safetensors]")
{
    CHECK(ValidateTensorInfo(MakeInfo("s", "F32", {}, 0, 4)) ==
          TensorInfoError::kOk);
    CHECK(ValidateTensorInfo(MakeInfo("w", "BF16", {2, 3}, 8, 20)) ==
          TensorInfoError::kOk);
    CHECK(ValidateTensorInfo(MakeInfo("empty", "U8", {0}, 16, 16)) ==
          TensorInfoError::kOk);
    CHECK(ValidateTensorInfo(MakeInfo("p", "U8", {4, 2}, 24, 32)) ==
          TensorInfoError::kOk);
}

TEST_CASE("ValidateTensorInfo: rejects malformed descriptors", "[safetensors]")
{
    CHECK(ValidateTensorInfo(MakeInfo("", "F32", {}, 0, 4)) ==
          TensorInfoError::kEmptyName);
    CHECK(ValidateTensorInfo(MakeInfo("x", "FP4", {}, 0, 4)) ==
          TensorInfoError::kUnknownDtype);
    CHECK(ValidateTensorInfo(MakeInfo("x", "F32", {-1, 4}, 0, 16)) ==
          TensorInfoError::kNegativeDimension);
    // numel overflows int64: 2^62 * 4 dims would wrap and under-allocate.
    CHECK(ValidateTensorInfo(MakeInfo("x", "F32", {1LL << 62, 4}, 0, 8)) ==
          TensorInfoError::kNumelOverflow);
    CHECK(ValidateTensorInfo(MakeInfo("x", "F32", {-1}, -4, 0)) ==
          TensorInfoError::kNegativeOffset);
    CHECK(ValidateTensorInfo(MakeInfo("x", "F32", {2}, 8, 4)) ==
          TensorInfoError::kReversedOffsets);
    // 2 F32 elements need 8 bytes; the header claims 4.
    CHECK(ValidateTensorInfo(MakeInfo("x", "F32", {2}, 0, 4)) ==
          TensorInfoError::kSizeMismatch);
    // 3 BF16 elements need 6 bytes; the header claims 8 (overlap with the
    // next tensor would silently corrupt both).
    CHECK(ValidateTensorInfo(MakeInfo("x", "BF16", {3}, 0, 8)) ==
          TensorInfoError::kSizeMismatch);
    // numel itself fits, but numel * 8 does not.
    CHECK(ValidateTensorInfo(MakeInfo("x", "F64", {1LL << 60}, 0, 0)) ==
          TensorInfoError::kByteSizeOverflow);
}

TEST_CASE("error enums map to distinct, non-empty messages", "[safetensors]")
{
    const TensorInfoError info_errors[] = {
        TensorInfoError::kOk,
        TensorInfoError::kEmptyName,
        TensorInfoError::kUnknownDtype,
        TensorInfoError::kNegativeDimension,
        TensorInfoError::kNumelOverflow,
        TensorInfoError::kByteSizeOverflow,
        TensorInfoError::kNegativeOffset,
        TensorInfoError::kReversedOffsets,
        TensorInfoError::kSizeMismatch,
    };
    std::vector<std::string> info_msgs;
    for (auto e : info_errors) {
        const std::string msg = TensorInfoErrorMessage(e);
        CHECK(!msg.empty());
        CHECK(msg != "unknown tensor info error");
        info_msgs.push_back(msg);
    }
    std::sort(info_msgs.begin(), info_msgs.end());
    CHECK(std::adjacent_find(info_msgs.begin(), info_msgs.end()) ==
          info_msgs.end());

    const TensorReadError read_errors[] = {
        TensorReadError::kOk,
        TensorReadError::kInvalidDescriptor,
        TensorReadError::kOpenFailed,
        TensorReadError::kShortHeader,
        TensorReadError::kHeaderTooLarge,
        TensorReadError::kOutOfBounds,
        TensorReadError::kShortRead,
    };
    std::vector<std::string> read_msgs;
    for (auto e : read_errors) {
        const std::string msg = TensorReadErrorMessage(e);
        CHECK(!msg.empty());
        CHECK(msg != "unknown tensor read error");
        read_msgs.push_back(msg);
    }
    std::sort(read_msgs.begin(), read_msgs.end());
    CHECK(std::adjacent_find(read_msgs.begin(), read_msgs.end()) ==
          read_msgs.end());
}

TEST_CASE("ParseTensorInfos: reads a well-formed header", "[safetensors]")
{
    const std::string json = R"({
        "__metadata__":{"format":"pt"},
        "model.embed.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},
        "model.scale":{"dtype":"F32","shape":[],"data_offsets":[12,16]},
        "model.norm.weight":{"dtype":"F32","shape":[0],"data_offsets":[16,16]}
    })";

    const auto report = ParseTensorInfos(json);
    REQUIRE(report.tensors.size() == 3);
    CHECK(report.rejected == 0);

    const auto &w = report.tensors[0];
    CHECK(w.name == "model.embed.weight");
    CHECK(w.dtype == "BF16");
    CHECK(w.shape == std::vector<int64_t>{2, 3});
    CHECK(w.data_begin == 0);
    CHECK(w.data_end == 12);
    CHECK(w.byte_size() == 12);

    // Rank-0 scalar: shape [] means ONE value, not an absent tensor.
    const auto &s = report.tensors[1];
    CHECK(s.name == "model.scale");
    CHECK(s.shape.empty());
    CHECK(s.numel() == 1);
    CHECK(s.data_end - s.data_begin == 4);

    CHECK(report.tensors[2].numel() == 0);
    CHECK(report.tensors[2].byte_size() == 0);
}

TEST_CASE("ParseTensorInfos: drops and counts malformed entries",
          "[safetensors]")
{
    const std::string json = R"({
        "ok":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
        "no_dtype":{"shape":[1],"data_offsets":[4,8]},
        "bad_dtype":{"dtype":"FP4","shape":[1],"data_offsets":[8,12]},
        "no_shape":{"dtype":"F32","data_offsets":[12,16]},
        "float_shape":{"dtype":"F32","shape":[1.5],"data_offsets":[16,20]},
        "neg_shape":{"dtype":"F32","shape":[-1],"data_offsets":[20,24]},
        "no_offsets":{"dtype":"F32","shape":[1]},
        "one_offset":{"dtype":"F32","shape":[1],"data_offsets":[24]},
        "three_offsets":{"dtype":"F32","shape":[1],"data_offsets":[24,28,32]},
        "neg_offset":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]},
        "short_span":{"dtype":"F32","shape":[2],"data_offsets":[32,36]},
        "not_an_object":[1,2,3],
        "overflow":{"dtype":"F32","shape":[9223372036854775807,4],"data_offsets":[36,40]}
    })";

    const auto report = ParseTensorInfos(json);
    REQUIRE(report.tensors.size() == 1);
    CHECK(report.tensors[0].name == "ok");
    CHECK(report.rejected == 12);
}

TEST_CASE("ParseTensorInfos: an unparseable header yields an empty report",
          "[safetensors]")
{
    for (const std::string bad : {"", "not json", "[1,2,3]", "{\"a\":",
                                  "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[0,4]}"}) {
        INFO("header: " << bad);
        const auto report = ParseTensorInfos(bad);
        CHECK(report.tensors.empty());
        CHECK(report.rejected == 0);
    }
}

// ── Reader fixtures ────────────────────────────────────────────────────────

namespace {

namespace fs = std::filesystem;

std::vector<uint8_t> SeqBytes(int n)
{
    std::vector<uint8_t> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = static_cast<uint8_t>(i + 1);
    return v;
}

// Writes [8-byte LE header length][header JSON][payload].
void WriteShardFile(const fs::path &path, const std::string &header,
                    const std::vector<uint8_t> &payload)
{
    std::ofstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    const uint64_t hlen = header.size();
    for (int i = 0; i < 8; ++i)
        f.put(static_cast<char>((hlen >> (8 * i)) & 0xff));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!payload.empty())
        f.write(reinterpret_cast<const char *>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
}

fs::path FreshDir(const std::string &tag)
{
    const fs::path dir = fs::temp_directory_path() / ("logestix_safetensors_" + tag);
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

} // namespace

TEST_CASE("read_safetensors_file: a file shorter than the length prefix yields nothing",
          "[safetensors]")
{
    const fs::path dir = FreshDir("short_file");
    std::ofstream(dir / "tiny.safetensors", std::ios::binary) << "abc";
    const auto shard =
        lsxcommon::read_safetensors_file((dir / "tiny.safetensors").string());
    CHECK(shard.tensors.empty());
    CHECK(shard.filename == "tiny.safetensors");
}

TEST_CASE("ReadTensorDataChecked: reads the declared byte range", "[safetensors]")
{
    const fs::path dir = FreshDir("read_ok");
    const std::string header =
        R"({"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
        R"("b":{"dtype":"F32","shape":[],"data_offsets":[8,12]},)"
        R"("z":{"dtype":"F32","shape":[0],"data_offsets":[12,12]}})";
    const auto payload = SeqBytes(12);
    WriteShardFile(dir / "model.safetensors", header, payload);

    const auto shard = lsxcommon::read_safetensors_file(
        (dir / "model.safetensors").string());
    REQUIRE(shard.tensors.size() == 3);
    CHECK(shard.file_size ==
          static_cast<int64_t>(8 + header.size() + payload.size()));

    const auto a = ReadTensorDataChecked(dir, shard, shard.tensors[0]);
    REQUIRE(a.ok());
    CHECK(a.bytes == std::vector<uint8_t>(payload.begin(), payload.begin() + 8));
    CHECK(a.detail.empty());

    const auto b = ReadTensorDataChecked(dir, shard, shard.tensors[1]);
    REQUIRE(b.ok());
    CHECK(b.bytes == std::vector<uint8_t>(payload.begin() + 8, payload.end()));

    // A legitimately empty tensor is a SUCCESS with zero bytes; only the
    // compat wrapper cannot tell that apart from a failed read.
    const auto z = ReadTensorDataChecked(dir, shard, shard.tensors[2]);
    CHECK(z.ok());
    CHECK(z.bytes.empty());

    // Compat wrapper returns the same payload for the converters.
    CHECK(ReadTensorData(dir, shard, shard.tensors[0]) == a.bytes);
}

TEST_CASE("ReadTensorDataChecked: rejects a descriptor before touching the file",
          "[safetensors]")
{
    const fs::path dir = FreshDir("read_desc");
    lsxcommon::SafetensorsModel::Shard shard;
    shard.filename = "does-not-exist.safetensors";

    TensorInfo bad = MakeInfo("w", "FP4", {2}, 0, 2);
    const auto r = ReadTensorDataChecked(dir, shard, bad);
    CHECK(r.error == TensorReadError::kInvalidDescriptor);
    CHECK(r.bytes.empty());
    CHECK(r.detail.find("unknown dtype") != std::string::npos);
    CHECK(ReadTensorData(dir, shard, bad).empty());
}

TEST_CASE("ReadTensorDataChecked: rejects malformed shard images", "[safetensors]")
{
    SECTION("missing file") {
        const fs::path dir = FreshDir("read_missing");
        lsxcommon::SafetensorsModel::Shard shard;
        shard.filename = "nope.safetensors";
        const TensorInfo info = MakeInfo("t", "U8", {1}, 0, 1);
        const auto r = ReadTensorDataChecked(dir, shard, info);
        CHECK(r.error == TensorReadError::kOpenFailed);
        CHECK(r.detail.find("nope.safetensors") != std::string::npos);
    }

    SECTION("file shorter than the 8-byte length") {
        const fs::path dir = FreshDir("read_short_hdr");
        std::ofstream(dir / "m.safetensors", std::ios::binary) << "abc";
        lsxcommon::SafetensorsModel::Shard shard;
        shard.filename = "m.safetensors";
        const TensorInfo info = MakeInfo("t", "U8", {1}, 0, 1);
        CHECK(ReadTensorDataChecked(dir, shard, info).error ==
              TensorReadError::kShortHeader);
    }

    SECTION("declared header longer than the file") {
        const fs::path dir = FreshDir("read_big_hdr");
        WriteShardFile(dir / "m.safetensors", "{\"t\":1}", {});
        lsxcommon::SafetensorsModel::Shard shard;
        shard.filename = "m.safetensors";
        // Overwrite the length to claim 1000 header bytes in a 15-byte file.
        std::fstream f(dir / "m.safetensors", std::ios::binary | std::ios::in |
                                                   std::ios::out);
        REQUIRE(f.is_open());
        const uint64_t hlen = 1000;
        for (int i = 0; i < 8; ++i)
            f.put(static_cast<char>((hlen >> (8 * i)) & 0xff));
        f.close();
        const TensorInfo info = MakeInfo("t", "U8", {1}, 0, 1);
        CHECK(ReadTensorDataChecked(dir, shard, info).error ==
              TensorReadError::kHeaderTooLarge);
    }

    SECTION("tensor range beyond the end of the file") {
        const fs::path dir = FreshDir("read_oob");
        const std::string header =
            R"({"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[100,108]}})";
        WriteShardFile(dir / "m.safetensors", header, SeqBytes(12));
        const auto shard = lsxcommon::read_safetensors_file(
            (dir / "m.safetensors").string());
        REQUIRE(shard.tensors.size() == 1);
        // The descriptor is internally consistent, so only the file bound can
        // reject it.
        CHECK(ValidateTensorInfo(shard.tensors[0]) == TensorInfoError::kOk);
        const auto r = ReadTensorDataChecked(dir, shard, shard.tensors[0]);
        CHECK(r.error == TensorReadError::kOutOfBounds);
        CHECK(r.detail.find("exceeds") != std::string::npos);
    }

    SECTION("an oversized shard.file_size cannot mask the real bound") {
        const fs::path dir = FreshDir("read_stale_size");
        const std::string header =
            R"({"a":{"dtype":"U8","shape":[64],"data_offsets":[0,64]}})";
        WriteShardFile(dir / "m.safetensors", header, SeqBytes(8));
        auto shard = lsxcommon::read_safetensors_file(
            (dir / "m.safetensors").string());
        REQUIRE(shard.tensors.size() == 1);
        shard.file_size = 1LL << 40;  // a stale descriptor must not help
        CHECK(ReadTensorDataChecked(dir, shard, shard.tensors[0]).error ==
              TensorReadError::kOutOfBounds);
    }
}

// ── Fuzz (deterministic, no external corpus/tooling) ───────────────────────
//
// Two entry points, both mutational: the header parser (pure, in memory) and
// the whole shard image (file I/O + reader). Invariants: whatever comes back
// is a validated descriptor, and a successful read is byte-identical to the
// slice an independent offset computation picks out of the mutated image.

namespace {

std::vector<uint8_t> BuildShardImage(const std::string &header,
                                     const std::vector<uint8_t> &payload)
{
    std::vector<uint8_t> out;
    const uint64_t hlen = header.size();
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<uint8_t>((hlen >> (8 * i)) & 0xff));
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void WriteBytes(const fs::path &path, const std::vector<uint8_t> &bytes)
{
    std::ofstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    if (!bytes.empty())
        f.write(reinterpret_cast<const char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
}

struct LogSilencer {
    spdlog::level::level_enum previous = spdlog::get_level();
    LogSilencer() { spdlog::set_level(spdlog::level::off); }
    ~LogSilencer() { spdlog::set_level(previous); }
};

} // namespace

TEST_CASE("fuzz: arbitrary header text only yields validated descriptors",
          "[safetensors][fuzz]")
{
    const std::string seed =
        R"({"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
        R"("b":{"dtype":"F32","shape":[],"data_offsets":[8,12]}})";
    std::mt19937_64 rng(0x5A5A5A5Aull);

    for (int iter = 0; iter < 2000; ++iter) {
        std::string buf = seed;
        const auto edits = 1 + rng() % 4;
        for (auto e = 0u; e < edits && !buf.empty(); ++e) {
            switch (rng() % 4) {
            case 0:
                buf[rng() % buf.size()] =
                    static_cast<char>(rng() & 0xff);
                break;
            case 1:
                buf.resize(rng() % buf.size());
                break;
            case 2:
                buf.insert(buf.begin() + static_cast<long>(rng() % buf.size()),
                           static_cast<char>(rng() & 0xff));
                break;
            default:
                buf.push_back(static_cast<char>(rng() & 0xff));
                break;
            }
        }
        INFO("iteration " << iter << " header: " << buf);
        const auto report = ParseTensorInfos(buf);
        for (const auto &t : report.tensors)
            REQUIRE(ValidateTensorInfo(t) == TensorInfoError::kOk);
    }
}

TEST_CASE("fuzz: mutated shard images never escape the reader's checks",
          "[safetensors][fuzz]")
{
    const fs::path dir = FreshDir("fuzz_read");
    const fs::path file = dir / "mut.safetensors";
    const std::string header =
        R"({"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
        R"("b":{"dtype":"F32","shape":[],"data_offsets":[8,12]},)"
        R"("c":{"dtype":"U8","shape":[8],"data_offsets":[12,20]}})";
    const auto image = BuildShardImage(header, SeqBytes(20));
    std::mt19937_64 rng(0xC0FFEEull);
    const LogSilencer silence;

    for (int iter = 0; iter < 1000; ++iter) {
        std::vector<uint8_t> bytes = image;
        const auto edits = 1 + rng() % 4;
        for (auto e = 0u; e < edits; ++e) {
            switch (rng() % 5) {
            case 0:
                if (!bytes.empty())
                    bytes[rng() % bytes.size()] ^=
                        static_cast<uint8_t>(1u << (rng() % 8));
                break;
            case 1:
                if (!bytes.empty())
                    bytes.resize(rng() % bytes.size());
                break;
            case 2: {  // rewrite the 8-byte header length
                const uint64_t len = rng() % (1ull << 40);
                for (int i = 0; i < 8 && i < static_cast<int>(bytes.size()); ++i)
                    bytes[static_cast<size_t>(i)] =
                        static_cast<uint8_t>((len >> (8 * i)) & 0xff);
                break;
            }
            case 3:
                bytes.push_back(static_cast<uint8_t>(rng()));
                break;
            default:
                if (bytes.size() > 8)
                    bytes[8 + rng() % (bytes.size() - 8)] = '[';
                break;
            }
        }
        WriteBytes(file, bytes);
        const auto shard =
            lsxcommon::read_safetensors_file(file.string());

        for (const auto &t : shard.tensors) {
            INFO("iteration " << iter << " tensor " << t.name);
            REQUIRE(ValidateTensorInfo(t) == TensorInfoError::kOk);
            const auto r = ReadTensorDataChecked(dir, shard, t);
            if (!r.ok())
                continue;

            REQUIRE(r.bytes.size() == static_cast<size_t>(t.byte_size()));
            // Independent oracle: recompute the slice from the image itself.
            uint64_t hlen = 0;
            for (int i = 0; i < 8; ++i)
                hlen |= static_cast<uint64_t>(bytes[static_cast<size_t>(i)])
                        << (8 * i);
            const uint64_t data_off = 8 + hlen;
            REQUIRE(data_off + static_cast<uint64_t>(t.data_end) <=
                    bytes.size());
            const std::vector<uint8_t> want(
                bytes.begin() + static_cast<long>(data_off + t.data_begin),
                bytes.begin() + static_cast<long>(data_off + t.data_end));
            REQUIRE(r.bytes == want);
        }
    }
}

TEST_CASE("VerifyAllTensorReads: reports the first unreadable tensor",
          "[safetensors]")
{
    const fs::path dir = FreshDir("integrity");
    const std::string header =
        R"({"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
        R"("b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}})";
    WriteShardFile(dir / "m.safetensors", header, SeqBytes(16));

    auto model = lsxcommon::scan_directory(dir.string());
    REQUIRE(model.shards.size() == 1);
    CHECK(lsxcommon::integrity::VerifyAllTensorReads(dir, model).ok());

    // Truncate the payload: the descriptors still parse, the read must fail
    // and name the tensor.
    auto bytes = BuildShardImage(header, SeqBytes(10));
    WriteBytes(dir / "m.safetensors", bytes);
    model = lsxcommon::scan_directory(dir.string());
    REQUIRE(model.shards.size() == 1);
    const auto status = lsxcommon::integrity::VerifyAllTensorReads(dir, model);
    CHECK(!status.ok());
    CHECK(status.ToString().find("'b'") != std::string::npos);
}
