#include "lsxnetwork/fabric_wire.h"

#include <cstring>

namespace lsxnetwork {
namespace {

void PutU16(std::byte* p, std::uint16_t v) {
    p[0] = static_cast<std::byte>(v & 0xFFu);
    p[1] = static_cast<std::byte>((v >> 8) & 0xFFu);
}

void PutU32(std::byte* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
        p[i] = static_cast<std::byte>((v >> (8 * i)) & 0xFFu);
}

void PutI32(std::byte* p, std::int32_t v) {
    PutU32(p, static_cast<std::uint32_t>(v));
}

void PutF64(std::byte* p, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i)
        p[i] = static_cast<std::byte>((bits >> (8 * i)) & 0xFFull);
}

std::uint16_t GetU16(const std::byte* p) {
    return static_cast<std::uint16_t>(
        std::to_integer<unsigned>(p[0]) |
        (std::to_integer<unsigned>(p[1]) << 8));
}

std::uint32_t GetU32(const std::byte* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(p[i]))
             << (8 * i);
    return v;
}

double GetF64(const std::byte* p) {
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
        bits |= static_cast<std::uint64_t>(std::to_integer<unsigned>(p[i]))
                << (8 * i);
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}  // namespace

std::vector<std::byte> EncodeFabricRequest(
    std::span<const std::int32_t> tokens, std::uint32_t max_gen) {
    std::vector<std::byte> out(kFabricRequestHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, max_gen);
    PutU32(p + 12, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < tokens.size(); ++i)
        PutI32(p + kFabricRequestHeaderBytes + i * 4u, tokens[i]);
    return out;
}

WireStatus DecodeFabricRequest(std::span<const std::byte> frame,
                               FabricRequestView& out) {
    if (frame.size() < kFabricRequestHeaderBytes) return WireStatus::kTruncated;
    const std::byte* p = frame.data();
    if (GetU32(p) != kFabricMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;

    const std::uint32_t n = GetU32(p + 12);
    const std::size_t avail = frame.size() - kFabricRequestHeaderBytes;
    if (static_cast<std::size_t>(n) > avail / 4u) return WireStatus::kTruncated;
    if (static_cast<std::size_t>(n) * 4u != avail) return WireStatus::kTrailing;

    out.max_gen = GetU32(p + 8);
    // The 16-byte header keeps the token region 4-byte aligned for every backing
    // buffer (std::vector and libzmq messages are >= 8-byte aligned).
    out.tokens = std::span<const std::int32_t>(
        reinterpret_cast<const std::int32_t*>(p + kFabricRequestHeaderBytes), n);
    return WireStatus::kOk;
}

std::vector<std::byte> EncodeFabricResponse(const FabricResponse& r) {
    const bool ok = r.status == static_cast<std::int32_t>(FabricStatus::kOk);
    const std::size_t body = ok ? r.tokens.size() * 4u : r.error.size();
    std::vector<std::byte> out(kFabricResponseHeaderBytes + body);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, static_cast<std::uint16_t>(r.status));
    PutU32(p + 8, ok ? static_cast<std::uint32_t>(r.tokens.size()) : 0u);
    PutU32(p + 12, r.eos_terminated ? 1u : 0u);
    PutF64(p + 16, r.ttft_ms);
    PutF64(p + 24, r.total_ms);
    PutF64(p + 32, r.tokens_per_sec);
    if (ok) {
        for (std::size_t i = 0; i < r.tokens.size(); ++i)
            PutI32(p + kFabricResponseHeaderBytes + i * 4u, r.tokens[i]);
    } else if (!r.error.empty()) {
        std::memcpy(p + kFabricResponseHeaderBytes, r.error.data(),
                    r.error.size());
    }
    return out;
}

WireStatus DecodeFabricResponse(std::span<const std::byte> frame,
                                FabricResponse& out) {
    out = FabricResponse{};
    if (frame.size() < kFabricResponseHeaderBytes)
        return WireStatus::kTruncated;
    const std::byte* p = frame.data();
    if (GetU32(p) != kFabricMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;

    out.status = static_cast<std::int32_t>(GetU16(p + 6));
    out.eos_terminated = (GetU32(p + 12) & 1u) != 0u;
    out.ttft_ms = GetF64(p + 16);
    out.total_ms = GetF64(p + 24);
    out.tokens_per_sec = GetF64(p + 32);

    const std::size_t avail = frame.size() - kFabricResponseHeaderBytes;
    if (out.status == static_cast<std::int32_t>(FabricStatus::kOk)) {
        const std::uint32_t n = GetU32(p + 8);
        if (static_cast<std::size_t>(n) > avail / 4u)
            return WireStatus::kTruncated;
        if (static_cast<std::size_t>(n) * 4u != avail)
            return WireStatus::kTrailing;
        const auto* first = reinterpret_cast<const std::int32_t*>(
            p + kFabricResponseHeaderBytes);
        out.tokens.assign(first, first + n);
    } else {
        out.error.assign(
            reinterpret_cast<const char*>(p + kFabricResponseHeaderBytes),
            avail);
    }
    return WireStatus::kOk;
}

}  // namespace lsxnetwork
