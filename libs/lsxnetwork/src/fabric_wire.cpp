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
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(p[0]) |
                                      (std::to_integer<unsigned>(p[1]) << 8));
}
std::uint32_t GetU32(const std::byte* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(p[i])) << (8 * i);
    return v;
}
double GetF64(const std::byte* p) {
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
        bits |= static_cast<std::uint64_t>(std::to_integer<unsigned>(p[i])) << (8 * i);
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Shared v2 stream header: magic, version, type, request_id.
void PutStreamHeader(std::byte* p, FabricFrameType type, std::uint32_t rid) {
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, static_cast<std::uint16_t>(type));
    PutU32(p + 8, rid);
}
WireStatus CheckStreamHeader(const std::byte* p, std::size_t size,
                             FabricFrameType& type, std::uint32_t& rid) {
    if (size < kFabricStreamHeaderBytes) return WireStatus::kTruncated;
    if (GetU32(p) != kFabricMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;
    const std::uint16_t t = GetU16(p + 6);
    if (t < 1 || t > 3) return WireStatus::kBadType;
    type = static_cast<FabricFrameType>(t);
    rid = GetU32(p + 8);
    return WireStatus::kOk;
}

}  // namespace

std::vector<std::byte> EncodeFabricRequest(std::uint32_t request_id,
                                           std::span<const std::int32_t> tokens,
                                           std::uint32_t max_gen) {
    std::vector<std::byte> out(kFabricRequestHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, request_id);
    PutU32(p + 12, max_gen);
    PutU32(p + 16, static_cast<std::uint32_t>(tokens.size()));
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
    const std::uint32_t n = GetU32(p + 16);
    const std::size_t avail = frame.size() - kFabricRequestHeaderBytes;
    if (static_cast<std::size_t>(n) > avail / 4u) return WireStatus::kTruncated;
    if (static_cast<std::size_t>(n) * 4u != avail) return WireStatus::kTrailing;
    out.request_id = GetU32(p + 8);
    out.max_gen = GetU32(p + 12);
    out.tokens = std::span<const std::int32_t>(
        reinterpret_cast<const std::int32_t*>(p + kFabricRequestHeaderBytes), n);
    return WireStatus::kOk;
}

std::vector<std::byte> EncodeFabricTokenBatch(
    std::uint32_t request_id, std::span<const std::int32_t> tokens) {
    std::vector<std::byte> out(kFabricStreamHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kTokenBatch, request_id);
    PutU32(p + 12, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < tokens.size(); ++i)
        PutI32(p + kFabricStreamHeaderBytes + i * 4u, tokens[i]);
    return out;
}

void EncodeFabricTokenBatch(std::vector<std::byte>& out,
                            std::uint32_t request_id,
                            std::span<const std::int32_t> tokens) {
    out.resize(kFabricStreamHeaderBytes + tokens.size() * 4u);
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kTokenBatch, request_id);
    PutU32(p + 12, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < tokens.size(); ++i)
        PutI32(p + kFabricStreamHeaderBytes + i * 4u, tokens[i]);
}

std::vector<std::byte> EncodeFabricDone(std::uint32_t request_id,
                                        bool eos_terminated,
                                        std::uint32_t total_tokens,
                                        double ttft_ms, double total_ms,
                                        double tokens_per_sec) {
    std::vector<std::byte> out(kFabricStreamHeaderBytes + kFabricDoneTailBytes);
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kDone, request_id);
    PutU32(p + 12, 0);  // reserved/count
    PutU32(p + 16, eos_terminated ? 1u : 0u);
    PutU32(p + 20, total_tokens);
    PutF64(p + 24, ttft_ms);
    PutF64(p + 32, total_ms);
    PutF64(p + 40, tokens_per_sec);
    return out;
}

std::vector<std::byte> EncodeFabricError(std::uint32_t request_id,
                                         std::string_view message) {
    std::vector<std::byte> out(kFabricStreamHeaderBytes + message.size());
    std::byte* p = out.data();
    PutStreamHeader(p, FabricFrameType::kError, request_id);
    PutU32(p + 12, static_cast<std::uint32_t>(message.size()));
    if (!message.empty())
        std::memcpy(p + kFabricStreamHeaderBytes, message.data(), message.size());
    return out;
}

WireStatus DecodeFabricStreamFrame(std::span<const std::byte> frame,
                                   FabricStreamFrame& out) {
    out = FabricStreamFrame{};
    FabricFrameType type{};
    std::uint32_t rid = 0;
    const WireStatus hs =
        CheckStreamHeader(frame.data(), frame.size(), type, rid);
    if (hs != WireStatus::kOk) return hs;
    out.type = type;
    out.request_id = rid;
    const std::byte* p = frame.data();
    const std::size_t avail = frame.size() - kFabricStreamHeaderBytes;

    if (type == FabricFrameType::kTokenBatch) {
        const std::uint32_t n = GetU32(p + 12);
        if (static_cast<std::size_t>(n) > avail / 4u) return WireStatus::kTruncated;
        if (static_cast<std::size_t>(n) * 4u != avail) return WireStatus::kTrailing;
        const auto* first =
            reinterpret_cast<const std::int32_t*>(p + kFabricStreamHeaderBytes);
        out.tokens.assign(first, first + n);
        return WireStatus::kOk;
    }
    if (type == FabricFrameType::kDone) {
        if (avail != kFabricDoneTailBytes) return WireStatus::kTrailing;
        out.eos_terminated = (GetU32(p + 16) & 1u) != 0u;
        out.total_tokens = GetU32(p + 20);
        out.ttft_ms = GetF64(p + 24);
        out.total_ms = GetF64(p + 32);
        out.tokens_per_sec = GetF64(p + 40);
        return WireStatus::kOk;
    }
    // kError
    const std::uint32_t len = GetU32(p + 12);
    if (static_cast<std::size_t>(len) > avail) return WireStatus::kTruncated;
    if (static_cast<std::size_t>(len) != avail) return WireStatus::kTrailing;
    out.error.assign(
        reinterpret_cast<const char*>(p + kFabricStreamHeaderBytes), len);
    return WireStatus::kOk;
}

std::vector<std::byte> EncodeFabricCancel(std::uint32_t request_id) {
    std::vector<std::byte> out(kFabricCancelBytes);
    std::byte* p = out.data();
    PutU32(p + 0, kFabricCancelMagic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, request_id);
    return out;
}

WireStatus DecodeFabricCancel(std::span<const std::byte> frame,
                              std::uint32_t& request_id) {
    if (frame.size() < kFabricCancelBytes) return WireStatus::kTruncated;
    if (frame.size() != kFabricCancelBytes) return WireStatus::kTrailing;
    const std::byte* p = frame.data();
    if (GetU32(p) != kFabricCancelMagic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;
    request_id = GetU32(p + 8);
    return WireStatus::kOk;
}

namespace {

std::vector<std::byte> EncodeProbe(std::uint32_t magic, std::uint32_t nonce) {
    std::vector<std::byte> out(kFabricPingBytes);
    std::byte* p = out.data();
    PutU32(p + 0, magic);
    PutU16(p + 4, kFabricVersion);
    PutU16(p + 6, 0);
    PutU32(p + 8, nonce);
    return out;
}

WireStatus DecodeProbe(std::span<const std::byte> frame, std::uint32_t magic,
                       std::uint32_t& nonce) {
    if (frame.size() < kFabricPingBytes) return WireStatus::kTruncated;
    if (frame.size() != kFabricPingBytes) return WireStatus::kTrailing;
    const std::byte* p = frame.data();
    if (GetU32(p) != magic) return WireStatus::kBadMagic;
    if (GetU16(p + 4) != kFabricVersion) return WireStatus::kBadVersion;
    nonce = GetU32(p + 8);
    return WireStatus::kOk;
}

}  // namespace

std::vector<std::byte> EncodeFabricPing(std::uint32_t nonce) {
    return EncodeProbe(kFabricPingMagic, nonce);
}
std::vector<std::byte> EncodeFabricPong(std::uint32_t nonce) {
    return EncodeProbe(kFabricPongMagic, nonce);
}
WireStatus DecodeFabricPing(std::span<const std::byte> frame,
                            std::uint32_t& nonce) {
    return DecodeProbe(frame, kFabricPingMagic, nonce);
}
WireStatus DecodeFabricPong(std::span<const std::byte> frame,
                            std::uint32_t& nonce) {
    return DecodeProbe(frame, kFabricPongMagic, nonce);
}

}  // namespace lsxnetwork
