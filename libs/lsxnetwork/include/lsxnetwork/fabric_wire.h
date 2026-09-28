#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lsxnetwork {

/// v2 streaming wire protocol (little-endian, explicit byte-wise encoding):
///   request    = 24-byte header + 4 * num_tokens bytes  (client -> server)
///   stream     = one message per frame, each: 16-byte header + body
///                token_batch: 16 + 4 * count bytes
///                done:        16 + 32 bytes
///                error:       16 + len bytes (UTF-8)
///   cancel     = 12 bytes (client -> server, separate PUSH/PULL channel)
inline constexpr std::uint32_t kFabricMagic = 0x4246584Cu;        // "LXFB" LE
inline constexpr std::uint32_t kFabricCancelMagic = 0x4346584Cu;  // "LXFC" LE
inline constexpr std::uint32_t kFabricPingMagic = 0x5046584Cu;    // "LXFP" LE
inline constexpr std::uint32_t kFabricPongMagic = 0x5146584Cu;    // "LXFQ" LE
inline constexpr std::uint16_t kFabricVersion = 2;

inline constexpr std::size_t kFabricRequestHeaderBytes = 24;
inline constexpr std::size_t kFabricStreamHeaderBytes = 16;
inline constexpr std::size_t kFabricDoneTailBytes = 32;  // flags4+total4+3*f64
inline constexpr std::size_t kFabricCancelBytes = 12;
inline constexpr std::size_t kFabricPingBytes = 12;

enum class WireStatus {
    kOk,
    kBadMagic,
    kBadVersion,
    kTruncated,
    kTrailing,
    kBadType,
};

enum class FabricFrameType : std::uint16_t {
    kTokenBatch = 1,
    kDone = 2,
    kError = 3,
};

/// Non-owning decoded request. `tokens` aliases the caller's frame buffer.
struct FabricRequestView {
    std::uint32_t request_id = 0;
    std::uint32_t max_gen = 0;  // 0 => server default
    std::span<const std::int32_t> tokens;
};

/// Owned decoded stream frame. Exactly one of the payload members is populated
/// according to `type`.
struct FabricStreamFrame {
    FabricFrameType type = FabricFrameType::kTokenBatch;
    std::uint32_t request_id = 0;
    std::vector<std::int32_t> tokens;  // kTokenBatch
    bool eos_terminated = false;       // kDone
    std::uint32_t total_tokens = 0;    // kDone
    double ttft_ms = 0.0;              // kDone
    double total_ms = 0.0;             // kDone
    double tokens_per_sec = 0.0;       // kDone
    std::string error;                 // kError
};

std::vector<std::byte> EncodeFabricRequest(std::uint32_t request_id,
                                           std::span<const std::int32_t> tokens,
                                           std::uint32_t max_gen);
WireStatus DecodeFabricRequest(std::span<const std::byte> frame,
                               FabricRequestView& out);

std::vector<std::byte> EncodeFabricTokenBatch(
    std::uint32_t request_id, std::span<const std::int32_t> tokens);
/// Same, writing into a caller-owned buffer (no per-call allocation): the
/// server's per-request token flush path reuses one buffer.
void EncodeFabricTokenBatch(std::vector<std::byte>& out,
                            std::uint32_t request_id,
                            std::span<const std::int32_t> tokens);
std::vector<std::byte> EncodeFabricDone(std::uint32_t request_id,
                                        bool eos_terminated,
                                        std::uint32_t total_tokens,
                                        double ttft_ms, double total_ms,
                                        double tokens_per_sec);
std::vector<std::byte> EncodeFabricError(std::uint32_t request_id,
                                         std::string_view message);
WireStatus DecodeFabricStreamFrame(std::span<const std::byte> frame,
                                   FabricStreamFrame& out);

std::vector<std::byte> EncodeFabricCancel(std::uint32_t request_id);
WireStatus DecodeFabricCancel(std::span<const std::byte> frame,
                              std::uint32_t& request_id);

/// Health-probe frames (12 bytes: magic, version, pad, nonce). The server
/// answers a ping with a pong carrying the same nonce on the same route, so a
/// client can verify a live backend without occupying the inference path.
std::vector<std::byte> EncodeFabricPing(std::uint32_t nonce);
std::vector<std::byte> EncodeFabricPong(std::uint32_t nonce);
WireStatus DecodeFabricPing(std::span<const std::byte> frame,
                            std::uint32_t& nonce);
WireStatus DecodeFabricPong(std::span<const std::byte> frame,
                            std::uint32_t& nonce);

}  // namespace lsxnetwork
