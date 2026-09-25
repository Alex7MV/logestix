#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace lsxnetwork {

/// Binary token protocol (little-endian, explicit byte-wise encoding):
///   request  = 16-byte header + 4 * num_tokens bytes
///   response = 40-byte header + (tokens | utf-8 error) bytes
inline constexpr std::uint32_t kFabricMagic = 0x4246584Cu;  // "LXFB" LE
inline constexpr std::uint16_t kFabricVersion = 1;

inline constexpr std::size_t kFabricRequestHeaderBytes = 16;
inline constexpr std::size_t kFabricResponseHeaderBytes = 40;

enum class FabricStatus : std::uint16_t {
    kOk = 0,
    kBadRequest = 1,
    kInferFailed = 2,
    kInternal = 3,
};

/// Codec verdict. kOk means the frame was accepted; every other value maps to
/// a `kBadRequest` reply on the server side.
enum class WireStatus {
    kOk,
    kBadMagic,
    kBadVersion,
    kTruncated,
    kTrailing,
};

/// Non-owning view into a decoded request frame. `tokens` aliases the caller's
/// frame buffer, which must outlive the view.
struct FabricRequestView {
    std::uint32_t max_gen = 0;  // 0 => server default
    std::span<const std::int32_t> tokens;
};

/// Owned response. On status != kOk, `tokens` is empty and `error` carries a
/// UTF-8 message (possibly empty).
struct FabricResponse {
    std::int32_t status = static_cast<std::int32_t>(FabricStatus::kOk);
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    double tokens_per_sec = 0.0;
    std::vector<std::int32_t> tokens;
    std::string error;
};

std::vector<std::byte> EncodeFabricRequest(std::span<const std::int32_t> tokens,
                                           std::uint32_t max_gen);

WireStatus DecodeFabricRequest(std::span<const std::byte> frame,
                               FabricRequestView& out);

std::vector<std::byte> EncodeFabricResponse(const FabricResponse& r);

WireStatus DecodeFabricResponse(std::span<const std::byte> frame,
                                FabricResponse& out);

}  // namespace lsxnetwork
