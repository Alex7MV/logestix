#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace lsxnetwork {
namespace spa {

/// A statically-embedded SPA asset served from read-only memory.
///
/// `data` points at an immutable, 64-byte cache-line aligned array produced by
/// the CMake step (see `lsx_embedded_www.h`). Because the asset table is a
/// compile-time constant, the static router can serve concurrent requests with
/// zero locking and zero per-request allocation — the body is written straight
/// from this buffer.
struct Asset {
    const char* path;           ///< URL path, e.g. "/" or "/css/styles.css".
    const std::uint8_t* data;   ///< 64-byte aligned immutable bytes.
    std::size_t size;           ///< byte length of `data`.
    const char* mime;           ///< HTTP Content-Type, e.g. "text/html; charset=utf-8".
};

}  // namespace spa
}  // namespace lsxnetwork
