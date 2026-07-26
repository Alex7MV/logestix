#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace gpu {

bool quantize_tensor_gpu(const uint8_t *raw_data, int64_t numel,
                         const std::string &dtype,
                         std::vector<uint8_t> &packed, float &scale);

}
} // namespace lsxcommon
