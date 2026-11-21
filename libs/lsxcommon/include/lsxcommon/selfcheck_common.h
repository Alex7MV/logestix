// selfcheck_common.h — helpers shared by the FSI/CODE/DET parity harnesses
// (dedup phase 5). Unfence was a byte-identical copy in fsi_selfcheck.cpp,
// code_selfcheck.cpp and det_selfcheck.cpp.
#pragma once

#include <string>
#include <string_view>

namespace lsxcommon {
namespace selfcheck {

// Strips surrounding whitespace and one markdown code fence (```lang ... ```),
// keeping the inner content. An unterminated fence keeps everything after the
// opener; a first line that is not a bare language tag stays in the content.
std::string Unfence(std::string_view raw);

} // namespace selfcheck
} // namespace lsxcommon