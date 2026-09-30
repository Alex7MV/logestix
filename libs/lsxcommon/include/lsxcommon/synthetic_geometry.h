#pragma once

// Test-only escape hatch, shared by every converter, loader and ModelBase that
// hard-gates against frozen geometry: synthetic fixtures use tiny geometry,
// production runs never set this. Canonical value "1".
//
// Host-only, model-neutral (no CUDA): safe to include from plain C++ TUs.
// Supersedes the per-family copies of the same predicate.

#include <cstdlib>

namespace lsxcommon {

inline bool SyntheticGeometryAllowed()
{
    const char *env = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    return env && env[0] == '1';
}

} // namespace lsxcommon
