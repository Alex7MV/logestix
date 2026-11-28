#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsxfabric {

// Renders WHY a model would load the profile it would load (design §2.7).
//
// One line per candidate of `<org>/<model>/`, with:
//   * the profile name and the codec mask artifact.fb declares,
//   * the verdict against the target's capability (`supported`),
//   * its rank in `preference` (0 = most desirable),
// and a final `chosen:` line naming the profile — or `no compatible profile`,
// when nothing on disk fits.
//
// `*rc` is 0 only when a profile would be chosen. An explicit profile path is a
// choice and is reported as such rather than re-ranked, so the flag answers the
// same question the loader would.
std::string ExplainArtifact(const std::string &model_or_path, uint64_t supported,
                            const std::vector<std::string> &preference,
                            int *rc);

} // namespace lsxfabric
