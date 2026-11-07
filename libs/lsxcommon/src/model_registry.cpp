#include <memory>

#include "lsxcommon/model_factory.h"

// The registration list is generated from models/*.json at configure time
// (tools/gen_model_manifest.py --registry). A model that exists as a manifest
// row and a class is therefore always reachable through LsxModelFactory; the
// list can no longer fall behind, which is what "that model does not exist"
// looked like before. tests/test_manifest_sync.cpp proves the list reaches
// every row, and each alias resolves to its owner's class.
//
// Included at FILE scope on purpose: the per-model headers cannot be expanded
// inside a function body.
#include "model_registry.inc"

namespace lsxcommon {

void EnsureModelRegistry()
{
    static bool init = [] {
        RegisterAllModels();
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon
