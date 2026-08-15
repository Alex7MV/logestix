#pragma once

#include "lsxhome/gui_bridge.h"

namespace lsxhome {

/// Applies the high-fidelity matte Blackwell/Cowork theme to the active
/// Dear ImGui style. Hardlocked palette and geometry (see gui_renderer.cpp).
void ApplyBlackwellCoworkTheme() noexcept;

/// Renders the full-viewport edge-to-edge dock root, then the seamless
/// sidebar / Token Throughput panel blocks hosted inside it.
void BuildWorkspaceSkeleton(GuiBridge& bridge) noexcept;

}  // namespace lsxhome
