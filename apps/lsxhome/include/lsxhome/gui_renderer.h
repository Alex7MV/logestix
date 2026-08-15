#pragma once

namespace lsxhome {

/// Applies the high-fidelity matte Blackwell/Cowork theme to the active
/// Dear ImGui style. Hardlocked palette and geometry (see gui_renderer.cpp).
void ApplyBlackwellCoworkTheme() noexcept;

/// Establishes the mosaic docking-grid backbone via DockSpaceOverViewport,
/// preparing the window tiles that host token text array blocks.
void BuildWorkspaceSkeleton() noexcept;

}  // namespace lsxhome
