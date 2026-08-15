// lsxhome — Blackwell matte/Cowork theme styling profile (Phase 1).
//
// Owns the hardlocked ImGuiStyle and the docking-grid skeleton render pass.
// Zero-allocation: the theme touches style state only, and the workspace
// skeleton reuses the existing viewport — no per-frame allocations.

#include "lsxhome/gui_renderer.h"

#include <imgui.h>

namespace lsxhome {
namespace {

constexpr ImVec4 MakeVec4(float r, float g, float b, float a = 1.0f) noexcept {
    return ImVec4(r, g, b, a);
}

}  // namespace

void ApplyBlackwellCoworkTheme() noexcept {
    ImGuiStyle& style = ImGui::GetStyle();

    // Hardlocked matte Blackwell palette.
    style.Colors[ImGuiCol_WindowBg] = MakeVec4(0.08f, 0.08f, 0.09f);
    style.Colors[ImGuiCol_ChildBg] = MakeVec4(0.10f, 0.10f, 0.12f);
    style.Colors[ImGuiCol_Text] = MakeVec4(0.88f, 0.88f, 0.90f);
    style.Colors[ImGuiCol_TextDisabled] = MakeVec4(0.58f, 0.58f, 0.62f);

    // Cowork geometry: frame and window border padding, all rounding at 6.0f.
    style.FramePadding = ImVec2(6.0f, 6.0f);
    style.WindowPadding = ImVec2(6.0f, 6.0f);
    style.ItemSpacing = ImVec2(6.0f, 6.0f);

    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;

    style.WindowRounding = 6.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 6.0f;
}

void BuildWorkspaceSkeleton() noexcept {
    ImGui::DockSpaceOverViewport();
}

}  // namespace lsxhome
