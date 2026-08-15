// lsxhome — Blackwell matte/Cowork theme styling profile (Phase 1).
//
// Owns the hardlocked ImGuiStyle and the edge-to-edge docking-grid skeleton
// render pass. Zero-allocation: the theme touches style state only, and the
// workspace skeleton reuses the existing viewport — no per-frame allocations.

#include "lsxhome/gui_renderer.h"

#include <imgui.h>
#include <implot.h>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdint>

namespace lsxhome {
namespace {

constexpr ImVec4 MakeVec4(float r, float g, float b, float a = 1.0f) noexcept {
    return ImVec4(r, g, b, a);
}

constexpr float kSidebarWidth = 300.0f;
constexpr int kPlotSamples = 256;

}  // namespace

void ApplyBlackwellCoworkTheme() noexcept {
    ImGuiStyle& style = ImGui::GetStyle();

    // Hardlocked matte Blackwell palette. Strict dark graphite panels, thin
    // dark-gray borders, deep matte-charcoal chrome and soft ivory text — no
    // blue tones remain anywhere.
    style.Colors[ImGuiCol_WindowBg] = MakeVec4(0.07f, 0.07f, 0.08f, 1.00f);
    style.Colors[ImGuiCol_ChildBg] = MakeVec4(0.11f, 0.11f, 0.12f, 1.00f);
    style.Colors[ImGuiCol_Border] = MakeVec4(0.18f, 0.18f, 0.20f, 1.00f);
    style.Colors[ImGuiCol_Button] = MakeVec4(0.14f, 0.14f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_ButtonHovered] = MakeVec4(0.20f, 0.20f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive] = MakeVec4(0.24f, 0.24f, 0.26f, 1.00f);
    style.Colors[ImGuiCol_Header] = MakeVec4(0.14f, 0.14f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_HeaderHovered] = MakeVec4(0.20f, 0.20f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_HeaderActive] = MakeVec4(0.24f, 0.24f, 0.26f, 1.00f);
    style.Colors[ImGuiCol_TitleBg] = MakeVec4(0.14f, 0.14f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_TitleBgActive] = MakeVec4(0.14f, 0.14f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = MakeVec4(0.14f, 0.14f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_Text] = MakeVec4(0.88f, 0.88f, 0.90f, 1.00f);
    style.Colors[ImGuiCol_TextDisabled] = MakeVec4(0.58f, 0.58f, 0.62f, 1.00f);

    // Cowork geometry: square main viewport, softly rounded panels and frames,
    // zero window chrome so the dock host fuses edge-to-edge with the screen.
    style.WindowRounding = 0.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.WindowBorderSize = 0.0f;
    style.ChildBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(8.0f, 8.0f);

    style.FramePadding = ImVec2(6.0f, 6.0f);
    style.WindowPadding = ImVec2(6.0f, 6.0f);
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 6.0f;
}

void BuildWorkspaceSkeleton(GuiBridge& bridge) noexcept {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    // Edge-to-edge root dock host. Covers the full viewport with zero chrome
    // and zero rounding, so no title bar, collapsing triangle, border, or
    // floating window can ever escape the graphite backdrop.
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const ImGuiWindowFlags host_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    ImGui::Begin("##LSXHomeDockHost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    // Lock the panels together across the full viewport. Passthru keeps the
    // dock transparent so the host's seamless children stay fully visible.
    ImGui::DockSpaceOverViewport(0, viewport,
                                 ImGuiDockNodeFlags_PassthruCentralNode);

    // Sidebar: matte charcoal panel, seamlessly inset into the root block.
    ImGui::BeginChild("##sidebar", ImVec2(kSidebarWidth, 0.0f), false);
    {
        ImGui::TextUnformatted("LogestiX Home");
        ImGui::Separator();
        ImGui::TextUnformatted("Sessions");

        ImGui::Separator();
        if (ImGui::Button("Drain pool")) {
            TokenPayload buf[256];
            const std::size_t n = bridge.DrainAll(buf, 256);
            for (std::size_t i = 0; i < n; ++i) {
                spdlog::info("token id={} seq={} text={}", buf[i].token_id,
                             buf[i].stream_seq, buf[i].text);
            }
        }
        ImGui::Text("Published: %zu\nDropped: %zu\nPoolSlots: %zu",
                    bridge.Published(), bridge.Dropped(),
                    GuiBridge::kPoolSlots);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Main content: Token Throughput ImPlot panel, seamless sibling block.
    ImGui::BeginChild("##main", ImVec2(0.0f, 0.0f), false);
    {
        static float ys[256];
        static int idx = 0;
        ys[idx] = static_cast<float>(bridge.Published());
        idx = (idx + 1) % kPlotSamples;
        if (ImPlot::BeginPlot("Token Throughput")) {
            ImPlot::PlotLine("published", ys, kPlotSamples);
            ImPlot::EndPlot();
        }
        ImGui::Separator();
        ImGui::TextUnformatted("Compute -> UI SPSC ring is live.");
    }
    ImGui::EndChild();

    ImGui::End();
}

}  // namespace lsxhome