// lsxhome (LogestiX Home) — Windows-native MVP desktop shell.
//
// Lifecycle: WinMain -> native window -> D3D11 hardware renderer + Dear ImGui;
// a compute thread publishes decoded tokens into the lock-free SPSC GuiBridge;
// the UI thread drains them every frame and renders the Cowork-style Blackwell
// theme. Strict separation: the UI thread never touches the model, and the
// compute thread never touches the window.

#include "backends/imgui_impl_win32.h"
#include "lsxhome/blackwell_theme.h"
#include "lsxhome/d3d11_renderer.h"
#include "lsxhome/gui_bridge.h"

#include <windows.h>

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>

// Dear ImGui's Win32 backend keeps WndProcHandler intentionally commented out;
// the application is responsible for this exact forward declaration.
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

constexpr const wchar_t* kWindowClass = L"LogestiXHomeWindow";
constexpr const wchar_t* kWindowTitle = L"lsxhome — LogestiX Home";

constexpr int kDefaultWidth = 1440;
constexpr int kDefaultHeight = 900;

// GLM-5.2 self-check token: 11751 decodes to "Paris".
constexpr std::uint32_t kParisTokenId = 11751;

lsxhome::D3D11Renderer* g_renderer = nullptr;

// ImGui's Win32 backend needs to see mouse/keyboard messages; forward them
// through this message pump before the switch chain below.


LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (g_renderer && g_renderer->Initialized()) {
        if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) {
            return 1;
        }
    }
    if (msg == WM_SIZE && g_renderer && wparam != SIZE_MINIMIZED) {
        g_renderer->Resize(LOWORD(lparam), HIWORD(lparam));
        return 0;
    }
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

void ComputeProducer(lsxhome::GuiBridge& bridge, std::atomic<bool>& running) {
    std::uint32_t seq = 0;
    while (running.load(std::memory_order_relaxed)) {
        lsxhome::TokenPayload t =
            lsxhome::MakeTokenPayload(kParisTokenId, seq++, "Paris");
        // Backpressure contract: dropping on a full pool is deterministic and
        // never blocks the compute stream.
        bridge.Publish(t);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        if ((seq & 7u) == 0) {
            for (int i = 0; i < 8; ++i) {
                bridge.Publish(t);
            }
        }
    }
}

void ApplyBlackwellTheme() noexcept {
    using namespace lsxhome::theme;

    auto as_vec4 = [](Color argb) {
        const float scale = 1.0f / 255.0f;
        return ImVec4(((argb >> 24) & 0xFF) * scale,  // r
                      ((argb >> 16) & 0xFF) * scale,  // g
                      ((argb >> 8) & 0xFF) * scale,   // b
                      (argb & 0xFF) * scale);         // a
    };

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(kGeometry.padding, kGeometry.padding);
    style.FramePadding = ImVec2(kGeometry.frame_padding, kGeometry.frame_padding);
    style.ItemSpacing = ImVec2(kGeometry.item_spacing, kGeometry.item_spacing);
    style.FrameRounding = kGeometry.rounding;
    style.ScrollbarSize = kGeometry.scrollbar_size;

    const BlackwellPalette& p = kPalette;
    style.Colors[ImGuiCol_WindowBg] = as_vec4(p.window_bg);
    style.Colors[ImGuiCol_ChildBg] = as_vec4(p.window_bg_alt);
    style.Colors[ImGuiCol_PopupBg] = as_vec4(p.window_bg);
    style.Colors[ImGuiCol_Border] = as_vec4(p.border);
    style.Colors[ImGuiCol_FrameBg] = as_vec4(p.field_bg);
    style.Colors[ImGuiCol_FrameBgHovered] = as_vec4(p.field_hover);
    style.Colors[ImGuiCol_Header] = as_vec4(p.field_hover);
    style.Colors[ImGuiCol_Text] = as_vec4(p.text);
    style.Colors[ImGuiCol_TextDisabled] = as_vec4(p.text_disabled);
    style.Colors[ImGuiCol_TextSelectedBg] = as_vec4(p.accent);
    style.Colors[ImGuiCol_Button] = as_vec4(p.accent);
    style.Colors[ImGuiCol_ButtonHovered] = as_vec4(p.accent_hover);
}

}  // namespace

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow) {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW);
    wc.lpszClassName = kWindowClass;
    if (!RegisterClassExW(&wc)) {
        return 1;
    }

    HWND hwnd = CreateWindowExW(
        0, kWindowClass, kWindowTitle, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, kDefaultWidth, kDefaultHeight,
        nullptr, nullptr, hInst, nullptr);
    if (!hwnd) {
        return 2;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    lsxhome::D3D11Renderer renderer;
    if (renderer.Init(hwnd, kDefaultWidth, kDefaultHeight) != 0) {
        return 3;
    }
    g_renderer = &renderer;

    lsxhome::GuiBridge bridge;
    std::atomic<bool> running{true};
    std::thread compute_thread(ComputeProducer, std::ref(bridge), std::ref(running));

    MSG msg = {};
    while (msg.message != WM_QUIT) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (msg.message == WM_QUIT) {
            break;
        }

        if (!renderer.BeginFrame()) {
            continue;
        }
        ApplyBlackwellTheme();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(renderer.Width()),
                                        static_cast<float>(renderer.Height())));
        constexpr ImGuiWindowFlags kFlags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus;
        if (ImGui::Begin("lsxhome", nullptr, kFlags)) {
            constexpr float kSidebarW = 300.0f;
            ImGui::BeginChild("##sidebar", ImVec2(kSidebarW, 0), true);
            {
                const auto& palette = lsxhome::theme::kPalette;
                const float accent_r = ((palette.accent >> 24) & 0xFF) / 255.f;
                const float accent_g = ((palette.accent >> 16) & 0xFF) / 255.f;
                const float accent_b = ((palette.accent >> 8) & 0xFF) / 255.f;
                ImGui::TextColored(ImVec4(accent_r, accent_g, accent_b, 1.0f),
                                   "LogestiX Home");
                ImGui::Separator();
                ImGui::TextUnformatted("Sessions");
                ImGui::EndChild();
            }

            ImGui::SameLine();
            ImGui::BeginChild("##main", ImVec2(0, 0), true);
            {
                if (ImGui::Button("Drain pool")) {
                    lsxhome::TokenPayload buf[256];
                    const std::size_t n = bridge.DrainAll(buf, 256);
                    for (std::size_t i = 0; i < n; ++i) {
                        spdlog::info("token id={} seq={} text={}", buf[i].token_id,
                                     buf[i].stream_seq, buf[i].text);
                    }
                }
                ImGui::Text("Published: %zu\nDropped: %zu\nPoolSlots: %zu",
                            bridge.Published(), bridge.Dropped(),
                            lsxhome::GuiBridge::kPoolSlots);
                ImGui::Separator();
                ImGui::TextUnformatted("Compute -> UI SPSC ring is live.");
                ImGui::EndChild();
            }
        }
        ImGui::End();

        renderer.EndFrame(true);
    }

    running.store(false, std::memory_order_relaxed);
    if (compute_thread.joinable()) {
        compute_thread.join();
    }

    renderer.Shutdown();
    g_renderer = nullptr;
    UnregisterClassW(kWindowClass, hInst);
    return static_cast<int>(msg.wParam);
}