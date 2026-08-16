// lsxhome (LogestiX Home) — Windows-native x64 desktop shell.
//
// Lifecycle: WinMain -> native window -> D3D12 hardware renderer + Dear ImGui
// (docking branch); a compute thread publishes decoded tokens into the
// lock-free SPSC GuiBridge; the UI thread drains them every frame and renders
// the Cowork-style Blackwell theme. Strict separation: the UI thread never
// touches the model, and the compute thread never touches the window.

#include "backends/imgui_impl_win32.h"
#include <imgui.h>

#include "lsxhome/d3d12_renderer.h"
#include "lsxhome/font_loader.h"
#include "lsxhome/gui_bridge.h"
#include "lsxhome/gui_renderer.h"

#include "lsxcommon/generation_cli.h"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// Dear ImGui's Win32 backend keeps WndProcHandler intentionally commented out;
// the application is responsible for this exact forward declaration.
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Generation path execution option: --run/-r routes the resolved prompt into
// the shell's token stream. A bare --run/-r (no value) falls back to the
// default blueprint prompt via lsxcommon::cli.
ABSL_FLAG(std::string, run, "", "generation prompt (--run \"PROMPT\" / -r \"PROMPT\")");

namespace {

constexpr const wchar_t* kWindowClass = L"LogestiXHomeWindow";
constexpr const wchar_t* kWindowTitle = L"lsxhome — LogestiX Home";

constexpr int kDefaultWidth = 1440;
constexpr int kDefaultHeight = 900;

// GLM-5.2 self-check token: 11751 decodes to "Paris".
constexpr std::uint32_t kParisTokenId = 11751;

lsxhome::D3D12Renderer* g_renderer = nullptr;

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

void ComputeProducer(lsxhome::GuiBridge& bridge, std::atomic<bool>& running,
                     const std::string& seed_prompt) {
    std::uint32_t seq = 0;
    while (running.load(std::memory_order_relaxed)) {
        // Route the generation prompt (resolved via --run/-r) into the shell's
        // token stream: the first payload carries the prompt text instead of
        // the demo token.
        const char* text = seed_prompt.empty() ? "Paris" : seed_prompt.c_str();
        lsxhome::TokenPayload t =
            lsxhome::MakeTokenPayload(kParisTokenId, seq++, text);
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

}  // namespace

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow) {
    // Canonicalize the generation-path flag (bare --run/-r -> default prompt)
    // before absl::ParseCommandLine consumes argv.
    auto normalized = lsxcommon::cli::NormalizeGenerationArgv(__argc, __argv);
    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(normalized.size());
    for (auto& arg : normalized) argv_ptrs.push_back(arg.data());
    absl::ParseCommandLine(static_cast<int>(argv_ptrs.size()), argv_ptrs.data());

    std::string run_prompt =
        lsxcommon::cli::ResolveGenerationPrompt(absl::GetFlag(FLAGS_run));

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

    lsxhome::D3D12Renderer renderer;
    if (renderer.Init(hwnd, kDefaultWidth, kDefaultHeight) != 0) {
        return 3;
    }
    g_renderer = &renderer;

    // Docking + multithreaded renderer are both required pieces of the shell.
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

    // imgui_freetype subpixel rasterizer is configured by the loader; the base
    // font comes from the embedded static TTF payload.
    lsxhome::FontLoader::LoadDefault(*io.Fonts);

    lsxhome::GuiBridge bridge;
    std::atomic<bool> running{true};
    std::thread compute_thread(
        ComputeProducer, std::ref(bridge), std::ref(running), run_prompt);

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
        lsxhome::ApplyBlackwellCoworkTheme();

        // Edge-to-edge docking root + seamless sidebar/panel layout.
        lsxhome::BuildWorkspaceSkeleton(bridge);

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