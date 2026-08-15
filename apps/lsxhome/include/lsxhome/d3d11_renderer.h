#pragma once

struct HWND__;
typedef HWND__* HWND;

namespace lsxhome {

/// DirectX 11 hardware-accelerated renderer for the lsxhome desktop shell.
///
/// Owns the D3D11 device, context, swap chain and render target view, plus
/// the Dear ImGui frame plumbing (ImGui_ImplWin32 + ImGui_ImplDX11). The UI
/// thread drives the full frame; compute results arrive only through the
/// lock-free `GuiBridge`, never through direct renderer access.
class D3D11Renderer {
public:
    static constexpr int kBackBufferCount = 2;
    static constexpr int kSrgbFormatSwiz = 2;  // DXGI_FORMAT_R8G8B8A8_UNORM

    D3D11Renderer() noexcept = default;
    ~D3D11Renderer() noexcept { Shutdown(); }

    D3D11Renderer(const D3D11Renderer&)            = delete;
    D3D11Renderer& operator=(const D3D11Renderer&) = delete;

    /// Creates the device, swap chain and ImGui context for @p hwnd.
    /// Returns 0 on success, -1 on any D3D/ImGui init failure.
    int Init(HWND hwnd, int width, int height) noexcept;
    void Shutdown() noexcept;

    /// Begins an ImGui frame: clears the back buffer, updates the device.
    /// Returns false if the swap chain is in a discard-resized state.
    bool BeginFrame() noexcept;
    void EndFrame(bool present) noexcept;

    /// Handles WM_SIZE -> swaps to the new client size (recreates RT view).
    void Resize(int width, int height) noexcept;

    int  Width() const noexcept { return width_; }
    int  Height() const noexcept { return height_; }
    bool Initialized() const noexcept { return initialized_; }

private:
    HWND    hwnd_      = nullptr;
    void*   device_    = nullptr;  // ID3D11Device*
    void*   ctx_       = nullptr;  // ID3D11DeviceContext*
    void*   swapchain_ = nullptr;  // IDXGISwapChain*
    void*   rtv_       = nullptr;  // ID3D11RenderTargetView*
    int     width_     = 1280;
    int     height_    = 800;
    bool    initialized_ = false;
};

}  // namespace lsxhome