#include "lsxhome/d3d11_renderer.h"

#include "backends/imgui_impl_dx11.h"
#include "backends/imgui_impl_win32.h"
#include <imgui.h>

#include <d3d11.h>
#include <windows.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace lsxhome {

namespace {

/// Creates the D3D11 device + swap chain for the given HWND (Blackwell target,
/// FL 11_0+, BGRA support so ImGui can blit its draw lists).
bool CreateDeviceAndSwapChain(HWND hwnd, int width, int height,
                              ID3D11Device** device, ID3D11DeviceContext** ctx,
                              IDXGISwapChain** swapchain) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferDesc.Width = static_cast<UINT>(width);
    sd.BufferDesc.Height = static_cast<UINT>(height);
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate = {60, 1};
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = D3D11Renderer::kBackBufferCount;
    sd.OutputWindow = hwnd;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    const D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0,
    };
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, feature_levels, 2,
        D3D11_SDK_VERSION, &sd, swapchain, device, nullptr, ctx);
    return SUCCEEDED(hr);
}

bool CreateRenderTargetView(ID3D11Device* device, IDXGISwapChain* swapchain,
                            ID3D11RenderTargetView** rtv) {
    ComPtr<ID3D11Texture2D> backbuffer;
    if (FAILED(swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer)))) {
        return false;
    }
    return SUCCEEDED(
        device->CreateRenderTargetView(backbuffer.Get(), nullptr, rtv));
}

void ReleaseImGui() noexcept {
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

}  // namespace

int D3D11Renderer::Init(HWND hwnd, int width, int height) noexcept {
    if (initialized_) {
        Shutdown();
    }
    hwnd_ = hwnd;
    width_ = width;
    height_ = height;

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    IDXGISwapChain* swapchain = nullptr;
    if (!CreateDeviceAndSwapChain(hwnd_, width_, height_, &device, &ctx,
                                  &swapchain)) {
        return -1;
    }

    ID3D11RenderTargetView* rtv = nullptr;
    if (!CreateRenderTargetView(device, swapchain, &rtv)) {
        swapchain->Release();
        ctx->Release();
        device->Release();
        return -1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    if (!ImGui_ImplWin32_Init(hwnd_) || !ImGui_ImplDX11_Init(device, ctx)) {
        ReleaseImGui();
        rtv->Release();
        swapchain->Release();
        ctx->Release();
        device->Release();
        return -1;
    }

    device_ = device;
    ctx_ = ctx;
    swapchain_ = swapchain;
    rtv_ = rtv;
    initialized_ = true;
    return 0;
}

void D3D11Renderer::Shutdown() noexcept {
    if (!initialized_) {
        return;
    }
    ReleaseImGui();
    if (rtv_) {
        reinterpret_cast<ID3D11RenderTargetView*>(rtv_)->Release();
        rtv_ = nullptr;
    }
    if (swapchain_) {
        reinterpret_cast<IDXGISwapChain*>(swapchain_)->Release();
        swapchain_ = nullptr;
    }
    if (ctx_) {
        reinterpret_cast<ID3D11DeviceContext*>(ctx_)->Release();
        ctx_ = nullptr;
    }
    if (device_) {
        reinterpret_cast<ID3D11Device*>(device_)->Release();
        device_ = nullptr;
    }
    initialized_ = false;
}

bool D3D11Renderer::BeginFrame() noexcept {
    if (!initialized_) {
        return false;
    }
    const float clear_color[4] = {0.043f, 0.043f, 0.063f, 1.0f};
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(ctx_);
    auto* rtv = reinterpret_cast<ID3D11RenderTargetView*>(rtv_);
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    ctx->ClearRenderTargetView(rtv, clear_color);

    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    return true;
}

void D3D11Renderer::EndFrame(bool present) noexcept {
    if (!initialized_) {
        return;
    }
    ImGui::Render();
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    if (present) {
        reinterpret_cast<IDXGISwapChain*>(swapchain_)->Present(1, 0);  // vsync
    }
}

void D3D11Renderer::Resize(int width, int height) noexcept {
    if (!initialized_ || width <= 0 || height <= 0) {
        return;
    }
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(ctx_);
    auto* swapchain = reinterpret_cast<IDXGISwapChain*>(swapchain_);
    auto* rtv = reinterpret_cast<ID3D11RenderTargetView*>(rtv_);
    auto* device = reinterpret_cast<ID3D11Device*>(device_);

    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    if (rtv) {
        rtv->Release();
        rtv_ = nullptr;
    }
    if (FAILED(swapchain->ResizeBuffers(0, static_cast<UINT>(width),
                                        static_cast<UINT>(height),
                                        DXGI_FORMAT_UNKNOWN, 0))) {
        return;
    }
    ID3D11RenderTargetView* new_rtv = nullptr;
    if (CreateRenderTargetView(device, swapchain, &new_rtv)) {
        rtv_ = new_rtv;
    }
    width_ = width;
    height_ = height;
}

}  // namespace lsxhome