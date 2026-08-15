# lsxhome — LogestiX Home, Windows-native MVP desktop shell.
#
# Hardware-accelerated GUI (D3D11 + Dear ImGui) consuming the lock-free SPSC
# token bridge from the GPU compute thread. Windows-only: the whole block is
# guarded so Linux presets keep building the headless apps unchanged.

if(NOT WIN32)
    return()
endif()

include(FetchContent)

# ── Dear ImGui (Windows GUI shell only) ─────────────────────────────────────
set(IMGUI_TAG v1.91.3 CACHE STRING "Dear ImGui tag for lsxhome")
FetchContent_Declare(
    dear-imgui
    GIT_REPOSITORY "https://github.com/ocornut/imgui.git"
    GIT_TAG        "${IMGUI_TAG}"
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(dear-imgui)
if(NOT dear-imgui_POPULATED)
    FetchContent_Populate(dear-imgui)
endif()

add_library(lsxhome_imgui STATIC
    ${dear-imgui_SOURCE_DIR}/imgui.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_draw.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_tables.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${dear-imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    ${dear-imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)
target_include_directories(lsxhome_imgui PUBLIC
    ${dear-imgui_SOURCE_DIR}
    ${dear-imgui_SOURCE_DIR}/backends
)

# ── lsxhome executable (WIN32 subsystem, no console) ────────────────────────
add_executable(lsxhome WIN32
    src/main_win32.cpp
    src/d3d11_renderer.cpp
)
target_include_directories(lsxhome PRIVATE include)
target_compile_definitions(lsxhome PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(lsxhome PRIVATE
    lsxhome_imgui
    lsxcommon
    arrow_shared
    arrow_cuda_shared
    d3d11
    dxgi
    dwmapi
)

# arrow.dll / arrow_cuda.dll next to the exe (same POST_BUILD pattern as the
# other four applications).
add_custom_command(TARGET lsxhome POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:arrow_shared>"
        "$<TARGET_FILE_DIR:lsxhome>"
)
add_custom_command(TARGET lsxhome POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:arrow_cuda_shared>"
        "$<TARGET_FILE_DIR:lsxhome>"
)
# lsxhome_imgui static lib doesn't need arrow; nothing to copy for it.

install(TARGETS lsxhome RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()