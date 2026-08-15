# lsxhome — LogestiX Home, Windows-native x64 desktop shell.
#
# Hardware-accelerated GUI (D3D12 immediate-mode stack) consuming the lock-free
# SPSC token bridge from the GPU compute thread. Windows-only: the whole block
# is guarded so Linux presets keep building the headless apps unchanged.
#
# GUI stack (third-party, vendored as SEPARATE static archives — MIT):
#   Dear ImGui  (docking branch)  — core + Win32 platform + D3D12 renderer
#   ImPlot                        — real-time charting channels
#   ImNodes                       — graph-layout node editor
#   FreeType 2.14.3               — imgui_freetype rasterizer, LCD subpixel AA

if(NOT WIN32)
    return()
endif()

include(FetchContent)

# ── Dear ImGui docking branch (D3D12 + FreeType subpixel) ───────────────────
set(LSXHOME_IMGUI_TAG v1.92.9-docking CACHE STRING "Dear ImGui docking-branch tag")
FetchContent_Declare(
    dear-imgui
    GIT_REPOSITORY "https://github.com/ocornut/imgui.git"
    GIT_TAG        "${LSXHOME_IMGUI_TAG}"
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(dear-imgui)
if(NOT dear-imgui_POPULATED)
    FetchContent_Populate(dear-imgui)
endif()

# ── ImPlot (real-time charting channels) ────────────────────────────────────
set(LSXHOME_IMPLOT_TAG v0.17 CACHE STRING "ImPlot tag")
FetchContent_Declare(
    implot
    GIT_REPOSITORY "https://github.com/epezent/implot.git"
    GIT_TAG        "${LSXHOME_IMPLOT_TAG}"
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(implot)
if(NOT implot_POPULATED)
    FetchContent_Populate(implot)
endif()

# ── ImNodes (graph-layout node editor) ──────────────────────────────────────
set(LSXHOME_IMNODES_TAG v0.5 CACHE STRING "ImNodes tag")
FetchContent_Declare(
    imnodes
    GIT_REPOSITORY "https://github.com/Nelarius/imnodes.git"
    GIT_TAG        "${LSXHOME_IMNODES_TAG}"
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(imnodes)
if(NOT imnodes_POPULATED)
    FetchContent_Populate(imnodes)
endif()

# ── FreeType (imgui_freetype abstraction layer, LCD subpixel AA) ────────────
set(LSXHOME_FREETYPE_TAG VER-2-14-3 CACHE STRING "FreeType tag")
FetchContent_Declare(
    freetype
    GIT_REPOSITORY "https://gitlab.freedesktop.org/freetype/freetype.git"
    GIT_TAG        "${LSXHOME_FREETYPE_TAG}"
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(freetype)
if(NOT freetype_POPULATED)
    FetchContent_Populate(freetype)
endif()

# FreeType ships FT_CONFIG_OPTION_SUBPIXEL_RENDERING commented out inside
# ftoption.h (was disabled for patents). imgui_freetype baboon-fed subpixel
# glyphs need pixel-LCD rendering, so unmute the define — the same patch-
# after-fetch pattern as the Arrow/Boost patch in the root CMakeLists.txt.
set(FT_CONFIG_OPTION_SUBPIXEL_FILE "${freetype_SOURCE_DIR}/include/freetype/config/ftoption.h")
if(EXISTS "${FT_CONFIG_OPTION_SUBPIXEL_FILE}")
    file(READ "${FT_CONFIG_OPTION_SUBPIXEL_FILE}" FT_OPTION_CONTENT)
    string(REPLACE
        "/* #define FT_CONFIG_OPTION_SUBPIXEL_RENDERING */"
        "#define FT_CONFIG_OPTION_SUBPIXEL_RENDERING"
        FT_OPTION_CONTENT "${FT_OPTION_CONTENT}")
    file(WRITE "${FT_CONFIG_OPTION_SUBPIXEL_FILE}" "${FT_OPTION_CONTENT}")
    unset(FT_OPTION_CONTENT)
endif()
unset(FT_CONFIG_OPTION_SUBPIXEL_FILE)

# Freetype targets must be static and must not pull system codec discovery
# (zlib/harfbuzz/brotli/png) — the GUI rasterizer only needs TrueType+LCD.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "lsxhome: freetype static" FORCE)
set(FT_DISABLE_ZLIB ON CACHE BOOL "lsxhome: no zlib in freetype" FORCE)
set(FT_DISABLE_BZIP2 ON CACHE BOOL "lsxhome: no bzip2 in freetype" FORCE)
set(FT_DISABLE_BROTLI ON CACHE BOOL "lsxhome: no brotli in freetype" FORCE)
set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "lsxhome: no harfbuzz in freetype" FORCE)
set(FT_DISABLE_PNG ON CACHE BOOL "lsxhome: no libpng in freetype" FORCE)
add_subdirectory(${freetype_SOURCE_DIR} ${freetype_BINARY_DIR} EXCLUDE_FROM_ALL)

# ── Dear ImGui static archive (docking headers + D3D12 + Win32) ─────────────
add_library(lsxhome_imgui STATIC
    ${dear-imgui_SOURCE_DIR}/imgui.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_draw.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_tables.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${dear-imgui_SOURCE_DIR}/imgui_demo.cpp
    ${dear-imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
    ${dear-imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    ${dear-imgui_SOURCE_DIR}/backends/imgui_impl_dx12.cpp
)
target_include_directories(lsxhome_imgui PUBLIC
    $<BUILD_INTERFACE:${dear-imgui_SOURCE_DIR}>
    $<BUILD_INTERFACE:${dear-imgui_SOURCE_DIR}/backends>
    $<BUILD_INTERFACE:${dear-imgui_SOURCE_DIR}/misc/freetype>
)
# imgui_freetype must see the compiled-in FreeType subpixel rasterizer, and
# imgui.cpp must default to the FreeType loader (imconfig-level IMGUI_ENABLE_FREETYPE).
target_compile_definitions(lsxhome_imgui PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    IMGUI_ENABLE_FREETYPE
    IMGUI_DISABLE_OBSOLETE_FUNCTIONS
)
# Execution-subsystem layers forced natively: D3D12 + Win32 graphics stack.
# /MACHINE:X64 everywhere: PE32+ is a hard requirement of the Blackwell toolchain.
target_link_libraries(lsxhome_imgui PUBLIC freetype)
target_compile_options(lsxhome_imgui PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/MACHINE:X64>)
if(MSVC)
    target_link_options(lsxhome_imgui PUBLIC /MACHINE:X64)
endif()

# ── ImPlot static (real-time charting channels) ─────────────────────────────
add_library(lsxhome_implot STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(lsxhome_implot PUBLIC
    $<BUILD_INTERFACE:${implot_SOURCE_DIR}>
)
target_link_libraries(lsxhome_implot PUBLIC lsxhome_imgui)
if(MSVC)
    target_link_options(lsxhome_implot PUBLIC /MACHINE:X64)
endif()

# ── ImNodes static (graph-layout node editor) ───────────────────────────────
add_library(lsxhome_imnodes STATIC
    ${imnodes_SOURCE_DIR}/imnodes.cpp
)
target_include_directories(lsxhome_imnodes PUBLIC
    $<BUILD_INTERFACE:${imnodes_SOURCE_DIR}>
)
target_link_libraries(lsxhome_imnodes PUBLIC lsxhome_imgui)
if(MSVC)
    target_link_options(lsxhome_imnodes PUBLIC /MACHINE:X64)
endif()

# ── TrueType font binary payload → static execution memory ──────────────────
# The resource loader ingests the TTF byte vector straight into the pre-allocated
# static memory space via a CMake embed pass (byte array in .rdata). Default font:
# Inter is freely redistributable; drop a TTF into lsxhome/assets/ and reconfigure.
set(LSXHOME_FONT_TTF "${CMAKE_CURRENT_SOURCE_DIR}/assets/Inter-Regular.ttf" CACHE FILEPATH
    "TrueType font vector embedded into the lsxhome static execution memory")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake)
lsx_embed_binary("${LSXHOME_FONT_TTF}"
    lsxhome_font_payload
    "${CMAKE_CURRENT_BINARY_DIR}/lsxhome_font_payload.cpp")
add_library(lsxhome_font STATIC "${CMAKE_CURRENT_BINARY_DIR}/lsxhome_font_payload.cpp")
target_include_directories(lsxhome_font PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
if(MSVC)
    target_link_options(lsxhome_font PUBLIC /MACHINE:X64)
endif()

# ── JetBrains Mono (monospace telemetry font) → static execution memory ──────
# Introduced under TDD (tests/test_lsxhome_font.cpp): the RED build failed with
# C1083 (missing lsxhome_mono_payload.h); this target is the minimal GREEN that
# bakes the JetBrains Mono TrueType vector into its own static archive.
set(LSXHOME_MONO_TTF "${CMAKE_CURRENT_SOURCE_DIR}/assets/JetBrainsMono-Regular.ttf" CACHE FILEPATH
    "JetBrains Mono TrueType vector embedded into the lsxhome static execution memory")
lsx_embed_binary("${LSXHOME_MONO_TTF}"
    lsxhome_mono_payload
    "${CMAKE_CURRENT_BINARY_DIR}/lsxhome_mono_payload.cpp")
add_library(lsxhome_mono STATIC "${CMAKE_CURRENT_BINARY_DIR}/lsxhome_mono_payload.cpp")
target_include_directories(lsxhome_mono PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
if(MSVC)
    target_link_options(lsxhome_mono PUBLIC /MACHINE:X64)
endif()

# ── lsxhome executable (WIN32 subsystem, no console) ────────────────────────
add_executable(lsxhome WIN32
    src/main_win32.cpp
    src/d3d12_renderer.cpp
    src/font_loader.cpp
)
target_include_directories(lsxhome PRIVATE include)
target_compile_definitions(lsxhome PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(lsxhome PRIVATE
    lsxhome_imgui
    lsxhome_implot
    lsxhome_imnodes
    lsxhome_font
    lsxhome_mono
    lsxcommon
    arrow_shared
    arrow_cuda_shared
    d3d12
    dxgi
    d3dcompiler
    dwmapi
    user32
    gdi32
    shell32
)
# Full x64 machine optimization block: /MACHINE:X64 emits the PE32+ image,
# /OPT:REF folds dead code, /OPT:ICF merges identical COMDATs (cross-TU,
# characteristically at high value for the IMGUI static stack).
if(MSVC)
    target_link_options(lsxhome PRIVATE /MACHINE:X64 /OPT:REF /OPT:ICF=5)
endif()

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