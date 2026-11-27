# libs/lsxstorage — the storage layer (namespace lsx::storage).
#
# Sits BELOW libs/lsxcommon on purpose: the load gate lives in
# ModelBase::BeginLoad (lsxcommon) and must be able to read artifact.fb, so the
# artifact reader cannot depend on lsxcommon or the dependency would cycle.
# links: FlatBuffers (submodule) + spdlog only.

# ── artifact.fb schema → generated header ───────────────────────────────────
# Generated at build time, never committed: a schema edit must be impossible to
# forget, so `artifact_generated.h` cannot drift from `artifact.fbs`.
set(_lsx_artifact_fbs "${CMAKE_CURRENT_SOURCE_DIR}/schema/artifact.fbs")
set(_lsx_artifact_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_lsx_artifact_gen_hdr "${_lsx_artifact_gen_dir}/artifact_generated.h")

add_custom_command(
    OUTPUT "${_lsx_artifact_gen_hdr}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_lsx_artifact_gen_dir}"
    COMMAND flatc --cpp --scoped-enums --gen-object-api --filename-suffix _generated
            -o "${_lsx_artifact_gen_dir}" "${_lsx_artifact_fbs}"
    DEPENDS "${_lsx_artifact_fbs}" flatc
    COMMENT "flatc: generating artifact_generated.h from artifact.fbs"
    VERBATIM)

add_library(lsxstorage STATIC
    src/roots.cpp
    src/artifact.cpp
    src/artifact_writer.cpp
    src/sha256.cpp
    src/verify.cpp
    "${_lsx_artifact_gen_hdr}"
)

# The generated header is PRIVATE: artifact.h exposes plain structs so that a
# consumer (lsxcommon, or any .cu TU that reaches it) never pulls FlatBuffers in.
target_include_directories(lsxstorage
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE
        "${_lsx_artifact_gen_dir}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"   # sha256.h, private to this target
)

target_link_libraries(lsxstorage PUBLIC
    flatbuffers
    spdlog::spdlog
)

# Arrow is needed by src/verify.cpp only, but the link must be PUBLIC: lsxstorage
# is a STATIC library, so a PRIVATE dependency does not reach the consumers that
# actually resolve the symbols (they failed to link with `undefined reference to
# arrow::...` from libarrow_cuda.so until this was public).
#
# The headers stay Arrow-free — artifact.h/artifact_writer.h expose plain
# structs — so a device TU that reaches lsxstorage gets an include dir it does
# not use, not a compile-time dependency on Arrow.
target_link_libraries(lsxstorage PUBLIC
    lsx_dep_includes
    arrow_shared
)

set_target_properties(lsxstorage PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
