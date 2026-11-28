# Model manifest wiring.
#
# `models/<org>/<model>.json` is the single source of truth for one ingested
# model. This module runs tools/gen_model_manifest.py at configure time and
# publishes three generated artifacts under ${CMAKE_BINARY_DIR}/generated/:
#
#   model_manifest.h      C++ table (guards, gate drivers, sync test)
#   model_targets.cmake   ctest registration for every model target
#   model_registry.inc    LsxModelFactory registration list
#
# Nothing here is committed: the build tree owns the output, so it cannot be
# hand-edited and cannot drift. The one committed artifact (docs/MODELS.md) is
# enforced by the generator's `--check` mode through test_manifest_sync.
#
# Failure policy: a manifest error fails CONFIGURE with the generator's own
# message. A model that cannot be described is not silently absent.
include_guard(GLOBAL)

find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)
set(LSX_PYTHON3 "${Python3_EXECUTABLE}")

set(LSX_MANIFEST_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")

# lsx_configure_model_manifest()
#
# Generates the artifacts and sets, in the CALLER's scope:
#   LSX_MANIFEST_MODELS        - list of model_type values (sorted)
#   LSX_MANIFEST_GENERATED_DIR - the directory holding the artifacts
#   LSX_MANIFEST_HEADER        - path to model_manifest.h
#   LSX_MANIFEST_TARGETS       - path to model_targets.cmake
#   LSX_MANIFEST_REGISTRY      - path to model_registry.inc
function(lsx_configure_model_manifest)
    set(_gen "${LSX_MANIFEST_GENERATED_DIR}")
    file(MAKE_DIRECTORY "${_gen}")

    execute_process(
        COMMAND "${LSX_PYTHON3}"
                "${PROJECT_SOURCE_DIR}/tools/gen_model_manifest.py"
                --root "${PROJECT_SOURCE_DIR}"
                --header "${_gen}/model_manifest.h"
                --cmake "${_gen}/model_targets.cmake"
                --registry "${_gen}/model_registry.inc"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
            "model manifest generation failed (exit ${_rc}):\n${_err}${_out}")
    endif()
    if(_out)
        message(STATUS "${_out}")
    endif()
    # Warnings (unclaimed test files while manifests are being authored) are
    # informational here; `--check` in test_manifest_sync makes them fatal.
    if(_err)
        message(STATUS "model manifest: ${_err}")
    endif()

    set(_models "")
    # models/<org>/<model>.json — one file per model, nested by the store's own
    # org segment. The stem is no longer the model_type (the file is named after
    # the store directory, which is a download spelling), so the label list is
    # used for messages only; identity comes from the manifest's own field.
    file(GLOB _manifest_files CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/models/*/*.json")
    foreach(_f IN LISTS _manifest_files)
        get_filename_component(_stem "${_f}" NAME_WE)
        get_filename_component(_org "${_f}" DIRECTORY)
        get_filename_component(_org "${_org}" NAME)
        list(APPEND _models "${_org}/${_stem}")
    endforeach()
    list(SORT _models)

    set(LSX_MANIFEST_MODELS "${_models}" PARENT_SCOPE)
    set(LSX_MANIFEST_GENERATED_DIR "${_gen}" PARENT_SCOPE)
    set(LSX_MANIFEST_HEADER "${_gen}/model_manifest.h" PARENT_SCOPE)
    set(LSX_MANIFEST_TARGETS "${_gen}/model_targets.cmake" PARENT_SCOPE)
    set(LSX_MANIFEST_REGISTRY "${_gen}/model_registry.inc" PARENT_SCOPE)
endfunction()
