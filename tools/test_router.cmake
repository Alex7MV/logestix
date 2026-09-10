# git-diff-driven ctest category router (script mode, no build system).
# Labels: fast (always) + inference (when the inference contour changed).
# perf is NEVER auto-run: `ctest --preset <p> -L perf` is a manual command.
#
# Usage (from the repo root):
#   cmake -DLSX_PRESET=<ctest-preset> [-DLSX_BASE=<ref>] [-DLSX_ALL=ON] \
#       [-DLSX_LIST=ON] -P tools/test_router.cmake
#   LSX_PRESET  required ctest preset name (e.g. linux-debug)
#   LSX_BASE    diff base ref, default origin/main (merge-base diff)
#   LSX_ALL=ON  force fast+inference regardless of the diff
#   LSX_LIST=ON print the selected labels and exit (no ctest run)
#
# Windows note: the inference sweep additionally applies
# -R "^(test_qwen|test_gemma)" per the AGENTS.md OS restriction.
cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED LSX_PRESET)
    message(FATAL_ERROR
        "usage: cmake -DLSX_PRESET=<ctest-preset> [-DLSX_BASE=<ref>] "
        "[-DLSX_ALL=ON] [-DLSX_LIST=ON] -P tools/test_router.cmake")
endif()
if(NOT DEFINED LSX_BASE)
    set(LSX_BASE "origin/main")
endif()
if(NOT DEFINED LSX_ALL)
    set(LSX_ALL OFF)
endif()
if(NOT DEFINED LSX_LIST)
    set(LSX_LIST OFF)
endif()

get_filename_component(LSX_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# ── Rule table (guard-checked by tests/test_ctest_labels_guard.cpp) ───────
# lsx_router_rule(<id> <labels> <regex>) — FIRST match wins; order matters.
# <labels> is space-separated fast|convert|inference.
macro(lsx_router_rule id labels regex)
    list(APPEND LSX_RULE_IDS "${id}")
    list(APPEND LSX_RULE_LABELS "${labels}")
    list(APPEND LSX_RULE_REGEXES "${regex}")
endmacro()

set(LSX_RULE_IDS "")
set(LSX_RULE_LABELS "")
set(LSX_RULE_REGEXES "")

# 1. Converter domain first (it also ships .cu quantization engines):
#    *_convert / *converter* / tokenizer_convert sources → convert.
lsx_router_rule(converters "convert" "libs/lsxcommon/.*(converter|_convert|tokenizer_convert)")
#    GPU quantize engine of the converters → convert + inference.
lsx_router_rule(quant_engine "convert inference" "libs/lsxcommon/.*safetensors_gpu")
# 2. Any other CUDA source anywhere → inference.
lsx_router_rule(cuda "inference" "\\.cuh?$")
# 3. lsxcommon: host-only fast-lexicon first, everything else → inference
#    (conservative default: a new unknown lsxcommon file is inference).
lsx_router_rule(lsxcommon_host "fast" "libs/lsxcommon/.*(tokenizer|unicode|pretokenizer|path_registry|schema|orchestrator)")
lsx_router_rule(lsxcommon "inference" "libs/lsxcommon/")
# 4. Network layer, apps → fast only.
lsx_router_rule(network "fast" "libs/lsxnetwork/")
lsx_router_rule(apps "fast" "apps/")
# 5. Test infra changed → every non-perf sweep (before the tooling rule,
#    else tests/CMakeLists.txt would be swallowed by CMakeLists\.txt$).
lsx_router_rule(tests "convert inference" "^tests/")
# 6. Repo-root build files, docs, tools, top-level *.md → fast only.
lsx_router_rule(tooling "fast" "^(CMakeLists\\.txt$|CMakePresets\\.json$|cmake/|docs/|tools/|[^/]+\\.md$)")
# 7. Everything else → fast.
lsx_router_rule(default "fast" ".")

set(LSX_LABELS "fast")
if(NOT LSX_ALL)
    execute_process(
        COMMAND git diff --name-only ${LSX_BASE}...HEAD
        WORKING_DIRECTORY "${LSX_REPO_ROOT}"
        OUTPUT_VARIABLE LSX_DIFF
        RESULT_VARIABLE LSX_DIFF_RC
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT LSX_DIFF_RC EQUAL 0 OR LSX_DIFF STREQUAL "")
        message(STATUS
            "test_router: no diff against ${LSX_BASE} — running fast only")
        set(LSX_DIFF "")
    endif()
    string(REPLACE "\n" ";" LSX_FILES "${LSX_DIFF}")
    foreach(f IN LISTS LSX_FILES)
        if(f STREQUAL "")
            continue()
        endif()
        list(LENGTH LSX_RULE_IDS LSX_N)
        set(i 0)
        while(i LESS LSX_N)
            list(GET LSX_RULE_LABELS ${i} lbl)
            list(GET LSX_RULE_REGEXES ${i} re)
            if(f MATCHES "${re}")
                string(REPLACE " " ";" lbls "${lbl}")
                foreach(lb IN LISTS lbls)
                    if(lb STREQUAL "fast")
                        continue()
                    endif()
                    list(FIND LSX_LABELS "${lb}" idx)
                    if(idx EQUAL -1)
                        list(APPEND LSX_LABELS "${lb}")
                    endif()
                endforeach()
                break()
            endif()
            math(EXPR i "${i} + 1")
        endwhile()
    endforeach()
else()
    list(APPEND LSX_LABELS "convert")
    list(APPEND LSX_LABELS "inference")
endif()

list(JOIN LSX_LABELS "|" LSX_LABEL_RE)
message(STATUS "test_router: labels = ${LSX_LABELS}")
if(LSX_LIST)
    return()
endif()

find_program(LSX_CTEST_EXE ctest REQUIRED)
set(LSX_FAILED OFF)
if(WIN32)
    # Per-label sweeps: the qwen|gemma name filter (AGENTS.md OS restriction)
    # must not shrink the fast run.
    foreach(sweep IN ITEMS fast convert inference)
        list(FIND LSX_LABELS "${sweep}" idx)
        if(idx EQUAL -1)
            continue()
        endif()
        if(sweep STREQUAL "fast")
            set(name_filter "")
        else()
            set(name_filter -R "^(test_qwen|test_gemma)")
        endif()
        execute_process(COMMAND ${LSX_CTEST_EXE} --preset ${LSX_PRESET}
            -L "^${sweep}$" ${name_filter} RESULT_VARIABLE rc)
        if(NOT rc EQUAL 0)
            set(LSX_FAILED ON)
        endif()
    endforeach()
else()
    execute_process(COMMAND ${LSX_CTEST_EXE} --preset ${LSX_PRESET}
        -L "^(${LSX_LABEL_RE})$" RESULT_VARIABLE LSX_RC)
    if(NOT LSX_RC EQUAL 0)
        set(LSX_FAILED ON)
    endif()
endif()
if(LSX_FAILED)
    message(FATAL_ERROR "test_router: ctest sweep failed")
endif()
return()
