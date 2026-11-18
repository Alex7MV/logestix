# Encoding-audit probe: reports how a converted vocabulary stores and decodes
# bytes (space token, byte-fallback pieces, lone high bytes) and whether a
# decode -> encode round-trip survives. No model load, no CUDA. Diagnostic, not
# part of the test suite. See docs/repro/engine-encoding-audit in lsxhome.
add_executable(tokenizer_audit tokenizer_audit.cpp)
target_link_libraries(tokenizer_audit PRIVATE lsxcommon arrow_shared)

# Parity probe: encodes a JSONL case file through a converted tree's tokenizer
# arrows and diffs against reference ids captured from the checkpoint. Same
# diagnostic contract as tokenizer_audit (no model load, no CUDA, not a test).
add_executable(tokenizer_parity tokenizer_parity.cpp)
target_link_libraries(tokenizer_parity PRIVATE lsxcommon arrow_shared)

# Inkling base real-tree numeric probe: TraceForward dump for the independent
# float64 reference + framed-Paris top-5 + optional teacher-forced NLL.
# Diagnostic, not part of the test suite (loads the real vault).
add_executable(inkling_base_probe inkling_base_probe.cpp)
target_include_directories(inkling_base_probe PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(inkling_base_probe PRIVATE lsxcommon arrow_shared)

# Short-prompt NaN diagnostic (temporary): Infer at n=1..20 tokens, report NaN
# count + max logit. Loads the real vault; not part of the test suite.
add_executable(inkling_base_short_probe inkling_base_short_probe.cpp)
target_include_directories(inkling_base_short_probe PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(inkling_base_short_probe PRIVATE lsxcommon arrow_shared)

# CODE-03 token dump (temporary): inspect the generation markers to pick the
# answer_start_token_id for the inline-reasoning golden. Not a test.
add_executable(inkling_base_code03_tokens inkling_base_code03_tokens.cpp)
target_include_directories(inkling_base_code03_tokens PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(inkling_base_code03_tokens PRIVATE lsxcommon arrow_shared)

if(WIN32)
    add_custom_command(TARGET tokenizer_audit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>" "$<TARGET_FILE_DIR:tokenizer_audit>"
    )
    add_custom_command(TARGET tokenizer_audit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>" "$<TARGET_FILE_DIR:tokenizer_audit>"
    )
endif()