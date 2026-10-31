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