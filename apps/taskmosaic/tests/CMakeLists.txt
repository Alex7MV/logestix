add_executable(test_taskmosaic test_taskmosaic.cpp)
target_link_libraries(test_taskmosaic PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic>"
    )
    add_custom_command(TARGET test_taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic>"
    )
endif()

add_test(NAME test_taskmosaic COMMAND test_taskmosaic)
set_tests_properties(test_taskmosaic PROPERTIES LABELS fast)

add_executable(test_taskmosaic_cli test_taskmosaic_cli.cpp ../src/cli.cpp)
target_include_directories(test_taskmosaic_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_cli>"
    )
    add_custom_command(TARGET test_taskmosaic_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_cli>"
    )
endif()

add_test(NAME test_taskmosaic_cli COMMAND test_taskmosaic_cli)
set_tests_properties(test_taskmosaic_cli PROPERTIES LABELS fast)

add_executable(test_taskmosaic_config test_taskmosaic_config.cpp ../src/config.cpp)
target_include_directories(test_taskmosaic_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_config PRIVATE lsxcommon simdjson::simdjson Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_config POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_config>"
    )
    add_custom_command(TARGET test_taskmosaic_config POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_config>"
    )
endif()

add_test(NAME test_taskmosaic_config COMMAND test_taskmosaic_config)
set_tests_properties(test_taskmosaic_config PROPERTIES LABELS fast)

add_executable(test_taskmosaic_backend test_taskmosaic_backend.cpp ../src/backend.cpp ../src/config.cpp)
target_include_directories(test_taskmosaic_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_backend PRIVATE lsxcommon lsxnetwork simdjson::simdjson Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_backend>"
    )
    add_custom_command(TARGET test_taskmosaic_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_backend>"
    )
endif()

add_test(NAME test_taskmosaic_backend COMMAND test_taskmosaic_backend)
set_tests_properties(test_taskmosaic_backend PROPERTIES LABELS fast)

add_executable(test_taskmosaic_bridge
    test_taskmosaic_bridge.cpp
    ../src/config.cpp ../src/backend.cpp ../src/chat_bridge.cpp)
target_include_directories(test_taskmosaic_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_taskmosaic_bridge PRIVATE lsxcommon lsxnetwork simdjson::simdjson httplib::httplib Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_taskmosaic_bridge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_bridge>"
    )
    add_custom_command(TARGET test_taskmosaic_bridge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_taskmosaic_bridge>"
    )
endif()

add_test(NAME test_taskmosaic_bridge COMMAND test_taskmosaic_bridge)
set_tests_properties(test_taskmosaic_bridge PROPERTIES LABELS fast)
set_tests_properties(test_taskmosaic_bridge PROPERTIES TIMEOUT 120)
