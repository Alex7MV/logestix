add_executable(test_lsxhome_gui test_lsxhome_gui.cpp)
target_include_directories(test_lsxhome_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_lsxhome_gui PRIVATE lsxcommon Catch2::Catch2WithMain)

find_package(Catch2 QUIET)
add_executable(test_lsxhome_font test_lsxhome_font.cpp)
target_include_directories(test_lsxhome_font PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/..
)
target_link_libraries(test_lsxhome_font PRIVATE lsxhome_mono Catch2::Catch2WithMain)

set(LSXHOME_TEST_TARGETS test_lsxhome_gui test_lsxhome_font)

foreach(test_target IN LISTS LSXHOME_TEST_TARGETS)
    if(WIN32)
        add_custom_command(TARGET ${test_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${test_target}>"
        )
        add_custom_command(TARGET ${test_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${test_target}>"
        )
    endif()
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()