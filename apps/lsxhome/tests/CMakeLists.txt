add_executable(test_lsxhome_gui test_lsxhome_gui.cpp)
target_include_directories(test_lsxhome_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_lsxhome_gui PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lsxhome_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lsxhome_gui>"
    )
    add_custom_command(TARGET test_lsxhome_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lsxhome_gui>"
    )
endif()

add_test(NAME test_lsxhome_gui COMMAND test_lsxhome_gui)