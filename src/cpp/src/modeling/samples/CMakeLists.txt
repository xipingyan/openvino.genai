# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

add_executable(modeling_qwen3_vl
    modeling_qwen3_vl.cpp
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat/load_image.cpp"
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_qwen3_vl PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_qwen3_vl PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>
    ${YAML_CPP_TARGET})

target_compile_definitions(modeling_qwen3_vl PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_qwen3_vl openvino_genai)

# Standalone video frame extraction tool (replaces Python extract_video_frames.py)
find_package(OpenCV QUIET COMPONENTS videoio imgcodecs imgproc core)
if(OpenCV_FOUND)
    add_executable(extract_video_frames extract_video_frames.cpp)
    target_include_directories(extract_video_frames PRIVATE
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(extract_video_frames PRIVATE
        ${OpenCV_LIBS})
    set_target_properties(extract_video_frames PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)
    if(WIN32)
        # Copy OpenCV DLLs next to the executable
        foreach(_opencv_target opencv_videoio opencv_imgcodecs opencv_imgproc opencv_core)
            if(TARGET ${_opencv_target})
                add_custom_command(TARGET extract_video_frames POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "$<TARGET_FILE:${_opencv_target}>"
                            "$<TARGET_FILE_DIR:extract_video_frames>"
                    VERBATIM)
            endif()
        endforeach()
    endif()
else()
    message(STATUS "OpenCV not found -- extract_video_frames tool will not be built")
endif()