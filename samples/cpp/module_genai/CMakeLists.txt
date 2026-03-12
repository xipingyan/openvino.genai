cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "module_genai_ut_app")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)
find_package(OpenCV REQUIRED)

# yaml-cpp dependency (use centralized cmake module or fetch if standalone)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/yaml-cpp.cmake)

# FFmpeg: optional, needed for audio extraction from video files
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG IMPORTED_TARGET QUIET
        libavformat libavcodec libswresample libavutil)
    if(FFMPEG_FOUND)
        message(STATUS "FFmpeg found — audio-in-video support enabled")
        add_compile_definitions(HAVE_FFMPEG)
        # libavformat pulls in libgdal/libhdf5 which both depend on libcurl;
        # link it explicitly to satisfy the @CURL_OPENSSL_4 versioned symbols.
        pkg_check_modules(LIBCURL IMPORTED_TARGET QUIET libcurl)
        if(NOT LIBCURL_FOUND)
            find_package(CURL QUIET)
        endif()
    else()
        message(STATUS "FFmpeg not found — audio-in-video will throw at runtime")
    endif()
endif()

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found! Check your installation and environment.")
endif()

function(add_sample_executable target_name)
    add_executable(${target_name}
        ${target_name}.cpp
        utils/utils.cpp
        utils/vision_utils.cpp
        utils/audio_utils.cpp
    )
    target_link_libraries(${target_name} PRIVATE openvino::genai ${OpenCV_LIBS} ${YAML_CPP_TARGET})
    if(FFMPEG_FOUND)
        target_link_libraries(${target_name} PRIVATE PkgConfig::FFMPEG)
        if(LIBCURL_FOUND)
            target_link_libraries(${target_name} PRIVATE PkgConfig::LIBCURL)
        elseif(CURL_FOUND)
            target_link_libraries(${target_name} PRIVATE CURL::libcurl)
        endif()
    endif()
    set_target_properties(${target_name} PROPERTIES
        # Ensure out-of-box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)

    # Install this sample binary as part of the default install target.
    install(TARGETS ${target_name}
            RUNTIME DESTINATION samples/module_genai/cpp/
            COMPONENT samples)

endfunction()

set (SAMPLE_LIST
    md_image_generation
    md_video_generation
    md_visual_language_chat
    md_omni)

foreach(sample IN LISTS SAMPLE_LIST)
    add_sample_executable(${sample})
endforeach()

# Install runtime configuration files for module_genai samples
install(DIRECTORY config_yaml
        DESTINATION samples/module_genai/cpp
        COMPONENT samples
        USE_SOURCE_PERMISSIONS)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

add_subdirectory(comfyui)
