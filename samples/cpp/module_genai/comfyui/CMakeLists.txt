cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "md_comfyui")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)
find_package(yaml-cpp REQUIRED)

# FFmpeg: optional, needed because vision_utils.cpp may use it
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG IMPORTED_TARGET QUIET
        libavformat libavcodec libswresample libavutil)
    if(FFMPEG_FOUND)
        add_compile_definitions(HAVE_FFMPEG)
        pkg_check_modules(LIBCURL IMPORTED_TARGET QUIET libcurl)
        if(NOT LIBCURL_FOUND)
            find_package(CURL QUIET)
        endif()
    endif()
endif()

add_executable(${TARGET_NAME}
  main.cpp
  ../utils/vision_utils.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(${TARGET_NAME} PRIVATE
  openvino::genai
  ${OpenCV_LIBS}
)
if(FFMPEG_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::FFMPEG)
    if(LIBCURL_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::LIBCURL)
    elseif(CURL_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE CURL::libcurl)
    endif()
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin/
        COMPONENT tools)