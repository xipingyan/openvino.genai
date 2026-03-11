// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file utils_image.hpp
 * @brief Image utility functions for loading and saving ov::Tensor images
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <filesystem>

#include <openvino/runtime/tensor.hpp>

namespace image_utils {

// ============================================================================
// Video Loading Structures
// ============================================================================

/**
 * @brief Options for video frame sampling (mirrors Python smart_nframes logic)
 */
struct VideoLoadOptions {
    float fps        = 2.0f;   ///< Target frames per second to sample
    int   min_frames = 4;      ///< Minimum number of frames to extract
    int   max_frames = 768;    ///< Maximum number of frames to extract
};

/**
 * @brief Result from load_video_with_audio
 *   - frames: [N, H, W, 3] u8, RGB, frames sampled at target fps
 *   - audio:  [1, num_samples] f32, 16 kHz mono (empty Tensor if use_audio_in_video=false)
 *   - sample_fps: actual sampling fps used
 */
struct VideoLoadResult {
    ov::Tensor frames;
    ov::Tensor audio;
    float      sample_fps = 0.0f;
};

// ============================================================================
// Image Loading Functions
// ============================================================================

/**
 * @brief Load a single image from file
 * @param image_path Path to the image file
 * @return ov::Tensor with shape [1, H, W, 3] in RGB format
 */
ov::Tensor load_image(const std::filesystem::path& image_path);

/**
 * @brief Load multiple images from a directory or a single image file
 * @param input_path Path to a directory or single image file
 * @return Vector of ov::Tensor images
 */
std::vector<ov::Tensor> load_images(const std::filesystem::path& input_path);

/**
 * @brief Load images from a directory as a video tensor (legacy)
 * @param input_path Path to directory containing image frames
 * @return ov::Tensor with shape [num_frames, H, W, 3]
 */
ov::Tensor load_video(const std::filesystem::path& input_path);

/**
 * @brief Load a video file: sample frames and optionally extract audio
 *
 * Frames are uniformly sampled at `opts.fps` (mirrors Python smart_nframes).
 * Each frame is resized via smart_resize (factor=28, pixel budget aware).
 * When use_audio_in_video=true the audio stream is decoded, resampled to
 * 16 kHz mono float32, and returned as ov::Tensor [1, num_samples].
 *
 * @param video_path         Path to the video file (mp4, avi, mkv, …)
 * @param use_audio_in_video Extract audio in addition to frames
 * @param opts               Frame sampling options
 * @return VideoLoadResult with frames, audio (may be empty), sample_fps
 */
VideoLoadResult load_video_with_audio(const std::filesystem::path& video_path,
                                      bool                         use_audio_in_video = false,
                                      const VideoLoadOptions&      opts               = {});

/**
 * @brief Create countdown frames for testing
 * @return ov::Tensor with shape [5, 240, 360, 3] containing countdown frames
 */
ov::Tensor create_countdown_frames();

// ============================================================================
// Image Saving Functions
// ============================================================================

/**
 * @brief Save ov::Tensor image to BMP file
 * @param filename Output file path
 * @param image Input tensor (HWC or NHWC format with 3 channels)
 * @param convert_rgb2bgr Whether to convert RGB to BGR (default: true)
 * @return true if successful, false otherwise
 */
bool save_image_bmp(const std::string& filename, const ov::Tensor& image, bool convert_rgb2bgr = true);

/**
 * @brief Generate unique filename with timestamp
 * @param prefix Filename prefix
 * @param suffix Filename suffix (default: ".bmp")
 * @return Generated filename with timestamp
 */
std::string generate_output_filename(const std::string& prefix, const std::string& suffix = ".bmp");

} // namespace image_utils
