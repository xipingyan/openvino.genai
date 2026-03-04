// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/model/qwen3_vl/vision_preprocess.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace ov::genai::module {

Qwen3VisionPreprocess::Qwen3VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type)
    : VisionPreprocess(model_type) {}

PreprocessOutput Qwen3VisionPreprocess::preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) {
    OPENVINO_ASSERT(images.empty() || videos.empty(), "Qwen3VisionPreprocess: images and videos cannot both be non-empty");
    OPENVINO_THROW("Qwen3VisionPreprocess::preprocess is not implemented yet");
    return {};
}

namespace qwen3vl_utils {
/**
 * @brief Smartly resizes dimensions based on pixel constraints and alignment factors.
 * * @param num_frames Number of frames in the video sequence.
 * @param height Original height.
 * @param width Original width.
 * @param temporal_factor Alignment factor for time/frames (default 2).
 * @param factor Spatial alignment factor for height/width (default 32).
 * @param min_pixels Minimum allowed total pixels (num_frames * h * w).
 * @param max_pixels Maximum allowed total pixels (num_frames * h * w).
 * @return ResizeResult containing the new h_bar and w_bar.
 */
ImageSize smart_resize(int num_frames,
                       int height,
                       int width,
                       int temporal_factor,
                       int factor,
                       size_t min_pixels,
                       size_t max_pixels) {
    // 1. Validation Checks
    if (height < factor || width < factor) {
        throw std::invalid_argument("Height or width must be larger than the alignment factor.");
    }

    double aspect_ratio = static_cast<double>(std::max(height, width)) / std::min(height, width);
    if (aspect_ratio > 200.0) {
        throw std::invalid_argument("Absolute aspect ratio must be smaller than 200.");
    }

    // 2. Initial alignment
    // h_bar and w_bar are rounded to the nearest multiple of 'factor'
    int h_bar = static_cast<int>(std::round(static_cast<double>(height) / factor)) * factor;
    int w_bar = static_cast<int>(std::round(static_cast<double>(width) / factor)) * factor;

    // t_bar is the temporal dimension aligned to 'temporal_factor'
    int t_bar = static_cast<int>(std::ceil(static_cast<double>(num_frames) / temporal_factor)) * temporal_factor;

    // Use size_t for volume calculation to prevent 32-bit integer overflow
    size_t current_pixels = static_cast<size_t>(t_bar) * h_bar * w_bar;

    // 3. Scale adjustment based on pixel budget
    if (current_pixels > max_pixels) {
        // Calculate downscaling ratio (beta)
        double beta = std::sqrt(static_cast<double>(num_frames) * height * width / max_pixels);
        // Floor to ensure we stay under the max_pixels limit after factor alignment
        h_bar = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (current_pixels < min_pixels) {
        // Calculate upscaling ratio (beta)
        double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(num_frames) * height * width));
        // Ceil to ensure we meet the min_pixels requirement
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }

    return {h_bar, w_bar};
}
}


}  // namespace ov::genai::module
