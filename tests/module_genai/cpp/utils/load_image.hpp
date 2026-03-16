// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <openvino/runtime/tensor.hpp>

namespace ov::genai::module::utils {
ov::Tensor load_image(const std::filesystem::path& image_path);
std::vector<ov::Tensor> load_images(const std::filesystem::path& image_path);

// input_path: a directory containing video frames as images, or a single video file (not supported yet, need ffmpeg to
// decode video into frames)
ov::Tensor load_video(const std::filesystem::path& input_path);

ov::Tensor create_countdown_frames();

namespace TEST_DATA {
std::string img_cat_120_100();
std::string img_dog_120_120();
ov::Tensor audio_dummy_data(float duration, int sample_rate = 16000);
};  // namespace TEST_DATA
}  // namespace ov::genai::module::utils