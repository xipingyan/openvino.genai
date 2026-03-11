// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"
#include <optional>
#include <vector>
#include <string>

namespace ov::genai::module {

int64_t count_visual_tokens(const ov::Tensor& grid_thw, int32_t spatial_merge_size);

std::vector<std::string> build_prompts(const std::vector<std::string> &prompts,
                                       std::optional<std::vector<ov::Tensor>>& image_grid_thw, 
                                       std::optional<std::vector<ov::Tensor>>& audio_features,
                                       std::optional<std::vector<ov::Tensor>>& video_grid_thw,
                                       std::optional<std::vector<int>>& use_audio_in_video,
                                       std::optional<std::vector<int>>& video_second_per_grid,
                                       std::optional<int64_t> spatial_merge_size,
                                       std::optional<int64_t> position_id_per_seconds);

}