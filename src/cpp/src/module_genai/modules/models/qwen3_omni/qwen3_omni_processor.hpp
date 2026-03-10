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
                                       const std::optional<ov::Tensor> &grid_thw,
                                       const std::optional<ov::Tensor> &audio_features,
                                       int64_t spatial_merge_size);

}