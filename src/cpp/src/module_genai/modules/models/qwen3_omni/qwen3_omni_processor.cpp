// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni_processor.hpp"

namespace ov::genai::module {

int64_t count_visual_tokens(const ov::Tensor& grid_thw, int32_t spatial_merge_size) {
    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto shape = grid_thw.get_shape();
    if (shape.size() != 2 || shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }
    const int64_t* grid = grid_thw.data<const int64_t>();
    int64_t total = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw values");
        }
        if (h % spatial_merge_size != 0 || w % spatial_merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by spatial_merge_size");
        }
        total += t * (h / spatial_merge_size) * (w / spatial_merge_size);
    }
    return total;
}

std::vector<std::string> build_prompts(const std::vector<std::string> &prompts,
                                       const std::optional<ov::Tensor> &grid_thw,
                                       const std::optional<ov::Tensor> &audio_features,
                                       int64_t spatial_merge_size) {
    std::vector<std::string> result;
    for (const auto& prompt : prompts) {
        std::stringstream ss;
        ss << "<|im_start|>user\n";
        int64_t image_token_num = 0;
        if (grid_thw.has_value()) {
            image_token_num = count_visual_tokens(grid_thw.value(), static_cast<int32_t>(spatial_merge_size));
        }
        int64_t audio_token_num = 0;
        if (audio_features.has_value()) {
            audio_token_num = static_cast<int64_t>(audio_features.value().get_shape()[0]);
        }

        if (image_token_num > 0) {
            ss << "<|vision_start|>";
            for (int64_t i = 0; i < image_token_num; ++i) ss << "<|image_pad|>";
            ss << "<|vision_end|>";
        }
        if (audio_token_num > 0) {
            ss << "<|audio_start|>";
            for (int64_t i = 0; i < audio_token_num; ++i) ss << "<|audio_pad|>";
            ss << "<|audio_end|>";
        }
        ss << prompt;
        ss << "<|im_end|>\n<|im_start|>assistant\n";
        result.push_back(ss.str());
    }
    return result;
}

}