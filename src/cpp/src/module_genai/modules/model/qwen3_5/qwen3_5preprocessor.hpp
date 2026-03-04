// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "openvino/runtime/tensor.hpp"
#include "qwen3_5config.hpp"
#include "module_genai/utils/vision_preprocess.hpp"

namespace ov::genai::module {

struct Qwen3_5PreprocessorOutput {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;

    // Video-specific outputs:
    ov::Tensor pixel_values_videos;
    ov::Tensor video_grid_thw;
};

class Qwen3_5Preprocessor {
public:
    explicit Qwen3_5Preprocessor(const std::filesystem::path& model_path);

    Qwen3_5PreprocessorOutput preprocess(const ov::Tensor &images);

    // @Param: video: shape [N, T, H, W, C] or [T, H, W, C], uint8, where N is batch size, T is temporal dimension
    // (number of frames), H and W are height and width of each frame, C is number of channels (3 for RGB). Output
    // grid_thw will be [N, 3] representing the grid dimensions for each video in the batch. Currently we only support
    // batch size of 1 for video input due to the complexity of handling variable-length videos and batching them
    // together. Future work may include support for batching multiple videos together.
    Qwen3_5PreprocessorOutput preprocess_video(const ov::Tensor &video);
private:
    Qwen3_5VisionPreprocessConfig m_preprocess_config;
    Qwen3_5VisionConfig m_vision_config;
    ov::Tensor m_pos_embed_weight;

    void load_pos_embed_weight(const std::filesystem::path& model_path);

    static ov::element::Type parse_ov_dtype(const std::string& s);

    std::pair<size_t, size_t> smart_resize(size_t height, size_t width, size_t factor);

    void resize_bilinear_to_chw(const uint8_t* src,
                                size_t src_h,
                                size_t src_w,
                                size_t channels,
                                bool nchw,
                                size_t dst_h,
                                size_t dst_w,
                                std::vector<float>& dst_chw);

    ov::Tensor build_pos_embeds(const ov::Tensor &grid_thw);

    static ov::Tensor to_f32(const ov::Tensor& src);

    std::pair<ov::Tensor, ov::Tensor> build_rotary_cos_sin(const ov::Tensor &grid_thw);
};

}
