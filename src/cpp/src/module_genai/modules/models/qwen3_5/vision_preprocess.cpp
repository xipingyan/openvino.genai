// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/models/qwen3_5/vision_preprocess.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "module_genai/utils/tensor_utils.hpp"

namespace ov::genai::module {

Qwen3_5VisionPreprocess::Qwen3_5VisionPreprocess(const std::filesystem::path& model_path, const std::string& device, VLMModelType model_type)
    : VisionPreprocess(model_type, device) {
    //   m_video_processor(std::make_unique<Qwen3_5VLVideoProcessor>(model_path)) {}
    m_preprocessor = std::make_shared<Qwen3_5Preprocessor>(model_path, device);
}

PreprocessOutput Qwen3_5VisionPreprocess::preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) {
    OPENVINO_ASSERT(images.empty() || videos.empty(), "Qwen3_5VisionPreprocess: images and videos cannot both be non-empty");
    OPENVINO_ASSERT(videos.size() == 1u || videos.empty(), "Qwen3_5VisionPreprocess: only a single video input is supported due to the complexity of handling variable-length videos and batching them together");

    ov::Tensor stack_images;
    PreprocessOutput preprocess_output {};
    if (images.size() > 0) {
        for (const auto &image : images) {
            auto output = m_preprocessor->preprocess(image);
            preprocess_output.pixel_values.push_back(std::move(output.pixel_values));
            preprocess_output.grid_thw.push_back(std::move(output.grid_thw));
            preprocess_output.pos_embeds.push_back(std::move(output.pos_embeds));
            preprocess_output.rotary_cos.push_back(std::move(output.rotary_cos));
            preprocess_output.rotary_sin.push_back(std::move(output.rotary_sin));
        }
    }

    if (videos.size() > 0) {
        for (const auto &video : videos) {
            auto output = m_preprocessor->preprocess_video(video);
            preprocess_output.pixel_values_videos.push_back(std::move(output.pixel_values_videos));
            preprocess_output.video_grid_thw.push_back(std::move(output.video_grid_thw));
            preprocess_output.video_pos_embeds.push_back(std::move(output.pos_embeds));
            preprocess_output.video_rotary_cos.push_back(std::move(output.rotary_cos));
            preprocess_output.video_rotary_sin.push_back(std::move(output.rotary_sin));
            preprocess_output.video_second_per_grid.push_back(output.video_second_per_grid);
        }
    }

    return preprocess_output;
}

// void Qwen3_5VisionPreprocess::result_to_output(std::map<std::string, OutputModule>& output) const {
//     output["pixel_values"].data = m_output.pixel_values;
//     output["grid_thw"].data = m_output.grid_thw;
//     output["pos_embeds"].data = m_output.pos_embeds;
//     output["rotary_cos"].data = m_output.rotary_cos;
//     output["rotary_sin"].data = m_output.rotary_sin;
// }

}  // namespace ov::genai::module
