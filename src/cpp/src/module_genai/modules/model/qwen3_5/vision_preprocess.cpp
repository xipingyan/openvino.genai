// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/model/qwen3_5/vision_preprocess.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "module_genai/utils/tensor_utils.hpp"

namespace ov::genai::module {

Qwen3_5VisionPreprocess::Qwen3_5VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type)
    : VisionPreprocess(model_type) {
    //   m_video_processor(std::make_unique<Qwen3_5VLVideoProcessor>(model_path)) {}
    m_preprocessor = std::make_shared<Qwen3_5Preprocessor>(model_path);
}

PreprocessOutput Qwen3_5VisionPreprocess::preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) {
    // OPENVINO_ASSERT(m_video_processor != nullptr);
    // OPENVINO_ASSERT(images.empty() || videos.empty(), "Qwen3_5VisionPreprocess: images and videos cannot both be non-empty");

    // if (!videos.empty()) {
    //     m_video_processor->preprocess(videos);
    //     return;
    // }
    // m_video_processor->preprocess(images);

    ov::Tensor stack_images;
    if (images.size() > 1) {
        stack_images = tensor_utils::stack(images, 0);
    } else if (images.size() == 1) {
        stack_images = images[0];
    } else {
        OPENVINO_THROW("No images provided for preprocessing");
    }
    auto output = m_preprocessor->preprocess(stack_images);

    PreprocessOutput preprocess_output;
    preprocess_output.pixel_values = std::move(output.pixel_values);
    preprocess_output.grid_thw = std::move(output.grid_thw);
    preprocess_output.pos_embeds = std::move(output.pos_embeds);
    preprocess_output.rotary_cos = std::move(output.rotary_cos);
    preprocess_output.rotary_sin = std::move(output.rotary_sin);
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
