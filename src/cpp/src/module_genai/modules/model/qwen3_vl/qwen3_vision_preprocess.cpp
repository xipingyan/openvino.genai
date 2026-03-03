// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/model/qwen3_vl/qwen3_vision_preprocess.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace ov::genai::module {

Qwen3VisionPreprocess::Qwen3VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type)
    : VisionPreprocess(model_type),
      m_video_processor(std::make_unique<Qwen3VLVideoProcessor>(model_path)) {}

void Qwen3VisionPreprocess::preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) {
    OPENVINO_ASSERT(m_video_processor != nullptr);
    OPENVINO_ASSERT(images.empty() || videos.empty(), "Qwen3VisionPreprocess: images and videos cannot both be non-empty");

    if (!videos.empty()) {
        m_video_processor->preprocess(videos);
        return;
    }
    m_video_processor->preprocess(images);
}

void Qwen3VisionPreprocess::result_to_output(std::map<std::string, OutputModule>& output) const {
    (void)output;
}

}  // namespace ov::genai::module
