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

}  // namespace ov::genai::module
