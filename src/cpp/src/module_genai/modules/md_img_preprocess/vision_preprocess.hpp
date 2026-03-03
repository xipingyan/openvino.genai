// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include "module_genai/module_base.hpp"
#include "module_genai/utils/video_utils.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai::module {

using OutputModule = IBaseModule::OutputModule;

// Vision preprocessing facade.
//
// Current implementation encapsulates Qwen3VLVideoProcessor, but the public
// API is intentionally model-agnostic so we can add more backends later.
class VisionPreprocess {
public:
    using PTR = std::shared_ptr<VisionPreprocess>;

    static PTR create(const std::filesystem::path& model_path, VLMModelType model_type = VLMModelType::QWEN3_VL);

    VisionPreprocess(const VisionPreprocess&) = delete;
    VisionPreprocess& operator=(const VisionPreprocess&) = delete;

    virtual ~VisionPreprocess() = default;

    // Preprocess images and videos.
    virtual void preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) = 0;

    virtual void result_to_output(std::map<std::string, OutputModule>& output) const = 0;

private:
    VisionPreprocess() = delete;

protected:
    explicit VisionPreprocess(VLMModelType model_type) : _model_type(model_type) {}
    VLMModelType _model_type;
};

}  // namespace ov::genai::module
