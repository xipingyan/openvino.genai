// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include "module_genai/pipeline/module_base.hpp"
#include "module_genai/utils/video_utils.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai::module {

using OutputModule = IBaseModule::OutputModule;

struct PreprocessOutput {
    std::vector<ov::Tensor> pixel_values;
    std::vector<ov::Tensor> grid_thw;
    std::vector<ov::Tensor> pos_embeds;
    std::vector<ov::Tensor> rotary_cos;
    std::vector<ov::Tensor> rotary_sin;

    // Video-specific outputs:
    std::vector<ov::Tensor> pixel_values_videos;
    std::vector<ov::Tensor> video_grid_thw;
    std::vector<ov::Tensor> video_pos_embeds;
    std::vector<ov::Tensor> video_rotary_cos;
    std::vector<ov::Tensor> video_rotary_sin;
    std::vector<int> video_second_per_grid;
};

// Vision preprocessing facade.
//
// The public API is intentionally model-agnostic so we can add more backends
// later. Currently, the factory returns a Qwen3_5VisionPreprocess instance
// (and QWEN3_VL is not yet implemented and returns nullptr).
class VisionPreprocess {
public:
    using PTR = std::shared_ptr<VisionPreprocess>;

    static PTR create(const std::filesystem::path& model_path, const std::string& device, VLMModelType model_type = VLMModelType::QWEN3_VL);

    VisionPreprocess(const VisionPreprocess&) = delete;
    VisionPreprocess& operator=(const VisionPreprocess&) = delete;

    virtual ~VisionPreprocess() = default;

    // Preprocess images and videos.
    virtual PreprocessOutput preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) = 0;

private:
    VisionPreprocess() = delete;

protected:
    explicit VisionPreprocess(VLMModelType model_type, const std::string& device) : _model_type(model_type), m_device(device) {}
    VLMModelType _model_type;
    std::string m_device;
};

}  // namespace ov::genai::module
