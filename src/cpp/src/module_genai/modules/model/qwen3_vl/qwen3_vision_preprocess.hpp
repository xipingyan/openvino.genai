// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "module_genai/modules/md_img_preprocess/vision_preprocess.hpp"
#include "module_genai/modules/model/qwen3_vl/qwen3_vl_video_processor.hpp"

namespace ov::genai::module {

class Qwen3VisionPreprocess final : public VisionPreprocess {
public:
    Qwen3VisionPreprocess() = delete;
    Qwen3VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type);

    void preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) override;

    void result_to_output(std::map<std::string, OutputModule>& output) const override;

private:
    std::unique_ptr<IVideoProcessor> m_video_processor;
};

}  // namespace ov::genai::module
