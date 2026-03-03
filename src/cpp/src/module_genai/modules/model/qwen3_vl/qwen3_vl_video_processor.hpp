// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "module_genai/modules/model/qwen3_5/qwen3_5config.hpp"
#include "module_genai/utils/video_processing_utils.hpp"
#include "module_genai/utils/video_utils.hpp"

namespace ov::genai::module {

class Qwen3VLVideoProcessor : public BaseVideoProcessor {
public:
    Qwen3VLVideoProcessor() = delete;
    explicit Qwen3VLVideoProcessor(const std::filesystem::path& model_path);
    void sample_frames(VideoMetadata metadata, int num_frames = 0, float fps = 0.0f);

    void preprocess(const std::vector<ov::Tensor>& frames);
};

}  // namespace ov::genai::module