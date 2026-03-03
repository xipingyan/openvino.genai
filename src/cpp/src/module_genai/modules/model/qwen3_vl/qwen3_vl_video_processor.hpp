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

class IVideoProcessor {
public:
    virtual ~IVideoProcessor() = default;

    virtual void sample_frames(VideoMetadata metadata, int num_frames = 0, float fps = 0.0f) = 0;
    virtual void preprocess(const std::vector<ov::Tensor>& frames) = 0;
};

class Qwen3VLVideoProcessor : public BaseVideoProcessor, public IVideoProcessor {
public:
    Qwen3VLVideoProcessor() = delete;
    explicit Qwen3VLVideoProcessor(const std::filesystem::path& model_path);
    void sample_frames(VideoMetadata metadata, int num_frames = 0, float fps = 0.0f) override;
    void preprocess(const std::vector<ov::Tensor>& frames) override;
};

}  // namespace ov::genai::module