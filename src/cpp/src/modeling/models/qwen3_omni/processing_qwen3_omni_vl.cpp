// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/processing_qwen3_omni_vl.hpp"

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3OmniVisionPreprocessor::Qwen3OmniVisionPreprocessor(
    const Qwen3OmniConfig& cfg,
    const Qwen3OmniVisionPreprocessConfig& preprocess_cfg)
    : preprocessor_(to_qwen3_omni_vl_cfg(cfg).vision, preprocess_cfg) {}

Qwen3OmniVisionInputs Qwen3OmniVisionPreprocessor::preprocess(const ov::Tensor& images,
                                                              const ov::Tensor& pos_embed_weight) const {
    return preprocessor_.preprocess(images, pos_embed_weight);
}

Qwen3OmniVisionInputs Qwen3OmniVisionPreprocessor::preprocess_video(
    const std::vector<ov::Tensor>& frames,
    const ov::Tensor& pos_embed_weight,
    size_t video_min_pixels,
    size_t video_max_pixels) const {
    return preprocessor_.preprocess_video(frames, pos_embed_weight, video_min_pixels, video_max_pixels);
}

int64_t Qwen3OmniVisionPreprocessor::count_visual_tokens(const ov::Tensor& grid_thw,
                                                         int32_t spatial_merge_size) {
    return Qwen3VLVisionPreprocessor::count_visual_tokens(grid_thw, spatial_merge_size);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
