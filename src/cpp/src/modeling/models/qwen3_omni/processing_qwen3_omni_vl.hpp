// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

using Qwen3OmniVisionInputs = Qwen3VLVisionInputs;
using Qwen3OmniVisionPreprocessConfig = Qwen3VLVisionPreprocessConfig;

struct Qwen3OmniTextIO {
    static constexpr const char* kInputIds = Qwen3VLTextIO::kInputIds;
    static constexpr const char* kInputsEmbeds = Qwen3VLTextIO::kInputsEmbeds;
    static constexpr const char* kAttentionMask = Qwen3VLTextIO::kAttentionMask;
    static constexpr const char* kPositionIds = Qwen3VLTextIO::kPositionIds;
    static constexpr const char* kBeamIdx = Qwen3VLTextIO::kBeamIdx;
    static constexpr const char* kVisualEmbeds = Qwen3VLTextIO::kVisualEmbeds;
    static constexpr const char* kVisualPosMask = Qwen3VLTextIO::kVisualPosMask;
    static constexpr const char* kDeepstackEmbedsPrefix = Qwen3VLTextIO::kDeepstackEmbedsPrefix;
    static constexpr const char* kAudioFeatures = "audio_features";
    static constexpr const char* kAudioPosMask = "audio_pos_mask";
    static constexpr const char* kLogits = Qwen3VLTextIO::kLogits;
};

class Qwen3OmniVisionPreprocessor {
public:
    Qwen3OmniVisionPreprocessor(const Qwen3OmniConfig& cfg,
                                const Qwen3OmniVisionPreprocessConfig& preprocess_cfg);

    Qwen3OmniVisionInputs preprocess(const ov::Tensor& images,
                                     const ov::Tensor& pos_embed_weight) const;

    /// Preprocess video frames (each [1, H, W, 3] u8) for the vision encoder.
    Qwen3OmniVisionInputs preprocess_video(const std::vector<ov::Tensor>& frames,
                                           const ov::Tensor& pos_embed_weight,
                                           size_t video_min_pixels = 128 * 28 * 28,
                                           size_t video_max_pixels = 768 * 28 * 28) const;

    static int64_t count_visual_tokens(const ov::Tensor& grid_thw,
                                       int32_t spatial_merge_size);

private:
    Qwen3VLVisionPreprocessor preprocessor_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
