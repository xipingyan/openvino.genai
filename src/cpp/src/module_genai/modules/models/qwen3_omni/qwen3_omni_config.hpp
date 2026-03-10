// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_OPENVINO_NEW_ARCH
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include <filesystem>
#include <optional>

namespace ov::genai::module {

struct Qwen3OmniVisionInput {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
};

struct Qwen3OmniAudioInput {
    ov::Tensor audio_features;
    ov::Tensor audio_feature_lengths;
};

struct Qwen3OmniVisionEmbeddingResult {
    ov::Tensor position_ids;
    ov::Tensor rope_deltas;
    std::optional<ov::Tensor> visual_embeds;
    std::optional<ov::Tensor> visual_pos_mask;
    std::optional<std::vector<ov::Tensor>> deepstack_embeds;
    std::optional<ov::Tensor> audio_embeds;
    std::optional<ov::Tensor> audio_pos_mask;
};

modeling::models::Qwen3VLConfig get_qwen3_omni_vl_config(const std::filesystem::path &config_path);

}
#endif
