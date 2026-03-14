// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3OmniAudioIO {
    static constexpr const char* kInputFeatures = "input_features";
    static constexpr const char* kFeatureAttentionMask = "feature_attention_mask";
    static constexpr const char* kAudioFeatureLengths = "audio_feature_lengths";
    static constexpr const char* kAudioFeatures = "audio_features";
};

std::shared_ptr<ov::Model> create_qwen3_omni_audio_encoder_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
