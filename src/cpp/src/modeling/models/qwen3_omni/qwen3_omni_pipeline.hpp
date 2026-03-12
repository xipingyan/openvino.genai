// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3OmniMultimodalInfo {
    nlohmann::json audios;
    nlohmann::json images;
    nlohmann::json videos;
    nlohmann::json video_kwargs;
};

struct Qwen3OmniAudioFeatureInfo {
	nlohmann::json input_ids;
	nlohmann::json attention_mask;
	nlohmann::json position_ids;
	nlohmann::json visual_pos_mask;
	nlohmann::json rope_deltas;
	nlohmann::json visual_embeds_padded;
	nlohmann::json deepstack_padded;
	nlohmann::json audio_features;
	nlohmann::json audio_pos_mask;
	nlohmann::json feature_attention_mask;
	nlohmann::json audio_feature_lengths;
	nlohmann::json video_second_per_grid;
};

int resolve_qwen3_omni_code_predictor_steps(
	const Qwen3OmniConfig& cfg,
	const Qwen3OmniPipelineBuildOptions& options = {});

Qwen3OmniMultimodalInfo process_qwen3_omni_multimodal_info(
	const nlohmann::json& conversations,
	bool use_audio_in_video,
	bool return_video_kwargs,
	const std::string& python_executable = "/home/wanglei/py_venv/dev_env/bin/python");

Qwen3OmniAudioFeatureInfo process_qwen3_omni_audio_feature_info(
	const nlohmann::json& conversations,
	const std::string& model_dir,
	bool use_audio_in_video,
	const std::string& python_executable = "/home/wanglei/py_venv/dev_env/bin/python");

void validate_qwen3_omni_pipeline_models(
	const Qwen3OmniPipelineModels& models,
	int expected_code_predictor_steps);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

