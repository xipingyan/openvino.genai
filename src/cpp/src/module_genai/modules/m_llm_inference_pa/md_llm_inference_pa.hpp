// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <set>
#include <string>

#include <yaml-cpp/yaml.h>

#include "module_genai/pipeline/module.hpp"
#include "module_genai/pipeline/module_type.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"

namespace ov::genai::module {

/// @brief LLM inference module using PA / ContinuousBatching backend.
///
/// Inputs:
///   - input_ids (OVTensor) or input_embeds (OVTensor)
///   - qwen3_omni extension: deepstacks (VecOVTensor), merged_embeds (OVTensor)
///
/// Output:
///   - generated_text (String)
///
/// Params:
///   - model_path (required)
///   - cache_dir (optional)
///   - max_new_tokens (optional)
///   - device (optional, overrides module-level device)
class LLMInferencePAModule : public IBaseModule {
protected:
	LLMInferencePAModule() = delete;
	LLMInferencePAModule(const IBaseModuleDesc::PTR& desc,
						 const PipelineDesc::PTR& pipeline_desc,
						 const VLMModelType& model_type);

public:
	~LLMInferencePAModule();

	using PTR = std::shared_ptr<LLMInferencePAModule>;
	static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc);
	static void print_static_config();
	static const ModuleSpec& get_spec();

	void run() override;

protected:
	struct InputsParams {
	public:
		DeclareClass_PTR_Create(InputsParams);
		ov::Tensor input_ids;
		ov::Tensor input_embeds;
		ov::Tensor merged_embeds;
		std::vector<ov::Tensor> deepstacks;
		bool has_input_ids = false;
		bool has_input_embeds = false;
		bool has_merged_embeds = false;
		bool has_deepstacks = false;
	};

	virtual InputsParams::PTR parse_inputs(InputsParams::PTR inputs_params = nullptr);
	bool initialize();
	std::string decode_first_result(const std::vector<ov::genai::EncodedGenerationResult>& results) const;

protected:
	std::unique_ptr<ov::genai::ContinuousBatchingPipeline> m_pipeline;
	ov::genai::GenerationConfig m_generation_config;

	std::set<int64_t> m_stop_ids;
	std::filesystem::path m_models_path;
	std::string m_device = "CPU";
	size_t m_max_new_tokens = 256;
	VLMModelType m_model_type;
};

REGISTER_MODULE_CONFIG(LLMInferencePAModule);

}  // namespace ov::genai::module
