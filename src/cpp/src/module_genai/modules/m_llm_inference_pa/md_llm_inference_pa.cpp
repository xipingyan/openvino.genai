// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference_pa.hpp"

#include "module_genai/pipeline/module_factory.hpp"

#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace ov::genai::module {

GENAI_REGISTER_MODULE_SAME(LLMInferencePAModule);

const ModuleSpec& LLMInferencePAModule::get_spec() {
	static const ModuleSpec spec = []() {
		ModuleSpec s("LLMInferencePAModule", "llm_inference_pa");
		s.add_input("input_ids", {DataType::OVTensor}, true);

		s.add_input("visual_embeds", {DataType::OVTensor}, true);
		s.add_input("visual_pos_mask", {DataType::OVTensor}, true);
		s.add_input("grid_thw", {DataType::VecOVTensor}, true);
		s.add_input("position_ids", {DataType::OVTensor}, true);
		s.add_input("rope_delta", {DataType::OVTensor}, true);
		s.add_input("deepstack_embeds", {DataType::VecOVTensor}, true);
		s.add_input("audio_embeds", {DataType::OVTensor}, true);
		s.add_input("audio_pos_mask", {DataType::OVTensor}, true);
		s.add_output("generated_text", {DataType::String});

		s.add_param("text_model_path", "text_model_path", true, "Directory containing text model and tokenizer files, need to be provided if text_embeds_model_path and llm_model_path are not provided");
		s.add_param("text_embeds_model_path", "text_embeds_model_path", true, "Directory containing text_embeds model and tokenizer files, text_embeds_model_path and llm_model_path need to be provided together");
		s.add_param("llm_model_path", "llm_model_path", true, "Directory containing llm model and tokenizer files, text_embeds_model_path and llm_model_path need to be provided together");

		s.add_param("cache_dir", "cache_dir", true, "Optional OpenVINO cache directory");
		s.add_param("max_new_tokens", "256", true, "Maximum number of new tokens to generate");
		s.add_param("device", "CPU", true, "Optional device override");
		s.add_param("split_text_model", "false", true, "Whether to split text model for better performance, only effective when text_model_path is provided");
		return s;
	}();
	return spec;
}

LLMInferencePAModule::PTR LLMInferencePAModule::create(const IBaseModuleDesc::PTR& desc,
													   const PipelineDesc::PTR& pipeline_desc) {
	const VLMModelType model_type = to_vlm_model_type(desc->model_type);
	return PTR(new LLMInferencePAModule(desc, pipeline_desc, model_type));
}

void LLMInferencePAModule::print_static_config() {
	std::cout << get_spec().to_yaml_template_string() << std::endl;
}

LLMInferencePAModule::LLMInferencePAModule(const IBaseModuleDesc::PTR& desc,
										   const PipelineDesc::PTR& pipeline_desc,
										   const VLMModelType& model_type)
	: IBaseModule(desc, pipeline_desc),
	  m_model_type(model_type) {
	check_params_with_spec(get_spec());
	if (!initialize()) {
		GENAI_ERR("Failed to initialize LLMInferencePAModule");
	}
}

LLMInferencePAModule::~LLMInferencePAModule() = default;

bool LLMInferencePAModule::initialize() {
	m_device = module_desc->device.empty() ? "CPU" : module_desc->device;

	const auto device_param = get_optional_param("device");
	if (!device_param.empty()) {
		m_device = device_param;
	}

	m_models_path = get_param("model_path");
	OPENVINO_ASSERT(!m_models_path.empty(), "LLMInferencePAModule: model_path is empty");
	OPENVINO_ASSERT(std::filesystem::is_directory(m_models_path),
					"LLMInferencePAModule: model_path must be an existing directory: ",
					m_models_path.string());

	check_cache_dir();

	const auto max_new_tokens_param = get_optional_param("max_new_tokens");
	if (!max_new_tokens_param.empty()) {
		m_max_new_tokens = str_to_size_t(max_new_tokens_param);
	}
	OPENVINO_ASSERT(m_max_new_tokens > 0, "LLMInferencePAModule: max_new_tokens must be > 0");

	m_generation_config = ov::genai::GenerationConfig{};
	m_generation_config.max_new_tokens = m_max_new_tokens;

	ov::AnyMap properties{};
	if (!m_cache_dir.empty()) {
		properties.insert({ov::cache_dir.name(), m_cache_dir});
	}

	const auto scheduler_config = ov::genai::utils::get_latency_oriented_scheduler_config();
	m_pipeline = std::make_unique<ov::genai::ContinuousBatchingPipeline>(
		m_models_path,
		scheduler_config,
		m_device,
		properties);

	const auto eos_id = m_pipeline->get_tokenizer().get_eos_token_id();
	if (eos_id >= 0) {
		m_stop_ids.insert(eos_id);
	}

	return true;
}

LLMInferencePAModule::InputsParams::PTR LLMInferencePAModule::parse_inputs(InputsParams::PTR inputs_params) {
	if (inputs_params == nullptr) {
		inputs_params = InputsParams::create();
	}

	if (exists_input("input_ids")) {
		inputs_params->input_ids = get_input("input_ids").as<ov::Tensor>();
		inputs_params->has_input_ids = true;
	}
	if (exists_input("input_embeds")) {
		inputs_params->input_embeds = get_input("input_embeds").as<ov::Tensor>();
		inputs_params->has_input_embeds = true;
	}
	if (exists_input("merged_embeds")) {
		inputs_params->merged_embeds = get_input("merged_embeds").as<ov::Tensor>();
		inputs_params->has_merged_embeds = true;
	}
	if (exists_input("deepstacks")) {
		inputs_params->deepstacks = get_input("deepstacks").as<std::vector<ov::Tensor>>();
		inputs_params->has_deepstacks = true;
	}

	OPENVINO_ASSERT(!(inputs_params->has_input_ids && (inputs_params->has_input_embeds || inputs_params->has_merged_embeds)),
					"LLMInferencePAModule: input_ids cannot be used together with input_embeds or merged_embeds");

	if (!inputs_params->has_input_embeds && inputs_params->has_merged_embeds) {
		inputs_params->input_embeds = inputs_params->merged_embeds;
		inputs_params->has_input_embeds = true;
	}

	OPENVINO_ASSERT(inputs_params->has_input_ids || inputs_params->has_input_embeds,
					"LLMInferencePAModule: one of input_ids/input_embeds/merged_embeds must be provided");

	OPENVINO_ASSERT(!inputs_params->has_deepstacks,
					"LLMInferencePAModule: deepstacks input is not supported in common PA implementation");

	return inputs_params;
}

std::string LLMInferencePAModule::decode_first_result(
	const std::vector<ov::genai::EncodedGenerationResult>& results) const {
	if (results.empty()) {
		return std::string();
	}
	const auto& first = results.front();
	if (first.m_generation_ids.empty() || first.m_generation_ids.front().empty()) {
		return std::string();
	}
	return m_pipeline->get_tokenizer().decode(first.m_generation_ids.front());
}

void LLMInferencePAModule::run() {
	GENAI_INFO("Running module: " + module_desc->name);

	prepare_inputs();
	const auto inputs_params = parse_inputs();

	const std::vector<ov::genai::GenerationConfig> sampling_params = {m_generation_config};

	std::string generated_text;
	if (inputs_params->has_input_ids) {
		const std::vector<ov::Tensor> input_ids = {inputs_params->input_ids};
		const auto results = m_pipeline->generate(input_ids, sampling_params);
		generated_text = decode_first_result(results);
	} else {
		const std::vector<ov::Tensor> input_embeds = {inputs_params->input_embeds};
		const auto results = m_pipeline->generate(input_embeds, sampling_params, std::monostate{}, std::nullopt, std::nullopt);
		generated_text = decode_first_result(results);
	}

	this->outputs["generated_text"].data = generated_text;
}

}  // namespace ov::genai::module
