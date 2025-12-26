// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference.hpp"
#include <fstream>

// ============ To Be Removed: laod position_ids_list from files (debug purpose) ============
std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> load_test_data_position_ids_list() {
    std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> position_ids_list;

    // pair_count: 1
    // pair_0_element_type: i64
    // pair_0_shape: 3,1,30
    // pair_0_byte_size: 720
    // pair_0_has_rope_delta: true
    // pair_0_rope_delta: -2

    ov::element::Type element_type = ov::element::i64;
    ov::Shape shape = {3, 1, 30};
    size_t byte_size = 720;
    bool has_rope_delta = true;
    int64_t rope_delta_value = -2;

    ov::Tensor tensor(element_type, shape);
    std::string bin_path = "ut_test_data/position_ids_0.bin";
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (bin_file.is_open()) {
        bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
        bin_file.close();
    } else {
        return position_ids_list;
    }

    std::optional<int64_t> rope_delta = has_rope_delta ? std::optional<int64_t>(rope_delta_value) : std::nullopt;
    position_ids_list.emplace_back(std::move(tensor), rope_delta);

    return position_ids_list;
}
// ============ To Be Removed: laod position_ids_list from files (debug purpose) ============

namespace ov {
namespace genai {
namespace module {

void LLMInferenceModule::print_static_config() {
    std::cout << R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:
  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "embeds_list"
        type: "VecOVTensor"
    outputs:
      - name: "position_ids_list"
        type: "VecOVTensor"
  llm_inference:
    type: "LLMInferenceModule"
    description: "LLM module for Continuous Batch pipeline"
    device: "CPU"
    inputs:
      - name: "embeds_list"
        type: "VecOVTensor"
        source: "pipeline_params.embeds_list"
      - name: "position_ids_list"
        type: "VecOVTensor"
        source: "pipeline_params.position_ids_list"
    outputs:
      - name: "generated_text"
        type: "String"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
      max_new_tokens: "256"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "generated_text"
        type: "String"
        source: "llm_inference.generated_text"
    )" << std::endl;
}

LLMInferenceModule::LLMInferenceModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    if (!initialize()) {
        std::cerr << "Failed to initialize LLMInferenceModule" << std::endl;
    }
}

bool LLMInferenceModule::load_generation_config(const std::string& config_path) {
    try {
        std::ifstream f(config_path);
        if (!f.is_open()) {
        	GENAI_ERR("Failed to open generation config file: " + config_path);
            return false;
        }
        m_generation_config = ov::genai::GenerationConfig(config_path);
        return true;
    } catch (const std::exception& e) {
    	GENAI_ERR(std::string("Error loading generation config: ") + e.what());
        return false;
    }
}

bool LLMInferenceModule::initialize() {
    const auto& params = module_desc->params;

    auto it_models_path = params.find("model_path");
    if (it_models_path == params.end()) {
    	GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: 'models_path' not found in params");
        return false;
    }
    std::filesystem::path models_path = it_models_path->second;

    // Get device
    std::string device = module_desc->device.empty() ? "GPU" : module_desc->device;

    // Force to use PA backend
    ov::AnyMap cfg{};
    cfg["ATTENTION_BACKEND"] = "PA";

    load_generation_config(it_models_path->second + "generation_config.json");
    // Override with parameters from module config
    auto apply_param = [&](const std::string& key, auto& target, auto converter) {
        auto it = params.find(key);
        if (it != params.end() && !it->second.empty()) {
            try {
                target = converter(it->second);
            } catch (...) {
                std::cerr << "Failed to parse parameter: " << key << std::endl;
            }
        }
    };

    apply_param("max_new_tokens", m_generation_config.max_new_tokens,
                [](const std::string& s) { return std::stoull(s); });
    apply_param("do_sample", m_generation_config.do_sample,
                [](const std::string& s) { return s == "true" || s == "1"; });
    apply_param("top_p", m_generation_config.top_p,
                [](const std::string& s) { return std::stof(s); });
    apply_param("top_k", m_generation_config.top_k,
                [](const std::string& s) { return std::stoull(s); });
    apply_param("temperature", m_generation_config.temperature,
                [](const std::string& s) { return std::stof(s); });
    apply_param("repetition_penalty", m_generation_config.repetition_penalty,
                [](const std::string& s) { return std::stof(s); });
    apply_param("apply_chat_template", m_generation_config.apply_chat_template,
                [](const std::string& s) { return s == "true" || s == "1"; });

    try {
		auto [properties, attention_backend] = utils::extract_attention_backend(cfg);
		auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
		m_cb_pipeline = std::make_unique<ov::genai::VLMPipeline::VLMContinuousBatchingAdapter>(models_path, scheduler_config, device, plugin_properties);
    	return true;
    } catch (const std::exception& e) {
    	GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: Failed to create pipeline: " + e.what());
        return false;
    }
}

void LLMInferenceModule::run() {
	GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    if (this->inputs.count("embeds") == 0 && this->inputs.count("embeds_list") == 0) {
    	GENAI_ERR("LlmInferenceModule[" + module_desc->name + "]: 'embeds' input not found");
    	return;
    }

    if(this->inputs.count("position_ids") == 0 && this->inputs.count("position_ids_list") == 0) {
    	GENAI_ERR("LlmInferenceModule[" + module_desc->name + "]: 'position_ids' input not found");
    	return;
    }

    ov::Tensor input_embeds;
    std::vector<ov::Tensor> input_embeds_list;
    std::pair<ov::Tensor, std::optional<int64_t>> input_position_ids;
    std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> input_position_ids_list;
    if (this->inputs.count("embeds")) {
    	input_embeds = this->inputs["embeds"].data.as<ov::Tensor>();
    }
    if (this->inputs.count("embeds_list")) {
    	input_embeds_list = this->inputs["embeds_list"].data.as<std::vector<ov::Tensor>>();
    }

    if (this->inputs.count("position_ids")) {
    	input_position_ids = this->inputs["position_ids"].data.as<std::pair<ov::Tensor, std::optional<int64_t>>>();
    }
    if (this->inputs.count("position_ids_list")) {
    	input_position_ids_list = this->inputs["position_ids_list"].data.as<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>();
        //TODO, Fixme, unit test purpose only, to be deleted after solution available for parameter passing around
        input_position_ids_list = load_test_data_position_ids_list();
    }

    ov::genai::VLMDecodedResults vlm_results;
    std::vector<ov::genai::GenerationConfig> configs_vec = {m_generation_config};
    std::vector<ov::genai::EncodedGenerationResult> results_vec;
    std::optional<std::vector<ov::Tensor>> opt_token_type_ids = std::nullopt;

    // TODO,FIXME: support single instance of embeds and input_position_ids case.
    results_vec = m_cb_pipeline->generate(input_embeds_list, configs_vec, std::monostate{}, opt_token_type_ids, input_position_ids_list);
    std::string generated_text = "";
    if (results_vec.size()) {
        auto& results = results_vec[0];
        // Decode the generated token IDs to text
        if (!results.m_generation_ids.empty() && !results.m_generation_ids[0].empty()) {
            generated_text = m_cb_pipeline->get_tokenizer().decode(results.m_generation_ids[0]);
    	    GENAI_INFO("LLM output: " + generated_text);
    	}
        this->outputs["generated_text"].data = generated_text;
    }

    GENAI_INFO("LLMInferenceModule[" + module_desc->name + "] generation completed.");
}

}  // namespace module
}  // namespace genai
}  // namespace ov
