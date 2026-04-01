// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni.hpp"

#include "openvino/core/except.hpp"
#include "utils.hpp"

namespace ov::genai::module {

LLMInferencePAModule_Qwen3Omni::LLMInferencePAModule_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                                                           const PipelineDesc::PTR& pipeline_desc,
                                                           const VLMModelType& model_type)
    : LLMInferencePAModule(desc, pipeline_desc, model_type) {
}

LLMInferencePAModule::InputsParams::PTR LLMInferencePAModule_Qwen3Omni::parse_inputs(
    LLMInferencePAModule::InputsParams::PTR inputs_params) {
    InputsParamsQwen3_OMNI::PTR cur_inputs;
    if (inputs_params == nullptr) {
        cur_inputs = InputsParamsQwen3_OMNI::create();
    } else {
        cur_inputs = std::static_pointer_cast<InputsParamsQwen3_OMNI>(inputs_params);
    }

    cur_inputs = std::dynamic_pointer_cast<InputsParamsQwen3_OMNI>(LLMInferencePAModule::parse_inputs(cur_inputs));

    if (exists_input("visual_embeds")) {
        cur_inputs->visual_embeds = get_input("visual_embeds").as<ov::Tensor>();
    }
    if (exists_input("visual_pos_mask")) {
        cur_inputs->visual_pos_mask = get_input("visual_pos_mask").as<ov::Tensor>();
    }
    if (exists_input("grid_thw")) {
        cur_inputs->grid_thw = get_input("grid_thw").as<std::vector<ov::Tensor>>();
    }
    if (exists_input("position_ids")) {
        cur_inputs->position_ids = get_input("position_ids").as<ov::Tensor>();
    }
    if (exists_input("rope_delta")) {
        cur_inputs->rope_deltas = get_input("rope_delta").as<ov::Tensor>();
    }
    if (exists_input("deepstack_embeds")) {
        cur_inputs->deepstack_embeds = get_input("deepstack_embeds").as<std::vector<ov::Tensor>>();
    }

    return std::static_pointer_cast<InputsParams>(cur_inputs);
}

bool LLMInferencePAModule_Qwen3Omni::initialize() {
    m_device = module_desc->device.empty() ? "CPU" : module_desc->device;

    const auto device_param = get_optional_param("device");
    if (!device_param.empty()) {
        m_device = device_param;
    }

    check_cache_dir();

    std::filesystem::path model_path;
    if (exists_param("model_path")) {
        model_path = get_param("model_path");
        if (std::filesystem::is_regular_file(model_path)) {
            model_path = model_path.parent_path();
        }
        std::filesystem::path text_embeds_model_path = model_path / "openvino_text_embeddings_model.xml";
        std::filesystem::path llm_model_path = model_path / "openvino_language_model.xml";
        OPENVINO_ASSERT(std::filesystem::exists(text_embeds_model_path),
                        "LLMInferencePAModule: openvino_text_embeddings_model.xml not found: ",
                        text_embeds_model_path.string());
        OPENVINO_ASSERT(std::filesystem::exists(llm_model_path),
                        "LLMInferencePAModule: openvino_language_model.xml not found: ",
                        llm_model_path.string());
        // Temp solution: embeds merger is called in this module, not continuseous batching pipeline, so the text embeddings model must be loaded together with language model as a single model.
        // In the future, if we want to load text embeddings model and language model separately, we can implement the embeds merger as a separate module and load text embeddings model in that module.
        std::filesystem::path merge_embeds_model_path = get_param("merge_embeds_model_path");
        OPENVINO_ASSERT(std::filesystem::is_regular_file(merge_embeds_model_path),
                        "LLMInferencePAModule: merge_embeds_model_path is not regular file: ",
                        merge_embeds_model_path.string());
        auto merge_embeds_model = utils::singleton_core().read_model(merge_embeds_model_path.string());
        auto merge_embeds_cm = utils::singleton_core().compile_model(merge_embeds_model, m_device);
        m_merge_embeds_infer_request = merge_embeds_cm.create_infer_request();
    } else {
        OPENVINO_THROW("LLMInferencePAModule: either text_model_path or both openvino_text_embeddings_model.xml and "
                       "openvino_language_model.xml must be provided");
    }

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

    m_pipeline =
        std::make_unique<ov::genai::ContinuousBatchingPipeline>(model_path, scheduler_config, m_device, properties);
    const auto eos_id = m_pipeline->get_tokenizer().get_eos_token_id();
    if (eos_id >= 0) {
        m_stop_ids.insert(eos_id);
    }
}

void LLMInferencePAModule_Qwen3Omni::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    const auto inputs_params = std::dynamic_pointer_cast<InputsParamsQwen3_OMNI>(parse_inputs());
    OPENVINO_ASSERT(inputs_params != nullptr, "LLMInferencePAModule_Qwen3Omni: failed to parse inputs");

    const bool has_visual_embeds = !inputs_params->visual_embeds.get_shape().empty();
    const bool has_visual_pos_mask = !inputs_params->visual_pos_mask.get_shape().empty();
    const bool has_grid_thw = !inputs_params->grid_thw.empty();
    const bool has_position_ids = !inputs_params->position_ids.get_shape().empty();
    const bool has_rope_deltas = !inputs_params->rope_deltas.get_shape().empty();
    const bool has_deepstack_embeds =
        inputs_params->deepstack_embeds.has_value() && !inputs_params->deepstack_embeds.value().empty();

    OPENVINO_ASSERT(!(has_visual_embeds || has_visual_pos_mask || has_grid_thw || has_position_ids || has_rope_deltas ||
                      has_deepstack_embeds),
                    "LLMInferencePAModule_Qwen3Omni: visual/audio/deepstack inputs are not supported by PA pipeline yet");

    const std::vector<ov::genai::GenerationConfig> sampling_params = {m_generation_config};

    std::string generated_text;
    if (inputs_params->has_input_ids) {
        const std::vector<ov::Tensor> input_ids = {inputs_params->input_ids};
        const auto results = m_pipeline->generate(input_ids, sampling_params);
        generated_text = decode_first_result(results);
    } else if (inputs_params->has_input_embeds) {
        const std::vector<ov::Tensor> input_embeds = {inputs_params->input_embeds};
        const auto results =
            m_pipeline->generate(input_embeds, sampling_params, std::monostate{}, std::nullopt, std::nullopt);
        generated_text = decode_first_result(results);
    } else {
        OPENVINO_ASSERT(false,
                        "LLMInferencePAModule_Qwen3Omni: one of input_ids/input_embeds/merged_embeds must be provided");
    }

    this->outputs["generated_text"].data = generated_text;
}

}  // namespace ov::genai::module
