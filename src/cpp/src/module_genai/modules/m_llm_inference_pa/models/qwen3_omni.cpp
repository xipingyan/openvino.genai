// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni.hpp"

#include "openvino/core/except.hpp"

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
