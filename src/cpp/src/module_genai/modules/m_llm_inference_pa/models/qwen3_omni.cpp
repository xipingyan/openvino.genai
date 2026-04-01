// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni.hpp"

#include "module_genai/utils/profiler.hpp"
#include "utils.hpp"

namespace ov::genai::module {

LLMInferencePAModule_Qwen3Omni::InputsParams::PTR LLMInferencePAModule_Qwen3Omni::parse_inputs(
    LLMInferencePAModule_Qwen3Omni::InputsParams::PTR inputs_params) {
     auto cur_inputs = inputs_params ? inputs_params : std::make_shared<InputsParamsQwen3_OMNI>();

    return std::static_pointer_cast<InputsParams>(cur_inputs);
}

LLMInferencePAModule_Qwen3Omni::LLMInferencePAModule_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                                                           const PipelineDesc::PTR& pipeline_desc,
                                                           const VLMModelType& model_type)
    : LLMInferencePAModule(desc, pipeline_desc, model_type) {
    //
    
}

void LLMInferencePAModule_Qwen3Omni::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    std::string generated_text;

    prepare_inputs();
    auto inputs_params = std::dynamic_pointer_cast<InputsParamsQwen3_OMNI>(parse_inputs());

    
    // GENAI_INFO("LLM output: " + generated_text);
    // this->outputs["generated_text"].data = generated_text;
}

namespace LLMInferenceSDPAModule_Utils {

}  // namespace LLMInferenceSDPAModule_Utils

}  // namespace ov::genai::module
