// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_text_encoder.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

void TextEncoderModule::print_static_config() {
    std::cout << R"(
  prompt_encoder:                       # Module Name
    type: "TextEncoderModule"
    description: "Encode prompt to prompt ids."
    device: "GPU"
    inputs:
      - name: "prompt"
        type: "String"            # [Optional] Support DataType: [String]
        source: "ParentModuleName.OutputPortName"
      - name: "prompts"
        type: "VecString"         # [Optional] Support DataType: [VecString]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "prompt_embedding"
        type: "OVRemoteTensor"     # Support DataType: [OVTensor, OVRemoteTensor]
      - name: "mask"
        type: "OVRemoteTensor"     # Support DataType: [OVTensor, OVRemoteTensor]
    params:
      model_path: "models/text_encoder.xml"  # Optional. OpenVINO IR
    )" << std::endl;
}

TextEncoderModule::TextEncoderModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    if (!initialize()) {
        std::cerr << "Failed to initiate TextEncoderModule" << std::endl;
    }
}

bool TextEncoderModule::initialize() {
    const auto& params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        std::cerr << "TextEncoderModule[" << module_desc->name << "]: 'model_path' not found in params" << std::endl;
        return false;
    }
    
    std::filesystem::path tokenizer_path = module_desc->get_full_path(it_path->second);

    m_tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, m_tokenization_params);
    OPENVINO_ASSERT(m_tokenizer_impl->m_ireq_queue_tokenizer != nullptr, std::string("Load tokenizer model fail: ") + tokenizer_path.c_str());
    return true;
}

void TextEncoderModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
    
    prepare_inputs();
    std::vector<std::string> m_prompts = {};
    if (this->inputs.find("prompts") != this->inputs.end()) {
        m_prompts = this->inputs["prompts"].data.as<std::vector<std::string>>();
    }
    if (this->inputs.find("prompt") != this->inputs.end()) {
        std::string single_prompt = this->inputs["prompt"].data.as<std::string>();
        m_prompts.insert(m_prompts.begin(), single_prompt);
    }

    auto encoded = run(m_prompts);

    this->outputs["input_ids"].data = encoded.input_ids;
    this->outputs["mask"].data = encoded.attention_mask;
}

TokenizedInputs TextEncoderModule::run(const std::vector<std::string>& prompts) {
    OPENVINO_ASSERT(m_tokenizer_impl, "TextEncoderModule is not initialized. Call initialize() first.");
    check_arguments(m_tokenization_params, {ov::genai::add_special_tokens.name(),
                                            ov::genai::max_length.name(),
                                            ov::genai::pad_to_max_length.name(),
                                            ov::genai::padding_side.name()});
    return m_tokenizer_impl->encode(prompts, m_tokenization_params);
}

}  // namespace module
}  // namespace genai
}  // namespace ov
