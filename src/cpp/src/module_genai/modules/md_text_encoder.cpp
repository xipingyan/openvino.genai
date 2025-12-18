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
    device: "GPU"
    inputs:
      - name: "prompts"
        type: "String"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "prompt_embedding"
        type: "OVRemoteTensor"
      - name: "mask"
        type: "OVRemoteTensor"
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
    
    std::filesystem::path tokenizer_path = it_path->second;
    m_tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, m_tokenization_params);
    return true;
}

void TextEncoderModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
    
    prepare_inputs();
    m_prompts = this->inputs["prompts_data"].data.as<std::vector<std::string>>();
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
