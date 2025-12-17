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

TextEncoderModule::TextEncoderModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    
}

bool TextEncoderModule::initialize() {
    const auto& params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        std::cerr << "TextEncoderModule[" << module_desc->name << "]: 'model_path' not found in params" << std::endl;
        return false;
    }
    
    std::filesystem::path tokenizer_path = it_path->second;
    return initialize(tokenizer_path, m_tokenization_params);
}

void TextEncoderModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
    
    // TODO: 从 inputs 中获取 prompt 数据
    if (m_prompt.empty()) {
        return;
    }
    auto encoded = run(m_prompt);
    // TODO: 设置outputs
}

bool TextEncoderModule::initialize(const std::filesystem::path& tokenizer_path,
                                   const ov::AnyMap& properties) {
    m_tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, properties);
    m_tokenization_params = properties;
    return true;
}

TokenizedInputs TextEncoderModule::run(const std::string& prompt) {
    // TODO: 似乎暂时缺少各模块的initiate？hard code here
    std::filesystem::path tokenizer_path = "/mnt/disk2/Qwen2.5-VL-7B-Instruct-INT4";
    m_tokenizer_impl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, m_tokenization_params);

    OPENVINO_ASSERT(m_tokenizer_impl, "TextEncoderModule is not initialized. Call initialize() first.");
    check_arguments(m_tokenization_params, {ov::genai::add_special_tokens.name(),
                                            ov::genai::max_length.name(),
                                            ov::genai::pad_to_max_length.name(),
                                            ov::genai::padding_side.name()});
    return m_tokenizer_impl->encode(prompt, m_tokenization_params);
}

}  // namespace module
}  // namespace genai
}  // namespace ov
