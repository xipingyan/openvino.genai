// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"

namespace ov {
namespace genai {
namespace module {

class TextEncoderModule : public IBaseModule {
protected:
    TextEncoderModule() = delete;
    TextEncoderModule(const IBaseModuleDesc::PTR& desc);

public:
    ~TextEncoderModule() {}

    void run() override;

    using PTR = std::shared_ptr<TextEncoderModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new TextEncoderModule(desc));
    }
    static void print_static_config();

private:
    std::shared_ptr<Tokenizer::TokenizerImpl> m_tokenizer_impl;
    ov::AnyMap m_tokenization_params = {};
    bool initialize();
    TokenizedInputs run(const std::vector<std::string>& prompts);
};

REGISTER_MODULE_CONFIG(TextEncoderModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
