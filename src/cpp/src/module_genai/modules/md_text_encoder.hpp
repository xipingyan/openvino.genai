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
    ~TextEncoderModule() {
        std::cout << "~TextEncoderModule is called." << std::endl;
    }

    void run() override;

    bool initialize(const std::filesystem::path& tokenizer_path,
                    const ov::AnyMap& properties = {});

    // TODO: 假定只有单条 prompt
    TokenizedInputs run(const std::string& prompt);

    using PTR = std::shared_ptr<TextEncoderModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new TextEncoderModule(desc));
    }

private:
    std::shared_ptr<Tokenizer::TokenizerImpl> m_tokenizer_impl;
    ov::AnyMap m_tokenization_params = {};
    std::string m_prompt = "Describe this image.";
};

}  // namespace module
}  // namespace genai
}  // namespace ov
