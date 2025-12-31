// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

#include "visual_language/qwen2vl/classes.hpp"

namespace ov {
namespace genai {
namespace module {
class ImagePreprocessModule : public IBaseModule {
protected:
    ImagePreprocessModule() = delete;
    ImagePreprocessModule(const IBaseModuleDesc::PTR& desc);

private:
    std::shared_ptr<VisionEncoderQwen2VL> encoder_ptr = nullptr;

public:
    ~ImagePreprocessModule();

    void run() override;

    using PTR = std::shared_ptr<ImagePreprocessModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new ImagePreprocessModule(desc));
    }

    static void print_static_config();
};

REGISTER_MODULE_CONFIG(ImagePreprocessModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
