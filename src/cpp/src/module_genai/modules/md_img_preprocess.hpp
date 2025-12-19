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
class ImagePreprocesModule : public IBaseModule {
protected:
    ImagePreprocesModule() = delete;
    ImagePreprocesModule(const IBaseModuleDesc::PTR& desc);

private:
    std::shared_ptr<VisionEncoderQwen2VL> encoder_ptr = nullptr;

public:
    ~ImagePreprocesModule();

    void run() override;

    using PTR = std::shared_ptr<ImagePreprocesModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new ImagePreprocesModule(desc));
    }

    static void print_static_config();
};

REGISTER_MODULE_CONFIG(ImagePreprocesModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
