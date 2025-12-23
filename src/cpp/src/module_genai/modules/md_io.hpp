// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
namespace ov {
namespace genai {
namespace module {

class ParameterModule : public IBaseModule {
protected:
    ParameterModule() = delete;
    ParameterModule(const IBaseModuleDesc::PTR& desc);

public:
    ~ParameterModule() {}

    void run() override {};

    void run(ov::AnyMap& inputs);

    using PTR = std::shared_ptr<ParameterModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new ParameterModule(desc));
    }

    static void print_static_config();
};

REGISTER_MODULE_CONFIG(ParameterModule);

class ResultModule : public IBaseModule {
protected:
    ResultModule() = delete;
    ResultModule(const IBaseModuleDesc::PTR& desc);

public:
    ~ResultModule()  {}

    void run() override {};
    void run(ov::AnyMap& outputs);

    using PTR = std::shared_ptr<ResultModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new ResultModule(desc));
    }

    static void print_static_config();
};

REGISTER_MODULE_CONFIG(ResultModule);
}  // namespace module
}  // namespace genai
}  // namespace ov
