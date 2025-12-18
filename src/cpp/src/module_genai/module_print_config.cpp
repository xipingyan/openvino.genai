// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_print_config.hpp"

#include "openvino/genai/module_genai/pipeline.hpp"

namespace ov {
namespace genai {
namespace module {

void PrintAllModulesConfig() {
    ModuleRegistry::getInstance().printAll();
}

std::vector<std::string> ListAllModules() {
    return ModuleRegistry::getInstance().listAll();
}

void PrintModuleConfig(const std::string& module_name) {
    ModuleRegistry::getInstance().printOne(module_name);
}

}  // namespace module
}  // namespace genai
}  // namespace ov
