// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>

#include "openvino/genai/module_genai/pipeline.hpp"

int main(int argc, char** argv) {
    using namespace ov::genai::module;

    // Minimal CLI:
    // - no args: print all modules' config/spec templates
    // - one arg: print only the specified module entry
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [ModuleName]" << std::endl;
        return 2;
    }

    const auto modules = ListAllModules();
    std::cout << "Registered modules: " << modules.size() << std::endl;

    if (argc == 2) {
        PrintModuleConfig(std::string(argv[1]));
        return 0;
    }

    PrintAllModulesConfig();
    return 0;
}
