// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <iostream>
#include <string>
#include <map>

namespace ov {
namespace genai {
namespace module {

class ModuleRegistry {
public:
    using ConfigFunc = std::function<void()>;

    struct Entry {
        std::string name;
        ConfigFunc print_func;
    };

    static ModuleRegistry& getInstance() {
        static ModuleRegistry instance;
        return instance;
    }

    void add(const std::string& name, ConfigFunc func) {
        entries[name] = func;
    }

    std::vector<std::string> listAll() const {
        std::vector<std::string> names;
        for (auto const& [name, _] : entries)
            names.push_back(name);
        return names;
    }

    void printOne(const std::string& name) const {
        if (entries.count(name)) {
            std::cout << "\n--- [" << name << " Configuration] ---" << std::endl;
            entries.at(name)();
        } else {
            std::cerr << "Module not found: " << name << std::endl;
        }
    }

    void printAll() const {
        for (auto const& [name, _] : entries) {
            printOne(name);
        }
    }

private:
    std::map<std::string, ConfigFunc> entries;
    ModuleRegistry() = default;
};

#define REGISTER_MODULE_CONFIG(ClassName)                                               \
    static bool ClassName##_entry = []() {                                              \
        ModuleRegistry::getInstance().add(#ClassName, &ClassName::print_static_config); \
        return true;                                                                    \
    }();

}  // namespace module
}  // namespace genai
}  // namespace ov
