// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_base.hpp"

namespace ov {
namespace genai {
namespace module {

IBaseModule::IBaseModule(const IBaseModuleDesc::PTR& desc) : module_desc(desc) {
    std::cout << "Init IBaseModule with module name : " << module_desc->name << std::endl;
    for (auto& input : desc->inputs) {
        this->inputs[input.source_module_out_name] = InputModule();
    }
    for (auto& output : desc->outputs) {
        this->outputs[output.name] = OutputModule();
    }
}

void IBaseModule::prepare_inputs() {
    for (auto& input : this->inputs) {
        const auto& parent_port_name = input.first;
        input.second.data = input.second.module_ptr->outputs[parent_port_name].data;
    }
}

const std::string& IBaseModule::get_module_name() const {
    return module_desc->name;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
