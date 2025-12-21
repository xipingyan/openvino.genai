// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_io.hpp"

namespace ov {
namespace genai {
namespace module {

void ParameterModule::print_static_config() {
    std::cout << R"(
  image:                        # Module Name
    type: "ParameterModule"
    description: "Input parameters. Supported DataType: [OVTensor, VecOVTensor, String, VecString]"
    outputs:
      - name: "image1_data"     # Input Name, should algin with pipeline.generate inputs.
        type: "OVTensor"
      - name: "image2_data"
        type: "OVTensor"
        )" << std::endl;
}

ParameterModule::ParameterModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    is_input_module = true;
    // std::cout << "ParameterModule:" << m_desc << std::endl;
}

void ParameterModule::run(ov::AnyMap& inputs) {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;

    for (auto& output : this->outputs) {
        OPENVINO_ASSERT(inputs.find(output.first) != inputs.end(), "Can't find input data:" + output.first);
        output.second.data = inputs[output.first];
        std::cout << "    Pass " << output.first << " to output port" << std::endl;
    }
}

void ResultModule::print_static_config() {
    std::cout << R"(
  pipeline_result:          # Module Name
    type: "ResultModule"
    description: "Output result. Supported DataType: [OVTensor, VecOVTensor, String, VecString]"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
    )" << std::endl;
}

ResultModule::ResultModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    is_output_module = true;
}

void ResultModule::run(ov::AnyMap& outputs) {
    prepare_inputs();

    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << std::endl;
    for (auto& port_name : module_desc->inputs) {
        auto raw_data = this->inputs[port_name.source_module_out_name].data;
        outputs[port_name.source_module_out_name] = raw_data;
        std::cout << "    Get output data from input port: " << port_name.source_module_out_name << std::endl;
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
