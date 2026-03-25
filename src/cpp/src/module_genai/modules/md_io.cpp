// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_io.hpp"

#include "module_genai/pipeline/module_factory.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(ParameterModule);
GENAI_REGISTER_MODULE_SAME(ResultModule);

const ModuleSpec& ParameterModule::get_spec() {
  static const ModuleSpec spec = []() {
    ModuleSpec s("ParameterModule", "image");
    const std::initializer_list<DataType> pass_through_types = {
      DataType::OVTensor,
      DataType::VecOVTensor,
      DataType::String,
      DataType::VecString,
    };
    s.add_output("image1_data", pass_through_types);
    s.add_output("image2_data", pass_through_types);
    return s;
  }();
  return spec;
}

const ModuleSpec& ResultModule::get_spec() {
  static const ModuleSpec spec = []() {
    ModuleSpec s("ResultModule", "pipeline_result");
    const std::initializer_list<DataType> pass_through_types = {
      DataType::OVTensor,
      DataType::VecOVTensor,
      DataType::String,
      DataType::VecString,
    };
    s.add_input("raw_data", pass_through_types);
    return s;
  }();
  return spec;
}

void ParameterModule::print_static_config() {
    std::cout << get_spec().to_yaml_template_string() << std::endl;
}

ParameterModule::ParameterModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
  check_params_with_spec(get_spec());
    is_input_module = true;
    // std::cout << "ParameterModule:" << m_desc << std::endl;
}
ParameterModule::~ParameterModule() {}

void ParameterModule::run(ov::AnyMap& inputs) {
    GENAI_INFO("Running module: " + module_desc->name);

    for (auto& output : this->outputs) {
        OPENVINO_ASSERT(inputs.find(output.first) != inputs.end(), "Can't find input data:" + output.first);
        output.second.data = inputs[output.first];
        GENAI_INFO("    Pass " + output.first + " to output port");
    }
}

void ParameterModule::run() {
    OPENVINO_ASSERT(false, "ParameterModule::run() should not be called");
}

void ResultModule::print_static_config() {
    std::cout << get_spec().to_yaml_template_string() << std::endl;
}

ResultModule::ResultModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
  check_params_with_spec(get_spec());
    is_output_module = true;
}

ResultModule::~ResultModule() {}

void ResultModule::run(ov::AnyMap& outputs) {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    
    for (auto& port_name : module_desc->inputs) {
        auto raw_data = this->inputs[port_name.name].data;
        GENAI_INFO("    pass '" + port_name.source_module_out_name + "' to pipeline outputs as '" + port_name.name + "'");
        outputs[port_name.name] = raw_data;
    }
}

void ResultModule::run() {
    OPENVINO_ASSERT(false, "ResultModule::run() should not be called");
}

}  // namespace module
}  // namespace genai
}  // namespace ov
