// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_audio_preprocess.hpp"

#include <chrono>
#include <thread>

#include "module_genai/module_factory.hpp"
#include "module_genai/utils/tensor_utils.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(AudioPreprocessModule);

void AudioPreprocessModule::print_static_config() {
    std::cout << R"(
  audio_preprocessor:           # Module Name
    type: "AudioPreprocessModule"
    device: "CPU"               # Optional, default to CPU
    description: "Audio preprocessing."
    inputs:
      - name: "audio"           # [optional]
        type: "OVTensor"        # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "audios"          # [Optional] multiple audios
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "input_features"  # Output port name.
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "feature_attention_mask"  # Output port name
        type: "OVTensor"                # Support DataType: [OVTensor]     
    params:
      model_path: "models_path"
    )" << std::endl;
}

AudioPreprocessModule::AudioPreprocessModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    std::string model_path = desc->get_full_path(desc->params["model_path"]);
    std::string device = desc->device;
    if (device.empty()) {
        device = "CPU";
    }

    _model_type = to_vlm_model_type(desc->model_type);
}

AudioPreprocessModule::~AudioPreprocessModule() {}

void AudioPreprocessModule::preprocess_audio(const bool& has_audios_input) {

}

void AudioPreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    bool has_audios_input = exists_input("audios");
    bool has_audio_input = exists_input("audio");
    bool has_audio = has_audios_input || has_audio_input;
    if (has_audio) {
        OPENVINO_ASSERT(
            !(has_audios_input && has_audio_input),
            "AudioPreprocessModule: Both 'audio' and 'audios' inputs exist. Please provide only one of them.");
    }

    if (has_audios_input || has_audio_input) {
        preprocess_audio(has_audios_input);
    } else {
        OPENVINO_THROW("AudioPreprocessModule[" + module_desc->name +
                       "]: No valid input found. Please provide one of the following inputs: 'audio', 'audios'.");
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
