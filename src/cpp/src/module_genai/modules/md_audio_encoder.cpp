// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_audio_encoder.hpp"
#include "module_genai/pipeline/module_factory.hpp"
#include "module_genai/utils/profiler.hpp"
#include "module_genai/utils/tensor_utils.hpp"

namespace ov::genai::module {

GENAI_REGISTER_MODULE_SAME(AudioEncoderModule);

void AudioEncoderModule::print_static_config() {
    std::cout << R"(
  audio_encoder:
    type: "AudioEncoderModule"
    description: "Encode raw audio to audio features."
    device: "GPU"
    inputs:
      - name: "input_features"
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "feature_attention_mask"
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "audio_features"
        type: "VecOVTensor"
      - name: "audio_feature_lengths"
        type: "OVTensor"
    params:
      model_path: "models_path"
    )" << std::endl;
}

AudioEncoderModule::AudioEncoderModule(const IBaseModuleDesc::PTR &desc, const PipelineDesc::PTR &pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize AudioEncoderModule");
    }
}

AudioEncoderModule::~AudioEncoderModule() {}

bool AudioEncoderModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("AudioEncoderModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
    std::string device = module_desc->device.empty() ? "GPU" : module_desc->device;
    if (model_path.extension() != ".xml") {
        model_path = model_path / "qwen3_omni_audio_encoder.xml";
    }
    if (!std::filesystem::exists(model_path)) {
        GENAI_ERR("AudioEncoderModule[" + module_desc->name + "]: Model file not found at: " + model_path.string());
        return false;
    }

    // Create and store the model in the request queue
    ov::Core core;
    auto model = core.read_model(model_path);
    auto compiled_model = core.compile_model(model, device);
    m_request_queue = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }
    );

    return true;
}

void AudioEncoderModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    if (!exists_input("input_features")) {
        OPENVINO_THROW("AudioEncoderModule[" + module_desc->name + "]: 'input_features' input not found");
    }
    if (!exists_input("feature_attention_mask")) {
        OPENVINO_THROW("AudioEncoderModule[" + module_desc->name + "]: 'feature_attention_mask' input not found");
    }
    auto input_features_vec = this->inputs["input_features"].data.as<std::vector<ov::Tensor>>();
    auto feature_attention_mask_vec = this->inputs["feature_attention_mask"].data.as<std::vector<ov::Tensor>>();

    ov::Tensor audio_feature_lengths_final(ov::element::i64, {feature_attention_mask_vec.size()});
    auto audio_feature_lengths_final_data = audio_feature_lengths_final.data<int64_t>();
    std::vector<ov::Tensor> audio_features_vec;
    for (size_t b = 0; b < input_features_vec.size(); b++) {
        auto input_features = tensor_utils::unsqueeze(input_features_vec[b], 0);
        auto feature_attention_mask = tensor_utils::unsqueeze(feature_attention_mask_vec[b], 0);
        ov::Tensor audio_feature_lengths(ov::element::i64, {feature_attention_mask.get_shape()[0]});
        ov::Tensor feature_attention_mask_i64(ov::element::i64, feature_attention_mask.get_shape());
        auto feature_attention_mask_i64_data = feature_attention_mask_i64.data<int64_t>();
        auto audio_feature_lengths_data = audio_feature_lengths.data<int64_t>();
        auto feature_attention_mask_data = feature_attention_mask.data<const int32_t>();
        for (size_t i = 0; i < feature_attention_mask.get_shape()[0]; ++i) {
            audio_feature_lengths_data[i] = 0;
            for (size_t j = 0; j < feature_attention_mask.get_shape()[1]; ++j) {
                if (feature_attention_mask_data[i * feature_attention_mask.get_shape()[1] + j]) {
                    ++audio_feature_lengths_data[i];
                }
                feature_attention_mask_i64_data[i * feature_attention_mask.get_shape()[1] + j] = static_cast<int64_t>(feature_attention_mask_data[i * feature_attention_mask.get_shape()[1] + j]);
            }
        }
        audio_feature_lengths_final_data[b] = audio_feature_lengths_data[0];

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_request_queue.get());
        ov::InferRequest& infer_request = infer_request_guard.get();
        infer_request.set_tensor("input_features", input_features);
        infer_request.set_tensor("feature_attention_mask", feature_attention_mask_i64);
        infer_request.set_tensor("audio_feature_lengths", audio_feature_lengths);
        {
            PROFILE(pm, "audio_encoder infer");
            infer_request.infer();
        }
        ov::Tensor audio_features_orig = infer_request.get_tensor("audio_features");

        if (audio_features_orig.get_shape().size() == 3) {
            size_t batch_size = audio_features_orig.get_shape()[0];
            size_t max_feature_length = audio_features_orig.get_shape()[1];
            size_t hidden_size = audio_features_orig.get_shape()[2];
            ov::Tensor audio_features(audio_features_orig.get_element_type(),
                                      {batch_size * max_feature_length, hidden_size});
            std::memcpy(audio_features.data(), audio_features_orig.data(),
                        audio_features_orig.get_byte_size());
            audio_features_vec.push_back(std::move(audio_features));
        } else {
            audio_features_vec.push_back(std::move(audio_features_orig));
        }
    
    }
    
    this->outputs["audio_features"].data = audio_features_vec;
    this->outputs["audio_feature_lengths"].data = audio_feature_lengths_final;
}


}