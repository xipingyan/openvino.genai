// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "module_genai/modules/md_text_to_speech/md_text_to_speech.hpp"

#if defined(ENABLE_MODELING_PRIVATE)

#    include "modeling_private/models/qwen3_omni/processing_qwen3_omni.hpp"

namespace ov::genai::module {

class TextToSpeechImpl_Qwen3Omni : public TextToSpeechModule {
public:
    TextToSpeechImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                               const PipelineDesc::PTR& pipeline_desc,
                               const VLMModelType& model_type);

    void run() override;

private:
    bool initialize();

    ov::Tensor text_embedding(ov::Tensor text_input_ids, ov::Tensor codec_input_ids, ov::Tensor codec_mask);

    std::vector<float> m_tts_pad_embed;
    void calc_tts_pad_embed();

    // Predictor AR infer.
    std::vector<int64_t> code_predictor_ar_infers(int cp_steps,
                                                  std::vector<float>& autoregressive_sequence,
                                                  size_t batch,
                                                  size_t hidden_size,
                                                  size_t cp_vocab_size,
                                                  float temperature,
                                                  size_t top_k,
                                                  float top_p,
                                                  std::mt19937& rng,
                                                  std::vector<std::vector<int64_t>>& all_layer_tokens,
                                                  int num_layers_total);

    std::pair<ov::Tensor, int> qwen3_omni_text_to_speech(const std::string& text);

private:
    modeling::models::Qwen3OmniProcessingConfig m_config;

    modeling::models::Qwen3TTSTalkerConfig m_talker_cfg;
    modeling::models::Qwen3TTSCodePredictorConfig m_cp_cfg;
    int m_cp_steps = 15;

    bool m_merge_ov_models = false;
    std::unique_ptr<ov::InferRequest> m_merged_infer_request = nullptr;  // Only used when m_merge_ov_models is true

    void load_code_predictor_models(const ov::AnyMap& tts_props);
    void merge_code_predictor_ov_models(std::vector<std::shared_ptr<ov::Model>>& ar_models,
                                        std::vector<std::shared_ptr<ov::Model>>& sce_models);
    std::vector<int64_t> code_predictor_ar_infers_merged_ov(int cp_steps,
                                                            std::vector<float>& autoregressive_sequence,
                                                            size_t batch,
                                                            size_t hidden_size,
                                                            size_t cp_vocab_size,
                                                            float temperature,
                                                            size_t top_k,
                                                            float top_p,
                                                            std::mt19937& rng,
                                                            std::vector<std::vector<int64_t>>& all_layer_tokens,
                                                            int num_layers_total);
};

}  // namespace ov::genai::module

#else

namespace ov::genai::module {

class TextToSpeechImpl_Qwen3Omni : public TextToSpeechModule {
public:
    TextToSpeechImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                               const PipelineDesc::PTR& pipeline_desc,
                               const VLMModelType& model_type)
        : TextToSpeechModule(desc, pipeline_desc, model_type) {
        OPENVINO_THROW("TextToSpeechImpl_Qwen3Omni is not implemented in open source build");
    }

    void run() override {
        OPENVINO_THROW("TextToSpeechImpl_Qwen3Omni is not implemented in open source build");
    }
};

}  // namespace ov::genai::module

#endif  // ENABLE_MODELING_PRIVATE
