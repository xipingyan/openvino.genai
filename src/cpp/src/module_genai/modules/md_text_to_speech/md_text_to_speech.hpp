// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/pipeline/module.hpp"
#include "module_genai/pipeline/pipeline_impl.hpp"

#include <memory>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "openvino/genai/tokenizer.hpp"

namespace ov::genai::module {

class TextToSpeechModule : public IBaseModule {
protected:
    TextToSpeechModule() = delete;
    TextToSpeechModule(const IBaseModuleDesc::PTR& desc,
                       const PipelineDesc::PTR& pipeline_desc,
                       const VLMModelType& model_type);

public:
    virtual ~TextToSpeechModule();

    using PTR = std::shared_ptr<TextToSpeechModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc);
    static void print_static_config();
    void run() override;

protected:
    static const ModuleSpec& get_spec();
    std::vector<std::string> parse_input_texts();

    std::optional<std::filesystem::path> get_model_path(const std::string &param_name);
    std::shared_ptr<ov::Model> load_model(const std::filesystem::path &model_path,
                                          const std::string &default_model_filename = "");

    std::vector<int64_t> make_mrope_positions(size_t start, size_t len, size_t batch);
    ov::Tensor make_causal_mask(size_t seq_len, size_t batch);
    int64_t sample_codec_token(const float* logits, size_t vocab_size,
                               float temperature, size_t top_k, float top_p,
                               float rep_penalty,
                               const std::vector<int64_t>* history,
                               const std::vector<int64_t>* suppress_tokens,
                               std::mt19937& rng);
    // Greedy decoding variant of sample_codec_token used when m_sample_codec_token_greedy_search is enabled.
    int64_t sample_codec_token_greedy(const float* logits, size_t vocab_size);

    ov::Tensor make_decode_mask(size_t past_len, size_t batch);
    std::pair<ov::Tensor, int> synthesize_fallback_tone(const std::string& text);

protected:
    VLMModelType m_model_type;
    std::string m_device = "CPU";

    std::unique_ptr<ov::InferRequest> m_embedding_infer;
    std::unique_ptr<ov::InferRequest> m_prefill_infer;
    std::unique_ptr<ov::InferRequest> m_decode_infer;
    std::unique_ptr<ov::InferRequest> m_codec_embedding_infer;
    std::vector<std::unique_ptr<ov::InferRequest>> m_code_predictor_ar_infers;
    std::vector<std::unique_ptr<ov::InferRequest>> m_code_predictor_single_codec_embed_infers;
    std::unique_ptr<ov::InferRequest> m_code_predictor_single_codec_embedding_infer;
    std::unique_ptr<ov::InferRequest> m_speech_decoder_infer;
    std::unique_ptr<Tokenizer> m_tokenizer;

    bool m_sample_codec_token_greedy_search =
        false;  // Enable greedy decoding in sample_codec_token when this flag is set.
                // Useful for fast debugging and deterministic behavior.
    bool m_merge_ar_and_sce_ov_models = false;  // Merge AR and SCE models into one OV model for better performance.
                                                // Requires "sample_codec_token_greedy_search=true".
    bool m_force_ar_model_inference_precision_f32 =
        false;  // Force AR model inference precision to f32, which can improve the performance of AR model inference
                // and also is required for some models (e.g. Qwen3-Omni) to get correct results when
                // "merge_ar_and_sce_ov_models" is enabled.
};

}
