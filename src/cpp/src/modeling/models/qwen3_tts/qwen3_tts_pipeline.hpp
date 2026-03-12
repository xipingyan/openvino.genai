// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS End-to-End Pipeline
//
// This file provides the complete TTS pipeline implementation:
//   Text -> Tokenizer -> Talker (prefill+decode) -> Code Predictor -> Speech Decoder -> Audio
//
// Usage:
//   Qwen3TTSPipeline pipeline(model_path, device);
//   auto audio = pipeline.generate("Hello, world!");
//===----------------------------------------------------------------------===//

#pragma once

#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

/**
 * @brief Qwen3-TTS Generation Configuration
 */
struct Qwen3TTSGenerationConfig {
    float temperature = 0.9f;
    size_t top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    int min_new_tokens = 2;
    int max_new_tokens = 2048;
    uint32_t seed = 42;
    
    // Temporary workaround: force stop after this many frames (0 = disabled, use EOS detection)
    // Use this when EOS is not being properly sampled
    int force_stop_after_frames = 0;
};

/**
 * @brief Qwen3-TTS Generation Result
 */
struct Qwen3TTSResult {
    ov::Tensor audio;           // [1, audio_samples] float32
    int64_t num_frames;         // Number of codec frames generated
    float generation_time_ms;   // Codec generation time
    float decode_time_ms;       // Speech decoder time
    int sample_rate;            // Audio sample rate (24000)
};

/**
 * @brief Qwen3-TTS End-to-End Pipeline
 * 
 * Pipeline:
 *   Text -> Tokenizer -> Talker (prefill+decode) -> Code Predictor -> Speech Decoder -> Audio
 * 
 * Components:
 *   - Talker: 28-layer decoder with mRoPE and KV cache
 *   - Code Predictor: 15 AR models for codec layers 1-15
 *   - Speech Decoder: RVQ dequantizer + PreTransformer + ConvNeXt decoder
 */
class Qwen3TTSPipeline {
public:
    /**
     * @brief Construct pipeline from model directory
     * @param models_path Path to Qwen3-TTS model directory
     * @param device Target device (CPU, GPU, NPU)
     * @param properties Optional device properties
     */
    Qwen3TTSPipeline(const std::filesystem::path& models_path,
                     const std::string& device,
                     const ov::AnyMap& properties = {});

    /**
     * @brief Generate audio from text
     * @param text Input text to synthesize
     * @param config Generation configuration
     * @return Generation result with audio tensor
     */
    Qwen3TTSResult generate(const std::string& text,
                            const Qwen3TTSGenerationConfig& config = {});

    /**
     * @brief Get audio sample rate
     */
    int get_sample_rate() const { return m_sample_rate; }

private:
    void init_model_config(const std::filesystem::path& root_dir);
    void load_models(const std::filesystem::path& models_path, 
                     const std::string& device, 
                     const ov::AnyMap& properties);
    
    // Talker generation
    std::vector<std::vector<int64_t>> generate_codec_tokens(
        const std::vector<int64_t>& text_token_ids,
        const Qwen3TTSGenerationConfig& config);
    
    // Speech decoder
    ov::Tensor decode_to_audio(const std::vector<std::vector<int64_t>>& all_layer_tokens);
    
    // Helper functions
    ov::Tensor create_causal_mask(size_t seq_len, size_t batch_size = 1);
    ov::Tensor create_decode_mask(size_t past_len, size_t batch_size = 1);
    std::vector<int64_t> create_mrope_positions(size_t start, size_t len, size_t batch = 1);
    int64_t sample_token(const float* logits, size_t vocab_size, 
                         float temperature, size_t top_k, float top_p,
                         float rep_penalty,
                         const std::vector<int64_t>* history = nullptr,
                         const std::vector<int64_t>* suppress_tokens = nullptr);

private:
    // Tokenizer
    std::unique_ptr<Tokenizer> m_tokenizer;
    
    // Talker configuration
    int32_t m_hidden_size = 2048;
    int32_t m_num_layers = 28;
    int32_t m_num_kv_heads = 8;
    int32_t m_head_dim = 128;
    int32_t m_vocab_size = 3072;
    
    // Special tokens
    int64_t m_tts_bos_token_id = 151672;
    int64_t m_tts_eos_token_id = 151673;
    int64_t m_tts_pad_token_id = 151671;
    int64_t m_codec_bos_id = 2149;
    int64_t m_codec_eos_id = 2150;
    int64_t m_codec_pad_id = 2148;
    int64_t m_codec_nothink_id = 2155;
    int64_t m_codec_think_bos_id = 2156;
    int64_t m_codec_think_eos_id = 2157;
    std::vector<int64_t> m_role_tokens = {151644, 77091, 198};  // <|im_start|>assistant\n
    
    // Code Predictor configuration
    int32_t m_cp_hidden_size = 1024;
    int32_t m_cp_vocab_size = 2048;
    
    // Speech Decoder configuration
    int32_t m_sample_rate = 24000;
    
    // Random generator for sampling
    std::mt19937 m_rng;
    
    // Inference requests
    ov::InferRequest m_embed_infer;
    ov::InferRequest m_talker_prefill_infer;
    ov::InferRequest m_talker_decode_infer;
    ov::InferRequest m_talker_codec_infer;
    std::vector<ov::InferRequest> m_cp_ar_infer;      // 15 AR models
    std::vector<ov::InferRequest> m_cp_embed_infer;   // 15 single codec embedding models
    ov::InferRequest m_cp_codec_infer;                // Combined codec embedding
    ov::InferRequest m_decoder_infer;
    
    // Cached embeddings
    std::vector<float> m_tts_pad_embed;  // Pre-computed tts_pad embedding for decode phase
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
