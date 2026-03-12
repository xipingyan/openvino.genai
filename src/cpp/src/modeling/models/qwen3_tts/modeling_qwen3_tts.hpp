// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Common Configuration Header
//
// This header contains shared configuration structures and KV cache types
// used across all Qwen3-TTS modules (Talker, CodePredictor, SpeechDecoder).
//
// For module-specific declarations, see:
//   - modeling_qwen3_tts_talker.hpp
//   - modeling_qwen3_tts_code_predictor.hpp
//   - modeling_qwen3_tts_speech_decoder.hpp
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

//===----------------------------------------------------------------------===//
// Qwen3 TTS Talker Config
//===----------------------------------------------------------------------===//
struct Qwen3TTSTalkerConfig {
    std::string architecture = "qwen3_tts_talker";
    int32_t hidden_size = 2048;
    int32_t num_attention_heads = 16;
    int32_t num_key_value_heads = 8;
    int32_t head_dim = 128;
    int32_t intermediate_size = 6144;
    int32_t num_hidden_layers = 28;
    int32_t vocab_size = 3072;           // codec vocab
    int32_t text_vocab_size = 151936;    // text vocab
    int32_t text_hidden_size = 896;      // text embedding hidden size
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;

    // mRoPE config
    bool mrope_interleaved = true;
    std::vector<int32_t> mrope_section = {24, 20, 20};  // [temporal, height, width]

    // Special tokens
    int32_t codec_eos_token_id = 2150;
    int32_t codec_bos_token_id = 2149;
    int32_t codec_pad_token_id = 2148;
};

//===----------------------------------------------------------------------===//
// Qwen3 TTS Code Predictor Config
//===----------------------------------------------------------------------===//
struct Qwen3TTSCodePredictorConfig {
    std::string architecture = "qwen3_tts_code_predictor";
    int32_t hidden_size = 1024;
    int32_t num_attention_heads = 16;
    int32_t num_key_value_heads = 8;
    int32_t head_dim = 128;              // (hidden_size * 2) / num_attention_heads
    int32_t intermediate_size = 3072;
    int32_t num_hidden_layers = 5;
    int32_t vocab_size = 2048;           // codec vocab for predictor
    int32_t num_code_groups = 16;        // 16 codec layers
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;

    // Projection size (from talker hidden to code predictor hidden)
    int32_t talker_hidden_size = 2048;   // for small_to_mtp_projection
};

//===----------------------------------------------------------------------===//
// Speech Decoder (12Hz) Config
//===----------------------------------------------------------------------===//
struct SpeechDecoderConfig {
    std::string architecture = "qwen3_tts_speech_decoder";

    // RVQ settings
    int32_t num_quantizers = 16;         // Number of RVQ layers
    int32_t codebook_size = 2048;        // Codebook size per layer
    int32_t codebook_dim = 32;           // Each codebook entry dimension

    // Pre-transformer
    int32_t latent_dim = 512;            // rvq_output_dim (input/output dim of pre_transformer)
    int32_t transformer_hidden = 1024;   // pre_transformer attention hidden size (num_heads * head_dim)
    int32_t transformer_heads = 8;       // pre_transformer attention heads
    int32_t transformer_head_dim = 128;  // head dimension (transformer_hidden / transformer_heads)
    int32_t transformer_layers = 8;      // pre_transformer layers
    int32_t transformer_intermediate = 2048;
    int32_t sliding_window = 72;
    float layer_scale_init = 0.1f;
    float rms_norm_eps = 1e-6f;

    // Decoder
    int32_t decoder_dim = 1536;          // initial decoder channel
    std::vector<int32_t> decoder_channel_mults = {1, 1, 1, 1};  // 4 stages
    std::vector<int32_t> decoder_dilations = {1, 3, 9};  // per-stage dilations

    // Upsampling ratios
    std::vector<int32_t> pre_upsample_ratios = {2, 2};  // pre-decoder upsample
    std::vector<int32_t> decoder_upsample_rates = {8, 5, 4, 3};  // decoder upsample

    // Output
    int32_t sample_rate = 24000;
    float rope_theta = 10000.0f;
};

//===----------------------------------------------------------------------===//
// KV Cache Output Structures
//===----------------------------------------------------------------------===//

// KV Cache output structure for attention
struct AttentionKVOutput {
    Tensor hidden_states;
    Tensor key_cache;    // [batch, num_kv_heads, seq_len, head_dim]
    Tensor value_cache;  // [batch, num_kv_heads, seq_len, head_dim]
};

// KV Cache output structure for decoder layer
struct DecoderLayerKVOutput {
    Tensor hidden_states;
    Tensor residual;
    Tensor key_cache;
    Tensor value_cache;
};

// KV Cache output structure for model
struct TalkerModelKVOutput {
    Tensor hidden_states;
    Tensor pre_norm_hidden;
    std::vector<Tensor> key_caches;    // [num_layers], each [batch, kv_heads, seq, head_dim]
    std::vector<Tensor> value_caches;  // [num_layers], each [batch, kv_heads, seq, head_dim]
};

// KV Cache output structure for Talker generation
struct TalkerGenerationKVOutput {
    Tensor logits;
    Tensor hidden_states;
    std::vector<Tensor> key_caches;
    std::vector<Tensor> value_caches;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
