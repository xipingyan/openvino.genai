// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Speech Decoder Module
//
// Implements the 12Hz speech decoder that converts RVQ codec tokens to audio:
//   1. RVQ Dequantizer: codes[16 layers] -> continuous embeddings
//   2. Pre-Conv: channel expansion (512 -> 1024)
//   3. Pre-Transformer: 8-layer transformer with sliding window attention
//   4. Pre-Decoder Upsample: 2x2 upsampling with ConvNeXt blocks
//   5. Decoder: 4 blocks with transposed conv upsampling (8x5x4x3 = 480x)
//   6. Output: SnakeBeta activation + conv to audio
//
// Total upsample factor: 4 (pre) * 480 (decoder) = 1920
// Input: [batch, 16, seq_len] (codec tokens)
// Output: [batch, seq_len * 1920] (audio samples at 24kHz)
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "modeling/builder_context.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/models/qwen3_tts/modeling_qwen3_tts.hpp"
#include "modeling/module.hpp"

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// RMSNorm is in ov::genai::modeling, access it via parent namespace
using modeling::RMSNorm;

//===----------------------------------------------------------------------===//
// RVQ Dequantizer
// Converts discrete RVQ codes into continuous embeddings
//===----------------------------------------------------------------------===//
class RVQDequantizer : public Module {
public:
    RVQDequantizer(BuilderContext& ctx, const std::string& name,
                   const SpeechDecoderConfig& cfg, Module* parent = nullptr);

    // Forward: codes[batch, 16, seq] -> embeddings[batch, seq, rvq_output_dim]
    Tensor forward(const Tensor& codes) const;

    // Weight accessors
    const Tensor& first_codebook() const;
    const Tensor& first_output_proj() const;
    const Tensor& rest_codebook(int idx) const;
    const Tensor& rest_output_proj() const;

private:
    SpeechDecoderConfig cfg_;
    WeightParameter* first_codebook_param_ = nullptr;
    WeightParameter* first_output_proj_param_ = nullptr;
    std::vector<WeightParameter*> rest_codebook_params_;
    WeightParameter* rest_output_proj_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Pre-Transformer Attention
// Standard attention with RoPE and optional sliding window
//===----------------------------------------------------------------------===//
class PreTransformerAttention : public Module {
public:
    PreTransformerAttention(BuilderContext& ctx, const std::string& name,
                            const SpeechDecoderConfig& cfg, Module* parent = nullptr);

    Tensor forward_no_cache(const Tensor& hidden_states,
                            const Tensor& rope_cos,
                            const Tensor& rope_sin,
                            const Tensor& causal_mask) const;

    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

private:
    int32_t num_heads_;
    int32_t head_dim_;
    int32_t hidden_size_;
    float scaling_;
    int32_t sliding_window_;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    RMSNorm q_norm_;
    RMSNorm k_norm_;
};

//===----------------------------------------------------------------------===//
// Pre-Transformer MLP
// SwiGLU MLP used in the pre-transformer
//===----------------------------------------------------------------------===//
class PreTransformerMLP : public Module {
public:
    PreTransformerMLP(BuilderContext& ctx, const std::string& name,
                      const SpeechDecoderConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

private:
    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Pre-Transformer Decoder Layer
// Single transformer layer with LayerScale
//===----------------------------------------------------------------------===//
class PreTransformerDecoderLayer : public Module {
public:
    PreTransformerDecoderLayer(BuilderContext& ctx, const std::string& name,
                               const SpeechDecoderConfig& cfg, Module* parent = nullptr);

    // Returns (hidden_states, residual)
    std::pair<Tensor, Tensor> forward_no_cache(const Tensor& hidden_states,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin,
                                               const Tensor& causal_mask,
                                               const std::optional<Tensor>& residual = std::nullopt) const;

    const Tensor& attn_layer_scale() const;
    const Tensor& mlp_layer_scale() const;

private:
    PreTransformerAttention self_attn_;
    PreTransformerMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
    WeightParameter* attn_layer_scale_param_ = nullptr;
    WeightParameter* mlp_layer_scale_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Pre-Transformer
// 8-layer transformer for processing RVQ embeddings
//===----------------------------------------------------------------------===//
class PreTransformer : public Module {
public:
    PreTransformer(BuilderContext& ctx, const std::string& name,
                   const SpeechDecoderConfig& cfg, Module* parent = nullptr);

    // Forward: x[batch, seq, latent_dim] -> output[batch, seq, latent_dim]
    Tensor forward(const Tensor& x) const;

    const Tensor& input_proj_weight() const;
    const Tensor& input_proj_bias() const;
    const Tensor& output_proj_weight() const;
    const Tensor& output_proj_bias() const;

private:
    SpeechDecoderConfig cfg_;
    std::vector<PreTransformerDecoderLayer> layers_;
    RMSNorm norm_;
    WeightParameter* input_proj_weight_param_ = nullptr;
    WeightParameter* input_proj_bias_param_ = nullptr;
    WeightParameter* output_proj_weight_param_ = nullptr;
    WeightParameter* output_proj_bias_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// SnakeBeta Activation
// Custom activation: x + 1/beta * sin^2(x * alpha)
//===----------------------------------------------------------------------===//
class SnakeBetaActivation : public Module {
public:
    SnakeBetaActivation(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

    const Tensor& alpha() const;
    const Tensor& beta() const;

private:
    WeightParameter* alpha_param_ = nullptr;
    WeightParameter* beta_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// ConvNeXt Block
// Used in pre-decoder upsampling
//===----------------------------------------------------------------------===//
class ConvNeXtBlock : public Module {
public:
    ConvNeXtBlock(BuilderContext& ctx, const std::string& name,
                  int32_t channels, Module* parent = nullptr);

    // Forward: x[batch, channels, seq] -> output[batch, channels, seq]
    Tensor forward(const Tensor& x) const;

private:
    int32_t channels_;
    WeightParameter* dwconv_weight_param_ = nullptr;
    WeightParameter* dwconv_bias_param_ = nullptr;
    WeightParameter* norm_weight_param_ = nullptr;
    WeightParameter* norm_bias_param_ = nullptr;
    WeightParameter* pwconv1_weight_param_ = nullptr;
    WeightParameter* pwconv1_bias_param_ = nullptr;
    WeightParameter* pwconv2_weight_param_ = nullptr;
    WeightParameter* pwconv2_bias_param_ = nullptr;
    WeightParameter* gamma_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Residual Unit
// Dilated causal conv with SnakeBeta activation
//===----------------------------------------------------------------------===//
class ResidualUnit : public Module {
public:
    ResidualUnit(BuilderContext& ctx, const std::string& name,
                 int32_t channels, int32_t dilation, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    int32_t channels_;
    int32_t dilation_;
    SnakeBetaActivation act1_;
    SnakeBetaActivation act2_;
    WeightParameter* conv1_weight_param_ = nullptr;
    WeightParameter* conv1_bias_param_ = nullptr;
    WeightParameter* conv2_weight_param_ = nullptr;
    WeightParameter* conv2_bias_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Decoder Block
// Transposed conv upsample + residual units
//===----------------------------------------------------------------------===//
class DecoderBlock : public Module {
public:
    DecoderBlock(BuilderContext& ctx, const std::string& name,
                 int32_t in_channels, int32_t out_channels, int32_t upsample_rate,
                 Module* parent = nullptr);

    // Forward: x[batch, in_ch, seq] -> output[batch, out_ch, seq * upsample_rate]
    Tensor forward(const Tensor& x) const;

private:
    int32_t in_channels_;
    int32_t out_channels_;
    int32_t upsample_rate_;
    SnakeBetaActivation snake_;
    WeightParameter* upsample_conv_weight_param_ = nullptr;
    WeightParameter* upsample_conv_bias_param_ = nullptr;
    std::vector<ResidualUnit> residual_units_;
};

//===----------------------------------------------------------------------===//
// Speech Decoder Model
// Complete decoder: codes -> audio
//===----------------------------------------------------------------------===//
class SpeechDecoderModel : public Module {
public:
    SpeechDecoderModel(BuilderContext& ctx, const SpeechDecoderConfig& cfg,
                       Module* parent = nullptr);

    // Forward: codes[batch, 16, seq_len] -> audio[batch, audio_len]
    Tensor forward(const Tensor& codes) const;

    // Calculate output audio length from input code length
    int64_t get_audio_length(int64_t code_length) const;

private:
    SpeechDecoderConfig cfg_;
    RVQDequantizer dequantizer_;
    PreTransformer pre_transformer_;

    // Pre-conv
    WeightParameter* pre_conv_weight_param_ = nullptr;
    WeightParameter* pre_conv_bias_param_ = nullptr;

    // Pre-decoder upsample
    std::vector<WeightParameter*> upsample_conv_weight_params_;
    std::vector<WeightParameter*> upsample_conv_bias_params_;
    std::vector<ConvNeXtBlock> upsample_blocks_;

    // Initial decoder conv
    WeightParameter* decoder_init_conv_weight_param_ = nullptr;
    WeightParameter* decoder_init_conv_bias_param_ = nullptr;

    // Decoder blocks
    std::vector<DecoderBlock> decoder_blocks_;

    // Final output
    SnakeBetaActivation final_snake_;
    WeightParameter* output_conv_weight_param_ = nullptr;
    WeightParameter* output_conv_bias_param_ = nullptr;

    int64_t total_upsample_;
};

//===----------------------------------------------------------------------===//
// Factory Function
// Creates a complete speech decoder OpenVINO model
//===----------------------------------------------------------------------===//
std::shared_ptr<ov::Model> create_qwen3_tts_speech_decoder_model(
    const SpeechDecoderConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
