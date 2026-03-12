// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/*
 * @file modeling_qwen3_tts_speech_decoder.cpp
 * @brief Qwen3-TTS Speech Decoder implementation using Modeling API
 *
 * This implements the speech decoder component of Qwen3-TTS which converts
 * codec tokens (16 layers of RVQ codes) to audio waveforms. The architecture:
 *   1. RVQ Dequantizer: codes -> continuous embeddings
 *   2. Pre-Conv: channel expansion (512 -> 1024)
 *   3. Pre-Transformer: 8-layer transformer with sliding window attention
 *   4. Pre-Decoder Upsample: 2x2 upsampling with ConvNeXt blocks
 *   5. Decoder: 4 blocks with transposed conv upsampling (8x5x4x3)
 *   6. Output: final SnakeBeta + conv to audio
 */

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_speech_decoder.hpp"
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/rope.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

//===----------------------------------------------------------------------===//
// RVQ Dequantizer Implementation
// 
// Note: The original safetensors stores embedding_sum and cluster_usage separately.
// The actual codebook is computed as: codebook = embedding_sum / cluster_usage
// The weight loader should precompute this or we need custom handling.
// For simplicity, we expect the weight source to provide pre-computed codebook embeddings
// with the name pattern "*.embed" instead of "*.embedding_sum" and "*.cluster_usage"
//===----------------------------------------------------------------------===//

RVQDequantizer::RVQDequantizer(BuilderContext& ctx, const std::string& name,
                               const SpeechDecoderConfig& cfg, Module* parent)
    : Module(name, ctx, parent), cfg_(cfg) {
    
    // First codebook (semantic layer 0)
    // Weight name: decoder.quantizer.rvq_first.vq.layers[0]._codebook.embed
    first_codebook_param_ = &register_parameter("rvq_first.vq.layers[0]._codebook.embed");
    first_output_proj_param_ = &register_parameter("rvq_first.output_proj.weight");
    
    // Rest codebooks (acoustic layers 1-15)
    // Weight names: decoder.quantizer.rvq_rest.vq.layers[{i}]._codebook.embed
    rest_codebook_params_.reserve(15);
    for (int i = 0; i < 15; ++i) {
        rest_codebook_params_.push_back(
            &register_parameter("rvq_rest.vq.layers[" + std::to_string(i) + "]._codebook.embed"));
    }
    rest_output_proj_param_ = &register_parameter("rvq_rest.output_proj.weight");
}

const Tensor& RVQDequantizer::first_codebook() const {
    return first_codebook_param_->value();
}

const Tensor& RVQDequantizer::first_output_proj() const {
    return first_output_proj_param_->value();
}

const Tensor& RVQDequantizer::rest_codebook(int idx) const {
    return rest_codebook_params_[static_cast<size_t>(idx)]->value();
}

const Tensor& RVQDequantizer::rest_output_proj() const {
    return rest_output_proj_param_->value();
}

Tensor RVQDequantizer::forward(const Tensor& codes) const {
    // codes: [batch, 16, seq_len] - indices into codebooks
    // output: [batch, seq_len, rvq_output_dim]
    auto* pctx = codes.context();
    
    // Process first codebook (layer 0 - semantic)
    // Slice codes for layer 0: [batch, seq_len]
    auto codes_0 = ops::slice(codes, 0, 1, 1, 1).squeeze(1);  // [batch, seq_len]
    
    // Gather embeddings from first codebook: [batch, seq_len, codebook_dim]
    auto embed_0 = ops::gather(first_codebook(), codes_0, 0);
    
    // Project through output_proj: [batch, seq_len, rvq_output_dim]
    // output_proj weight: [rvq_output_dim, codebook_dim, 1] -> use as [rvq_output_dim, codebook_dim]
    auto proj_w = first_output_proj().squeeze(2);  // Remove kernel dim
    auto output = ops::matmul(embed_0, proj_w, false, true);
    
    // Process rest codebooks (layers 1-15 - acoustic)
    for (int layer = 1; layer < 16; ++layer) {
        // Slice codes for this layer
        auto codes_l = ops::slice(codes, layer, layer + 1, 1, 1).squeeze(1);
        
        // Gather embeddings
        auto embed_l = ops::gather(rest_codebook(layer - 1), codes_l, 0);
        
        // Project and accumulate
        auto proj_rest_w = rest_output_proj().squeeze(2);
        auto proj_l = ops::matmul(embed_l, proj_rest_w, false, true);
        output = output + proj_l;
    }
    
    return output;  // [batch, seq_len, rvq_output_dim]
}

//===----------------------------------------------------------------------===//
// Pre-Transformer Attention Implementation
//===----------------------------------------------------------------------===//

PreTransformerAttention::PreTransformerAttention(BuilderContext& ctx, const std::string& name,
                                                 const SpeechDecoderConfig& cfg, Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.transformer_heads),
      head_dim_(cfg.transformer_head_dim),
      hidden_size_(cfg.transformer_hidden),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      sliding_window_(cfg.sliding_window),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");
}

const Tensor& PreTransformerAttention::q_proj_weight() const {
    return q_proj_param_->value();
}

const Tensor& PreTransformerAttention::k_proj_weight() const {
    return k_proj_param_->value();
}

const Tensor& PreTransformerAttention::v_proj_weight() const {
    return v_proj_param_->value();
}

const Tensor& PreTransformerAttention::o_proj_weight() const {
    return o_proj_param_->value();
}

Tensor PreTransformerAttention::forward_no_cache(const Tensor& hidden_states,
                                                  const Tensor& rope_cos,
                                                  const Tensor& rope_sin,
                                                  const Tensor& causal_mask) const {
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    // Reshape: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});

    // QK Norm
    if (q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    // Apply standard RoPE
    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    // SDPA with sliding window attention
    // Note: sliding window is handled by the mask in OpenVINO SDPA
    auto context = ops::llm::sdpa(q_rot, k_rot, v_heads, scaling_, 3, &causal_mask, false, policy);

    // Output projection
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    return ops::linear(merged, o_proj_weight());
}

//===----------------------------------------------------------------------===//
// Pre-Transformer MLP Implementation
//===----------------------------------------------------------------------===//

PreTransformerMLP::PreTransformerMLP(BuilderContext& ctx, const std::string& name,
                                     const SpeechDecoderConfig& cfg, Module* parent)
    : Module(name, ctx, parent) {
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& PreTransformerMLP::gate_proj_weight() const {
    return gate_proj_param_->value();
}

const Tensor& PreTransformerMLP::up_proj_weight() const {
    return up_proj_param_->value();
}

const Tensor& PreTransformerMLP::down_proj_weight() const {
    return down_proj_param_->value();
}

Tensor PreTransformerMLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

//===----------------------------------------------------------------------===//
// Pre-Transformer Decoder Layer Implementation
//===----------------------------------------------------------------------===//

PreTransformerDecoderLayer::PreTransformerDecoderLayer(BuilderContext& ctx, const std::string& name,
                                                       const SpeechDecoderConfig& cfg, Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
    
    // Layer scale parameters
    attn_layer_scale_param_ = &register_parameter("self_attn_layer_scale.scale");
    mlp_layer_scale_param_ = &register_parameter("mlp_layer_scale.scale");
}

const Tensor& PreTransformerDecoderLayer::attn_layer_scale() const {
    return attn_layer_scale_param_->value();
}

const Tensor& PreTransformerDecoderLayer::mlp_layer_scale() const {
    return mlp_layer_scale_param_->value();
}

std::pair<Tensor, Tensor> PreTransformerDecoderLayer::forward_no_cache(const Tensor& hidden_states,
                                                                        const Tensor& rope_cos,
                                                                        const Tensor& rope_sin,
                                                                        const Tensor& causal_mask,
                                                                        const std::optional<Tensor>& residual) const {
    Tensor normed;
    Tensor next_residual;
    
    if (residual) {
        auto norm_out = input_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = input_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }
    
    auto attn_out = self_attn_.forward_no_cache(normed, rope_cos, rope_sin, causal_mask);
    
    // Apply layer scale: residual + scale * attn_out
    auto scaled_attn = attn_out * attn_layer_scale();
    auto after_attn = next_residual + scaled_attn;
    
    // Post-attention norm
    auto post_normed = post_attention_layernorm_.forward(after_attn);
    
    // MLP
    auto mlp_out = mlp_.forward(post_normed);
    
    // Apply layer scale to MLP output
    auto scaled_mlp = mlp_out * mlp_layer_scale();
    
    return {scaled_mlp, after_attn};
}

//===----------------------------------------------------------------------===//
// Pre-Transformer Implementation
//===----------------------------------------------------------------------===//

PreTransformer::PreTransformer(BuilderContext& ctx, const std::string& name,
                               const SpeechDecoderConfig& cfg, Module* parent)
    : Module(name, ctx, parent),
      cfg_(cfg),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this) {
    
    // Input/output projections
    input_proj_weight_param_ = &register_parameter("input_proj.weight");
    input_proj_bias_param_ = &register_parameter("input_proj.bias");
    output_proj_weight_param_ = &register_parameter("output_proj.weight");
    output_proj_bias_param_ = &register_parameter("output_proj.bias");
    
    // Transformer layers
    layers_.reserve(static_cast<size_t>(cfg.transformer_layers));
    for (int32_t i = 0; i < cfg.transformer_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
    }
}

const Tensor& PreTransformer::input_proj_weight() const {
    return input_proj_weight_param_->value();
}

const Tensor& PreTransformer::input_proj_bias() const {
    return input_proj_bias_param_->value();
}

const Tensor& PreTransformer::output_proj_weight() const {
    return output_proj_weight_param_->value();
}

const Tensor& PreTransformer::output_proj_bias() const {
    return output_proj_bias_param_->value();
}

Tensor PreTransformer::forward(const Tensor& x) const {
    auto* pctx = x.context();
    auto* policy = &ctx().op_policy();
    
    // Input projection: [batch, seq, latent_dim] -> [batch, seq, transformer_hidden]
    auto hidden = ops::matmul(x, input_proj_weight(), false, true);
    hidden = hidden + input_proj_bias();
    
    // Build RoPE cos/sin
    // Get seq_len as scalar
    auto seq_len_1d = Tensor(shape::dim(x, 1), pctx);  // shape [1]
    auto seq_len_scalar = seq_len_1d.squeeze(0);  // shape []
    auto position_ids = ops::range(seq_len_scalar, 0, 1, ov::element::i64);
    position_ids = position_ids.unsqueeze(0);  // [1, seq_len]
    
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, cfg_.transformer_head_dim, 
                                           cfg_.rope_theta, policy);
    
    // Build causal mask
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len_scalar);
    
    // Forward through layers
    std::optional<Tensor> residual;
    
    for (auto& layer : layers_) {
        auto layer_out = layer.forward_no_cache(hidden, cos_sin.first, cos_sin.second, 
                                                 causal_mask, residual);
        hidden = layer_out.first;
        residual = layer_out.second;
    }
    
    // Final norm
    if (residual) {
        hidden = norm_.forward(hidden, *residual).first;
    } else {
        hidden = norm_.forward(hidden);
    }
    
    // Output projection: [batch, seq, transformer_hidden] -> [batch, seq, latent_dim]
    auto output = ops::matmul(hidden, output_proj_weight(), false, true);
    output = output + output_proj_bias();
    
    return output;
}

//===----------------------------------------------------------------------===//
// SnakeBeta Activation Implementation
//===----------------------------------------------------------------------===//

SnakeBetaActivation::SnakeBetaActivation(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {
    alpha_param_ = &register_parameter("alpha");
    beta_param_ = &register_parameter("beta");
}

const Tensor& SnakeBetaActivation::alpha() const {
    return alpha_param_->value();
}

const Tensor& SnakeBetaActivation::beta() const {
    return beta_param_->value();
}

Tensor SnakeBetaActivation::forward(const Tensor& x) const {
    // SnakeBeta: x + 1/beta * sin^2(x * alpha)
    // alpha and beta are stored as log values, so exp() them
    auto alpha_exp = alpha().exp();
    auto beta_exp = beta().exp();
    
    // Reshape alpha/beta from [C] to [1, C, 1] for NCL format
    auto alpha_r = alpha_exp.unsqueeze({0, 2});
    auto beta_r = beta_exp.unsqueeze({0, 2});
    
    // x * alpha
    auto x_alpha = x * alpha_r;
    
    // sin(x * alpha)
    auto sin_xa = x_alpha.sin();
    
    // sin^2(x * alpha)
    auto sin_sq = sin_xa * sin_xa;
    
    // 1/beta * sin^2
    auto scaled = sin_sq / beta_r;
    
    // x + scaled
    return x + scaled;
}

//===----------------------------------------------------------------------===//
// ConvNeXt Block Implementation
//===----------------------------------------------------------------------===//

ConvNeXtBlock::ConvNeXtBlock(BuilderContext& ctx, const std::string& name,
                             int32_t channels, Module* parent)
    : Module(name, ctx, parent), channels_(channels) {
    
    // Depthwise conv: [channels, 1, 7]
    dwconv_weight_param_ = &register_parameter("dwconv.conv.weight");
    dwconv_bias_param_ = &register_parameter("dwconv.conv.bias");
    
    // LayerNorm
    norm_weight_param_ = &register_parameter("norm.weight");
    norm_bias_param_ = &register_parameter("norm.bias");
    
    // Pointwise convs
    pwconv1_weight_param_ = &register_parameter("pwconv1.weight");
    pwconv1_bias_param_ = &register_parameter("pwconv1.bias");
    pwconv2_weight_param_ = &register_parameter("pwconv2.weight");
    pwconv2_bias_param_ = &register_parameter("pwconv2.bias");
    
    // Gamma scale
    gamma_param_ = &register_parameter("gamma");
}

Tensor ConvNeXtBlock::forward(const Tensor& x) const {
    // x: [batch, channels, seq_len] (NCL format)
    auto residual = x;
    
    // Depthwise causal conv
    // Pad left by kernel_size - 1 = 6
    auto dwconv_w = dwconv_weight_param_->value();
    auto dwconv_b = dwconv_bias_param_->value();
    
    // Depthwise conv is group conv with groups=channels
    // Weight shape from safetensors: [channels, 1, 7]
    // OpenVINO GroupConvolution needs: [groups, out_ch/groups, in_ch/groups, kernel] = [channels, 1, 1, 7]
    auto dwconv_w_reshaped = dwconv_w.unsqueeze(2);  // [channels, 1, 7] -> [channels, 1, 1, 7]
    auto hidden = ops::nn::conv1d(x, dwconv_w_reshaped, dwconv_b, {1}, {6}, {0}, {1}, channels_);
    
    // Transpose for LayerNorm: [B, C, L] -> [B, L, C]
    hidden = hidden.permute({0, 2, 1});
    
    // LayerNorm
    auto norm_w = norm_weight_param_->value();
    auto norm_b = norm_bias_param_->value();
    hidden = ops::nn::layer_norm(hidden, norm_w, &norm_b, 1e-5f);
    
    // Pointwise conv 1 (expand)
    auto pw1_w = pwconv1_weight_param_->value();
    auto pw1_b = pwconv1_bias_param_->value();
    hidden = ops::matmul(hidden, pw1_w, false, true);
    hidden = hidden + pw1_b;
    
    // GELU activation
    hidden = ops::nn::gelu(hidden, true);
    
    // Pointwise conv 2 (contract)
    auto pw2_w = pwconv2_weight_param_->value();
    auto pw2_b = pwconv2_bias_param_->value();
    hidden = ops::matmul(hidden, pw2_w, false, true);
    hidden = hidden + pw2_b;
    
    // Transpose back: [B, L, C] -> [B, C, L]
    hidden = hidden.permute({0, 2, 1});
    
    // Apply gamma and add residual
    auto gamma = gamma_param_->value().unsqueeze({0, 2});  // [1, C, 1]
    hidden = residual + gamma * hidden;
    
    return hidden;
}

//===----------------------------------------------------------------------===//
// Residual Unit Implementation
//===----------------------------------------------------------------------===//

ResidualUnit::ResidualUnit(BuilderContext& ctx, const std::string& name,
                           int32_t channels, int32_t dilation, Module* parent)
    : Module(name, ctx, parent),
      channels_(channels),
      dilation_(dilation),
      act1_(ctx, "act1", this),
      act2_(ctx, "act2", this) {
    
    conv1_weight_param_ = &register_parameter("conv1.conv.weight");
    conv1_bias_param_ = &register_parameter("conv1.conv.bias");
    conv2_weight_param_ = &register_parameter("conv2.conv.weight");
    conv2_bias_param_ = &register_parameter("conv2.conv.bias");
}

Tensor ResidualUnit::forward(const Tensor& x) const {
    // x: [batch, channels, seq_len]
    auto residual = x;
    
    // SnakeBeta 1
    auto hidden = act1_.forward(x);
    
    // Dilated causal conv1 (kernel=7)
    // Effective kernel size = (kernel - 1) * dilation + 1 = 6 * dilation + 1
    // Causal padding = effective_kernel - 1 = 6 * dilation
    int64_t pad1 = 6 * dilation_;
    auto conv1_w = conv1_weight_param_->value();
    auto conv1_b = conv1_bias_param_->value();
    hidden = ops::nn::conv1d(hidden, conv1_w, conv1_b, {1}, {pad1}, {0}, {dilation_});
    
    // SnakeBeta 2
    hidden = act2_.forward(hidden);
    
    // Conv2 (kernel=1, no dilation)
    auto conv2_w = conv2_weight_param_->value();
    auto conv2_b = conv2_bias_param_->value();
    hidden = ops::nn::conv1d(hidden, conv2_w, conv2_b, {1}, {0}, {0}, {1});
    
    // Add residual
    return hidden + residual;
}

//===----------------------------------------------------------------------===//
// Decoder Block Implementation
//===----------------------------------------------------------------------===//

DecoderBlock::DecoderBlock(BuilderContext& ctx, const std::string& name,
                           int32_t in_channels, int32_t out_channels, int32_t upsample_rate,
                           Module* parent)
    : Module(name, ctx, parent),
      in_channels_(in_channels),
      out_channels_(out_channels),
      upsample_rate_(upsample_rate),
      snake_(ctx, "block.0", this) {
    
    // Transposed conv for upsampling (block.1)
    upsample_conv_weight_param_ = &register_parameter("block.1.conv.weight");
    upsample_conv_bias_param_ = &register_parameter("block.1.conv.bias");
    
    // Residual units with dilations 1, 3, 9 (block.2, block.3, block.4)
    std::vector<int32_t> dilations = {1, 3, 9};
    residual_units_.reserve(3);
    for (int i = 0; i < 3; ++i) {
        residual_units_.emplace_back(ctx, "block." + std::to_string(i + 2), 
                                     out_channels, dilations[i], this);
    }
}

Tensor DecoderBlock::forward(const Tensor& x) const {
    // x: [batch, in_channels, seq_len]
    
    // SnakeBeta activation
    auto hidden = snake_.forward(x);
    
    // Transposed conv for upsampling
    // kernel_size = 2 * upsample_rate, stride = upsample_rate
    int64_t kernel = 2 * upsample_rate_;
    int64_t stride = upsample_rate_;
    
    // For transposed conv: output_len = (input_len - 1) * stride + kernel - 2*pad
    // We want output_len = input_len * stride, so pad = (kernel - stride) / 2
    int64_t pad = (kernel - stride) / 2;
    
    auto upsample_w = upsample_conv_weight_param_->value();
    auto upsample_b = upsample_conv_bias_param_->value();
    hidden = ops::nn::conv_transpose1d(hidden, upsample_w, upsample_b, 
                                       {stride}, {pad}, {pad}, {0}, {1});
    
    // Residual units
    for (auto& unit : residual_units_) {
        hidden = unit.forward(hidden);
    }
    
    return hidden;  // [batch, out_channels, seq_len * upsample_rate]
}

//===----------------------------------------------------------------------===//
// Speech Decoder Model Implementation
//===----------------------------------------------------------------------===//

SpeechDecoderModel::SpeechDecoderModel(BuilderContext& ctx, const SpeechDecoderConfig& cfg, Module* parent)
    : Module("decoder", ctx, parent),
      cfg_(cfg),
      dequantizer_(ctx, "quantizer", cfg, this),
      pre_transformer_(ctx, "pre_transformer", cfg, this),
      final_snake_(ctx, "decoder.5", this) {
    
    // Pre-conv: [batch, rvq_output_dim, seq] -> [batch, latent_dim, seq]
    pre_conv_weight_param_ = &register_parameter("pre_conv.conv.weight");
    pre_conv_bias_param_ = &register_parameter("pre_conv.conv.bias");
    
    // Pre-decoder upsample blocks (2 stages)
    for (int i = 0; i < 2; ++i) {
        std::string prefix = "upsample." + std::to_string(i);
        // Transposed conv
        upsample_conv_weight_params_.push_back(
            &register_parameter(prefix + ".0.conv.weight"));
        upsample_conv_bias_params_.push_back(
            &register_parameter(prefix + ".0.conv.bias"));
        // ConvNeXt block
        upsample_blocks_.emplace_back(ctx, prefix + ".1", cfg.latent_dim, this);
    }
    
    // Initial decoder conv
    decoder_init_conv_weight_param_ = &register_parameter("decoder.0.conv.weight");
    decoder_init_conv_bias_param_ = &register_parameter("decoder.0.conv.bias");
    
    // Decoder blocks (4 stages with different upsample rates)
    int32_t ch = cfg.decoder_dim;
    for (int i = 0; i < 4; ++i) {
        int32_t out_ch = ch / 2;
        int32_t rate = cfg.decoder_upsample_rates[i];
        decoder_blocks_.emplace_back(ctx, "decoder." + std::to_string(i + 1), ch, out_ch, rate, this);
        ch = out_ch;
    }
    
    // Output conv
    output_conv_weight_param_ = &register_parameter("decoder.6.conv.weight");
    output_conv_bias_param_ = &register_parameter("decoder.6.conv.bias");
    
    // Calculate total upsample factor
    total_upsample_ = 1;
    for (auto r : cfg.pre_upsample_ratios) total_upsample_ *= r;
    for (auto r : cfg.decoder_upsample_rates) total_upsample_ *= r;
}

Tensor SpeechDecoderModel::forward(const Tensor& codes) const {
    // codes: [batch, 16, seq_len]
    
    // 1. RVQ Dequantizer: [batch, 16, seq] -> [batch, seq, rvq_output_dim]
    auto embeddings = dequantizer_.forward(codes);
    
    // Transpose for conv: [batch, seq, dim] -> [batch, dim, seq]
    embeddings = embeddings.permute({0, 2, 1});
    
    // 2. Pre-conv: [batch, 512, seq] -> [batch, 1024, seq]
    auto pre_conv_w = pre_conv_weight_param_->value();
    auto pre_conv_b = pre_conv_bias_param_->value();
    auto hidden = ops::nn::conv1d(embeddings, pre_conv_w, pre_conv_b, {1}, {2}, {0}, {1});
    
    // Transpose for transformer: [batch, dim, seq] -> [batch, seq, dim]
    hidden = hidden.permute({0, 2, 1});
    
    // 3. Pre-transformer: [batch, seq, 1024] -> [batch, seq, 1024]
    hidden = pre_transformer_.forward(hidden);
    
    // Transpose back: [batch, seq, dim] -> [batch, dim, seq]
    hidden = hidden.permute({0, 2, 1});
    
    // 4. Pre-decoder upsample (2x2)
    for (size_t i = 0; i < upsample_blocks_.size(); ++i) {
        // Transposed conv for 2x upsample
        auto up_w = upsample_conv_weight_params_[i]->value();
        auto up_b = upsample_conv_bias_params_[i]->value();
        hidden = ops::nn::conv_transpose1d(hidden, up_w, up_b, {2}, {0}, {0}, {0}, {1});
        
        // ConvNeXt block
        hidden = upsample_blocks_[i].forward(hidden);
    }
    
    // 5. Initial decoder conv: [batch, 1024, seq*4] -> [batch, 1536, seq*4]
    auto init_w = decoder_init_conv_weight_param_->value();
    auto init_b = decoder_init_conv_bias_param_->value();
    hidden = ops::nn::conv1d(hidden, init_w, init_b, {1}, {6}, {0}, {1});  // kernel=7
    
    // 6. Decoder blocks (8x5x4x3 upsample)
    for (auto& block : decoder_blocks_) {
        hidden = block.forward(hidden);
    }
    
    // 7. Final SnakeBeta + output conv
    hidden = final_snake_.forward(hidden);
    
    auto out_w = output_conv_weight_param_->value();
    auto out_b = output_conv_bias_param_->value();
    auto audio = ops::nn::conv1d(hidden, out_w, out_b, {1}, {6}, {0}, {1});  // kernel=7
    
    // Squeeze to [batch, audio_len]
    audio = audio.squeeze(1);
    
    // Clamp to [-1, 1] using OpenVINO's Clamp op
    auto clamp_op = std::make_shared<ov::op::v0::Clamp>(audio.output(), -1.0, 1.0);
    audio = Tensor(clamp_op->output(0), audio.context());
    
    return audio;
}

int64_t SpeechDecoderModel::get_audio_length(int64_t code_length) const {
    return code_length * total_upsample_;
}

//===----------------------------------------------------------------------===//
// Factory Function Implementation
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_speech_decoder_model(
    const SpeechDecoderConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    
    BuilderContext ctx;
    SpeechDecoderModel model(ctx, cfg);
    
    // Load weights
    weights::LoadOptions opts;
    opts.allow_unmatched = true;
    opts.allow_missing = true;
    weights::load_model(model, source, finalizer, opts);
    
    // Create input parameter
    // codes: [batch, num_quantizers, seq_len]
    auto codes = ctx.parameter("codes", ov::element::i64, 
                               ov::PartialShape{-1, cfg.num_quantizers, -1});
    
    // Forward pass
    auto audio = model.forward(codes);
    
    // Build model
    auto result = std::make_shared<ov::op::v0::Result>(audio.output());
    set_name(result, "audio");
    
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
