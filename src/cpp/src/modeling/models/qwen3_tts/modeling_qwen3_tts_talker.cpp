// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_talker.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/rope.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_source.hpp"

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
// Qwen3TTSTextProjection Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSTextProjection::Qwen3TTSTextProjection(BuilderContext& ctx,
                                               const std::string& name,
                                               const Qwen3TTSTalkerConfig& cfg,
                                               Module* parent)
    : Module(name, ctx, parent) {
    linear_fc1_weight_param_ = &register_parameter("linear_fc1.weight");
    linear_fc1_bias_param_ = &register_parameter("linear_fc1.bias");
    linear_fc2_weight_param_ = &register_parameter("linear_fc2.weight");
    linear_fc2_bias_param_ = &register_parameter("linear_fc2.bias");
}

const Tensor& Qwen3TTSTextProjection::linear_fc1_weight() const {
    if (!linear_fc1_weight_param_) {
        OPENVINO_THROW("Qwen3TTSTextProjection linear_fc1.weight not registered");
    }
    return linear_fc1_weight_param_->value();
}

const Tensor* Qwen3TTSTextProjection::linear_fc1_bias() const {
    return (linear_fc1_bias_param_ && linear_fc1_bias_param_->is_bound())
               ? &linear_fc1_bias_param_->value()
               : nullptr;
}

const Tensor& Qwen3TTSTextProjection::linear_fc2_weight() const {
    if (!linear_fc2_weight_param_) {
        OPENVINO_THROW("Qwen3TTSTextProjection linear_fc2.weight not registered");
    }
    return linear_fc2_weight_param_->value();
}

const Tensor* Qwen3TTSTextProjection::linear_fc2_bias() const {
    return (linear_fc2_bias_param_ && linear_fc2_bias_param_->is_bound())
               ? &linear_fc2_bias_param_->value()
               : nullptr;
}

Tensor Qwen3TTSTextProjection::forward(const Tensor& x) const {
    // ResizeMLP: input[896] -> fc1[2048] -> silu -> fc2[2048]
    auto h = ops::linear(x, linear_fc1_weight());
    if (auto* bias = linear_fc1_bias()) {
        h = h + *bias;
    }
    h = ops::silu(h);
    h = ops::linear(h, linear_fc2_weight());
    if (auto* bias = linear_fc2_bias()) {
        h = h + *bias;
    }
    return h;
}

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerAttention Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSTalkerAttention::Qwen3TTSTalkerAttention(BuilderContext& ctx,
                                                 const std::string& name,
                                                 const Qwen3TTSTalkerConfig& cfg,
                                                 Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3TTSTalkerAttention head configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }

    register_module("q_norm", &q_norm_);
    register_module("k_norm", &k_norm_);

    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    q_bias_param_ = &register_parameter("q_proj.bias");
    k_bias_param_ = &register_parameter("k_proj.bias");
    v_bias_param_ = &register_parameter("v_proj.bias");
    o_bias_param_ = &register_parameter("o_proj.bias");

    if (!cfg.attention_bias) {
        q_bias_param_->set_optional(true);
        k_bias_param_->set_optional(true);
        v_bias_param_->set_optional(true);
        o_bias_param_->set_optional(true);
    }
}

const Tensor& Qwen3TTSTalkerAttention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerAttention q_proj not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3TTSTalkerAttention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerAttention k_proj not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3TTSTalkerAttention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerAttention v_proj not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3TTSTalkerAttention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerAttention o_proj not registered");
    }
    return o_proj_param_->value();
}

const Tensor* Qwen3TTSTalkerAttention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* Qwen3TTSTalkerAttention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* Qwen3TTSTalkerAttention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* Qwen3TTSTalkerAttention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor Qwen3TTSTalkerAttention::forward_no_cache(const Tensor& hidden_states,
                                                 const Tensor& rope_cos,
                                                 const Tensor& rope_sin,
                                                 const Tensor& causal_mask) const {
    // Q, K, V projections
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    if (auto* bias = q_proj_bias()) {
        q = q + *bias;
    }
    if (auto* bias = k_proj_bias()) {
        k = k + *bias;
    }
    if (auto* bias = v_proj_bias()) {
        v = v + *bias;
    }

    // Reshape to [B, T, num_heads, head_dim] and permute to [B, num_heads, T, head_dim]
    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    // Apply QK normalization (Qwen3 style)
    if (q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    // Apply RoPE
    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    // Expand KV for GQA
    auto k_expanded = ops::llm::repeat_kv(k_rot, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_heads, num_heads_, num_kv_heads_, head_dim_);

    // SDPA with causal mask
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &causal_mask, false, policy);

    // Merge heads: [B, num_heads, T, head_dim] -> [B, T, hidden_size]
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});

    // Output projection
    auto out = ops::linear(merged, o_proj_weight());
    if (auto* bias = o_proj_bias()) {
        out = out + *bias;
    }
    return out;
}

AttentionKVOutput Qwen3TTSTalkerAttention::forward_with_cache(
    const Tensor& hidden_states,
    const Tensor& rope_cos,
    const Tensor& rope_sin,
    const Tensor& attention_mask,
    const std::optional<Tensor>& past_key,
    const std::optional<Tensor>& past_value) const {
    // Q, K, V projections
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    if (auto* bias = q_proj_bias()) {
        q = q + *bias;
    }
    if (auto* bias = k_proj_bias()) {
        k = k + *bias;
    }
    if (auto* bias = v_proj_bias()) {
        v = v + *bias;
    }

    // Reshape to [B, T, num_heads, head_dim] and permute to [B, num_heads, T, head_dim]
    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    // Apply QK normalization (Qwen3 style)
    if (q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    // Apply RoPE
    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    // Concatenate with past KV cache if present
    Tensor k_combined = k_rot;
    Tensor v_combined = v_heads;
    if (past_key.has_value() && past_value.has_value()) {
        k_combined = ops::concat({*past_key, k_rot}, 2);  // concat on seq_len dimension
        v_combined = ops::concat({*past_value, v_heads}, 2);
    }

    // Expand KV for GQA
    auto k_expanded = ops::llm::repeat_kv(k_combined, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_combined, num_heads_, num_kv_heads_, head_dim_);

    // Use the 4D attention mask directly (0.0 = attend, -inf = mask)
    // attention_mask is already in the correct format [batch, 1, q_len, kv_len]
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &attention_mask, false, policy);

    // Merge heads: [B, num_heads, T, head_dim] -> [B, T, hidden_size]
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});

    // Output projection
    auto out = ops::linear(merged, o_proj_weight());
    if (auto* bias = o_proj_bias()) {
        out = out + *bias;
    }

    return AttentionKVOutput{out, k_combined, v_combined};
}

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerMLP Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSTalkerMLP::Qwen3TTSTalkerMLP(BuilderContext& ctx,
                                     const std::string& name,
                                     const Qwen3TTSTalkerConfig& cfg,
                                     Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3TTSTalker MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3TTSTalkerMLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerMLP gate_proj not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& Qwen3TTSTalkerMLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerMLP up_proj not registered");
    }
    return up_proj_param_->value();
}

const Tensor& Qwen3TTSTalkerMLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("Qwen3TTSTalkerMLP down_proj not registered");
    }
    return down_proj_param_->value();
}

Tensor Qwen3TTSTalkerMLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerDecoderLayer Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSTalkerDecoderLayer::Qwen3TTSTalkerDecoderLayer(BuilderContext& ctx,
                                                       const std::string& name,
                                                       const Qwen3TTSTalkerConfig& cfg,
                                                       Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
    register_module("self_attn", &self_attn_);
    register_module("mlp", &mlp_);
    register_module("input_layernorm", &input_layernorm_);
    register_module("post_attention_layernorm", &post_attention_layernorm_);
}

std::pair<Tensor, Tensor> Qwen3TTSTalkerDecoderLayer::forward_no_cache(
    const Tensor& hidden_states,
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
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);

    return {mlp_out, post_norm.second};
}

DecoderLayerKVOutput Qwen3TTSTalkerDecoderLayer::forward_with_cache(
    const Tensor& hidden_states,
    const Tensor& rope_cos,
    const Tensor& rope_sin,
    const Tensor& attention_mask,
    const std::optional<Tensor>& residual,
    const std::optional<Tensor>& past_key,
    const std::optional<Tensor>& past_value) const {
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

    auto attn_result = self_attn_.forward_with_cache(normed, rope_cos, rope_sin, attention_mask, past_key, past_value);
    auto post_norm = post_attention_layernorm_.forward(attn_result.hidden_states, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);

    return DecoderLayerKVOutput{mlp_out, post_norm.second, attn_result.key_cache, attn_result.value_cache};
}

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerModel Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSTalkerModel::Qwen3TTSTalkerModel(BuilderContext& ctx,
                                         const Qwen3TTSTalkerConfig& cfg,
                                         Module* parent)
    : Module("model", ctx, parent),
      cfg_(cfg),
      text_embedding_(ctx, "text_embedding", this),
      codec_embedding_(ctx, "codec_embedding", this),
      // text_projection uses absolute path "talker.text_projection" since it's not under model
      text_projection_(ctx, "talker.text_projection", cfg, nullptr),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim
                                 : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    register_module("text_embedding", &text_embedding_);
    register_module("codec_embedding", &codec_embedding_);
    register_module("norm", &norm_);

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
        register_module("layers[" + std::to_string(i) + "]", &layers_.back());
    }
}

std::pair<Tensor, Tensor> Qwen3TTSTalkerModel::build_mrope_cos_sin(const Tensor& position_ids) const {
    auto* op_ctx = position_ids.context();
    const int32_t half_dim = head_dim_ / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim_);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta_, exponent);
    }

    auto inv_freq_const = ops::const_vec(op_ctx, inv_freq);
    Tensor inv_freq_tensor(inv_freq_const, op_ctx);
    auto inv_freq_reshaped = inv_freq_tensor.reshape({1, 1, static_cast<int64_t>(half_dim)}, false);

    // position_ids shape: [3, B, T] for [temporal, height, width]
    auto pos_t = ops::slice(position_ids, 0, 1, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_h = ops::slice(position_ids, 1, 2, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_w = ops::slice(position_ids, 2, 3, 1, 0).squeeze(0).to(ov::element::f32);

    auto freqs_t = pos_t.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_h = pos_h.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_w = pos_w.unsqueeze(2) * inv_freq_reshaped;

    if (!cfg_.mrope_interleaved) {
        return {freqs_t.cos(), freqs_t.sin()};
    }

    // Stack and apply mRoPE interleaving
    auto freqs_all = ops::tensor::stack({freqs_t, freqs_h, freqs_w}, 0);
    auto freqs = ops::rope::mrope_interleaved(freqs_all, cfg_.mrope_section);
    return {freqs.cos(), freqs.sin()};
}

std::pair<Tensor, Tensor> Qwen3TTSTalkerModel::forward_no_cache(const Tensor& inputs_embeds,
                                                                const Tensor& position_ids) {
    auto hidden_states = inputs_embeds;
    auto* policy = &ctx().op_policy();

    // Build mRoPE cos/sin
    auto [rope_cos, rope_sin] = build_mrope_cos_sin(position_ids);

    // Build causal mask
    auto seq_len = Tensor(shape::dim(position_ids, 2), position_ids.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);

    // Forward through layers
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward_no_cache(hidden_states, rope_cos, rope_sin, causal_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }

    // Apply final norm
    Tensor pre_norm_hidden;
    if (residual) {
        auto norm_out = norm_.forward(hidden_states, *residual);
        pre_norm_hidden = norm_out.first;
    } else {
        pre_norm_hidden = norm_.forward(hidden_states);
    }

    return {pre_norm_hidden, hidden_states};
}

TalkerModelKVOutput Qwen3TTSTalkerModel::forward_with_cache(const Tensor& inputs_embeds,
                                                           const Tensor& position_ids,
                                                           const Tensor& attention_mask,
                                                           const std::vector<Tensor>& past_keys,
                                                           const std::vector<Tensor>& past_values) {
    auto hidden_states = inputs_embeds;

    // Build mRoPE cos/sin
    auto [rope_cos, rope_sin] = build_mrope_cos_sin(position_ids);

    // attention_mask: [batch, kv_len] where 1=attend, 0=mask
    // The actual causal mask will be built inside each attention layer using
    // build_kv_causal_mask_with_attention which correctly handles q_len vs kv_len

    // Forward through layers, collecting KV caches
    std::vector<Tensor> key_caches;
    std::vector<Tensor> value_caches;
    key_caches.reserve(layers_.size());
    value_caches.reserve(layers_.size());

    std::optional<Tensor> residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
        std::optional<Tensor> past_key = (i < past_keys.size()) ? std::optional<Tensor>(past_keys[i]) : std::nullopt;
        std::optional<Tensor> past_value =
            (i < past_values.size()) ? std::optional<Tensor>(past_values[i]) : std::nullopt;

        auto layer_out =
            layers_[i].forward_with_cache(hidden_states, rope_cos, rope_sin, attention_mask, residual, past_key, past_value);
        hidden_states = layer_out.hidden_states;
        residual = layer_out.residual;
        key_caches.push_back(layer_out.key_cache);
        value_caches.push_back(layer_out.value_cache);
    }

    // Apply final norm
    Tensor pre_norm_hidden;
    if (residual) {
        auto norm_out = norm_.forward(hidden_states, *residual);
        pre_norm_hidden = norm_out.first;
    } else {
        pre_norm_hidden = norm_.forward(hidden_states);
    }

    return TalkerModelKVOutput{pre_norm_hidden, hidden_states, std::move(key_caches), std::move(value_caches)};
}

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerForConditionalGeneration Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSTalkerForConditionalGeneration::Qwen3TTSTalkerForConditionalGeneration(BuilderContext& ctx,
                                                                               const Qwen3TTSTalkerConfig& cfg,
                                                                               Module* parent)
    : Module("talker", ctx, parent), cfg_(cfg), model_(ctx, cfg, this), codec_head_(ctx, "codec_head", this) {
    // model_ is already named "model" and is a child of "talker" -> talker.model.xxx
    register_module("model", &model_);
    // text_projection uses absolute path "talker.text_projection" (defined in model_)
    // codec_head is a direct child -> talker.codec_head.xxx
    register_module("codec_head", &codec_head_);
}

std::pair<Tensor, Tensor> Qwen3TTSTalkerForConditionalGeneration::forward_no_cache(const Tensor& inputs_embeds,
                                                                                   const Tensor& position_ids) {
    auto [hidden_states, pre_norm] = model_.forward_no_cache(inputs_embeds, position_ids);
    auto logits = codec_head_.forward(hidden_states);
    return {logits, hidden_states};
}

TalkerGenerationKVOutput Qwen3TTSTalkerForConditionalGeneration::forward_with_cache(
    const Tensor& inputs_embeds,
    const Tensor& position_ids,
    const Tensor& attention_mask,
    const std::vector<Tensor>& past_keys,
    const std::vector<Tensor>& past_values) {
    auto model_out = model_.forward_with_cache(inputs_embeds, position_ids, attention_mask, past_keys, past_values);
    auto logits = codec_head_.forward(model_out.hidden_states);
    return TalkerGenerationKVOutput{logits, model_out.hidden_states, std::move(model_out.key_caches),
                                    std::move(model_out.value_caches)};
}

//===----------------------------------------------------------------------===//
// Factory Functions - Embedding Models
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_embedding_model(const Qwen3TTSTalkerConfig& cfg,
                                                            ov::genai::modeling::weights::WeightSource& source,
                                                            ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;

    // Create modules for embedding lookup
    // Note: HF weight paths are talker.model.xxx for embeddings, talker.text_projection for projection
    VocabEmbedding text_embedding(ctx, "talker.model.text_embedding", nullptr);
    VocabEmbedding codec_embedding(ctx, "talker.model.codec_embedding", nullptr);
    Qwen3TTSTextProjection text_projection(ctx, "talker.text_projection", cfg, nullptr);

    // Load weights
    auto lenient = ov::genai::modeling::weights::LoadOptions::lenient();
    ov::genai::modeling::weights::load_model(text_embedding, source, finalizer, lenient);
    ov::genai::modeling::weights::load_model(codec_embedding, source, finalizer, lenient);
    ov::genai::modeling::weights::load_model(text_projection, source, finalizer, lenient);

    // Create inputs - ctx.parameter() returns Tensor directly
    auto text_input_ids = ctx.parameter("text_input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto codec_input_ids = ctx.parameter("codec_input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    // codec_mask: [batch, seq] float mask, 1.0 = use codec, 0.0 = text only
    auto codec_mask = ctx.parameter("codec_mask", ov::element::f32, ov::PartialShape{-1, -1});

    // Forward: text_embed = text_projection(text_embedding(text_ids))
    //          codec_embed = codec_embedding(codec_ids) * codec_mask
    //          inputs_embeds = text_embed + codec_embed
    auto text_embed = text_embedding.forward(text_input_ids);
    auto text_projected = text_projection.forward(text_embed);
    auto codec_embed = codec_embedding.forward(codec_input_ids);
    // Apply mask: expand mask to [batch, seq, 1] for broadcasting with [batch, seq, hidden]
    auto mask_expanded = codec_mask.unsqueeze({2});
    auto masked_codec_embed = codec_embed * mask_expanded;
    auto inputs_embeds = text_projected + masked_codec_embed;

    // Build output
    auto result = std::make_shared<ov::op::v0::Result>(inputs_embeds.output());
    set_name(result, "inputs_embeds");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_tts_codec_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;

    // Create codec embedding module only
    // Note: HF weight path is talker.model.codec_embedding
    VocabEmbedding codec_embedding(ctx, "talker.model.codec_embedding", nullptr);
    ov::genai::modeling::weights::load_model(codec_embedding, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create input - ctx.parameter() returns Tensor directly
    auto codec_input_ids = ctx.parameter("codec_input_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // Forward: codec_embed = codec_embedding(codec_ids)
    auto codec_embed = codec_embedding.forward(codec_input_ids);

    // Build output
    auto result = std::make_shared<ov::op::v0::Result>(codec_embed.output());
    set_name(result, "codec_embeds");

    return ctx.build_model({result->output(0)});
}

//===----------------------------------------------------------------------===//
// Factory Functions - Talker Models
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_talker_model(const Qwen3TTSTalkerConfig& cfg,
                                                         ov::genai::modeling::weights::WeightSource& source,
                                                         ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSTalkerForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create inputs - ctx.parameter() returns Tensor directly
    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{3, -1, -1});

    // Forward (no cache)
    auto [logits, hidden_states] = model.forward_no_cache(inputs_embeds, position_ids);

    // Build outputs
    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden_states.output());
    set_name(hidden_result, "hidden_states");

    return ctx.build_model({logits_result->output(0), hidden_result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_tts_talker_prefill_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSTalkerForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create inputs - ctx.parameter() returns Tensor directly
    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{3, -1, -1});
    // attention_mask: [batch, 1, seq_len, seq_len] float mask
    // 0.0 = attend, -inf = mask (causal mask)
    auto attention_mask = ctx.parameter("attention_mask", ov::element::f32, ov::PartialShape{-1, 1, -1, -1});

    // Empty past KV caches for prefill (seq_len = 0)
    // Following working version: create input parameters even for prefill
    std::vector<Tensor> past_keys, past_values;
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto past_key = ctx.parameter("past_key_" + std::to_string(i), ov::element::f32,
                                      ov::PartialShape{-1, cfg.num_key_value_heads, 0, cfg.head_dim});
        auto past_value = ctx.parameter("past_value_" + std::to_string(i), ov::element::f32,
                                        ov::PartialShape{-1, cfg.num_key_value_heads, 0, cfg.head_dim});
        past_keys.push_back(past_key);
        past_values.push_back(past_value);
    }
    
    auto result = model.forward_with_cache(inputs_embeds, position_ids, attention_mask, past_keys, past_values);

    // Build outputs
    std::vector<ov::Output<ov::Node>> outputs;

    auto logits_result = std::make_shared<ov::op::v0::Result>(result.logits.output());
    set_name(logits_result, "logits");
    outputs.push_back(logits_result->output(0));

    auto hidden_result = std::make_shared<ov::op::v0::Result>(result.hidden_states.output());
    set_name(hidden_result, "hidden_states");
    outputs.push_back(hidden_result->output(0));

    // Output KV caches for each layer (named present_key_* to match working version)
    for (size_t i = 0; i < result.key_caches.size(); ++i) {
        auto key_result = std::make_shared<ov::op::v0::Result>(result.key_caches[i].output());
        set_name(key_result, "present_key_" + std::to_string(i));
        outputs.push_back(key_result->output(0));

        auto value_result = std::make_shared<ov::op::v0::Result>(result.value_caches[i].output());
        set_name(value_result, "present_value_" + std::to_string(i));
        outputs.push_back(value_result->output(0));
    }

    return ctx.build_model(outputs);
}

std::shared_ptr<ov::Model> create_qwen3_tts_talker_decode_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSTalkerForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create inputs - ctx.parameter() returns Tensor directly
    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, 1, cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{3, -1, 1});
    // attention_mask: [batch, 1, 1, total_len] float mask
    // 0.0 = attend (decode attends to all past tokens)
    auto attention_mask = ctx.parameter("attention_mask", ov::element::f32, ov::PartialShape{-1, 1, 1, -1});

    // Create past KV cache inputs for each layer
    std::vector<Tensor> past_keys;
    std::vector<Tensor> past_values;
    past_keys.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    past_values.reserve(static_cast<size_t>(cfg.num_hidden_layers));

    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto past_key = ctx.parameter("past_key_" + std::to_string(i), ov::element::f32,
                                      ov::PartialShape{-1, cfg.num_key_value_heads, -1, cfg.head_dim});
        auto past_value = ctx.parameter("past_value_" + std::to_string(i), ov::element::f32,
                                        ov::PartialShape{-1, cfg.num_key_value_heads, -1, cfg.head_dim});
        past_keys.push_back(past_key);
        past_values.push_back(past_value);
    }

    // Forward with KV cache
    auto result = model.forward_with_cache(inputs_embeds, position_ids, attention_mask, past_keys, past_values);

    // Build outputs
    std::vector<ov::Output<ov::Node>> outputs;

    auto logits_result = std::make_shared<ov::op::v0::Result>(result.logits.output());
    set_name(logits_result, "logits");
    outputs.push_back(logits_result->output(0));

    auto hidden_result = std::make_shared<ov::op::v0::Result>(result.hidden_states.output());
    set_name(hidden_result, "hidden_states");
    outputs.push_back(hidden_result->output(0));

    // Output updated KV caches for each layer (named present_key_* to match working version)
    for (size_t i = 0; i < result.key_caches.size(); ++i) {
        auto key_result = std::make_shared<ov::op::v0::Result>(result.key_caches[i].output());
        set_name(key_result, "present_key_" + std::to_string(i));
        outputs.push_back(key_result->output(0));

        auto value_result = std::make_shared<ov::op::v0::Result>(result.value_caches[i].output());
        set_name(value_result, "present_value_" + std::to_string(i));
        outputs.push_back(value_result->output(0));
    }

    return ctx.build_model(outputs);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
