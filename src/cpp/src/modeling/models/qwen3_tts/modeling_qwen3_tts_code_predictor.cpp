// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_code_predictor.hpp"

#include <cmath>
#include <limits>
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
// Code Predictor Attention Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSCodePredictorAttention::Qwen3TTSCodePredictorAttention(BuilderContext& ctx,
                                                               const std::string& name,
                                                               const Qwen3TTSCodePredictorConfig& cfg,
                                                               Module* parent)
    : Module(name, ctx, parent),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads),
      head_dim_(cfg.head_dim),
      scaling_(1.0f / std::sqrt(static_cast<float>(cfg.head_dim))) {
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");
}

const Tensor& Qwen3TTSCodePredictorAttention::q_proj_weight() const {
    return q_proj_param_->value();
}

const Tensor& Qwen3TTSCodePredictorAttention::k_proj_weight() const {
    return k_proj_param_->value();
}

const Tensor& Qwen3TTSCodePredictorAttention::v_proj_weight() const {
    return v_proj_param_->value();
}

const Tensor& Qwen3TTSCodePredictorAttention::o_proj_weight() const {
    return o_proj_param_->value();
}

Tensor Qwen3TTSCodePredictorAttention::forward_no_cache(const Tensor& hidden_states,
                                                        const Tensor& rope_cos,
                                                        const Tensor& rope_sin,
                                                        const Tensor& causal_mask) const {
    // Q/K/V projections
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    // Reshape to heads: [B, T, H*D] -> [B, H, T, D]
    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    // Q/K normalization
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

    // Expand KV for GQA
    auto k_expanded = ops::llm::repeat_kv(k_rot, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_heads, num_heads_, num_kv_heads_, head_dim_);

    // SDPA
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &causal_mask, false, policy);

    // Merge heads: [B, H, T, D] -> [B, T, H*D]
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});

    // Output projection
    auto out = ops::linear(merged, o_proj_weight());
    return out;
}

//===----------------------------------------------------------------------===//
// Code Predictor MLP Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSCodePredictorMLP::Qwen3TTSCodePredictorMLP(BuilderContext& ctx,
                                                   const std::string& name,
                                                   const Qwen3TTSCodePredictorConfig& cfg,
                                                   Module* parent)
    : Module(name, ctx, parent) {
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3TTSCodePredictorMLP::gate_proj_weight() const {
    return gate_proj_param_->value();
}

const Tensor& Qwen3TTSCodePredictorMLP::up_proj_weight() const {
    return up_proj_param_->value();
}

const Tensor& Qwen3TTSCodePredictorMLP::down_proj_weight() const {
    return down_proj_param_->value();
}

Tensor Qwen3TTSCodePredictorMLP::forward(const Tensor& x) const {
    // SwiGLU: silu(gate) * up
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto activated = ops::silu(gate) * up;
    return ops::linear(activated, down_proj_weight());
}

//===----------------------------------------------------------------------===//
// Code Predictor Decoder Layer Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSCodePredictorDecoderLayer::Qwen3TTSCodePredictorDecoderLayer(BuilderContext& ctx,
                                                                     const std::string& name,
                                                                     const Qwen3TTSCodePredictorConfig& cfg,
                                                                     Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> Qwen3TTSCodePredictorDecoderLayer::forward_no_cache(
    const Tensor& hidden_states,
    const Tensor& rope_cos,
    const Tensor& rope_sin,
    const Tensor& causal_mask,
    const std::optional<Tensor>& residual) const {
    // Pre-norm attention
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

    // Post-attention norm + MLP
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);

    return {mlp_out, post_norm.second};
}

//===----------------------------------------------------------------------===//
// Code Predictor Model Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSCodePredictorModel::Qwen3TTSCodePredictorModel(BuilderContext& ctx,
                                                       const Qwen3TTSCodePredictorConfig& cfg,
                                                       Module* parent)
    : Module("model", ctx, parent),
      cfg_(cfg),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      codec_embeddings_(),
      head_dim_(cfg.head_dim),
      rope_theta_(cfg.rope_theta) {
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        // Use canonical format layers[N] - WeightNameMapper converts HF layers.N to layers[N]
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
    }
    // Create 15 codec embeddings (for layers 0-14, predicting 1-15)
    // Weight name format: codec_embedding[i] to match canonical name conversion
    // HF name "codec_embedding.0" -> canonical "codec_embedding[0]"
    codec_embeddings_.reserve(15);
    for (int i = 0; i < 15; ++i) {
        codec_embeddings_.emplace_back(ctx, "codec_embedding[" + std::to_string(i) + "]", this);
    }
}

Tensor Qwen3TTSCodePredictorModel::forward_no_cache(const Tensor& inputs_embeds,
                                                    const Tensor& position_ids) const {
    auto hidden_states = inputs_embeds;

    // Build RoPE cos/sin (standard RoPE, not mRoPE)
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);

    // Build causal mask
    auto seq_len = Tensor(shape::dim(position_ids, 1), position_ids.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);

    // Forward through layers
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward_no_cache(hidden_states, cos_sin.first, cos_sin.second, causal_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }

    // Final norm
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

Tensor Qwen3TTSCodePredictorModel::get_codec_embed(const Tensor& codec_ids, int layer_idx) const {
    return codec_embeddings_[layer_idx].forward(codec_ids);
}

VocabEmbedding& Qwen3TTSCodePredictorModel::codec_embedding(int layer_idx) {
    return codec_embeddings_[layer_idx];
}

//===----------------------------------------------------------------------===//
// Code Predictor For Conditional Generation Implementation
//===----------------------------------------------------------------------===//

Qwen3TTSCodePredictorForConditionalGeneration::Qwen3TTSCodePredictorForConditionalGeneration(
    BuilderContext& ctx,
    const Qwen3TTSCodePredictorConfig& cfg,
    Module* parent)
    : Module("talker.code_predictor", ctx, nullptr),  // Use absolute path since it's nested under talker
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_heads_(),
      needs_projection_(cfg.talker_hidden_size != cfg.hidden_size) {
    // Create 15 lm_heads (for predicting layers 1-15)
    // Weight name format: lm_head[i] to match canonical name conversion
    // HF name "lm_head.0" -> canonical "lm_head[0]"
    lm_heads_.reserve(15);
    for (int i = 0; i < 15; ++i) {
        lm_heads_.emplace_back(ctx, "lm_head[" + std::to_string(i) + "]", this);
    }

    // Register projection weights if needed
    // HF path: talker.code_predictor.small_to_mtp_projection.weight/bias
    if (needs_projection_) {
        projection_weight_ = &register_parameter("small_to_mtp_projection.weight");
        projection_bias_ = &register_parameter("small_to_mtp_projection.bias");
    }
}

Tensor Qwen3TTSCodePredictorForConditionalGeneration::forward_no_cache(const Tensor& inputs_embeds,
                                                                       const Tensor& position_ids,
                                                                       int step) const {
    // Apply projection if needed (talker_hidden_size -> hidden_size)
    Tensor projected_embeds = inputs_embeds;
    if (needs_projection_ && projection_weight_ && projection_weight_->is_bound()) {
        // Linear: x @ W^T + b
        projected_embeds = ops::linear(inputs_embeds, projection_weight_->value());
        if (projection_bias_ && projection_bias_->is_bound()) {
            projected_embeds = projected_embeds + projection_bias_->value();
        }
    }

    // Forward through transformer
    auto hidden_states = model_.forward_no_cache(projected_embeds, position_ids);

    // Get logits from the specified lm_head, at the LAST position only
    // hidden_states shape: [batch, seq, hidden_size]
    // Slice to get hidden[:, -1:, :] for the last position
    auto last_hidden = ops::slice(hidden_states, -1, std::numeric_limits<int64_t>::max(), 1, 1);  // [batch, 1, hidden_size]

    // Use appropriate lm_head for this step
    return lm_heads_[step].forward(last_hidden);
}

Tensor Qwen3TTSCodePredictorForConditionalGeneration::get_codec_embed(const Tensor& codec_ids,
                                                                      int layer_idx) const {
    return model_.get_codec_embed(codec_ids, layer_idx);
}

Tensor Qwen3TTSCodePredictorForConditionalGeneration::get_codec_embeds_sum(
    const std::vector<Tensor>& codec_ids_list) const {
    // Sum all codec embeddings
    Tensor result = model_.get_codec_embed(codec_ids_list[0], 0);
    for (size_t i = 1; i < codec_ids_list.size() && i < 15; ++i) {
        result = result + model_.get_codec_embed(codec_ids_list[i], static_cast<int>(i));
    }
    return result;
}

Qwen3TTSCodePredictorModel& Qwen3TTSCodePredictorForConditionalGeneration::model() {
    return model_;
}

VocabEmbedding& Qwen3TTSCodePredictorForConditionalGeneration::codec_embedding(int layer_idx) {
    return model_.codec_embedding(layer_idx);
}

LMHead& Qwen3TTSCodePredictorForConditionalGeneration::lm_head(int step) {
    return lm_heads_[step];
}

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create inputs
    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // Forward and create outputs for all 15 lm_heads
    auto hidden_states = model.model().forward_no_cache(inputs_embeds, position_ids);

    std::vector<ov::Output<ov::Node>> outputs;
    for (int i = 0; i < 15; ++i) {
        auto logits = model.lm_head(i).forward(hidden_states);
        auto result = std::make_shared<ov::op::v0::Result>(logits.output());
        set_name(result, "logits_" + std::to_string(i));
        outputs.push_back(result->output(0));
    }

    return ctx.build_model(outputs);
}

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_ar_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int generation_step,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create inputs
    // Note: inputs_embeds uses talker_hidden_size because the input comes from Talker + codec embeddings
    // The projection from talker_hidden_size -> hidden_size is done inside forward_no_cache
    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, cfg.talker_hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // Forward for specific generation step
    auto logits = model.forward_no_cache(inputs_embeds, position_ids, generation_step);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create 15 inputs (one for each codec layer 1-15)
    std::vector<Tensor> codec_inputs;
    codec_inputs.reserve(15);
    for (int i = 0; i < 15; ++i) {
        auto input = ctx.parameter("codec_input_" + std::to_string(i), ov::element::i64, ov::PartialShape{-1, -1});
        codec_inputs.push_back(input);
    }

    // Sum all embeddings
    auto embeds_sum = model.get_codec_embeds_sum(codec_inputs);

    auto result = std::make_shared<ov::op::v0::Result>(embeds_sum.output());
    set_name(result, "codec_embeds_sum");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_single_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int codec_layer,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, source, finalizer, ov::genai::modeling::weights::LoadOptions::lenient());

    // Create input for specific codec layer
    auto codec_input =
        ctx.parameter("codec_input", ov::element::i64, ov::PartialShape{-1, -1});

    // Get embedding for specific layer
    auto codec_embed = model.get_codec_embed(codec_input, codec_layer);

    auto result = std::make_shared<ov::op::v0::Result>(codec_embed.output());
    set_name(result, "codec_embed");

    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
