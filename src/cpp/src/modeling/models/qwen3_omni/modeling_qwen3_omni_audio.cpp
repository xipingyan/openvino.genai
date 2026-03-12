// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_audio.hpp"

#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

int64_t feature_bins_after_cnn(int64_t num_mel_bins) {
    int64_t out = (num_mel_bins + 1) / 2;
    out = (out + 1) / 2;
    out = (out + 1) / 2;
    return out;
}

ov::genai::modeling::Tensor apply_audio_activation(const ov::genai::modeling::Tensor& input,
                                                   const std::string& activation_function) {
    const bool use_tanh_approx = activation_function == "gelu_new" || activation_function == "gelu_pytorch_tanh";
    return ov::genai::modeling::ops::nn::gelu(input, use_tanh_approx);
}

ov::genai::modeling::Tensor build_audio_sinusoidal_positional_embedding(const ov::genai::modeling::Tensor& hidden_states,
                                                                        int32_t channels) {
    OPENVINO_ASSERT(channels > 0 && channels % 2 == 0, "Audio positional embedding requires even channels");
    auto* op_ctx = hidden_states.context();

    const int32_t half = channels / 2;
    std::vector<float> inv_timescales(static_cast<size_t>(half));
    if (half == 1) {
        inv_timescales[0] = 1.0f;
    } else {
        const float log_timescale_increment = std::log(10000.0f) / static_cast<float>(half - 1);
        for (int32_t i = 0; i < half; ++i) {
            inv_timescales[static_cast<size_t>(i)] = std::exp(-log_timescale_increment * static_cast<float>(i));
        }
    }

    auto seq_len_1d = ov::genai::modeling::shape::dim(hidden_states, 1);
    auto seq_len = ov::genai::modeling::Tensor(seq_len_1d, op_ctx).squeeze(0);
    auto positions = ov::genai::modeling::ops::range(seq_len, 0, 1, ov::element::f32).unsqueeze(1);

    auto inv = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(op_ctx, inv_timescales), op_ctx).unsqueeze(0);
    auto scaled_time = positions * inv;
    auto positional_embedding = ov::genai::modeling::ops::concat({scaled_time.sin(), scaled_time.cos()}, 1);
    return positional_embedding.unsqueeze(0);
}

ov::genai::modeling::Tensor downsample_audio_lengths(const ov::genai::modeling::Tensor& input_lengths) {
    auto lengths = input_lengths.to(ov::element::f32);
    lengths = (lengths + 1.0f) / 2.0f;
    lengths = (lengths + 1.0f) / 2.0f;
    lengths = (lengths + 1.0f) / 2.0f;
    return lengths.to(ov::element::i64);
}

int64_t feature_len_after_3x_stride2(int64_t length) {
    int64_t out = (length + 1) / 2;
    out = (out + 1) / 2;
    out = (out + 1) / 2;
    return std::max<int64_t>(1, out);
}

ov::genai::modeling::Tensor build_audio_padding_mask(const ov::genai::modeling::Tensor& hidden_states,
                                                     const ov::genai::modeling::Tensor& audio_feature_lengths,
                                                     int32_t n_window,
                                                     int32_t n_window_infer) {
    auto* op_ctx = hidden_states.context();
    auto seq_len_1d = ov::genai::modeling::shape::dim(hidden_states, 1);
    auto seq_len = ov::genai::modeling::Tensor(seq_len_1d, op_ctx).squeeze(0);

    auto input_lengths = audio_feature_lengths.to(ov::element::i64);
    auto lengths_shape = ov::genai::modeling::shape::of(input_lengths);
    auto seq_len_b = ov::genai::modeling::shape::broadcast_to(ov::genai::modeling::Tensor(seq_len_1d, op_ctx), lengths_shape);
    auto one_i64 = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(op_ctx, int64_t(1)), op_ctx);
    auto downsampled_lengths = downsample_audio_lengths(input_lengths);
    auto already_after_cnn = ov::genai::modeling::ops::less_equal(input_lengths, seq_len_b);
    auto lengths = ov::genai::modeling::ops::where(already_after_cnn, input_lengths, downsampled_lengths);
    auto clamped_lower = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::Maximum>(lengths.output(), one_i64.output())->output(0),
        op_ctx);
    auto clamped_lengths = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::Minimum>(clamped_lower.output(), seq_len_b.output())->output(0),
        op_ctx);

    auto positions = ov::genai::modeling::ops::range(seq_len, 0, 1, ov::element::i64).unsqueeze(0);
    auto valid_lens = (clamped_lengths - one_i64).unsqueeze(1);
    auto key_valid = ov::genai::modeling::ops::less_equal(positions, valid_lens);
    auto query_valid = ov::genai::modeling::ops::less_equal(positions, valid_lens);

    const int64_t base_window = std::max<int64_t>(1, static_cast<int64_t>(n_window) * 2);
    const int64_t ratio = std::max<int64_t>(1, static_cast<int64_t>(n_window_infer) / base_window);
    const int64_t chunk_size = std::max<int64_t>(1, feature_len_after_3x_stride2(base_window) * ratio);
    auto chunk_size_tensor = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(op_ctx, chunk_size), op_ctx);

    auto row_idx = positions.transpose({1, 0});
    auto col_idx = positions;
    auto row_chunk = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::Divide>(row_idx.output(), chunk_size_tensor.output())->output(0),
        op_ctx);
    auto col_chunk = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::Divide>(col_idx.output(), chunk_size_tensor.output())->output(0),
        op_ctx);
    auto same_chunk = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::Equal>(row_chunk.output(), col_chunk.output())->output(0),
        op_ctx)
                          .unsqueeze(0);

    auto query_valid_3d = query_valid.unsqueeze(2);
    auto key_valid_3d = key_valid.unsqueeze(1);
    auto chunk_and_query = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::LogicalAnd>(same_chunk.output(), query_valid_3d.output())->output(0),
        op_ctx);
    auto valid_3d = ov::genai::modeling::Tensor(
        std::make_shared<ov::opset13::LogicalAnd>(chunk_and_query.output(), key_valid_3d.output())->output(0),
        op_ctx);
    auto valid = valid_3d.unsqueeze(1);

    auto zero = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(op_ctx, 0.0f), op_ctx);
    auto neg_inf = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(op_ctx, -65504.0f), op_ctx);
    return ov::genai::modeling::ops::where(valid, zero, neg_inf);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

class Qwen3OmniAudioAttention : public Module {
public:
    Qwen3OmniAudioAttention(BuilderContext& ctx,
                            const std::string& name,
                            const Qwen3OmniAudioConfig& cfg,
                            Module* parent = nullptr)
        : Module(name, ctx, parent),
          embed_dim_(cfg.d_model),
          num_heads_(cfg.encoder_attention_heads),
          head_dim_(cfg.d_model / std::max<int32_t>(1, cfg.encoder_attention_heads)),
          scaling_(1.0f / std::sqrt(static_cast<float>(std::max<int32_t>(1, head_dim_)))) {
        q_proj_weight_ = &register_parameter("q_proj.weight");
        q_proj_bias_ = &register_parameter("q_proj.bias");
        k_proj_weight_ = &register_parameter("k_proj.weight");
        k_proj_bias_ = &register_parameter("k_proj.bias");
        v_proj_weight_ = &register_parameter("v_proj.weight");
        v_proj_bias_ = &register_parameter("v_proj.bias");
        out_proj_weight_ = &register_parameter("out_proj.weight");
        out_proj_bias_ = &register_parameter("out_proj.bias");
    }

    Tensor forward(const Tensor& hidden_states, const Tensor* attention_mask = nullptr) const {
        auto q = ops::linear(hidden_states, q_proj_weight()) + q_proj_bias();
        auto k = ops::linear(hidden_states, k_proj_weight()) + k_proj_bias();
        auto v = ops::linear(hidden_states, v_proj_weight()) + v_proj_bias();

        auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto k_heads = k.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto v_heads = v.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});

        auto* policy = &ctx().op_policy();
        auto context = ops::llm::sdpa(q_heads, k_heads, v_heads, scaling_, 3, attention_mask, false, policy);
        auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, embed_dim_});
        return ops::linear(merged, out_proj_weight()) + out_proj_bias();
    }

private:
    const Tensor& q_proj_weight() const { return q_proj_weight_->value(); }
    const Tensor& q_proj_bias() const { return q_proj_bias_->value(); }
    const Tensor& k_proj_weight() const { return k_proj_weight_->value(); }
    const Tensor& k_proj_bias() const { return k_proj_bias_->value(); }
    const Tensor& v_proj_weight() const { return v_proj_weight_->value(); }
    const Tensor& v_proj_bias() const { return v_proj_bias_->value(); }
    const Tensor& out_proj_weight() const { return out_proj_weight_->value(); }
    const Tensor& out_proj_bias() const { return out_proj_bias_->value(); }

    int32_t embed_dim_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    float scaling_ = 1.0f;

    WeightParameter* q_proj_weight_ = nullptr;
    WeightParameter* q_proj_bias_ = nullptr;
    WeightParameter* k_proj_weight_ = nullptr;
    WeightParameter* k_proj_bias_ = nullptr;
    WeightParameter* v_proj_weight_ = nullptr;
    WeightParameter* v_proj_bias_ = nullptr;
    WeightParameter* out_proj_weight_ = nullptr;
    WeightParameter* out_proj_bias_ = nullptr;
};

class Qwen3OmniAudioEncoderLayer : public Module {
public:
    Qwen3OmniAudioEncoderLayer(BuilderContext& ctx,
                               const std::string& name,
                               const Qwen3OmniAudioConfig& cfg,
                               Module* parent = nullptr)
        : Module(name, ctx, parent),
          self_attn_(ctx, "self_attn", cfg, this),
          embed_dim_(cfg.d_model),
          intermediate_dim_(cfg.encoder_ffn_dim),
                    eps_(cfg.layer_norm_eps),
                    activation_function_(cfg.activation_function) {
        register_module("self_attn", &self_attn_);

        self_attn_ln_weight_ = &register_parameter("self_attn_layer_norm.weight");
        self_attn_ln_bias_ = &register_parameter("self_attn_layer_norm.bias");
        fc1_weight_ = &register_parameter("fc1.weight");
        fc1_bias_ = &register_parameter("fc1.bias");
        fc2_weight_ = &register_parameter("fc2.weight");
        fc2_bias_ = &register_parameter("fc2.bias");
        final_ln_weight_ = &register_parameter("final_layer_norm.weight");
        final_ln_bias_ = &register_parameter("final_layer_norm.bias");
    }

    Tensor forward(const Tensor& hidden_states, const Tensor* attention_mask = nullptr) const {
        auto residual = hidden_states;
        auto normed = ops::nn::layer_norm(hidden_states, self_attn_ln_weight(), &self_attn_ln_bias(), eps_, -1);
        auto attn_out = self_attn_.forward(normed, attention_mask);
        auto after_attn = residual + attn_out;

        auto residual2 = after_attn;
        auto normed2 = ops::nn::layer_norm(after_attn, final_ln_weight(), &final_ln_bias(), eps_, -1);
        auto ff = ops::linear(normed2, fc1_weight()) + fc1_bias();
        ff = apply_audio_activation(ff, activation_function_);
        ff = ops::linear(ff, fc2_weight()) + fc2_bias();
        return residual2 + ff;
    }

private:
    const Tensor& self_attn_ln_weight() const { return self_attn_ln_weight_->value(); }
    const Tensor& self_attn_ln_bias() const { return self_attn_ln_bias_->value(); }
    const Tensor& fc1_weight() const { return fc1_weight_->value(); }
    const Tensor& fc1_bias() const { return fc1_bias_->value(); }
    const Tensor& fc2_weight() const { return fc2_weight_->value(); }
    const Tensor& fc2_bias() const { return fc2_bias_->value(); }
    const Tensor& final_ln_weight() const { return final_ln_weight_->value(); }
    const Tensor& final_ln_bias() const { return final_ln_bias_->value(); }

    Qwen3OmniAudioAttention self_attn_;
    int32_t embed_dim_ = 0;
    int32_t intermediate_dim_ = 0;
    float eps_ = 1e-5f;
    std::string activation_function_;

    WeightParameter* self_attn_ln_weight_ = nullptr;
    WeightParameter* self_attn_ln_bias_ = nullptr;
    WeightParameter* fc1_weight_ = nullptr;
    WeightParameter* fc1_bias_ = nullptr;
    WeightParameter* fc2_weight_ = nullptr;
    WeightParameter* fc2_bias_ = nullptr;
    WeightParameter* final_ln_weight_ = nullptr;
    WeightParameter* final_ln_bias_ = nullptr;
};

class Qwen3OmniAudioEncoderFront : public Module {
public:
    Qwen3OmniAudioEncoderFront(BuilderContext& ctx,
                               const std::string& name,
                               const Qwen3OmniAudioConfig& cfg,
                               Module* parent = nullptr)
        : Module(name, ctx, parent),
          cfg_(cfg),
                    eps_(cfg.layer_norm_eps) {
        conv2d1_weight_ = &register_parameter("conv2d1.weight");
        conv2d1_bias_ = &register_parameter("conv2d1.bias");
        conv2d2_weight_ = &register_parameter("conv2d2.weight");
        conv2d2_bias_ = &register_parameter("conv2d2.bias");
        conv2d3_weight_ = &register_parameter("conv2d3.weight");
        conv2d3_bias_ = &register_parameter("conv2d3.bias");

        conv_out_weight_ = &register_parameter("conv_out.weight");
        conv_out_bias_ = &register_parameter("conv_out.bias");
        conv_out_bias_->set_optional(true);

        ln_post_weight_ = &register_parameter("ln_post.weight");
        ln_post_bias_ = &register_parameter("ln_post.bias");

        proj1_weight_ = &register_parameter("proj1.weight");
        proj1_bias_ = &register_parameter("proj1.bias");
        proj2_weight_ = &register_parameter("proj2.weight");
        proj2_bias_ = &register_parameter("proj2.bias");

        layers_.reserve(static_cast<size_t>(std::max(0, cfg.encoder_layers)));
        for (int32_t i = 0; i < cfg.encoder_layers; ++i) {
            const std::string layer_name = std::string("layers.") + std::to_string(i);
            layers_.emplace_back(ctx, layer_name, cfg, this);
            register_module(layer_name, &layers_.back());
        }
    }

    Tensor forward(const Tensor& input_features,
                   const Tensor* feature_attention_mask = nullptr,
                   const Tensor* audio_feature_lengths = nullptr) const {
        auto x = input_features.unsqueeze(1);

        x = ops::nn::conv2d(x,
                            conv2d1_weight(),
                            conv2d1_bias(),
                            {2, 2},
                            {1, 1},
                            {1, 1});
        x = apply_audio_activation(x, cfg_.activation_function);

        x = ops::nn::conv2d(x,
                            conv2d2_weight(),
                            conv2d2_bias(),
                            {2, 2},
                            {1, 1},
                            {1, 1});
        x = apply_audio_activation(x, cfg_.activation_function);

        x = ops::nn::conv2d(x,
                            conv2d3_weight(),
                            conv2d3_bias(),
                            {2, 2},
                            {1, 1},
                            {1, 1});
        x = apply_audio_activation(x, cfg_.activation_function);

        const int64_t out_f = feature_bins_after_cnn(cfg_.num_mel_bins);
        const int64_t hidden_channels = cfg_.downsample_hidden_size > 0 ? cfg_.downsample_hidden_size : conv2d_channels();
        const int64_t conv_out_in = hidden_channels * out_f;
        auto x_seq = x.permute({0, 3, 1, 2}).reshape({0, 0, conv_out_in});

        auto hidden = ops::linear(x_seq, conv_out_weight());
        if (conv_out_bias_param_bound()) {
            hidden = hidden + conv_out_bias();
        }

        auto positional_embedding = build_audio_sinusoidal_positional_embedding(hidden, cfg_.d_model).to(hidden.dtype());
        hidden = hidden + positional_embedding;

        Tensor attn_mask;
        const Tensor* attn_mask_ptr = nullptr;
        if (audio_feature_lengths) {
            attn_mask = build_audio_padding_mask(hidden,
                                                 *audio_feature_lengths,
                                                 cfg_.n_window,
                                                 cfg_.n_window_infer);
            attn_mask_ptr = &attn_mask;
        } else if (feature_attention_mask) {
            auto lengths = ops::reduce_sum(feature_attention_mask->to(ov::element::i64), 1, false);
            attn_mask = build_audio_padding_mask(hidden,
                                                 lengths,
                                                 cfg_.n_window,
                                                 cfg_.n_window_infer);
            attn_mask_ptr = &attn_mask;
        }

        for (const auto& layer : layers_) {
            hidden = layer.forward(hidden, attn_mask_ptr);
        }

        hidden = ops::nn::layer_norm(hidden, ln_post_weight(), &ln_post_bias(), eps_, -1);

        hidden = ops::linear(hidden, proj1_weight()) + proj1_bias();
        hidden = apply_audio_activation(hidden, cfg_.activation_function);
        hidden = ops::linear(hidden, proj2_weight()) + proj2_bias();
        return hidden;
    }

private:
    int64_t conv2d_channels() const {
        if (!conv2d1_weight_param()) {
            return 0;
        }
        const auto shape = conv2d1_weight().output().get_shape();
        return shape.empty() ? 0 : static_cast<int64_t>(shape[0]);
    }

    bool conv_out_bias_param_bound() const {
        return conv_out_bias_ && conv_out_bias_->is_bound();
    }

    const Tensor& conv2d1_weight() const { return conv2d1_weight_->value(); }
    const Tensor& conv2d1_bias() const { return conv2d1_bias_->value(); }
    const Tensor& conv2d2_weight() const { return conv2d2_weight_->value(); }
    const Tensor& conv2d2_bias() const { return conv2d2_bias_->value(); }
    const Tensor& conv2d3_weight() const { return conv2d3_weight_->value(); }
    const Tensor& conv2d3_bias() const { return conv2d3_bias_->value(); }

    const Tensor& conv_out_weight() const { return conv_out_weight_->value(); }
    const Tensor& conv_out_bias() const { return conv_out_bias_->value(); }

    const Tensor& ln_post_weight() const { return ln_post_weight_->value(); }
    const Tensor& ln_post_bias() const { return ln_post_bias_->value(); }

    const Tensor& proj1_weight() const { return proj1_weight_->value(); }
    const Tensor& proj1_bias() const { return proj1_bias_->value(); }
    const Tensor& proj2_weight() const { return proj2_weight_->value(); }
    const Tensor& proj2_bias() const { return proj2_bias_->value(); }

    const WeightParameter* conv2d1_weight_param() const { return conv2d1_weight_; }

    Qwen3OmniAudioConfig cfg_;
    float eps_;

    WeightParameter* conv2d1_weight_ = nullptr;
    WeightParameter* conv2d1_bias_ = nullptr;
    WeightParameter* conv2d2_weight_ = nullptr;
    WeightParameter* conv2d2_bias_ = nullptr;
    WeightParameter* conv2d3_weight_ = nullptr;
    WeightParameter* conv2d3_bias_ = nullptr;

    WeightParameter* conv_out_weight_ = nullptr;
    WeightParameter* conv_out_bias_ = nullptr;

    WeightParameter* ln_post_weight_ = nullptr;
    WeightParameter* ln_post_bias_ = nullptr;

    WeightParameter* proj1_weight_ = nullptr;
    WeightParameter* proj1_bias_ = nullptr;
    WeightParameter* proj2_weight_ = nullptr;
    WeightParameter* proj2_bias_ = nullptr;

    std::vector<Qwen3OmniAudioEncoderLayer> layers_;
};

std::shared_ptr<ov::Model> create_qwen3_omni_audio_encoder_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3OmniAudioEncoderFront model(ctx, "", cfg.audio);

    PrefixMappedWeightSource thinker_source(source, "thinker.audio_tower.");
    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    (void)ov::genai::modeling::weights::load_model(model, thinker_source, finalizer, options);

    auto input_features = ctx.parameter(Qwen3OmniAudioIO::kInputFeatures,
                                        ov::element::f32,
                                        ov::PartialShape{-1, cfg.audio.num_mel_bins, -1});
    auto feature_attention_mask = ctx.parameter(Qwen3OmniAudioIO::kFeatureAttentionMask,
                                                ov::element::i64,
                                                ov::PartialShape{-1, -1});
    auto audio_feature_lengths = ctx.parameter(Qwen3OmniAudioIO::kAudioFeatureLengths,
                                               ov::element::i64,
                                               ov::PartialShape{-1});
    (void)feature_attention_mask;
    (void)audio_feature_lengths;

    auto audio_features = model.forward(input_features, &feature_attention_mask, &audio_feature_lengths);
    auto result = std::make_shared<ov::op::v0::Result>(audio_features.output());
    set_name(result, Qwen3OmniAudioIO::kAudioFeatures);
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
