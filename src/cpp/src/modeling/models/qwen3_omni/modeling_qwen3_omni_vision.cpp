// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"

#include <algorithm>
#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3OmniVisionOutput {
    struct TokenHeadDebug {
        int32_t block_idx = -1;
        Tensor q_heads;
        Tensor k_heads;
        Tensor attn_prob;
        Tensor context;
    };

    Tensor visual_embeds;
    std::vector<Tensor> deepstack_embeds;
    std::vector<Tensor> block_hiddens;
    std::vector<TokenHeadDebug> token267_head_debugs;
    Tensor preblock0_patch_embed_input_5d;
    Tensor preblock0_patch_embed_conv_5d;
    Tensor preblock0_patch_embed;
    Tensor preblock0_with_pos;
    Tensor block0_input;
    Tensor block0_norm1;
    Tensor block0_qkv;
    Tensor block0_attn_out;
    Tensor block0_resid1;
    Tensor block0_norm2;
    Tensor block0_mlp_out;
    Tensor block0_mlp_fc1;
    Tensor block0_mlp_act;
    Tensor block0_mlp_fc2;
    Tensor block16_input;
    Tensor block16_norm1;
    Tensor block16_qkv;
    Tensor block25_norm1;
    Tensor block25_qkv;
    Tensor block25_attn_out;
    Tensor block25_resid1;
    Tensor block25_norm2;
    Tensor block25_mlp_out;
    Tensor block25_mlp_fc1;
    Tensor block25_mlp_act;
    Tensor block25_mlp_fc2;
    Tensor block26_norm1;
    Tensor block26_qkv;
    Tensor block26_attn_out;
    Tensor block26_resid1;
    Tensor block26_norm2;
    Tensor block26_mlp_out;
    Tensor block26_mlp_fc1;
    Tensor block26_mlp_act;
    Tensor block26_mlp_fc2;
};

class Qwen3OmniVisionPatchEmbed : public Module {
public:
    struct DebugOutput {
        Tensor input_5d;
        Tensor conv_out_5d;
    };

    Qwen3OmniVisionPatchEmbed(BuilderContext& ctx,
                              const std::string& name,
                              const Qwen3OmniVisionConfig& cfg,
                              Module* parent = nullptr)
        : Module(name, ctx, parent),
          in_channels_(cfg.in_channels),
          patch_size_(cfg.patch_size),
          temporal_patch_size_(cfg.temporal_patch_size),
          embed_dim_(cfg.hidden_size) {
        weight_param_ = &register_parameter("proj.weight");
        bias_param_ = &register_parameter("proj.bias");
    }

    Tensor forward(const Tensor& pixel_values, DebugOutput* debug = nullptr) const {
        auto x = pixel_values.reshape({0, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
        x = x.to(weight().dtype());
        if (debug) {
            debug->input_5d = x;
        }
        const std::vector<int64_t> strides = {temporal_patch_size_, patch_size_, patch_size_};
        const std::vector<int64_t> pads = {0, 0, 0};
        Tensor conv;
        if (const auto* b = bias()) {
            conv = ops::nn::conv3d(x, weight(), *b, strides, pads, pads);
        } else {
            conv = ops::nn::conv3d(x, weight(), strides, pads, pads);
        }
        if (debug) {
            debug->conv_out_5d = conv;
        }
        return conv.reshape({0, embed_dim_});
    }

private:
    const Tensor& weight() const {
        if (!weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionPatchEmbed weight parameter not registered");
        }
        return weight_param_->value();
    }

    const Tensor* bias() const {
        return (bias_param_ && bias_param_->is_bound()) ? &bias_param_->value() : nullptr;
    }

    int32_t in_channels_ = 0;
    int32_t patch_size_ = 0;
    int32_t temporal_patch_size_ = 0;
    int32_t embed_dim_ = 0;

    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
};

class Qwen3OmniVisionAttention : public Module {
public:
    struct DebugOutput {
        Tensor qkv;
        Tensor proj_out;
        Tensor token_q_heads;
        Tensor token_k_heads;
        Tensor token_attn_prob;
        Tensor token_context;
    };

    Qwen3OmniVisionAttention(BuilderContext& ctx,
                             const std::string& name,
                             const Qwen3OmniVisionConfig& cfg,
                             Module* parent = nullptr)
        : Module(name, ctx, parent),
          hidden_size_(cfg.hidden_size),
          num_heads_(cfg.num_heads),
          head_dim_(cfg.hidden_size / cfg.num_heads),
          scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
        if (hidden_size_ <= 0 || num_heads_ <= 0 || head_dim_ <= 0) {
            OPENVINO_THROW("Invalid Qwen3OmniVisionAttention configuration");
        }
        if (hidden_size_ != num_heads_ * head_dim_) {
            OPENVINO_THROW("Qwen3OmniVisionAttention hidden_size must equal num_heads * head_dim");
        }
        if (head_dim_ % 2 != 0) {
            OPENVINO_THROW("Qwen3OmniVisionAttention head_dim must be even for rotary embeddings");
        }
        qkv_weight_param_ = &register_parameter("qkv.weight");
        qkv_bias_param_ = &register_parameter("qkv.bias");
        proj_weight_param_ = &register_parameter("proj.weight");
        proj_bias_param_ = &register_parameter("proj.bias");
    }

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin,
                   const Tensor* attention_mask = nullptr,
                   DebugOutput* debug = nullptr) const {
        auto qkv_weight_t = qkv_weight().to(hidden_states.dtype());
        const Tensor* qkv_bias_t_ptr = nullptr;
        Tensor qkv_bias_t;
        if (const auto* b = qkv_bias()) {
            qkv_bias_t = b->to(hidden_states.dtype());
            qkv_bias_t_ptr = &qkv_bias_t;
        }
        auto qkv = add_bias_if_present(ops::linear(hidden_states, qkv_weight_t), qkv_bias_t_ptr);
        if (debug) {
            debug->qkv = qkv;
        }
        auto qkv_reshaped = qkv.reshape({0, 3, num_heads_, head_dim_});
        auto q = ops::slice(qkv_reshaped, 0, 1, 1, 1).squeeze(1);
        auto k = ops::slice(qkv_reshaped, 1, 2, 1, 1).squeeze(1);
        auto v = ops::slice(qkv_reshaped, 2, 3, 1, 1).squeeze(1);

        auto q_rot = apply_rotary(q, rotary_cos, rotary_sin);
        auto k_rot = apply_rotary(k, rotary_cos, rotary_sin);

        auto q_heads = q_rot.permute({1, 0, 2}).unsqueeze(0);
        auto k_heads = k_rot.permute({1, 0, 2}).unsqueeze(0);
        auto v_heads = v.permute({1, 0, 2}).unsqueeze(0);

        if (debug) {
            auto* ctx = q_heads.context();
            // Build a safe dynamic index: min(token_debug_index_, seq_len - 1)
            // q_heads shape: (1, num_heads, seq_len, head_dim) — seq_len is axis 2
            // shape::dim returns a 1D [1] tensor
            auto seq_len_1d = shape::dim(q_heads, 2);
            auto minus1   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1LL});
            auto one      = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto step     = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto axes     = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2LL});
            auto desired  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {token_debug_index_});
            auto max_valid   = std::make_shared<ov::op::v1::Add>(seq_len_1d, minus1)->output(0);
            auto safe_start  = std::make_shared<ov::op::v1::Minimum>(max_valid, desired)->output(0);
            auto safe_stop   = std::make_shared<ov::op::v1::Add>(safe_start, one)->output(0);
            auto q_slice = std::make_shared<ov::opset13::Slice>(
                q_heads.output(), safe_start, safe_stop, step, axes);
            auto k_slice = std::make_shared<ov::opset13::Slice>(
                k_heads.output(), safe_start, safe_stop, step, axes);
            debug->token_q_heads = Tensor(q_slice->output(0), ctx).squeeze(2).squeeze(0);
            debug->token_k_heads = Tensor(k_slice->output(0), ctx).squeeze(2).squeeze(0);
        }

        auto scores = ops::matmul(q_heads.to(ov::element::f32), k_heads.to(ov::element::f32), false, true) * scaling_;
        if (attention_mask) {
            scores = scores + attention_mask->to(ov::element::f32);
        }
        auto probs = scores.softmax(3).to(q_heads.dtype());
        auto context = ops::matmul(probs, v_heads.to(probs.dtype()));
        if (debug) {
            auto* ctx = probs.context();
            // probs shape: (1, num_heads, seq_len, seq_len) — seq_len is axis 2
            // shape::dim returns a 1D [1] tensor
            auto seq_len_1d = shape::dim(probs, 2);
            auto minus1   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1LL});
            auto one      = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto step     = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1LL});
            auto axes     = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2LL});
            auto desired  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {token_debug_index_});
            auto max_valid   = std::make_shared<ov::op::v1::Add>(seq_len_1d, minus1)->output(0);
            auto safe_start  = std::make_shared<ov::op::v1::Minimum>(max_valid, desired)->output(0);
            auto safe_stop   = std::make_shared<ov::op::v1::Add>(safe_start, one)->output(0);
            auto prob_slice = std::make_shared<ov::opset13::Slice>(
                probs.output(), safe_start, safe_stop, step, axes);
            auto ctx_slice = std::make_shared<ov::opset13::Slice>(
                context.output(), safe_start, safe_stop, step, axes);
            debug->token_attn_prob = Tensor(prob_slice->output(0), ctx).squeeze(2).squeeze(0);
            debug->token_context   = Tensor(ctx_slice->output(0), ctx).squeeze(2).squeeze(0);
        }
        const int64_t attn_out_dim = static_cast<int64_t>(hidden_size_);
        auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
        auto merged_2d = merged.squeeze(0);
        auto proj_weight_t = proj_weight().to(merged_2d.dtype());
        const Tensor* proj_bias_t_ptr = nullptr;
        Tensor proj_bias_t;
        if (const auto* b = proj_bias()) {
            proj_bias_t = b->to(merged_2d.dtype());
            proj_bias_t_ptr = &proj_bias_t;
        }
        auto proj_out = add_bias_if_present(ops::linear(merged_2d, proj_weight_t), proj_bias_t_ptr);
        if (debug) {
            debug->proj_out = proj_out;
        }
        return proj_out;
    }

private:
    const Tensor& qkv_weight() const {
        if (!qkv_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionAttention qkv weight parameter not registered");
        }
        return qkv_weight_param_->value();
    }

    const Tensor* qkv_bias() const {
        return (qkv_bias_param_ && qkv_bias_param_->is_bound()) ? &qkv_bias_param_->value() : nullptr;
    }

    const Tensor& proj_weight() const {
        if (!proj_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionAttention proj weight parameter not registered");
        }
        return proj_weight_param_->value();
    }

    const Tensor* proj_bias() const {
        return (proj_bias_param_ && proj_bias_param_->is_bound()) ? &proj_bias_param_->value() : nullptr;
    }

    Tensor apply_rotary(const Tensor& x,
                        const Tensor& cos,
                        const Tensor& sin) const {
        auto orig_dtype = x.dtype();
        auto x_f = x.to(ov::element::f32);
        auto cos_f = cos.to(ov::element::f32).unsqueeze(1);
        auto sin_f = sin.to(ov::element::f32).unsqueeze(1);

        const int64_t half = static_cast<int64_t>(head_dim_ / 2);
        auto x1 = ops::slice(x_f, 0, half, 1, 2);
        auto x2 = ops::slice(x_f, half, static_cast<int64_t>(head_dim_), 1, 2);
        auto rotated = ops::concat({-x2, x1}, 2);

        auto out = x_f * cos_f + rotated * sin_f;
        return out.to(orig_dtype);
    }

    int32_t hidden_size_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    float scaling_ = 1.0f;
    const int64_t token_debug_index_ = 267;

    WeightParameter* qkv_weight_param_ = nullptr;
    WeightParameter* qkv_bias_param_ = nullptr;
    WeightParameter* proj_weight_param_ = nullptr;
    WeightParameter* proj_bias_param_ = nullptr;
};

class Qwen3OmniVisionMLP : public Module {
public:
    struct DebugOutput {
        Tensor fc1;
        Tensor act;
        Tensor fc2;
    };

    Qwen3OmniVisionMLP(BuilderContext& ctx,
                       const std::string& name,
                       const Qwen3OmniVisionConfig& cfg,
                       Module* parent = nullptr)
        : Module(name, ctx, parent) {
        if (!cfg.hidden_act.empty() && cfg.hidden_act != "gelu_pytorch_tanh") {
            OPENVINO_THROW("Unsupported Qwen3OmniVision MLP activation: ", cfg.hidden_act);
        }
        fc1_weight_param_ = &register_parameter("linear_fc1.weight");
        fc1_bias_param_ = &register_parameter("linear_fc1.bias");
        fc2_weight_param_ = &register_parameter("linear_fc2.weight");
        fc2_bias_param_ = &register_parameter("linear_fc2.bias");
    }

    Tensor forward(const Tensor& hidden_states, DebugOutput* debug = nullptr) const {
        auto fc1_weight_t = fc1_weight().to(hidden_states.dtype());
        const Tensor* fc1_bias_t_ptr = nullptr;
        Tensor fc1_bias_t;
        if (const auto* b = fc1_bias()) {
            fc1_bias_t = b->to(hidden_states.dtype());
            fc1_bias_t_ptr = &fc1_bias_t;
        }
        auto fc1 = add_bias_if_present(ops::linear(hidden_states, fc1_weight_t), fc1_bias_t_ptr);
        auto act = ops::nn::gelu(fc1, true);
        auto fc2_weight_t = fc2_weight().to(act.dtype());
        const Tensor* fc2_bias_t_ptr = nullptr;
        Tensor fc2_bias_t;
        if (const auto* b = fc2_bias()) {
            fc2_bias_t = b->to(act.dtype());
            fc2_bias_t_ptr = &fc2_bias_t;
        }
        auto fc2 = add_bias_if_present(ops::linear(act, fc2_weight_t), fc2_bias_t_ptr);
        if (debug) {
            debug->fc1 = fc1;
            debug->act = act;
            debug->fc2 = fc2;
        }
        return fc2;
    }

private:
    const Tensor& fc1_weight() const {
        if (!fc1_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionMLP fc1 weight parameter not registered");
        }
        return fc1_weight_param_->value();
    }

    const Tensor* fc1_bias() const {
        return (fc1_bias_param_ && fc1_bias_param_->is_bound()) ? &fc1_bias_param_->value() : nullptr;
    }

    const Tensor& fc2_weight() const {
        if (!fc2_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionMLP fc2 weight parameter not registered");
        }
        return fc2_weight_param_->value();
    }

    const Tensor* fc2_bias() const {
        return (fc2_bias_param_ && fc2_bias_param_->is_bound()) ? &fc2_bias_param_->value() : nullptr;
    }

    WeightParameter* fc1_weight_param_ = nullptr;
    WeightParameter* fc1_bias_param_ = nullptr;
    WeightParameter* fc2_weight_param_ = nullptr;
    WeightParameter* fc2_bias_param_ = nullptr;
};

class Qwen3OmniVisionBlock : public Module {
public:
    struct DebugOutput {
        Tensor input;
        Tensor norm1;
        Tensor qkv;
        Tensor attn_out;
        Tensor token_q_heads;
        Tensor token_k_heads;
        Tensor token_attn_prob;
        Tensor token_context;
        Tensor resid1;
        Tensor norm2;
        Tensor mlp_out;
        Tensor mlp_fc1;
        Tensor mlp_act;
        Tensor mlp_fc2;
    };

    Qwen3OmniVisionBlock(BuilderContext& ctx,
                         const std::string& name,
                         const Qwen3OmniVisionConfig& cfg,
                         Module* parent = nullptr)
        : Module(name, ctx, parent),
          attn_(ctx, "attn", cfg, this),
          mlp_(ctx, "mlp", cfg, this) {
        register_module("attn", &attn_);
        register_module("mlp", &mlp_);

        norm1_weight_param_ = &register_parameter("norm1.weight");
        norm1_bias_param_ = &register_parameter("norm1.bias");
        norm2_weight_param_ = &register_parameter("norm2.weight");
        norm2_bias_param_ = &register_parameter("norm2.bias");
    }

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin,
                   const Tensor* attention_mask = nullptr,
                   DebugOutput* debug = nullptr) const {
        auto norm1_weight_t = norm1_weight().to(hidden_states.dtype());
        auto norm1_bias_t = norm1_bias().to(hidden_states.dtype());
        auto norm1 = ops::nn::layer_norm(hidden_states, norm1_weight_t, &norm1_bias_t, eps_, -1);
        Qwen3OmniVisionAttention::DebugOutput attn_debug;
        auto attn_out = attn_.forward(norm1, rotary_cos, rotary_sin, attention_mask, debug ? &attn_debug : nullptr);
        auto resid1 = hidden_states + attn_out;
        auto norm2_weight_t = norm2_weight().to(resid1.dtype());
        auto norm2_bias_t = norm2_bias().to(resid1.dtype());
        auto norm2 = ops::nn::layer_norm(resid1, norm2_weight_t, &norm2_bias_t, eps_, -1);
        Qwen3OmniVisionMLP::DebugOutput mlp_debug;
        auto mlp_out = mlp_.forward(norm2, debug ? &mlp_debug : nullptr);
        if (debug) {
            debug->input = hidden_states;
            debug->norm1 = norm1;
            debug->qkv = attn_debug.qkv;
            debug->attn_out = attn_out;
            debug->token_q_heads = attn_debug.token_q_heads;
            debug->token_k_heads = attn_debug.token_k_heads;
            debug->token_attn_prob = attn_debug.token_attn_prob;
            debug->token_context = attn_debug.token_context;
            debug->resid1 = resid1;
            debug->norm2 = norm2;
            debug->mlp_out = mlp_out;
            debug->mlp_fc1 = mlp_debug.fc1;
            debug->mlp_act = mlp_debug.act;
            debug->mlp_fc2 = mlp_debug.fc2;
        }
        return resid1 + mlp_out;
    }

private:
    const Tensor& norm1_weight() const {
        if (!norm1_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionBlock norm1 weight parameter not registered");
        }
        return norm1_weight_param_->value();
    }

    const Tensor& norm1_bias() const {
        if (!norm1_bias_param_) {
            OPENVINO_THROW("Qwen3OmniVisionBlock norm1 bias parameter not registered");
        }
        return norm1_bias_param_->value();
    }

    const Tensor& norm2_weight() const {
        if (!norm2_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionBlock norm2 weight parameter not registered");
        }
        return norm2_weight_param_->value();
    }

    const Tensor& norm2_bias() const {
        if (!norm2_bias_param_) {
            OPENVINO_THROW("Qwen3OmniVisionBlock norm2 bias parameter not registered");
        }
        return norm2_bias_param_->value();
    }

    Qwen3OmniVisionAttention attn_;
    Qwen3OmniVisionMLP mlp_;
    float eps_ = 1e-6f;

    WeightParameter* norm1_weight_param_ = nullptr;
    WeightParameter* norm1_bias_param_ = nullptr;
    WeightParameter* norm2_weight_param_ = nullptr;
    WeightParameter* norm2_bias_param_ = nullptr;
};

class Qwen3OmniVisionPatchMerger : public Module {
public:
    Qwen3OmniVisionPatchMerger(BuilderContext& ctx,
                               const std::string& name,
                               const Qwen3OmniVisionConfig& cfg,
                               bool use_postshuffle_norm,
                               Module* parent = nullptr)
        : Module(name, ctx, parent),
          hidden_size_(cfg.hidden_size),
          merged_hidden_size_(cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size),
          use_postshuffle_norm_(use_postshuffle_norm) {
        norm_weight_param_ = &register_parameter("norm.weight");
        norm_bias_param_ = &register_parameter("norm.bias");
        fc1_weight_param_ = &register_parameter("linear_fc1.weight");
        fc1_bias_param_ = &register_parameter("linear_fc1.bias");
        fc2_weight_param_ = &register_parameter("linear_fc2.weight");
        fc2_bias_param_ = &register_parameter("linear_fc2.bias");
    }

    Tensor forward(const Tensor& hidden_states) const {
        Tensor x;
        if (use_postshuffle_norm_) {
            auto reshaped = hidden_states.reshape({-1, merged_hidden_size_});
            auto norm_weight_t = norm_weight().to(reshaped.dtype());
            auto norm_bias_t = norm_bias().to(reshaped.dtype());
            x = ops::nn::layer_norm(reshaped, norm_weight_t, &norm_bias_t, eps_, -1);
        } else {
            auto norm_weight_t = norm_weight().to(hidden_states.dtype());
            auto norm_bias_t = norm_bias().to(hidden_states.dtype());
            auto normed = ops::nn::layer_norm(hidden_states, norm_weight_t, &norm_bias_t, eps_, -1);
            x = normed.reshape({-1, merged_hidden_size_});
        }
        auto fc1_weight_t = fc1_weight().to(x.dtype());
        const Tensor* fc1_bias_t_ptr = nullptr;
        Tensor fc1_bias_t;
        if (const auto* b = fc1_bias()) {
            fc1_bias_t = b->to(x.dtype());
            fc1_bias_t_ptr = &fc1_bias_t;
        }
        auto fc1 = add_bias_if_present(ops::linear(x, fc1_weight_t), fc1_bias_t_ptr);
        auto act = ops::nn::gelu(fc1, true);
        auto fc2_weight_t = fc2_weight().to(act.dtype());
        const Tensor* fc2_bias_t_ptr = nullptr;
        Tensor fc2_bias_t;
        if (const auto* b = fc2_bias()) {
            fc2_bias_t = b->to(act.dtype());
            fc2_bias_t_ptr = &fc2_bias_t;
        }
        return add_bias_if_present(ops::linear(act, fc2_weight_t), fc2_bias_t_ptr);
    }

private:
    const Tensor& norm_weight() const {
        if (!norm_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionPatchMerger norm weight parameter not registered");
        }
        return norm_weight_param_->value();
    }

    const Tensor& norm_bias() const {
        if (!norm_bias_param_) {
            OPENVINO_THROW("Qwen3OmniVisionPatchMerger norm bias parameter not registered");
        }
        return norm_bias_param_->value();
    }

    const Tensor& fc1_weight() const {
        if (!fc1_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionPatchMerger fc1 weight parameter not registered");
        }
        return fc1_weight_param_->value();
    }

    const Tensor* fc1_bias() const {
        return (fc1_bias_param_ && fc1_bias_param_->is_bound()) ? &fc1_bias_param_->value() : nullptr;
    }

    const Tensor& fc2_weight() const {
        if (!fc2_weight_param_) {
            OPENVINO_THROW("Qwen3OmniVisionPatchMerger fc2 weight parameter not registered");
        }
        return fc2_weight_param_->value();
    }

    const Tensor* fc2_bias() const {
        return (fc2_bias_param_ && fc2_bias_param_->is_bound()) ? &fc2_bias_param_->value() : nullptr;
    }

    int32_t hidden_size_ = 0;
    int32_t merged_hidden_size_ = 0;
    bool use_postshuffle_norm_ = false;
    float eps_ = 1e-6f;

    WeightParameter* norm_weight_param_ = nullptr;
    WeightParameter* norm_bias_param_ = nullptr;
    WeightParameter* fc1_weight_param_ = nullptr;
    WeightParameter* fc1_bias_param_ = nullptr;
    WeightParameter* fc2_weight_param_ = nullptr;
    WeightParameter* fc2_bias_param_ = nullptr;
};

class Qwen3OmniVisionModel : public Module {
public:
    Qwen3OmniVisionModel(BuilderContext& ctx,
                         const Qwen3OmniVisionConfig& cfg,
                         Module* parent = nullptr)
        : Module("visual", ctx, parent),
          cfg_(cfg),
          patch_embed_(ctx, "patch_embed", cfg, this),
          blocks_(),
          merger_(ctx, "merger", cfg, false, this),
          deepstack_mergers_(),
          deepstack_indexes_(cfg.deepstack_visual_indexes) {
        register_module("patch_embed", &patch_embed_);
        register_module("merger", &merger_);

        blocks_.reserve(static_cast<size_t>(cfg.depth));
        for (int32_t i = 0; i < cfg.depth; ++i) {
            const std::string block_name = std::string("blocks.") + std::to_string(i);
            blocks_.emplace_back(ctx, block_name, cfg, this);
            register_module(block_name, &blocks_.back());
        }

        deepstack_mergers_.reserve(deepstack_indexes_.size());
        for (size_t i = 0; i < deepstack_indexes_.size(); ++i) {
            const std::string merger_name = std::string("deepstack_merger_list.") + std::to_string(i);
            deepstack_mergers_.emplace_back(ctx, merger_name, cfg, true, this);
            register_module(merger_name, &deepstack_mergers_.back());
        }
    }

    Qwen3OmniVisionOutput forward(const Tensor& pixel_values,
                                  const Tensor& grid_thw,
                                  const Tensor& pos_embeds,
                                  const Tensor& rotary_cos,
                                  const Tensor& rotary_sin,
                                  const Tensor* attention_mask = nullptr) {
        (void)grid_thw;
        Qwen3OmniVisionPatchEmbed::DebugOutput patch_debug;
        auto hidden_states = patch_embed_.forward(pixel_values, &patch_debug);
        Qwen3OmniVisionOutput output;
        output.preblock0_patch_embed_input_5d = patch_debug.input_5d;
        output.preblock0_patch_embed_conv_5d = patch_debug.conv_out_5d;
        output.preblock0_patch_embed = hidden_states;
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype());
        output.preblock0_with_pos = hidden_states;
        output.deepstack_embeds.reserve(deepstack_mergers_.size());
        output.block_hiddens.reserve(blocks_.size());
        output.token267_head_debugs.reserve(4);
        const size_t debug_block_idx_0 = 0;
        const size_t debug_block_idx_16 = 16;
        const size_t debug_block_idx_25 = 25;
        const size_t debug_block_idx_26 = 26;
        const std::vector<size_t> token_debug_block_indices = {16, 20, 24, 26};

        for (size_t layer_idx = 0; layer_idx < blocks_.size(); ++layer_idx) {
            const bool needs_full_debug =
                (layer_idx == debug_block_idx_0 || layer_idx == debug_block_idx_25 || layer_idx == debug_block_idx_26);
            const bool needs_token_debug =
                std::find(token_debug_block_indices.begin(), token_debug_block_indices.end(), layer_idx) != token_debug_block_indices.end();
            if (needs_full_debug || needs_token_debug) {
                Qwen3OmniVisionBlock::DebugOutput block_debug;
                hidden_states = blocks_[layer_idx].forward(hidden_states, rotary_cos, rotary_sin, attention_mask, &block_debug);
                if (layer_idx == debug_block_idx_0) {
                    output.block0_input = block_debug.input;
                    output.block0_norm1 = block_debug.norm1;
                    output.block0_qkv = block_debug.qkv;
                    output.block0_attn_out = block_debug.attn_out;
                    output.block0_resid1 = block_debug.resid1;
                    output.block0_norm2 = block_debug.norm2;
                    output.block0_mlp_out = block_debug.mlp_out;
                    output.block0_mlp_fc1 = block_debug.mlp_fc1;
                    output.block0_mlp_act = block_debug.mlp_act;
                    output.block0_mlp_fc2 = block_debug.mlp_fc2;
                }
                if (layer_idx == debug_block_idx_16) {
                    output.block16_input = block_debug.input;
                    output.block16_norm1 = block_debug.norm1;
                    output.block16_qkv = block_debug.qkv;
                }
                if (needs_token_debug) {
                    Qwen3OmniVisionOutput::TokenHeadDebug token_debug;
                    token_debug.block_idx = static_cast<int32_t>(layer_idx);
                    token_debug.q_heads = block_debug.token_q_heads;
                    token_debug.k_heads = block_debug.token_k_heads;
                    token_debug.attn_prob = block_debug.token_attn_prob;
                    token_debug.context = block_debug.token_context;
                    output.token267_head_debugs.push_back(token_debug);
                }
                if (layer_idx == debug_block_idx_25) {
                    output.block25_norm1 = block_debug.norm1;
                    output.block25_qkv = block_debug.qkv;
                    output.block25_attn_out = block_debug.attn_out;
                    output.block25_resid1 = block_debug.resid1;
                    output.block25_norm2 = block_debug.norm2;
                    output.block25_mlp_out = block_debug.mlp_out;
                    output.block25_mlp_fc1 = block_debug.mlp_fc1;
                    output.block25_mlp_act = block_debug.mlp_act;
                    output.block25_mlp_fc2 = block_debug.mlp_fc2;
                } else if (layer_idx == debug_block_idx_26) {
                    output.block26_norm1 = block_debug.norm1;
                    output.block26_qkv = block_debug.qkv;
                    output.block26_attn_out = block_debug.attn_out;
                    output.block26_resid1 = block_debug.resid1;
                    output.block26_norm2 = block_debug.norm2;
                    output.block26_mlp_out = block_debug.mlp_out;
                    output.block26_mlp_fc1 = block_debug.mlp_fc1;
                    output.block26_mlp_act = block_debug.mlp_act;
                    output.block26_mlp_fc2 = block_debug.mlp_fc2;
                }
            } else {
                hidden_states = blocks_[layer_idx].forward(hidden_states, rotary_cos, rotary_sin, attention_mask);
            }
            output.block_hiddens.push_back(hidden_states);
            auto it = std::find(deepstack_indexes_.begin(),
                                deepstack_indexes_.end(),
                                static_cast<int32_t>(layer_idx));
            if (it != deepstack_indexes_.end()) {
                size_t ds_idx = static_cast<size_t>(std::distance(deepstack_indexes_.begin(), it));
                if (ds_idx < deepstack_mergers_.size()) {
                    output.deepstack_embeds.push_back(deepstack_mergers_[ds_idx].forward(hidden_states));
                }
            }
        }

        output.visual_embeds = merger_.forward(hidden_states);
        return output;
    }

private:
    Qwen3OmniVisionConfig cfg_;
    Qwen3OmniVisionPatchEmbed patch_embed_;
    std::vector<Qwen3OmniVisionBlock> blocks_;
    Qwen3OmniVisionPatchMerger merger_;
    std::vector<Qwen3OmniVisionPatchMerger> deepstack_mergers_;
    std::vector<int32_t> deepstack_indexes_;
};

std::shared_ptr<ov::Model> create_qwen3_omni_vision_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    PrefixMappedWeightSource thinker_source(source, "thinker.");

    BuilderContext ctx;
    Qwen3OmniVisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});
    model.packed_mapping().rules.push_back({"visual.merger.ln_q.", "visual.merger.norm.", 0});
    model.packed_mapping().rules.push_back({"visual.merger.mlp.0.", "visual.merger.linear_fc1.", 0});
    model.packed_mapping().rules.push_back({"visual.merger.mlp.2.", "visual.merger.linear_fc2.", 0});

    const size_t deepstack_count = cfg.vision.deepstack_visual_indexes.size();
    for (size_t i = 0; i < deepstack_count; ++i) {
        const std::string src_base = std::string("visual.merger_list.") + std::to_string(i) + ".";
        const std::string dst_base = std::string("visual.deepstack_merger_list.") + std::to_string(i) + ".";
        model.packed_mapping().rules.push_back({src_base + "ln_q.", dst_base + "norm.", 0});
        model.packed_mapping().rules.push_back({src_base + "mlp.0.", dst_base + "linear_fc1.", 0});
        model.packed_mapping().rules.push_back({src_base + "mlp.2.", dst_base + "linear_fc2.", 0});
    }

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, thinker_source, finalizer, options);
    (void)report;

    auto pixel_values = ctx.parameter(Qwen3VLVisionIO::kPixelValues,
                                      ov::element::f32,
                                      ov::PartialShape{-1,
                                                       cfg.vision.in_channels,
                                                       cfg.vision.temporal_patch_size,
                                                       cfg.vision.patch_size,
                                                       cfg.vision.patch_size});
    auto grid_thw = ctx.parameter(Qwen3VLVisionIO::kGridThw,
                                  ov::element::i64,
                                  ov::PartialShape{-1, 3});
    auto pos_embeds = ctx.parameter(Qwen3VLVisionIO::kPosEmbeds,
                                    ov::element::f32,
                                    ov::PartialShape{-1, cfg.vision.hidden_size});
    auto rotary_cos = ctx.parameter(Qwen3VLVisionIO::kRotaryCos,
                                    ov::element::f32,
                                    ov::PartialShape{-1, cfg.vision.hidden_size / cfg.vision.num_heads});
    auto rotary_sin = ctx.parameter(Qwen3VLVisionIO::kRotarySin,
                                    ov::element::f32,
                                    ov::PartialShape{-1, cfg.vision.hidden_size / cfg.vision.num_heads});
    auto attention_mask = ctx.parameter("attention_mask",
                                        ov::element::f32,
                                        ov::PartialShape{1, 1, -1, -1});

    auto output = model.forward(pixel_values, grid_thw, pos_embeds, rotary_cos, rotary_sin, &attention_mask);

    ov::OutputVector results;
    results.reserve(1 + output.deepstack_embeds.size() + output.block_hiddens.size());

    auto visual = std::make_shared<ov::op::v0::Result>(output.visual_embeds.output());
    set_name(visual, Qwen3VLVisionIO::kVisualEmbeds);
    results.push_back(visual->output(0));

    for (size_t i = 0; i < output.deepstack_embeds.size(); ++i) {
        auto ds_result = std::make_shared<ov::op::v0::Result>(output.deepstack_embeds[i].output());
        std::string name = std::string(Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i);
        set_name(ds_result, name);
        results.push_back(ds_result->output(0));
    }

    for (size_t i = 0; i < output.block_hiddens.size(); ++i) {
        auto blk_result = std::make_shared<ov::op::v0::Result>(output.block_hiddens[i].output());
        std::string name = std::string("vision_block_hidden.") + std::to_string(i);
        set_name(blk_result, name);
        results.push_back(blk_result->output(0));
    }

    for (const auto& token_debug : output.token267_head_debugs) {
        const std::string base = std::string("vision_block") + std::to_string(token_debug.block_idx) + ".token267.";

        auto q_heads = std::make_shared<ov::op::v0::Result>(token_debug.q_heads.output());
        set_name(q_heads, base + "q_heads");
        results.push_back(q_heads->output(0));

        auto k_heads = std::make_shared<ov::op::v0::Result>(token_debug.k_heads.output());
        set_name(k_heads, base + "k_heads");
        results.push_back(k_heads->output(0));

        auto attn_prob = std::make_shared<ov::op::v0::Result>(token_debug.attn_prob.output());
        set_name(attn_prob, base + "attn_prob");
        results.push_back(attn_prob->output(0));

        auto context = std::make_shared<ov::op::v0::Result>(token_debug.context.output());
        set_name(context, base + "context");
        results.push_back(context->output(0));
    }

    if (cfg.vision.depth > 0) {
        auto preblock0_patch_embed_input_5d =
            std::make_shared<ov::op::v0::Result>(output.preblock0_patch_embed_input_5d.output());
        set_name(preblock0_patch_embed_input_5d, "vision_preblock0_patch_embed_input_5d");
        results.push_back(preblock0_patch_embed_input_5d->output(0));

        auto preblock0_patch_embed_conv_5d =
            std::make_shared<ov::op::v0::Result>(output.preblock0_patch_embed_conv_5d.output());
        set_name(preblock0_patch_embed_conv_5d, "vision_preblock0_patch_embed_conv_5d");
        results.push_back(preblock0_patch_embed_conv_5d->output(0));

        auto preblock0_patch_embed = std::make_shared<ov::op::v0::Result>(output.preblock0_patch_embed.output());
        set_name(preblock0_patch_embed, "vision_preblock0_patch_embed");
        results.push_back(preblock0_patch_embed->output(0));

        auto preblock0_with_pos = std::make_shared<ov::op::v0::Result>(output.preblock0_with_pos.output());
        set_name(preblock0_with_pos, "vision_preblock0_with_pos");
        results.push_back(preblock0_with_pos->output(0));

        auto block0_input = std::make_shared<ov::op::v0::Result>(output.block0_input.output());
        set_name(block0_input, "vision_block0.input");
        results.push_back(block0_input->output(0));

        auto block0_norm1 = std::make_shared<ov::op::v0::Result>(output.block0_norm1.output());
        set_name(block0_norm1, "vision_block0.norm1");
        results.push_back(block0_norm1->output(0));

        auto block0_qkv = std::make_shared<ov::op::v0::Result>(output.block0_qkv.output());
        set_name(block0_qkv, "vision_block0.qkv");
        results.push_back(block0_qkv->output(0));

        auto block0_attn = std::make_shared<ov::op::v0::Result>(output.block0_attn_out.output());
        set_name(block0_attn, "vision_block0.attn_out");
        results.push_back(block0_attn->output(0));

        auto block0_resid1 = std::make_shared<ov::op::v0::Result>(output.block0_resid1.output());
        set_name(block0_resid1, "vision_block0.resid1");
        results.push_back(block0_resid1->output(0));

        auto block0_norm2 = std::make_shared<ov::op::v0::Result>(output.block0_norm2.output());
        set_name(block0_norm2, "vision_block0.norm2");
        results.push_back(block0_norm2->output(0));

        auto block0_mlp = std::make_shared<ov::op::v0::Result>(output.block0_mlp_out.output());
        set_name(block0_mlp, "vision_block0.mlp_out");
        results.push_back(block0_mlp->output(0));

        auto block0_mlp_fc1 = std::make_shared<ov::op::v0::Result>(output.block0_mlp_fc1.output());
        set_name(block0_mlp_fc1, "vision_block0.mlp_fc1");
        results.push_back(block0_mlp_fc1->output(0));

        auto block0_mlp_act = std::make_shared<ov::op::v0::Result>(output.block0_mlp_act.output());
        set_name(block0_mlp_act, "vision_block0.mlp_act");
        results.push_back(block0_mlp_act->output(0));

        auto block0_mlp_fc2 = std::make_shared<ov::op::v0::Result>(output.block0_mlp_fc2.output());
        set_name(block0_mlp_fc2, "vision_block0.mlp_fc2");
        results.push_back(block0_mlp_fc2->output(0));
    }

    if (cfg.vision.depth > 16) {
        auto block16_input = std::make_shared<ov::op::v0::Result>(output.block16_input.output());
        set_name(block16_input, "vision_block16.input");
        results.push_back(block16_input->output(0));

        auto block16_norm1 = std::make_shared<ov::op::v0::Result>(output.block16_norm1.output());
        set_name(block16_norm1, "vision_block16.norm1");
        results.push_back(block16_norm1->output(0));

        auto block16_qkv = std::make_shared<ov::op::v0::Result>(output.block16_qkv.output());
        set_name(block16_qkv, "vision_block16.qkv");
        results.push_back(block16_qkv->output(0));
    }

    if (cfg.vision.depth > 25) {
        auto block25_norm1 = std::make_shared<ov::op::v0::Result>(output.block25_norm1.output());
        set_name(block25_norm1, "vision_block25.norm1");
        results.push_back(block25_norm1->output(0));

        auto block25_qkv = std::make_shared<ov::op::v0::Result>(output.block25_qkv.output());
        set_name(block25_qkv, "vision_block25.qkv");
        results.push_back(block25_qkv->output(0));

        auto block25_attn = std::make_shared<ov::op::v0::Result>(output.block25_attn_out.output());
        set_name(block25_attn, "vision_block25.attn_out");
        results.push_back(block25_attn->output(0));

        auto block25_resid1 = std::make_shared<ov::op::v0::Result>(output.block25_resid1.output());
        set_name(block25_resid1, "vision_block25.resid1");
        results.push_back(block25_resid1->output(0));

        auto block25_norm2 = std::make_shared<ov::op::v0::Result>(output.block25_norm2.output());
        set_name(block25_norm2, "vision_block25.norm2");
        results.push_back(block25_norm2->output(0));

        auto block25_mlp = std::make_shared<ov::op::v0::Result>(output.block25_mlp_out.output());
        set_name(block25_mlp, "vision_block25.mlp_out");
        results.push_back(block25_mlp->output(0));

        auto block25_mlp_fc1 = std::make_shared<ov::op::v0::Result>(output.block25_mlp_fc1.output());
        set_name(block25_mlp_fc1, "vision_block25.mlp_fc1");
        results.push_back(block25_mlp_fc1->output(0));

        auto block25_mlp_act = std::make_shared<ov::op::v0::Result>(output.block25_mlp_act.output());
        set_name(block25_mlp_act, "vision_block25.mlp_act");
        results.push_back(block25_mlp_act->output(0));

        auto block25_mlp_fc2 = std::make_shared<ov::op::v0::Result>(output.block25_mlp_fc2.output());
        set_name(block25_mlp_fc2, "vision_block25.mlp_fc2");
        results.push_back(block25_mlp_fc2->output(0));
    }

    if (cfg.vision.depth > 26) {
        auto block26_norm1 = std::make_shared<ov::op::v0::Result>(output.block26_norm1.output());
        set_name(block26_norm1, "vision_block26.norm1");
        results.push_back(block26_norm1->output(0));

        auto block26_qkv = std::make_shared<ov::op::v0::Result>(output.block26_qkv.output());
        set_name(block26_qkv, "vision_block26.qkv");
        results.push_back(block26_qkv->output(0));

        auto block26_attn = std::make_shared<ov::op::v0::Result>(output.block26_attn_out.output());
        set_name(block26_attn, "vision_block26.attn_out");
        results.push_back(block26_attn->output(0));

        auto block26_resid1 = std::make_shared<ov::op::v0::Result>(output.block26_resid1.output());
        set_name(block26_resid1, "vision_block26.resid1");
        results.push_back(block26_resid1->output(0));

        auto block26_norm2 = std::make_shared<ov::op::v0::Result>(output.block26_norm2.output());
        set_name(block26_norm2, "vision_block26.norm2");
        results.push_back(block26_norm2->output(0));

        auto block26_mlp = std::make_shared<ov::op::v0::Result>(output.block26_mlp_out.output());
        set_name(block26_mlp, "vision_block26.mlp_out");
        results.push_back(block26_mlp->output(0));

        auto block26_mlp_fc1 = std::make_shared<ov::op::v0::Result>(output.block26_mlp_fc1.output());
        set_name(block26_mlp_fc1, "vision_block26.mlp_fc1");
        results.push_back(block26_mlp_fc1->output(0));

        auto block26_mlp_act = std::make_shared<ov::op::v0::Result>(output.block26_mlp_act.output());
        set_name(block26_mlp_act, "vision_block26.mlp_act");
        results.push_back(block26_mlp_act->output(0));

        auto block26_mlp_fc2 = std::make_shared<ov::op::v0::Result>(output.block26_mlp_fc2.output());
        set_name(block26_mlp_fc2, "vision_block26.mlp_fc2");
        results.push_back(block26_mlp_fc2->output(0));
    }

    return ctx.build_model(results);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
