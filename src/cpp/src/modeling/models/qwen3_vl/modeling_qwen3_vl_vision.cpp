// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl/modeling_qwen3_vl_vision.hpp"

#include <algorithm>
#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
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

Qwen3VLVisionPatchEmbed::Qwen3VLVisionPatchEmbed(BuilderContext& ctx,
                                                 const std::string& name,
                                                 const Qwen3VLVisionConfig& cfg,
                                                 Module* parent)
    : Module(name, ctx, parent),
      in_channels_(cfg.in_channels),
      patch_size_(cfg.patch_size),
      temporal_patch_size_(cfg.temporal_patch_size),
      embed_dim_(cfg.hidden_size) {
    weight_param_ = &register_parameter("proj.weight");
    bias_param_ = &register_parameter("proj.bias");
}

const Tensor& Qwen3VLVisionPatchEmbed::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionPatchEmbed weight parameter not registered");
    }
    return weight_param_->value();
}

const Tensor* Qwen3VLVisionPatchEmbed::bias() const {
    return (bias_param_ && bias_param_->is_bound()) ? &bias_param_->value() : nullptr;
}

Tensor Qwen3VLVisionPatchEmbed::forward(const Tensor& pixel_values) const {
    auto x = pixel_values.reshape({0, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    x = x.to(weight().dtype());
    const std::vector<int64_t> strides = {temporal_patch_size_, patch_size_, patch_size_};
    const std::vector<int64_t> pads = {0, 0, 0};
    Tensor conv;
    if (const auto* b = bias()) {
        conv = ops::nn::conv3d(x, weight(), *b, strides, pads, pads);
    } else {
        conv = ops::nn::conv3d(x, weight(), strides, pads, pads);
    }
    return conv.reshape({0, embed_dim_});
}

Qwen3VLVisionAttention::Qwen3VLVisionAttention(BuilderContext& ctx,
                                               const std::string& name,
                                               const Qwen3VLVisionConfig& cfg,
                                               Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      num_heads_(cfg.num_heads),
      head_dim_(cfg.head_dim()),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (hidden_size_ <= 0 || num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3VLVisionAttention configuration");
    }
    if (hidden_size_ != num_heads_ * head_dim_) {
        OPENVINO_THROW("Qwen3VLVisionAttention hidden_size must equal num_heads * head_dim");
    }
    if (head_dim_ % 2 != 0) {
        OPENVINO_THROW("Qwen3VLVisionAttention head_dim must be even for rotary embeddings");
    }
    qkv_weight_param_ = &register_parameter("qkv.weight");
    qkv_bias_param_ = &register_parameter("qkv.bias");
    proj_weight_param_ = &register_parameter("proj.weight");
    proj_bias_param_ = &register_parameter("proj.bias");
}

const Tensor& Qwen3VLVisionAttention::qkv_weight() const {
    if (!qkv_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionAttention qkv weight parameter not registered");
    }
    return qkv_weight_param_->value();
}

const Tensor* Qwen3VLVisionAttention::qkv_bias() const {
    return (qkv_bias_param_ && qkv_bias_param_->is_bound()) ? &qkv_bias_param_->value() : nullptr;
}

const Tensor& Qwen3VLVisionAttention::proj_weight() const {
    if (!proj_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionAttention proj weight parameter not registered");
    }
    return proj_weight_param_->value();
}

const Tensor* Qwen3VLVisionAttention::proj_bias() const {
    return (proj_bias_param_ && proj_bias_param_->is_bound()) ? &proj_bias_param_->value() : nullptr;
}

Tensor Qwen3VLVisionAttention::apply_rotary(const Tensor& x,
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

Tensor Qwen3VLVisionAttention::forward(const Tensor& hidden_states,
                                       const Tensor& rotary_cos,
                                       const Tensor& rotary_sin) const {
    auto qkv = add_bias_if_present(ops::linear(hidden_states, qkv_weight()), qkv_bias());
    auto qkv_reshaped = qkv.reshape({0, 3, num_heads_, head_dim_});
    auto q = ops::slice(qkv_reshaped, 0, 1, 1, 1).squeeze(1);
    auto k = ops::slice(qkv_reshaped, 1, 2, 1, 1).squeeze(1);
    auto v = ops::slice(qkv_reshaped, 2, 3, 1, 1).squeeze(1);

    auto q_rot = apply_rotary(q, rotary_cos, rotary_sin);
    auto k_rot = apply_rotary(k, rotary_cos, rotary_sin);

    auto q_heads = q_rot.permute({1, 0, 2}).unsqueeze(0);
    auto k_heads = k_rot.permute({1, 0, 2}).unsqueeze(0);
    auto v_heads = v.permute({1, 0, 2}).unsqueeze(0);

    auto* policy = &ctx().op_policy();
    auto context = ops::llm::sdpa(q_heads, k_heads, v_heads, scaling_, 3, nullptr, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(hidden_size_);
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto merged_2d = merged.squeeze(0);
    auto out = add_bias_if_present(ops::linear(merged_2d, proj_weight()), proj_bias());
    return out;
}

Qwen3VLVisionMLP::Qwen3VLVisionMLP(BuilderContext& ctx,
                                   const std::string& name,
                                   const Qwen3VLVisionConfig& cfg,
                                   Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "gelu_pytorch_tanh") {
        OPENVINO_THROW("Unsupported Qwen3VLVision MLP activation: ", cfg.hidden_act);
    }
    fc1_weight_param_ = &register_parameter("linear_fc1.weight");
    fc1_bias_param_ = &register_parameter("linear_fc1.bias");
    fc2_weight_param_ = &register_parameter("linear_fc2.weight");
    fc2_bias_param_ = &register_parameter("linear_fc2.bias");
}

const Tensor& Qwen3VLVisionMLP::fc1_weight() const {
    if (!fc1_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionMLP fc1 weight parameter not registered");
    }
    return fc1_weight_param_->value();
}

const Tensor* Qwen3VLVisionMLP::fc1_bias() const {
    return (fc1_bias_param_ && fc1_bias_param_->is_bound()) ? &fc1_bias_param_->value() : nullptr;
}

const Tensor& Qwen3VLVisionMLP::fc2_weight() const {
    if (!fc2_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionMLP fc2 weight parameter not registered");
    }
    return fc2_weight_param_->value();
}

const Tensor* Qwen3VLVisionMLP::fc2_bias() const {
    return (fc2_bias_param_ && fc2_bias_param_->is_bound()) ? &fc2_bias_param_->value() : nullptr;
}

Tensor Qwen3VLVisionMLP::forward(const Tensor& hidden_states) const {
    auto fc1 = add_bias_if_present(ops::linear(hidden_states, fc1_weight()), fc1_bias());
    auto act = ops::nn::gelu(fc1, true);
    return add_bias_if_present(ops::linear(act, fc2_weight()), fc2_bias());
}

Qwen3VLVisionBlock::Qwen3VLVisionBlock(BuilderContext& ctx,
                                       const std::string& name,
                                       const Qwen3VLVisionConfig& cfg,
                                       Module* parent)
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

const Tensor& Qwen3VLVisionBlock::norm1_weight() const {
    if (!norm1_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionBlock norm1 weight parameter not registered");
    }
    return norm1_weight_param_->value();
}

const Tensor& Qwen3VLVisionBlock::norm1_bias() const {
    if (!norm1_bias_param_) {
        OPENVINO_THROW("Qwen3VLVisionBlock norm1 bias parameter not registered");
    }
    return norm1_bias_param_->value();
}

const Tensor& Qwen3VLVisionBlock::norm2_weight() const {
    if (!norm2_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionBlock norm2 weight parameter not registered");
    }
    return norm2_weight_param_->value();
}

const Tensor& Qwen3VLVisionBlock::norm2_bias() const {
    if (!norm2_bias_param_) {
        OPENVINO_THROW("Qwen3VLVisionBlock norm2 bias parameter not registered");
    }
    return norm2_bias_param_->value();
}

Tensor Qwen3VLVisionBlock::forward(const Tensor& hidden_states,
                                   const Tensor& rotary_cos,
                                   const Tensor& rotary_sin) const {
    auto norm1 = ops::nn::layer_norm(hidden_states, norm1_weight(), &norm1_bias(), eps_, -1);
    auto attn_out = attn_.forward(norm1, rotary_cos, rotary_sin);
    auto resid1 = hidden_states + attn_out;
    auto norm2 = ops::nn::layer_norm(resid1, norm2_weight(), &norm2_bias(), eps_, -1);
    auto mlp_out = mlp_.forward(norm2);
    return resid1 + mlp_out;
}

Qwen3VLVisionPatchMerger::Qwen3VLVisionPatchMerger(BuilderContext& ctx,
                                                   const std::string& name,
                                                   const Qwen3VLVisionConfig& cfg,
                                                   bool use_postshuffle_norm,
                                                   Module* parent)
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

const Tensor& Qwen3VLVisionPatchMerger::norm_weight() const {
    if (!norm_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionPatchMerger norm weight parameter not registered");
    }
    return norm_weight_param_->value();
}

const Tensor& Qwen3VLVisionPatchMerger::norm_bias() const {
    if (!norm_bias_param_) {
        OPENVINO_THROW("Qwen3VLVisionPatchMerger norm bias parameter not registered");
    }
    return norm_bias_param_->value();
}

const Tensor& Qwen3VLVisionPatchMerger::fc1_weight() const {
    if (!fc1_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionPatchMerger fc1 weight parameter not registered");
    }
    return fc1_weight_param_->value();
}

const Tensor* Qwen3VLVisionPatchMerger::fc1_bias() const {
    return (fc1_bias_param_ && fc1_bias_param_->is_bound()) ? &fc1_bias_param_->value() : nullptr;
}

const Tensor& Qwen3VLVisionPatchMerger::fc2_weight() const {
    if (!fc2_weight_param_) {
        OPENVINO_THROW("Qwen3VLVisionPatchMerger fc2 weight parameter not registered");
    }
    return fc2_weight_param_->value();
}

const Tensor* Qwen3VLVisionPatchMerger::fc2_bias() const {
    return (fc2_bias_param_ && fc2_bias_param_->is_bound()) ? &fc2_bias_param_->value() : nullptr;
}

Tensor Qwen3VLVisionPatchMerger::forward(const Tensor& hidden_states) const {
    Tensor x;
    if (use_postshuffle_norm_) {
        auto reshaped = hidden_states.reshape({-1, merged_hidden_size_});
        x = ops::nn::layer_norm(reshaped, norm_weight(), &norm_bias(), eps_, -1);
    } else {
        auto normed = ops::nn::layer_norm(hidden_states, norm_weight(), &norm_bias(), eps_, -1);
        x = normed.reshape({-1, merged_hidden_size_});
    }
    auto fc1 = add_bias_if_present(ops::linear(x, fc1_weight()), fc1_bias());
    auto act = ops::nn::gelu(fc1, true);
    return add_bias_if_present(ops::linear(act, fc2_weight()), fc2_bias());
}

Qwen3VLVisionModel::Qwen3VLVisionModel(BuilderContext& ctx,
                                       const Qwen3VLVisionConfig& cfg,
                                       Module* parent)
    : Module(Qwen3VLModuleNames::kVision, ctx, parent),
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
        blocks_.emplace_back(ctx, Qwen3VLModuleNames::vision_block(i), cfg, this);
        register_module(Qwen3VLModuleNames::vision_block(i), &blocks_.back());
    }

    deepstack_mergers_.reserve(deepstack_indexes_.size());
    for (size_t i = 0; i < deepstack_indexes_.size(); ++i) {
        deepstack_mergers_.emplace_back(ctx, Qwen3VLModuleNames::deepstack_merger(static_cast<int32_t>(i)),
                                        cfg, true, this);
        register_module(Qwen3VLModuleNames::deepstack_merger(static_cast<int32_t>(i)),
                        &deepstack_mergers_.back());
    }
}

Qwen3VLVisionOutput Qwen3VLVisionModel::forward(const Tensor& pixel_values,
                                                const Tensor& grid_thw,
                                                const Tensor& pos_embeds,
                                                const Tensor& rotary_cos,
                                                const Tensor& rotary_sin) {
    (void)grid_thw;
    auto hidden_states = patch_embed_.forward(pixel_values);
    hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype());

    Qwen3VLVisionOutput output;
    output.deepstack_embeds.reserve(deepstack_mergers_.size());

    for (size_t layer_idx = 0; layer_idx < blocks_.size(); ++layer_idx) {
        hidden_states = blocks_[layer_idx].forward(hidden_states, rotary_cos, rotary_sin);
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

Qwen3VLVisionPatchEmbed& Qwen3VLVisionModel::patch_embed() {
    return patch_embed_;
}

Qwen3VLVisionPatchMerger& Qwen3VLVisionModel::merger() {
    return merger_;
}

std::shared_ptr<ov::Model> create_qwen3_vl_vision_model(
    const Qwen3VLConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3VLVisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
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
                                    ov::PartialShape{-1, cfg.vision.head_dim()});
    auto rotary_sin = ctx.parameter(Qwen3VLVisionIO::kRotarySin,
                                    ov::element::f32,
                                    ov::PartialShape{-1, cfg.vision.head_dim()});

    auto output = model.forward(pixel_values, grid_thw, pos_embeds, rotary_cos, rotary_sin);

    ov::OutputVector results;
    results.reserve(1 + output.deepstack_embeds.size());

    auto visual = std::make_shared<ov::op::v0::Result>(output.visual_embeds.output());
    set_name(visual, Qwen3VLVisionIO::kVisualEmbeds);
    results.push_back(visual->output(0));

    for (size_t i = 0; i < output.deepstack_embeds.size(); ++i) {
        auto ds_result = std::make_shared<ov::op::v0::Result>(output.deepstack_embeds[i].output());
        std::string name = std::string(Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i);
        set_name(ds_result, name);
        results.push_back(ds_result->output(0));
    }

    return ctx.build_model(results);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
