// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3VLVisionOutput {
    Tensor visual_embeds;
    std::vector<Tensor> deepstack_embeds;
};

class Qwen3VLVisionPatchEmbed : public Module {
public:
    Qwen3VLVisionPatchEmbed(BuilderContext& ctx, const std::string& name, const Qwen3VLVisionConfig& cfg,
                            Module* parent = nullptr);

    Tensor forward(const Tensor& pixel_values) const;

private:
    const Tensor& weight() const;
    const Tensor* bias() const;

    int32_t in_channels_ = 0;
    int32_t patch_size_ = 0;
    int32_t temporal_patch_size_ = 0;
    int32_t embed_dim_ = 0;

    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
};

class Qwen3VLVisionAttention : public Module {
public:
    Qwen3VLVisionAttention(BuilderContext& ctx, const std::string& name, const Qwen3VLVisionConfig& cfg,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin) const;

private:
    const Tensor& qkv_weight() const;
    const Tensor* qkv_bias() const;
    const Tensor& proj_weight() const;
    const Tensor* proj_bias() const;

    Tensor apply_rotary(const Tensor& x,
                        const Tensor& cos,
                        const Tensor& sin) const;

    int32_t hidden_size_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    float scaling_ = 1.0f;

    WeightParameter* qkv_weight_param_ = nullptr;
    WeightParameter* qkv_bias_param_ = nullptr;
    WeightParameter* proj_weight_param_ = nullptr;
    WeightParameter* proj_bias_param_ = nullptr;
};

class Qwen3VLVisionMLP : public Module {
public:
    Qwen3VLVisionMLP(BuilderContext& ctx, const std::string& name, const Qwen3VLVisionConfig& cfg,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& fc1_weight() const;
    const Tensor* fc1_bias() const;
    const Tensor& fc2_weight() const;
    const Tensor* fc2_bias() const;

    WeightParameter* fc1_weight_param_ = nullptr;
    WeightParameter* fc1_bias_param_ = nullptr;
    WeightParameter* fc2_weight_param_ = nullptr;
    WeightParameter* fc2_bias_param_ = nullptr;
};

class Qwen3VLVisionBlock : public Module {
public:
    Qwen3VLVisionBlock(BuilderContext& ctx, const std::string& name, const Qwen3VLVisionConfig& cfg,
                       Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin) const;

private:
    const Tensor& norm1_weight() const;
    const Tensor& norm1_bias() const;
    const Tensor& norm2_weight() const;
    const Tensor& norm2_bias() const;

    Qwen3VLVisionAttention attn_;
    Qwen3VLVisionMLP mlp_;
    float eps_ = 1e-6f;

    WeightParameter* norm1_weight_param_ = nullptr;
    WeightParameter* norm1_bias_param_ = nullptr;
    WeightParameter* norm2_weight_param_ = nullptr;
    WeightParameter* norm2_bias_param_ = nullptr;
};

class Qwen3VLVisionPatchMerger : public Module {
public:
    Qwen3VLVisionPatchMerger(BuilderContext& ctx,
                             const std::string& name,
                             const Qwen3VLVisionConfig& cfg,
                             bool use_postshuffle_norm,
                             Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& norm_weight() const;
    const Tensor& norm_bias() const;
    const Tensor& fc1_weight() const;
    const Tensor* fc1_bias() const;
    const Tensor& fc2_weight() const;
    const Tensor* fc2_bias() const;

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

class Qwen3VLVisionModel : public Module {
public:
    Qwen3VLVisionModel(BuilderContext& ctx, const Qwen3VLVisionConfig& cfg, Module* parent = nullptr);

    Qwen3VLVisionOutput forward(const Tensor& pixel_values,
                                const Tensor& grid_thw,
                                const Tensor& pos_embeds,
                                const Tensor& rotary_cos,
                                const Tensor& rotary_sin);

    Qwen3VLVisionPatchEmbed& patch_embed();
    Qwen3VLVisionPatchMerger& merger();

private:
    Qwen3VLVisionConfig cfg_;
    Qwen3VLVisionPatchEmbed patch_embed_;
    std::vector<Qwen3VLVisionBlock> blocks_;
    Qwen3VLVisionPatchMerger merger_;
    std::vector<Qwen3VLVisionPatchMerger> deepstack_mergers_;
    std::vector<int32_t> deepstack_indexes_;
};

std::shared_ptr<ov::Model> create_qwen3_vl_vision_model(
    const Qwen3VLConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

