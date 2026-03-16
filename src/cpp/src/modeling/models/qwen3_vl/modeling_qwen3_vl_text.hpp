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
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

class EmbeddingInjector : public Module {
public:
    EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    // visual_embeds is expected to be aligned with inputs_embeds (e.g., [B, S, H]).
    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& visual_embeds,
                   const Tensor& visual_pos_mask) const;
};

class DeepstackInjector : public Module {
public:
    DeepstackInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    // deepstack_embeds is expected to be aligned with hidden_states (e.g., [B, S, H]).
    Tensor forward(const Tensor& hidden_states,
                   const Tensor& visual_pos_mask,
                   const Tensor& deepstack_embeds) const;
};

class Qwen3VLTextAttention : public Module {
public:
    Qwen3VLTextAttention(BuilderContext& ctx,
                         const std::string& name,
                         const Qwen3VLTextConfig& cfg,
                         Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    const Tensor* q_proj_bias() const;
    const Tensor* k_proj_bias() const;
    const Tensor* v_proj_bias() const;
    const Tensor* o_proj_bias() const;

    std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                              const Tensor& values,
                                              const Tensor& beam_idx) const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    float scaling_ = 1.0f;

    RMSNorm q_norm_;
    RMSNorm k_norm_;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;
};

class Qwen3VLTextMLP : public Module {
public:
    Qwen3VLTextMLP(BuilderContext& ctx, const std::string& name, const Qwen3VLTextConfig& cfg,
                   Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen3VLTextDecoderLayer : public Module {
public:
    Qwen3VLTextDecoderLayer(BuilderContext& ctx,
                            const std::string& name,
                            const Qwen3VLTextConfig& cfg,
                            Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    Qwen3VLTextAttention self_attn_;
    Qwen3VLTextMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class Qwen3VLTextModel : public Module {
public:
    Qwen3VLTextModel(BuilderContext& ctx, const Qwen3VLTextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr,
                   const std::vector<Tensor>* deepstack_embeds = nullptr);

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr,
                          const std::vector<Tensor>* deepstack_embeds = nullptr);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    std::pair<Tensor, Tensor> build_mrope_cos_sin(const Tensor& position_ids) const;

    Qwen3VLTextConfig cfg_;
    VocabEmbedding embed_tokens_;
    EmbeddingInjector embedding_injector_;
    DeepstackInjector deepstack_injector_;
    std::vector<Qwen3VLTextDecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
};

class Qwen3VLTextForCausalLM : public Module {
public:
    Qwen3VLTextForCausalLM(BuilderContext& ctx, const Qwen3VLTextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr,
                   const std::vector<Tensor>* deepstack_embeds = nullptr);

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr,
                          const std::vector<Tensor>* deepstack_embeds = nullptr);

    Qwen3VLTextModel& model();
    LMHead& lm_head();

private:
    Qwen3VLTextConfig cfg_;
    Qwen3VLTextModel model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen3_vl_text_model(
    const Qwen3VLConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds = false,
    bool enable_visual_inputs = true);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

