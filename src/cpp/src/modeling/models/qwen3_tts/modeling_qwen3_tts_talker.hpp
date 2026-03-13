// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Talker Module
//
// The Talker is the main autoregressive model for generating Layer-0 codec
// tokens. It uses mRoPE (multi-dimensional rotary position embedding) for
// temporal position encoding.
//
// Architecture: 28-layer transformer with GQA (16 heads, 8 KV heads)
// Input: text embeddings + codec embeddings -> combined embedding
// Output: Layer-0 codec token logits
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
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

//===----------------------------------------------------------------------===//
// Qwen3TTSTextProjection - Projects text embeddings to talker hidden size
//
// Architecture: ResizeMLP (text_hidden_size -> hidden_size)
//   input[896] -> fc1[2048] -> silu -> fc2[2048] -> output
//===----------------------------------------------------------------------===//
class Qwen3TTSTextProjection : public Module {
public:
    Qwen3TTSTextProjection(BuilderContext& ctx, const std::string& name,
                           const Qwen3TTSTalkerConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& linear_fc1_weight() const;
    const Tensor* linear_fc1_bias() const;
    const Tensor& linear_fc2_weight() const;
    const Tensor* linear_fc2_bias() const;

    WeightParameter* linear_fc1_weight_param_ = nullptr;
    WeightParameter* linear_fc1_bias_param_ = nullptr;
    WeightParameter* linear_fc2_weight_param_ = nullptr;
    WeightParameter* linear_fc2_bias_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerAttention - Multi-head attention with mRoPE
//
// Uses GQA (Grouped Query Attention): 16 Q heads, 8 KV heads
// Uses mRoPE with mrope_section = [24, 20, 20] for temporal/height/width
//===----------------------------------------------------------------------===//
class Qwen3TTSTalkerAttention : public Module {
public:
    Qwen3TTSTalkerAttention(BuilderContext& ctx, const std::string& name,
                            const Qwen3TTSTalkerConfig& cfg, Module* parent = nullptr);

    // Forward without KV cache (for prefill or no-cache mode)
    Tensor forward_no_cache(const Tensor& hidden_states,
                            const Tensor& rope_cos,
                            const Tensor& rope_sin,
                            const Tensor& causal_mask) const;

    // Forward with KV cache (for decode phase)
    // attention_mask: [batch, kv_len] where 1=attend, 0=mask
    AttentionKVOutput forward_with_cache(const Tensor& hidden_states,
                                         const Tensor& rope_cos,
                                         const Tensor& rope_sin,
                                         const Tensor& attention_mask,
                                         const std::optional<Tensor>& past_key,
                                         const std::optional<Tensor>& past_value) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;
    const Tensor* q_proj_bias() const;
    const Tensor* k_proj_bias() const;
    const Tensor* v_proj_bias() const;
    const Tensor* o_proj_bias() const;

    int32_t num_heads_;
    int32_t num_kv_heads_;
    int32_t head_dim_;
    int32_t hidden_size_;
    float scaling_;

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

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerMLP - SwiGLU feedforward network
//
// Architecture: gate = silu(x @ gate_proj), up = x @ up_proj
//              output = (gate * up) @ down_proj
//===----------------------------------------------------------------------===//
class Qwen3TTSTalkerMLP : public Module {
public:
    Qwen3TTSTalkerMLP(BuilderContext& ctx, const std::string& name,
                      const Qwen3TTSTalkerConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerDecoderLayer - Single decoder layer
//
// Architecture: layernorm -> attention -> residual -> layernorm -> mlp -> residual
//===----------------------------------------------------------------------===//
class Qwen3TTSTalkerDecoderLayer : public Module {
public:
    Qwen3TTSTalkerDecoderLayer(BuilderContext& ctx, const std::string& name,
                               const Qwen3TTSTalkerConfig& cfg, Module* parent = nullptr);

    // Forward without KV cache
    std::pair<Tensor, Tensor> forward_no_cache(const Tensor& hidden_states,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin,
                                               const Tensor& causal_mask,
                                               const std::optional<Tensor>& residual) const;

    // Forward with KV cache
    // attention_mask: [batch, kv_len] where 1=attend, 0=mask
    DecoderLayerKVOutput forward_with_cache(const Tensor& hidden_states,
                                            const Tensor& rope_cos,
                                            const Tensor& rope_sin,
                                            const Tensor& attention_mask,
                                            const std::optional<Tensor>& residual,
                                            const std::optional<Tensor>& past_key,
                                            const std::optional<Tensor>& past_value) const;

private:
    Qwen3TTSTalkerAttention self_attn_;
    Qwen3TTSTalkerMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerModel - 28-layer transformer decoder
//
// Contains: embed_tokens + text_projection + 28 decoder layers + norm
//===----------------------------------------------------------------------===//
class Qwen3TTSTalkerModel : public Module {
public:
    Qwen3TTSTalkerModel(BuilderContext& ctx, const Qwen3TTSTalkerConfig& cfg,
                        Module* parent = nullptr);

    // Forward without KV cache (prefill or full sequence)
    // Returns: (hidden_states, pre_norm_hidden)
    std::pair<Tensor, Tensor> forward_no_cache(const Tensor& inputs_embeds,
                                               const Tensor& position_ids);

    // Forward with KV cache (decode phase)
    // attention_mask: [batch, kv_len] where 1=attend, 0=mask
    TalkerModelKVOutput forward_with_cache(const Tensor& inputs_embeds,
                                           const Tensor& position_ids,
                                           const Tensor& attention_mask,
                                           const std::vector<Tensor>& past_keys,
                                           const std::vector<Tensor>& past_values);

    // Accessors for sub-modules
    VocabEmbedding& text_embedding() { return text_embedding_; }
    VocabEmbedding& codec_embedding() { return codec_embedding_; }
    Qwen3TTSTextProjection& text_projection() { return text_projection_; }
    RMSNorm& norm() { return norm_; }

private:
    // Build mRoPE cos/sin from position_ids [3, B, T]
    std::pair<Tensor, Tensor> build_mrope_cos_sin(const Tensor& position_ids) const;

    Qwen3TTSTalkerConfig cfg_;
    VocabEmbedding text_embedding_;
    VocabEmbedding codec_embedding_;
    Qwen3TTSTextProjection text_projection_;
    std::vector<Qwen3TTSTalkerDecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_;
    float rope_theta_;
};

//===----------------------------------------------------------------------===//
// Qwen3TTSTalkerForConditionalGeneration - Full Talker with lm_head
//
// Wraps Qwen3TTSTalkerModel and adds codec_head for token generation
//===----------------------------------------------------------------------===//
class Qwen3TTSTalkerForConditionalGeneration : public Module {
public:
    Qwen3TTSTalkerForConditionalGeneration(BuilderContext& ctx,
                                           const Qwen3TTSTalkerConfig& cfg,
                                           Module* parent = nullptr);

    // Forward without KV cache
    // Returns: (logits, hidden_states)
    std::pair<Tensor, Tensor> forward_no_cache(const Tensor& inputs_embeds,
                                               const Tensor& position_ids);

    // Forward with KV cache
    // attention_mask: [batch, kv_len] where 1=attend, 0=mask
    TalkerGenerationKVOutput forward_with_cache(const Tensor& inputs_embeds,
                                                const Tensor& position_ids,
                                                const Tensor& attention_mask,
                                                const std::vector<Tensor>& past_keys,
                                                const std::vector<Tensor>& past_values);

    Qwen3TTSTalkerModel& model() { return model_; }
    LMHead& codec_head() { return codec_head_; }

private:
    Qwen3TTSTalkerConfig cfg_;
    Qwen3TTSTalkerModel model_;
    LMHead codec_head_;  // Maps hidden_size -> vocab_size (3072)
};

//===----------------------------------------------------------------------===//
// Factory Functions - Embedding Models
//===----------------------------------------------------------------------===//

// Create Embedding model (for converting text/codec tokens to embeddings)
// Input:
//   - text_input_ids: [batch, seq] text token IDs
//   - codec_input_ids: [batch, seq] codec token IDs
// Output:
//   - inputs_embeds: [batch, seq, hidden_size] combined embeddings
//                    = text_projection(text_embedding(text_ids)) + codec_embedding(codec_ids)
std::shared_ptr<ov::Model> create_qwen3_tts_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create codec-only embedding model (for decode phase - no text projection!)
// Input:
//   - codec_input_ids: [batch, 1] codec token ID
// Output:
//   - codec_embeds: [batch, 1, hidden_size] codec embedding
std::shared_ptr<ov::Model> create_qwen3_tts_codec_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

//===----------------------------------------------------------------------===//
// Factory Functions - Talker Models
//===----------------------------------------------------------------------===//

// Create Talker model (for Layer-0 codec generation, no KV cache)
std::shared_ptr<ov::Model> create_qwen3_tts_talker_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Talker Prefill model - processes initial sequence, outputs KV caches
// Inputs: inputs_embeds [B, T, hidden], position_ids [3, B, T], attention_mask [B, 1, T, T]
// Outputs: logits [B, T, vocab], hidden_states [B, T, hidden],
//          key_cache_0..N-1 [B, kv_heads, T, head_dim],
//          value_cache_0..N-1 [B, kv_heads, T, head_dim]
std::shared_ptr<ov::Model> create_qwen3_tts_talker_prefill_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Talker Decode model - generates one token at a time using KV caches
// Inputs: inputs_embeds [B, 1, hidden], position_ids [3, B, 1], attention_mask [B, 1, 1, total_len],
//         past_key_0..N-1 [B, kv_heads, past_len, head_dim],
//         past_value_0..N-1 [B, kv_heads, past_len, head_dim]
// Outputs: logits [B, 1, vocab], hidden_states [B, 1, hidden],
//          key_cache_0..N-1 [B, kv_heads, total_len, head_dim],
//          value_cache_0..N-1 [B, kv_heads, total_len, head_dim]
std::shared_ptr<ov::Model> create_qwen3_tts_talker_decode_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
