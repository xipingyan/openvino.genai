// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// ────────────────────────────────────────────────────────────────────────────
// Qwen3Omni config structs
// Qwen3Omni reuses the vision / text sub-configs from Qwen3VL but wraps them
// inside a "thinker_config" JSON section and adds audio-specific fields.
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniAudioEncoderConfig {
    int32_t num_mel_bins = 128;
    int32_t encoder_layers = 32;
    int32_t encoder_attention_heads = 20;
    int32_t encoder_ffn_dim = 5120;
    int32_t d_model = 1280;
    int32_t max_source_positions = 1500;
    int32_t n_window = 100;
    int32_t output_dim = 3584;
    int32_t n_window_infer = 400;
    int32_t conv_chunksize = 500;
    int32_t downsample_hidden_size = 480;
    float dropout = 0.0f;
    float attention_dropout = 0.0f;
    float activation_dropout = 0.0f;
    float initializer_range = 0.02f;
    bool scale_embedding = false;
    std::string activation_function = "gelu";
};

struct Qwen3OmniThinkerConfig {
    // Vision and text configs are shared with Qwen3VL
    Qwen3VLVisionConfig vision;
    Qwen3VLTextConfig   text;
    Qwen3OmniAudioEncoderConfig audio;

    int32_t image_token_id = 151655;
    int32_t video_token_id = 151656;
    int32_t audio_token_id = 151646;
    int32_t vision_start_token_id = 151652;
    int32_t vision_end_token_id = 151653;
    int32_t audio_start_token_id = 151647;
    int32_t audio_end_token_id = 151648;
    int32_t position_id_per_seconds = 25;

    void finalize();
    void validate() const;

    static Qwen3OmniThinkerConfig from_json(const nlohmann::json& data);
    static Qwen3OmniThinkerConfig from_json_file(const std::filesystem::path& config_path);
};

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniTalkerConfig
// Mirrors Python Qwen3OmniTalkerConfig (configuration_qwen3_omni.py:758).
// Sub-sub-configs (TalkerTextConfig, TalkerCodePredictorConfig) are opaque at
// this level — we only expose the flat fields consumed by GenAI pipelines.
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniTalkerConfig {
    // Flat fields consumed by GenAI pipelines
    int32_t num_code_groups          = 32;
    int32_t thinker_hidden_size      = 2048;
    int32_t codec_eos_token_id       = 4198;
    int32_t accept_hidden_layer      = 18;
    int32_t codec_nothink_id         = 4203;
    int32_t codec_think_bos_id       = 4204;
    int32_t codec_think_eos_id       = 4205;
    int32_t codec_pad_id             = 4196;
    int32_t codec_bos_id             = 4197;
    int32_t audio_token_id           = 151646;
    int32_t image_token_id           = 151655;
    int32_t video_token_id           = 151656;
    int32_t vision_start_token_id    = 151652;
    int32_t position_id_per_seconds  = 25;
    int32_t audio_start_token_id     = 151669;

    void finalize();
    void validate() const;

    static Qwen3OmniTalkerConfig from_json(const nlohmann::json& data);
};

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniCode2WavConfig
// Mirrors Python Qwen3OmniCode2WavConfig (configuration_qwen3_omni.py:894).
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniCode2WavConfig {
    int32_t  codebook_size              = 2048;
    int32_t  hidden_size                = 1024;
    int32_t  max_position_embeddings    = 8000;
    float    rope_theta                 = 10000.0f;
    int32_t  num_attention_heads        = 16;
    int32_t  num_key_value_heads        = 16;
    bool     attention_bias             = false;
    int32_t  sliding_window             = 72;
    int32_t  intermediate_size          = 3072;
    std::string hidden_act              = "silu";
    float    layer_scale_initial_scale  = 0.01f;
    float    rms_norm_eps               = 1e-5f;
    int32_t  num_hidden_layers          = 8;
    int32_t  num_quantizers             = 16;
    std::vector<int32_t> upsample_rates    = {8, 5, 4, 3};
    std::vector<int32_t> upsampling_ratios = {2, 2};
    int32_t  decoder_dim                = 1536;
    float    attention_dropout          = 0.0f;

    void finalize();
    void validate() const;

    static Qwen3OmniCode2WavConfig from_json(const nlohmann::json& data);
};

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniConfig  (top-level, analogous to Qwen3VLConfig)
// Mirrors Python Qwen3OmniConfig (configuration_qwen3_omni.py:1009).
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniConfig {
    std::string model_type = "qwen3_omni";
    std::vector<std::string> architectures;

    Qwen3OmniThinkerConfig  thinker;
    Qwen3OmniTalkerConfig   talker;
    Qwen3OmniCode2WavConfig code2wav;

    bool    enable_audio_output  = true;
    int32_t im_start_token_id    = 151644;
    int32_t im_end_token_id      = 151645;
    int32_t tts_pad_token_id     = 151671;
    int32_t tts_bos_token_id     = 151672;
    int32_t tts_eos_token_id     = 151673;
    int32_t system_token_id      = 8948;
    int32_t user_token_id        = 872;
    int32_t assistant_token_id   = 77091;

    void finalize();
    void validate() const;

    static Qwen3OmniConfig from_json(const nlohmann::json& data);
    static Qwen3OmniConfig from_json_file(const std::filesystem::path& config_path);
};

// ────────────────────────────────────────────────────────────────────────────
// IO name constants
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniVisionIO {
    // Shared with Qwen3VL
    static constexpr const char* kPixelValues = "pixel_values";
    static constexpr const char* kGridThw     = "grid_thw";
    static constexpr const char* kPosEmbeds   = "pos_embeds";
    static constexpr const char* kRotaryCos   = "rotary_cos";
    static constexpr const char* kRotarySin   = "rotary_sin";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kDeepstackEmbedsPrefix = "deepstack_embeds";
};

struct Qwen3OmniAudioIO {
    static constexpr const char* kInputFeatures     = "input_features";
    static constexpr const char* kFeatureAttentionMask = "feature_attention_mask";
    static constexpr const char* kAudioEmbeds       = "audio_embeds";
};

struct Qwen3OmniTextIO {
    static constexpr const char* kInputIds       = "input_ids";
    static constexpr const char* kInputsEmbeds   = "inputs_embeds";
    static constexpr const char* kAttentionMask  = "attention_mask";
    static constexpr const char* kPositionIds    = "position_ids";
    static constexpr const char* kBeamIdx        = "beam_idx";
    static constexpr const char* kVisualEmbeds   = "visual_embeds";
    static constexpr const char* kVisualPosMask  = "visual_pos_mask";
    static constexpr const char* kAudioFeatures  = "audio_features";
    static constexpr const char* kAudioPosMask   = "audio_pos_mask";
    static constexpr const char* kLogits         = "logits";
};

// ────────────────────────────────────────────────────────────────────────────
// Audio length helpers
// Mirrors _get_feat_extract_output_lengths() from processing_qwen3_omni.py
// ────────────────────────────────────────────────────────────────────────────

/// Compute the number of audio tokens produced by the convolutional
/// feature-extractor + encoder stack for a given raw input length.
int64_t get_feat_extract_output_length(int64_t input_length);

// ────────────────────────────────────────────────────────────────────────────
// Input plan for the language-model
// (same layout as Qwen3VLInputPlan but adds an audio pos-mask)
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniInputPlan {
    ov::Tensor position_ids;    // [3, B, S]  (mrope: t / h / w)
    ov::Tensor visual_pos_mask; // [B, S] boolean – marks image/video token positions
    ov::Tensor audio_pos_mask;  // [B, S] boolean – marks audio token positions
    ov::Tensor rope_deltas;     // [B, 1]
};

// ────────────────────────────────────────────────────────────────────────────
// Input planner
// Mirrors Qwen3VLInputPlanner extended with audio token awareness.
// ────────────────────────────────────────────────────────────────────────────

class Qwen3OmniInputPlanner {
public:
    explicit Qwen3OmniInputPlanner(const Qwen3OmniThinkerConfig& cfg);

    /// Build position_ids, visual_pos_mask, audio_pos_mask and rope_deltas
    /// for a prefill step.
    /// Mirrors Qwen3OmniPreTrainedModelForConditionalGeneration.get_rope_index().
    ///
    /// \param input_ids         [B, S]  i64
    /// \param attention_mask    [B, S]  (optional) i32/i64/bool/u8
    /// \param image_grid_thw    [N_img, 3]  i64  (optional)
    /// \param video_grid_thw    [N_vid, 3]  i64  (optional)
    /// \param audio_seqlens     [N_aud]     i64  (optional) – already-computed
    ///                          audio token lengths (output of get_feat_extract_output_length)
    /// \param second_per_grids  per-video seconds-per-temporal-grid (optional)
    /// \param use_audio_in_video interleave audio tokens inside video blocks
    Qwen3OmniInputPlan build_plan(const ov::Tensor& input_ids,
                                  const ov::Tensor* attention_mask     = nullptr,
                                  const ov::Tensor* image_grid_thw     = nullptr,
                                  const ov::Tensor* video_grid_thw     = nullptr,
                                  const ov::Tensor* audio_seqlens      = nullptr,
                                  const std::vector<float>* second_per_grids = nullptr,
                                  bool use_audio_in_video              = false) const;

    /// Build position_ids for a decode step given cached rope_deltas.
    static ov::Tensor build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                int64_t past_length,
                                                int64_t seq_len);

    /// Scatter flat visual embeddings into a [B, S, H] tensor according to
    /// the boolean visual_pos_mask.
    static ov::Tensor scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                            const ov::Tensor& visual_pos_mask);

    /// Scatter flat audio embeddings into a [B, S, H] tensor.
    static ov::Tensor scatter_audio_embeds(const ov::Tensor& audio_embeds,
                                           const ov::Tensor& audio_pos_mask);

    /// Convenience overload for deepstack embeddings.
    static std::vector<ov::Tensor> scatter_deepstack_embeds(
        const std::vector<ov::Tensor>& deepstack_embeds,
        const ov::Tensor& visual_pos_mask);

private:
    ov::Tensor build_pos_mask(const ov::Tensor& input_ids,
                              const ov::Tensor* attention_mask,
                              int64_t token_id) const;

    int64_t image_token_id_          = 0;
    int64_t video_token_id_          = 0;
    int64_t audio_token_id_          = 0;
    int64_t vision_start_token_id_   = 0;
    int64_t vision_end_token_id_     = 0;
    int64_t audio_start_token_id_    = 0;
    int64_t audio_end_token_id_      = 0;
    int32_t spatial_merge_size_      = 2;
    int32_t position_id_per_seconds_ = 25;
};

// ────────────────────────────────────────────────────────────────────────────
// Preprocessor config (image / video side — mirrors Qwen3VL)
// ────────────────────────────────────────────────────────────────────────────

using Qwen3OmniVisionPreprocessConfig = Qwen3VLVisionPreprocessConfig;

// ────────────────────────────────────────────────────────────────────────────
// Vision preprocessor (image / video side — thin wrapper around Qwen3VL)
// ────────────────────────────────────────────────────────────────────────────

using Qwen3OmniVisionInputs      = Qwen3VLVisionInputs;
using Qwen3OmniVisionPreprocessor = Qwen3VLVisionPreprocessor;

// ────────────────────────────────────────────────────────────────────────────
// Audio feature-extractor output
// ────────────────────────────────────────────────────────────────────────────

struct Qwen3OmniAudioInputs {
    ov::Tensor input_features;         // [B, n_mels, T]  f32
    ov::Tensor feature_attention_mask; // [B, T]           i32 / bool
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
