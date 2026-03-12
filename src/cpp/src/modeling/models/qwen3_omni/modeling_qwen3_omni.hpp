// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "modeling/models/qwen3_tts/modeling_qwen3_tts.hpp"

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

struct Qwen3OmniRopeConfig {
    bool mrope_interleaved = false;
    std::vector<int32_t> mrope_section = {24, 20, 20};
    std::string rope_type = "default";
};

struct Qwen3OmniTextConfig {
    std::string model_type = "qwen3_omni_text";
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 5000000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    float attention_dropout = 0.0f;
    bool tie_word_embeddings = false;
    std::string dtype = "float16";
    Qwen3OmniRopeConfig rope;

    int32_t kv_heads() const;
    int32_t resolved_head_dim() const;
    void finalize();
    void validate() const;
};

struct Qwen3OmniVisionConfig {
    std::string model_type = "qwen3_omni_vision";
    int32_t depth = 0;
    int32_t hidden_size = 0;
    std::string hidden_act = "gelu_pytorch_tanh";
    int32_t intermediate_size = 0;
    int32_t num_heads = 0;
    int32_t in_channels = 3;
    int32_t patch_size = 16;
    int32_t spatial_merge_size = 2;
    int32_t temporal_patch_size = 2;
    int32_t out_hidden_size = 0;
    int32_t num_position_embeddings = 0;
    std::vector<int32_t> deepstack_visual_indexes;

    void finalize();
};

struct Qwen3OmniAudioConfig {
    std::string model_type = "qwen3_omni_audio_encoder";
    int32_t num_mel_bins = 128;
    int32_t downsample_hidden_size = 0;
    int32_t encoder_layers = 0;
    int32_t encoder_attention_heads = 0;
    int32_t encoder_ffn_dim = 0;
    int32_t d_model = 0;
    int32_t output_dim = 0;
    int32_t n_window = 0;
    int32_t n_window_infer = 0;
    float layer_norm_eps = 1e-5f;
    std::string activation_function = "gelu";

    void finalize();
};

struct Qwen3OmniConfig {
    std::string model_type = "qwen3_omni";
    std::vector<std::string> architectures;

    Qwen3OmniTextConfig text;
    Qwen3OmniVisionConfig vision;
    Qwen3OmniAudioConfig audio;

    int32_t image_token_id = 0;
    int32_t video_token_id = 0;
    int32_t audio_token_id = 0;
    int32_t vision_start_token_id = 0;
    int32_t vision_end_token_id = 0;
    int32_t audio_start_token_id = 0;
    int32_t audio_end_token_id = 0;

    bool tie_word_embeddings = false;

    // TTS special token IDs (top-level in config.json)
    int32_t tts_pad_token_id = 151671;
    int32_t tts_eos_token_id = 151673;

    nlohmann::json talker_config_raw;
    nlohmann::json code2wav_config_raw;

    void finalize();
    void validate() const;

    static Qwen3OmniConfig from_json(const nlohmann::json& data);
    static Qwen3OmniConfig from_json_file(const std::filesystem::path& config_path);
};

struct Qwen3OmniPipelineBuildOptions {
    bool use_inputs_embeds_for_text = false;
    bool enable_multimodal_inputs = true;
    int code_predictor_steps = -1;
};

struct Qwen3OmniPipelineModels {
    std::shared_ptr<ov::Model> text;
    std::shared_ptr<ov::Model> vision;
    std::shared_ptr<ov::Model> talker_embedding;
    std::shared_ptr<ov::Model> talker_codec_embedding;
    std::shared_ptr<ov::Model> talker;
    std::shared_ptr<ov::Model> talker_prefill;
    std::shared_ptr<ov::Model> talker_decode;
    std::vector<std::shared_ptr<ov::Model>> code_predictor_ar;
    std::shared_ptr<ov::Model> code_predictor_codec_embedding;
    std::vector<std::shared_ptr<ov::Model>> code_predictor_single_codec_embedding;
    std::shared_ptr<ov::Model> speech_decoder;
};

std::shared_ptr<ov::Model> create_qwen3_omni_text_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds = false,
    bool enable_multimodal_inputs = true);

std::shared_ptr<ov::Model> create_qwen3_omni_vision_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_audio_encoder_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

Qwen3TTSTalkerConfig to_qwen3_omni_talker_config(const Qwen3OmniConfig& cfg);
Qwen3TTSCodePredictorConfig to_qwen3_omni_code_predictor_config(const Qwen3OmniConfig& cfg);
SpeechDecoderConfig to_qwen3_omni_speech_decoder_config(const Qwen3OmniConfig& cfg);

std::shared_ptr<ov::Model> create_qwen3_omni_talker_embedding_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_talker_codec_embedding_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_talker_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_talker_prefill_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_talker_decode_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_code_predictor_ar_model(
    const Qwen3OmniConfig& cfg,
    int generation_step,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_code_predictor_codec_embed_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_code_predictor_single_codec_embed_model(
    const Qwen3OmniConfig& cfg,
    int codec_layer,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_omni_speech_decoder_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

Qwen3OmniPipelineModels create_qwen3_omni_pipeline_models(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const Qwen3OmniPipelineBuildOptions& options = {});

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
