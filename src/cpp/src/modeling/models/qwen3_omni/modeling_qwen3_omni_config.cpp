// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"

#include <fstream>

#include <openvino/core/except.hpp>

#include "json_utils.hpp"

namespace {

void parse_rope_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3OmniRopeConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "mrope_interleaved", cfg.mrope_interleaved);
    read_json_param(data, "mrope_section", cfg.mrope_section);
    read_json_param(data, "rope_type", cfg.rope_type);
}

void parse_text_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3OmniTextConfig& cfg) {
    using ov::genai::utils::read_json_param;

    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "vocab_size", cfg.vocab_size);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_hidden_layers", cfg.num_hidden_layers);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads", cfg.num_key_value_heads);
    read_json_param(data, "head_dim", cfg.head_dim);
    read_json_param(data, "max_position_embeddings", cfg.max_position_embeddings);
    read_json_param(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_param(data, "rope_theta", cfg.rope_theta);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "attention_bias", cfg.attention_bias);
    read_json_param(data, "attention_dropout", cfg.attention_dropout);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);
    read_json_param(data, "dtype", cfg.dtype);

    if (data.contains("rope_scaling")) {
        parse_rope_config(data.at("rope_scaling"), cfg.rope);
    }
    if (data.contains("rope_parameters")) {
        parse_rope_config(data.at("rope_parameters"), cfg.rope);
    }

    cfg.finalize();
}

void parse_vision_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3OmniVisionConfig& cfg) {
    using ov::genai::utils::read_json_param;

    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "depth", cfg.depth);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_heads", cfg.num_heads);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "spatial_merge_size", cfg.spatial_merge_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "out_hidden_size", cfg.out_hidden_size);
    read_json_param(data, "num_position_embeddings", cfg.num_position_embeddings);
    read_json_param(data, "deepstack_visual_indexes", cfg.deepstack_visual_indexes);

    cfg.finalize();
}

void parse_audio_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3OmniAudioConfig& cfg) {
    using ov::genai::utils::read_json_param;

    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "num_mel_bins", cfg.num_mel_bins);
    read_json_param(data, "downsample_hidden_size", cfg.downsample_hidden_size);
    read_json_param(data, "encoder_layers", cfg.encoder_layers);
    read_json_param(data, "encoder_attention_heads", cfg.encoder_attention_heads);
    read_json_param(data, "encoder_ffn_dim", cfg.encoder_ffn_dim);
    read_json_param(data, "d_model", cfg.d_model);
    read_json_param(data, "output_dim", cfg.output_dim);
    read_json_param(data, "n_window", cfg.n_window);
    read_json_param(data, "n_window_infer", cfg.n_window_infer);
    read_json_param(data, "layer_norm_eps", cfg.layer_norm_eps);
    read_json_param(data, "activation_function", cfg.activation_function);

    cfg.finalize();
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

std::filesystem::path resolve_qwen3_omni_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

void read_qwen3_omni_json(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

Qwen3VLConfig to_qwen3_omni_vl_cfg(const Qwen3OmniConfig& cfg) {
    Qwen3VLConfig out;

    out.model_type = "qwen3_omni";
    out.architectures = cfg.architectures;

    out.text.model_type = cfg.text.model_type;
    out.text.vocab_size = cfg.text.vocab_size;
    out.text.hidden_size = cfg.text.hidden_size;
    out.text.intermediate_size = cfg.text.intermediate_size;
    out.text.num_hidden_layers = cfg.text.num_hidden_layers;
    out.text.num_attention_heads = cfg.text.num_attention_heads;
    out.text.num_key_value_heads = cfg.text.num_key_value_heads;
    out.text.head_dim = cfg.text.head_dim;
    out.text.max_position_embeddings = cfg.text.max_position_embeddings;
    out.text.rms_norm_eps = cfg.text.rms_norm_eps;
    out.text.rope_theta = cfg.text.rope_theta;
    out.text.hidden_act = cfg.text.hidden_act;
    out.text.attention_bias = cfg.text.attention_bias;
    out.text.attention_dropout = cfg.text.attention_dropout;
    out.text.tie_word_embeddings = cfg.text.tie_word_embeddings;
    out.text.dtype = cfg.text.dtype;
    out.text.rope.mrope_interleaved = cfg.text.rope.mrope_interleaved;
    out.text.rope.mrope_section = cfg.text.rope.mrope_section;
    out.text.rope.rope_type = cfg.text.rope.rope_type;
    out.text.finalize();

    out.vision.model_type = cfg.vision.model_type;
    out.vision.depth = cfg.vision.depth;
    out.vision.hidden_size = cfg.vision.hidden_size;
    out.vision.hidden_act = cfg.vision.hidden_act;
    out.vision.intermediate_size = cfg.vision.intermediate_size;
    out.vision.num_heads = cfg.vision.num_heads;
    out.vision.in_channels = cfg.vision.in_channels;
    out.vision.patch_size = cfg.vision.patch_size;
    out.vision.spatial_merge_size = cfg.vision.spatial_merge_size;
    out.vision.temporal_patch_size = cfg.vision.temporal_patch_size;
    out.vision.out_hidden_size = cfg.vision.out_hidden_size;
    out.vision.num_position_embeddings = cfg.vision.num_position_embeddings;
    out.vision.deepstack_visual_indexes = cfg.vision.deepstack_visual_indexes;
    out.vision.finalize();

    out.image_token_id = cfg.image_token_id;
    out.video_token_id = cfg.video_token_id;
    out.vision_start_token_id = cfg.vision_start_token_id;
    out.vision_end_token_id = cfg.vision_end_token_id;
    out.tie_word_embeddings = cfg.tie_word_embeddings;
    out.finalize();

    return out;
}

int32_t Qwen3OmniTextConfig::kv_heads() const {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
}

int32_t Qwen3OmniTextConfig::resolved_head_dim() const {
    if (head_dim > 0) {
        return head_dim;
    }
    if (hidden_size <= 0 || num_attention_heads <= 0) {
        return 0;
    }
    return hidden_size / num_attention_heads;
}

void Qwen3OmniTextConfig::finalize() {
    if (num_key_value_heads <= 0) {
        num_key_value_heads = num_attention_heads;
    }
    if (head_dim <= 0 && hidden_size > 0 && num_attention_heads > 0) {
        head_dim = hidden_size / num_attention_heads;
    }
}

void Qwen3OmniTextConfig::validate() const {
    if (hidden_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0 || resolved_head_dim() <= 0) {
        OPENVINO_THROW("Invalid qwen3_omni text config");
    }
}

void Qwen3OmniVisionConfig::finalize() {
    if (out_hidden_size <= 0) {
        out_hidden_size = hidden_size;
    }
}

void Qwen3OmniAudioConfig::finalize() {
    if (downsample_hidden_size <= 0) {
        downsample_hidden_size = d_model;
    }
    if (n_window_infer <= 0) {
        n_window_infer = n_window;
    }
}

void Qwen3OmniConfig::finalize() {
    text.finalize();
    vision.finalize();
    audio.finalize();

    if (tie_word_embeddings) {
        text.tie_word_embeddings = true;
    }
}

void Qwen3OmniConfig::validate() const {
    text.validate();
}

Qwen3OmniConfig Qwen3OmniConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;

    Qwen3OmniConfig cfg;

    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "image_token_id", cfg.image_token_id);
    read_json_param(data, "video_token_id", cfg.video_token_id);
    read_json_param(data, "audio_token_id", cfg.audio_token_id);
    read_json_param(data, "vision_start_token_id", cfg.vision_start_token_id);
    read_json_param(data, "vision_end_token_id", cfg.vision_end_token_id);
    read_json_param(data, "audio_start_token_id", cfg.audio_start_token_id);
    read_json_param(data, "audio_end_token_id", cfg.audio_end_token_id);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);
    read_json_param(data, "tts_pad_token_id", cfg.tts_pad_token_id);
    read_json_param(data, "tts_eos_token_id", cfg.tts_eos_token_id);

    const nlohmann::json* thinker = nullptr;
    if (data.contains("thinker_config") && data.at("thinker_config").is_object()) {
        thinker = &data.at("thinker_config");
        read_json_param(*thinker, "audio_token_id", cfg.audio_token_id);
        read_json_param(*thinker, "image_token_id", cfg.image_token_id);
        read_json_param(*thinker, "video_token_id", cfg.video_token_id);
        read_json_param(*thinker, "audio_start_token_id", cfg.audio_start_token_id);
        read_json_param(*thinker, "vision_start_token_id", cfg.vision_start_token_id);
    }

    if (thinker && thinker->contains("text_config")) {
        parse_text_config(thinker->at("text_config"), cfg.text);
    } else if (data.contains("text_config")) {
        parse_text_config(data.at("text_config"), cfg.text);
    } else {
        parse_text_config(data, cfg.text);
    }

    if (thinker && thinker->contains("vision_config")) {
        parse_vision_config(thinker->at("vision_config"), cfg.vision);
    } else if (data.contains("vision_config")) {
        parse_vision_config(data.at("vision_config"), cfg.vision);
    }

    if (thinker && thinker->contains("audio_config")) {
        parse_audio_config(thinker->at("audio_config"), cfg.audio);
    } else if (data.contains("audio_config")) {
        parse_audio_config(data.at("audio_config"), cfg.audio);
    }

    if (data.contains("talker_config") && data.at("talker_config").is_object()) {
        cfg.talker_config_raw = data.at("talker_config");
    }
    if (data.contains("code2wav_config") && data.at("code2wav_config").is_object()) {
        cfg.code2wav_config_raw = data.at("code2wav_config");
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

Qwen3OmniConfig Qwen3OmniConfig::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    read_qwen3_omni_json(resolve_qwen3_omni_config_path(config_path), data);
    return from_json(data);
}

Qwen3TTSTalkerConfig to_qwen3_omni_talker_config(const Qwen3OmniConfig& cfg) {
    Qwen3TTSTalkerConfig talker_cfg;
    const auto& raw = cfg.talker_config_raw;
    if (!raw.is_object()) {
        return talker_cfg;
    }

    // Qwen3-Omni stores the talker transformer config under "text_config" sub-object.
    // Fall back to text_config values when not present at the top level.
    const nlohmann::json empty_obj = nlohmann::json::object();
    const auto& tc = (raw.contains("text_config") && raw.at("text_config").is_object())
                         ? raw.at("text_config") : empty_obj;

    auto val_i32 = [&](const char* key, int32_t def) -> int32_t {
        return raw.value(key, tc.value(key, def));
    };
    auto val_f = [&](const char* key, float def) -> float {
        return raw.value(key, tc.value(key, def));
    };
    auto val_f64 = [&](const char* key, double def) -> double {
        return raw.value(key, tc.value(key, def));
    };
    auto val_str = [&](const char* key, const std::string& def) -> std::string {
        return raw.value(key, tc.value(key, def));
    };
    auto val_bool = [&](const char* key, bool def) -> bool {
        return raw.value(key, tc.value(key, def));
    };

    talker_cfg.hidden_size = val_i32("hidden_size", talker_cfg.hidden_size);
    talker_cfg.num_attention_heads = val_i32("num_attention_heads", talker_cfg.num_attention_heads);
    talker_cfg.num_key_value_heads = val_i32("num_key_value_heads", talker_cfg.num_key_value_heads);
    talker_cfg.head_dim = val_i32("head_dim", talker_cfg.head_dim);
    talker_cfg.intermediate_size = val_i32("intermediate_size", talker_cfg.intermediate_size);
    talker_cfg.num_hidden_layers = val_i32("num_hidden_layers", talker_cfg.num_hidden_layers);
    talker_cfg.vocab_size = val_i32("vocab_size", talker_cfg.vocab_size);
    talker_cfg.text_vocab_size = val_i32("text_vocab_size", talker_cfg.text_vocab_size);
    talker_cfg.text_hidden_size = raw.value("thinker_hidden_size",
                                            raw.value("text_hidden_size", talker_cfg.text_hidden_size));
    talker_cfg.rms_norm_eps = static_cast<float>(val_f64("rms_norm_eps", static_cast<double>(talker_cfg.rms_norm_eps)));
    talker_cfg.rope_theta = static_cast<float>(val_f64("rope_theta", static_cast<double>(talker_cfg.rope_theta)));
    talker_cfg.hidden_act = val_str("hidden_act", talker_cfg.hidden_act);
    talker_cfg.attention_bias = val_bool("attention_bias", talker_cfg.attention_bias);
    talker_cfg.mrope_interleaved = raw.value("mrope_interleaved", talker_cfg.mrope_interleaved);

    if (raw.contains("mrope_section") && raw.at("mrope_section").is_array()) {
        talker_cfg.mrope_section.clear();
        for (const auto& section : raw.at("mrope_section")) {
            talker_cfg.mrope_section.push_back(section.get<int32_t>());
        }
    } else if (tc.contains("mrope_section") && tc.at("mrope_section").is_array()) {
        talker_cfg.mrope_section.clear();
        for (const auto& section : tc.at("mrope_section")) {
            talker_cfg.mrope_section.push_back(section.get<int32_t>());
        }
    }

    talker_cfg.codec_eos_token_id = raw.value("codec_eos_token_id", talker_cfg.codec_eos_token_id);
    // Omni uses "codec_bos_id" / "codec_pad_id" instead of "codec_bos_token_id" / "codec_pad_token_id"
    talker_cfg.codec_bos_token_id = raw.value("codec_bos_token_id",
                                              raw.value("codec_bos_id", talker_cfg.codec_bos_token_id));
    talker_cfg.codec_pad_token_id = raw.value("codec_pad_token_id",
                                              raw.value("codec_pad_id", talker_cfg.codec_pad_token_id));
    return talker_cfg;
}

Qwen3TTSCodePredictorConfig to_qwen3_omni_code_predictor_config(const Qwen3OmniConfig& cfg) {
    Qwen3TTSCodePredictorConfig cp_cfg;
    const auto& raw = cfg.talker_config_raw;
    if (!raw.is_object()) {
        return cp_cfg;
    }

    // The code predictor's talker_hidden_size matches the talker's hidden_size,
    // which in Omni is under text_config.
    const nlohmann::json empty_obj2 = nlohmann::json::object();
    const auto& tc2 = (raw.contains("text_config") && raw.at("text_config").is_object())
                          ? raw.at("text_config") : empty_obj2;
    cp_cfg.talker_hidden_size = raw.value("hidden_size", tc2.value("hidden_size", cp_cfg.talker_hidden_size));
    if (!raw.contains("code_predictor_config") || !raw.at("code_predictor_config").is_object()) {
        return cp_cfg;
    }

    const auto& cp_raw = raw.at("code_predictor_config");
    cp_cfg.hidden_size = cp_raw.value("hidden_size", cp_cfg.hidden_size);
    cp_cfg.num_attention_heads = cp_raw.value("num_attention_heads", cp_cfg.num_attention_heads);
    cp_cfg.num_key_value_heads = cp_raw.value("num_key_value_heads", cp_cfg.num_key_value_heads);
    cp_cfg.head_dim = cp_raw.value("head_dim", cp_cfg.head_dim);
    cp_cfg.intermediate_size = cp_raw.value("intermediate_size", cp_cfg.intermediate_size);
    cp_cfg.num_hidden_layers = cp_raw.value("num_hidden_layers", cp_cfg.num_hidden_layers);
    cp_cfg.vocab_size = cp_raw.value("vocab_size", cp_cfg.vocab_size);
    cp_cfg.num_code_groups = cp_raw.value("num_code_groups", cp_cfg.num_code_groups);
    cp_cfg.rms_norm_eps = cp_raw.value("rms_norm_eps", cp_cfg.rms_norm_eps);
    cp_cfg.rope_theta = cp_raw.value("rope_theta", cp_cfg.rope_theta);
    cp_cfg.hidden_act = cp_raw.value("hidden_act", cp_cfg.hidden_act);
    cp_cfg.attention_bias = cp_raw.value("attention_bias", cp_cfg.attention_bias);
    return cp_cfg;
}

SpeechDecoderConfig to_qwen3_omni_speech_decoder_config(const Qwen3OmniConfig& cfg) {
    SpeechDecoderConfig decoder_cfg;
    const auto& raw = cfg.code2wav_config_raw;
    if (!raw.is_object()) {
        return decoder_cfg;
    }

    const nlohmann::json* config_node = &raw;
    if (raw.contains("speech_decoder_config") && raw.at("speech_decoder_config").is_object()) {
        config_node = &raw.at("speech_decoder_config");
    }

    const auto& decoder_raw = *config_node;
    decoder_cfg.num_quantizers = decoder_raw.value("num_quantizers", decoder_cfg.num_quantizers);
    decoder_cfg.codebook_size = decoder_raw.value("codebook_size", decoder_cfg.codebook_size);

    // Qwen3-Omni code2wav schema fields
    const int32_t hidden_size = decoder_raw.value("hidden_size", decoder_cfg.transformer_hidden);
    decoder_cfg.codebook_dim = decoder_raw.value("codebook_dim", hidden_size);
    decoder_cfg.latent_dim = decoder_raw.value("latent_dim", hidden_size);
    decoder_cfg.transformer_hidden = decoder_raw.value("transformer_hidden", hidden_size);
    decoder_cfg.transformer_heads = decoder_raw.value("num_attention_heads",
                                                      decoder_raw.value("transformer_heads", decoder_cfg.transformer_heads));
    decoder_cfg.transformer_head_dim = decoder_raw.value(
        "head_dim",
        decoder_raw.value("transformer_head_dim", decoder_cfg.transformer_head_dim));
    decoder_cfg.transformer_layers = decoder_raw.value("num_hidden_layers",
                                                       decoder_raw.value("transformer_layers", decoder_cfg.transformer_layers));
    decoder_cfg.transformer_intermediate = decoder_raw.value(
        "intermediate_size",
        decoder_raw.value("transformer_intermediate", decoder_cfg.transformer_intermediate));
    decoder_cfg.sliding_window = decoder_raw.value("sliding_window", decoder_cfg.sliding_window);
    decoder_cfg.layer_scale_init = decoder_raw.value(
        "layer_scale_initial_scale",
        decoder_raw.value("layer_scale_init", decoder_cfg.layer_scale_init));
    decoder_cfg.rms_norm_eps = decoder_raw.value("rms_norm_eps", decoder_cfg.rms_norm_eps);
    decoder_cfg.decoder_dim = decoder_raw.value("decoder_dim", decoder_cfg.decoder_dim);
    decoder_cfg.sample_rate = decoder_raw.value("sample_rate", decoder_cfg.sample_rate);
    decoder_cfg.rope_theta = decoder_raw.value("rope_theta", decoder_cfg.rope_theta);

    if (decoder_raw.contains("decoder_channel_mults") && decoder_raw.at("decoder_channel_mults").is_array()) {
        decoder_cfg.decoder_channel_mults.clear();
        for (const auto& value : decoder_raw.at("decoder_channel_mults")) {
            decoder_cfg.decoder_channel_mults.push_back(value.get<int32_t>());
        }
    }
    if (decoder_raw.contains("decoder_dilations") && decoder_raw.at("decoder_dilations").is_array()) {
        decoder_cfg.decoder_dilations.clear();
        for (const auto& value : decoder_raw.at("decoder_dilations")) {
            decoder_cfg.decoder_dilations.push_back(value.get<int32_t>());
        }
    }
    if (decoder_raw.contains("pre_upsample_ratios") && decoder_raw.at("pre_upsample_ratios").is_array()) {
        decoder_cfg.pre_upsample_ratios.clear();
        for (const auto& value : decoder_raw.at("pre_upsample_ratios")) {
            decoder_cfg.pre_upsample_ratios.push_back(value.get<int32_t>());
        }
    }
    if (decoder_raw.contains("decoder_upsample_rates") && decoder_raw.at("decoder_upsample_rates").is_array()) {
        decoder_cfg.decoder_upsample_rates.clear();
        for (const auto& value : decoder_raw.at("decoder_upsample_rates")) {
            decoder_cfg.decoder_upsample_rates.push_back(value.get<int32_t>());
        }
    }

    if (decoder_raw.contains("upsampling_ratios") && decoder_raw.at("upsampling_ratios").is_array()) {
        decoder_cfg.pre_upsample_ratios.clear();
        for (const auto& value : decoder_raw.at("upsampling_ratios")) {
            decoder_cfg.pre_upsample_ratios.push_back(value.get<int32_t>());
        }
    }

    if (decoder_raw.contains("upsample_rates") && decoder_raw.at("upsample_rates").is_array()) {
        decoder_cfg.decoder_upsample_rates.clear();
        for (const auto& value : decoder_raw.at("upsample_rates")) {
            decoder_cfg.decoder_upsample_rates.push_back(value.get<int32_t>());
        }
    }

    if (decoder_cfg.transformer_heads > 0 && decoder_cfg.transformer_hidden > 0) {
        decoder_cfg.transformer_head_dim = decoder_cfg.transformer_hidden / decoder_cfg.transformer_heads;
    }

    return decoder_cfg;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
