// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/qwen3_omni_pipeline.hpp"

#include <algorithm>

#include <openvino/core/except.hpp>

#include "modeling/models/qwen3_omni/processing_qwen3_omni_audio.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vision.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int resolve_qwen3_omni_code_predictor_steps(
    const Qwen3OmniConfig& cfg,
    const Qwen3OmniPipelineBuildOptions& options) {
    auto code_predictor_cfg = to_qwen3_omni_code_predictor_config(cfg);
    const int max_steps = std::max(1, code_predictor_cfg.num_code_groups - 1);
    return options.code_predictor_steps > 0 ? std::min(options.code_predictor_steps, max_steps) : max_steps;
}

Qwen3OmniMultimodalInfo process_qwen3_omni_multimodal_info(
    const nlohmann::json& conversations,
    bool use_audio_in_video,
    bool return_video_kwargs,
    const std::string& python_executable) {
    Qwen3OmniMultimodalInfo output;

    output.audios = Qwen3OmniAudioProcess::process_audio_info_via_python(
        conversations,
        use_audio_in_video,
        python_executable);

    auto vision = Qwen3OmniVisionProcess::process_vision_info_via_python(
        conversations,
        return_video_kwargs,
        python_executable);

    if (!vision.is_array() || vision.size() < 2) {
        OPENVINO_THROW("qwen3_omni vision bridge output must be array [images, videos, ...]");
    }

    output.images = vision[0];
    output.videos = vision[1];

    if (return_video_kwargs) {
        if (vision.size() < 3) {
            OPENVINO_THROW("qwen3_omni vision bridge output missing video kwargs");
        }
        output.video_kwargs = vision[2];
    } else {
        output.video_kwargs = nlohmann::json::object();
    }

    return output;
}

Qwen3OmniAudioFeatureInfo process_qwen3_omni_audio_feature_info(
    const nlohmann::json& conversations,
    const std::string& model_dir,
    bool use_audio_in_video,
    const std::string& python_executable) {
    Qwen3OmniAudioFeatureInfo output;
    auto data = Qwen3OmniAudioProcess::process_audio_features_via_python(
        conversations,
        model_dir,
        use_audio_in_video,
        python_executable);

    if (!data.is_object()) {
        OPENVINO_THROW("qwen3_omni audio feature bridge output must be object");
    }

    output.input_ids = data.value("input_ids", nlohmann::json());
    output.attention_mask = data.value("attention_mask", nlohmann::json());
    output.position_ids = data.value("position_ids", nlohmann::json());
    output.visual_pos_mask = data.value("visual_pos_mask", nlohmann::json());
    output.rope_deltas = data.value("rope_deltas", nlohmann::json());
    output.visual_embeds_padded = data.value("visual_embeds_padded", nlohmann::json());
    output.deepstack_padded = data.value("deepstack_padded", nlohmann::json::array());
    output.audio_features = data.value("audio_features", nlohmann::json());
    output.audio_pos_mask = data.value("audio_pos_mask", nlohmann::json());
    output.feature_attention_mask = data.value("feature_attention_mask", nlohmann::json());
    output.audio_feature_lengths = data.value("audio_feature_lengths", nlohmann::json());
    output.video_second_per_grid = data.value("video_second_per_grid", nlohmann::json());
    return output;
}

void validate_qwen3_omni_pipeline_models(
    const Qwen3OmniPipelineModels& models,
    int expected_code_predictor_steps) {
    if (!models.text || !models.vision || !models.talker_embedding || !models.talker_codec_embedding ||
        !models.talker || !models.talker_prefill || !models.talker_decode ||
        !models.code_predictor_codec_embedding || !models.speech_decoder) {
        OPENVINO_THROW("qwen3_omni pipeline build incomplete: one or more required modules are null");
    }

    if (expected_code_predictor_steps < 1) {
        OPENVINO_THROW("qwen3_omni pipeline expected_code_predictor_steps must be >= 1");
    }

    if (static_cast<int>(models.code_predictor_ar.size()) != expected_code_predictor_steps ||
        static_cast<int>(models.code_predictor_single_codec_embedding.size()) != expected_code_predictor_steps) {
        OPENVINO_THROW("qwen3_omni pipeline build incomplete: code predictor model counts mismatch");
    }

    for (int i = 0; i < expected_code_predictor_steps; ++i) {
        if (!models.code_predictor_ar[i] || !models.code_predictor_single_codec_embedding[i]) {
            OPENVINO_THROW("qwen3_omni pipeline build incomplete: null code predictor sub-model at step ", i);
        }
    }
}

Qwen3OmniPipelineModels create_qwen3_omni_pipeline_models(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const Qwen3OmniPipelineBuildOptions& options) {
    Qwen3OmniPipelineModels models;

    models.text = create_qwen3_omni_text_model(
        cfg,
        source,
        finalizer,
        options.use_inputs_embeds_for_text,
        options.enable_multimodal_inputs);
    models.vision = create_qwen3_omni_vision_model(cfg, source, finalizer);

    models.talker_embedding = create_qwen3_omni_talker_embedding_model(cfg, source, finalizer);
    models.talker_codec_embedding = create_qwen3_omni_talker_codec_embedding_model(cfg, source, finalizer);
    models.talker = create_qwen3_omni_talker_model(cfg, source, finalizer);
    models.talker_prefill = create_qwen3_omni_talker_prefill_model(cfg, source, finalizer);
    models.talker_decode = create_qwen3_omni_talker_decode_model(cfg, source, finalizer);

    const int steps = resolve_qwen3_omni_code_predictor_steps(cfg, options);

    models.code_predictor_ar.reserve(static_cast<size_t>(steps));
    models.code_predictor_single_codec_embedding.reserve(static_cast<size_t>(steps));
    for (int step = 0; step < steps; ++step) {
        models.code_predictor_ar.push_back(create_qwen3_omni_code_predictor_ar_model(cfg, step, source, finalizer));
        models.code_predictor_single_codec_embedding.push_back(
            create_qwen3_omni_code_predictor_single_codec_embed_model(cfg, step, source, finalizer));
    }

    models.code_predictor_codec_embedding = create_qwen3_omni_code_predictor_codec_embed_model(cfg, source, finalizer);
    models.speech_decoder = create_qwen3_omni_speech_decoder_model(cfg, source, finalizer);
    validate_qwen3_omni_pipeline_models(models, steps);

    return models;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
