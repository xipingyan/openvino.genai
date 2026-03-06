// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef ENABLE_OPENVINO_NEW_ARCH
#include "qwen3_omni_config.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include "json_utils.hpp"

namespace ov::genai::module {

modeling::models::Qwen3VLConfig get_qwen3_omni_vl_config(const std::filesystem::path &config_path) {
    if (!std::filesystem::exists(config_path)) {
        OPENVINO_THROW("Config file not found: ", config_path.string());
    }
    nlohmann::json data;
    std::ifstream file(config_path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", config_path.string());
    }
    file >> data;
    
    if (!data.contains("thinker_config")) {
        OPENVINO_THROW("Missing 'thinker_config' in config file: ", config_path.string());
    }
    nlohmann::json thinker_config = data["thinker_config"];
    
    using utils::read_json_param;
    modeling::models::Qwen3VLConfig vl_config;
    
    read_json_param(thinker_config, "model_type", vl_config.model_type);
    
    if (thinker_config.contains("text_config")) {
        nlohmann::json text_config_data = thinker_config["text_config"];
        read_json_param(text_config_data, "model_type", vl_config.text.model_type);
        read_json_param(text_config_data, "vocab_size", vl_config.text.vocab_size);
        read_json_param(text_config_data, "hidden_size", vl_config.text.hidden_size);
        read_json_param(text_config_data, "intermediate_size", vl_config.text.intermediate_size);
        read_json_param(text_config_data, "num_hidden_layers", vl_config.text.num_hidden_layers);
        read_json_param(text_config_data, "num_attention_heads", vl_config.text.num_attention_heads);
        read_json_param(text_config_data, "num_key_value_heads", vl_config.text.num_key_value_heads);
        read_json_param(text_config_data, "head_dim", vl_config.text.head_dim);
        read_json_param(text_config_data, "max_position_embeddings", vl_config.text.max_position_embeddings);
        read_json_param(text_config_data, "rms_norm_eps", vl_config.text.rms_norm_eps);
        read_json_param(text_config_data, "rope_theta", vl_config.text.rope_theta);
        read_json_param(text_config_data, "hidden_act", vl_config.text.hidden_act);
        read_json_param(text_config_data, "attention_bias", vl_config.text.attention_bias);
        read_json_param(text_config_data, "attention_dropout", vl_config.text.attention_dropout);
        read_json_param(text_config_data, "tie_word_embeddings", vl_config.text.tie_word_embeddings);
        read_json_param(text_config_data, "dtype", vl_config.text.dtype);
        
        if (text_config_data.contains("rope_scaling")) {
            nlohmann::json rope_scaling = text_config_data["rope_scaling"];
            read_json_param(rope_scaling, "mrope_interleaved", vl_config.text.rope.mrope_interleaved);
            read_json_param(rope_scaling, "mrope_section", vl_config.text.rope.mrope_section);
            read_json_param(rope_scaling, "rope_type", vl_config.text.rope.rope_type);
        }
        
        vl_config.text.finalize();
    }
    
    if (thinker_config.contains("vision_config")) {
        nlohmann::json vision_config_data = thinker_config["vision_config"];
        read_json_param(vision_config_data, "model_type", vl_config.vision.model_type);
        read_json_param(vision_config_data, "depth", vl_config.vision.depth);
        read_json_param(vision_config_data, "hidden_size", vl_config.vision.hidden_size);
        read_json_param(vision_config_data, "hidden_act", vl_config.vision.hidden_act);
        read_json_param(vision_config_data, "intermediate_size", vl_config.vision.intermediate_size);
        read_json_param(vision_config_data, "num_heads", vl_config.vision.num_heads);
        read_json_param(vision_config_data, "in_channels", vl_config.vision.in_channels);
        read_json_param(vision_config_data, "patch_size", vl_config.vision.patch_size);
        read_json_param(vision_config_data, "spatial_merge_size", vl_config.vision.spatial_merge_size);
        read_json_param(vision_config_data, "temporal_patch_size", vl_config.vision.temporal_patch_size);
        read_json_param(vision_config_data, "out_hidden_size", vl_config.vision.out_hidden_size);
        read_json_param(vision_config_data, "num_position_embeddings", vl_config.vision.num_position_embeddings);
        read_json_param(vision_config_data, "deepstack_visual_indexes", vl_config.vision.deepstack_visual_indexes);
        read_json_param(vision_config_data, "initializer_range", vl_config.vision.initializer_range);
        
        vl_config.vision.finalize();
    }
    
    read_json_param(thinker_config, "image_token_id", vl_config.image_token_id);
    read_json_param(thinker_config, "video_token_id", vl_config.video_token_id);
    read_json_param(thinker_config, "vision_start_token_id", vl_config.vision_start_token_id);
    read_json_param(thinker_config, "vision_end_token_id", vl_config.vision_end_token_id);
    
    if (thinker_config.contains("text_config") && thinker_config["text_config"].contains("tie_word_embeddings")) {
        read_json_param(thinker_config["text_config"], "tie_word_embeddings", vl_config.tie_word_embeddings);
    }
    
    vl_config.finalize();
    
    return vl_config;
}

}

#endif
