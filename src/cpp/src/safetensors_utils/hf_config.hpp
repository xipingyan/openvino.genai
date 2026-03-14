// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief HuggingFace model configuration parsed from config.json
 */
struct HFConfig {
    // Model architecture
    std::string model_type;           // "qwen3", "llama", "mistral", etc.
    std::vector<std::string> architectures;  // ["Qwen3ForCausalLM"]
    
    // Model dimensions
    int hidden_size = 0;              // 2560 for Qwen3-4B
    int intermediate_size = 0;        // 9728 for Qwen3-4B (FFN hidden size)
    int num_hidden_layers = 0;        // 36 for Qwen3-4B
    int num_attention_heads = 0;      // 32 for Qwen3-4B
    int num_key_value_heads = 0;      // 8 for Qwen3-4B (GQA)
    int head_dim = 0;                 // 128 for Qwen3-4B
    int vocab_size = 0;               // 151936 for Qwen3-4B
    int max_position_embeddings = 0;  // 40960 for Qwen3-4B

    // Youtu (MLA) specific dimensions
    int q_lora_rank = 0;
    int kv_lora_rank = 0;
    int qk_rope_head_dim = 0;
    int qk_nope_head_dim = 0;
    int qk_head_dim = 0;
    int v_head_dim = 0;

    int expert_count = 0;             // MoE: number of experts
    int expert_used_count = 0;        // MoE: top-k experts per token
    int moe_intermediate_size = 0;    // MoE FFN hidden size (if different)
    
    // Normalization
    float rms_norm_eps = 1e-6f;       // RMS normalization epsilon
    
    // RoPE (Rotary Position Embedding)
    float rope_theta = 10000.0f;      // 1000000 for Qwen3
    std::string rope_scaling;         // Raw JSON string or empty if null/absent
    bool rope_interleave = false;

    // Attention/MLP flags
    bool attention_bias = false;
    bool mlp_bias = false;
    
    // Activation
    std::string hidden_act = "silu";  // "silu", "gelu", etc.
    
    // Token IDs
    int bos_token_id = 0;
    int eos_token_id = 0;
    int pad_token_id = -1;
    
    // Weight tying
    bool tie_word_embeddings = false;
    
    // Data type
    std::string torch_dtype = "float16";  // "bfloat16", "float16", "float32"
    
    // Computed properties
    int kv_heads() const { 
        return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads; 
    }
    
    int head_size() const {
        if (qk_head_dim > 0) {
            return qk_head_dim;
        }
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
};

/**
 * @brief Load HuggingFace config from config.json
 * 
 * @param model_dir Path to the model directory containing config.json
 * @return HFConfig Parsed configuration
 * @throws std::runtime_error if config.json is not found or invalid
 */
HFConfig load_hf_config(const std::filesystem::path& model_dir);

/**
 * @brief Convert HFConfig to a map for compatibility with existing code
 * 
 * This allows reusing building_blocks.cpp which expects a config map
 */
using ConfigValue = std::variant<std::monostate, float, int, std::string, std::vector<std::string>>;
std::map<std::string, ConfigValue> config_to_map(const HFConfig& config);

/**
 * @brief In-Flight weight compression configuration
 * 
 * Used to configure dynamic weight quantization during model loading.
 * The quantization is applied on-the-fly as weights are loaded from
 * safetensors files, minimizing peak memory usage.
 */
struct InFlightCompressionConfig {
    /**
     * @brief Compression mode enumeration
     */
    enum class Mode {
        NONE,       ///< No compression (keep original precision)
        INT4_SYM,   ///< 4-bit symmetric quantization
        INT4_ASYM,  ///< 4-bit asymmetric quantization  
        INT8_SYM,   ///< 8-bit symmetric quantization
        INT8_ASYM,  ///< 8-bit asymmetric quantization
    };
    
    /// Enable/disable compression
    bool enabled = false;
    
    /// Compression mode
    Mode mode = Mode::INT4_SYM;
    
    /// Quantization bit width (4 or 8)
    int bits() const {
        switch (mode) {
            case Mode::INT4_SYM:
            case Mode::INT4_ASYM:
                return 4;
            case Mode::INT8_SYM:
            case Mode::INT8_ASYM:
                return 8;
            default:
                return 16;  // No compression
        }
    }
    
    /// Is symmetric quantization
    bool symmetric() const {
        return mode == Mode::INT4_SYM || mode == Mode::INT8_SYM;
    }
    
    /// Group size for grouped quantization (-1 = per-channel)
    int group_size = 128;
    
    /// Ratio of weights to compress (1.0 = all, 0.0 = none)
    /// Weights below sensitivity threshold may skip compression
    float ratio = 1.0f;
    
    /**
     * @brief Parse compression mode from string
     * @param mode_str Mode string: "int4_sym", "int4_asym", "int8_sym", "int8_asym"
     * @return Compression mode
     */
    static Mode parse_mode(const std::string& mode_str) {
        if (mode_str == "int4_sym" || mode_str == "INT4_SYM") return Mode::INT4_SYM;
        if (mode_str == "int4_asym" || mode_str == "INT4_ASYM") return Mode::INT4_ASYM;
        if (mode_str == "int8_sym" || mode_str == "INT8_SYM") return Mode::INT8_SYM;
        if (mode_str == "int8_asym" || mode_str == "INT8_ASYM") return Mode::INT8_ASYM;
        return Mode::NONE;
    }
    
    /**
     * @brief Get mode string representation
     */
    std::string mode_string() const {
        switch (mode) {
            case Mode::INT4_SYM: return "int4_sym";
            case Mode::INT4_ASYM: return "int4_asym";
            case Mode::INT8_SYM: return "int8_sym";
            case Mode::INT8_ASYM: return "int8_asym";
            default: return "none";
        }
    }
    
    /**
     * @brief Convert compression mode to GGUF tensor type
     * @return Corresponding GGUF type for the compression mode, or 0 if NONE
     */
    int to_gguf_type() const;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
