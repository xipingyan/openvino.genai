// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file weight_name_mapper.hpp
 * @brief Utilities for normalizing weight names across different formats
 */

#pragma once

#include <regex>
#include <string>
#include <unordered_map>

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Normalize weight names to canonical format
 * 
 * Different model formats use different naming conventions:
 * - GGUF: blk.0.attn_q.weight
 * - HuggingFace: model.layers.0.self_attn.q_proj.weight
 * - Canonical (modeling API): model.layers[0].self_attn.q_proj.weight
 * 
 * This class provides utilities to convert between formats.
 */
class WeightNameMapper {
public:
    /**
     * @brief Convert GGUF weight name to canonical format
     * 
     * Examples:
     * - blk.0.attn_q.weight -> model.layers[0].self_attn.q_proj.weight
     * - blk.0.attn_k.weight -> model.layers[0].self_attn.k_proj.weight
     * - blk.0.ffn_gate.weight -> model.layers[0].mlp.gate_proj.weight
     * - output.weight -> lm_head.weight
     * - output_norm.weight -> model.norm.weight
     * 
     * @param gguf_name Weight name in GGUF format
     * @return Weight name in canonical format
     */
    static std::string from_gguf(const std::string& gguf_name);

    /**
     * @brief Convert HuggingFace weight name to canonical format
     * 
     * Examples:
     * - model.layers.0.self_attn.q_proj.weight -> model.layers[0].self_attn.q_proj.weight
     * - model.layers.10.mlp.gate_proj.weight -> model.layers[10].mlp.gate_proj.weight
     * 
     * @param hf_name Weight name in HuggingFace format
     * @return Weight name in canonical format
     */
    static std::string from_hf(const std::string& hf_name);

    /**
     * @brief Convert canonical format to GGUF format
     * 
     * @param canonical_name Weight name in canonical format
     * @return Weight name in GGUF format
     */
    static std::string to_gguf(const std::string& canonical_name);

    /**
     * @brief Convert canonical format to HuggingFace format
     * 
     * @param canonical_name Weight name in canonical format
     * @return Weight name in HuggingFace format
     */
    static std::string to_hf(const std::string& canonical_name);

    /**
     * @brief Check if a weight name is in GGUF format
     */
    static bool is_gguf_format(const std::string& name);

    /**
     * @brief Check if a weight name is in HuggingFace format
     */
    static bool is_hf_format(const std::string& name);

    /**
     * @brief Check if a weight name is in canonical format
     */
    static bool is_canonical_format(const std::string& name);

private:
    // GGUF to canonical name mappings
    static const std::unordered_map<std::string, std::string>& gguf_to_canonical_map();
    
    // Regex pattern for HF layer names
    static const std::regex& hf_layer_pattern();
    
    // Regex pattern for canonical layer names
    static const std::regex& canonical_layer_pattern();
};

}  // namespace loaders
}  // namespace genai
}  // namespace ov
