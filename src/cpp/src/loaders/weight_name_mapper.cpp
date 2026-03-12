// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "loaders/weight_name_mapper.hpp"

#include <sstream>

namespace ov {
namespace genai {
namespace loaders {

namespace {

// Replace first occurrence of a substring
std::string replace_first(const std::string& str, const std::string& from, const std::string& to) {
    auto pos = str.find(from);
    if (pos == std::string::npos) return str;
    std::string result = str;
    result.replace(pos, from.length(), to);
    return result;
}

// Replace all occurrences of a substring
std::string replace_all(const std::string& str, const std::string& from, const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

}  // namespace

const std::unordered_map<std::string, std::string>& WeightNameMapper::gguf_to_canonical_map() {
    static const std::unordered_map<std::string, std::string> map = {
        // Embeddings
        {"token_embd.weight", "model.embed_tokens.weight"},
        
        // Final norm and output
        {"output_norm.weight", "model.norm.weight"},
        {"output.weight", "lm_head.weight"},
        
        // Per-layer mappings (template patterns)
        // These are applied after layer number extraction
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q.bias", "self_attn.q_proj.bias"},
        {"attn_k.bias", "self_attn.k_proj.bias"},
        {"attn_v.bias", "self_attn.v_proj.bias"},
        {"attn_output.bias", "self_attn.o_proj.bias"},
        {"attn_norm.weight", "input_layernorm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
    };
    return map;
}

const std::regex& WeightNameMapper::hf_layer_pattern() {
    // Matches: model.layers.0.xxx or model.layers.10.xxx (segment boundary-aware)
    static const std::regex pattern(R"((^|\.)model\.layers\.(\d+)\.)");
    return pattern;
}

const std::regex& WeightNameMapper::canonical_layer_pattern() {
    // Matches: model.layers[0].xxx or model.layers[10].xxx (segment boundary-aware)
    static const std::regex pattern(R"((^|\.)model\.layers\[(\d+)\]\.)");
    return pattern;
}

std::string WeightNameMapper::from_gguf(const std::string& gguf_name) {
    // Handle direct mappings first
    const auto& map = gguf_to_canonical_map();
    auto it = map.find(gguf_name);
    if (it != map.end()) {
        return it->second;
    }
    
    // Handle layer patterns: blk.N.xxx -> model.layers[N].xxx
    std::regex blk_pattern(R"(blk\.(\d+)\.(.+))");
    std::smatch match;
    
    if (std::regex_match(gguf_name, match, blk_pattern)) {
        int layer_num = std::stoi(match[1].str());
        std::string suffix = match[2].str();
        
        // Convert suffix using mapping
        auto suffix_it = map.find(suffix);
        if (suffix_it != map.end()) {
            suffix = suffix_it->second;
        }
        
        std::stringstream ss;
        ss << "model.layers[" << layer_num << "]." << suffix;
        return ss.str();
    }
    
    // Return as-is if no conversion needed
    return gguf_name;
}

std::string WeightNameMapper::from_hf(const std::string& hf_name) {
    // Convert model.layers.N.xxx to model.layers[N].xxx
    std::string result = hf_name;
    
    std::smatch match;
    if (std::regex_search(result, match, hf_layer_pattern())) {
        result = std::regex_replace(result,
                                    hf_layer_pattern(),
                                    "$1model.layers[$2].",
                                    std::regex_constants::format_first_only);
    }
    
    // Convert specific array patterns that model builders use [N] indexing for
    // Only convert known array patterns, NOT nn.Sequential child modules (.0., .1., etc.)
    
    // Pattern: lm_head.N. or lm_head.N.weight/bias (Qwen3-TTS Code Predictor)
    static const std::regex lm_head_pattern(R"(\.lm_head\.(\d+)\.)");
    result = std::regex_replace(result, lm_head_pattern, ".lm_head[$1].");
    
    // Pattern: codec_embedding.N. (Qwen3-TTS Code Predictor)
    static const std::regex codec_embedding_pattern(R"(\.codec_embedding\.(\d+)\.)");
    result = std::regex_replace(result, codec_embedding_pattern, ".codec_embedding[$1].");
    
    return result;
}

std::string WeightNameMapper::to_gguf(const std::string& canonical_name) {
    // Build reverse mapping
    static std::unordered_map<std::string, std::string> reverse_map;
    static bool initialized = false;
    
    if (!initialized) {
        for (const auto& [gguf, canonical] : gguf_to_canonical_map()) {
            reverse_map[canonical] = gguf;
        }
        initialized = true;
    }
    
    // Check direct mappings
    auto it = reverse_map.find(canonical_name);
    if (it != reverse_map.end()) {
        return it->second;
    }
    
    // Handle layer patterns: model.layers[N].xxx -> blk.N.xxx
    std::smatch match;
    if (std::regex_search(canonical_name, match, canonical_layer_pattern())) {
        int layer_num = std::stoi(match[1].str());
        std::string suffix = match.suffix().str();
        
        // Try to convert suffix
        auto suffix_it = reverse_map.find(suffix);
        if (suffix_it != reverse_map.end()) {
            suffix = suffix_it->second;
        }
        
        std::stringstream ss;
        ss << "blk." << layer_num << "." << suffix;
        return ss.str();
    }
    
    return canonical_name;
}

std::string WeightNameMapper::to_hf(const std::string& canonical_name) {
    // Convert model.layers[N].xxx to model.layers.N.xxx
    std::string result = canonical_name;
    
    std::smatch match;
    if (std::regex_search(result, match, canonical_layer_pattern())) {
        result = std::regex_replace(result,
                                    canonical_layer_pattern(),
                                    "$1model.layers.$2.",
                                    std::regex_constants::format_first_only);
    }
    
    return result;
}

bool WeightNameMapper::is_gguf_format(const std::string& name) {
    return name.find("blk.") == 0 || 
           name == "token_embd.weight" ||
           name == "output_norm.weight" ||
           name == "output.weight";
}

bool WeightNameMapper::is_hf_format(const std::string& name) {
    static const std::regex hf_pattern(R"((^|\.)model\.layers\.\d+\.)");
    return std::regex_search(name, hf_pattern);
}

bool WeightNameMapper::is_canonical_format(const std::string& name) {
    return std::regex_search(name, canonical_layer_pattern()) ||
           name.find("model.embed_tokens") == 0 ||
           name.find("model.norm") == 0 ||
           name.find("lm_head") == 0;
}

}  // namespace loaders
}  // namespace genai
}  // namespace ov
