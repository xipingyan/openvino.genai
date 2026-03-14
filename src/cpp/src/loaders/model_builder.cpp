// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file model_builder.cpp
 * @brief Implementation of ModelBuilder registry
 *
 * This file contains only the ModelBuilder registry implementation.
 * Architecture-specific build functions are defined in their respective
 * model files (e.g., qwen3_dense.cpp) and self-register at static init.
 *
 * To add a new model:
 * 1. Create the model class in modeling/models/xxx.cpp
 * 2. Add build_xxx_model() function in the same file
 * 3. Add static registration at the end of that file
 * 4. No changes needed to this file!
 */

#include "loaders/model_builder.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace ov {
namespace genai {
namespace loaders {

ModelBuilder& ModelBuilder::instance() {
    static ModelBuilder instance;
    return instance;
}

bool ModelBuilder::register_architecture(const std::string& arch_name, ArchitectureBuilder builder) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    std::string normalized = normalize_arch_name(arch_name);
    
    // Check if already registered
    if (m_builders.find(normalized) != m_builders.end()) {
        // Allow re-registration, return true
        m_builders[normalized] = std::move(builder);
        return true;
    }
    
    m_builders[normalized] = std::move(builder);
    m_registered_archs.push_back(arch_name);
    return true;
}

bool ModelBuilder::has_architecture(const std::string& arch_name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string normalized = normalize_arch_name(arch_name);
    return m_builders.find(normalized) != m_builders.end();
}

std::vector<std::string> ModelBuilder::registered_architectures() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_registered_archs;
}

std::shared_ptr<ov::Model> ModelBuilder::build(
    const ModelConfig& config,
    modeling::weights::WeightSource& weight_source,
    modeling::weights::WeightFinalizer& weight_finalizer) const {
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    std::string normalized = normalize_arch_name(config.architecture);
    auto it = m_builders.find(normalized);
    
    if (it == m_builders.end()) {
        // Try model_type if architecture not found
        normalized = normalize_arch_name(config.model_type);
        it = m_builders.find(normalized);
    }
    
    if (it == m_builders.end()) {
        throw std::runtime_error(
            "Unsupported architecture: '" + config.architecture + 
            "' (model_type: '" + config.model_type + "'). "
            "Registered architectures: " + 
            [this]() {
                std::string archs;
                for (const auto& a : m_registered_archs) {
                    if (!archs.empty()) archs += ", ";
                    archs += a;
                }
                return archs.empty() ? "(none)" : archs;
            }());
    }
    
    return it->second(config, weight_source, weight_finalizer);
}

std::string ModelBuilder::normalize_arch_name(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    
    return result;
}

}  // namespace loaders
}  // namespace genai
}  // namespace ov
