// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file model_builder.hpp
 * @brief Factory for building models from configuration and weights
 * 
 * ModelBuilder provides a unified interface for constructing OpenVINO models
 * from different architectures using the modeling API.
 */

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "loaders/model_config.hpp"
#include "modeling/weights/weight_source.hpp"
#include "modeling/weights/weight_finalizer.hpp"

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Builder function type for architecture-specific model construction
 * 
 * @param config Unified model configuration
 * @param weight_source Source for loading weights
 * @param weight_finalizer Finalizer for processing weights
 * @return Constructed OpenVINO model
 */
using ArchitectureBuilder = std::function<std::shared_ptr<ov::Model>(
    const ModelConfig& config,
    modeling::weights::WeightSource& weight_source,
    modeling::weights::WeightFinalizer& weight_finalizer)>;

/**
 * @brief Factory for building models from different architectures
 * 
 * ModelBuilder uses the modeling API (Qwen3ForCausalLM, etc.) to construct
 * OpenVINO models. It provides a unified entry point for model construction
 * regardless of the source format (GGUF, Safetensors, etc.).
 */
class ModelBuilder {
public:
    /**
     * @brief Get singleton instance
     */
    static ModelBuilder& instance();

    /**
     * @brief Register an architecture builder
     * 
     * @param architecture Architecture name (e.g., "qwen3", "llama")
     * @param builder Builder function
     * @return true if registration was successful
     */
    bool register_architecture(const std::string& architecture, ArchitectureBuilder builder);

    /**
     * @brief Check if an architecture is supported
     */
    bool has_architecture(const std::string& architecture) const;

    /**
     * @brief Get list of registered architectures
     */
    std::vector<std::string> registered_architectures() const;

    /**
     * @brief Build a model from configuration and weights
     * 
     * @param config Model configuration
     * @param weight_source Source for loading weights
     * @param weight_finalizer Finalizer for processing weights
     * @return Constructed OpenVINO model
     * @throws std::runtime_error if architecture is not supported
     */
    std::shared_ptr<ov::Model> build(
        const ModelConfig& config,
        modeling::weights::WeightSource& weight_source,
        modeling::weights::WeightFinalizer& weight_finalizer) const;

private:
    ModelBuilder() = default;
    ~ModelBuilder() = default;
    ModelBuilder(const ModelBuilder&) = delete;
    ModelBuilder& operator=(const ModelBuilder&) = delete;

    // Normalize architecture name for case-insensitive comparison
    static std::string normalize_arch_name(const std::string& name);

    mutable std::mutex m_mutex;
    std::map<std::string, ArchitectureBuilder> m_builders;
    std::vector<std::string> m_registered_archs;
};

/**
 * @brief Helper macro for registering architecture builders
 * 
 * Usage:
 * @code
 * REGISTER_ARCHITECTURE("qwen3", build_qwen3_model);
 * @endcode
 */
#define REGISTER_ARCHITECTURE(arch, builder_fn) \
    static bool _registered_arch_##arch = \
        ov::genai::loaders::ModelBuilder::instance().register_architecture(arch, builder_fn)

}  // namespace loaders
}  // namespace genai
}  // namespace ov
