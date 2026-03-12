// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file model_loader.hpp
 * @brief Abstract interface for model loaders supporting different formats
 * 
 * This file defines the IModelLoader interface that all format-specific loaders
 * must implement. The interface is designed to work seamlessly with the existing
 * modeling API (WeightSource, WeightFinalizer, Module, etc.).
 */

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <openvino/openvino.hpp>

#include "loaders/model_config.hpp"
#include "modeling/weights/weight_source.hpp"
#include "modeling/weights/weight_finalizer.hpp"

namespace ov {
namespace genai {

// Type aliases for convenience
using WeightSource = modeling::weights::WeightSource;
using WeightFinalizer = modeling::weights::WeightFinalizer;

namespace loaders {

/**
 * @brief Supported model formats
 */
enum class ModelFormat {
    Auto,           ///< Auto-detect format from path
    GGUF,           ///< GGUF format (llama.cpp)
    Safetensors,    ///< HuggingFace Safetensors format
    // OpenVINO IR support planned for future releases
};

/**
 * @brief Convert ModelFormat to string for logging
 */
inline std::string to_string(ModelFormat format) {
    switch (format) {
        case ModelFormat::Auto: return "Auto";
        case ModelFormat::GGUF: return "GGUF";
        case ModelFormat::Safetensors: return "Safetensors";
        default: return "Unknown";
    }
}

/**
 * @brief Tokenizer pair (tokenizer + detokenizer)
 */
using TokenizerPair = std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>;

/**
 * @brief Abstract interface for model loaders
 * 
 * Each format-specific loader (GGUF, Safetensors, OpenVINO) implements this interface.
 * The interface is designed to integrate with the existing modeling API:
 * - WeightSource: provides access to weight tensors
 * - WeightFinalizer: handles weight post-processing (e.g., dequantization)
 * 
 * Usage:
 * @code
 * auto loader = LoaderRegistry::instance().get_loader_for_path(model_path);
 * auto config = loader->load_config(model_path);
 * auto source = loader->create_weight_source(model_path);
 * auto finalizer = loader->create_weight_finalizer(model_path);
 * auto model = ModelBuilder::build(config, *source, *finalizer);
 * @endcode
 */
class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    /**
     * @brief Check if this loader supports the given path
     * 
     * @param path Path to model file or directory
     * @return true if this loader can handle the path
     */
    virtual bool supports(const std::string& path) const = 0;

    /**
     * @brief Load model configuration from path
     * 
     * @param model_path Path to model file or directory
     * @return Unified ModelConfig structure
     * @throws std::runtime_error if config cannot be loaded
     */
    virtual ModelConfig load_config(const std::string& model_path) const = 0;

    /**
     * @brief Create a weight source for the model
     * 
     * The WeightSource provides access to weight tensors by name.
     * Weight names are normalized to the canonical format used by the modeling API.
     * 
     * @param model_path Path to model file or directory
     * @return Shared pointer to WeightSource implementation
     */
    virtual std::shared_ptr<WeightSource> create_weight_source(
        const std::string& model_path) const = 0;

    /**
     * @brief Create a weight finalizer for the model
     * 
     * The WeightFinalizer handles post-processing of weights, such as:
     * - Dequantization for GGUF (Q4_0, Q8_0, etc.)
     * - Direct constant creation for Safetensors (no processing needed)
     * 
     * @param config Model configuration
     * @return Shared pointer to WeightFinalizer implementation
     */
    virtual std::shared_ptr<WeightFinalizer> create_weight_finalizer(
        const ModelConfig& config) const = 0;

    /**
     * @brief Load tokenizer and detokenizer models
     * 
     * @param model_path Path to model file or directory
     * @return Pair of (tokenizer, detokenizer) models
     * @throws std::runtime_error if tokenizer cannot be loaded
     */
    virtual TokenizerPair load_tokenizer(const std::string& model_path) const = 0;
};

}  // namespace loaders
}  // namespace genai
}  // namespace ov
