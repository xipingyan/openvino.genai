// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file safetensors_loader.hpp
 * @brief Safetensors format model loader implementation.
 * 
 * This loader handles HuggingFace models stored in safetensors format.
 * It reads config.json for model configuration and *.safetensors files for weights.
 */

#include <string>
#include <vector>
#include <unordered_map>

#include "loaders/model_loader.hpp"
#include "loaders/model_config.hpp"

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Model loader for Safetensors format (HuggingFace).
 * 
 * The SafetensorsLoader provides:
 * - Reading config.json to create ModelConfig
 * - Creating WeightSource that reads tensors from *.safetensors files
 * - Support for sharded models (model-00001-of-00003.safetensors, etc.)
 * - Loading tokenizer from tokenizer.json / tokenizer_config.json
 * 
 * Weight naming convention:
 * - HF uses: model.layers.N.xxx format for layer weights
 * - Converted to canonical: model.layers[N].xxx
 * 
 * @see IModelLoader for interface documentation
 */
class SafetensorsLoader : public IModelLoader {
public:
    SafetensorsLoader() = default;
    ~SafetensorsLoader() override = default;

    // Non-copyable, non-movable
    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&&) = delete;
    SafetensorsLoader& operator=(SafetensorsLoader&&) = delete;

    /**
     * @brief Check if this loader supports the given path.
     * @param path Path to file or directory
     * @return true if path contains safetensors files and config.json
     */
    bool supports(const std::string& path) const override;

    /**
     * @brief Load model configuration from config.json.
     * @param path Path to model directory
     * @return ModelConfig populated from config.json
     */
    ModelConfig load_config(const std::string& path) const override;

    /**
     * @brief Create weight source for safetensors files.
     * @param path Path to model directory
     * @return WeightSource that provides tensor data
     */
    std::shared_ptr<WeightSource> create_weight_source(
        const std::string& path) const override;

    /**
     * @brief Create weight finalizer (identity for safetensors).
     * @param config Model configuration
     * @return WeightFinalizer for post-processing tensors
     */
    std::shared_ptr<WeightFinalizer> create_weight_finalizer(
        const ModelConfig& config) const override;

    /**
     * @brief Load tokenizer from tokenizer.json.
     * @param path Path to model directory
     * @return Tokenizer/detokenizer pair
     */
    TokenizerPair load_tokenizer(const std::string& path) const override;

private:
    /**
     * @brief Find all safetensors files in directory.
     * @param dir_path Directory path
     * @return List of safetensors file paths in correct order
     */
    std::vector<std::string> find_safetensors_files(const std::string& dir_path) const;

    /**
     * @brief Parse model.safetensors.index.json for sharded models.
     * @param index_path Path to index file
     * @return Weight name to file mapping
     */
    std::unordered_map<std::string, std::string> parse_index_file(
        const std::string& index_path) const;
};

}  // namespace loaders
}  // namespace genai
}  // namespace ov
