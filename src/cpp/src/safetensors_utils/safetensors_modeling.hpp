// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <memory>

#include "openvino/openvino.hpp"
#include "modeling/weights/quantization_config.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Create an OpenVINO model from HuggingFace safetensors format
 * 
 * @deprecated Use loaders::LoaderRegistry and loaders::ModelBuilder instead.
 * 
 * Migration example:
 * @code
 * auto& registry = loaders::LoaderRegistry::instance();
 * auto loader = registry.get_loader_for_path(model_dir.string());
 * auto config = loader->load_config(model_dir.string());
 * auto source = loader->create_weight_source(model_dir.string());
 * auto finalizer = loader->create_weight_finalizer(config);
 * auto model = loaders::ModelBuilder::instance().build(config, *source, *finalizer);
 * @endcode
 * 
 * This function:
 * 1. Loads config.json to get model configuration
 * 2. Loads all safetensors weight files
 * 3. Constructs the ov::Model using building blocks
 * 
 * @param model_dir Path to the HuggingFace model directory
 * @param enable_save_ov_model If true, save the generated model to disk
 * @param quant_config Optional quantization configuration for in-flight quantization
 * @return std::shared_ptr<ov::Model> The constructed OpenVINO model
 * @throws std::runtime_error if model creation fails
 */
[[deprecated("Use loaders::LoaderRegistry and loaders::ModelBuilder instead")]]
std::shared_ptr<ov::Model> create_from_safetensors(
    const std::filesystem::path& model_dir,
    bool enable_save_ov_model = false,
    const ov::genai::modeling::weights::QuantizationConfig& quant_config = ov::genai::modeling::weights::QuantizationConfig());

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
