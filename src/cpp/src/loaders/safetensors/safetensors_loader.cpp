// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file safetensors_loader.cpp
 * @brief Implementation of SafetensorsLoader for HuggingFace models.
 * 
 * This loader wraps the existing safetensors_utils implementation to provide
 * a unified IModelLoader interface for loading HuggingFace safetensors models.
 */

#include "loaders/safetensors/safetensors_loader.hpp"
#include "loaders/loader_registry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Include existing safetensors utilities (unified implementation)
#include "safetensors_utils/hf_config.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

// Modeling API quantization support
#include "modeling/weights/quantization_config.hpp"

// JSON parsing
#include <nlohmann/json.hpp>
#include <sstream>
#include <cstdlib>

namespace ov {
namespace genai {
namespace loaders {

namespace fs = std::filesystem;

// Note: SafetensorsWeightSource is now defined in safetensors_utils/safetensors_weight_source.hpp
// It handles HF name format (layers.N) to canonical format (layers[N]) conversion.

// Note: We reuse safetensors::SafetensorsWeightFinalizer from safetensors_utils/
// which handles BF16/F16 to F32 conversion for consistency with GGUF behavior.

// =============================================================================
// SafetensorsLoader Implementation
// =============================================================================

bool SafetensorsLoader::supports(const std::string& path) const {
    fs::path p(path);
    
    // Must be a directory
    if (!fs::is_directory(p)) {
        return false;
    }
    
    // Must have config.json
    if (!fs::exists(p / "config.json")) {
        return false;
    }
    
    // Must have safetensors files
    bool has_single = fs::exists(p / "model.safetensors");
    bool has_sharded = fs::exists(p / "model.safetensors.index.json");
    
    // Check for any .safetensors files
    bool has_any_safetensors = false;
    if (!has_single && !has_sharded) {
        for (const auto& entry : fs::directory_iterator(p)) {
            if (entry.path().extension() == ".safetensors") {
                has_any_safetensors = true;
                break;
            }
        }
    }
    
    return has_single || has_sharded || has_any_safetensors;
}

ModelConfig SafetensorsLoader::load_config(const std::string& path) const {
    fs::path config_file = fs::path(path) / "config.json";
    return ModelConfig::from_hf_json(config_file);
}

std::shared_ptr<WeightSource> SafetensorsLoader::create_weight_source(
    const std::string& path) const {
    
    fs::path model_dir(path);
    
    // Use existing safetensors loading logic
    auto data = safetensors::load_safetensors(model_dir);
    
    // Use unified SafetensorsWeightSource from safetensors_utils
    return std::make_shared<safetensors::SafetensorsWeightSource>(std::move(data));
}

std::shared_ptr<WeightFinalizer> SafetensorsLoader::create_weight_finalizer(
    const ModelConfig& config) const {
    
    using namespace ov::genai::modeling::weights;
    
    // Check if quantization config is provided in model config (from properties)
    QuantizationConfig quant_config;
    if (config.quantization_config.has_value()) {
        quant_config = *config.quantization_config;
    } else {
        // Fallback: Parse quantization configuration from environment variables
        quant_config = parse_quantization_config_from_env();
    }
    
    // Create finalizer with quantization config
    return std::make_shared<safetensors::SafetensorsWeightFinalizer>(quant_config);
}

TokenizerPair SafetensorsLoader::load_tokenizer(const std::string& path) const {
    // NOTE: This function is currently not used in the actual workflow.
    // Tokenizer loading is handled by the Tokenizer class (tokenizer_impl.cpp)
    // which automatically detects HuggingFace tokenizer.json and converts it
    // to OpenVINO format using Python openvino_tokenizers if needed.
    // This interface exists for potential future unified loader integration.
    std::cout << "[SafetensorsLoader::load_tokenizer] Called but NOT used - "
              << "Tokenizer class handles loading independently. Path: " << path << std::endl;
    return TokenizerPair{nullptr, nullptr};
}

std::vector<std::string> SafetensorsLoader::find_safetensors_files(
    const std::string& dir_path) const {
    
    fs::path dir(dir_path);
    std::vector<std::string> files;
    
    // Check for single file
    if (fs::exists(dir / "model.safetensors")) {
        files.push_back((dir / "model.safetensors").string());
        return files;
    }
    
    // Check for sharded files using index
    fs::path index_path = dir / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        auto mapping = parse_index_file(index_path.string());
        
        // Collect unique file names
        std::set<std::string> unique_files;
        for (const auto& [_, file] : mapping) {
            unique_files.insert(file);
        }
        
        for (const auto& file : unique_files) {
            files.push_back((dir / file).string());
        }
        
        // Sort for consistent ordering
        std::sort(files.begin(), files.end());
        return files;
    }
    
    // Fallback: find all .safetensors files
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors") {
            files.push_back(entry.path().string());
        }
    }
    
    std::sort(files.begin(), files.end());
    return files;
}

std::unordered_map<std::string, std::string> SafetensorsLoader::parse_index_file(
    const std::string& index_path) const {
    
    std::unordered_map<std::string, std::string> weight_to_file;
    
    std::ifstream ifs(index_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open index file: " + index_path);
    }
    
    nlohmann::json index;
    ifs >> index;
    
    if (index.contains("weight_map")) {
        for (auto& [weight_name, file_name] : index["weight_map"].items()) {
            weight_to_file[weight_name] = file_name.get<std::string>();
        }
    }
    
    return weight_to_file;
}

// Register loader with the registry
namespace {
    static bool registered = []() {
        LoaderRegistry::instance().register_loader(
            ModelFormat::Safetensors,
            []() { return std::make_unique<SafetensorsLoader>(); }
        );
        return true;
    }();
}

}  // namespace loaders
}  // namespace genai
}  // namespace ov
