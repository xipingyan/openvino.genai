// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "loaders/loader_registry.hpp"

#include <fstream>
#include <stdexcept>

namespace ov {
namespace genai {
namespace loaders {

LoaderRegistry& LoaderRegistry::instance() {
    static LoaderRegistry registry;
    return registry;
}

bool LoaderRegistry::register_loader(ModelFormat format, LoaderFactory factory) {
    if (format == ModelFormat::Auto) {
        return false;  // Cannot register Auto format
    }
    factories_[format] = std::move(factory);
    return true;
}

std::unique_ptr<IModelLoader> LoaderRegistry::get_loader(ModelFormat format) const {
    auto it = factories_.find(format);
    if (it == factories_.end()) {
        return nullptr;
    }
    return it->second();
}

ModelFormat LoaderRegistry::detect_format(const std::filesystem::path& path) const {
    if (!std::filesystem::exists(path)) {
        return ModelFormat::Auto;
    }

    // Check for GGUF file
    if (std::filesystem::is_regular_file(path)) {
        if (path.extension() == ".gguf") {
            return ModelFormat::GGUF;
        }
        
        // Check GGUF magic number
        std::ifstream file(path, std::ios::binary);
        if (file) {
            char magic[4];
            file.read(magic, 4);
            if (std::string(magic, 4) == "GGUF") {
                return ModelFormat::GGUF;
            }
        }
    }

    // Check for directory-based formats
    if (std::filesystem::is_directory(path)) {
        // Check for Safetensors (HuggingFace format)
        bool has_config = std::filesystem::exists(path / "config.json");
        bool has_safetensors = std::filesystem::exists(path / "model.safetensors") ||
                               std::filesystem::exists(path / "model.safetensors.index.json");
        
        if (has_config && has_safetensors) {
            return ModelFormat::Safetensors;
        }
    }

    return ModelFormat::Auto;
}

std::unique_ptr<IModelLoader> LoaderRegistry::get_loader_for_path(
    const std::filesystem::path& path) const {
    
    // Try to detect format
    ModelFormat format = detect_format(path);
    std::string path_str = path.string();

    if (format != ModelFormat::Auto) {
        auto loader = get_loader(format);
        if (loader && loader->supports(path_str)) {
            return loader;
        }
    }

    // Try all registered loaders
    for (const auto& [fmt, factory] : factories_) {
        auto loader = factory();
        if (loader && loader->supports(path_str)) {
            return loader;
        }
    }

    throw std::runtime_error("No loader found for path: " + path_str);
}

bool LoaderRegistry::has_loader(ModelFormat format) const {
    return factories_.find(format) != factories_.end();
}

std::vector<ModelFormat> LoaderRegistry::registered_formats() const {
    std::vector<ModelFormat> formats;
    formats.reserve(factories_.size());
    for (const auto& [format, _] : factories_) {
        formats.push_back(format);
    }
    return formats;
}

}  // namespace loaders
}  // namespace genai
}  // namespace ov
