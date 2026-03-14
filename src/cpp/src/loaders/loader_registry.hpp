// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file loader_registry.hpp
 * @brief Registry for model loaders with auto-detection support
 */

#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "loaders/model_loader.hpp"

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Factory function type for creating loaders
 */
using LoaderFactory = std::function<std::unique_ptr<IModelLoader>()>;

/**
 * @brief Singleton registry for model loaders
 * 
 * LoaderRegistry manages the registration and retrieval of format-specific loaders.
 * It supports auto-detection of model format from file path.
 * 
 * Usage:
 * @code
 * // Auto-detect and get loader
 * auto loader = LoaderRegistry::instance().get_loader_for_path(model_path);
 * 
 * // Get specific format loader
 * auto gguf_loader = LoaderRegistry::instance().get_loader(ModelFormat::GGUF);
 * @endcode
 */
class LoaderRegistry {
public:
    /**
     * @brief Get singleton instance
     */
    static LoaderRegistry& instance();

    /**
     * @brief Register a loader factory
     * 
     * @param format Model format this loader handles
     * @param factory Factory function to create the loader
     * @return true if registration was successful
     */
    bool register_loader(ModelFormat format, LoaderFactory factory);

    /**
     * @brief Get loader for a specific format
     * 
     * @param format Model format
     * @return Unique pointer to loader, or nullptr if not registered
     */
    std::unique_ptr<IModelLoader> get_loader(ModelFormat format) const;

    /**
     * @brief Auto-detect format and get appropriate loader
     * 
     * @param path Path to model file or directory
     * @return Unique pointer to loader that can handle the path
     * @throws std::runtime_error if no loader supports the path
     */
    std::unique_ptr<IModelLoader> get_loader_for_path(
        const std::filesystem::path& path) const;

    /**
     * @brief Detect model format from path
     * 
     * @param path Path to model file or directory
     * @return Detected format, or ModelFormat::Auto if unknown
     */
    ModelFormat detect_format(const std::filesystem::path& path) const;

    /**
     * @brief Check if a format is registered
     */
    bool has_loader(ModelFormat format) const;

    /**
     * @brief Get list of registered formats
     */
    std::vector<ModelFormat> registered_formats() const;

private:
    LoaderRegistry() = default;
    ~LoaderRegistry() = default;
    LoaderRegistry(const LoaderRegistry&) = delete;
    LoaderRegistry& operator=(const LoaderRegistry&) = delete;

    std::map<ModelFormat, LoaderFactory> factories_;
};

/**
 * @brief Helper macro for registering loaders at static initialization
 * 
 * Usage:
 * @code
 * // In gguf_loader.cpp
 * REGISTER_MODEL_LOADER(ModelFormat::GGUF, GGUFLoader);
 * @endcode
 */
#define REGISTER_MODEL_LOADER(format, loader_class) \
    static bool _registered_##loader_class = \
        ov::genai::loaders::LoaderRegistry::instance().register_loader( \
            format, \
            []() { return std::make_unique<loader_class>(); })

}  // namespace loaders
}  // namespace genai
}  // namespace ov
