// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file loaders.hpp
 * @brief Main include file for the multi-format model loaders module.
 * 
 * This module provides a unified interface for loading models from different formats:
 * - GGUF (llama.cpp format)
 * - Safetensors (HuggingFace format)
 * 
 * Usage example:
 * @code
 * #include "loaders/loaders.hpp"
 * 
 * using namespace ov::genai::loaders;
 * 
 * // Auto-detect format and get appropriate loader
 * auto loader = LoaderRegistry::instance().get_loader_for_path("/path/to/model");
 * 
 * // Load configuration
 * auto config = loader->load_config("/path/to/model");
 * 
 * // Create weight source and finalizer
 * auto weight_source = loader->create_weight_source("/path/to/model");
 * auto weight_finalizer = loader->create_weight_finalizer(config);
 * 
 * // Build the model using ModelBuilder
 * auto model = ModelBuilder::instance().build(config, *weight_source, properties);
 * 
 * // Load tokenizer
 * auto [tokenizer, detokenizer] = loader->load_tokenizer("/path/to/model");
 * @endcode
 */

#pragma once

// Core interfaces
#include "loaders/model_loader.hpp"
#include "loaders/model_config.hpp"
#include "loaders/loader_registry.hpp"
#include "loaders/model_builder.hpp"
#include "loaders/weight_name_mapper.hpp"

// Format-specific loaders are included conditionally in CMakeLists
// They are registered automatically via static initialization
