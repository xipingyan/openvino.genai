// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Define implementation before including header
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors_utils/safetensors.hh"

#include "safetensors_utils/safetensors_loader.hpp"

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <set>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace safetensors {

// =============================================================================
// MmapHolder Implementation
// =============================================================================

MmapHolder::MmapHolder() = default;

MmapHolder::~MmapHolder() = default;

MmapHolder::MmapHolder(MmapHolder&& other) noexcept
    : m_st(std::move(other.m_st))
    , m_data_buffer(other.m_data_buffer)
    , m_valid(other.m_valid) {
    other.m_data_buffer = nullptr;
    other.m_valid = false;
}

MmapHolder& MmapHolder::operator=(MmapHolder&& other) noexcept {
    if (this != &other) {
        m_st = std::move(other.m_st);
        m_data_buffer = other.m_data_buffer;
        m_valid = other.m_valid;
        other.m_data_buffer = nullptr;
        other.m_valid = false;
    }
    return *this;
}

void MmapHolder::load(const std::filesystem::path& file_path) {
    m_st = std::make_unique<::safetensors::safetensors_t>();
    std::string warn, err;
    
    bool success = ::safetensors::mmap_from_file(file_path.string(), m_st.get(), &warn, &err);
    
    if (!success) {
        throw std::runtime_error("Failed to mmap safetensors file: " + file_path.string() + 
                                 ", error: " + err);
    }
    
    if (!warn.empty()) {
        std::cerr << "Warning loading " << file_path << ": " << warn << std::endl;
    }
    
    // Validate data offsets
    if (!::safetensors::validate_data_offsets(*m_st, err)) {
        throw std::runtime_error("Invalid data offsets in " + file_path.string() + ": " + err);
    }
    
    m_data_buffer = m_st->databuffer_addr;
    m_valid = true;
}

const ::safetensors::safetensors_t& MmapHolder::get_st() const {
    if (!m_valid || !m_st) {
        throw std::runtime_error("MmapHolder not loaded");
    }
    return *m_st;
}

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/**
 * @brief Check if modeling API is enabled via environment variable
 */
bool is_modeling_api_enabled() {
    const char* env = std::getenv("OV_GENAI_USE_MODELING_API");
    if (env == nullptr) {
        return false;  // Default: disabled
    }
    std::string val(env);
    return (val == "1" || val == "true" || val == "TRUE");
}

bool is_unified_loader_enabled() {
    const char* env = std::getenv("OV_GENAI_USE_UNIFIED_LOADER");
    if (env == nullptr) {
        return false;
    }
    std::string val(env);
    return (val == "1" || val == "true" || val == "TRUE");
}

/**
 * @brief Check if zero-copy mode is enabled via environment variable
 * 
 * Environment variable: OV_GENAI_USE_ZERO_COPY
 * - "0" or "false" or "FALSE": Disable zero-copy (use memcpy)
 * - "1" or "true" or "TRUE": Enable zero-copy
 * - unset: Enable zero-copy only if modeling API is enabled (building_blocks doesn't support zero-copy)
 */
bool is_zero_copy_enabled() {
    const char* env = std::getenv("OV_GENAI_USE_ZERO_COPY");
    if (env == nullptr) {
        // Default: enable for modeling API and unified loader path.
        // Building-blocks path does not support zero-copy safely.
        return is_modeling_api_enabled() || is_unified_loader_enabled();
    }
    std::string val(env);
    return !(val == "0" || val == "false" || val == "FALSE");
}

// Parse model.safetensors.index.json to get shard file list
std::vector<std::string> parse_index_json(const std::filesystem::path& index_path) {
    std::ifstream file(index_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + index_path.string());
    }

    nlohmann::json index_json;
    try {
        file >> index_json;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse " + index_path.string() + ": " + e.what());
    }

    if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
        throw std::runtime_error("weight_map not found or invalid in " + index_path.string());
    }

    // Extract unique file names from weight_map values.
    std::set<std::string> files;
    for (auto it = index_json["weight_map"].begin(); it != index_json["weight_map"].end(); ++it) {
        if (!it.value().is_string()) {
            throw std::runtime_error("Invalid weight_map entry for key '" + it.key() + "' in " + index_path.string());
        }
        files.insert(it.value().get<std::string>());
    }

    return std::vector<std::string>(files.begin(), files.end());
}

}  // anonymous namespace

ov::element::Type convert_dtype(int safetensors_dtype) {
    using namespace ::safetensors;
    
    switch (safetensors_dtype) {
        case kBOOL:     return ov::element::boolean;
        case kUINT8:    return ov::element::u8;
        case kINT8:     return ov::element::i8;
        case kINT16:    return ov::element::i16;
        case kUINT16:   return ov::element::u16;
        case kFLOAT16:  return ov::element::f16;
        case kBFLOAT16: return ov::element::bf16;
        case kFLOAT8_E4M3: return ov::element::f8e4m3;
        case kINT32:    return ov::element::i32;
        case kUINT32:   return ov::element::u32;
        case kFLOAT32:  return ov::element::f32;
        case kFLOAT64:  return ov::element::f64;
        case kINT64:    return ov::element::i64;
        case kUINT64:   return ov::element::u64;
        default:
            throw std::runtime_error("Unknown safetensors dtype: " + std::to_string(safetensors_dtype));
    }
}

bool is_safetensors_model(const std::filesystem::path& model_dir) {
    // Check for single file
    if (std::filesystem::exists(model_dir / "model.safetensors")) {
        return true;
    }
    
    // Check for sharded files
    if (std::filesystem::exists(model_dir / "model.safetensors.index.json")) {
        return true;
    }
    
    // Check for any .safetensors file
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            return true;
        }
    }
    
    return false;
}

SafetensorsData load_safetensors_file(const std::filesystem::path& file_path) {
    SafetensorsData result;
    
    const bool zero_copy = is_zero_copy_enabled();
    
    // Create mmap holder
    auto holder = std::make_shared<MmapHolder>();
    holder->load(file_path);
    
    // Keep holder alive
    result.mmap_holders.push_back(holder);
    
    const auto& st = holder->get_st();
    
    // Process tensors
    const auto& keys = st.tensors.keys();
    for (size_t i = 0; i < keys.size(); i++) {
        const std::string& name = keys[i];
        ::safetensors::tensor_t tensor;
        
        if (!st.tensors.at(name, &tensor)) {
            continue;
        }
        
        // Convert dtype and shape
        ov::element::Type ov_dtype = convert_dtype(static_cast<int>(tensor.dtype));
        ov::Shape ov_shape(tensor.shape.begin(), tensor.shape.end());
        
        // Store tensor metadata
        TensorInfo info;
        info.name = name;
        info.dtype = ov_dtype;
        info.shape = ov_shape;
        result.tensor_infos[name] = info;
        
        if (zero_copy) {
            // Zero-copy mode: Store mmap info for later SharedBuffer creation
            TensorMmapInfo mmap_info;
            mmap_info.holder = holder;
            mmap_info.offset = tensor.data_offsets[0];
            mmap_info.size = tensor.data_offsets[1] - tensor.data_offsets[0];
            result.tensor_mmap_info[name] = mmap_info;
        } else {
            // Legacy mode: Copy data to ov::Tensor
            const uint8_t* data_ptr = st.databuffer_addr + tensor.data_offsets[0];
            size_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];
            
            ov::Tensor ov_tensor(ov_dtype, ov_shape);
            std::memcpy(ov_tensor.data(), data_ptr, data_size);
            result.tensors[name] = std::move(ov_tensor);
        }
    }
    
    // Copy metadata (small strings, not a concern)
    const auto& meta_keys = st.metadata.keys();
    for (size_t i = 0; i < meta_keys.size(); i++) {
        const std::string& key = meta_keys[i];
        std::string value;
        if (st.metadata.at(key, &value)) {
            result.metadata[key] = value;
        }
    }
    
    return result;
}

SafetensorsData load_safetensors(const std::filesystem::path& model_dir) {
    SafetensorsData result;
    
    std::vector<std::filesystem::path> files_to_load;
    
    // Check for single file model
    std::filesystem::path single_file = model_dir / "model.safetensors";
    if (std::filesystem::exists(single_file)) {
        files_to_load.push_back(single_file);
    } else {
        // Check for sharded model
        std::filesystem::path index_file = model_dir / "model.safetensors.index.json";
        if (std::filesystem::exists(index_file)) {
            auto shard_names = parse_index_json(index_file);
            for (const auto& name : shard_names) {
                files_to_load.push_back(model_dir / name);
            }
        } else {
            // Load all .safetensors files in directory
            for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
                if (entry.path().extension() == ".safetensors") {
                    files_to_load.push_back(entry.path());
                }
            }
        }
    }
    
    if (files_to_load.empty()) {
        throw std::runtime_error("No safetensors files found in " + model_dir.string());
    }
    
    // Sort files for deterministic loading order
    std::sort(files_to_load.begin(), files_to_load.end());
    
    const bool zero_copy = is_zero_copy_enabled();
    const char* mode_str = zero_copy ? "Zero-Copy" : "Copy";
    
    std::cout << "[Safetensors " << mode_str << "] Loading " << files_to_load.size() << " file(s)..." << std::endl;
    
    // Load each file
    for (const auto& file_path : files_to_load) {
        std::cout << "[Safetensors " << mode_str << "] " 
                  << (zero_copy ? "Mapping" : "Loading") << " " << file_path.filename() << "..." << std::endl;
        
        auto start_load = std::chrono::high_resolution_clock::now();
        SafetensorsData file_data = load_safetensors_file(file_path);
        auto end_load = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
        std::cout << "  -> Time: " << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        
        // Merge mmap holders (keeps mmap alive)
        for (auto& holder : file_data.mmap_holders) {
            result.mmap_holders.push_back(std::move(holder));
        }
        
        // Merge tensor mmap info (zero-copy references)
        for (auto& [name, mmap_info] : file_data.tensor_mmap_info) {
            if (result.tensor_mmap_info.count(name)) {
                std::cerr << "Warning: Duplicate tensor name: " << name << std::endl;
            }
            result.tensor_mmap_info[name] = std::move(mmap_info);
        }
        
        // Merge tensors (legacy mode)
        for (auto& [name, tensor] : file_data.tensors) {
            if (result.tensors.count(name)) {
                std::cerr << "Warning: Duplicate tensor name: " << name << std::endl;
            }
            result.tensors[name] = std::move(tensor);
        }
        
        // Merge tensor infos (metadata only)
        for (auto& [name, info] : file_data.tensor_infos) {
            result.tensor_infos[name] = std::move(info);
        }
        
        // Merge metadata (first file wins for duplicates)
        for (auto& [key, value] : file_data.metadata) {
            if (!result.metadata.count(key)) {
                result.metadata[key] = std::move(value);
            }
        }
    }
    
    if (zero_copy) {
        std::cout << "[Safetensors Zero-Copy] Mapped " << result.tensor_infos.size() 
                  << " tensors (no memory copy)" << std::endl;
    } else {
        std::cout << "[Safetensors Copy] Loaded " << result.tensors.size() 
                  << " tensors (with memory copy)" << std::endl;
    }
    
    return result;
}

std::string map_hf_weight_name(const std::string& hf_name) {
    // For now, return the same name
    // HuggingFace naming is already quite standardized
    // We may need to add mapping later if building_blocks expects different names
    return hf_name;
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
