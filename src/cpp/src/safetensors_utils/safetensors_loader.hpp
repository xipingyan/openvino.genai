// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <memory>

#include "openvino/genai/visibility.hpp"
#include "openvino/openvino.hpp"

// Forward declare safetensors_t to avoid including the large header
namespace safetensors {
struct safetensors_t;
}

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Tensor metadata
 */
struct OPENVINO_GENAI_EXPORTS TensorInfo {
    std::string name;
    ov::element::Type dtype;
    ov::Shape shape;
};

/**
 * @brief Holder for mmap'ed safetensors file
 * 
 * This class owns the mmap lifetime. When destroyed, it will unmap the file.
 * Multiple tensors can reference the same MmapHolder through shared_ptr,
 * ensuring the mmap stays valid as long as any tensor needs it.
 */
class OPENVINO_GENAI_EXPORTS MmapHolder {
public:
    MmapHolder();
    ~MmapHolder();
    
    // Non-copyable
    MmapHolder(const MmapHolder&) = delete;
    MmapHolder& operator=(const MmapHolder&) = delete;
    
    // Movable
    MmapHolder(MmapHolder&& other) noexcept;
    MmapHolder& operator=(MmapHolder&& other) noexcept;
    
    /**
     * @brief Load safetensors file via mmap
     * @param file_path Path to .safetensors file
     * @throws std::runtime_error on failure
     */
    void load(const std::filesystem::path& file_path);
    
    /**
     * @brief Check if loaded successfully
     */
    bool is_valid() const { return m_valid; }
    
    /**
     * @brief Get the data buffer address (start of tensor data)
     */
    const uint8_t* data_buffer() const { return m_data_buffer; }
    
    /**
     * @brief Get safetensors_t reference for tensor iteration
     */
    const ::safetensors::safetensors_t& get_st() const;
    
private:
    std::unique_ptr<::safetensors::safetensors_t> m_st;
    const uint8_t* m_data_buffer = nullptr;  // Points into mmap
    bool m_valid = false;
};

/**
 * @brief Mmap info for a single tensor
 */
struct OPENVINO_GENAI_EXPORTS TensorMmapInfo {
    std::shared_ptr<MmapHolder> holder;  // Keeps mmap alive
    size_t offset;                        // Offset from data_buffer
    size_t size;                          // Size in bytes
};

/**
 * @brief Result of loading safetensors files
 * 
 * Two modes based on OV_GENAI_USE_ZERO_COPY environment variable:
 * - Mmap mode (zero_copy=1): tensor_infos + tensor_mmap_info (no data copy)
 * - Copy mode (zero_copy=0): tensor_infos + tensors (data copied to ov::Tensor)
 */
struct OPENVINO_GENAI_EXPORTS SafetensorsData {
    // Tensor metadata (always populated)
    std::unordered_map<std::string, TensorInfo> tensor_infos;
    std::map<std::string, std::string> metadata;
    
    // Mmap mode: keeps mmap files alive
    std::vector<std::shared_ptr<MmapHolder>> mmap_holders;
    
    // Mmap mode: tensor name -> mmap pointer info
    std::unordered_map<std::string, TensorMmapInfo> tensor_mmap_info;
    
    // Copy mode: tensor data (empty in mmap mode)
    std::unordered_map<std::string, ov::Tensor> tensors;
};

/**
 * @brief Check if a directory contains safetensors model files
 * 
 * @param model_dir Path to model directory
 * @return true if safetensors files are found
 */
OPENVINO_GENAI_EXPORTS bool is_safetensors_model(const std::filesystem::path& model_dir);

/**
 * @brief Load all safetensors files from a HuggingFace model directory (Zero-Copy)
 * 
 * This function:
 * 1. Reads model.safetensors.index.json to find all shard files
 * 2. Loads each shard file using mmap for efficiency
 * 3. Stores tensor metadata WITHOUT copying data (zero-copy)
 * 
 * Use SafetensorsWeightSource::get_shared_buffer() to create SharedBuffer
 * for ov::Constant construction.
 * 
 * @param model_dir Path to the model directory
 * @return SafetensorsData containing tensor metadata and mmap holders
 * @throws std::runtime_error if loading fails
 */
OPENVINO_GENAI_EXPORTS SafetensorsData load_safetensors(const std::filesystem::path& model_dir);

/**
 * @brief Load a single safetensors file (Zero-Copy)
 * 
 * @param file_path Path to the .safetensors file
 * @return SafetensorsData containing tensor metadata
 */
OPENVINO_GENAI_EXPORTS SafetensorsData load_safetensors_file(const std::filesystem::path& file_path);

/**
 * @brief Map HuggingFace weight names to internal names
 * 
 * HuggingFace naming: model.layers.0.self_attn.q_proj.weight
 * Internal naming may differ based on the target format
 */
OPENVINO_GENAI_EXPORTS std::string map_hf_weight_name(const std::string& hf_name);

/**
 * @brief Convert safetensors dtype to OpenVINO element type
 */
OPENVINO_GENAI_EXPORTS ov::element::Type convert_dtype(int safetensors_dtype);

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
