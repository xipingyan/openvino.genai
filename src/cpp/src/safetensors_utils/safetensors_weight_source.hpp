// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>
#include <openvino/runtime/aligned_buffer.hpp>

#include "modeling/weights/weight_source.hpp"
#include "safetensors_utils/safetensors_loader.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Weight source for safetensors format
 *
 * This class provides access to weights loaded from HuggingFace safetensors files.
 * It handles name conversion from HF format (layers.N) to canonical format (layers[N])
 * for seamless integration with the modeling API.
 * 
 * The implementation uses mmap for efficient memory access. Tensors returned by
 * get_tensor() hold references to the mmap data, keeping it alive.
 * 
 * Usage:
 * @code
 * auto data = load_safetensors(model_dir);
 * SafetensorsWeightSource source(std::move(data));
 * const auto& tensor = source.get_tensor("layers[0].self_attn.q_proj.weight");
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS SafetensorsWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    /**
     * @brief Construct from loaded SafetensorsData (takes ownership)
     * 
     * @param data Loaded safetensors data (will be moved)
     */
    explicit SafetensorsWeightSource(SafetensorsData data);

    std::vector<std::string> keys() const override;
    bool has(const std::string& name) const override;
    
    /**
     * @brief Get tensor by name
     * 
     * Returns a tensor that references mmap data. The tensor holds the mmap
     * lifetime, so the data remains valid as long as the tensor is alive.
     * 
     * @param name Canonical weight name (e.g., "layers[0].self_attn.q_proj.weight")
     * @return Reference to the tensor
     */
    const ov::Tensor& get_tensor(const std::string& name) const override;

private:
    /**
     * @brief Get tensor metadata (dtype, shape) - used internally by get_tensor()
     */
    const TensorInfo& get_info(const std::string& name) const;

    /**
     * @brief Get HF name from canonical name
     */
    std::string get_hf_name(const std::string& canonical_name) const;
    
    SafetensorsData m_data;  // Owns the data (tensors for copy mode, mmap for mmap mode)
    std::vector<std::string> m_keys;  // Canonical names
    std::unordered_map<std::string, std::string> m_canonical_to_hf;  // canonical -> HF name mapping
    
    // Cache for mmap mode only - stores tensors created from mmap data
    // (In copy mode, tensors are already in m_data.tensors)
    mutable std::unordered_map<std::string, ov::Tensor> m_tensor_cache;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
