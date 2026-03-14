// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_source.hpp"

#include <stdexcept>

#include "loaders/weight_name_mapper.hpp"

namespace ov {
namespace genai {
namespace safetensors {

// =============================================================================
// SafetensorsWeightSource Implementation
// =============================================================================

SafetensorsWeightSource::SafetensorsWeightSource(SafetensorsData data)
    : m_data(std::move(data)) {
    
    // Build key list with canonical names (HF format -> canonical format)
    // tensor_infos is always populated by load_safetensors()
    m_keys.reserve(m_data.tensor_infos.size());
    for (const auto& [name, _] : m_data.tensor_infos) {
        std::string canonical = loaders::WeightNameMapper::from_hf(name);
        m_keys.push_back(canonical);
        m_canonical_to_hf[canonical] = name;
    }
}

std::vector<std::string> SafetensorsWeightSource::keys() const {
    return m_keys;
}

std::string SafetensorsWeightSource::get_hf_name(const std::string& canonical_name) const {
    // Check cache first
    auto cache_it = m_canonical_to_hf.find(canonical_name);
    if (cache_it != m_canonical_to_hf.end()) {
        return cache_it->second;
    }
    // Try dynamic conversion
    return loaders::WeightNameMapper::to_hf(canonical_name);
}

bool SafetensorsWeightSource::has(const std::string& name) const {
    std::string hf_name = get_hf_name(name);
    
    if (m_data.tensor_mmap_info.find(hf_name) != m_data.tensor_mmap_info.end()) {
        return true;
    }
    if (m_data.tensor_infos.find(hf_name) != m_data.tensor_infos.end()) {
        return true;
    }
    // Try direct name lookup
    if (m_data.tensor_mmap_info.find(name) != m_data.tensor_mmap_info.end()) {
        return true;
    }
    return m_data.tensor_infos.find(name) != m_data.tensor_infos.end();
}

const ov::Tensor& SafetensorsWeightSource::get_tensor(const std::string& name) const {
    // Copy mode: tensors already loaded in m_data.tensors
    if (!m_data.tensors.empty()) {
        std::string hf_name = get_hf_name(name);
        auto it = m_data.tensors.find(hf_name);
        if (it != m_data.tensors.end()) {
            return it->second;
        }
        it = m_data.tensors.find(name);
        if (it != m_data.tensors.end()) {
            return it->second;
        }
    }
    
    // Mmap mode: check cache first
    auto cache_it = m_tensor_cache.find(name);
    if (cache_it != m_tensor_cache.end()) {
        return cache_it->second;
    }
    
    // Create tensor from mmap data
    std::string hf_name = get_hf_name(name);
    auto it = m_data.tensor_mmap_info.find(hf_name);
    if (it == m_data.tensor_mmap_info.end()) {
        it = m_data.tensor_mmap_info.find(name);
    }
    if (it == m_data.tensor_mmap_info.end()) {
        throw std::runtime_error("Weight mmap info not found: " + name);
    }
    
    const auto& mmap_info = it->second;
    const auto& info = get_info(name);
    const uint8_t* data_ptr = mmap_info.holder->data_buffer() + mmap_info.offset;
    
    // Create tensor pointing to mmap data, binding mmap lifetime to tensor
    ov::Tensor view_tensor(info.dtype, info.shape, const_cast<void*>(static_cast<const void*>(data_ptr)));
    ov::Tensor tensor(view_tensor, mmap_info.holder);
    
    // Cache the tensor
    m_tensor_cache[name] = tensor;
    return m_tensor_cache[name];
}

const TensorInfo& SafetensorsWeightSource::get_info(const std::string& name) const {
    std::string hf_name = get_hf_name(name);
    
    auto it = m_data.tensor_infos.find(hf_name);
    if (it != m_data.tensor_infos.end()) {
        return it->second;
    }
    
    // Try direct name
    it = m_data.tensor_infos.find(name);
    if (it != m_data.tensor_infos.end()) {
        return it->second;
    }
    
    throw std::runtime_error("Weight info not found: " + name + " (tried: " + hf_name + ")");
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
