// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_weight_source.hpp"

#include <algorithm>

#include <openvino/core/except.hpp>

#include "loaders/weight_name_mapper.hpp"

namespace {

bool ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

namespace ov {
namespace genai {
namespace gguf {

GGUFWeightSource::GGUFWeightSource(const std::unordered_map<std::string, ov::Tensor>& consts) : consts_(consts) {
    keys_.reserve(consts_.size());
    for (const auto& kv : consts_) {
        if (ends_with(kv.first, ".weight")) {
            // Convert GGUF name to canonical name
            std::string canonical = loaders::WeightNameMapper::from_gguf(kv.first);
            keys_.push_back(canonical);
            canonical_to_gguf_[canonical] = kv.first;
        }
    }
    std::sort(keys_.begin(), keys_.end());
}

std::vector<std::string> GGUFWeightSource::keys() const {
    return keys_;
}

bool GGUFWeightSource::has(const std::string& name) const {
    // Try canonical name lookup
    if (canonical_to_gguf_.count(name) > 0) {
        return true;
    }
    // Try direct GGUF name lookup
    return consts_.count(name) > 0;
}

const ov::Tensor& GGUFWeightSource::get_tensor(const std::string& name) const {
    // Try canonical name lookup first
    auto it = canonical_to_gguf_.find(name);
    if (it != canonical_to_gguf_.end()) {
        auto tensor_it = consts_.find(it->second);
        if (tensor_it != consts_.end()) {
            return tensor_it->second;
        }
    }
    
    // Try direct GGUF name lookup
    auto direct_it = consts_.find(name);
    if (direct_it != consts_.end()) {
        return direct_it->second;
    }
    
    OPENVINO_THROW("Unknown GGUF tensor: ", name);
}

}  // namespace gguf
}  // namespace genai
}  // namespace ov
