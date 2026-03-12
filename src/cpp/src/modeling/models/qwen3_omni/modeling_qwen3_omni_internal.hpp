// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

std::filesystem::path resolve_qwen3_omni_config_path(const std::filesystem::path& path);
void read_qwen3_omni_json(const std::filesystem::path& path, nlohmann::json& data);
Qwen3VLConfig to_qwen3_omni_vl_cfg(const Qwen3OmniConfig& cfg);

class PrefixMappedWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    PrefixMappedWeightSource(ov::genai::modeling::weights::WeightSource& source, std::string prefix)
        : source_(source),
          prefix_(std::move(prefix)) {}

    std::vector<std::string> keys() const override {
        auto names = source_.keys();
        std::vector<std::string> filtered;
        filtered.reserve(names.size());
        for (const auto& name : names) {
            if (name.rfind(prefix_, 0) == 0) {
                filtered.push_back(name.substr(prefix_.size()));
            }
        }
        return filtered;
    }

    bool has(const std::string& name) const override {
        return source_.has(name) || source_.has(prefix_ + name);
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        if (source_.has(name)) {
            return source_.get_tensor(name);
        }
        return source_.get_tensor(prefix_ + name);
    }

    void release_tensor(const std::string& name) override {
        if (source_.has(name)) {
            source_.release_tensor(name);
            return;
        }
        source_.release_tensor(prefix_ + name);
    }

private:
    ov::genai::modeling::weights::WeightSource& source_;
    std::string prefix_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
