// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/module_genai/pipeline.hpp"
#include "module_genai/module_base.hpp"

namespace ov {
namespace genai {

namespace module {

struct DataTypeConverter {
private:
    static const std::unordered_map<DataType, std::string> kTypeToString;

    static const std::unordered_map<std::string, DataType> kStringToType;

    static std::unordered_map<std::string, DataType> create_string_to_type_map() {
        std::unordered_map<std::string, DataType> map;
        for (const auto& pair : kTypeToString) {
            map[pair.second] = pair.first;
        }
        return map;
    }

public:
    static std::string toString(DataType type) {
        auto it = kTypeToString.find(type);
        OPENVINO_ASSERT(it != kTypeToString.end(), "Unknown DataType value: " + std::to_string(static_cast<int>(type)));
        return it->second;
    }

    static DataType fromString(const std::string& str) {
        auto it = kStringToType.find(str);
        OPENVINO_ASSERT(it != kStringToType.end(), "Unknown DataType string: " + str);
        return it->second;
    }
};
}  // namespace module
}  // namespace genai
}  // namespace ov