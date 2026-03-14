// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <assert.h>
#include <stdio.h>

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <variant>

#include "openvino/openvino.hpp"

extern "C" {
#include <gguflib.h>
}

/**
 * @brief Extended tensor types for in-flight quantization
 * 
 * These types extend the gguf_tensor_type enum to support dynamic
 * quantization modes not present in the original GGUF format.
 * Values start at 100 to avoid collision with gguflib types.
 */
namespace ov_extended_types {

constexpr int GGUF_TYPE_INFLIGHT_INT4_SYM = 100;   ///< FP16 → INT4 symmetric quantization
constexpr int GGUF_TYPE_INFLIGHT_INT4_ASYM = 101;  ///< FP16 → INT4 asymmetric quantization
constexpr int GGUF_TYPE_INFLIGHT_INT8_SYM = 102;   ///< FP16 → INT8 symmetric quantization
constexpr int GGUF_TYPE_INFLIGHT_INT8_ASYM = 103;  ///< FP16 → INT8 asymmetric quantization
constexpr int GGUF_TYPE_AWQ_4BIT = 104;            ///< AWQ 4-bit pre-quantized format

/**
 * @brief Check if the type is an in-flight quantization type
 */
inline bool is_inflight_type(int type) {
    return type >= GGUF_TYPE_INFLIGHT_INT4_SYM && type <= GGUF_TYPE_INFLIGHT_INT8_ASYM;
}

/**
 * @brief Check if the type is INT4 (symmetric or asymmetric)
 */
inline bool is_int4_type(int type) {
    return type == GGUF_TYPE_INFLIGHT_INT4_SYM || type == GGUF_TYPE_INFLIGHT_INT4_ASYM;
}

/**
 * @brief Check if the type is INT8 (symmetric or asymmetric)
 */
inline bool is_int8_type(int type) {
    return type == GGUF_TYPE_INFLIGHT_INT8_SYM || type == GGUF_TYPE_INFLIGHT_INT8_ASYM;
}

/**
 * @brief Check if the type uses symmetric quantization
 */
inline bool is_symmetric_type(int type) {
    return type == GGUF_TYPE_INFLIGHT_INT4_SYM || type == GGUF_TYPE_INFLIGHT_INT8_SYM;
}

}  // namespace ov_extended_types

using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;

using GGUFLoad = std::tuple<std::unordered_map<std::string, GGUFMetaData>,
                            std::unordered_map<std::string, ov::Tensor>,
                            std::unordered_map<std::string, gguf_tensor_type>>;

template <typename... Args>
std::string format(std::string fmt, Args... args);

ov::Shape get_shape(const gguf_tensor& tensor);

void gguf_load_quantized(std::unordered_map<std::string, ov::Tensor>& a,
                         std::unordered_map<std::string, gguf_tensor_type>& qtype_map,
                         const gguf_tensor& tensor);

std::tuple<std::map<std::string, GGUFMetaData>,
           std::unordered_map<std::string, ov::Tensor>,
           std::unordered_map<std::string, gguf_tensor_type>>
load_gguf(const std::string& file);

GGUFLoad get_gguf_data(const std::string& file, bool is_tokenizer=false);
