// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/quantization_selector.hpp"

namespace ov {
namespace genai {

// Forward declare QuantizedWeight
namespace rtn {
struct QuantizedWeight;
}

namespace safetensors {

// Re-export modeling API types for convenience and backward compatibility
using QuantizationConfig = ov::genai::modeling::weights::QuantizationConfig;
using WeightSelectionConfig = ov::genai::modeling::weights::WeightSelectionConfig;

/**
 * @brief Weight finalizer for safetensors format with optional in-flight quantization
 *
 * Supports:
 * - Direct weight loading (F32/F16/BF16)
 * - In-flight INT4/INT8 quantization during loading
 * - Zero-copy mode for efficiency
 * - Custom weight selection for quantization
 */
class OPENVINO_GENAI_EXPORTS SafetensorsWeightFinalizer : public ov::genai::modeling::weights::WeightFinalizer {
public:
    SafetensorsWeightFinalizer();
    explicit SafetensorsWeightFinalizer(const QuantizationConfig& config);
    ~SafetensorsWeightFinalizer() override;

    ov::genai::modeling::weights::FinalizedWeight finalize(
        const std::string& name,
        ov::genai::modeling::weights::WeightSource& source,
        ov::genai::modeling::OpContext& ctx) override;

private:
    /**
     * @brief Perform in-flight quantization on a tensor
     * @return QuantizedWeight struct containing compressed weights, scales, and zero-points
     */
    rtn::QuantizedWeight quantize_weight(
        const std::string& name,
        const ov::Tensor& tensor,
        ov::genai::modeling::weights::WeightSource& source,
        ov::genai::modeling::OpContext& ctx);
    
    /**
     * @brief Direct FP8 to INT4/INT8 quantization without intermediate F32 tensor
     * 
     * This is a memory-optimized version that processes FP8 weights directly,
     * avoiding the creation of a full F32 tensor which would consume 4x memory.
     * 
     * @param fp8_tensor Input FP8 weight tensor
     * @param scale_inv FP8 scale tensor
     * @return QuantizedWeight struct containing compressed weights, scales, and zero-points
     */
    rtn::QuantizedWeight quantize_fp8_weight_direct(
        const std::string& name,
        const ov::Tensor& fp8_tensor,
        const ov::Tensor& scale_inv,
        ov::genai::modeling::weights::WeightSource& source,
        ov::genai::modeling::OpContext& ctx);
    
    /**
     * @brief Create dequantization subgraph from quantized result
     * Supports both INT4 (packed in U8) and INT8 quantization
     */
    ov::Output<ov::Node> create_dequant_subgraph(
        const std::string& name,
        const rtn::QuantizedWeight& quant_result,
        const ov::Shape& original_shape);
    
    /**
     * @brief Check if weight name corresponds to a MoE weight
     */
    bool is_moe_weight(const std::string& name) const;
    
    /**
     * @brief Create MoE-specific subgraph with accessible scales and zero-points
     * Returns FinalizedWeight with dequantized weight as primary and scales/zps in auxiliary
     */
    ov::genai::modeling::weights::FinalizedWeight create_moe_subgraph(
        const std::string& name,
        const rtn::QuantizedWeight& quant_result,
        const ov::Shape& original_shape,
        ov::genai::modeling::OpContext& ctx);

    bool has_fp8_scale_inv(ov::genai::modeling::weights::WeightSource& source,
                           const std::string& name) const;

    ov::Output<ov::Node> create_fp8_dequant_subgraph(
        const std::string& name,
        const ov::Tensor& weight_fp8,
        const ov::Tensor& scale_inv,
        ov::genai::modeling::OpContext& ctx);

    /**
     * @brief Dequantize FP8 tensor to F32 in CPU memory
     * Used when we want to further quantize the weight to INT4/INT8
     */
    ov::Tensor dequantize_fp8_to_f32(
        const ov::Tensor& weight_fp8,
        const ov::Tensor& scale_inv) const;

    ov::Output<ov::Node> expand_block_scale_to_weight(
        const ov::Tensor& scale_inv,
        const ov::Shape& weight_shape,
        ov::genai::modeling::OpContext& ctx) const;

    std::vector<int64_t> make_interleaved_scale_shape(const ov::Shape& scale_shape) const;

    std::vector<int64_t> make_interleaved_tile_repeats(const ov::Shape& scale_shape,
                                                       const ov::Shape& weight_shape) const;

    std::unordered_map<std::string, ov::Output<ov::Node>> cache_;
    ov::genai::modeling::weights::QuantizationSelector selector_;

    // Statistics
    size_t total_weights_ = 0;
    size_t quantized_weights_ = 0;

    // Timing stats
    double total_fetch_time_ms_ = 0;
    double total_quant_time_ms_ = 0;
    double total_graph_time_ms_ = 0;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
