#pragma once

#include <memory>

#include "openvino/core/node.hpp"

namespace ov::genai::module {
namespace qwen3_utils {

std::shared_ptr<ov::Node> create_f32_nchw_input(std::shared_ptr<ov::Node> input);

/**
 * Creates a bicubic resize operation using OpenVINO nodes
 * @param input The input tensor node to resize
 * @param target_size Node containing the target width and height [height, width]
 * @return Node representing the resized tensor
 */
std::shared_ptr<ov::Node> create_bicubic_resize(std::shared_ptr<ov::Node> input,
                                                const std::shared_ptr<ov::Node>& target_size);

/**
 * Creates a normalization operation using OpenVINO nodes.
 * @param input The input tensor node to normalize. Expected to be float32
 *              (e.g. converted from uint8 image data in [0, 255] range).
 * @param mean  Node containing the mean values for each channel (float32),
 *              broadcastable to the input tensor.
 * @param scale Node containing the per-channel multiplicative scale
 *              (e.g. 1 / (std * 255)) applied after mean subtraction,
 *              broadcastable to the input tensor.
 * @return Node representing the normalized tensor.
 */
std::shared_ptr<ov::Node> create_normalization(std::shared_ptr<ov::Node> input,
                                               const std::shared_ptr<ov::Node>& mean,
                                               const std::shared_ptr<ov::Node>& scale);

/**
 * @brief Creates a node that reshapes and transposes the input tensor to match the
 *        functionality of reshape_image_patches.
 * @param input The input node to reshape and transpose.
 * @param reshape_shape A constant node containing the target shape dimensions.
 * @return A node representing the reshaped and transposed tensor.
 */
std::shared_ptr<ov::Node> create_transpose_patches(std::shared_ptr<ov::Node> input,
                                                   const std::shared_ptr<ov::Node>& reshape_dims,
                                                   const std::shared_ptr<ov::Node>& transpose_order);

std::shared_ptr<ov::Node> create_flatten_patches(std::shared_ptr<ov::Node> input,
                                                 const std::shared_ptr<ov::Node>& flatten_shape);
}
}  // namespace ov::genai::module
