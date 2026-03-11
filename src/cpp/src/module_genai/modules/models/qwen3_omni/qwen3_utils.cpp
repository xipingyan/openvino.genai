#include "qwen3_utils.hpp"

#include <openvino/op/tile.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/interpolate.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/reshape.hpp>

namespace ov::genai::module {
namespace qwen3_utils {

std::shared_ptr<ov::Node> create_f32_nchw_input(std::shared_ptr<ov::Node> input) {
    auto raw_images_f32 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    auto img_trans = std::make_shared<ov::op::v1::Transpose>(
        raw_images_f32,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    return img_trans;
}

/**
 * Creates a bicubic resize operation using OpenVINO nodes
 * @param input The input tensor node to resize
 * @param target_size Node containing the target width and height [height, width]
 * @return Node representing the resized tensor
 */
std::shared_ptr<ov::Node> create_bicubic_resize(std::shared_ptr<ov::Node> input,
                                                const std::shared_ptr<ov::Node>& target_size) {
    // Create axes for height and width dimensions (assuming NCHW layout)
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});

    // Configure interpolation attributes for bicubic resize
    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.cube_coeff = -0.75f;  // Standard bicubic coefficient
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = false;

    // Create interpolate operation
    auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, target_size, axes, attrs);

    return interpolate;
}

/**
 * Creates a normalization operation using OpenVINO nodes
 * @param input The input tensor node to normalize (uint8 format)
 * @param mean Node containing the mean values for each channel
 * @param std Node containing the standard deviation values for each channel
 * @return Node representing the normalized tensor
 */
std::shared_ptr<ov::Node> create_normalization(std::shared_ptr<ov::Node> input,
                                               const std::shared_ptr<ov::Node>& mean,
                                               const std::shared_ptr<ov::Node>& std) {
    // clamp to 0 ~ 255
    auto image_clamp = std::make_shared<ov::op::v0::Clamp>(input, 0, 255);
    // Subtract mean
    auto mean_subtracted = std::make_shared<ov::op::v1::Subtract>(image_clamp, mean);

    // Divide by std
    auto normalized = std::make_shared<ov::op::v1::Multiply>(mean_subtracted, std);

    return normalized;
}

/**
 * @brief Creates a node that reshapes and transposes the input tensor to match the
 *        functionality of reshape_image_patches.
 * @param input The input node to reshape and transpose.
 * @param reshape_shape A constant node containing the target shape dimensions.
 * @return A node representing the reshaped and transposed tensor.
 */
std::shared_ptr<ov::Node> create_transpose_patches(std::shared_ptr<ov::Node> input,
                                                   const std::shared_ptr<ov::Node>& reshape_dims,
                                                   const std::shared_ptr<ov::Node>& transpose_order) {
    // Reshape input to the required dimensions
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, reshape_dims, true);

    // Transpose the reshaped tensor
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, transpose_order);

    return transposed;
}

std::shared_ptr<ov::Node> create_flatten_patches(std::shared_ptr<ov::Node> input,
                                                 const std::shared_ptr<ov::Node>& flatten_shape) {
    // Reshape (flatten) the input tensor
    auto flattened = std::make_shared<ov::op::v1::Reshape>(input, flatten_shape, true);

    return flattened;
}
}  // namespace qwen3_utils
}  // namespace ov::genai::module