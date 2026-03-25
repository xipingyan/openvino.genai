// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_img_preprocess.hpp"

#include "module_genai/pipeline/module_factory.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/modules/models/qwen3_5/qwen3_5preprocessor.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(ImagePreprocessModule);

const ModuleSpec& ImagePreprocessModule::get_spec() {
  static const ModuleSpec spec = []() {
    ModuleSpec s("ImagePreprocessModule", "image_preprocessor");
    s.add_input("image", {DataType::OVTensor}, true);
    s.add_input("images", {DataType::VecOVTensor}, true);
    s.add_output("raw_data", {DataType::OVTensor});
    s.add_output("source_size", {DataType::VecInt});
    s.add_output("raw_datas", {DataType::VecOVTensor});
    s.add_output("source_sizes", {DataType::VecVecInt});
    s.add_output("pixel_values", {DataType::VecOVTensor});
    s.add_output("grid_thw", {DataType::VecOVTensor});
    s.add_output("pos_embeds", {DataType::VecOVTensor});
    s.add_output("rotary_cos", {DataType::VecOVTensor});
    s.add_output("rotary_sin", {DataType::VecOVTensor});
    s.add_param("model_path", "models/openvino_vision_embeddings_model.xml");
    return s;
  }();
  return spec;
}

void ImagePreprocessModule::print_static_config() {
    std::cout << get_spec().to_yaml_template_string() << std::endl;
}

ImagePreprocessModule::ImagePreprocessModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
  check_params_with_spec(get_spec());
    std::string model_path = desc->get_full_path(desc->params["model_path"]);
    std::string device = desc->device;

    _model_type = to_vlm_model_type(desc->model_type);

    _vision_preprocess_ptr = VisionPreprocess::create(model_path, device, _model_type);
    if (_vision_preprocess_ptr == nullptr) {
        _encoder_ptr = VisionEncoder::create(model_path, _model_type, device);
        OPENVINO_ASSERT(_encoder_ptr != nullptr,
                        "Failed to create VisionEncoder for ImagePreprocessModule: " + desc->name);
    }
}

ImagePreprocessModule::~ImagePreprocessModule() {}

void ImagePreprocessModule::run_image(const bool& has_images_input) {
    std::vector<ov::Tensor> images_data;
    if (has_images_input) {
        images_data = get_input("images").as<std::vector<ov::Tensor>>();
    } else {
        images_data.push_back(get_input("image").as<ov::Tensor>());
    }

    if (_vision_preprocess_ptr) {
        auto output = _vision_preprocess_ptr->preprocess(images_data, {});
        this->outputs["pixel_values"].data = output.pixel_values;
        this->outputs["grid_thw"].data = output.grid_thw;
        this->outputs["pos_embeds"].data = output.pos_embeds;
        this->outputs["rotary_cos"].data = output.rotary_cos;
        this->outputs["rotary_sin"].data = output.rotary_sin;
    } else {
        std::vector<ov::Tensor> output_tensors;
        std::vector<ImageSize> output_sizes;
        for (size_t i = 0; i < images_data.size(); ++i) {
            auto encoded_img = _encoder_ptr->encode(images_data[i], ov::AnyMap{});
            output_tensors.push_back(encoded_img.resized_source);
            output_sizes.push_back(encoded_img.resized_source_size);
        }

        if (has_images_input) {
            this->outputs["raw_datas"].data = output_tensors;
            std::vector<std::vector<int>> sizes_vec;
            for (const auto& sz : output_sizes) {
                sizes_vec.push_back({static_cast<int>(sz.height), static_cast<int>(sz.width)});
            }
            this->outputs["source_sizes"].data = sizes_vec;
        } else {
            this->outputs["raw_data"].data = output_tensors[0];
            this->outputs["source_size"].data =
                std::vector<int>{static_cast<int>(output_sizes[0].height), static_cast<int>(output_sizes[0].width)};
        }
    }
}

void ImagePreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    bool has_images_input = exists_input("images");
    bool has_image_input = exists_input("image");
    bool has_image = has_images_input || has_image_input;
    if (has_image) {
        OPENVINO_ASSERT(
            !(has_images_input && has_image_input),
            "ImagePreprocessModule: Both 'image' and 'images' inputs exist. Please provide only one of them.");
    }

    if (has_images_input || has_image_input) {
      run_image(has_images_input);
    } else {
        OPENVINO_THROW("ImagePreprocessModule[" + module_desc->name +
                       "]: No valid input found. Please provide one of the following inputs: 'image', 'images'.");
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
