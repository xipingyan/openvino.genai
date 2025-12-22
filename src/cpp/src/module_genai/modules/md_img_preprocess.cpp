// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_img_preprocess.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

void ImagePreprocesModule::print_static_config() {
    std::cout << R"(
  image_preprocessor:       # Module Name
    type: "ImagePreprocessModule"
    device: "CPU"
    description: "Image or Video preprocessing."
    inputs:
      - name: "InputName_1"     # single image
        type: "OVTensor"        # Support DataType: [OVTensor, OVRemoteTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "InputName_2"     # multiple images
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "raw_data"        # Output port name
        type: "OVTensor"        # Support DataType: [OVTensor, OVRemoteTensor]
      - name: "source_size"     # Output port name
        type: "VecInt"          # Support DataType: [VecInt]
    params:
      target_resolution: [224, 224]   # optional
      mean: [0.485, 0.456, 0.406]     # optional
      std: [0.229, 0.224, 0.225]      # optional
      model_path: "models/openvino_vision_embeddings_model.xml"
    )" << std::endl;
}

ImagePreprocesModule::ImagePreprocesModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    std::string model_path = desc->get_full_path(desc->params["model_path"]);
    std::string device = desc->params["device"];
    if (device.empty()) {
        device = "CPU";
    }

    encoder_ptr = std::make_shared<VisionEncoderQwen2VL>(std::filesystem::path(model_path), device, ov::AnyMap{});
}

ImagePreprocesModule::~ImagePreprocesModule() {}

void ImagePreprocesModule::run() {
    prepare_inputs();

    auto image1_data = this->inputs["image1_data"].data.as<ov::Tensor>();
    auto encoded_img = encoder_ptr->encode(image1_data, ov::AnyMap{});
  
    auto thw_tensor = ov::Tensor(ov::element::i32, ov::Shape(3));
    thw_tensor.data<int>()[0] = 3;
    thw_tensor.data<int>()[1] = 4;
    thw_tensor.data<int>()[2] = 5;
    this->outputs["raw_data"].data = encoded_img.resized_source;
    this->outputs["source_size"].data = std::vector<int>{encoded_img.resized_source_size.height, encoded_img.resized_source_size.width};
}

}  // namespace module
}  // namespace genai
}  // namespace ov
