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
    inputs:
      - name: "InputName_1"     # single image
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "InputName_2"     # multiple images
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "raw_data"        # Output port name
        type: "OVTensor"        # DataType
      - name: "thw"
        type: "OVTensor"
    params:
      target_resolution: [224, 224]   # optional
      mean: [0.485, 0.456, 0.406]     # optional
      std: [0.229, 0.224, 0.225]      # optional
    )" << std::endl;
}

ImagePreprocesModule::ImagePreprocesModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {}

inline ov::Tensor get_input_tensor(const std::string& name,
                                   std::map<std::string, IBaseModule::InputModule>& inputs,
                                   const bool& optional = false) {
    // inputs[name].module_ptr->outputs
    // for (auto& input : inputs) {
    //     if (input.out_port_name == name) {
    //         input.module_ptr->outputs
    //         return input.data.as<ov::Tensor>();
    //     }
    // }
    OPENVINO_ASSERT(!optional, "Can't find input tensor with name " + name);
    return ov::Tensor();
}

void ImagePreprocesModule::run() {
    prepare_inputs();

    auto image1_data = this->inputs["image1_data"].data.as<ov::Tensor>();
    auto image2_data = this->inputs["image2_data"].data.as<ov::Tensor>();

    ov::Tensor img_f32 = ov::Tensor(ov::element::f32, image1_data.get_shape());
    float* out_data = img_f32.data<float>();
    if (image1_data.get_element_type() == ov::element::u8) {
        uint8_t* in_data = image1_data.data<uint8_t>();
        for (int i = 0; i < image1_data.get_size(); i++) {
            out_data[i] = in_data[i] / 2;
        }
    } else if (image1_data.get_element_type() == ov::element::f32) {
        float* in_data = image1_data.data<float>();
        for (int i = 0; i < image1_data.get_size(); i++) {
            out_data[i] = in_data[i] / 2;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(700));

    auto thw_tensor = ov::Tensor(ov::element::i32, ov::Shape(3));
    thw_tensor.data<int>()[0] = 3;
    thw_tensor.data<int>()[1] = 4;
    thw_tensor.data<int>()[2] = 5;
    this->outputs["raw_data"].data = img_f32;
    this->outputs["thw"].data = thw_tensor;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
