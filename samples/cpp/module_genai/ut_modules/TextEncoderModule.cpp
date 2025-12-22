// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class TextEncoderModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(TextEncoderModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "prompts_data"
        type: "String"

  prompt_encoder:
    type: "TextEncoderModule"
    device: "GPU"
    inputs:
      - name: "prompts"
        type: "String"
        source: "pipeline_params.prompts_data"
    outputs:
      - name: "input_ids"
        type: "OVTensor"
      - name: "mask"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "input_ids"
        type: "OVTensor"
        source: "prompt_encoder.input_ids"
      - name: "mask"
        type: "OVTensor"
        source: "prompt_encoder.mask"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["prompts_data"] = std::vector<std::string>{"This is a sample prompt."};
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("input_ids").as<ov::Tensor>();
        auto expected_input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* data_ptr = expected_input_ids.data<int64_t>();
        std::vector<int64_t> values = {1986, 374, 264, 6077, 9934, 13};
        std::copy(values.begin(), values.end(), data_ptr);

        CHECK(compare_tensors(output, expected_input_ids), "input_ids do not match expected values");

        auto mask = pipe.get_output("mask").as<ov::Tensor>();
        auto expected_mask = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* mask_data_ptr = expected_mask.data<int64_t>();
        std::vector<int64_t> mask_values = {1, 1, 1, 1, 1, 1};
        std::copy(mask_values.begin(), mask_values.end(), mask_data_ptr);
        CHECK(compare_tensors(mask, expected_mask), "mask not match expected values");
    }
};

REGISTER_MODULE_TEST(TextEncoderModuleTest);
