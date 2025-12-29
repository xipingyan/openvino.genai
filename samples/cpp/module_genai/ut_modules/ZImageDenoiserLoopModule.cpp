// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class ZImageDenoiserLoopModule : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(ZImageDenoiserLoopModule)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  denoiser_loop:
    type: "ZImageDenoiserLoopModule"
    device: "CPU"
    inputs:
      - name: "prompt_embed"
        type: "OVTensor"
        source: "pipeline_params.prompt_embed"
      - name: "num_inference_steps"
        type: "Int"
        source: "pipeline_params.num_inference_steps"
      - name: "width"
        type: "Int"
        source: "pipeline_params.width"
      - name: "height"
        type: "Int"
        source: "pipeline_params.height"
    outputs:
      - name: "latent"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        auto prompt_embed = ut_randn_tensor(ov::Shape{101, 2560}, 42);
        inputs["prompt_embed"] = prompt_embed;
        inputs["num_inference_steps"] = 5;
        inputs["width"] = 128;
        inputs["height"] = 128;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("latent").as<ov::Tensor>();
        std::vector<float> expected_ouput = { 
          0.329547, 0.660284, 0.467982, 0.449864, 0.523435, 1.11912, 0.244269, -0.327588, -0.83454, -1.47006
        };
        CHECK(compare_big_tensor(output, expected_ouput, 1e-2), "latent do not match expected values");
    }
};

REGISTER_MODULE_TEST(ZImageDenoiserLoopModule);
