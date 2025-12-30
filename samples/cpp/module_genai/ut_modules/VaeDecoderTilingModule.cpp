// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class VaeDecoderTilingModule : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(VaeDecoderTilingModule)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:
  vae_decoder_tiling:
    type: "VaeDecoderTilingModule"
    device: "CPU"
    inputs:
        - name: "latent"
        type: "OVTensor"
    outputs:
      - name: "latent"
        type: "OVTensor"
    params:
      tile_overlap_factor: "0.25"
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        auto latent = ut_randn_tensor(ov::Shape{1, 16, 240, 240}, 42);
        inputs["latent"] = latent;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("latent").as<ov::Tensor>();
        std::vector<float> expected_ouput = { 
          0.0279331, -0.0194968, -0.158097, 0.142582, -0.313633, -0.452601, 0.107033, 0.305759, -0.0610831, 0.136313
        };
        CHECK(compare_big_tensor(output, expected_ouput, 1e-2), "latent do not match expected values");
    }
};

REGISTER_MODULE_TEST(VaeDecoderTilingModule);
