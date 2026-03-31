// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"

using namespace ov::genai::module;

// Parameters for test:
//   string: input mode ("input_ids")
//   string: device
using test_params = std::tuple<std::string, std::string>;

class LLMInferencePAModuleTest : public ModuleTestBase,
                                 public ::testing::TestWithParam<test_params> {
private:
    std::string m_mode;
    std::string m_device;

    const std::string m_module_name = "pa_llm_infer";

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& mode = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        std::string result;
        result += "Device_" + sanitize_for_gtest(device);
        result += "_Mode_" + sanitize_for_gtest(mode);
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_mode, m_device) = GetParam();
    }

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen2_5_vl";

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node llm_pa;
        llm_pa["type"] = "LLMInferencePAModule";
        llm_pa["device"] = m_device;
        llm_pa["description"] = "LLM Inference PA Module test.";

        YAML::Node inputs;
        if (m_mode == "input_ids") {
            inputs.push_back(input_node("input_ids", "OVTensor"));
        } else {
            OPENVINO_THROW("Unsupported test mode: " + m_mode);
        }
        llm_pa["inputs"] = inputs;

        YAML::Node outputs;
        outputs.push_back(output_node("generated_text", "String"));
        llm_pa["outputs"] = outputs;

        YAML::Node params;
        params["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        params["max_new_tokens"] = "16";
        llm_pa["params"] = params;

        pipeline_modules[m_module_name] = llm_pa;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        if (m_mode == "input_ids") {
            std::vector<int64_t> token_values = {9707};
            ov::Tensor input_ids(ov::element::i64, {1, token_values.size()});
            std::copy(token_values.begin(), token_values.end(), input_ids.data<int64_t>());
            inputs["input_ids"] = input_ids;
            return inputs;
        }

        OPENVINO_THROW("Unsupported test mode: " + m_mode);
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        const auto generated_text = pipe.get_output("generated_text").as<std::string>();
        EXPECT_FALSE(generated_text.empty()) << "Generated text should not be empty";
    }
};

TEST_P(LLMInferencePAModuleTest, ModuleTest) {
    run();
}

static std::vector<test_params> g_test_params = {
    {"input_ids", TEST_MODEL::get_device()},
};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         LLMInferencePAModuleTest,
                         ::testing::ValuesIn(g_test_params),
                         LLMInferencePAModuleTest::get_test_case_name);
