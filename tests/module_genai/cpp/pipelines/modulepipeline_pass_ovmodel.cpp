// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <thread>

#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "module_genai/module_base.hpp"
#include "module_genai/module_factory.hpp"
#include "utils/load_image.hpp"
#include "utils/model_yaml.hpp"
#include "utils/utils.hpp"

// Test for ModulePipeline pass ov::Model as parameter.
// Config yaml only take model_path as param, can't pass ov::Model directly. So we need to
// pass ov::Model in models_map parameter of ModulePipeline constructor.

// Test focus on 3 cases:
// case_1. ModulePipeline with ov::Model passed in models_map, and main module use model from models_map.
// case_2. ModulePipeline with ov::Model passed in models_map, and submodule use model from param.
// case_3. Pass multiple ov::Model;

// Define test parameters:
// bool: Single ov::Model or multiple ov::Model;
// bool: main module or submodule use model from models_map
using test_params = std::tuple<bool, bool>;

// Define Dummy Modules for testing
namespace ov::genai::module {

static bool g_single_ov_model;
static bool g_main_module_use_model;
static int g_dummy_module_a_got_ovmodel_count = 0;
static int g_dummy_module_b_got_ovmodel_count = 0;

class DummyModuleA : public IBaseModule {
protected:
    DummyModuleA() = delete;
    DummyModuleA(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
        : IBaseModule(desc, pipeline_desc) {
        // First ov::Model will be init in IBaseModule constructor.
        if (!g_single_ov_model) {
            // Initialize second ov::Model from models_map
            m_ov_model_2 = get_ov_model_from_cfg_models_map("ov_model_2", true);
        }
    }
public:
    ~DummyModuleA() {};
    using PTR = std::shared_ptr<DummyModuleA>;
    static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) {
        // Can't use std::make_shared with protected ctor.
        return PTR(new DummyModuleA(desc, pipeline_desc));
    }
    static void print_static_config() {}

    void run() override {
        if (is_dummy_module_a()) {
            g_dummy_module_a_got_ovmodel_count += (m_ov_model == nullptr ? 0 : 1);
            g_dummy_module_a_got_ovmodel_count += (m_ov_model_2 == nullptr ? 0 : 1);
        } else {
            g_dummy_module_b_got_ovmodel_count += (m_ov_model == nullptr ? 0 : 1);
            g_dummy_module_b_got_ovmodel_count = m_ov_model_2 == nullptr ? 1 : 2;
        }
    }

private:
    std::shared_ptr<ov::Model> m_ov_model_2 = nullptr;
    bool is_dummy_module_a() { return (get_name() == "dummy_module_a"); }
};

REGISTER_MODULE_CONFIG(DummyModuleA);
GENAI_REGISTER_MODULE(ov::genai::module::ModuleType::DummyModuleBase, DummyModuleA);
}  // namespace ov::genai::module

class PipelineTestPassOvModel : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    bool _single_ov_model = true;
    bool _main_module_use_model = true;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& single_ov_model = std::get<0>(obj.param);
        const auto& main_module_use_model = std::get<1>(obj.param);
        std::string result;
        result += single_ov_model ? "Pass1OVModel_" : "PassMultipleOVModel_";
        result += main_module_use_model ? "MainModuleUseOVModel" : "SubModuleUseOVModel";
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_single_ov_model, _main_module_use_model) = GetParam();
        ov::genai::module::g_single_ov_model = _single_ov_model;
        ov::genai::module::g_main_module_use_model = _main_module_use_model;

        // load ov model.
        ov::Core core;
        auto model_1 = core.read_model(TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4() + "openvino_text_embeddings_model.xml");

        m_models_map.clear();
        std::string module_key = _main_module_use_model ? "dummy_module_a" : "dummy_module_b";
        m_models_map[module_key].clear();
        m_models_map[module_key]["ov_model"] = model_1;
        if (!_single_ov_model) {
            auto model_2 =
                core.read_model(TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4() + "openvino_vision_embeddings_model.xml");
            m_models_map[module_key]["ov_model_2"] = model_2;
        }
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "DummyModel";

        YAML::Node pipeline_modules = config["pipeline_modules"];
        // Modules graph
        /*          input_module
         *               |
         *         dummy_module_a --- submodule --> dummy_module_b
         *               |
         *          output_module
         */

        {
            YAML::Node input_module;
            input_module["type"] = "ParameterModule";
            YAML::Node outputs;
            outputs.push_back(output_node("input_data", "OVTensor"));
            input_module["outputs"] = outputs;
            pipeline_modules["input_node"] = input_module;
        }

        {
            YAML::Node dummy_module_a;
            dummy_module_a["type"] = "DummyModuleBase";
            YAML::Node inputs_a;
            inputs_a.push_back(input_node("input_data", "OVTensor", "input_node.input_data"));
            dummy_module_a["inputs"] = inputs_a;
            YAML::Node outputs_a;
            outputs_a.push_back(output_node("output_data", "OVTensor"));
            dummy_module_a["outputs"] = outputs_a;
            YAML::Node params;
            params["submodule"] = "submodule_dummy_module_b";
            dummy_module_a["params"] = params;
            pipeline_modules["dummy_module_a"] = dummy_module_a;
        }

        {
            YAML::Node output_module;
            output_module["type"] = "ResultModule";
            YAML::Node inputs;
            inputs.push_back(input_node("output_data", "OVTensor", "dummy_module_a.output_data"));
            output_module["inputs"] = inputs;
            pipeline_modules["output_node"] = output_module;
        }

        YAML::Node submodule;
        submodule["name"] = "submodule_dummy_module_b";
        {
            YAML::Node dummy_module_b;
            dummy_module_b["type"] = "DummyModuleBase";
            YAML::Node inputs;
            inputs.push_back(input_node("input_data", "OVTensor"));
            dummy_module_b["inputs"] = inputs;
            YAML::Node outputs;
            outputs.push_back(output_node("output_data", "OVTensor"));
            dummy_module_b["outputs"] = outputs;
            submodule.push_back(dummy_module_b);
        }
        config["submodules"] = submodule;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["input_data"] = ov::Tensor();
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Check DummyModuleA got ov::Model count
        int expected_model_count = _single_ov_model ? 1 : 2;
        if (_main_module_use_model) {
            EXPECT_EQ(ov::genai::module::g_dummy_module_a_got_ovmodel_count, expected_model_count);
            EXPECT_EQ(ov::genai::module::g_dummy_module_b_got_ovmodel_count, 0);
        } else {
            EXPECT_EQ(ov::genai::module::g_dummy_module_a_got_ovmodel_count, 0);
            EXPECT_EQ(ov::genai::module::g_dummy_module_b_got_ovmodel_count, expected_model_count);
        }
    }
};

TEST_P(PipelineTestPassOvModel, ModuleTest) {
    run();
}

static std::vector<test_params> g_test_params = {{true, true}, {true, false}, {false, true}};

INSTANTIATE_TEST_SUITE_P(PipelineTestSuite,
                         PipelineTestPassOvModel,
                         ::testing::ValuesIn(g_test_params),
                         PipelineTestPassOvModel::get_test_case_name);
