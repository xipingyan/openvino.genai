// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <thread>
#include <chrono>
#include <filesystem>
#include <openvino/genai/module_genai/pipeline.hpp>

#include "utils/load_image.hpp"
#include "utils/utils.hpp"
#include "utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/model_yaml.hpp"


// Define test parameters: 
// bool: generate_async or generate;
// vector<int>: module ids of sync execution order when run: generate_async();
using test_params = std::tuple<bool, std::vector<std::string>>;
namespace ov::genai::module {
    extern std::thread::id FakeModuleA_thread_id;
    extern std::thread::id FakeModuleB_thread_id;
}

class PipelineGenerateAsyncSubmodulesTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::vector<std::string> _sync_execution_module_names;
    bool not_in_sync_execution_modules(const std::string& module_name) const {
        return std::find(_sync_execution_module_names.begin(), _sync_execution_module_names.end(), module_name) ==
               _sync_execution_module_names.end();
    }

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& async = std::get<0>(obj.param);
        const auto& sync_exec_module_names = std::get<1>(obj.param);
        std::string result;
        result += std::string("ThreadMode_") + (async ? "Async" : "Sync");
        if (async) {
            result += "_SyncModules_";
            for (size_t i = 0; i < sync_exec_module_names.size(); ++i) {
                result += sync_exec_module_names[i];
                if (i < sync_exec_module_names.size() - 1)
                    result += "_";
            }
            if (sync_exec_module_names.size() == 0) {
                result += "None";
            }
        }
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_async, _sync_execution_module_names) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "FakeModel";

        YAML::Node pipeline_modules = config["pipeline_modules"];
        // Modules graph
        /*          input_module
         *         /            \              submodules
         *        /              \                  |
         *  fake_module_a    fake_module_b --> fake_module_c
         *         \            /
         *          \          /
         *          output_module
         */

        {
            YAML::Node input_module;
            input_module["type"] = "ParameterModule";
            YAML::Node outputs;
            outputs.push_back(output_node("input_data_1", "OVTensor"));
            outputs.push_back(output_node("input_data_2", "OVTensor"));
            input_module["outputs"] = outputs;
            pipeline_modules["input_node"] = input_module;
        }

        {
            YAML::Node fake_module_a;
            fake_module_a["type"] = "FakeModuleA";
            YAML::Node inputs_a;
            inputs_a.push_back(input_node("input_data", "OVTensor", "input_node.input_data_1"));
            fake_module_a["inputs"] = inputs_a;
            YAML::Node outputs_a;
            outputs_a.push_back(output_node("output_data", "OVTensor"));
            fake_module_a["outputs"] = outputs_a;
            pipeline_modules["fake_module_a"] = fake_module_a;

            fake_module_a["thread_mode"] = (m_async && not_in_sync_execution_modules("fake_module_a")) ? "ASYNC" : "SYNC";
        }

        {
            YAML::Node fake_module_b;
            fake_module_b["type"] = "FakeModuleB";
            YAML::Node inputs_b;
            inputs_b.push_back(input_node("input_data", "OVTensor", "input_node.input_data_2"));
            fake_module_b["inputs"] = inputs_b;
            YAML::Node outputs_b;
            outputs_b.push_back(output_node("output_data", "OVTensor"));
            fake_module_b["outputs"] = outputs_b;
            pipeline_modules["fake_module_b"] = fake_module_b;

            fake_module_b["thread_mode"] = (m_async && not_in_sync_execution_modules("fake_module_b")) ? "ASYNC" : "SYNC";
        }

        {
            YAML::Node output_module;
            output_module["type"] = "ResultModule";
            YAML::Node inputs;
            inputs.push_back(input_node("output_data_1", "OVTensor", "fake_module_a.output_data"));
            inputs.push_back(input_node("output_data_2", "OVTensor", "fake_module_b.output_data"));
            output_module["inputs"] = inputs;
            pipeline_modules["output_node"] = output_module;
        }

        //  submodules
        YAML::Node submodules;
        {
            YAML::Node submodule_entry;
            submodule_entry["name"] = "fake_submodule_pipeline";
            submodules.push_back(submodule_entry);
        }
        config["submodules"] = submodules;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["input_data_1"] = ov::Tensor();
        inputs["input_data_2"] = ov::Tensor();
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        if (m_async && _sync_execution_module_names.size() == 0) {
            EXPECT_NE(ov::genai::module::FakeModuleA_thread_id, ov::genai::module::FakeModuleB_thread_id);
        } else {
            EXPECT_EQ(ov::genai::module::FakeModuleA_thread_id, ov::genai::module::FakeModuleB_thread_id);
        }
    }
};

TEST_P(PipelineGenerateAsyncSubmodulesTest, ModuleTest) {
    run();
}

static std::vector<test_params> g_test_params = {
    {true, {"fake_module_a", "fake_module_b"}},
    {true, {}},
    {false, {}}
};

INSTANTIATE_TEST_SUITE_P(
    PipelineTestSuite,
    PipelineGenerateAsyncSubmodulesTest,
    ::testing::ValuesIn(g_test_params),
    PipelineGenerateAsyncSubmodulesTest::get_test_case_name
);
