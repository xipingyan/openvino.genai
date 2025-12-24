// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <string>
#include <vector>

#include "utils.hpp"
#include "load_image.hpp"
#include <yaml-cpp/yaml.h>

#ifndef CHECK
#    define CHECK(cond, msg)                                                   \
        do {                                                                   \
            if (!(cond)) {                                                     \
                throw std::runtime_error(std::string("Check failed: ") + msg); \
            }                                                                  \
        } while (0)
#endif

class ModuleTestBase {
public:
    virtual ~ModuleTestBase() = default;

    void run() {
        std::string config_path = generate_yaml();

        ov::genai::module::ModulePipeline pipe(config_path);

        ov::AnyMap inputs = prepare_inputs();
        pipe.generate(inputs);

        verify_outputs(pipe);

        // Cleanup
        if (std::filesystem::exists(config_path)) {
            std::filesystem::remove(config_path);
        }
    }

protected:
    std::string m_test_name;
    virtual std::string get_yaml_content() = 0;
    virtual ov::AnyMap prepare_inputs() = 0;
    virtual void verify_outputs(ov::genai::module::ModulePipeline& pipe) = 0;

    std::string save_yaml(const YAML::Node& config) {
        std::string filename = "temp_" + m_test_name + ".yaml";
        std::ofstream out(filename);
        out << config;
        out.close();
        return filename;
    }

    virtual std::string generate_yaml() {
        std::string cur_module_cfg = get_yaml_content();
        YAML::Node config = YAML::Load(cur_module_cfg);

        OPENVINO_ASSERT(config["pipeline_modules"], "Test yaml config miss 'pipeline_modules'.");

        YAML::Node modules = config["pipeline_modules"];
        std::map<std::string, std::string> extracted_params;
        std::map<std::string, std::string> extracted_results;

        if (modules.size() != 1) {
            return save_yaml(config);
        }

        // 1. 遍历提取数据
        std::string test_module_name;
        for (auto it = modules.begin(); it != modules.end(); ++it) {
            test_module_name = it->first.as<std::string>();
            // 处理 inputs
            YAML::Node inputs = it->second["inputs"];
            if (inputs && inputs.IsSequence()) {
                for (const auto& input : inputs) {
                    std::string source = input["source"].as<std::string>("");
                    if (source.find("pipeline_params.") == 0) {
                        std::string param_name = source.substr(16);
                        std::string type = input["type"].as<std::string>("");
                        extracted_params[param_name] = type;
                    }
                }
            }

            // 处理 outputs
            YAML::Node outputs = it->second["outputs"];
            if (outputs && outputs.IsSequence()) {
                for (const auto& output : outputs) {
                    std::string name = output["name"].as<std::string>("");
                    std::string type = output["type"].as<std::string>("");
                    extracted_results[name] = type;
                }
            }
        }

        // 2. 构建 pipeline_params 节点 (使用独立的 Node)
        YAML::Node params_node;
        params_node["type"] = "ParameterModule";
        YAML::Node p_outputs_seq;  // 创建 Sequence 节点
        for (const auto& param : extracted_params) {
            YAML::Node item;
            item["name"] = param.first;
            item["type"] = param.second;
            p_outputs_seq.push_back(item);
        }
        if (p_outputs_seq.size() > 0) {
            params_node["outputs"] = p_outputs_seq;
        }
        config["pipeline_modules"]["pipeline_params"] = params_node;

        // 3. 构建 pipeline_results 节点 (使用另一个独立的 Node)
        YAML::Node results_node;
        results_node["type"] = "ResultModule";
        YAML::Node r_inputs_seq;  // 创建 Sequence 节点
        for (const auto& result : extracted_results) {
            YAML::Node item;
            item["name"] = result.first;
            item["type"] = result.second;
            item["source"] = test_module_name + "." + result.first;
            r_inputs_seq.push_back(item);
        }
        if (r_inputs_seq.size() > 0) {
            results_node["inputs"] = r_inputs_seq;
        }
        config["pipeline_modules"]["pipeline_results"] = results_node;

        // 4. 导出文件
        std::string filename = "temp_" + m_test_name + ".yaml";
        std::ofstream out(filename);
        out << config;  // 直接输出 config
        out.close();

        return filename;
    }

    virtual bool compare_tensors(const ov::Tensor& output, const ov::Tensor& expected) {
        if (output.get_shape() != expected.get_shape() || output.get_element_type() != expected.get_element_type()) {
            return false;
        }
        size_t byte_size = output.get_byte_size();
        return std::memcmp(output.data(), expected.data(), byte_size) == 0;
    }

    template<typename T>
    bool compare_big_tensor(const ov::Tensor& output, const std::vector<T>& expected_top) {
        int real_size = std::min(expected_top.size(), output.get_size());
        bool bresult = true;
        for (int i = 0; i < real_size; ++i) {
            T val = static_cast<T>(output.data<T>()[i]);
            if (val != expected_top[i]) {
                bresult = false;
                std::cout << "Mismatch at index " << i << ": expected " << expected_top[i] << ", got " << val << std::endl;
            }
        }
        return bresult;
    }

    bool compare_big_tensor(const ov::Tensor& output, const std::vector<float>& expected_top, const float& thr = 1e-3) {
        int real_size = std::min(expected_top.size(), output.get_size());
        bool bresult = true;
        for (int i = 0; i < real_size; ++i) {
            float val = static_cast<float>(output.data<float>()[i]);
            if (std::fabs(val - expected_top[i]) > thr) {
                bresult = false;
                std::cout << "Mismatch at index " << i << ": expected " << expected_top[i] << ", got " << val << std::endl;
            }
        }
        return bresult;
    }

    bool compare_shape(const ov::Shape& shape1, const ov::Shape& shape2) {
        if (shape1.size() != shape2.size()) {
            return false;
        }
        for (size_t i = 0; i < shape1.size(); ++i) {
            if (shape1[i] != shape2[i]) {
                return false;
            }
        }
        return true;
    }
};

#ifndef DEFINE_MODULE_TEST_CONSTRUCTOR
#    define DEFINE_MODULE_TEST_CONSTRUCTOR(CLASS_NAME) \
        CLASS_NAME() = delete;                         \
        CLASS_NAME(const std::string& test_name) {     \
            m_test_name = test_name;                   \
        }
#endif

class TestRegistry {
public:
    using Creator = std::function<std::shared_ptr<ModuleTestBase>()>;

    static TestRegistry& get() {
        static TestRegistry instance;
        return instance;
    }

    void register_test(const std::string& name, Creator creator) {
        tests[name] = creator;
    }

    const std::map<std::string, Creator>& get_tests() const {
        return tests;
    }

private:
    std::map<std::string, Creator> tests;
};

struct TestRegistrar {
    TestRegistrar(const std::string& name, TestRegistry::Creator creator) {
        TestRegistry::get().register_test(name, creator);
    }
};

#define REGISTER_MODULE_TEST(CLASS_NAME)                                                               \
    static TestRegistrar registrar_##CLASS_NAME(#CLASS_NAME, []() -> std::shared_ptr<ModuleTestBase> { \
        return std::make_shared<CLASS_NAME>(#CLASS_NAME);                                                         \
    });
