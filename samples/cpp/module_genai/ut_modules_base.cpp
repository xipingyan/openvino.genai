// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ut_modules_base.hpp"

YAML::Node ModuleTestBase::generate_yaml() {
    std::string cur_module_cfg = get_yaml_content();
    YAML::Node config = YAML::Load(cur_module_cfg);

    OPENVINO_ASSERT(config["pipeline_modules"], "Test yaml config miss 'pipeline_modules'.");

    YAML::Node modules = config["pipeline_modules"];
    std::map<std::string, std::string> extracted_params;
    std::map<std::string, std::string> extracted_results;

    if (modules.size() != 1) {
        return config;
    }

    // When only one module is specified, the ParameterModule and ResultModule will be automatically inferred and added.
    std::string auto_padding_param_name = "pipeline_params";
    std::string auto_padding_result_name = "pipeline_results";

    // only one node recursive
    std::string test_module_name;
    for (auto it = modules.begin(); it != modules.end(); ++it) {
        test_module_name = it->first.as<std::string>();
        // get inputs
        YAML::Node inputs = it->second["inputs"];
        if (inputs && inputs.IsSequence()) {
            for (const auto& input : inputs) {
                std::string input_name = input["name"].as<std::string>("");
                std::string source = input["source"].as<std::string>(std::string());
                if (source.empty()) {
                    std::string type = input["type"].as<std::string>("");
                    extracted_params[input_name] = type;
                    it->second["inputs"][0]["source"] = auto_padding_param_name + "." + input_name;
                    continue;
                } else if (source.find(auto_padding_param_name + ".") == 0) {
                    std::string param_name = source.substr(auto_padding_param_name.size() + 1);
                    std::string type = input["type"].as<std::string>("");
                    extracted_params[param_name] = type;
                } else {
                    OPENVINO_ASSERT(false, "Error: Input[" + input_name + "] source format error. ");
                }
            }
        }

        // get outputs
        YAML::Node outputs = it->second["outputs"];
        if (outputs && outputs.IsSequence()) {
            for (const auto& output : outputs) {
                std::string name = output["name"].as<std::string>("");
                std::string type = output["type"].as<std::string>("");
                extracted_results[name] = type;
            }
        }
    }

    // pipeline_params
    YAML::Node params_node;
    params_node["type"] = "ParameterModule";
    YAML::Node outputs_seq;
    for (const auto& param : extracted_params) {
        YAML::Node item;
        item["name"] = param.first;
        item["type"] = param.second;
        outputs_seq.push_back(item);
    }
    if (outputs_seq.size() > 0) {
        params_node["outputs"] = outputs_seq;
    }
    config["pipeline_modules"][auto_padding_param_name] = params_node;

    // pipeline_results
    YAML::Node results_node;
    results_node["type"] = "ResultModule";
    YAML::Node inputs_seq;
    for (const auto& result : extracted_results) {
        YAML::Node item;
        item["name"] = result.first;
        item["type"] = result.second;
        item["source"] = test_module_name + "." + result.first;
        inputs_seq.push_back(item);
    }
    if (inputs_seq.size() > 0) {
        results_node["inputs"] = inputs_seq;
    }
    config["pipeline_modules"][auto_padding_result_name] = results_node;

    return config;
}

ov::Tensor ModuleTestBase::ut_randn_tensor(const ov::Shape& shape, size_t seed) {
    ov::Tensor rand_tensor(ov::element::f32, shape);
    float* rand_tensor_data = rand_tensor.data<float>();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < rand_tensor.get_size(); ++i) {
        rand_tensor_data[i] = dist(rng);
    }

    return rand_tensor;
}

bool ModuleTestBase::compare_shape(const ov::Shape& shape1, const ov::Shape& shape2) {
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

bool ModuleTestBase::compare_tensors(const ov::Tensor& output, const ov::Tensor& expected) {
    if (output.get_shape() != expected.get_shape() || output.get_element_type() != expected.get_element_type()) {
        return false;
    }
    size_t byte_size = output.get_byte_size();
    return std::memcmp(output.data(), expected.data(), byte_size) == 0;
}

bool ModuleTestBase::compare_big_tensor(const ov::Tensor& output,
                                        const std::vector<float>& expected_top,
                                        const float& thr) {
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