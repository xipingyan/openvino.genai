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

    virtual std::string generate_yaml() {
        std::string filename = "temp_" + m_test_name + ".yaml";
        std::ofstream out(filename);
        out << get_yaml_content();
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

#ifndef CHECK_RESULT
#    define CHECK_RESULT(cond, msg)                                            \
        do {                                                                   \
            if (!(cond)) {                                                     \
                throw std::runtime_error(std::string("Check failed: ") + msg); \
            }                                                                  \
        } while (0)
#endif
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
