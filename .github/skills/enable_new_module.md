---
name: 实现新module
description: 实现新module的步骤与注意事项，所有module的代码位置为：src/cpp/src/module_genai/modules/
tags: [openvino, module, gtest]
---

# Skill: 实现新module

## Steps
1. 实现一个空的module，例如名字为：NewModule,
   - 新建文件加：src/cpp/src/module_genai/modules/new_module, src/cpp/src/module_genai/modules/new_module/models
   - 新建文件：src/cpp/src/module_genai/modules/new_module/new_module.hpp and src/cpp/src/module_genai/modules/new_module/new_module.cpp
   - 参考：src/cpp/src/module_genai/modules/m_llm_inference_pa/md_llm_inference_pa.hpp，实现new_module.hpp，例如必须继承自public IBaseModule，module特有的参数放到private下面，如果有可能被针对具体model实现时，model内部要用的变量，放到protected下面。
   - 参考：src/cpp/src/module_genai/modules/m_llm_inference_pa/md_llm_inference_pa.cpp，实现new_module.cpp，这里所有的操作都是一个common的，可以被大多数model共用。
   - 如果是一些模型特殊的操作，放到models目录下，例如：src/cpp/src/module_genai/modules/md_llm_inference_sdpa/models/qwen3_omni.hpp， 它需要继承自：NewModule

2. 实现对应module的测试，测试代码位置为：tests/module_genai/cpp/modules
   - 测试基本机构可以参考：tests/module_genai/cpp/modules/AudioPreprocesModule.cpp
   - 名字为：Module名字+Test
   - 自己定义：test_params， 并需要添加注释，解释参数的意义，最好添加上期望的结果中一小部分数据，用于验证。
   - 需要继承自：public ModuleTestBase, public ::testing::TestWithParam<test_params>
   - 需要_threshold，一般默认为：1e-2
   - 在get_test_case_name中，尽量能反应使用的每个参数，expected value除外。
   - 在get_yaml_content中，尽量使用YAML::Node去构建config，尽量复用一些utils函数。
   - 整个测试，需要一个全局的namespace，例如：namespace AudioPreprocessModuleTest

3. 提供需要在module中实现的具体功能，可以输入到copilot input或者提供一个新的skill.md

4. 根据具体功能，先完成测试，再完善对应module内容。
   - get_spec： 实现module所需的inputs,outputs, params 注册。
   - print_static_config：返回string格式的注册后的参数列表。
   - 构造函数：需要调用基类构造函数，还必须调用： check_params_with_spec(get_spec())， 检查真是输入的参数，是否被注册。
   - 构造函数中，如果代码太长，可以封装成多个initialize函数，例如：init_models, init_config等。
   - run，runtime的执行函数，
      - 需要调用：	GENAI_INFO("Running module: " + module_desc->name);
      - prepare_inputs(), 从之前的节点拿到当前module需要用到的inputs。
      - 解析inputs
      - 具体执行
      - 返回output到this->outputs

## Goal
1. 没有编译问题，单元测试通过。

## Fast build
1. Build:
   - `cd ~/mygithub/modular_genai/openvino.genai`
   - `../build_genai.sh`
2. Run:
   - `./build/tests/module_genai/cpp/genai_modules_test --gtest_filter="*具体的测试模块都名字*"`

## Exit Criteria
当满足以下条件时视为完成：
1. build 通过；
2. `genai_modules_test` 无崩溃；测试通过。
