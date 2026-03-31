---
name: 模块：md_llm_inference_pa
description: md_llm_inference_pa具体的内容
tags: [openvino, module, gtest]
---

# Skill: md_llm_inference_pa模块都具体内容

## 功能
1. 调用src/cpp/include/openvino/genai/continuous_batching_pipeline.hpp相关PA能力做模型推理。
2. 尽量不要修改src/cpp/include/openvino/genai和src/cpp/src/continuous_batching下面的代码。
3. inputs：
   - input_ids
   - input_embeds
4. outputs：
   - generated_text
5. params：
   - model_path（必选）
   - cache_dir（可选）
   - max_new_tokens（可选，默认256）
   - device（可选，优先级高于module.device）

## 实现步骤（参考 md_llm_inference_sdpa.hpp / md_text_to_speech.hpp）

1. 新建模块文件
   - src/cpp/src/module_genai/modules/m_llm_inference_pa/md_llm_inference_pa.hpp
   - src/cpp/src/module_genai/modules/m_llm_inference_pa/md_llm_inference_pa.cpp

2. 头文件结构
   - 继承 `public IBaseModule`
   - 提供：`create/get_spec/print_static_config/run`
   - 定义 `InputsParams`，至少包含：
     - `input_ids`
     - `input_embeds`
     - `merged_embeds`
     - `deepstacks`

3. cpp实现
   - `get_spec`：注册inputs/outputs/params
   - 构造函数：调用 `check_params_with_spec(get_spec())`
   - `initialize`：
     - 解析 `model_path/device/cache_dir/max_new_tokens`
     - 初始化PA pipeline
   - `run`：
     - `GENAI_INFO("Running module: " + module_desc->name);`
     - `prepare_inputs()`
     - `parse_inputs()`
     - 调用PA生成并写入 `outputs["generated_text"]`

4. 参数与输入校验
   - 必须使用 `OPENVINO_ASSERT(...)`
   - 不允许静默修正配置
   - `input_ids` 与 `input_embeds/merged_embeds` 互斥

5. 新增测试
   - 测试目录：tests/module_genai/cpp/modules
   - 参考：tests/module_genai/cpp/modules/AudioPreprocesModule.cpp
   - 至少包含：
     - 基础推理冒烟测试（输出非空）
     - 输入/参数校验测试（必要时）

## Model qwen3_omni一些特有的功能

1. inputs:
   - deepstacks
   - merged_embeds

## 完成标准

1. 编译通过。
<!-- 2. `genai_modules_test --gtest_filter="*LLMInferencePAModule*"` 通过 -->
2. 本地的验证脚本为： 
```
cd ~/mygithub/modular_genai/openvino.genai/tests/module_genai/cpp
./run_test_xiping.sh 
```
