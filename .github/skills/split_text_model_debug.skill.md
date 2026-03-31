---
name: 实现拆分text_model的功能
description: split model的代码在 src/cpp/src/modeling/samples/utils/split_text_model.cpp
tags: [openvino, model-graph, parameter, split-model]
---

# Skill: 实现`split_text_model`并在run_qwen3_omni_decode_split_models中调用

## Goal
1. 实现split_text_model，编译，初步查看结果正常。
2. 在run_qwen3_omni_decode_split_models中调用拆分后的模型。
3. run_qwen3_omni_decode_split_models和run_qwen3_omni_decode跑出来的结果应该一样。可以通过macro（ENABLE_SPLIT_MODEL=0、1）区分;

## Scope
- Target file: `src/cpp/src/modeling/samples/utils/split_text_model.cpp`
- Related app: `src/cpp/src/modeling/samples/test_split_text_model.cpp`
- Call file: `src/cpp/src/module_genai/modules/md_llm_inference_sdpa/models/qwen3_omni.cpp`

## Fast Repro
1. Build:
   - `cd ~/mygithub/modular_genai/openvino.genai`
   - `../build_genai.sh`
2. Run:
   - `./build/src/cpp/src/modeling/samples/test_split_text_model`
   - `# 注意：当前目录在 openvino.genai，脚本在上一级 modular_genai`
   - `ENABLE_SPLIT_MODEL=0 RUN_OMNI=1 ../run_samples_cpp.sh`
   - `ENABLE_SPLIT_MODEL=1 RUN_OMNI=1 ../run_samples_cpp.sh`

## Common Pitfall
- 在 `openvino.genai` 目录执行 `./run_samples_cpp.sh` 会报 `No such file or directory`（exit code 127）。
- 正确命令：`../run_samples_cpp.sh`。

## Verification Checklist
- [ ] `cd ~/mygithub/modular_genai/openvino.genai && ../build_genai.sh` 通过
- [ ] `test_split_text_model` 可运行无 `AssertFailure`
- [ ] 控制台参数打印与预期一致（无“未声明但被引用”的参数）
- [ ] `embedding_merge_model.xml/bin` 与 `llm_model.xml/bin` 可成功序列化

## Exit Criteria
当满足以下条件时视为修复完成：
1. build 通过；
2. `test_split_text_model` 无崩溃；
3. run_qwen3_omni_decode_split_models和run_qwen3_omni_decode跑出来的结果应该一样。
