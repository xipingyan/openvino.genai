---
name: 模块：md_llm_inference_pa
description: md_llm_inference_pa具体的内容
tags: [openvino, module, gtest]
---

# Skill: md_llm_inference_pa模块都具体内容

## 功能
1. 调用src/cpp/include/openvino/genai/continuous_batching_pipeline.hpp中的API，做模型推理
2. 尽量不要修改src/cpp/include/openvino/genai和src/cpp/src/continuous_batching下面的代码。必要修改时，需要提示，和用户交互进行修改。
3. inputs:
   - input_ids
   - input_embeds
4. outputs:
   - generated_text
5. params:
   - model_path
   - cache_dir
   - max_new_tokens
   - device

## Model qwen3_omni一些特有的功能

1. inputs:
   - deepstacks
   - merged_embeds
