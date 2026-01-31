# Module GenAI Samples

Build samples, refer: openvino.genai/README_Module_GenAI.md

<details>
<summary>visual_language_chat</summary>


#### Test model: Qwen2.5-VL-3B-Instruct
```
cd openvino.genai
app=./build/samples/cpp/module_genai/md_visual_language_chat
cfg=./samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml
prompt="Please describe the image"
img=./tests/module_genai/cpp/test_data/cat_120_100.png
$app $cfg "$prompt" $img
```

</details>