# Unit Test

Each module's unit test.

# How to run

Refer script: run_test.sh
Please prepare your test models and copy to [PATH](./test_models/)

```
cd [PATH]/openvino.genai/tests/module_genai/cpp

export DATA_DIR=./test_data
export MODEL_DIR=./test_models
export DUMP_YAML=1  # Dump config yaml to file.
export DEVICE=GPU   # Default CPU.
export ENABLE_PROFILE=1 # Dump profiling data. default 0.

<!-- Copy libopenvino_tokenizers -->
OV_TOKENIZERS_LIB_PATH=../../../build/openvino_genai/libopenvino_tokenizers.so
cp ${OV_TOKENIZERS_LIB_PATH} ../../../build/tests/module_genai/cpp/

../../../build/tests/module_genai/cpp/genai_modules_test

<!-- Filter test example -->
../../../build/tests/module_genai/cpp/genai_modules_test --gtest_filter="*cat_120_100_dog_120_120*"

```

# Supported model list

Corresponding yaml config [path](../../../samples/cpp/module_genai/config_yaml/)

```
Qwen2.5-VL-3B-Instruct
Z-Image-Turbo-fp16-ov
Wan2.1-T2V-1.3B-Diffusers
Qwen3.5-0.8B
Qwen3.5-35B-A3B-Base_VL_OV_IR
Qwen3-Omni-4B-Instruct-multilingual
```
