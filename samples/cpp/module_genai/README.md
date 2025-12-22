# UT For module_genai pipeline

This is UT for modular GenAI. Please verify the correctness of [pipeline](./ut_pipelines/config.yaml) before merging code.

#### Build GenAI

Refer [Guide](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/docs/BUILD.md)

```
ut_build.sh
```

#### Module Unit Test

Preprare your model, take qwen2.5-vl as example: `openvino.genai/samples/cpp/module_genai/ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/`

```
./ut_modules.sh
```

#### Pipeline Unit test

```bash
./ut_pipelines.sh
```