# UT For module_genai pipeline

This is UT for modular GenAI. Please verify the correctness of [pipeline](./ut_pipelines/config.yaml) before merging code.

#### Build GenAI

Refer [Guide](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/docs/BUILD.md)

```
ut_build.sh
```

#### Module Unit Test

```
./ut_modules.sh
```

#### Pipeline Unit test

```bash
./ut_pipelines.sh
```