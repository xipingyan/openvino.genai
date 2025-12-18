# UT For module_genai pipeline

This is UT for modular GenAI. Please verify the correctness of [pipeline](./config_pipeline/config_qwen2_5_vl.yaml) before merging code.

Step1: Build GenAI

Step2: Add new modules in `./config_pipeline/config_qwen2_5_vl.yaml`

Step3: Build sample test and run

```bash
./ut_build.sh
./ut_run.sh
```

Note: Remember to modify your own model_path in yaml file, and modify openvino dir in `ut_build.sh` and `ut_run.sh`.
