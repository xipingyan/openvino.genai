# Dev Tools

Dev-only Python utilities for model preparation and testing. **Not included in release builds.**

## Setup

Each tool may have its own `requirements_<tool>.txt`. Install as needed:

```bash
pip install -r requirements_hf_tokenizer_to_ov.txt
```

## Tools

### hf_tokenizer_to_ov.py

Convert a HuggingFace tokenizer to OpenVINO IR (`openvino_tokenizer.xml` + `openvino_detokenizer.xml`).

Dependencies: `requirements_hf_tokenizer_to_ov.txt`

```bash
# Basic usage — output defaults to model_id directory
python hf_tokenizer_to_ov.py --help

# Specify output directory
python hf_tokenizer_to_ov.py Qwen/Qwen3-0.6B -o ./qwen3_tokenizer_ir
```


### qwen3_omni_case_compare.py

Automated test harness: runs Qwen3-Omni inference cases across multiple devices and precision modes, generating a JSON report with performance metrics and text outputs.

Dependencies: Python standard library only (no extra install needed).

```bat
python qwen3_omni_case_compare.py ^
    --model-dir D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
    --image path\to\image.jpg ^
    --test-audio path\to\audio.wav ^
    --cpp-bin path\to\modeling_qwen3_omni.exe ^
    --out-json reports\case_compare.json ^
    --devices CPU,GPU ^
    --precisions fp32,inf_fp16_kv_int8
```

See [modeling README](../../../modeling/README.md) for full options.
