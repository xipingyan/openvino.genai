# OpenVINO GenAI C++ Modeling Usage Guide

This directory contains the C++ modeling API for running supported models such as Qwen3-Omni inference
with OpenVINO, including text generation, vision understanding, audio understanding,
and text-to-speech (TTS) synthesis.

```
modeling/
├── models/    # supported models implementations
├── layers/               # Reusable ov::Model building blocks (attention, RMSNorm, etc.)
├── ops/                  # Custom OpenVINO operations
├── weights/              # Weight loading and quantization utilities
├── samples/              # Sample executables
│   └── tools/              # Dev-only Python utilities (see tools/README.md)
```

<details>
<summary>Prerequisites</summary>

- **Model weights**: HuggingFace Model checkpoint directory
  containing `model-*.safetensors`, `config.json`, `tokenizer.json`, and
  `preprocessor_config.json`.
- **OpenVINO**: Source-built OpenVINO (2026.1.0+).
- **Python**: Python 3.12 with `transformers`, `torch`, `numpy`, `Pillow` installed
  (required for vision/audio preprocessing bridge).
- **openvino_tokenizers**: The Python package `openvino_tokenizers` must be importable
  (used to convert HuggingFace `tokenizer.json` to OpenVINO XML at runtime).

### Environment Setup (Windows)

```bat
set OV_DIR=<path\to\openvino>
set GENAI_DIR=<path\to\openvino.genai>

REM OpenVINO runtime DLLs and openvino_genai DLL
set PATH=%OV_DIR%\bin\intel64\RelWithDebInfo;%GENAI_DIR%\build-master\openvino_genai;%PATH%

REM Source-built OpenVINO Python bindings + openvino_tokenizers Python package
set PYTHONPATH=%GENAI_DIR%\thirdparty\openvino_tokenizers\python;%OV_DIR%\bin\intel64\RelWithDebInfo\python;%PYTHONPATH%
set OPENVINO_LIB_PATHS=%OV_DIR%\bin\intel64\RelWithDebInfo
```

### Environment Setup (Linux)

```bash
export OV_DIR=<path/to/openvino>
export GENAI_DIR=<path/to/openvino.genai>

# OpenVINO runtime libraries and openvino_genai library
export LD_LIBRARY_PATH=$OV_DIR/bin/intel64/RelWithDebInfo:$GENAI_DIR/build-master/openvino_genai:$LD_LIBRARY_PATH

# Source-built OpenVINO Python bindings + openvino_tokenizers Python package
export PYTHONPATH=$GENAI_DIR/thirdparty/openvino_tokenizers/python:$OV_DIR/bin/intel64/RelWithDebInfo/python:$PYTHONPATH
export OPENVINO_LIB_PATHS=$OV_DIR/bin/intel64/RelWithDebInfo
```

</details>

<details>
<summary>Sample Executables</summary>

### Case 1: Image + Text → Text (`modeling_qwen3_omni`)

Loads the Qwen3-Omni text and vision models from safetensors, preprocesses an image,
runs vision encoding and autoregressive text decoding.

**Windows:**
```bat
modeling_qwen3_omni.exe ^
    --model-dir  path\to\model ^
    --image      path\to\image.jpg ^
    --prompt     "Describe this image in detail." ^
    --device     CPU ^
    --precision  fp32 ^
    --output-tokens 64
```

**Linux:**
```bash
./modeling_qwen3_omni \
    --model-dir  path/to/model \
    --image      path/to/image.jpg \
    --prompt     "Describe this image in detail." \
    --device     CPU \
    --precision  fp32 \
    --output-tokens 64
```

**Required arguments:**

| Argument | Description |
|---|---|
| `--model-dir PATH` | HuggingFace model directory with safetensors and config files |
| `--image PATH` | Input image file (JPEG, PNG, etc.) |

</details>

<details>
<summary>Cases 2–5: Multimodal → Text + TTS</summary>

### Cases 2–5: Multimodal → Text + TTS (`modeling_qwen3_omni_tts_min`)

Supports image, audio, video inputs with text-to-speech output. Uses positional arguments.

**Windows:**
```bat
modeling_qwen3_omni_tts_min.exe ^
    path\to\model ^
    <CASE_ID> ^
    "<TEXT_PROMPT>" ^
    output.wav ^
    [IMAGE_PATH] ^
    [AUDIO_PATH] ^
    [DEVICE] ^
    [MAX_NEW_TOKENS] ^
    [PRECISION] ^
    [VIDEO_FRAMES_DIR]
```

**Linux:**
```bash
./modeling_qwen3_omni_tts_min \
    path/to/model \
    <CASE_ID> \
    "<TEXT_PROMPT>" \
    output.wav \
    [IMAGE_PATH] \
    [AUDIO_PATH] \
    [DEVICE] \
    [MAX_NEW_TOKENS] \
    [PRECISION] \
    [VIDEO_FRAMES_DIR]
```

</details>

<details>
<summary>Test Cases & Examples</summary>

## Test Cases

| Case | Input | Output | Description |
|---|---|---|---|
| **1** | image + text | text | Visual question answering — describe or analyze an image |
| **2** | image + text | text + TTS | Image description with synthesized speech audio output |
| **3** | audio + text | text + TTS | Audio understanding (e.g., identify sounds) with speech reply |
| **4** | image + audio + text | text + TTS | Combined image and audio understanding with speech reply |
| **5** | image + video + audio + text | text + TTS | Full multimodal: image, video frames, audio, and text prompt |

### Example: Case 2 — Image Description with TTS

**Windows:**
```bat
modeling_qwen3_omni_tts_min.exe ^
    path\to\model ^
    2 ^
    "Describe this image and provide a speech response." ^
    case2_output.wav ^
    path\to\image.jpg ^
    none ^
    CPU ^
    32 ^
    fp32
```

**Linux:**
```bash
./modeling_qwen3_omni_tts_min \
    path/to/model \
    2 \
    "Describe this image and provide a speech response." \
    case2_output.wav \
    path/to/image.jpg \
    none \
    CPU \
    32 \
    fp32
```

### Example: Case 3 — Audio Understanding with TTS

**Windows:**
```bat
modeling_qwen3_omni_tts_min.exe ^
    path\to\model ^
    3 ^
    "What sound do you hear in the audio? Answer in one short sentence." ^
    case3_output.wav ^
    none ^
    path\to\audio.wav ^
    CPU ^
    32 ^
    fp32
```

**Linux:**
```bash
./modeling_qwen3_omni_tts_min \
    path/to/model \
    3 \
    "What sound do you hear in the audio? Answer in one short sentence." \
    case3_output.wav \
    none \
    path/to/audio.wav \
    CPU \
    32 \
    fp32
```

### Example: Case 5 — Full Multimodal (Image + Video + Audio + Text)

Requires pre-extracted video frames (use `extract_video_frames`):

**Windows:**
```bat
REM Step 1: Extract video frames
extract_video_frames.exe ^
    --video path\to\video.mp4 ^
    --output-dir frames_dir ^
    --max-frames 4

REM Step 2: Run Case 5
modeling_qwen3_omni_tts_min.exe ^
    path\to\model ^
    5 ^
    "Describe the scene in the image, video, and audio." ^
    case5_output.wav ^
    path\to\image.jpg ^
    path\to\audio.wav ^
    CPU ^
    32 ^
    fp32 ^
    frames_dir
```

**Linux:**
```bash
# Step 1: Extract video frames
./extract_video_frames \
    --video path/to/video.mp4 \
    --output-dir frames_dir \
    --max-frames 4

# Step 2: Run Case 5
./modeling_qwen3_omni_tts_min \
    path/to/model \
    5 \
    "Describe the scene in the image, video, and audio." \
    case5_output.wav \
    path/to/image.jpg \
    path/to/audio.wav \
    CPU \
    32 \
    fp32 \
    frames_dir
```

</details>

<details>
<summary>Precision Modes</summary>

## Precision Modes

Control inference precision and KV-cache compression via the `--precision` argument:

| Mode | Inference | KV Cache | Weights | Description |
|---|---|---|---|---|
| `mixed` | bf16 | bf16 | original | Default mixed precision |
| `fp32` | f32 | f32 | original | Full FP32 precision |
| `inf_fp16_kv_int8` | f16 | u8 | original | FP16 inference with INT8 KV cache |
| `inf_fp32_kv_fp32_w_int4_asym` | f32 | f32 | INT4 asym | FP32 with INT4 weight quantization |
| `inf_fp32_kv_fp32_w_int8` | f32 | f32 | INT8 asym | FP32 with INT8 weight quantization |
| `inf_fp32_kv_int8_w_int4_asym` | f32 | u8 | INT4 asym | FP32, INT8 KV cache, INT4 weights |
| `inf_fp16_kv_int8_w_int4_asym` | f16 | u8 | INT4 asym | FP16, INT8 KV cache, INT4 weights |

Aliases: `fp32_kv8` → `inf_fp32_kv_int8`, `fp16_kv8` → `inf_fp16_kv_int8`, etc.

</details>

<details>
<summary>Automated Case Comparison</summary>

## Automated Case Comparison (`tools/qwen3_omni_case_compare.py`)

Runs all cases across multiple devices and precision modes, generating a JSON report
with performance metrics and text outputs for comparison.

**Windows:**
```bat
python tools/qwen3_omni_case_compare.py ^
    --model-dir     path\to\model ^
    --image         path\to\image.jpg ^
    --test-audio    path\to\audio.wav ^
    --video         path\to\video.mp4 ^
    --case5-audio   path\to\case5_audio.wav ^
    --case5-image   path\to\case5_image.jpg ^
    --case5-prompt-file  path\to\prompt.txt ^
    --cpp-bin       path\to\modeling_qwen3_omni.exe ^
    --cpp-tts-bin   path\to\modeling_qwen3_omni_tts_min.exe ^
    --out-json      reports\case_compare.json ^
    --max-new-tokens 32 ^
    --max-video-frames 4 ^
    --devices       CPU,GPU.1 ^
    --precisions    fp32,inf_fp16_kv_int8,inf_fp32_kv_fp32_w_int4_asym ^
    --timeout       600 ^
    --cpp-only
```

**Linux:**
```bash
python tools/qwen3_omni_case_compare.py \
    --model-dir     path/to/model \
    --image         path/to/image.jpg \
    --test-audio    path/to/audio.wav \
    --video         path/to/video.mp4 \
    --case5-audio   path/to/case5_audio.wav \
    --case5-image   path/to/case5_image.jpg \
    --case5-prompt-file  path/to/prompt.txt \
    --cpp-bin       path/to/modeling_qwen3_omni \
    --cpp-tts-bin   path/to/modeling_qwen3_omni_tts_min \
    --out-json      reports/case_compare.json \
    --max-new-tokens 32 \
    --max-video-frames 4 \
    --devices       CPU,GPU.1 \
    --precisions    fp32,inf_fp16_kv_int8,inf_fp32_kv_fp32_w_int4_asym \
    --timeout       600 \
    --cpp-only
```

</details>

<details>
<summary>C++ Modeling API Overview</summary>

## C++ Modeling API Overview

The modeling API builds `ov::Model` graphs from safetensor weights at runtime:

```cpp
#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vision.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

// 1. Load config and weights
auto cfg = Qwen3OmniConfig::from_json_file(model_dir);
auto data = ov::genai::safetensors::load_safetensors(model_dir);
ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));
ov::genai::safetensors::SafetensorsWeightFinalizer finalizer;

// 2. Build ov::Model for text and vision
auto text_model = create_qwen3_omni_text_model(cfg, source, finalizer);
auto vision_model = create_qwen3_omni_vision_model(cfg, source, finalizer);

// 3. Compile and run inference
ov::Core core;
auto compiled_text = core.compile_model(text_model, "CPU");
auto compiled_vision = core.compile_model(vision_model, "CPU");

// 4. Preprocess image
Qwen3OmniVisionPreprocessor preprocessor(cfg, pre_cfg);
auto vision_inputs = preprocessor.preprocess(image_tensor, pos_embed_weight);

// 5. Run vision encoder
auto vision_request = compiled_vision.create_infer_request();
vision_request.set_tensor("pixel_values", vision_inputs.pixel_values);
vision_request.set_tensor("grid_thw", vision_inputs.grid_thw);
// ... set other inputs ...
vision_request.infer();
auto visual_embeds = vision_request.get_tensor("visual_embeds");

// 6. Run autoregressive text generation with tokenizer
ov::genai::Tokenizer tokenizer(model_dir);
auto text_request = compiled_text.create_infer_request();
// ... feed input_ids, attention_mask, visual_embeds, position_ids ...
// ... decode loop with argmax sampling ...
```

</details>
