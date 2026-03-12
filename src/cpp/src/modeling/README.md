# OpenVINO GenAI Modeling — Qwen3-Omni Usage Guide

This directory contains the C++ modeling API for running **Qwen3-Omni-4B** inference
with OpenVINO, including text generation, vision understanding, audio understanding,
and text-to-speech (TTS) synthesis.

## Directory Structure

```
modeling/
├── models/qwen3_omni/    # Qwen3-Omni model implementations
│   ├── modeling_qwen3_omni.hpp          # Text model (thinker) builder
│   ├── modeling_qwen3_omni_audio.hpp    # Audio (talker/TTS) model builder
│   ├── processing_qwen3_omni_audio.hpp  # Audio preprocessing (WAV → mel spectrogram)
│   ├── processing_qwen3_omni_vl.hpp     # Vision-Language processing
│   ├── processing_qwen3_omni_vision.hpp # Vision preprocessing (image → pixel values)
│   └── whisper_mel_spectrogram.hpp      # Whisper-style mel spectrogram extractor
├── layers/               # Reusable ov::Model building blocks (attention, RMSNorm, etc.)
├── ops/                  # Custom OpenVINO operations
├── weights/              # Weight loading and quantization utilities
├── samples/              # Sample executables
│   ├── modeling_qwen3_omni.cpp          # Case 1: image+text → text
│   ├── modeling_qwen3_omni_tts_min.cpp  # Cases 2–5: multimodal → text + TTS
│   └── extract_video_frames.cpp         # Video frame extraction tool
└── qwen3_omni_case_compare.py           # Automated test harness (all cases × devices × precisions)
```

## Prerequisites

- **Model weights**: HuggingFace Qwen3-Omni-4B-Instruct checkpoint directory
  containing `model-*.safetensors`, `config.json`, `tokenizer.json`, and
  `preprocessor_config.json`.
- **OpenVINO**: Source-built OpenVINO (2026.1.0+).
- **Python**: Python 3.12 with `transformers`, `torch`, `numpy`, `Pillow` installed
  (required for vision/audio preprocessing bridge).
- **openvino_tokenizers**: The Python package `openvino_tokenizers` must be importable
  (used to convert HuggingFace `tokenizer.json` to OpenVINO XML at runtime).

### Environment Setup (Windows)

```bat
set OV_DIR=C:\work\ws_tmp\openvino.xzhan34
set GENAI_DIR=C:\work\ws_tmp\openvino.genai.xzhan34

REM OpenVINO runtime DLLs and openvino_genai DLL
set PATH=%OV_DIR%\bin\intel64\RelWithDebInfo;%GENAI_DIR%\build-master\openvino_genai;%PATH%

REM Source-built OpenVINO Python bindings + openvino_tokenizers Python package
set PYTHONPATH=%GENAI_DIR%\thirdparty\openvino_tokenizers\python;%OV_DIR%\bin\intel64\RelWithDebInfo\python;%PYTHONPATH%
set OPENVINO_LIB_PATHS=%OV_DIR%\bin\intel64\RelWithDebInfo
```

## Sample Executables

### Case 1: Image + Text → Text (`modeling_qwen3_omni`)

Loads the Qwen3-Omni text and vision models from safetensors, preprocesses an image,
runs vision encoding and autoregressive text decoding.

```
modeling_qwen3_omni.exe ^
    --model-dir  D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
    --image      path\to\image.jpg ^
    --prompt     "Describe this image in detail." ^
    --device     CPU ^
    --precision  fp32 ^
    --output-tokens 64
```

**Required arguments:**

| Argument | Description |
|---|---|
| `--model-dir PATH` | HuggingFace model directory with safetensors and config files |
| `--image PATH` | Input image file (JPEG, PNG, etc.) |

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--prompt TEXT` | `"What can you see"` | User text prompt |
| `--device NAME` | `CPU` | OpenVINO device (`CPU`, `GPU`, `GPU.1`, etc.) |
| `--precision MODE` | `mixed` | Inference precision mode (see below) |
| `--output-tokens N` | `64` | Maximum number of tokens to generate |
| `--dump-dir PATH` | *(none)* | Directory to dump intermediate tensors for debugging |
| `--dump-ir-dir PATH` | *(none)* | Directory to save compiled IR models |

### Cases 2–5: Multimodal → Text + TTS (`modeling_qwen3_omni_tts_min`)

Supports image, audio, video inputs with text-to-speech output. Uses positional arguments.

```
modeling_qwen3_omni_tts_min.exe ^
    D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
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

**Positional arguments (in order):**

| # | Argument | Required | Description |
|---|---|---|---|
| 1 | `MODEL_DIR` | Yes | HuggingFace model directory |
| 2 | `CASE_ID` | Yes | Test case identifier (2, 3, 4, or 5) |
| 3 | `TEXT_PROMPT` | Yes | User text prompt |
| 4 | `WAV_OUT` | Yes | Output WAV file path for synthesized speech |
| 5 | `IMAGE_PATH` | No | Input image (use `none` to skip) |
| 6 | `AUDIO_PATH` | No | Input audio WAV file (use `none` to skip) |
| 7 | `DEVICE` | No | OpenVINO device (default: `CPU`) |
| 8 | `MAX_NEW_TOKENS` | No | Max generation tokens (default: `64`) |
| 9 | `PRECISION` | No | Precision mode (default: `fp32`) |
| 10 | `VIDEO_FRAMES_DIR` | No | Directory of extracted video frames (use `none` to skip) |

## Test Cases

| Case | Input | Output | Description |
|---|---|---|---|
| **1** | image + text | text | Visual question answering — describe or analyze an image |
| **2** | image + text | text + TTS | Image description with synthesized speech audio output |
| **3** | audio + text | text + TTS | Audio understanding (e.g., identify sounds) with speech reply |
| **4** | image + audio + text | text + TTS | Combined image and audio understanding with speech reply |
| **5** | image + video + audio + text | text + TTS | Full multimodal: image, video frames, audio, and text prompt |

### Example: Case 2 — Image Description with TTS

```bat
modeling_qwen3_omni_tts_min.exe ^
    D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
    2 ^
    "Describe this image and provide a speech response." ^
    case2_output.wav ^
    path\to\image.jpg ^
    none ^
    CPU ^
    32 ^
    fp32
```

### Example: Case 3 — Audio Understanding with TTS

```bat
modeling_qwen3_omni_tts_min.exe ^
    D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
    3 ^
    "What sound do you hear in the audio? Answer in one short sentence." ^
    case3_output.wav ^
    none ^
    path\to\audio.wav ^
    CPU ^
    32 ^
    fp32
```

### Example: Case 5 — Full Multimodal (Image + Video + Audio + Text)

Requires pre-extracted video frames (use `extract_video_frames.exe`):

```bat
REM Step 1: Extract video frames
extract_video_frames.exe ^
    --video path\to\video.mp4 ^
    --output-dir frames_dir ^
    --max-frames 4

REM Step 2: Run Case 5
modeling_qwen3_omni_tts_min.exe ^
    D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
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

## Automated Case Comparison (`qwen3_omni_case_compare.py`)

Runs all cases across multiple devices and precision modes, generating a JSON report
with performance metrics and text outputs for comparison.

```bat
python qwen3_omni_case_compare.py ^
    --model-dir     D:\models\Qwen3-Omni-4B-Instruct-multilingual ^
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

**Key arguments:**

| Argument | Description |
|---|---|
| `--model-dir` | HuggingFace model directory |
| `--image` | Default image for Cases 1–4 |
| `--test-audio` | Default audio for Cases 3–4 |
| `--video` | Video file for Case 5 (frames extracted automatically) |
| `--case5-image` | Case 5 specific image (falls back to `--image`) |
| `--case5-audio` | Case 5 specific audio (falls back to `--test-audio`) |
| `--case5-prompt-file` | Text file containing the Case 5 prompt |
| `--cpp-bin` | Path to `modeling_qwen3_omni` executable (Case 1) |
| `--cpp-tts-bin` | Path to `modeling_qwen3_omni_tts_min` executable (Cases 2–5) |
| `--out-json` | Output JSON report path |
| `--devices` | Comma-separated devices: `CPU`, `GPU`, `GPU.1` |
| `--precisions` | Comma-separated precision modes |
| `--cases` | Run specific cases only (e.g., `--cases 1,5`) |
| `--max-new-tokens` | Token generation limit per case |
| `--max-video-frames` | Max video frames to extract for Case 5 |
| `--timeout` | Per-case timeout in seconds (default: 600) |
| `--cpp-only` | Skip Python reference inference, run C++ cases only |

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
