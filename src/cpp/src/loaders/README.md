# Multi-Format Model Loaders

This module provides a unified interface for loading models from different file formats.

## Supported Formats

| Format | Status | Loader | Description |
|--------|--------|--------|-------------|
| **GGUF** | ✅ Tested | `GGUFLoader` | llama.cpp GGUF format |
| **Safetensors** | ✅ Tested | `SafetensorsLoader` | HuggingFace Safetensors format |
| **OpenVINO IR** | 🚧 TODO | - | Native OpenVINO IR format (.xml/.bin) |

## Supported Model Architectures

| Architecture | Status | Models |
|--------------|--------|--------|
| **Qwen3** | ✅ Tested | Qwen3-0.6B, Qwen3-4B |
| **SmolLM3** | ✅ Tested | SmolLM3-3B |
| **Qwen2** | 🚧 Pending | Requires validation |
| **LLaMA** | 🚧 TODO | - |
| **Mistral** | 🚧 TODO | - |

## Usage

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OV_GENAI_USE_UNIFIED_LOADER` | `0` | Set to `1` to enable new unified loader |
| `OV_GENAI_USE_MODELING_API` | `1` | Set to `1` to use new modeling API (legacy path) |

### Example

```cpp
// Set environment variable before running
// export OV_GENAI_USE_UNIFIED_LOADER=1

// Load model (auto-detects format)
auto model = read_model("path/to/model");
```

## Testing

Tested configurations:

| Model | Format | Configurations | Status |
|-------|--------|----------------|--------|
| Qwen3-0.6B-BF16.gguf | GGUF | 3 env configs | ✅ Pass |
| qwen3-4b-hf | Safetensors | 3 env configs | ✅ Pass |
| SmolLM3-3B | Safetensors | Unified loader | ✅ Pass |

Test command:
```bash
greedy_causal_lm.exe <model_path> "question: what is ffmpeg?" GPU 1 1 100
```

## TODO

- load_tokenizer?
