
# Module GenAI Python samples

This folder contains Python samples for **Module GenAI** based pipelines.

The Python bindings are provided by `openvino_genai` which is installed under `openvino.genai/install/python` after building.

## Setup

From the repo root:

```bash
cd openvino.genai

# Activate the repo venv (created in the repo root)
source ../python-env/bin/activate

# Set OpenVINO runtime environment (repo helper script)
source ../source_ov.sh

# Make openvino_genai importable
export PYTHONPATH="$PWD/install/python:${PYTHONPATH}"

# Ensure runtime libraries are found
export LD_LIBRARY_PATH="$PWD/install/runtime/lib/intel64:${LD_LIBRARY_PATH}"
```

## Visual language chat

This sample runs a VLM pipeline (e.g. Qwen2.5-VL-3B-Instruct) using a ModulePipeline config YAML.

<details>
	<summary>Command</summary>

```bash
python3 ./samples/python/module_genai/md_visual_language_chat.py \
	./samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml \
	"Describe this image" \
	./samples/cpp/module_genai/test_data/demo.png
```

</details>

Notes:

- Update model paths inside the config YAML if you keep models in a different location.
- `image_path` / `video_path` are optional; the sample maps inputs based on `ParameterModule.outputs`.

## Image generation

Use the Z-Image pipeline Python sample.

<details>
	<summary>Command</summary>

```bash
python3 ./samples/python/module_genai/md_image_generation.py \
	--model_path ./samples/cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov \
	--device GPU \
	--prompt "A cozy cabin in a snowy forest" \
	--height 1040 \
	--width 1040
```

</details>

## More samples

- `md_video_generation.py` (video generation)
- `md_cowork_with_torch.py` (mix ModulePipeline with Torch steps)

