#!/usr/bin/env python3
# pyright: reportMissingImports=false
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml
from openvino import Tensor
import openvino_genai

def _load_image_tensor(image_path: str) -> Tensor:
    pic = Image.open(image_path).convert("RGB")
    arr = np.array(pic, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    # NHWC with batch size 1
    return Tensor(arr[None, ...])


def _load_video_tensor(video_path: str) -> Tensor:
    entry = Path(video_path)
    if not entry.exists():
        raise FileNotFoundError(video_path)

    if entry.is_file():
        # Minimal behavior: treat a single file as a 1-frame video
        return _load_image_tensor(str(entry))

    frames: list[np.ndarray] = []
    for file in sorted(entry.iterdir()):
        if not file.is_file():
            continue
        try:
            pic = Image.open(file).convert("RGB")
        except Exception:
            continue
        frames.append(np.array(pic, dtype=np.uint8))

    if not frames:
        raise RuntimeError(f"No readable image frames in directory: {video_path}")

    # [N, H, W, 3]
    return Tensor(np.stack(frames, axis=0))


def _find_parameter_module_outputs(cfg_yaml_path: Path) -> list[dict[str, Any]]:
	cfg = yaml.safe_load(cfg_yaml_path.read_text(encoding="utf-8"))
	pipeline_modules = cfg.get("pipeline_modules")
	if not isinstance(pipeline_modules, dict):
		raise RuntimeError("Invalid config: missing 'pipeline_modules'")

	for _, module in pipeline_modules.items():
		if isinstance(module, dict) and module.get("type") == "ParameterModule":
			outputs = module.get("outputs")
			if isinstance(outputs, list):
				return [x for x in outputs if isinstance(x, dict)]
			raise RuntimeError("Invalid config: ParameterModule.outputs is not a list")

	raise RuntimeError("Could not find ParameterModule in config YAML.")


def _build_inputs_from_cfg(cfg_yaml_path: Path, prompt: str, image_path: str, video_path: str) -> dict[str, Any]:
	outputs = _find_parameter_module_outputs(cfg_yaml_path)
	inputs: dict[str, Any] = {}

	for out in outputs:
		name = out.get("name")
		typ = out.get("type")
		if not isinstance(name, str) or not isinstance(typ, str):
			continue

		if typ == "String" or "String" in typ:
			if not prompt:
				raise RuntimeError("Prompt string is empty.")
			# Keep it simple: most module_genai configs use a single prompt.
			inputs[name] = prompt
			continue

		if "Tensor" in typ:
			if video_path:
				inputs[name] = _load_video_tensor(video_path)
			elif image_path:
				inputs[name] = _load_image_tensor(image_path)
			else:
				raise RuntimeError("Image/Video path is empty.")
			continue

	# Fallback for common naming when YAML parsing didn't match.
	if not inputs:
		if not prompt:
			raise RuntimeError("Prompt string is empty.")
		if not (image_path or video_path):
			raise RuntimeError("Image/Video path is empty.")
		inputs["prompts_data"] = prompt
		inputs["img1"] = _load_video_tensor(video_path) if video_path else _load_image_tensor(image_path)

	return inputs


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Module GenAI VLM sample (Python): <Config_Yaml> <Prompt> <Image_Path:Optional> <Video_Path:Optional>"
	)
	parser.add_argument("config_yaml", help="Path to config.yaml")
	parser.add_argument("prompt", help="Prompt text")
	parser.add_argument("image_path", nargs="?", default="", help="Optional image path")
	parser.add_argument("video_path", nargs="?", default="", help="Optional video path (dir of frames)")
	args = parser.parse_args()

	cfg_yaml_path = Path(args.config_yaml)
	if not cfg_yaml_path.exists():
		raise FileNotFoundError(str(cfg_yaml_path))

	inputs = _build_inputs_from_cfg(cfg_yaml_path, args.prompt, args.image_path, args.video_path)

	pipe = openvino_genai.ModulePipeline(str(cfg_yaml_path))
	pipe.generate(**inputs)

	generated = pipe.get_output("generated_text")
	print("Generation Result:", str(generated))
	return 0

if __name__ == "__main__":
    raise SystemExit(main())

