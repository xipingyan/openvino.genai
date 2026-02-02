# Copyright (C) 2026 Intel Corporation
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
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Add batch dimension: [H, W, 3] -> [1, H, W, 3]
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return Tensor(arr)


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


def _parse_inputs_from_yaml_cfg_for_vlm(cfg_yaml_path: Path, prompt: str, image_path: str, video_path: str) -> dict[str, Any]:
    outputs = _find_parameter_module_outputs(cfg_yaml_path)
    inputs: dict[str, Any] = {}

    for out in outputs:
        name = out.get("name")
        typ = out.get("type")
        if "prompt" in name.lower() and typ == "String":
            inputs[name] = prompt
            continue

        if "image" in name.lower() or "img" in name.lower() and typ == "OVTensor":
            if image_path:
                inputs[name] = _load_image_tensor(image_path)
            else:
                raise RuntimeError("Image path is empty.")
            continue

        if "video" in name.lower() and typ == "OVTensor":
            if video_path:
                inputs[name] = _load_video_tensor(video_path)
            else:
                raise RuntimeError("Video path is empty.")
            continue

    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Module GenAI VLM sample (Python). "
            "Preferred usage: --cfg <config.yaml> --prompt <text> [--img <image>] [--video <frames_dir>]"
        )
    )

    # Preferred flags
    parser.add_argument("--cfg", dest="cfg", default="", help="Path to config.yaml")
    parser.add_argument("--prompt", dest="prompt", default="", help="Prompt text")
    parser.add_argument("--img", dest="img", default="", help="Optional image path")
    parser.add_argument("--video", dest="video", default="", help="Optional video path (dir of frames)")

    # Backward-compatible positionals
    parser.add_argument("cfg_pos", nargs="?", default="", help=argparse.SUPPRESS)
    parser.add_argument("prompt_pos", nargs="?", default="", help=argparse.SUPPRESS)
    parser.add_argument("img_pos", nargs="?", default="", help=argparse.SUPPRESS)
    parser.add_argument("video_pos", nargs="?", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg_value = args.cfg or args.cfg_pos
    prompt_value = args.prompt or args.prompt_pos
    img_value = args.img or args.img_pos
    video_value = args.video or args.video_pos

    if not cfg_value:
        raise RuntimeError("Missing --cfg <config.yaml>")
    if not prompt_value:
        raise RuntimeError("Missing --prompt <text>")

    cfg_yaml_path = Path(cfg_value)
    if not cfg_yaml_path.exists():
        raise FileNotFoundError(str(cfg_yaml_path))

    inputs = _parse_inputs_from_yaml_cfg_for_vlm(cfg_yaml_path, prompt_value, img_value, video_value)

    for key, value in inputs.items():
        print(f"[Input] {key}: {str(value)}")

    print("type(inputs):", type(inputs["img1"]))

    pipe = openvino_genai.ModulePipeline(str(cfg_yaml_path))
    pipe.generate(**inputs)

    generated = pipe.get_output("generated_text")
    print("Generation Result:", str(generated))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

