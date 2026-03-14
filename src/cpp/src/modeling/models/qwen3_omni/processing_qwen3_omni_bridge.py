#!/usr/bin/env python3

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LINUX_RELEASE_BASE = Path("/home/wanglei/model/Qwen3_omni/qwen3_omni_4B_final_release")
_RELEASE_DIR_NAME = "qwen3_omni_4B_final_release"


def _find_release_base(model_dir: str = "") -> Path:
    """Locate the qwen3_omni_4B_final_release directory.

    Search order:
      1. QWEN3_OMNI_RELEASE_BASE env var
      2. Sibling of model_dir  (e.g. .../wangleis/qwen3_omni_4B_final_release)
      3. Hard-coded Linux fallback
    """
    env_base = os.environ.get("QWEN3_OMNI_RELEASE_BASE", "")
    if env_base:
        p = Path(env_base)
        if p.exists():
            return p
    if model_dir:
        candidate = Path(model_dir).parent / _RELEASE_DIR_NAME
        if candidate.exists():
            return candidate
    if _LINUX_RELEASE_BASE.exists():
        return _LINUX_RELEASE_BASE
    raise FileNotFoundError(
        f"Cannot find {_RELEASE_DIR_NAME}. Set QWEN3_OMNI_RELEASE_BASE or "
        f"place it as a sibling of your model directory."
    )


def _default_utils_dir(model_dir: str = "") -> Path:
    return _find_release_base(model_dir) / "qwen_omni_utils" / "v2_5"


def _default_internal_src(model_dir: str = "") -> Path:
    base = _find_release_base(model_dir)
    matches = glob.glob(str(base / "transformers-internal-q3o-dense-*" / "src"))
    if not matches:
        raise FileNotFoundError("Cannot locate internal transformers source for qwen3_omni")
    return Path(matches[0])


def _default_model_dir() -> Path:
    return Path("/home/wanglei/model/Qwen3_omni/Qwen3-Omni-4B-Instruct-multilingual")


def _compute_audio_features(conversations, model_dir: str, use_audio_in_video: bool, utils_dir: Path):
    if not isinstance(conversations, list) or len(conversations) == 0:
        return {
            "input_ids": None,
            "attention_mask": None,
            "position_ids": None,
            "visual_pos_mask": None,
            "rope_deltas": None,
            "visual_embeds_padded": None,
            "deepstack_padded": [],
            "audio_features": None,
            "audio_pos_mask": None,
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "video_second_per_grid": None,
        }

    import torch

    use_model_dir = model_dir if model_dir else str(_default_model_dir())
    base = _find_release_base(use_model_dir)
    internal_src = _default_internal_src(use_model_dir)
    if str(internal_src) not in sys.path:
        sys.path.insert(0, str(internal_src))
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    from transformers import Qwen3OmniForConditionalGeneration, Qwen3OmniProcessor

    use_model_dir = model_dir if model_dir else str(_default_model_dir())
    model = Qwen3OmniForConditionalGeneration.from_pretrained(
        use_model_dir,
        dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    processor = Qwen3OmniProcessor.from_pretrained(use_model_dir, trust_remote_code=True, use_fast=False)

    process_mm_info = None
    try:
        from qwen_omni_utils import process_mm_info as _process_mm_info

        process_mm_info = _process_mm_info
    except Exception:
        process_mm_info = None

    audio_py = utils_dir / "audio_process.py"
    vision_py = utils_dir / "vision_process.py"
    if process_mm_info is None:
        if not audio_py.exists() or not vision_py.exists():
            raise FileNotFoundError(
                f"Cannot find qwen_omni utils files under {utils_dir}. "
                f"Expected {audio_py.name} and {vision_py.name}"
            )
        audio_mod = _load_module("qwen3_omni_audio_process_audio_features", audio_py)
        vision_mod = _load_module("qwen3_omni_vision_process_audio_features", vision_py)

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    if process_mm_info is not None:
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)
    else:
        audios = audio_mod.process_audio_info(conversations, use_audio_in_video)
        images, videos = vision_mod.process_vision_info(conversations, False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"].long()
        token_embeds = model.thinker.get_input_embeddings()(input_ids)

        audio_features = None
        if "input_features" in inputs and "feature_attention_mask" in inputs:
            audio_feature_lengths = inputs.get("audio_feature_lengths", None)
            audio_features = model.thinker.get_audio_features(
                inputs["input_features"],
                inputs["feature_attention_mask"],
                audio_feature_lengths,
            )

        image_features = None
        image_features_multiscale = None
        if "pixel_values" in inputs and "image_grid_thw" in inputs:
            image_features, image_features_multiscale = model.thinker.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])

        _, _, audio_mask = model.thinker.get_placeholder_mask(
            input_ids,
            inputs_embeds=token_embeds,
        )
        image_mask = None
        if image_features is not None:
            image_mask, _, _ = model.thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=token_embeds,
                image_features=image_features,
            )

        image_grid_thw = inputs["image_grid_thw"].long() if "image_grid_thw" in inputs else None
        video_second_per_grid = inputs["video_second_per_grid"] if "video_second_per_grid" in inputs else None
        if "audio_feature_lengths" in inputs:
            audio_seqlens = inputs["audio_feature_lengths"].long()
        elif "feature_attention_mask" in inputs:
            audio_seqlens = inputs["feature_attention_mask"].to(torch.long).sum(dim=-1)
        else:
            audio_seqlens = None
        position_ids, rope_deltas = model.thinker.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask,
            use_audio_in_video,
            audio_seqlens,
            video_second_per_grid,
        )
        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        rope_deltas = rope_deltas - delta0

        if audio_features is None:
            hidden = token_embeds.shape[-1]
            audio_features = torch.zeros((input_ids.shape[0], 0, hidden), dtype=torch.float32)
            audio_pos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        else:
            audio_features = audio_features.to(torch.float32)
            audio_pos_mask = audio_mask[..., 0].to(torch.bool)

        audio_features_padded = torch.zeros((input_ids.shape[0], input_ids.shape[1], token_embeds.shape[-1]), dtype=torch.float32)
        if audio_features.numel() > 0 and audio_pos_mask is not None:
            audio_features_padded[audio_pos_mask] = audio_features

        if image_mask is None:
            visual_pos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        else:
            visual_pos_mask = image_mask[..., 0].to(torch.bool)

        hidden = token_embeds.shape[-1]
        visual_embeds_padded = torch.zeros((input_ids.shape[0], input_ids.shape[1], hidden), dtype=torch.float32)
        deepstack_padded = []
        if image_features is not None and image_mask is not None:
            visual_embeds_padded[visual_pos_mask] = image_features.to(torch.float32)
            if image_features_multiscale is not None:
                for ds in image_features_multiscale:
                    cur = torch.zeros((input_ids.shape[0], input_ids.shape[1], ds.shape[-1]), dtype=torch.float32)
                    cur[visual_pos_mask] = ds.to(torch.float32)
                    deepstack_padded.append(cur)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids.long(),
        "visual_pos_mask": visual_pos_mask,
        "rope_deltas": rope_deltas.long(),
        "visual_embeds_padded": visual_embeds_padded,
        "deepstack_padded": deepstack_padded,
        "audio_features": audio_features_padded,
        "audio_pos_mask": audio_pos_mask,
        "feature_attention_mask": inputs.get("feature_attention_mask", None),
        "audio_feature_lengths": inputs.get("audio_feature_lengths", None),
        "video_second_per_grid": inputs.get("video_second_per_grid", None),
    }


def _to_serializable(value):
    try:
        import torch
    except Exception:
        torch = None

    try:
        from PIL import Image
    except Exception:
        Image = None

    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if torch is not None and torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
        return {
            "kind": "tensor",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tolist(),
        }
    if Image is not None and isinstance(value, Image.Image):
        arr = np.array(value)
        return {
            "kind": "image",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tolist(),
        }
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audio", "vision", "audio-features"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--return-video-kwargs", action="store_true")
    parser.add_argument("--model-dir", default=os.environ.get("QWEN3_OMNI_MODEL_DIR", ""))
    parser.add_argument("--utils-dir", default=os.environ.get("QWEN3_OMNI_UTILS_PATH", ""))
    args = parser.parse_args()

    utils_dir = Path(args.utils_dir) if args.utils_dir else _default_utils_dir(args.model_dir)
    audio_py = utils_dir / "audio_process.py"
    vision_py = utils_dir / "vision_process.py"

    if args.mode in ("audio", "vision") and (not audio_py.exists() or not vision_py.exists()):
        raise FileNotFoundError(
            f"Cannot find qwen_omni utils files under {utils_dir}. "
            f"Expected {audio_py.name} and {vision_py.name}"
        )

    audio_mod = None
    vision_mod = None
    if args.mode in ("audio", "vision"):
        audio_mod = _load_module("qwen3_omni_audio_process_bridge", audio_py)
        vision_mod = _load_module("qwen3_omni_vision_process_bridge", vision_py)

    with open(args.input, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list) or len(conversations) == 0:
        if args.mode == "audio":
            result = None
        elif args.mode == "audio-features":
            result = {
                "input_ids": None,
                "attention_mask": None,
                "position_ids": None,
                "visual_pos_mask": None,
                "rope_deltas": None,
                "visual_embeds_padded": None,
                "deepstack_padded": [],
                "audio_features": None,
                "audio_pos_mask": None,
                "feature_attention_mask": None,
                "audio_feature_lengths": None,
                "video_second_per_grid": None,
            }
        else:
            if args.return_video_kwargs:
                result = (None, None, {"fps": []})
            else:
                result = (None, None)
    elif args.mode == "audio":
        result = audio_mod.process_audio_info(conversations, args.use_audio_in_video)
    elif args.mode == "audio-features":
        result = _compute_audio_features(conversations, args.model_dir, args.use_audio_in_video, utils_dir)
    else:
        result = vision_mod.process_vision_info(conversations, args.return_video_kwargs)

    serializable = _to_serializable(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
