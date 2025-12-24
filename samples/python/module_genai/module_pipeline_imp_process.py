#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor
from pathlib import Path

# pip install pillow

def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return openvino_genai.StreamingStatus.RUNNING".


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', default="", help="Image file or dir with images")
    parser.add_argument('device', nargs='?', default='CPU', help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    # GPU and NPU can be used as well.
    # Note: If NPU is selected, only the language model will be run on the NPU.
    enable_compile_cache = dict()
    if args.device == "GPU":
        # Cache compiled models on disk for GPU to save time on the next run.
        # It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"

    pipe = openvino_genai.ModulePipeline()

    # config = openvino_genai.GenerationConfig()
    # config.max_new_tokens = 100

    # pipe.start_chat()
    inputs = dict()
    inputs["prompt"] = ""
    inputs["images"] = rgbs
    pipe.generate(inputs)
    # pipe.finish_chat()


if '__main__' == __name__:
    main()
