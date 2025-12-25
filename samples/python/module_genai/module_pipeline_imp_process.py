#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor
from pathlib import Path
import yaml

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
    # image_data = np.array(pic)
    # return Tensor(image_data)

    # 3dim to 4dim with batch size 1
    return Tensor(np.stack([pic], axis=0))

def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', default="", help="Image file or dir with images")
    parser.add_argument('model_dir', default="", help="Path to the directory with models")
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

    # yaml config
    cfg_data = {
        'global_context': {
            'model_type': 'qwen2_5_vl'
        },
        'pipeline_modules': {
            'pipeline_params': {
                'type': 'ParameterModule',
                'device': args.device,
                'description': 'Pipeline parameters module.',
                'outputs': [
                    {
                        'name': 'img1',
                        'type': 'OVTensor'
                    }
                ]
            },
            'image_preprocessor': {
                'type': 'ImagePreprocessModule',
                'device': args.device,
                'description': 'Image or Video preprocessing.',
                'inputs': [
                    {
                        'name': 'image',
                        'type': 'OVTensor',
                        'source': 'pipeline_params.img1'
                    }
                ],
                'outputs': [
                    {
                        'name': 'raw_data',
                        'type': 'OVTensor'
                    },
                    {
                        'name': 'source_size',
                        'type': 'VecInt'
                    }
                ],
                'params': {
                    'target_resolution': str([224, 224]),
                    'mean': str([0.485, 0.456, 0.406]),
                    'std': str([0.229, 0.224, 0.225]),
                    'model_path': args.model_dir
                }
            },
            'pipeline_results': {
                'type': 'ResultModule',
                'inputs': [
                    {
                        'name': 'raw_data',
                        'type': 'OVTensor',
                        'source': 'image_preprocessor.raw_data'
                    },
                    {
                        'name': 'source_size',
                        'type': 'VecInt',
                        'source': 'image_preprocessor.source_size'
                    }
                ]
            }
        }
    }
    cfg_yaml = yaml.dump(cfg_data)
    # convert yaml str to local file
    fn = "module_pipeline_imp_process.yaml"
    with open(fn, "w") as f:
        f.write(cfg_yaml)

    pipe = openvino_genai.ModulePipeline(fn)

    # config = openvino_genai.GenerationConfig()
    # config.max_new_tokens = 100

    # pipe.start_chat()
    print("rgbs[0] shape:", rgbs[0].get_shape())
    pipe.generate(img1=rgbs[0])
    source_size = pipe.get_output("source_size")
    print("Output source_size:", source_size)

    # pipe.finish_chat()


if '__main__' == __name__:
    main()
