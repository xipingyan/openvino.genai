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


def run_pipeline_test(pipe, **kwargs):
    print("rgbs[0] shape:", kwargs['img1'].get_shape())
    pipe.generate(**kwargs)
    source_size = pipe.get_output("source_size")
    print("Output source_size:", source_size)

def get_yaml_config(image_model_path: str, device: str) -> str:
    img_preprocess_cfg = {
         'global_context': {
            'model_type': 'qwen2_5_vl'
        },
        'pipeline_modules': {
            'image_preprocessor': {
                'type': 'ImagePreprocessModule',
                'device': device,
                'description': 'Image or Video preprocessing.',
                'inputs': [
                    {
                        'name': 'image',
                        'type': 'OVTensor',
                        # For only one module, don't need to specify `source`. it will be set during runtime.
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
                    'model_path': image_model_path
                }
            }
        }
    }
    return yaml.dump(img_preprocess_cfg)

def get_yaml_full_config(image_model_path: str, device: str) -> str:
    cfg_data = {
        'global_context': {
            'model_type': 'qwen2_5_vl'
        },
        'pipeline_modules': {
            'pipeline_params': {
                'type': 'ParameterModule',
                'device': device,
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
                'device': device,
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
                    'model_path': image_model_path
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
    return yaml.dump(cfg_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', default="", help="Image file or dir with images")
    parser.add_argument('model_dir', default="", help="Path to the directory with models")
    parser.add_argument('device', nargs='?', default='CPU', help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    enable_compile_cache = dict()
    if args.device == "GPU":
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"

    # yaml config
    cfg_yaml_content = get_yaml_config(args.model_dir, args.device)

    inputs = {'img1': rgbs[0]}
    
    # Test 1: Config File Path
    print("\n--- Test 1: Initialize with Config File Path ---")
    fn = "module_pipeline_imp_process.yaml"
    with open(fn, "w") as f:
        f.write(cfg_yaml_content)

    pipe_from_file = openvino_genai.ModulePipeline(config_yaml_path=fn)
    run_pipeline_test(pipe_from_file, **inputs)

    # Test 2: Config String
    print("\n--- Test 2: Initialize with Config String ---")
    # Pass the yaml content string directly
    pipe_from_string = openvino_genai.ModulePipeline(config_yaml_content=cfg_yaml_content)
    run_pipeline_test(pipe_from_string, **inputs)


if '__main__' == __name__:
    main()
