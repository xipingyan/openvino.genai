# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import cv2
from PIL import Image
import logging as log
from .model_utils import get_param_from_file
from .parse_json_data import parse_text_json_data

def get_text_prompt(args):
    text_list = []
    output_data_list, is_json_data = get_param_from_file(args, 'prompt')
    if is_json_data is True:
        text_param_list = parse_text_json_data(output_data_list)
        if len(text_param_list) > 0:
            for text in text_param_list:
                text_list.append(text)
    else:
        text_list.append(output_data_list[0])
    return text_list


def print_frames_number(func):
    def inner(video_path, decym_frames):
        log.info(f"Input video file: {video_path}")
        if decym_frames is not None:
            log.info(f"Requested to reduce into {decym_frames} frames")
        out_frames = func(video_path, decym_frames)
        log.info(f"Final frames number: {len(out_frames)}")
        return out_frames
    return inner

@print_frames_number
def split_video_into_frames(video_path, decym_frames=None):
    supported_files = set([".mp4"])

    assert os.path.exists(video_path), f"no input video file: {video_path}"
    assert video_path.suffix.lower() in supported_files, "no supported video file"
    cap = cv2.VideoCapture(video_path)

    output_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        output_frames.append(pil_image)
    if decym_frames is None:
        return output_frames
    if int(decym_frames) == 0:
        return output_frames

    # decimation procedure
    # decim_fames is required frame number if positive
    # or decimation factor if negative

    decym_frames = int(decym_frames)
    if decym_frames > 0:
        if len(output_frames) <= decym_frames:
            return output_frames
        decym_factor = int(len(output_frames) / decym_frames)
    else: decym_factor = -decym_frames
    if decym_factor >= 2:
        return list(output_frames[::decym_factor])
    return output_frames
