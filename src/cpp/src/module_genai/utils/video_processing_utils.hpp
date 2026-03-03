// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <cstdint>
#include <utility>
#include <vector>
#include <string>

// Refer: transformers/video_processing_utils.py
class BaseVideoProcessor
{
protected:
    bool resample = false;
    std::vector<float> image_mean = {};
    std::vector<float> image_std = {};
    std::vector<int> size = {};
    int size_divisor = 0;
    bool default_to_square = true;
    std::vector<int> crop_size = {};
    bool do_resize = false;
    bool do_center_crop = false;
    bool do_rescale = false;
    float rescale_factor = 1.0f / 255.0f;
    bool do_normalize = false;
    bool do_convert_rgb = false;
    bool do_sample_frames = false;
    float fps = 0.0f;
    int num_frames = 0;
    bool return_metadata = false;
public:
    BaseVideoProcessor(/* args */) = default;
    ~BaseVideoProcessor() = default;
};