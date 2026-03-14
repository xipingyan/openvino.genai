// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3OmniVisionResizeConfig {
    int64_t image_factor = 28;
    int64_t min_pixels = 4 * 28 * 28;
    int64_t max_pixels = 16384 * 28 * 28;
    int64_t max_ratio = 200;
};

struct Qwen3OmniVideoFrameConfig {
    int64_t frame_factor = 2;
    float fps = 2.0f;
    int64_t min_frames = 4;
    int64_t max_frames = 768;
};

class Qwen3OmniVisionProcess {
public:
    static int64_t round_by_factor(int64_t value, int64_t factor);
    static int64_t ceil_by_factor(int64_t value, int64_t factor);
    static int64_t floor_by_factor(int64_t value, int64_t factor);

    static std::pair<int64_t, int64_t> smart_resize(
        int64_t height,
        int64_t width,
        const Qwen3OmniVisionResizeConfig& cfg = {});

    static int64_t smart_nframes(
        const nlohmann::json& video_info,
        int64_t total_frames,
        double video_fps,
        const Qwen3OmniVideoFrameConfig& cfg = {});

    static std::vector<nlohmann::json> extract_vision_info(const nlohmann::json& conversations);

    static nlohmann::json process_vision_info_via_python(
        const nlohmann::json& conversations,
        bool return_video_kwargs = false,
        const std::string& python_executable = "/home/wanglei/py_venv/dev_env/bin/python");
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
