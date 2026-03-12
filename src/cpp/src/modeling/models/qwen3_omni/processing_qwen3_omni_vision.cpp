// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/processing_qwen3_omni_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <openvino/core/except.hpp>

namespace {

std::filesystem::path make_temp_path(const std::string& stem) {
    auto tmp_dir = std::filesystem::temp_directory_path();
    auto name = stem + "_" + std::to_string(std::rand()) + ".json";
    return tmp_dir / name;
}

std::string shell_quote(const std::string& value) {
#ifdef _WIN32
    // Windows CMD uses double quotes for quoting
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
#endif
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int64_t Qwen3OmniVisionProcess::round_by_factor(int64_t value, int64_t factor) {
    return static_cast<int64_t>(std::llround(static_cast<double>(value) / static_cast<double>(factor))) * factor;
}

int64_t Qwen3OmniVisionProcess::ceil_by_factor(int64_t value, int64_t factor) {
    return static_cast<int64_t>(std::ceil(static_cast<double>(value) / static_cast<double>(factor))) * factor;
}

int64_t Qwen3OmniVisionProcess::floor_by_factor(int64_t value, int64_t factor) {
    return static_cast<int64_t>(std::floor(static_cast<double>(value) / static_cast<double>(factor))) * factor;
}

std::pair<int64_t, int64_t> Qwen3OmniVisionProcess::smart_resize(
    int64_t height,
    int64_t width,
    const Qwen3OmniVisionResizeConfig& cfg) {
    if (height <= 0 || width <= 0 || cfg.image_factor <= 0) {
        OPENVINO_THROW("Invalid image shape/factor");
    }

    const auto max_side = std::max(height, width);
    const auto min_side = std::min(height, width);
    if (max_side / min_side > cfg.max_ratio) {
        OPENVINO_THROW("absolute aspect ratio must be smaller than ", cfg.max_ratio);
    }

    int64_t h_bar = std::max<int64_t>(cfg.image_factor, round_by_factor(height, cfg.image_factor));
    int64_t w_bar = std::max<int64_t>(cfg.image_factor, round_by_factor(width, cfg.image_factor));

    const double pixels = static_cast<double>(height) * static_cast<double>(width);
    if (h_bar * w_bar > cfg.max_pixels) {
        const double beta = std::sqrt(pixels / static_cast<double>(cfg.max_pixels));
        h_bar = floor_by_factor(static_cast<int64_t>(std::llround(height / beta)), cfg.image_factor);
        w_bar = floor_by_factor(static_cast<int64_t>(std::llround(width / beta)), cfg.image_factor);
    } else if (h_bar * w_bar < cfg.min_pixels) {
        const double beta = std::sqrt(static_cast<double>(cfg.min_pixels) / pixels);
        h_bar = ceil_by_factor(static_cast<int64_t>(std::llround(height * beta)), cfg.image_factor);
        w_bar = ceil_by_factor(static_cast<int64_t>(std::llround(width * beta)), cfg.image_factor);
    }

    return {h_bar, w_bar};
}

int64_t Qwen3OmniVisionProcess::smart_nframes(
    const nlohmann::json& video_info,
    int64_t total_frames,
    double video_fps,
    const Qwen3OmniVideoFrameConfig& cfg) {
    const bool has_fps = video_info.contains("fps");
    const bool has_nframes = video_info.contains("nframes");
    if (has_fps && has_nframes) {
        OPENVINO_THROW("Only accept either `fps` or `nframes`");
    }

    int64_t nframes = 0;
    if (has_nframes) {
        nframes = round_by_factor(video_info.at("nframes").get<int64_t>(), cfg.frame_factor);
    } else {
        const double use_fps = has_fps ? video_info.at("fps").get<double>() : cfg.fps;
        const int64_t min_frames = ceil_by_factor(
            video_info.value("min_frames", cfg.min_frames), cfg.frame_factor);
        const int64_t max_frames = floor_by_factor(
            std::min<int64_t>(video_info.value("max_frames", cfg.max_frames), total_frames), cfg.frame_factor);

        double estimated = static_cast<double>(total_frames) / std::max(video_fps, 1e-6) * use_fps;
        estimated = std::min<double>(std::max<double>(estimated, min_frames), max_frames);
        estimated = std::min<double>(estimated, total_frames);
        nframes = floor_by_factor(static_cast<int64_t>(std::llround(estimated)), cfg.frame_factor);
    }

    if (!(cfg.frame_factor <= nframes && nframes <= total_frames)) {
        OPENVINO_THROW("nframes should be in interval [", cfg.frame_factor, ", ", total_frames, "]");
    }
    return nframes;
}

std::vector<nlohmann::json> Qwen3OmniVisionProcess::extract_vision_info(const nlohmann::json& conversations) {
    std::vector<nlohmann::json> out;

    auto append_from_conversation = [&out](const nlohmann::json& conversation) {
        if (!conversation.is_array()) {
            return;
        }
        for (const auto& message : conversation) {
            if (!message.contains("content") || !message.at("content").is_array()) {
                continue;
            }
            for (const auto& element : message.at("content")) {
                const bool has_vision_key =
                    element.contains("image") || element.contains("image_url") || element.contains("video");
                const std::string type = element.value("type", "");
                const bool is_vision_type =
                    type == "image" || type == "image_url" || type == "video";
                if (has_vision_key || is_vision_type) {
                    out.push_back(element);
                }
            }
        }
    };

    if (conversations.is_array() && !conversations.empty() && conversations.front().is_object()) {
        append_from_conversation(conversations);
        return out;
    }

    if (conversations.is_array()) {
        for (const auto& conversation : conversations) {
            append_from_conversation(conversation);
        }
    }

    return out;
}

nlohmann::json Qwen3OmniVisionProcess::process_vision_info_via_python(
    const nlohmann::json& conversations,
    bool return_video_kwargs,
    const std::string& python_executable) {
    const auto bridge_script = std::filesystem::path(__FILE__).parent_path() / "processing_qwen3_omni_bridge.py";
    if (!std::filesystem::exists(bridge_script)) {
        OPENVINO_THROW("Bridge script not found: ", bridge_script.string());
    }

    const auto input_path = make_temp_path("qwen3_omni_vision_in");
    const auto output_path = make_temp_path("qwen3_omni_vision_out");

    {
        std::ofstream input_file(input_path);
        if (!input_file.is_open()) {
            OPENVINO_THROW("Failed to create temp input file: ", input_path.string());
        }
        input_file << conversations.dump();
    }

    std::string cmd = shell_quote(python_executable) + " " +
                      shell_quote(bridge_script.string()) +
                      " --mode vision --input " + shell_quote(input_path.string()) +
                      " --output " + shell_quote(output_path.string());
    if (return_video_kwargs) {
        cmd += " --return-video-kwargs";
    }

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::filesystem::remove(input_path);
        std::filesystem::remove(output_path);
        OPENVINO_THROW("Python bridge vision process failed with code: ", ret);
    }

    nlohmann::json result;
    {
        std::ifstream output_file(output_path);
        if (!output_file.is_open()) {
            std::filesystem::remove(input_path);
            std::filesystem::remove(output_path);
            OPENVINO_THROW("Failed to read bridge output file: ", output_path.string());
        }
        output_file >> result;
    }

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
    return result;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
