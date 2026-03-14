// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

enum class Qwen3OmniAudioInputType {
    kUnknown,
    kNumpyArray,
    kDataUri,
    kHttpUrl,
    kFileUrl,
    kLocalPath,
};

struct Qwen3OmniAudioEntry {
    std::string source;
    Qwen3OmniAudioInputType type = Qwen3OmniAudioInputType::kUnknown;
    bool from_video = false;
};

class Qwen3OmniAudioProcess {
public:
    static Qwen3OmniAudioInputType classify_source(const nlohmann::json& value);
    static std::vector<Qwen3OmniAudioEntry> extract_audio_entries(
        const nlohmann::json& conversations,
        bool use_audio_in_video);

    static nlohmann::json process_audio_info_via_python(
        const nlohmann::json& conversations,
        bool use_audio_in_video,
        const std::string& python_executable = "/home/wanglei/py_venv/dev_env/bin/python");

    static nlohmann::json process_audio_features_via_python(
        const nlohmann::json& conversations,
        const std::string& model_dir,
        bool use_audio_in_video,
        const std::string& python_executable = "/home/wanglei/py_venv/dev_env/bin/python");
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
