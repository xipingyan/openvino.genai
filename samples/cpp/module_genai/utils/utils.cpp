
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <filesystem>
#include <fstream>
#include <openvino/runtime/tensor.hpp>
#include <sstream>
#include <string>

#include "yaml-cpp/yaml.h"

namespace utils {

bool readFileToString(const std::string& filename, std::string& content) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        content.clear();
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    file.close();
    return true;
}

std::string any_to_string(const ov::Any& value) {
    if (value.is<std::string>()) {
        return value.as<std::string>();
    }
    if (value.is<int>()) {
        return std::to_string(value.as<int>());
    }
    if (value.is<int64_t>()) {
        return std::to_string(value.as<int64_t>());
    }
    if (value.is<float>()) {
        return std::to_string(value.as<float>());
    }
    if (value.is<double>()) {
        return std::to_string(value.as<double>());
    }
    if (value.is<bool>()) {
        return value.as<bool>() ? "true" : "false";
    }
    if (value.is<ov::Tensor>()) {
        return "ov::Tensor[" + value.as<ov::Tensor>().get_shape().to_string() + "]";
    }
    return "<unsupported>";
}

YAML::Node find_param_module_in_yaml(const std::filesystem::path& cfg_yaml_path) {
    YAML::Node config = YAML::LoadFile(cfg_yaml_path.string());
    auto pipeline_modules = config["pipeline_modules"];
    // loop pipeline_modules to find a node with type "ParameterModule"
    for (const auto& module : pipeline_modules) {
        if (module.second["type"] && module.second["type"].as<std::string>() == "ParameterModule") {
            return module.second["outputs"];
        }
    }
    throw std::runtime_error("Could not find ParameterModule in config YAML.");
}

bool contains_key(const std::string& name, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        if (name.find(key) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string get_input_arg(int argc, char* argv[], const std::string& key, const std::string& default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == key && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return default_value;
}

OmniInputParams parse_omni_input_params(int argc, char* argv[]) {
    OmniInputParams params {};
    int i = 0;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "-prompt") {
            if (i + 1 < argc) {
                params.prompts.emplace_back(argv[i + 1]);
            } else {
                params.prompts.emplace_back("");
            }
        } else if (arg == "-img") {
            if (i + 1 < argc) {
                params.image_paths.emplace_back(argv[i + 1]);
            } else {
                params.image_paths.emplace_back("");
            }
        } else if (arg == "-video") {
            if (i + 1 < argc) {
                params.video_paths.emplace_back(argv[i + 1]);
            } else {
                params.video_paths.emplace_back("");
            }
        } else if (arg == "-audio") {
            if (i + 1 < argc) {
                params.audio_paths.emplace_back(argv[i + 1]);
            } else {
                params.audio_paths.emplace_back("");
            }
        } else if (arg == "-use_audio_in_video") {
            if (i + 1 < argc) {
                params.use_audio_in_video = std::stoi(argv[i + 1]) != 0;
            }
        }
        i++;
    }
    return params;
}

}  // namespace utils