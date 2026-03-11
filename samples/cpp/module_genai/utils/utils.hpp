
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace utils {

struct OmniInputParams {
    std::vector<std::string> prompts;
    std::vector<std::string> image_paths;
    std::vector<std::string> video_paths;
    std::vector<std::string> audio_paths;
    bool use_audio_in_video = false;
};

std::string get_input_arg(int argc,
                          char* argv[],
                          const std::string& key,
                          const std::string& default_value = std::string());

std::string any_to_string(const ov::Any& value);

bool readFileToString(const std::string& filename, std::string& content);

// Check if name contains any of the keys in the keys vector
bool contains_key(const std::string& name, const std::vector<std::string>& keys);

// Find parameter module from yaml config file
YAML::Node find_param_module_in_yaml(const std::filesystem::path& cfg_yaml_path);

OmniInputParams parse_omni_input_params(int argc, char* argv[]);

}  // namespace utils
