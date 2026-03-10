// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <openvino/runtime/properties.hpp>

#include <stdexcept>

#include "utils/vision_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "utils/utils.hpp"
#include "utils/audio_utils.hpp"

inline ov::AnyMap parse_inputs_from_yaml_cfg_for_vlm(const std::filesystem::path& cfg_yaml_path,
                                                     const std::string& prompt = std::string{},
                                                     const std::string& image_path = std::string{},
                                                     const std::string& video_path = std::string{},
                                                     const std::string& audio_path = std::string{}) {
    ov::AnyMap inputs;
    YAML::Node input_params = utils::find_param_module_in_yaml(cfg_yaml_path);

    // Loop input_params to find "prompt", "image", "video", "audio"
    for (const auto& entry : input_params) {
        if (!entry["name"] || !entry["type"]) {
            continue;
        }

        const std::string param_name = entry["name"].as<std::string>();
        const std::string param_type = entry["type"].as<std::string>();

        if (param_type == "String" && utils::contains_key(param_name, {"prompt"})) {
            if (prompt.empty()) {
                throw std::runtime_error("Prompt string is empty.");
            }
            inputs[param_name] = prompt;
            continue;
        }

        if (param_type == "OVTensor" && utils::contains_key(param_name, {"img", "image"})) {
            if (image_path.empty()) {
                throw std::runtime_error("Image path is empty.");
            }
            inputs[param_name] = image_utils::load_image(image_path);
            continue;
        }

        if (param_type == "OVTensor" && utils::contains_key(param_name, {"video"})) {
            if (video_path.empty()) {
                throw std::runtime_error("Video path is empty.");
            }
            inputs[param_name] = image_utils::load_video(video_path);
            continue;
        }

        if (param_type == "OVTensor" && utils::contains_key(param_name, {"audio"})) {
            if (audio_path.empty()) {
                throw std::runtime_error("Audio path is empty.");
            }
            inputs[param_name] = audio_utils::load_audio(audio_path);
            continue;
        }
    }
    return inputs;
}

int main(int argc, char* argv[]) {
    try {
        if (argc <= 1) {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                     "\n"
                                     "  -cfg config.yaml \n"
                                     "  -cache_dir: [Optional] string path, default empty\n"
                                     "  -prompt: input prompt\n"
                                     "  -img: [Optional] image path\n"
                                     "  -video: [Optional] video path\n"
                                     "  -audio: [Optional] audio path\n"
                                     "  -warmup: [Optional] number of warmup runs, default 0\n"
                                     "  -perf: [Optional] set to 1 to print performance metrics, default 0\n");
        }

        std::filesystem::path config_path = utils::get_input_arg(argc, argv, "-cfg", std::string{});
        std::string cache_dir = utils::get_input_arg(argc, argv, "-cache_dir", std::string{});
        std::string prompt = utils::get_input_arg(argc, argv, "-prompt", std::string{});
        std::string img_path = utils::get_input_arg(argc, argv, "-img", std::string{});
        std::string video_path = utils::get_input_arg(argc, argv, "-video", std::string{});
        std::string audio_path = utils::get_input_arg(argc, argv, "-audio", std::string{});
        int warmup = std::stoi(utils::get_input_arg(argc, argv, "-warmup", std::string("0")));
        bool perf = std::stoi(utils::get_input_arg(argc, argv, "-perf", std::string("0")));

        ov::AnyMap inputs = parse_inputs_from_yaml_cfg_for_vlm(config_path, prompt, img_path, video_path, audio_path);

        for (const auto& [key, value] : inputs) {
            std::cout << "[Input] " << key << ": ";
            if (value.is<std::string>()) {
                std::cout << value.as<std::string>();
            } else if (value.is<ov::Tensor>()) {
                const auto& tensor = value.as<ov::Tensor>();
                std::cout << "Tensor (rank=" << tensor.get_shape().size() << ")";
            } else {
                std::cout << "<non-string input>";
            }
            std::cout << std::endl;
        }

        ov::AnyMap properties{};
        if (!cache_dir.empty()) {
            properties.insert({ov::cache_dir(cache_dir)});
        }

        ov::genai::module::ModulePipeline pipe(config_path, properties);

        for (int i = 0; i < warmup; ++i) {
            std::cout << "[Warmup] Run " << (i + 1) << "/" << warmup << std::endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            pipe.generate(inputs);
            auto t2 = std::chrono::high_resolution_clock::now();
            if (perf) {
                auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                std::cout << "[Warmup] Duration: " << diff << " ms" << std::endl;
            }
        }

        std::cout << "[Generation] Running main generation..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();

        pipe.generate(inputs);

        auto t2 = std::chrono::high_resolution_clock::now();
        if (perf) {
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "[Generation] Duration: " << diff << " ms" << std::endl;
        }

        std::cout << "Generation Result: " << pipe.get_output("generated_text").as<std::string>() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}