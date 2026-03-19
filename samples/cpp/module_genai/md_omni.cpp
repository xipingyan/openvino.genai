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

inline ov::AnyMap parse_inputs_for_omni(const utils::OmniInputParams& params) {
    ov::AnyMap inputs;

    if (!params.prompts.empty()) {
        inputs["prompts"] = params.prompts;
    }
    if (!params.image_paths.empty()) {
        std::vector<ov::Tensor> images;
        for (const auto& image_path : params.image_paths) {
            images.push_back(image_utils::load_image(image_path));
        }
        inputs["images"] = images;
    }

    std::vector<ov::Tensor> audios {};
    std::vector<int> use_audio_in_video {};

    if (!params.video_paths.empty()) {
        std::vector<ov::Tensor> videos;
        for (const auto& video_path : params.video_paths) {
            image_utils::VideoLoadResult video = image_utils::load_video_with_audio(video_path, params.use_audio_in_video);
            videos.push_back(video.frames);
            if (params.use_audio_in_video) {
                audios.push_back(video.audio);
                use_audio_in_video.push_back(1);
            } else {
                use_audio_in_video.push_back(0);
            }
        }
        inputs["videos"] = videos;
    }

    if (!params.audio_paths.empty()) {
        for (const auto& audio_path : params.audio_paths) {
            audios.push_back(audio_utils::load_audio(audio_path));
        }
    }

    if (!audios.empty()) {
        inputs["audios"] = audios;
    }
    if (!use_audio_in_video.empty()) {
        inputs["use_audio_in_video"] = use_audio_in_video;
    }

    return inputs;
}

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string> args(argv, argv + argc);
        auto usage_prompts = std::string{"Usage: "} + argv[0] +
                             "\n"
                             "  -h or --help for more details\n"
                             "  -cfg config.yaml \n"
                             "  -cache_dir: [Optional] string path, default empty\n"
                             "  -prompt: input prompt\n"
                             "  -img: [Optional] image path\n"
                             "  -video: [Optional] video path\n"
                             "  -audio: [Optional] audio path\n"
                             "  -use_audio_in_video: [Optional] set to 1 if the video contains audio and you want to "
                             "use the audio, default 0\n"
                             "  -tts: [Optional] set to 1 to use tts, default 0\n"
                             "  -warmup: [Optional] number of warmup runs, default 0\n"
                             "  -perf: [Optional] set to 1 to print performance metrics, default 0\n";
        if (argc <= 1) {
            throw std::runtime_error(usage_prompts);
        } else if (utils::contains_key("-h", args) ||
                   utils::contains_key("--help", args)) {
            std::cout << usage_prompts << std::endl;
            return EXIT_SUCCESS;
        }

        std::filesystem::path config_path = utils::get_input_arg(argc, argv, "-cfg", std::string{});
        std::string cache_dir = utils::get_input_arg(argc, argv, "-cache_dir", std::string{});
        int warmup = std::stoi(utils::get_input_arg(argc, argv, "-warmup", std::string("0")));
        bool perf = std::stoi(utils::get_input_arg(argc, argv, "-perf", std::string("0")));
        bool use_tts = std::stoi(utils::get_input_arg(argc, argv, "-tts", std::string("0"))) != 0;

        utils::OmniInputParams input_params = utils::parse_omni_input_params(argc, argv);
        ov::AnyMap inputs = parse_inputs_for_omni(input_params);

        std::cout << "Pipeline inputs:" << std::endl;
        for (const auto& [key, value] : inputs) {
            std::cout << "  - [" << key << "]: " << utils::any_to_string(value) << std::endl;
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

        if (!use_tts) {
            std::cout << "Generation Result: " << pipe.get_output("generated_text").as<std::string>() << std::endl;
        } else {
            std::vector<ov::Tensor> audios = pipe.get_output("audios").as<std::vector<ov::Tensor>>();
            std::vector<int> sample_rates = pipe.get_output("sample_rates").as<std::vector<int>>();
            std::vector<std::string> texts = pipe.get_output("generated_texts").as<std::vector<std::string>>();
            for (size_t i = 0; i < audios.size(); ++i) {
                std::string output_path = "output_audio_" + std::to_string(i) + ".wav";
                auto audio_data = audios[i].data<const float>();
                const size_t sample_count = audios[i].get_size();
                std::cout << "Generated text: " << texts[i] << std::endl;
                audio_utils::write_wav(output_path, audio_data, sample_count, sample_rates[i]);
                std::cout << "Saved generated audio to: " << output_path << std::endl;
            }
        }
        
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}