// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include "utils.hpp"
#include <openvino/genai/module_genai/pipeline.hpp>
#include <filesystem>
#include <chrono>

static std::string get_config_ymal_path(int argc, char *argv[]) {
    if (argc > 1) {
        return std::string(argv[1]);
    }
    return "config.yaml";
}

static bool print_subword(std::string &&subword)
{
    return !(std::cout << subword << std::flush);
}

int test_genai_module_pipeline(int argc, char *argv[])
{
    std::cout << "== Init ModulePipeline" << std::endl;
    std::string config_fn = get_config_ymal_path(argc, argv);
    std::cout << "  == config_fn: " << config_fn << std::endl;
    ov::genai::module::ModulePipeline pipe(config_fn);
    ov::genai::module::PrintAllModulesConfig();

    std::string prompt = "Please describle this image";
    ov::Tensor image = utils::load_image("test_image.jpg");

    // std::cout << "question:\n";
    // std::getline(std::cin, prompt);
    for (int l = 0; l < 1; l++)
    {
        std::cout << "== Loop: [" << l << "] " << std::endl;
        std::cout << "Input image1_data first value: " << (int)image.data<uint8_t>()[0] << ", data type: " << image.get_element_type() << std::endl;
        // pipe.start_chat();

        ov::AnyMap inputs;
        inputs["prompts_data"] = prompt;
        inputs["image1_data"] = image;
        inputs["image2_data"] = image;

        auto t1 = std::chrono::high_resolution_clock::now();
        pipe.generate(inputs);
        // auto aa = pipe.generate(inputs, ov::genai::streamer(print_subword));
        auto t2 = std::chrono::high_resolution_clock::now();
        // std::cout << "result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

        // pipe.finish_chat();

        auto output_raw_data = pipe.get_output("raw_data").as<ov::Tensor>();
        std::cout << "Output raw data first value: " << output_raw_data.data<float>()[0] << std::endl;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    try
    {
        test_genai_module_pipeline(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Catch exceptions: " << error.what() << '\n';
    }
    return EXIT_SUCCESS;
}
