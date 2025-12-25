// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include "utils.hpp"
#include <openvino/genai/module_genai/pipeline.hpp>
#include <filesystem>
#include <chrono>
#include "ut_modules_base.hpp"

static std::string get_config_ymal_path(int argc, char *argv[]) {
    if (argc > 2) {
        return std::string(argv[2]);
    }
    return "config.yaml";
}

static bool compare_big_tensor(const ov::Tensor& output,
                               const std::vector<float>& expected_top,
                               const float& thr = 1e-3) {
    int real_size = std::min(expected_top.size(), output.get_size());
    bool bresult = true;
    for (int i = 0; i < real_size; ++i) {
        float val = static_cast<float>(output.data<float>()[i]);
        if (std::fabs(val - expected_top[i]) > thr) {
            bresult = false;
            std::cout << "Mismatch at index " << i << ": expected " << expected_top[i] << ", got " << val << std::endl;
        }
    }
    return bresult;
}

static bool print_subword(std::string &&subword)
{
    return !(std::cout << subword << std::flush);
}

int test_genai_module_ut_pipelines(int argc, char *argv[])
{
    std::cout << "== Init ModulePipeline" << std::endl;
    std::string config_fn = get_config_ymal_path(argc, argv);
    std::cout << "  == config_fn: " << config_fn << std::endl;
    ov::genai::module::ModulePipeline pipe(config_fn);
    ov::genai::module::PrintAllModulesConfig();

    ov::Tensor image = utils::load_image("ut_test_data/cat_120_100.png");

    // std::cout << "question:\n";
    // std::getline(std::cin, prompt);
    for (int l = 0; l < 1; l++)
    {
        std::cout << "== Loop: [" << l << "] " << std::endl;
        // pipe.start_chat();

        ov::AnyMap inputs;
        inputs["prompts_data"] = std::vector<std::string>{"Please describle this image"};
        inputs["img1"] = image;

        auto t1 = std::chrono::high_resolution_clock::now();
        pipe.generate(inputs);
        // auto aa = pipe.generate(inputs, ov::genai::streamer(print_subword));
        auto t2 = std::chrono::high_resolution_clock::now();
        // std::cout << "result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

        // pipe.finish_chat();

        auto output_merged_embedding = pipe.get_output("merged_embedding").as<ov::Tensor>();

        std::vector<float> expected_merged_embedding_start = {
            -0.0235291, 0.000759125, -0.0151825, -0.0136642, -0.00987244, 0.00531387, -0.00151825, 0.00683212, 0.012146, 0.0220184
        };
        CHECK(compare_big_tensor(output_merged_embedding, expected_merged_embedding_start, 1e-2), "merged_embedding not match expected values");
        std::cout << "Pipeline test passed" << std::endl;
    }
    return EXIT_SUCCESS;
}
