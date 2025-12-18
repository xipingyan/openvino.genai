
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>
#include <filesystem>

#include <fstream>
#include <string>
#include <sstream>
inline bool readFileToString(const std::string &filename, std::string &content)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        content.clear();
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    file.close();
    return true;
}

class CTestParam {
    public:
    CTestParam(){}
    void pasre_params(int argc, char *argv[])
    {
        auto help_fun = std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <video/img> <IMAGE_FILE OR DIR_WITH_IMAGES> <device> <prompts>");

        if (argc == 1)
        {
            throw help_fun;
        }
    
        if (2 == argc && std::string(argv[1]) == std::string("-h"))
        {
            throw help_fun;
        }
    
        if (3 <= argc)
        {
            input_video = std::string(argv[2]) == "video";
        }
    
        if (4 <= argc)
        {
            img_video_path = argv[3];
        }
        if (5 <= argc)
        {
            device = argv[4];
        }
        if (6 <= argc)
        {
            if (!readFileToString(argv[5], prompt))
            {
                prompt = argv[5];
            }
        }
        if (7 <= argc)
        {
            if (!readFileToString(argv[6], prompt2))
            {
                prompt2 = argv[6];
            }
        }

        model_path = argv[1];
        print_param();
    }

    void print_param() {
        std::cout << "== Params:" << std::endl;
        std::cout << "    model_path = " << model_path << std::endl;
        std::cout << "    input_video = " << input_video << std::endl;
        std::cout << "    img_video_path = " << img_video_path << std::endl;
        std::cout << "    device = " << device << std::endl;
        std::cout << "    prompt = " << prompt << std::endl;
        std::cout << "    prompt2 = " << prompt2 << std::endl;
    }

    std::string img_video_path = "../../cat_1.jpg";
    std::string model_path = "../../ov_model_i8/";
    bool input_video = true;
    std::string device = "GPU";
    std::string prompt;
    std::string prompt2;
};