// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "private.hpp"

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        if (std::string(argv[1]) == std::string("ut_modules")) {
            test_genai_module_ut_modules(argc, argv);
        }
        else if (std::string(argv[1]) == std::string("ut_pipelines")) {
            test_genai_module_ut_pipelines(argc, argv);
        }
    }
    else {
        std::cout << "Please specify test to run: ut_modules or ut_pipelines" << std::endl;
    }
    return EXIT_SUCCESS;
}
