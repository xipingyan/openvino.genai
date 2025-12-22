// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ut_modules_base.hpp"

int test_genai_module_ut_modules(int argc, char *argv[])
{
    std::cout << "== Starting Module Tests ==" << std::endl;
    
    const auto& tests = TestRegistry::get().get_tests();
    
    if (tests.empty()) {
        std::cout << "No tests registered." << std::endl;
        return EXIT_SUCCESS;
    }

    int passed = 0;
    int failed = 0;
    std::vector<std::string> failed_tests;

    for (const auto& [name, creator] : tests) {
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Running Test: " << name << std::endl;
        try {
            auto test = creator();
            test->run();
            std::cout << "Test " << name << " PASSED" << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << "Test " << name << " FAILED: " << e.what() << std::endl;
            failed++;
            failed_tests.emplace_back(name);
        } catch (...) {
            std::cout << "Test " << name << " FAILED: Unknown error" << std::endl;
            failed++;
            failed_tests.emplace_back(name);
        }
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Tests Completed. Passed: " << passed << ", Failed: " << failed << std::endl;
    if (failed > 0) {
        std::cout << "All failed:" << std::endl;
        for (const auto& failed_name : failed_tests) {
            std::cout << " - " << failed_name << std::endl;
        }
    }

    return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

