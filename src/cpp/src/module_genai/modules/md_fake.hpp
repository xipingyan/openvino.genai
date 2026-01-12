// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"


#include "module_genai/module_base.hpp"
#include "module_genai/module_desc.hpp"
namespace ov::genai::module {

#define FAKE_MODULE_CONSTRUCTOR(FM_NAME, TM)                        \
    class FM_NAME : public ov::genai::module::IBaseModule {         \
        DeclareModuleConstructor(FM_NAME);                          \
                                                                    \
    public:                                                         \
        std::thread::id m_thread_id;                                \
    };                                                              \
    void FM_NAME::print_static_config() {}                          \
    FM_NAME::~FM_NAME() {}                                          \
    FM_NAME::run() {                                                \
        m_thread_id = std::this_thread::get_id();                   \
        std::this_thread::sleep_for(std::chrono::milliseconds(TM)); \
    }

FAKE_MODULE_CONSTRUCTOR(FakeModuleA, 2);
FAKE_MODULE_CONSTRUCTOR(FakeModuleB, 3);
FAKE_MODULE_CONSTRUCTOR(FakeModuleC, 4);
FAKE_MODULE_CONSTRUCTOR(FakeModuleD, 2);
}  // namespace ov::genai::module