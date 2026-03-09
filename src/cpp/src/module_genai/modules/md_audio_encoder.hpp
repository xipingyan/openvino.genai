// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

namespace ov::genai::module {

class AudioEncoderModule : public IBaseModule {
    DeclareModuleConstructor(AudioEncoderModule);

private:
    bool initialize();

    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_request_queue;
};

}
