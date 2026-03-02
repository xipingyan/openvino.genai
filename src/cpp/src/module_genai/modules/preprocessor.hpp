// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <any>
#include <openvino/runtime/tensor.hpp>

namespace ov::genai::module {

class Preprocessor {
public:
    Preprocessor() = default;
    virtual ~Preprocessor() = default;

    virtual std::any preprocess(const ov::Tensor &images) = 0;
};

}