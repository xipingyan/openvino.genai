// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/qwen3_vl/classes.hpp"

namespace ov::genai {

// Qwen3.5 VLM is currently implemented as a thin wrapper over Qwen3-VL.
class VisionEncoderQwen3_5 : public VisionEncoderQwen3VL {
public:
    using VisionEncoderQwen3VL::VisionEncoderQwen3VL;
};

class InputsEmbedderQwen3_5 : public InputsEmbedderQwen3VL {
public:
    using InputsEmbedderQwen3VL::InputsEmbedderQwen3VL;
};

}  // namespace ov::genai
