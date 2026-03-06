// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_OPENVINO_NEW_ARCH
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include <filesystem>

namespace ov::genai::module {

modeling::models::Qwen3VLConfig get_qwen3_omni_vl_config(const std::filesystem::path &config_path);

}
#endif
