// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <openvino/runtime/tensor.hpp>

namespace audio_utils {

ov::Tensor load_audio(const std::filesystem::path& audio_path);

void write_wav(const std::string& filename, const float* samples, size_t num_samples, int sample_rate);

}