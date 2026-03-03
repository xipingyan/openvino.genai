// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_vl_video_processor.hpp"

#include "openvino/core/except.hpp"

namespace ov::genai::module {

Qwen3VLVideoProcessor::Qwen3VLVideoProcessor(const std::filesystem::path& model_path) {
    (void)model_path;
    do_resize = true;
}

void Qwen3VLVideoProcessor::sample_frames(VideoMetadata metadata, int num_frames, float fps) {
    (void)metadata;
    if (num_frames > 0) {
        this->num_frames = num_frames;
    }
    if (fps > 0.0f) {
        this->fps = fps;
    }
    do_sample_frames = true;
}

void Qwen3VLVideoProcessor::preprocess(const std::vector<ov::Tensor>& frames) {
    if (frames.empty()) {
        return;
    }

    for (const auto& frame : frames) {
        if (frame.get_element_type() != ov::element::u8) {
            OPENVINO_THROW("Qwen3VLVideoProcessor expects u8 frames");
        }

        const auto shape = frame.get_shape();
        if (shape.size() != 3 && shape.size() != 4) {
            OPENVINO_THROW("Qwen3VLVideoProcessor frame must have shape [H,W,C] or [1,H,W,C]");
        }
        const size_t channels = shape.back();
        if (channels != 3) {
            OPENVINO_THROW("Qwen3VLVideoProcessor expects 3-channel RGB frames");
        }
    }
}

}  // namespace ov::genai::module