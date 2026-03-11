// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni_processor.hpp"
#include "logger.hpp"

namespace ov::genai::module {

int64_t count_visual_tokens(const ov::Tensor& grid_thw, int32_t spatial_merge_size) {
    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto shape = grid_thw.get_shape();
    if (shape.size() != 2 || shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }
    const int64_t* grid = grid_thw.data<const int64_t>();
    int64_t total = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw values");
        }
        if (h % spatial_merge_size != 0 || w % spatial_merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by spatial_merge_size");
        }
        total += t * (h / spatial_merge_size) * (w / spatial_merge_size);
    }
    return total;
}

std::vector<std::string> build_prompts(const std::vector<std::string> &prompts,
                                       std::optional<std::vector<ov::Tensor>>& image_grid_thw, 
                                       std::optional<std::vector<ov::Tensor>>& audio_features,
                                       std::optional<std::vector<ov::Tensor>>& video_grid_thw,
                                       std::optional<std::vector<int>>& use_audio_in_video,
                                       std::optional<std::vector<int>>& video_second_per_grid,
                                       std::optional<int64_t> spatial_merge_size,
                                       std::optional<int64_t> position_id_per_seconds) {
    std::vector<std::string> result;
    for (const auto& prompt : prompts) {
        std::stringstream ss;
        ss << "<|im_start|>user\n";

        if (image_grid_thw.has_value() && !image_grid_thw->empty()) {
            if (!position_id_per_seconds.has_value()) {
                OPENVINO_THROW("position_id_per_seconds is required when image_grid_thw is provided");
            }
            for (const auto &grid_thw : image_grid_thw.value()) {
                int64_t image_token_num = count_visual_tokens(grid_thw, static_cast<int32_t>(spatial_merge_size.value()));
                if (image_token_num > 0) {
                    ss << "<|vision_start|>";
                    for (int64_t i = 0; i < image_token_num; ++i) ss << "<|image_pad|>";
                    ss << "<|vision_end|>";
                }
            }
        }

        // Counts audio features actually consumed in the video loop below.
        // Declared here so the standalone-audio section can use it as a starting
        // offset without re-counting flags (which would be wrong when audio was
        // requested but unavailable for a given video).
        size_t audio_idx = 0;

        if (video_grid_thw.has_value() && !video_grid_thw->empty()) {
            if (!position_id_per_seconds.has_value()) {
                OPENVINO_THROW("position_id_per_seconds is required when video_grid_thw is provided");
            }
            if (use_audio_in_video.has_value()) {
                if (use_audio_in_video->size() != video_grid_thw->size()) {
                    OPENVINO_THROW("use_audio_in_video size must match video_grid_thw size");
                }
            }
            if (video_second_per_grid.has_value()) {
                if (video_second_per_grid->size() != video_grid_thw->size()) {
                    OPENVINO_THROW("video_second_per_grid size must match video_grid_thw size");
                }
            } else {
                OPENVINO_THROW("video_second_per_grid is required when video_grid_thw is provided");
            }
            for (size_t i = 0; i < video_grid_thw->size(); i++) {
                const auto& vgrid = video_grid_thw.value()[i];

                bool do_audio_in_video = use_audio_in_video.has_value()
                                      && use_audio_in_video.value()[i]
                                      && audio_features.has_value()
                                      && audio_idx < audio_features->size();

                if (do_audio_in_video) {
                    const int64_t* gd = vgrid.data<const int64_t>();
                    const auto gs = vgrid.get_shape();
                    int64_t T, H, W;
                    if (gs.size() == 2 && gs[1] == 3) {
                        T = gd[0]; H = gd[1]; W = gd[2];
                    } else if (gs.size() == 1 && gs[0] == 3) {
                        T = gd[0]; H = gd[1]; W = gd[2];
                    } else {
                        OPENVINO_THROW("build_prompts: unexpected grid_thw shape for video ", i);
                    }

                    const int64_t merge = spatial_merge_size.value();
                    const int64_t H_m = H / merge;
                    const int64_t W_m = W / merge;
                    const float   spg = static_cast<float>((*video_second_per_grid)[i]);
                    const float   pip = static_cast<float>(position_id_per_seconds.value());

                    std::vector<float> video_pos;
                    video_pos.reserve(static_cast<size_t>(T * H_m * W_m));
                    for (int64_t t = 0; t < T; ++t) {
                        float pos = static_cast<float>(t) * spg * pip;
                        for (int64_t p = 0; p < H_m * W_m; ++p)
                            video_pos.push_back(pos);
                    }

                    const int64_t audio_len =
                        static_cast<int64_t>(audio_features->at(audio_idx).get_shape()[0]);

                    ss << "<|vision_start|><|audio_start|>";
                    size_t vi = 0;
                    int64_t ai = 0;
                    while (vi < video_pos.size() && ai < audio_len) {
                        if (video_pos[vi] <= static_cast<float>(ai)) {
                            ss << "<|video_pad|>";
                            ++vi;
                        } else {
                            ss << "<|audio_pad|>";
                            ++ai;
                        }
                    }
                    while (vi < video_pos.size()) { ss << "<|video_pad|>"; ++vi; }
                    while (ai < audio_len)        { ss << "<|audio_pad|>";  ++ai; }
                    ss << "<|audio_end|><|vision_end|>";

                    ++audio_idx;
                    continue;
                } else if (use_audio_in_video.has_value() && use_audio_in_video.value()[i]) {
                    // do_audio_in_video was false despite the flag being set:
                    // audio_features is absent or exhausted for this video.
                    GENAI_WARN("audio in video requested for video ", i, " but no audio feature available");
                }

                int64_t video_token_num = count_visual_tokens(vgrid, static_cast<int32_t>(spatial_merge_size.value()));
                if (video_token_num > 0) {
                    ss << "<|vision_start|>";
                    for (int64_t k = 0; k < video_token_num; ++k) ss << "<|video_pad|>";
                    ss << "<|vision_end|>";
                }
            }
        }

        if (audio_features.has_value() && !audio_features->empty()) {
            // audio_idx counts the audio features that were actually consumed inside
            // the video loop above (only incremented on the do_audio_in_video path,
            // not on the warning/fallback path).  Use it directly so that a video
            // whose audio was absent or too short does NOT advance the offset.
            for (size_t i = audio_idx; i < audio_features->size(); i++) {
                auto audio_token_num = static_cast<int64_t>(audio_features->at(i).get_shape()[0]);
                if (audio_token_num > 0) {
                    ss << "<|audio_start|>";
                    for (int64_t i = 0; i < audio_token_num; ++i) ss << "<|audio_pad|>";
                    ss << "<|audio_end|>";
                }
            }
        }
        ss << prompt;
        ss << "<|im_end|>\n<|im_start|>assistant\n";
        result.push_back(ss.str());
    }
    return result;
}

}