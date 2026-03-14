// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "json_utils.hpp"

namespace {

void read_config_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

void read_preprocess_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open preprocessor config: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

void parse_rope_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3VLRopeConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "mrope_interleaved", cfg.mrope_interleaved);
    read_json_param(data, "mrope_section", cfg.mrope_section);
    read_json_param(data, "rope_type", cfg.rope_type);
}

void parse_text_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3VLTextConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "vocab_size", cfg.vocab_size);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_hidden_layers", cfg.num_hidden_layers);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads", cfg.num_key_value_heads);
    read_json_param(data, "head_dim", cfg.head_dim);
    read_json_param(data, "max_position_embeddings", cfg.max_position_embeddings);
    read_json_param(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_param(data, "rope_theta", cfg.rope_theta);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "attention_bias", cfg.attention_bias);
    read_json_param(data, "attention_dropout", cfg.attention_dropout);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);
    read_json_param(data, "dtype", cfg.dtype);

    if (data.contains("rope_scaling")) {
        parse_rope_config(data.at("rope_scaling"), cfg.rope);
    }
    if (data.contains("rope_parameters")) {
        parse_rope_config(data.at("rope_parameters"), cfg.rope);
    }

    cfg.finalize();
}

void parse_vision_config(const nlohmann::json& data, ov::genai::modeling::models::Qwen3VLVisionConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "depth", cfg.depth);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_heads", cfg.num_heads);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "spatial_merge_size", cfg.spatial_merge_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "out_hidden_size", cfg.out_hidden_size);
    read_json_param(data, "num_position_embeddings", cfg.num_position_embeddings);
    read_json_param(data, "deepstack_visual_indexes", cfg.deepstack_visual_indexes);
    read_json_param(data, "initializer_range", cfg.initializer_range);

    cfg.finalize();
}

void validate_index_range(const std::vector<int32_t>& indexes,
                          int32_t upper_bound,
                          const std::string& name) {
    for (int32_t idx : indexes) {
        if (idx < 0 || idx >= upper_bound) {
            OPENVINO_THROW("Invalid ", name, " index: ", idx);
        }
    }
}

template <typename T>
bool mask_value_at(const ov::Tensor& mask, size_t index) {
    const T* data = mask.data<const T>();
    return data[index] != static_cast<T>(0);
}

bool mask_value(const ov::Tensor& mask, size_t index) {
    switch (mask.get_element_type()) {
        case ov::element::boolean:
            return mask_value_at<char>(mask, index);
        case ov::element::i32:
            return mask_value_at<int32_t>(mask, index);
        case ov::element::i64:
            return mask_value_at<int64_t>(mask, index);
        case ov::element::u8:
            return mask_value_at<uint8_t>(mask, index);
        default:
            OPENVINO_THROW("Unsupported attention_mask dtype");
    }
}

void set_bool(ov::Tensor& mask, size_t index, bool value) {
    auto* data = mask.data<char>();
    data[index] = value ? 1 : 0;
}

std::pair<size_t, size_t> smart_resize(size_t height,
                                       size_t width,
                                       size_t factor,
                                       size_t min_pixels,
                                       size_t max_pixels) {
    if (height < factor || width < factor) {
        OPENVINO_THROW("Height and width must be >= resize factor");
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        OPENVINO_THROW("Absolute aspect ratio must be smaller than 200");
    }

    auto round_to_factor = [factor](double value) {
        return static_cast<size_t>(std::round(value / static_cast<double>(factor)) * factor);
    };

    size_t h_bar = round_to_factor(static_cast<double>(height));
    size_t w_bar = round_to_factor(static_cast<double>(width));

    const double pixels = static_cast<double>(height) * static_cast<double>(width);
    if (static_cast<double>(h_bar) * static_cast<double>(w_bar) > static_cast<double>(max_pixels)) {
        double beta = std::sqrt(pixels / static_cast<double>(max_pixels));
        h_bar = std::max(factor, static_cast<size_t>(std::floor(height / beta / factor) * factor));
        w_bar = std::max(factor, static_cast<size_t>(std::floor(width / beta / factor) * factor));
    } else if (static_cast<double>(h_bar) * static_cast<double>(w_bar) < static_cast<double>(min_pixels)) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / pixels);
        h_bar = static_cast<size_t>(std::ceil(height * beta / factor) * factor);
        w_bar = static_cast<size_t>(std::ceil(width * beta / factor) * factor);
    }

    return {h_bar, w_bar};
}

ov::Tensor to_f32(const ov::Tensor& src) {
    if (src.get_element_type() == ov::element::f32) {
        return src;
    }
    ov::Tensor dst(ov::element::f32, src.get_shape());
    float* out = dst.data<float>();
    const size_t total = src.get_size();

    if (src.get_element_type() == ov::element::f16) {
        const auto* in = src.data<const ov::float16>();
        for (size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }
    if (src.get_element_type() == ov::element::bf16) {
        const auto* in = src.data<const ov::bfloat16>();
        for (size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }
    if (src.get_element_type() == ov::element::f64) {
        const auto* in = src.data<const double>();
        for (size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }

    OPENVINO_THROW("Unsupported pos_embed dtype for Qwen3VL preprocessing");
}

void resize_bilinear_to_chw(const uint8_t* src,
                            size_t src_h,
                            size_t src_w,
                            size_t channels,
                            bool nchw,
                            size_t dst_h,
                            size_t dst_w,
                            const std::array<float, 3>& mean,
                            const std::array<float, 3>& std,
                            std::vector<float>& dst_chw) {
    dst_chw.assign(channels * dst_h * dst_w, 0.0f);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);

    auto fetch = [&](size_t y, size_t x, size_t c) -> float {
        size_t idx = 0;
        if (nchw) {
            idx = (c * src_h + y) * src_w + x;
        } else {
            idx = (y * src_w + x) * channels + c;
        }
        return static_cast<float>(src[idx]);
    };

    for (size_t y = 0; y < dst_h; ++y) {
        float in_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(src_h - 1)));
        size_t y0 = static_cast<size_t>(std::floor(in_y));
        size_t y1 = std::min(y0 + 1, src_h - 1);
        float wy = in_y - static_cast<float>(y0);
        for (size_t x = 0; x < dst_w; ++x) {
            float in_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            in_x = std::max(0.0f, std::min(in_x, static_cast<float>(src_w - 1)));
            size_t x0 = static_cast<size_t>(std::floor(in_x));
            size_t x1 = std::min(x0 + 1, src_w - 1);
            float wx = in_x - static_cast<float>(x0);
            const float w00 = (1.0f - wy) * (1.0f - wx);
            const float w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx);
            const float w11 = wy * wx;

            for (size_t c = 0; c < channels; ++c) {
                float v = 0.0f;
                v += w00 * fetch(y0, x0, c);
                v += w01 * fetch(y0, x1, c);
                v += w10 * fetch(y1, x0, c);
                v += w11 * fetch(y1, x1, c);
                const float norm = (v / 255.0f - mean[c]) / std[c];
                const size_t out_idx = (c * dst_h + y) * dst_w + x;
                dst_chw[out_idx] = norm;
            }
        }
    }
}

struct PreparedImage {
    std::vector<float> data;
    size_t frames = 0;
    size_t height = 0;
    size_t width = 0;
    int64_t grid_t = 0;
    int64_t grid_h = 0;
    int64_t grid_w = 0;
};

ov::Tensor build_pos_embeds(const ov::Tensor& pos_embed_weight,
                            const ov::Tensor& grid_thw,
                            int32_t merge_size) {
    auto weight_f32 = to_f32(pos_embed_weight);
    const auto weight_shape = weight_f32.get_shape();
    if (weight_shape.size() != 2) {
        OPENVINO_THROW("pos_embed weight must be 2D");
    }
    const int64_t num_pos = static_cast<int64_t>(weight_shape[0]);
    const int64_t hidden = static_cast<int64_t>(weight_shape[1]);
    const int64_t num_grid = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(num_pos))));
    if (num_grid * num_grid != num_pos) {
        OPENVINO_THROW("pos_embed num_position_embeddings must be a square");
    }

    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto grid_shape = grid_thw.get_shape();
    if (grid_shape.size() != 2 || grid_shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }

    const int64_t* grid = grid_thw.data<const int64_t>();
    const size_t num_images = grid_shape[0];
    int64_t total_tokens = 0;
    for (size_t i = 0; i < num_images; ++i) {
        total_tokens += grid[i * 3 + 0] * grid[i * 3 + 1] * grid[i * 3 + 2];
    }

    ov::Tensor pos_embeds(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(hidden)});
    float* out = pos_embeds.data<float>();
    const float* weight = weight_f32.data<const float>();
    size_t offset = 0;

    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw for pos_embed interpolation");
        }
        if (h % merge_size != 0 || w % merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by merge_size");
        }

        std::vector<float> h_pos(static_cast<size_t>(h));
        std::vector<float> w_pos(static_cast<size_t>(w));
        if (h == 1) {
            h_pos[0] = 0.0f;
        } else {
            const float step = static_cast<float>(num_grid - 1) / static_cast<float>(h - 1);
            for (int64_t y = 0; y < h; ++y) {
                h_pos[static_cast<size_t>(y)] = static_cast<float>(y) * step;
            }
        }
        if (w == 1) {
            w_pos[0] = 0.0f;
        } else {
            const float step = static_cast<float>(num_grid - 1) / static_cast<float>(w - 1);
            for (int64_t x = 0; x < w; ++x) {
                w_pos[static_cast<size_t>(x)] = static_cast<float>(x) * step;
            }
        }

        std::vector<int64_t> h_floor(static_cast<size_t>(h));
        std::vector<int64_t> h_ceil(static_cast<size_t>(h));
        std::vector<float> h_lerp(static_cast<size_t>(h));
        for (int64_t y = 0; y < h; ++y) {
            float val = h_pos[static_cast<size_t>(y)];
            int64_t f = static_cast<int64_t>(std::floor(val));
            int64_t c = std::min(f + 1, num_grid - 1);
            h_floor[static_cast<size_t>(y)] = f;
            h_ceil[static_cast<size_t>(y)] = c;
            h_lerp[static_cast<size_t>(y)] = val - static_cast<float>(f);
        }

        std::vector<int64_t> w_floor(static_cast<size_t>(w));
        std::vector<int64_t> w_ceil(static_cast<size_t>(w));
        std::vector<float> w_lerp(static_cast<size_t>(w));
        for (int64_t x = 0; x < w; ++x) {
            float val = w_pos[static_cast<size_t>(x)];
            int64_t f = static_cast<int64_t>(std::floor(val));
            int64_t c = std::min(f + 1, num_grid - 1);
            w_floor[static_cast<size_t>(x)] = f;
            w_ceil[static_cast<size_t>(x)] = c;
            w_lerp[static_cast<size_t>(x)] = val - static_cast<float>(f);
        }

        std::vector<float> pos_hw(static_cast<size_t>(h * w * hidden));
        for (int64_t yy = 0; yy < h; ++yy) {
            const float dh = h_lerp[static_cast<size_t>(yy)];
            const int64_t y0 = h_floor[static_cast<size_t>(yy)];
            const int64_t y1 = h_ceil[static_cast<size_t>(yy)];
            const int64_t base_y0 = y0 * num_grid;
            const int64_t base_y1 = y1 * num_grid;
            for (int64_t xx = 0; xx < w; ++xx) {
                const float dw = w_lerp[static_cast<size_t>(xx)];
                const int64_t x0 = w_floor[static_cast<size_t>(xx)];
                const int64_t x1 = w_ceil[static_cast<size_t>(xx)];
                const float w00 = (1.0f - dh) * (1.0f - dw);
                const float w01 = (1.0f - dh) * dw;
                const float w10 = dh * (1.0f - dw);
                const float w11 = dh * dw;

                const int64_t idx00 = base_y0 + x0;
                const int64_t idx01 = base_y0 + x1;
                const int64_t idx10 = base_y1 + x0;
                const int64_t idx11 = base_y1 + x1;

                const size_t out_base = static_cast<size_t>((yy * w + xx) * hidden);
                const size_t w00_base = static_cast<size_t>(idx00 * hidden);
                const size_t w01_base = static_cast<size_t>(idx01 * hidden);
                const size_t w10_base = static_cast<size_t>(idx10 * hidden);
                const size_t w11_base = static_cast<size_t>(idx11 * hidden);

                for (int64_t hidx = 0; hidx < hidden; ++hidx) {
                    float value = 0.0f;
                    value += w00 * weight[w00_base + static_cast<size_t>(hidx)];
                    value += w01 * weight[w01_base + static_cast<size_t>(hidx)];
                    value += w10 * weight[w10_base + static_cast<size_t>(hidx)];
                    value += w11 * weight[w11_base + static_cast<size_t>(hidx)];
                    pos_hw[out_base + static_cast<size_t>(hidx)] = value;
                }
            }
        }

        const int64_t merged_h = h / merge_size;
        const int64_t merged_w = w / merge_size;
        for (int64_t tt = 0; tt < t; ++tt) {
            (void)tt;
            for (int64_t bh = 0; bh < merged_h; ++bh) {
                for (int64_t bw = 0; bw < merged_w; ++bw) {
                    for (int64_t mh = 0; mh < merge_size; ++mh) {
                        for (int64_t mw = 0; mw < merge_size; ++mw) {
                            const int64_t h_idx = bh * merge_size + mh;
                            const int64_t w_idx = bw * merge_size + mw;
                            const size_t src_base = static_cast<size_t>((h_idx * w + w_idx) * hidden);
                            const size_t dst_base = offset * static_cast<size_t>(hidden);
                            std::copy_n(pos_hw.data() + src_base, static_cast<size_t>(hidden), out + dst_base);
                            offset++;
                        }
                    }
                }
            }
        }
    }

    return pos_embeds;
}

std::pair<ov::Tensor, ov::Tensor> build_rotary_cos_sin(
    const ov::Tensor& grid_thw,
    const ov::genai::modeling::models::Qwen3VLVisionConfig& cfg,
    int32_t merge_size) {
    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto grid_shape = grid_thw.get_shape();
    if (grid_shape.size() != 2 || grid_shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }
    const int64_t* grid = grid_thw.data<const int64_t>();
    const size_t num_images = grid_shape[0];

    int64_t total_tokens = 0;
    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        total_tokens += t * h * w;
    }

    const int32_t head_dim = cfg.head_dim();
    if (head_dim <= 0 || head_dim % 2 != 0) {
        OPENVINO_THROW("Invalid head_dim for Qwen3VL rotary embedding");
    }
    const int32_t rotary_dim = head_dim / 2;
    if (rotary_dim % 2 != 0) {
        OPENVINO_THROW("Vision rotary_dim must be even");
    }
    const int32_t inv_len = rotary_dim / 2;
    const float theta = 10000.0f;

    std::vector<float> inv_freq(static_cast<size_t>(inv_len));
    for (int32_t i = 0; i < inv_len; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(theta, exponent);
    }

    ov::Tensor rotary_cos(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(head_dim)});
    ov::Tensor rotary_sin(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(head_dim)});
    float* cos_out = rotary_cos.data<float>();
    float* sin_out = rotary_sin.data<float>();

    size_t offset = 0;
    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw for rotary embedding");
        }
        if (h % merge_size != 0 || w % merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by merge_size");
        }
        const int64_t merged_h = h / merge_size;
        const int64_t merged_w = w / merge_size;

        for (int64_t tt = 0; tt < t; ++tt) {
            (void)tt;
            for (int64_t bh = 0; bh < merged_h; ++bh) {
                for (int64_t bw = 0; bw < merged_w; ++bw) {
                    for (int64_t mh = 0; mh < merge_size; ++mh) {
                        for (int64_t mw = 0; mw < merge_size; ++mw) {
                            const int64_t row = bh * merge_size + mh;
                            const int64_t col = bw * merge_size + mw;
                            float* cos_ptr = cos_out + offset * static_cast<size_t>(head_dim);
                            float* sin_ptr = sin_out + offset * static_cast<size_t>(head_dim);

                            for (int32_t j = 0; j < inv_len; ++j) {
                                const float inv = inv_freq[static_cast<size_t>(j)];
                                const float row_freq = static_cast<float>(row) * inv;
                                const float col_freq = static_cast<float>(col) * inv;
                                const float cos_row = std::cos(row_freq);
                                const float sin_row = std::sin(row_freq);
                                const float cos_col = std::cos(col_freq);
                                const float sin_col = std::sin(col_freq);

                                const int32_t row_idx = j;
                                const int32_t col_idx = inv_len + j;

                                cos_ptr[row_idx] = cos_row;
                                sin_ptr[row_idx] = sin_row;
                                cos_ptr[col_idx] = cos_col;
                                sin_ptr[col_idx] = sin_col;
                            }
                            for (int32_t j = 0; j < rotary_dim; ++j) {
                                cos_ptr[rotary_dim + j] = cos_ptr[j];
                                sin_ptr[rotary_dim + j] = sin_ptr[j];
                            }

                            offset++;
                        }
                    }
                }
            }
        }
    }

    return {rotary_cos, rotary_sin};
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t Qwen3VLTextConfig::kv_heads() const {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
}

int32_t Qwen3VLTextConfig::resolved_head_dim() const {
    if (head_dim > 0) {
        return head_dim;
    }
    if (hidden_size > 0 && num_attention_heads > 0) {
        return hidden_size / num_attention_heads;
    }
    return 0;
}

void Qwen3VLTextConfig::finalize() {
    if (num_key_value_heads <= 0) {
        num_key_value_heads = num_attention_heads;
    }
    if (head_dim <= 0) {
        head_dim = resolved_head_dim();
    }
    if (rope.mrope_section.empty()) {
        rope.mrope_section = {24, 20, 20};
    }
}

void Qwen3VLTextConfig::validate() const {
    if (hidden_size <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.hidden_size must be > 0");
    }
    if (num_hidden_layers <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_hidden_layers must be > 0");
    }
    if (num_attention_heads <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_attention_heads must be > 0");
    }
    if (kv_heads() <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_key_value_heads must be > 0");
    }
    if (num_attention_heads % kv_heads() != 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_attention_heads must be divisible by num_key_value_heads");
    }
    if (resolved_head_dim() <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.head_dim must be > 0");
    }
    if (hidden_size % num_attention_heads != 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.hidden_size must be divisible by num_attention_heads");
    }
    if (rope.mrope_interleaved && rope.mrope_section.size() != 3) {
        OPENVINO_THROW("Qwen3VLTextConfig.mrope_section must have 3 elements");
    }
}

int32_t Qwen3VLVisionConfig::head_dim() const {
    if (num_heads <= 0) {
        return 0;
    }
    return hidden_size / num_heads;
}

void Qwen3VLVisionConfig::finalize() {
    if (out_hidden_size <= 0) {
        out_hidden_size = hidden_size;
    }
}

void Qwen3VLVisionConfig::validate() const {
    if (depth <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.depth must be > 0");
    }
    if (hidden_size <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.hidden_size must be > 0");
    }
    if (num_heads <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.num_heads must be > 0");
    }
    if (hidden_size % num_heads != 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.hidden_size must be divisible by num_heads");
    }
    if (patch_size <= 0 || spatial_merge_size <= 0 || temporal_patch_size <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig patch/merge sizes must be > 0");
    }
    if (out_hidden_size <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.out_hidden_size must be > 0");
    }
    if (num_position_embeddings <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.num_position_embeddings must be > 0");
    }
    if (!deepstack_visual_indexes.empty()) {
        validate_index_range(deepstack_visual_indexes, depth, "deepstack_visual_indexes");
    }
}

void Qwen3VLConfig::finalize() {
    if (model_type.empty()) {
        model_type = "qwen3_vl";
    }
    text.finalize();
    vision.finalize();
    if (text.tie_word_embeddings) {
        tie_word_embeddings = true;
    }
    if (tie_word_embeddings) {
        text.tie_word_embeddings = true;
    }
}

void Qwen3VLConfig::validate() const {
    if (model_type != "qwen3_vl") {
        OPENVINO_THROW("Unsupported model_type: ", model_type);
    }
    text.validate();
    vision.validate();
    if (image_token_id < 0 || vision_start_token_id < 0 || vision_end_token_id < 0) {
        OPENVINO_THROW("Invalid token ids in Qwen3VLConfig");
    }
}

Qwen3VLConfig Qwen3VLConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    Qwen3VLConfig cfg;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "image_token_id", cfg.image_token_id);
    read_json_param(data, "video_token_id", cfg.video_token_id);
    read_json_param(data, "vision_start_token_id", cfg.vision_start_token_id);
    read_json_param(data, "vision_end_token_id", cfg.vision_end_token_id);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);

    if (data.contains("text_config")) {
        parse_text_config(data.at("text_config"), cfg.text);
    } else {
        OPENVINO_THROW("Qwen3VLConfig is missing text_config");
    }

    if (data.contains("vision_config")) {
        parse_vision_config(data.at("vision_config"), cfg.vision);
    } else {
        OPENVINO_THROW("Qwen3VLConfig is missing vision_config");
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

Qwen3VLConfig Qwen3VLConfig::from_json_file(const std::filesystem::path& config_path) {
    auto resolved = resolve_config_path(config_path);
    if (!std::filesystem::exists(resolved)) {
        OPENVINO_THROW("Config file not found: ", resolved.string());
    }
    nlohmann::json data;
    read_config_json_file(resolved, data);
    return from_json(data);
}

std::string Qwen3VLModuleNames::vision_block(int32_t index) {
    return std::string("blocks.") + std::to_string(index);
}

std::string Qwen3VLModuleNames::deepstack_merger(int32_t index) {
    return std::string("deepstack_merger_list.") + std::to_string(index);
}

std::string Qwen3VLModuleNames::text_layer(int32_t index) {
    return std::string("layers.") + std::to_string(index);
}

std::vector<std::string> Qwen3VLGraphSpec::vision_required_inputs(bool use_external_pos_embeds) {
    std::vector<std::string> inputs = {
        Qwen3VLVisionIO::kPixelValues,
        Qwen3VLVisionIO::kGridThw,
    };
    if (use_external_pos_embeds) {
        inputs.emplace_back(Qwen3VLVisionIO::kPosEmbeds);
        inputs.emplace_back(Qwen3VLVisionIO::kRotaryCos);
        inputs.emplace_back(Qwen3VLVisionIO::kRotarySin);
    }
    return inputs;
}

std::vector<std::string> Qwen3VLGraphSpec::vision_outputs(const Qwen3VLVisionConfig& cfg) {
    std::vector<std::string> outputs = {Qwen3VLVisionIO::kVisualEmbeds};
    for (size_t i = 0; i < cfg.deepstack_visual_indexes.size(); ++i) {
        outputs.push_back(std::string(Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i));
    }
    return outputs;
}

std::vector<std::string> Qwen3VLGraphSpec::text_required_inputs(bool use_inputs_embeds) {
    std::vector<std::string> inputs = {
        Qwen3VLTextIO::kAttentionMask,
        Qwen3VLTextIO::kPositionIds,
        Qwen3VLTextIO::kBeamIdx,
    };
    if (use_inputs_embeds) {
        inputs.emplace_back(Qwen3VLTextIO::kInputsEmbeds);
    } else {
        inputs.emplace_back(Qwen3VLTextIO::kInputIds);
    }
    return inputs;
}

std::vector<std::string> Qwen3VLGraphSpec::text_optional_inputs() {
    return {
        Qwen3VLTextIO::kInputsEmbeds,
        Qwen3VLTextIO::kVisualEmbeds,
        Qwen3VLTextIO::kVisualPosMask,
        Qwen3VLTextIO::kDeepstackEmbedsPrefix,
    };
}

Qwen3VLInputPlanner::Qwen3VLInputPlanner(const Qwen3VLConfig& cfg)
    : image_token_id_(cfg.image_token_id),
      vision_start_token_id_(cfg.vision_start_token_id),
      spatial_merge_size_(cfg.vision.spatial_merge_size) {}

ov::Tensor Qwen3VLInputPlanner::build_visual_pos_mask(const ov::Tensor& input_ids,
                                                      const ov::Tensor* attention_mask) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for Qwen3VLInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    ov::Tensor mask(ov::element::boolean, shape);
    const int64_t* ids = input_ids.data<const int64_t>();
    const size_t total = input_ids.get_size();

    for (size_t idx = 0; idx < total; ++idx) {
        bool active = ids[idx] == image_token_id_;
        if (attention_mask && !mask_value(*attention_mask, idx)) {
            active = false;
        }
        set_bool(mask, idx, active);
    }
    return mask;
}

Qwen3VLInputPlan Qwen3VLInputPlanner::build_plan(const ov::Tensor& input_ids,
                                                 const ov::Tensor* attention_mask,
                                                 const ov::Tensor* image_grid_thw) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for Qwen3VLInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    if (image_grid_thw) {
        const auto grid_shape = image_grid_thw->get_shape();
        if (image_grid_thw->get_element_type() != ov::element::i64) {
            OPENVINO_THROW("image_grid_thw must be i64");
        }
        if (grid_shape.size() != 2 || grid_shape[1] != 3) {
            OPENVINO_THROW("image_grid_thw must have shape [N, 3]");
        }
    }
    if (spatial_merge_size_ <= 0) {
        OPENVINO_THROW("spatial_merge_size must be > 0");
    }
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];

    ov::Tensor position_ids(ov::element::i64, {3, batch, seq_len});
    std::memset(position_ids.data(), 0, position_ids.get_byte_size());

    ov::Tensor rope_deltas(ov::element::i64, {batch, 1});
    std::memset(rope_deltas.data(), 0, rope_deltas.get_byte_size());

    auto visual_pos_mask = build_visual_pos_mask(input_ids, attention_mask);

    const int64_t* ids = input_ids.data<const int64_t>();
    const int64_t* grid = image_grid_thw ? image_grid_thw->data<const int64_t>() : nullptr;
    const size_t grid_rows = image_grid_thw ? image_grid_thw->get_shape().at(0) : 0;
    size_t grid_index = 0;

    auto pos_data = position_ids.data<int64_t>();
    auto delta_data = rope_deltas.data<int64_t>();

    for (size_t b = 0; b < batch; ++b) {
        std::vector<int64_t> tokens;
        std::vector<size_t> active_indices;
        tokens.reserve(seq_len);
        active_indices.reserve(seq_len);

        for (size_t s = 0; s < seq_len; ++s) {
            const size_t idx = b * seq_len + s;
            if (attention_mask && !mask_value(*attention_mask, idx)) {
                continue;
            }
            tokens.push_back(ids[idx]);
            active_indices.push_back(s);
        }

        if (tokens.empty()) {
            delta_data[b] = 0;
            continue;
        }

        std::vector<int64_t> pos_t;
        std::vector<int64_t> pos_h;
        std::vector<int64_t> pos_w;
        pos_t.reserve(tokens.size());
        pos_h.reserve(tokens.size());
        pos_w.reserve(tokens.size());

        int64_t last_max = -1;
        size_t st = 0;
        size_t local_grid_index = grid_index;

        auto append_text = [&](size_t length) {
            if (length == 0) {
                return;
            }
            const int64_t base = last_max + 1;
            for (size_t i = 0; i < length; ++i) {
                const int64_t value = base + static_cast<int64_t>(i);
                pos_t.push_back(value);
                pos_h.push_back(value);
                pos_w.push_back(value);
            }
            last_max = base + static_cast<int64_t>(length) - 1;
        };

        auto append_visual = [&](int64_t t, int64_t h, int64_t w) {
            if (t <= 0 || h <= 0 || w <= 0) {
                OPENVINO_THROW("Invalid grid_thw values in Qwen3VLInputPlanner");
            }
            const int64_t llm_grid_t = t;
            const int64_t llm_grid_h = h / spatial_merge_size_;
            const int64_t llm_grid_w = w / spatial_merge_size_;
            if (llm_grid_h <= 0 || llm_grid_w <= 0) {
                OPENVINO_THROW("Invalid spatial_merge_size for grid_thw");
            }
            const int64_t base = last_max + 1;
            int64_t max_dim = 0;
            for (int64_t tt = 0; tt < llm_grid_t; ++tt) {
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        pos_t.push_back(base + tt);
                        pos_h.push_back(base + hh);
                        pos_w.push_back(base + ww);
                        max_dim = std::max(max_dim, std::max(tt, std::max(hh, ww)));
                    }
                }
            }
            last_max = base + max_dim;
        };

        if (image_grid_thw) {
            while (true) {
                auto start_it = tokens.begin() + static_cast<std::vector<int64_t>::difference_type>(st);
                auto it = std::find(start_it, tokens.end(), image_token_id_);
                if (it == tokens.end()) {
                    break;
                }
                const size_t ed = static_cast<size_t>(std::distance(tokens.begin(), it));
                if (local_grid_index >= grid_rows) {
                    OPENVINO_THROW("image_grid_thw entries are fewer than image tokens");
                }
                append_text(ed - st);
                const int64_t t = grid[local_grid_index * 3 + 0];
                const int64_t h = grid[local_grid_index * 3 + 1];
                const int64_t w = grid[local_grid_index * 3 + 2];
                append_visual(t, h, w);

                const int64_t llm_grid_h = h / spatial_merge_size_;
                const int64_t llm_grid_w = w / spatial_merge_size_;
                const int64_t visual_len = t * llm_grid_h * llm_grid_w;
                if (ed + static_cast<size_t>(visual_len) > tokens.size()) {
                    OPENVINO_THROW("Image tokens length does not match grid_thw");
                }
                st = ed + static_cast<size_t>(visual_len);
                local_grid_index += 1;
            }
        }

        if (st < tokens.size()) {
            append_text(tokens.size() - st);
        }

        if (pos_t.size() != tokens.size()) {
            OPENVINO_THROW("Position ids length mismatch");
        }

        int64_t max_pos = pos_t.empty() ? 0 : pos_t.front();
        for (size_t i = 0; i < pos_t.size(); ++i) {
            max_pos = std::max(max_pos, pos_t[i]);
            max_pos = std::max(max_pos, pos_h[i]);
            max_pos = std::max(max_pos, pos_w[i]);
        }

        for (size_t i = 0; i < tokens.size(); ++i) {
            const size_t s = active_indices[i];
            const size_t base = b * seq_len + s;
            pos_data[0 * batch * seq_len + base] = pos_t[i];
            pos_data[1 * batch * seq_len + base] = pos_h[i];
            pos_data[2 * batch * seq_len + base] = pos_w[i];
        }

        if (attention_mask) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t idx = b * seq_len + s;
                if (mask_value(*attention_mask, idx)) {
                    continue;
                }
                pos_data[0 * batch * seq_len + idx] = 1;
                pos_data[1 * batch * seq_len + idx] = 1;
                pos_data[2 * batch * seq_len + idx] = 1;
            }
        }

        delta_data[b] = max_pos + 1 - static_cast<int64_t>(seq_len);
        grid_index = local_grid_index;
    }

    return {position_ids, visual_pos_mask, rope_deltas};
}

ov::Tensor Qwen3VLInputPlanner::scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                                      const ov::Tensor& visual_pos_mask) {
    const auto mask_shape = visual_pos_mask.get_shape();
    if (mask_shape.size() != 2) {
        OPENVINO_THROW("visual_pos_mask must have shape [B, S]");
    }
    const auto embeds_shape = visual_embeds.get_shape();
    if (embeds_shape.size() != 2) {
        OPENVINO_THROW("visual_embeds must have shape [V, H]");
    }
    const size_t batch = mask_shape[0];
    const size_t seq_len = mask_shape[1];
    const size_t hidden = embeds_shape[1];

    ov::Tensor out(visual_embeds.get_element_type(), {batch, seq_len, hidden});
    std::memset(out.data(), 0, out.get_byte_size());

    const size_t elem_size = visual_embeds.get_element_type().size();
    const size_t row_bytes = hidden * elem_size;

    const char* src = static_cast<const char*>(visual_embeds.data());
    char* dst = static_cast<char*>(out.data());

    size_t visual_idx = 0;
    const size_t total = batch * seq_len;
    for (size_t idx = 0; idx < total; ++idx) {
        if (!mask_value(visual_pos_mask, idx)) {
            continue;
        }
        if (visual_idx >= embeds_shape[0]) {
            OPENVINO_THROW("visual_embeds shorter than visual_pos_mask");
        }
        std::memcpy(dst + idx * row_bytes, src + visual_idx * row_bytes, row_bytes);
        visual_idx++;
    }
    if (visual_idx != embeds_shape[0]) {
        OPENVINO_THROW("visual_embeds length does not match visual_pos_mask");
    }
    return out;
}

std::vector<ov::Tensor> Qwen3VLInputPlanner::scatter_deepstack_embeds(
    const std::vector<ov::Tensor>& deepstack_embeds,
    const ov::Tensor& visual_pos_mask) {
    std::vector<ov::Tensor> out;
    out.reserve(deepstack_embeds.size());
    for (const auto& embed : deepstack_embeds) {
        out.push_back(scatter_visual_embeds(embed, visual_pos_mask));
    }
    return out;
}

ov::Tensor Qwen3VLInputPlanner::build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                          int64_t past_length,
                                                          int64_t seq_len) {
    if (rope_deltas.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("rope_deltas must be i64");
    }
    if (past_length < 0 || seq_len <= 0) {
        OPENVINO_THROW("Invalid past_length or seq_len for decode position ids");
    }
    const auto shape = rope_deltas.get_shape();
    size_t batch = 0;
    if (shape.size() == 1) {
        batch = shape[0];
    } else if (shape.size() == 2) {
        if (shape[1] != 1) {
            OPENVINO_THROW("rope_deltas must have shape [B] or [B, 1]");
        }
        batch = shape[0];
    } else {
        OPENVINO_THROW("rope_deltas must have shape [B] or [B, 1]");
    }

    ov::Tensor position_ids(ov::element::i64, {3, batch, static_cast<size_t>(seq_len)});
    auto* out = position_ids.data<int64_t>();
    const int64_t* deltas = rope_deltas.data<const int64_t>();
    const size_t plane_stride = batch * static_cast<size_t>(seq_len);

    for (size_t b = 0; b < batch; ++b) {
        const int64_t base = past_length + deltas[b];
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t value = base + s;
            const size_t idx = b * static_cast<size_t>(seq_len) + static_cast<size_t>(s);
            out[idx] = value;
            out[plane_stride + idx] = value;
            out[2 * plane_stride + idx] = value;
        }
    }

    return position_ids;
}

Qwen3VLVisionPreprocessConfig Qwen3VLVisionPreprocessConfig::from_json_file(
    const std::filesystem::path& path) {
    nlohmann::json data;
    read_preprocess_json_file(path, data);
    Qwen3VLVisionPreprocessConfig cfg;
    using ov::genai::utils::read_json_param;
    read_json_param(data, "size.shortest_edge", cfg.min_pixels);
    read_json_param(data, "size.longest_edge", cfg.max_pixels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "merge_size", cfg.merge_size);
    read_json_param(data, "image_mean", cfg.image_mean);
    read_json_param(data, "image_std", cfg.image_std);
    return cfg;
}

Qwen3VLVisionPreprocessor::Qwen3VLVisionPreprocessor(
    const Qwen3VLVisionConfig& vision_cfg,
    const Qwen3VLVisionPreprocessConfig& preprocess_cfg)
    : vision_cfg_(vision_cfg),
      preprocess_cfg_(preprocess_cfg) {
    if (vision_cfg_.patch_size != preprocess_cfg_.patch_size) {
        OPENVINO_THROW("patch_size mismatch between vision config and preprocessor config");
    }
    if (vision_cfg_.temporal_patch_size != preprocess_cfg_.temporal_patch_size) {
        OPENVINO_THROW("temporal_patch_size mismatch between vision config and preprocessor config");
    }
    if (vision_cfg_.spatial_merge_size != preprocess_cfg_.merge_size) {
        OPENVINO_THROW("merge_size mismatch between vision config and preprocessor config");
    }
}

Qwen3VLVisionInputs Qwen3VLVisionPreprocessor::preprocess(const ov::Tensor& images,
                                                          const ov::Tensor& pos_embed_weight) const {
    const auto img_shape = images.get_shape();
    if (img_shape.size() != 3 && img_shape.size() != 4) {
        OPENVINO_THROW("images must have shape [H, W, C] or [B, H, W, C]");
    }
    if (images.get_element_type() != ov::element::u8) {
        OPENVINO_THROW("images must be u8 for Qwen3VL preprocessing");
    }

    const bool has_batch = img_shape.size() == 4;
    const size_t batch = has_batch ? img_shape[0] : 1;
    const size_t in_h = has_batch ? img_shape[1] : img_shape[0];
    const size_t in_w = has_batch ? img_shape[2] : img_shape[1];
    const size_t channels = has_batch ? img_shape[3] : img_shape[2];
    if (channels != 3) {
        OPENVINO_THROW("images must have 3 channels");
    }

    const size_t factor = static_cast<size_t>(preprocess_cfg_.patch_size * preprocess_cfg_.merge_size);
    const uint8_t* src = images.data<const uint8_t>();
    const bool nchw = false;

    std::vector<PreparedImage> prepared;
    prepared.reserve(batch);

    for (size_t b = 0; b < batch; ++b) {
        const uint8_t* src_img = src + b * in_h * in_w * channels;
        size_t out_h = in_h;
        size_t out_w = in_w;
        if (preprocess_cfg_.do_resize) {
            auto resized = smart_resize(in_h,
                                        in_w,
                                        factor,
                                        static_cast<size_t>(preprocess_cfg_.min_pixels),
                                        static_cast<size_t>(preprocess_cfg_.max_pixels));
            out_h = resized.first;
            out_w = resized.second;
        }
        if (out_h % preprocess_cfg_.patch_size != 0 || out_w % preprocess_cfg_.patch_size != 0) {
            OPENVINO_THROW("Resized image must be divisible by patch_size");
        }

        std::vector<float> frame;
        resize_bilinear_to_chw(src_img,
                               in_h,
                               in_w,
                               channels,
                               nchw,
                               out_h,
                               out_w,
                               preprocess_cfg_.image_mean,
                               preprocess_cfg_.image_std,
                               frame);

        const size_t frames = 1;
        size_t padded_frames = frames;
        if (frames % static_cast<size_t>(preprocess_cfg_.temporal_patch_size) != 0) {
            padded_frames += static_cast<size_t>(preprocess_cfg_.temporal_patch_size) -
                             (frames % static_cast<size_t>(preprocess_cfg_.temporal_patch_size));
        }

        std::vector<float> stacked(padded_frames * frame.size());
        for (size_t t = 0; t < padded_frames; ++t) {
            const size_t dst_offset = t * frame.size();
            std::copy(frame.begin(), frame.end(), stacked.begin() + dst_offset);
        }

        PreparedImage item;
        item.data = std::move(stacked);
        item.frames = padded_frames;
        item.height = out_h;
        item.width = out_w;
        item.grid_t = static_cast<int64_t>(padded_frames / static_cast<size_t>(preprocess_cfg_.temporal_patch_size));
        item.grid_h = static_cast<int64_t>(out_h / static_cast<size_t>(preprocess_cfg_.patch_size));
        item.grid_w = static_cast<int64_t>(out_w / static_cast<size_t>(preprocess_cfg_.patch_size));
        prepared.push_back(std::move(item));
    }

    int64_t total_patches = 0;
    for (const auto& item : prepared) {
        total_patches += item.grid_t * item.grid_h * item.grid_w;
    }

    ov::Tensor grid_thw(ov::element::i64, {batch, 3});
    auto* grid = grid_thw.data<int64_t>();
    for (size_t b = 0; b < prepared.size(); ++b) {
        grid[b * 3 + 0] = prepared[b].grid_t;
        grid[b * 3 + 1] = prepared[b].grid_h;
        grid[b * 3 + 2] = prepared[b].grid_w;
    }

    const size_t patch_size = static_cast<size_t>(preprocess_cfg_.patch_size);
    const size_t temporal_patch = static_cast<size_t>(preprocess_cfg_.temporal_patch_size);
    const size_t merge_size = static_cast<size_t>(preprocess_cfg_.merge_size);
    const size_t patch_stride = channels * temporal_patch * patch_size * patch_size;

    ov::Tensor pixel_values(ov::element::f32,
                            {static_cast<size_t>(total_patches),
                             channels,
                             temporal_patch,
                             patch_size,
                             patch_size});
    float* out = pixel_values.data<float>();
    size_t patch_offset = 0;

    for (const auto& item : prepared) {
        const size_t height = item.height;
        const size_t width = item.width;
        const size_t frame_stride = channels * height * width;
        const float* data = item.data.data();
        const size_t grid_t = static_cast<size_t>(item.grid_t);
        const size_t grid_h = static_cast<size_t>(item.grid_h);
        const size_t grid_w = static_cast<size_t>(item.grid_w);
        if (grid_h % merge_size != 0 || grid_w % merge_size != 0) {
            OPENVINO_THROW("grid_h/grid_w must be divisible by merge_size");
        }
        const size_t merged_h = grid_h / merge_size;
        const size_t merged_w = grid_w / merge_size;

        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t bh = 0; bh < merged_h; ++bh) {
                for (size_t bw = 0; bw < merged_w; ++bw) {
                    for (size_t mh = 0; mh < merge_size; ++mh) {
                        for (size_t mw = 0; mw < merge_size; ++mw) {
                            float* dst = out + patch_offset * patch_stride;
                            size_t dst_idx = 0;
                            const size_t h_idx = (bh * merge_size + mh) * patch_size;
                            const size_t w_idx = (bw * merge_size + mw) * patch_size;
                            for (size_t c = 0; c < channels; ++c) {
                                for (size_t tp = 0; tp < temporal_patch; ++tp) {
                                    const size_t t_idx = (t * temporal_patch + tp) * frame_stride;
                                    for (size_t ph = 0; ph < patch_size; ++ph) {
                                        for (size_t pw = 0; pw < patch_size; ++pw) {
                                            const size_t src_idx =
                                                t_idx + (c * height + h_idx + ph) * width + w_idx + pw;
                                            dst[dst_idx++] = data[src_idx];
                                        }
                                    }
                                }
                            }
                            patch_offset++;
                        }
                    }
                }
            }
        }
    }

    auto pos_embeds = build_pos_embeds(pos_embed_weight, grid_thw, preprocess_cfg_.merge_size);
    auto rotary = build_rotary_cos_sin(grid_thw, vision_cfg_, preprocess_cfg_.merge_size);

    return {pixel_values, grid_thw, pos_embeds, rotary.first, rotary.second};
}

Qwen3VLVisionInputs Qwen3VLVisionPreprocessor::preprocess_video(
    const std::vector<ov::Tensor>& frames,
    const ov::Tensor& pos_embed_weight,
    size_t video_min_pixels,
    size_t video_max_pixels) const {
    if (frames.empty()) {
        OPENVINO_THROW("preprocess_video: frames must not be empty");
    }

    // All frames must be [1, H, W, 3] or [H, W, 3] u8
    size_t in_h = 0, in_w = 0;
    for (size_t f = 0; f < frames.size(); ++f) {
        const auto s = frames[f].get_shape();
        if (frames[f].get_element_type() != ov::element::u8) {
            OPENVINO_THROW("preprocess_video: frame must be u8");
        }
        size_t fh, fw, fc;
        if (s.size() == 3) {
            fh = s[0]; fw = s[1]; fc = s[2];
        } else if (s.size() == 4 && s[0] == 1) {
            fh = s[1]; fw = s[2]; fc = s[3];
        } else {
            OPENVINO_THROW("preprocess_video: frame shape must be [H,W,3] or [1,H,W,3]");
        }
        if (fc != 3) {
            OPENVINO_THROW("preprocess_video: frame must have 3 channels");
        }
        if (f == 0) { in_h = fh; in_w = fw; }
        // Use first frame dimensions for smart_resize; all frames are resized to the same size
    }

    const size_t factor = static_cast<size_t>(preprocess_cfg_.patch_size * preprocess_cfg_.merge_size);
    const size_t channels = 3;
    size_t out_h = in_h, out_w = in_w;
    if (preprocess_cfg_.do_resize) {
        auto resized = smart_resize(in_h, in_w, factor, video_min_pixels, video_max_pixels);
        out_h = resized.first;
        out_w = resized.second;
    }

    const size_t nframes = frames.size();
    const size_t temporal_patch = static_cast<size_t>(preprocess_cfg_.temporal_patch_size);
    size_t padded_nframes = nframes;
    if (nframes % temporal_patch != 0) {
        padded_nframes += temporal_patch - (nframes % temporal_patch);
    }

    // Resize each frame to CHW float, then stack temporally
    const size_t frame_chw_size = channels * out_h * out_w;
    std::vector<float> stacked(padded_nframes * frame_chw_size, 0.0f);

    for (size_t f = 0; f < padded_nframes; ++f) {
        // For padding frames, duplicate the last real frame
        const size_t src_idx = (f < nframes) ? f : (nframes - 1);
        const auto& frm = frames[src_idx];
        const auto s = frm.get_shape();
        size_t fh = (s.size() == 4) ? s[1] : s[0];
        size_t fw = (s.size() == 4) ? s[2] : s[1];
        const uint8_t* src = frm.data<const uint8_t>();

        std::vector<float> frame_chw;
        resize_bilinear_to_chw(src, fh, fw, channels, /*nchw=*/false,
                               out_h, out_w,
                               preprocess_cfg_.image_mean,
                               preprocess_cfg_.image_std,
                               frame_chw);
        std::copy(frame_chw.begin(), frame_chw.end(),
                  stacked.begin() + static_cast<std::ptrdiff_t>(f * frame_chw_size));
    }

    // Build grid_thw: single video entity
    const int64_t grid_t = static_cast<int64_t>(padded_nframes / temporal_patch);
    const int64_t grid_h = static_cast<int64_t>(out_h / static_cast<size_t>(preprocess_cfg_.patch_size));
    const int64_t grid_w = static_cast<int64_t>(out_w / static_cast<size_t>(preprocess_cfg_.patch_size));

    ov::Tensor grid_thw(ov::element::i64, {1, 3});
    auto* grid = grid_thw.data<int64_t>();
    grid[0] = grid_t;
    grid[1] = grid_h;
    grid[2] = grid_w;

    const int64_t total_patches = grid_t * grid_h * grid_w;
    const size_t patch_size = static_cast<size_t>(preprocess_cfg_.patch_size);
    const size_t merge_size = static_cast<size_t>(preprocess_cfg_.merge_size);
    const size_t patch_stride = channels * temporal_patch * patch_size * patch_size;

    ov::Tensor pixel_values(ov::element::f32,
                            {static_cast<size_t>(total_patches),
                             channels, temporal_patch, patch_size, patch_size});
    float* out = pixel_values.data<float>();
    const float* data = stacked.data();
    const size_t frame_stride = channels * out_h * out_w;
    const size_t merged_h = static_cast<size_t>(grid_h) / merge_size;
    const size_t merged_w = static_cast<size_t>(grid_w) / merge_size;
    size_t patch_offset = 0;

    for (int64_t t = 0; t < grid_t; ++t) {
        for (size_t bh = 0; bh < merged_h; ++bh) {
            for (size_t bw = 0; bw < merged_w; ++bw) {
                for (size_t mh = 0; mh < merge_size; ++mh) {
                    for (size_t mw = 0; mw < merge_size; ++mw) {
                        float* dst = out + patch_offset * patch_stride;
                        size_t dst_idx = 0;
                        const size_t h_idx = (bh * merge_size + mh) * patch_size;
                        const size_t w_idx = (bw * merge_size + mw) * patch_size;
                        for (size_t c = 0; c < channels; ++c) {
                            for (size_t tp = 0; tp < temporal_patch; ++tp) {
                                const size_t t_idx = (static_cast<size_t>(t) * temporal_patch + tp) * frame_stride;
                                for (size_t ph = 0; ph < patch_size; ++ph) {
                                    for (size_t pw = 0; pw < patch_size; ++pw) {
                                        const size_t src_idx_val =
                                            t_idx + (c * out_h + h_idx + ph) * out_w + w_idx + pw;
                                        dst[dst_idx++] = data[src_idx_val];
                                    }
                                }
                            }
                        }
                        patch_offset++;
                    }
                }
            }
        }
    }

    auto pos_embeds = build_pos_embeds(pos_embed_weight, grid_thw, preprocess_cfg_.merge_size);
    auto rotary = build_rotary_cos_sin(grid_thw, vision_cfg_, preprocess_cfg_.merge_size);

    return {pixel_values, grid_thw, pos_embeds, rotary.first, rotary.second};
}

int64_t Qwen3VLVisionPreprocessor::count_visual_tokens(const ov::Tensor& grid_thw,
                                                       int32_t spatial_merge_size) {
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

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
