// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/processing_qwen3_omni.hpp"

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

// ────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ────────────────────────────────────────────────────────────────────────────

void read_config_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

// ────────────────────────────────────────────────────────────────────────────
// Attention-mask helpers (shared with Qwen3VL logic)
// ────────────────────────────────────────────────────────────────────────────

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
            OPENVINO_THROW("Unsupported attention_mask dtype in Qwen3Omni");
    }
}

void set_bool(ov::Tensor& mask, size_t index, bool value) {
    auto* data = mask.data<char>();
    data[index] = value ? 1 : 0;
}

// ────────────────────────────────────────────────────────────────────────────
// Scatter helper shared by both visual and audio embeddings
// ────────────────────────────────────────────────────────────────────────────

ov::Tensor scatter_embeds_impl(const ov::Tensor& flat_embeds,
                               const ov::Tensor& pos_mask,
                               const char* caller_name) {
    const auto mask_shape = pos_mask.get_shape();
    if (mask_shape.size() != 2) {
        OPENVINO_THROW(caller_name, ": pos_mask must have shape [B, S]");
    }
    const auto embeds_shape = flat_embeds.get_shape();
    // Accept both [V, H] (2-D) and [B, V, H] (3-D).
    // In the 3-D case the first dimension is a batch axis and each batch item
    // has its own V vectors.
    const bool is_3d = (embeds_shape.size() == 3);
    if (!is_3d && embeds_shape.size() != 2) {
        OPENVINO_THROW(caller_name, ": flat_embeds must have shape [V, H] or [B, V, H]");
    }

    const size_t batch   = mask_shape[0];
    const size_t seq_len = mask_shape[1];
    const size_t hidden  = is_3d ? embeds_shape[2] : embeds_shape[1];
    // Total number of embedding vectors per batch item.
    const size_t v_per_batch = is_3d ? embeds_shape[1] : embeds_shape[0];
    // For 2-D input treat all V vectors as belonging to one shared pool that
    // is consumed left-to-right across the whole batch (original behaviour).
    const size_t total_v = is_3d ? (embeds_shape[0] * embeds_shape[1]) : embeds_shape[0];

    ov::Tensor out(flat_embeds.get_element_type(), {batch, seq_len, hidden});
    std::memset(out.data(), 0, out.get_byte_size());

    const size_t elem_size = flat_embeds.get_element_type().size();
    const size_t row_bytes = hidden * elem_size;

    const char* src = static_cast<const char*>(flat_embeds.data());
    char*       dst = static_cast<char*>(out.data());

    if (is_3d) {
        // Per-batch scattering: batch item b uses embeds[b, 0..V-1, :].
        for (size_t b = 0; b < batch; ++b) {
            size_t embed_idx = 0;
            // src base for this batch item
            const char* src_b = src + b * v_per_batch * row_bytes;
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t global_idx = b * seq_len + s;
                if (!mask_value(pos_mask, global_idx)) {
                    continue;
                }
                if (embed_idx >= v_per_batch) {
                    OPENVINO_THROW(caller_name, ": flat_embeds[", b, "] shorter than pos_mask row");
                }
                std::memcpy(dst + global_idx * row_bytes,
                            src_b + embed_idx * row_bytes,
                            row_bytes);
                ++embed_idx;
            }
        }
    } else {
        // 2-D: shared pool across the whole batch (original behaviour).
        size_t embed_idx = 0;
        const size_t total = batch * seq_len;
        for (size_t idx = 0; idx < total; ++idx) {
            if (!mask_value(pos_mask, idx)) {
                continue;
            }
            if (embed_idx >= total_v) {
                OPENVINO_THROW(caller_name, ": flat_embeds shorter than pos_mask");
            }
            std::memcpy(dst + idx * row_bytes, src + embed_idx * row_bytes, row_bytes);
            ++embed_idx;
        }
        if (embed_idx != total_v) {
            OPENVINO_THROW(caller_name, ": flat_embeds length does not match pos_mask");
        }
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Audio-config JSON parsing
// ────────────────────────────────────────────────────────────────────────────

void parse_audio_config(
    const nlohmann::json& data,
    ov::genai::modeling::models::Qwen3OmniAudioEncoderConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "num_mel_bins",           cfg.num_mel_bins);
    read_json_param(data, "encoder_layers",         cfg.encoder_layers);
    read_json_param(data, "encoder_attention_heads",cfg.encoder_attention_heads);
    read_json_param(data, "encoder_ffn_dim",        cfg.encoder_ffn_dim);
    read_json_param(data, "d_model",                cfg.d_model);
    read_json_param(data, "dropout",                cfg.dropout);
    read_json_param(data, "attention_dropout",      cfg.attention_dropout);
    read_json_param(data, "activation_function",    cfg.activation_function);
    read_json_param(data, "activation_dropout",     cfg.activation_dropout);
    read_json_param(data, "scale_embedding",        cfg.scale_embedding);
    read_json_param(data, "initializer_range",      cfg.initializer_range);
    read_json_param(data, "max_source_positions",   cfg.max_source_positions);
    read_json_param(data, "n_window",               cfg.n_window);
    read_json_param(data, "output_dim",             cfg.output_dim);
    read_json_param(data, "n_window_infer",         cfg.n_window_infer);
    read_json_param(data, "conv_chunksize",         cfg.conv_chunksize);
    read_json_param(data, "downsample_hidden_size", cfg.downsample_hidden_size);
}

// ────────────────────────────────────────────────────────────────────────────
// Vision / text config parsing (reuses qwen3_vl internals via free functions)
// ────────────────────────────────────────────────────────────────────────────

void parse_rope_config(const nlohmann::json& data,
                       ov::genai::modeling::models::Qwen3VLRopeConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "mrope_interleaved", cfg.mrope_interleaved);
    read_json_param(data, "mrope_section",     cfg.mrope_section);
    read_json_param(data, "rope_type",         cfg.rope_type);
}

void parse_text_config(const nlohmann::json& data,
                       ov::genai::modeling::models::Qwen3VLTextConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type",             cfg.model_type);
    read_json_param(data, "vocab_size",             cfg.vocab_size);
    read_json_param(data, "hidden_size",            cfg.hidden_size);
    read_json_param(data, "intermediate_size",      cfg.intermediate_size);
    read_json_param(data, "num_hidden_layers",      cfg.num_hidden_layers);
    read_json_param(data, "num_attention_heads",    cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads",    cfg.num_key_value_heads);
    read_json_param(data, "head_dim",               cfg.head_dim);
    read_json_param(data, "max_position_embeddings",cfg.max_position_embeddings);
    read_json_param(data, "rms_norm_eps",           cfg.rms_norm_eps);
    read_json_param(data, "rope_theta",             cfg.rope_theta);
    read_json_param(data, "hidden_act",             cfg.hidden_act);
    read_json_param(data, "attention_bias",         cfg.attention_bias);
    read_json_param(data, "attention_dropout",      cfg.attention_dropout);
    read_json_param(data, "tie_word_embeddings",    cfg.tie_word_embeddings);
    read_json_param(data, "dtype",                  cfg.dtype);

    if (data.contains("rope_scaling")) {
        parse_rope_config(data.at("rope_scaling"), cfg.rope);
    }
    if (data.contains("rope_parameters")) {
        parse_rope_config(data.at("rope_parameters"), cfg.rope);
    }

    cfg.finalize();
}

void parse_vision_config(const nlohmann::json& data,
                         ov::genai::modeling::models::Qwen3VLVisionConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type",              cfg.model_type);
    read_json_param(data, "depth",                   cfg.depth);
    read_json_param(data, "hidden_size",             cfg.hidden_size);
    read_json_param(data, "hidden_act",              cfg.hidden_act);
    read_json_param(data, "intermediate_size",       cfg.intermediate_size);
    read_json_param(data, "num_heads",               cfg.num_heads);
    read_json_param(data, "in_channels",             cfg.in_channels);
    read_json_param(data, "patch_size",              cfg.patch_size);
    read_json_param(data, "spatial_merge_size",      cfg.spatial_merge_size);
    read_json_param(data, "temporal_patch_size",     cfg.temporal_patch_size);
    read_json_param(data, "out_hidden_size",         cfg.out_hidden_size);
    read_json_param(data, "num_position_embeddings", cfg.num_position_embeddings);
    read_json_param(data, "deepstack_visual_indexes",cfg.deepstack_visual_indexes);
    read_json_param(data, "initializer_range",       cfg.initializer_range);

    cfg.finalize();
}

}  // anonymous namespace

// ────────────────────────────────────────────────────────────────────────────
namespace ov {
namespace genai {
namespace modeling {
namespace models {

// ────────────────────────────────────────────────────────────────────────────
// get_feat_extract_output_length
//
// Python equivalent (processing_qwen3_omni.py):
//
//   def _get_feat_extract_output_lengths(input_lengths):
//       input_lengths_leave = input_lengths % 100
//       feat_lengths  = (input_lengths_leave - 1) // 2 + 1
//       output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
//                       + (input_lengths // 100) * 13
//       return output_lengths
// ────────────────────────────────────────────────────────────────────────────

int64_t get_feat_extract_output_length(int64_t input_length) {
    const int64_t leave       = input_length % 100;
    const int64_t feat_len    = (leave - 1) / 2 + 1;
    const int64_t output_len  = ((feat_len - 1) / 2 + 1 - 1) / 2 + 1
                                + (input_length / 100) * 13;
    return output_len;
}

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniThinkerConfig
// ────────────────────────────────────────────────────────────────────────────

void Qwen3OmniThinkerConfig::finalize() {
    text.finalize();
    vision.finalize();
}

void Qwen3OmniThinkerConfig::validate() const {
    text.validate();
    vision.validate();
    if (image_token_id < 0) {
        OPENVINO_THROW("Qwen3OmniThinkerConfig.image_token_id must be >= 0");
    }
    if (audio_token_id < 0) {
        OPENVINO_THROW("Qwen3OmniThinkerConfig.audio_token_id must be >= 0");
    }
    if (vision_start_token_id < 0) {
        OPENVINO_THROW("Qwen3OmniThinkerConfig.vision_start_token_id must be >= 0");
    }
}

Qwen3OmniThinkerConfig Qwen3OmniThinkerConfig::from_json(const nlohmann::json& root) {
    using ov::genai::utils::read_json_param;

    // The Python model wraps thinker fields under "thinker_config".
    // Fall back to the root if the key is absent (e.g. direct thinker JSON).
    const nlohmann::json& data =
        root.contains("thinker_config") ? root.at("thinker_config") : root;

    Qwen3OmniThinkerConfig cfg;

    if (data.contains("text_config")) {
        parse_text_config(data.at("text_config"), cfg.text);
    }

    if (data.contains("vision_config")) {
        parse_vision_config(data.at("vision_config"), cfg.vision);
    }

    if (data.contains("audio_config")) {
        parse_audio_config(data.at("audio_config"), cfg.audio);
    }

    read_json_param(data, "image_token_id",        cfg.image_token_id);
    read_json_param(data, "video_token_id",         cfg.video_token_id);
    read_json_param(data, "audio_token_id",         cfg.audio_token_id);
    read_json_param(data, "vision_start_token_id",  cfg.vision_start_token_id);
    read_json_param(data, "vision_end_token_id",    cfg.vision_end_token_id);
    read_json_param(data, "audio_start_token_id",   cfg.audio_start_token_id);
    read_json_param(data, "audio_end_token_id",     cfg.audio_end_token_id);
    read_json_param(data, "position_id_per_seconds",cfg.position_id_per_seconds);

    cfg.finalize();
    cfg.validate();
    return cfg;
}

Qwen3OmniThinkerConfig Qwen3OmniThinkerConfig::from_json_file(
    const std::filesystem::path& config_path) {
    auto resolved = resolve_config_path(config_path);
    if (!std::filesystem::exists(resolved)) {
        OPENVINO_THROW("Config file not found: ", resolved.string());
    }
    nlohmann::json data;
    read_config_json_file(resolved, data);
    return from_json(data);
}

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniInputPlanner
// ────────────────────────────────────────────────────────────────────────────

Qwen3OmniInputPlanner::Qwen3OmniInputPlanner(const Qwen3OmniThinkerConfig& cfg)
    : image_token_id_(cfg.image_token_id),
      video_token_id_(cfg.video_token_id),
      audio_token_id_(cfg.audio_token_id),
      vision_start_token_id_(cfg.vision_start_token_id),
      vision_end_token_id_(cfg.vision_end_token_id),
      audio_start_token_id_(cfg.audio_start_token_id),
      audio_end_token_id_(cfg.audio_end_token_id),
      spatial_merge_size_(cfg.vision.spatial_merge_size),
      position_id_per_seconds_(cfg.position_id_per_seconds) {}

// Build a boolean mask that is true at every position where `token_id` appears
// in input_ids (and the optional attention_mask is non-zero).
ov::Tensor Qwen3OmniInputPlanner::build_pos_mask(const ov::Tensor& input_ids,
                                                  const ov::Tensor* attention_mask,
                                                  int64_t token_id) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for Qwen3OmniInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }

    ov::Tensor mask(ov::element::boolean, shape);
    const int64_t* ids  = input_ids.data<const int64_t>();
    const size_t   total = input_ids.get_size();

    for (size_t idx = 0; idx < total; ++idx) {
        bool active = (ids[idx] == token_id);
        if (attention_mask && !mask_value(*attention_mask, idx)) {
            active = false;
        }
        set_bool(mask, idx, active);
    }
    return mask;
}

Qwen3OmniInputPlan Qwen3OmniInputPlanner::build_plan(
    const ov::Tensor& input_ids,
    const ov::Tensor* attention_mask,
    const ov::Tensor* image_grid_thw,
    const ov::Tensor* video_grid_thw,
    const ov::Tensor* audio_seqlens,
    const std::vector<float>* second_per_grids,
    bool use_audio_in_video) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for Qwen3OmniInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    // Validate grid tensors
    auto validate_grid = [&](const ov::Tensor* g, const char* name) {
        if (!g) return;
        if (g->get_element_type() != ov::element::i64)
            OPENVINO_THROW(name, " must be i64");
        const auto& gs = g->get_shape();
        if (gs.size() != 2 || gs[1] != 3)
            OPENVINO_THROW(name, " must have shape [N, 3]");
    };
    validate_grid(image_grid_thw, "image_grid_thw");
    validate_grid(video_grid_thw, "video_grid_thw");
    if (audio_seqlens) {
        if (audio_seqlens->get_element_type() != ov::element::i64)
            OPENVINO_THROW("audio_seqlens must be i64");
        if (audio_seqlens->get_shape().size() != 1)
            OPENVINO_THROW("audio_seqlens must have shape [N]");
    }
    if (spatial_merge_size_ <= 0)
        OPENVINO_THROW("spatial_merge_size must be > 0");

    const size_t batch   = shape[0];
    const size_t seq_len = shape[1];

    ov::Tensor position_ids(ov::element::i64, {3, batch, seq_len});
    std::memset(position_ids.data(), 0, position_ids.get_byte_size());
    ov::Tensor rope_deltas(ov::element::i64, {batch, 1});
    std::memset(rope_deltas.data(), 0, rope_deltas.get_byte_size());

    // Build pos-masks over the full [B, S] space (includes padding positions)
    // visual_pos_mask marks image AND video token positions
    ov::Tensor visual_pos_mask = build_pos_mask(input_ids, attention_mask, image_token_id_);
    {
        // also mark video tokens
        const int64_t* ids_ptr = input_ids.data<const int64_t>();
        const size_t total = input_ids.get_size();
        for (size_t idx = 0; idx < total; ++idx) {
            if (ids_ptr[idx] == video_token_id_) {
                if (!attention_mask || mask_value(*attention_mask, idx))
                    set_bool(visual_pos_mask, idx, true);
            }
        }
    }
    auto audio_pos_mask = build_pos_mask(input_ids, attention_mask, audio_token_id_);

    const int64_t* ids       = input_ids.data<const int64_t>();
    const int64_t* img_grid  = image_grid_thw ? image_grid_thw->data<const int64_t>() : nullptr;
    const int64_t* vid_grid  = video_grid_thw ? video_grid_thw->data<const int64_t>() : nullptr;
    const int64_t* aud_lens  = audio_seqlens  ? audio_seqlens->data<const int64_t>()  : nullptr;
    const size_t img_rows = image_grid_thw ? image_grid_thw->get_shape()[0] : 0;
    const size_t vid_rows = video_grid_thw ? video_grid_thw->get_shape()[0] : 0;
    const size_t aud_rows = audio_seqlens  ? audio_seqlens->get_shape()[0]  : 0;

    auto* pos_data   = position_ids.data<int64_t>();
    auto* delta_data = rope_deltas.data<int64_t>();

    size_t image_idx = 0;
    size_t video_idx = 0;
    size_t audio_idx = 0;

    for (size_t b = 0; b < batch; ++b) {
        // ── Collect non-padded tokens for this batch item ──────────────────
        std::vector<int64_t> tokens;
        std::vector<size_t>  active_seq_pos;
        tokens.reserve(seq_len);
        active_seq_pos.reserve(seq_len);
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t idx = b * seq_len + s;
            if (attention_mask && !mask_value(*attention_mask, idx))
                continue;
            tokens.push_back(ids[idx]);
            active_seq_pos.push_back(s);
        }
        if (tokens.empty()) {
            delta_data[b] = 0;
            continue;
        }

        // ── Count multimodal blocks so we know how many iterations to do ───
        // Mirrors Python: count vision_start occurrences that lead to
        // image / video tokens, plus audio_start occurrences.
        size_t remain_images = 0, remain_videos = 0, remain_audios = 0;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            if (tokens[i] == vision_start_token_id_) {
                if (tokens[i + 1] == image_token_id_)  ++remain_images;
                else if (use_audio_in_video && tokens[i + 1] == audio_start_token_id_) ++remain_videos;
                else if (!use_audio_in_video && tokens[i + 1] == video_token_id_) ++remain_videos;
            }
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] == audio_start_token_id_) ++remain_audios;
        }
        // In audio-in-video mode, each video block consumes one audio slot too
        if (use_audio_in_video) remain_audios -= remain_videos;
        const size_t multimodal_nums = (use_audio_in_video)
            ? (remain_images + remain_audios)   // standalone audio + images
            : (remain_images + remain_videos + remain_audios);

        std::vector<int64_t> pos_t, pos_h, pos_w;
        pos_t.reserve(tokens.size());
        pos_h.reserve(tokens.size());
        pos_w.reserve(tokens.size());

        // ── Helper: sequential 1-D positions (text / bos / eos / audio) ───
        // Mirrors: torch.arange(length).expand(3,-1) + st_idx
        auto append_seq = [&](size_t length) {
            if (length == 0) return;
            const int64_t base = pos_t.empty() ? 0 : (std::max({pos_t.back(), pos_h.back(), pos_w.back()}) + 1);
            for (size_t i = 0; i < length; ++i) {
                const int64_t v = base + static_cast<int64_t>(i);
                pos_t.push_back(v);
                pos_h.push_back(v);
                pos_w.push_back(v);
            }
        };

        // ── Helper: get_llm_pos_ids_for_vision ────────────────────────────
        // t_index: per-frame temporal position (may be non-unit stride)
        // Produces llm_grid_t * llm_grid_h * llm_grid_w triplets.
        auto append_visual_with_tindex = [&](const std::vector<double>& t_index,
                                              int64_t h, int64_t w) {
            const int64_t llm_grid_h = h / spatial_merge_size_;
            const int64_t llm_grid_w = w / spatial_merge_size_;
            if (llm_grid_h <= 0 || llm_grid_w <= 0)
                OPENVINO_THROW("Invalid spatial_merge_size or grid h/w");
            const int64_t start_idx = pos_t.empty()
                ? 0
                : (std::max({pos_t.back(), pos_h.back(), pos_w.back()}) + 1);
            double max_t_val = 0.0;
            for (const auto& tv : t_index) max_t_val = std::max(max_t_val, tv);
            for (size_t ti = 0; ti < t_index.size(); ++ti) {
                const int64_t t_pos = start_idx + static_cast<int64_t>(std::round(t_index[ti]));
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        pos_t.push_back(t_pos);
                        pos_h.push_back(start_idx + hh);
                        pos_w.push_back(start_idx + ww);
                    }
                }
            }
        };

        size_t st = 0;  // current position in tokens[]

        for (size_t mm = 0; mm < multimodal_nums; ++mm) {
            // Find the next vision_start and audio_start from st onwards
            size_t ed_vision = tokens.size() + 1;  // sentinel
            size_t ed_audio  = tokens.size() + 1;

            if (remain_images > 0 || remain_videos > 0) {
                for (size_t i = st; i < tokens.size(); ++i) {
                    if (tokens[i] == vision_start_token_id_) { ed_vision = i; break; }
                }
            }
            if (remain_audios > 0) {
                for (size_t i = st; i < tokens.size(); ++i) {
                    if (tokens[i] == audio_start_token_id_) { ed_audio = i; break; }
                }
            }

            const size_t min_ed = std::min(ed_vision, ed_audio);
            if (min_ed == tokens.size() + 1) break;  // no more blocks

            const size_t text_len = min_ed - st;

            // Determine bos_len / eos_len
            // audio-in-video: vision_start immediately followed by audio_start
            bool is_audio_in_video = use_audio_in_video
                && min_ed == ed_vision
                && ed_vision + 1 < tokens.size()
                && tokens[ed_vision + 1] == audio_start_token_id_;
            const size_t bos_len = is_audio_in_video ? 2 : 1;
            const size_t eos_len = is_audio_in_video ? 2 : 1;

            // Text before this block
            append_seq(text_len);
            st += text_len;

            // BOS token(s)
            append_seq(bos_len);
            st += bos_len;

            // ── Audio-only block ──────────────────────────────────────────
            if (min_ed == ed_audio && !is_audio_in_video) {
                // Count actual audio tokens between audio_start and audio_end
                // in the token stream (more reliable than aud_lens which may
                // be computed from a different length formula).
                size_t scan = min_ed + 1 /*skip audio_start*/;
                while (scan < tokens.size() && tokens[scan] != audio_end_token_id_)
                    ++scan;
                const int64_t audio_len = static_cast<int64_t>(scan - (min_ed + 1));
                // Keep audio_idx in sync so callers can still use aud_lens for
                // other purposes (e.g., audio encoder feeding).
                if (audio_idx < aud_rows) ++audio_idx;
                append_seq(static_cast<size_t>(audio_len));
                st += static_cast<size_t>(audio_len);
                --remain_audios;

            // ── Image-only block ──────────────────────────────────────────
            } else if (min_ed == ed_vision
                       && ed_vision + 1 < tokens.size()
                       && tokens[ed_vision + 1] == image_token_id_) {
                if (image_idx >= img_rows)
                    OPENVINO_THROW("image_grid_thw has fewer rows than image tokens");
                const int64_t g_t = img_grid[image_idx * 3 + 0];
                const int64_t g_h = img_grid[image_idx * 3 + 1];
                const int64_t g_w = img_grid[image_idx * 3 + 2];
                // t_index = arange(g_t) * 1 * position_id_per_seconds
                std::vector<double> t_index(static_cast<size_t>(g_t));
                for (int64_t ti = 0; ti < g_t; ++ti)
                    t_index[static_cast<size_t>(ti)] = ti * position_id_per_seconds_;
                append_visual_with_tindex(t_index, g_h, g_w);
                const int64_t image_len = g_t * (g_h / spatial_merge_size_) * (g_w / spatial_merge_size_);
                st += static_cast<size_t>(image_len);
                ++image_idx;
                --remain_images;

            // ── Video-only block ──────────────────────────────────────────
            } else if (min_ed == ed_vision
                       && ed_vision + 1 < tokens.size()
                       && tokens[ed_vision + 1] == video_token_id_
                       && !is_audio_in_video) {
                if (video_idx >= vid_rows)
                    OPENVINO_THROW("video_grid_thw has fewer rows than video tokens");
                const int64_t g_t = vid_grid[video_idx * 3 + 0];
                const int64_t g_h = vid_grid[video_idx * 3 + 1];
                const int64_t g_w = vid_grid[video_idx * 3 + 2];
                const double spg = (second_per_grids && video_idx < second_per_grids->size())
                    ? static_cast<double>((*second_per_grids)[video_idx]) : 1.0;
                std::vector<double> t_index(static_cast<size_t>(g_t));
                for (int64_t ti = 0; ti < g_t; ++ti)
                    t_index[static_cast<size_t>(ti)] = ti * spg * position_id_per_seconds_;
                append_visual_with_tindex(t_index, g_h, g_w);
                const int64_t video_len = g_t * (g_h / spatial_merge_size_) * (g_w / spatial_merge_size_);
                st += static_cast<size_t>(video_len);
                ++video_idx;
                --remain_videos;

            // ── Audio-in-Video block ──────────────────────────────────────
            } else if (is_audio_in_video) {
                // Count actual audio tokens from the audio_start that follows
                // vision_start in an audio-in-video block.
                const size_t audio_start_pos = ed_vision + 1; // audio_start_token_id
                size_t scan_av = audio_start_pos + 1;
                while (scan_av < tokens.size() && tokens[scan_av] != audio_end_token_id_)
                    ++scan_av;
                const int64_t audio_len = static_cast<int64_t>(scan_av - (audio_start_pos + 1));
                if (audio_idx < aud_rows) ++audio_idx;
                if (video_idx >= vid_rows)
                    OPENVINO_THROW("video_grid_thw has fewer rows than video tokens (audio-in-video)");
                const int64_t g_t = vid_grid[video_idx * 3 + 0];
                const int64_t g_h = vid_grid[video_idx * 3 + 1];
                const int64_t g_w = vid_grid[video_idx * 3 + 2];
                const double spg = (second_per_grids && video_idx < second_per_grids->size())
                    ? static_cast<double>((*second_per_grids)[video_idx]) : 1.0;
                const int64_t llm_grid_h = g_h / spatial_merge_size_;
                const int64_t llm_grid_w = g_w / spatial_merge_size_;
                const int64_t video_len = g_t * llm_grid_h * llm_grid_w;

                // Build video and audio pos-id vectors separately, then merge
                const int64_t start_idx = pos_t.empty()
                    ? 0
                    : (std::max({pos_t.back(), pos_h.back(), pos_w.back()}) + 1);

                // video pos ids
                std::vector<int64_t> vpt, vph, vpw;
                vpt.reserve(static_cast<size_t>(video_len));
                vph.reserve(static_cast<size_t>(video_len));
                vpw.reserve(static_cast<size_t>(video_len));
                for (int64_t ti = 0; ti < g_t; ++ti) {
                    const int64_t tp = start_idx + static_cast<int64_t>(std::round(ti * spg * position_id_per_seconds_));
                    for (int64_t hh = 0; hh < llm_grid_h; ++hh)
                        for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                            vpt.push_back(tp); vph.push_back(start_idx + hh); vpw.push_back(start_idx + ww);
                        }
                }
                // audio pos ids
                std::vector<int64_t> apt, aph, apw;
                apt.reserve(static_cast<size_t>(audio_len));
                for (int64_t ai = 0; ai < audio_len; ++ai) {
                    const int64_t v = start_idx + ai;
                    apt.push_back(v); aph.push_back(v); apw.push_back(v);
                }
                // Merge: video_t[vi] <= audio_t[ai] → video first
                size_t vi = 0, ai = 0;
                while (vi < vpt.size() && ai < apt.size()) {
                    if (vpt[vi] <= apt[ai]) {
                        pos_t.push_back(vpt[vi]); pos_h.push_back(vph[vi]); pos_w.push_back(vpw[vi++]);
                    } else {
                        pos_t.push_back(apt[ai]); pos_h.push_back(aph[ai]); pos_w.push_back(apw[ai++]);
                    }
                }
                while (vi < vpt.size()) { pos_t.push_back(vpt[vi]); pos_h.push_back(vph[vi]); pos_w.push_back(vpw[vi++]); }
                while (ai < apt.size()) { pos_t.push_back(apt[ai]); pos_h.push_back(aph[ai]); pos_w.push_back(apw[ai++]); }

                st += static_cast<size_t>(video_len + audio_len);
                ++video_idx;
                --remain_videos;
                --remain_audios;
            }

            // EOS token(s)
            append_seq(eos_len);
            st += eos_len;
        }

        // Remaining text after all multimodal blocks
        if (st < tokens.size())
            append_seq(tokens.size() - st);

        if (pos_t.size() != tokens.size())
            OPENVINO_THROW("Position ids length mismatch in Qwen3Omni: pos=",
                           pos_t.size(), " tokens=", tokens.size());

        // Compute max position
        int64_t max_pos = 0;
        for (size_t i = 0; i < pos_t.size(); ++i)
            max_pos = std::max(max_pos, std::max(pos_t[i], std::max(pos_h[i], pos_w[i])));

        // Write back to output tensor
        for (size_t i = 0; i < tokens.size(); ++i) {
            const size_t s    = active_seq_pos[i];
            const size_t base = b * seq_len + s;
            pos_data[0 * batch * seq_len + base] = pos_t[i];
            pos_data[1 * batch * seq_len + base] = pos_h[i];
            pos_data[2 * batch * seq_len + base] = pos_w[i];
        }

        // Fill padded positions with 1 (same convention as Qwen3VL)
        if (attention_mask) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t idx = b * seq_len + s;
                if (mask_value(*attention_mask, idx)) continue;
                pos_data[0 * batch * seq_len + idx] = 1;
                pos_data[1 * batch * seq_len + idx] = 1;
                pos_data[2 * batch * seq_len + idx] = 1;
            }
        }

        delta_data[b] = max_pos + 1 - static_cast<int64_t>(seq_len);
    }

    return {position_ids, visual_pos_mask, audio_pos_mask, rope_deltas};
}

ov::Tensor Qwen3OmniInputPlanner::build_decode_position_ids(
    const ov::Tensor& rope_deltas,
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

    ov::Tensor position_ids(ov::element::i64,
                            {3, batch, static_cast<size_t>(seq_len)});
    auto* out            = position_ids.data<int64_t>();
    const int64_t* deltas = rope_deltas.data<const int64_t>();
    const size_t plane_stride = batch * static_cast<size_t>(seq_len);

    for (size_t b = 0; b < batch; ++b) {
        const int64_t base = past_length + deltas[b];
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t value = base + s;
            const size_t idx = b * static_cast<size_t>(seq_len)
                               + static_cast<size_t>(s);
            out[idx]                      = value;
            out[plane_stride + idx]       = value;
            out[2 * plane_stride + idx]   = value;
        }
    }
    return position_ids;
}

ov::Tensor Qwen3OmniInputPlanner::scatter_visual_embeds(
    const ov::Tensor& visual_embeds,
    const ov::Tensor& visual_pos_mask) {
    return scatter_embeds_impl(visual_embeds, visual_pos_mask,
                               "Qwen3OmniInputPlanner::scatter_visual_embeds");
}

ov::Tensor Qwen3OmniInputPlanner::scatter_audio_embeds(
    const ov::Tensor& audio_embeds,
    const ov::Tensor& audio_pos_mask) {
    return scatter_embeds_impl(audio_embeds, audio_pos_mask,
                               "Qwen3OmniInputPlanner::scatter_audio_embeds");
}

std::vector<ov::Tensor> Qwen3OmniInputPlanner::scatter_deepstack_embeds(
    const std::vector<ov::Tensor>& deepstack_embeds,
    const ov::Tensor& visual_pos_mask) {
    std::vector<ov::Tensor> out;
    out.reserve(deepstack_embeds.size());
    for (const auto& embed : deepstack_embeds) {
        out.push_back(scatter_visual_embeds(embed, visual_pos_mask));
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniTalkerConfig
// ────────────────────────────────────────────────────────────────────────────

void Qwen3OmniTalkerConfig::finalize() {
    // Nothing to derive for now; reserved for future computed fields.
}

void Qwen3OmniTalkerConfig::validate() const {
    if (num_code_groups <= 0) {
        OPENVINO_THROW("Qwen3OmniTalkerConfig: num_code_groups must be positive");
    }
    if (thinker_hidden_size <= 0) {
        OPENVINO_THROW("Qwen3OmniTalkerConfig: thinker_hidden_size must be positive");
    }
}

Qwen3OmniTalkerConfig Qwen3OmniTalkerConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    Qwen3OmniTalkerConfig cfg;
    read_json_param(data, "num_code_groups",         cfg.num_code_groups);
    read_json_param(data, "thinker_hidden_size",     cfg.thinker_hidden_size);
    read_json_param(data, "codec_eos_token_id",      cfg.codec_eos_token_id);
    read_json_param(data, "accept_hidden_layer",     cfg.accept_hidden_layer);
    read_json_param(data, "codec_nothink_id",        cfg.codec_nothink_id);
    read_json_param(data, "codec_think_bos_id",      cfg.codec_think_bos_id);
    read_json_param(data, "codec_think_eos_id",      cfg.codec_think_eos_id);
    read_json_param(data, "codec_pad_id",            cfg.codec_pad_id);
    read_json_param(data, "codec_bos_id",            cfg.codec_bos_id);
    read_json_param(data, "audio_token_id",          cfg.audio_token_id);
    read_json_param(data, "image_token_id",          cfg.image_token_id);
    read_json_param(data, "video_token_id",          cfg.video_token_id);
    read_json_param(data, "vision_start_token_id",   cfg.vision_start_token_id);
    read_json_param(data, "position_id_per_seconds", cfg.position_id_per_seconds);
    read_json_param(data, "audio_start_token_id",    cfg.audio_start_token_id);
    cfg.finalize();
    cfg.validate();
    return cfg;
}

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniCode2WavConfig
// ────────────────────────────────────────────────────────────────────────────

void Qwen3OmniCode2WavConfig::finalize() {
    // Nothing to derive for now.
}

void Qwen3OmniCode2WavConfig::validate() const {
    if (codebook_size <= 0) {
        OPENVINO_THROW("Qwen3OmniCode2WavConfig: codebook_size must be positive");
    }
    if (hidden_size <= 0) {
        OPENVINO_THROW("Qwen3OmniCode2WavConfig: hidden_size must be positive");
    }
    if (num_hidden_layers <= 0) {
        OPENVINO_THROW("Qwen3OmniCode2WavConfig: num_hidden_layers must be positive");
    }
    if (num_quantizers <= 0) {
        OPENVINO_THROW("Qwen3OmniCode2WavConfig: num_quantizers must be positive");
    }
}

Qwen3OmniCode2WavConfig Qwen3OmniCode2WavConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    Qwen3OmniCode2WavConfig cfg;
    read_json_param(data, "codebook_size",             cfg.codebook_size);
    read_json_param(data, "hidden_size",               cfg.hidden_size);
    read_json_param(data, "max_position_embeddings",   cfg.max_position_embeddings);
    read_json_param(data, "rope_theta",                cfg.rope_theta);
    read_json_param(data, "num_attention_heads",       cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads",       cfg.num_key_value_heads);
    read_json_param(data, "attention_bias",            cfg.attention_bias);
    read_json_param(data, "sliding_window",            cfg.sliding_window);
    read_json_param(data, "intermediate_size",         cfg.intermediate_size);
    read_json_param(data, "hidden_act",                cfg.hidden_act);
    read_json_param(data, "layer_scale_initial_scale", cfg.layer_scale_initial_scale);
    read_json_param(data, "rms_norm_eps",              cfg.rms_norm_eps);
    read_json_param(data, "num_hidden_layers",         cfg.num_hidden_layers);
    read_json_param(data, "num_quantizers",            cfg.num_quantizers);
    read_json_param(data, "upsample_rates",            cfg.upsample_rates);
    read_json_param(data, "upsampling_ratios",         cfg.upsampling_ratios);
    read_json_param(data, "decoder_dim",               cfg.decoder_dim);
    read_json_param(data, "attention_dropout",         cfg.attention_dropout);
    cfg.finalize();
    cfg.validate();
    return cfg;
}

// ────────────────────────────────────────────────────────────────────────────
// Qwen3OmniConfig  (top-level)
// ────────────────────────────────────────────────────────────────────────────

void Qwen3OmniConfig::finalize() {
    thinker.finalize();
    talker.finalize();
    code2wav.finalize();
}

void Qwen3OmniConfig::validate() const {
    thinker.validate();
    talker.validate();
    code2wav.validate();
}

Qwen3OmniConfig Qwen3OmniConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    Qwen3OmniConfig cfg;

    read_json_param(data, "model_type",           cfg.model_type);
    read_json_param(data, "architectures",         cfg.architectures);
    read_json_param(data, "enable_audio_output",   cfg.enable_audio_output);
    read_json_param(data, "im_start_token_id",     cfg.im_start_token_id);
    read_json_param(data, "im_end_token_id",       cfg.im_end_token_id);
    read_json_param(data, "tts_pad_token_id",      cfg.tts_pad_token_id);
    read_json_param(data, "tts_bos_token_id",      cfg.tts_bos_token_id);
    read_json_param(data, "tts_eos_token_id",      cfg.tts_eos_token_id);
    read_json_param(data, "system_token_id",       cfg.system_token_id);
    read_json_param(data, "user_token_id",         cfg.user_token_id);
    read_json_param(data, "assistant_token_id",    cfg.assistant_token_id);

    // thinker_config: reuse ThinkerConfig::from_json which already handles
    // the nested "thinker_config" key internally.
    cfg.thinker = Qwen3OmniThinkerConfig::from_json(data);

    if (data.contains("talker_config")) {
        cfg.talker = Qwen3OmniTalkerConfig::from_json(data.at("talker_config"));
    }

    if (data.contains("code2wav_config")) {
        cfg.code2wav = Qwen3OmniCode2WavConfig::from_json(data.at("code2wav_config"));
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

Qwen3OmniConfig Qwen3OmniConfig::from_json_file(const std::filesystem::path& config_path) {
    auto resolved = resolve_config_path(config_path);
    if (!std::filesystem::exists(resolved)) {
        OPENVINO_THROW("Config file not found: ", resolved.string());
    }
    nlohmann::json data;
    read_config_json_file(resolved, data);
    return from_json(data);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
