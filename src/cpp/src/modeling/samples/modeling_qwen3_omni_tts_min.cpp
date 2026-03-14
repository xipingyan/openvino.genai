// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
// Minimal declaration to avoid pulling in all of windows.h
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);
#endif

#include <openvino/openvino.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "load_image.hpp"
#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_omni/modeling_qwen3_omni_audio.hpp"
#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_audio.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vl.hpp"
#include "modeling/models/qwen3_omni/whisper_mel_spectrogram.hpp"

using namespace ov::genai::modeling::models;

namespace {

void write_wav(const std::string& filename, const float* samples, size_t num_samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create WAV file: " + filename);
    }

    int32_t data_size = static_cast<int32_t>(num_samples * sizeof(int16_t));
    int32_t file_size = 36 + data_size;
    int16_t audio_format = 1;
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * 2;
    int16_t block_align = num_channels * 2;
    int16_t bits_per_sample = 16;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);

    for (size_t i = 0; i < num_samples; ++i) {
        float v = std::max(-1.0f, std::min(1.0f, samples[i]));
        int16_t s = static_cast<int16_t>(v * 32767.0f);
        file.write(reinterpret_cast<char*>(&s), 2);
    }
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("Expected logits shape [1,S,V]");
    }
    const size_t seq = shape[1];
    const size_t vocab = shape[2];
    const auto* data = logits.data<const float>() + (seq - 1) * vocab;
    size_t best_idx = 0;
    float best_val = data[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    return static_cast<int64_t>(best_idx);
}

ov::Tensor make_text_position_ids_i64(size_t seq_len) {
    ov::Tensor t(ov::element::i64, {3, 1, seq_len});
    auto* p = t.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) {
        const int64_t v = static_cast<int64_t>(i);
        p[i] = v;
        p[seq_len + i] = v;
        p[2 * seq_len + i] = v;
    }
    return t;
}

ov::Tensor make_decode_position_ids_i64(int64_t pos) {
    ov::Tensor t(ov::element::i64, {3, 1, 1});
    auto* p = t.data<int64_t>();
    p[0] = pos;
    p[1] = pos;
    p[2] = pos;
    return t;
}

// --- Helper functions for vision pipeline ---

std::string build_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    for (int64_t i = 0; i < image_tokens; ++i) prompt += "<|image_pad|>";
    prompt += "<|vision_end|>";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

/**
 * Build a chat prompt containing audio (and optionally image) placeholders.
 *
 * Format:  <|im_start|>user\n
 *          [<|vision_start|> N×<|image_pad|> <|vision_end|>]   ← if image_tokens > 0
 *          <|audio_start|> M×<|audio_pad|> <|audio_end|>
 *          <user_text>
 *          <|im_end|>\n<|im_start|>assistant\n
 */
std::string build_audio_prompt(const std::string& user_prompt,
                               int64_t audio_tokens,
                               int64_t image_tokens = 0,
                               int64_t video_tokens = 0) {
    std::string prompt = "<|im_start|>user\n";
    if (image_tokens > 0) {
        prompt += "<|vision_start|>";
        for (int64_t i = 0; i < image_tokens; ++i) prompt += "<|image_pad|>";
        prompt += "<|vision_end|>";
    }
    if (video_tokens > 0) {
        prompt += "<|vision_start|>";
        for (int64_t i = 0; i < video_tokens; ++i) prompt += "<|video_pad|>";
        prompt += "<|vision_end|>";
    }
    prompt += "<|audio_start|>";
    for (int64_t i = 0; i < audio_tokens; ++i) prompt += "<|audio_pad|>";
    prompt += "<|audio_end|>";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

/**
 * Build a prompt containing video placeholders (no audio).
 */
std::string build_video_prompt(const std::string& user_prompt, int64_t video_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    for (int64_t i = 0; i < video_tokens; ++i) prompt += "<|video_pad|>";
    prompt += "<|vision_end|>";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

/**
 * Load video frames from a directory of image files using stb_image (via load_image).
 * Files are sorted lexicographically. Returns a vector of [1,H,W,3] u8 tensors.
 */
std::vector<ov::Tensor> load_video_frames(const std::filesystem::path& frames_dir) {
    if (!std::filesystem::is_directory(frames_dir)) {
        throw std::runtime_error("Video frames path is not a directory: " + frames_dir.string());
    }
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(frames_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("No image files (jpg/png/bmp) found in: " + frames_dir.string());
    }
    std::vector<ov::Tensor> frames;
    frames.reserve(paths.size());
    for (const auto& p : paths) {
        frames.push_back(utils::load_image(p));
    }
    return frames;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor t(type, shape);
    std::memset(t.data(), 0, t.get_byte_size());
    return t;
}

/// Concatenate two tensors along dimension 0 (must have same rank and matching other dims).
ov::Tensor concat_tensors_dim0(const ov::Tensor& a, const ov::Tensor& b) {
    auto sa = a.get_shape();
    auto sb = b.get_shape();
    OPENVINO_ASSERT(sa.size() == sb.size(), "concat_tensors_dim0: rank mismatch");
    auto out_shape = sa;
    out_shape[0] = sa[0] + sb[0];
    ov::Tensor out(a.get_element_type(), out_shape);
    const size_t a_bytes = a.get_byte_size();
    const size_t b_bytes = b.get_byte_size();
    std::memcpy(out.data(), a.data(), a_bytes);
    std::memcpy(static_cast<uint8_t*>(out.data()) + a_bytes, b.data(), b_bytes);
    return out;
}

ov::Tensor build_vision_attention_mask(const ov::Tensor& grid_thw) {
    const auto* g = grid_thw.data<const int64_t>();
    const size_t rows = grid_thw.get_shape()[0];
    std::vector<size_t> segments;
    size_t total = 0;
    for (size_t i = 0; i < rows; ++i) {
        const size_t hw = static_cast<size_t>(g[i * 3 + 1] * g[i * 3 + 2]);
        for (int64_t f = 0; f < g[i * 3]; ++f) {
            segments.push_back(hw);
            total += hw;
        }
    }
    ov::Tensor mask(ov::element::f32, {1, 1, total, total});
    auto* d = mask.data<float>();
    std::fill_n(d, mask.get_size(), -std::numeric_limits<float>::infinity());
    size_t s = 0;
    for (size_t len : segments) {
        for (size_t r = s; r < s + len; ++r)
            std::fill(d + r * total + s, d + r * total + s + len, 0.0f);
        s += len;
    }
    return mask;
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    for (const auto& n : {"thinker.model.visual.pos_embed.weight",
                           "model.visual.pos_embed.weight",
                           "visual.pos_embed.weight",
                           "pos_embed.weight"}) {
        if (source.has(n)) return n;
    }
    for (const auto& n : source.keys()) {
        if (n.find("pos_embed.weight") != std::string::npos) return n;
    }
    throw std::runtime_error("Cannot find visual pos_embed.weight in safetensors");
}


// --- Precision mode support (aligned with modeling_qwen3_omni.cpp) ---

enum class PrecisionMode {
    kMixed,
    kDefault,
    kFP32,
    kInfFp16KvInt8,
    kInfFp16KvInt4,
    kInfFp32KvFp32WInt4Asym,
    kInfFp16KvInt8WInt4Asym,
};

static std::string to_lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static PrecisionMode parse_precision_mode(const std::string& value) {
    const auto m = to_lower_copy(value);
    if (m == "mixed" || m == "0" || m == "false" || m == "off") return PrecisionMode::kMixed;
    if (m == "default")                                          return PrecisionMode::kDefault;
    if (m == "fp32" || m == "1" || m == "true" || m == "on")     return PrecisionMode::kFP32;
    if (m == "inf_fp16_kv_int8" || m == "fp16_kv8")              return PrecisionMode::kInfFp16KvInt8;
    if (m == "inf_fp16_kv_int4" || m == "fp16_kv4")              return PrecisionMode::kInfFp16KvInt4;
    if (m == "inf_fp32_kv_fp32_w_int4_asym")                     return PrecisionMode::kInfFp32KvFp32WInt4Asym;
    if (m == "inf_fp16_kv_int8_w_int4_asym")                     return PrecisionMode::kInfFp16KvInt8WInt4Asym;
    throw std::runtime_error("Invalid PRECISION_MODE: " + value);
}

static std::string precision_mode_to_string(PrecisionMode mode) {
    switch (mode) {
    case PrecisionMode::kMixed:                     return "mixed";
    case PrecisionMode::kDefault:                   return "default";
    case PrecisionMode::kFP32:                      return "fp32";
    case PrecisionMode::kInfFp16KvInt8:             return "inf_fp16_kv_int8";
    case PrecisionMode::kInfFp16KvInt4:             return "inf_fp16_kv_int4";
    case PrecisionMode::kInfFp32KvFp32WInt4Asym:    return "inf_fp32_kv_fp32_w_int4_asym";
    case PrecisionMode::kInfFp16KvInt8WInt4Asym:    return "inf_fp16_kv_int8_w_int4_asym";
    }
    return "unknown";
}

/// Build ov::AnyMap compile properties based on precision mode.
static ov::AnyMap compile_props_for_precision(PrecisionMode mode) {
    ov::AnyMap props;
    switch (mode) {
    case PrecisionMode::kFP32:
    case PrecisionMode::kInfFp32KvFp32WInt4Asym:
        props.emplace(ov::hint::inference_precision.name(), ov::element::f32);
        break;
    case PrecisionMode::kDefault:
        props.emplace(ov::hint::inference_precision.name(), ov::element::bf16);
        break;
    case PrecisionMode::kInfFp16KvInt8:
    case PrecisionMode::kInfFp16KvInt4:
    case PrecisionMode::kInfFp16KvInt8WInt4Asym:
        props.emplace(ov::hint::inference_precision.name(), ov::element::f16);
        break;
    default:
        break;
    }
    // KV cache precision
    if (mode == PrecisionMode::kInfFp16KvInt8 || mode == PrecisionMode::kInfFp16KvInt8WInt4Asym) {
        props.emplace(ov::hint::kv_cache_precision.name(), ov::element::u8);
    }
    return props;
}

/// Set runtime_options on the text model according to precision mode.
static void set_text_model_precision(std::shared_ptr<ov::Model>& text_model, PrecisionMode mode) {
    switch (mode) {
    case PrecisionMode::kFP32:
    case PrecisionMode::kInfFp32KvFp32WInt4Asym:
        text_model->set_rt_info(ov::element::f32, {"runtime_options", ov::hint::kv_cache_precision.name()});
        break;
    case PrecisionMode::kDefault:
        text_model->set_rt_info(ov::element::bf16, {"runtime_options", ov::hint::kv_cache_precision.name()});
        break;
    case PrecisionMode::kInfFp16KvInt8:
    case PrecisionMode::kInfFp16KvInt4:
    case PrecisionMode::kInfFp16KvInt8WInt4Asym:
        text_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
        break;
    default:
        break;
    }
    if (mode == PrecisionMode::kInfFp16KvInt8 || mode == PrecisionMode::kInfFp16KvInt8WInt4Asym) {
        text_model->set_rt_info(ov::element::u8, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
}

// --- Text generation with optional vision / audio ---

struct TextGenResult {
    std::string text;
    int64_t prompt_tokens = 0;
    int64_t output_tokens = 0;
    double model_load_ms = 0.0;
    double audio_encode_ms = 0.0;
    double vision_encode_ms = 0.0;
    double preprocess_ms = 0.0;   // tokenization + scatter + plan
    double ttft_ms = 0.0;         // time to first token (prefill infer)
    double decode_ms = 0.0;       // all decode steps
    double tpot_ms = 0.0;         // time per output token
    double throughput = 0.0;      // tokens/s
    double text_gen_total_ms = 0.0;
};

static double elapsed_ms(const std::chrono::steady_clock::time_point& a,
                          const std::chrono::steady_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

TextGenResult run_text_generation(const std::filesystem::path& model_dir,
                                ov::Core& core,
                                ov::genai::modeling::weights::WeightSource& source,
                                ov::genai::modeling::weights::WeightFinalizer& finalizer,
                                const Qwen3OmniConfig& cfg,
                                const std::string& user_prompt,
                                const std::filesystem::path& image_path,
                                const std::filesystem::path& audio_path,
                                const std::filesystem::path& video_path,
                                int max_new_tokens,
                                const std::string& device,
                                PrecisionMode precision_mode = PrecisionMode::kFP32) {
    TextGenResult perf;
    const auto t_start = std::chrono::steady_clock::now();

    const bool has_image = !image_path.empty() && std::filesystem::exists(image_path);
    const bool has_audio = !audio_path.empty() && std::filesystem::exists(audio_path);
    const bool has_video = !video_path.empty() && std::filesystem::exists(video_path) && std::filesystem::is_directory(video_path);
    auto vl_cfg = to_qwen3_omni_vl_cfg(cfg);
    ov::AnyMap props = compile_props_for_precision(precision_mode);
    std::cout << "[TextGen] precision_mode: " << precision_mode_to_string(precision_mode) << std::endl;

    const bool has_multimodal = has_image || has_audio || has_video;
    auto text_model = create_qwen3_omni_text_model(cfg, source, finalizer, false, has_multimodal);
    set_text_model_precision(text_model, precision_mode);

    ov::genai::Tokenizer tokenizer(model_dir);

    ov::Tensor input_ids, attention_mask, position_ids, visual_padded, visual_pos_mask, rope_deltas;
    ov::Tensor audio_features_padded, audio_pos_mask;
    std::vector<ov::Tensor> deepstack_padded;

    std::shared_ptr<ov::Model> vision_model;
    if (has_image || has_video) {
        // Build vision model whenever image or video is present
        vision_model = create_qwen3_omni_vision_model(cfg, source, finalizer);
    }

    auto compiled_text = core.compile_model(text_model, device, props);
    const auto t_model_load = std::chrono::steady_clock::now();
    perf.model_load_ms = elapsed_ms(t_start, t_model_load);

    if (has_audio) {
        // ==========================================
        // Native C++ audio path (no Python bridge)
        // ==========================================
        const auto t_audio_start = std::chrono::steady_clock::now();
        // Step 1: Read WAV → mel spectrogram
        std::cout << "[Audio] Reading WAV: " << audio_path.string() << std::endl;
        auto waveform = read_wav_to_float32(audio_path.string());
        std::cout << "[Audio]   waveform samples: " << waveform.size()
                  << " (" << (waveform.size() / 16000.0) << " seconds)" << std::endl;

        auto mel_cfg_path = model_dir / "preprocessor_config.json";
        WhisperFeatureExtractorConfig mel_cfg;
        if (std::filesystem::exists(mel_cfg_path))
            mel_cfg = WhisperFeatureExtractorConfig::from_json_file(mel_cfg_path);

        auto mel = extract_whisper_mel_features(waveform, mel_cfg);
        const int64_t T_frames = mel.audio_feature_length;
        std::cout << "[Audio]   mel frames (T_frames): " << T_frames << std::endl;

        // Step 2: Chunk mel for audio encoder (n_window*2 = 100 frames per chunk)
        const int32_t chunk_size = cfg.audio.n_window * 2;  // typically 100
        auto chunked = chunk_mel_for_audio_encoder(mel, chunk_size);
        std::cout << "[Audio]   chunked into " << chunked.chunk_lengths.size()
                  << " chunks, expected total tokens: " << chunked.total_output_tokens << std::endl;

        // Step 3: Build & run audio encoder
        std::cout << "[Audio] Building audio encoder model..." << std::endl;
        auto audio_encoder_model = create_qwen3_omni_audio_encoder_model(cfg, source, finalizer);
        auto compiled_audio = core.compile_model(audio_encoder_model, device, props);
        auto areq = compiled_audio.create_infer_request();

        areq.set_tensor(Qwen3OmniAudioIO::kInputFeatures, chunked.input_features);
        areq.set_tensor(Qwen3OmniAudioIO::kFeatureAttentionMask, chunked.feature_attention_mask);
        areq.set_tensor(Qwen3OmniAudioIO::kAudioFeatureLengths, chunked.audio_feature_lengths);

        std::cout << "[Audio] Running audio encoder inference..." << std::endl;
        areq.infer();

        ov::Tensor encoder_output = areq.get_tensor(Qwen3OmniAudioIO::kAudioFeatures);
        std::cout << "[Audio]   raw encoder output shape: ";
        for (size_t d = 0; d < encoder_output.get_shape().size(); ++d) {
            if (d) std::cout << "x";
            std::cout << encoder_output.get_shape()[d];
        }
        std::cout << std::endl;

        // Step 4: Extract valid tokens from chunked output
        ov::Tensor audio_features = extract_valid_audio_tokens(encoder_output, chunked.cnn_output_lengths);
        const auto af_shape = audio_features.get_shape();  // [total_tokens, output_dim]
        const int64_t num_audio_tokens = static_cast<int64_t>(af_shape[0]);
        const int64_t audio_hidden = static_cast<int64_t>(af_shape[1]);
        std::cout << "[Audio]   audio features after extraction: [" << num_audio_tokens
                  << ", " << audio_hidden << "]" << std::endl;

        // Release audio encoder resources to free memory before vision encoder
        encoder_output = {};
        areq = {};
        compiled_audio = {};
        audio_encoder_model.reset();
        chunked = {};
        mel = {};
        waveform.clear();
        waveform.shrink_to_fit();
        std::cout << "[Audio] Audio encoder resources released." << std::endl;
        const auto t_audio_end = std::chrono::steady_clock::now();
        perf.audio_encode_ms = elapsed_ms(t_audio_start, t_audio_end);

        // Step 3: Compute expected token count & build prompt
        const int64_t expected_tokens = get_audio_token_count(T_frames);
        std::cout << "[Audio]   expected audio tokens from formula: " << expected_tokens << std::endl;
        if (num_audio_tokens != expected_tokens) {
            std::cout << "[Audio]   WARNING: encoder produced " << num_audio_tokens
                      << " tokens, formula expected " << expected_tokens
                      << ". Using encoder output count." << std::endl;
        }

        // ==========================================
        // If image and/or video is also present, run vision encoder
        // ==========================================
        ov::Tensor visual_embeds;
        std::vector<ov::Tensor> ds_embeds;
        int64_t img_tok = 0;
        int64_t vid_tok = 0;
        ov::Tensor combined_grid_thw;
        if (has_image || has_video) {
            const auto t_vis_start = std::chrono::steady_clock::now();
            auto compiled_vision = core.compile_model(vision_model, device, props);
            const ov::Tensor pos_embed = source.get_tensor(resolve_pos_embed_name(source));
            Qwen3OmniVisionPreprocessConfig pre_cfg;
            const auto pre_cfg_path = model_dir / "preprocessor_config.json";
            if (std::filesystem::exists(pre_cfg_path))
                pre_cfg = Qwen3OmniVisionPreprocessConfig::from_json_file(pre_cfg_path);
            Qwen3OmniVisionPreprocessor preprocessor(cfg, pre_cfg);

            // Lambda: run vision encoder on preprocessed inputs
            struct VisionOut { ov::Tensor embeds; std::vector<ov::Tensor> ds; };
            auto run_vision_encoder = [&](const Qwen3OmniVisionInputs& vis) -> VisionOut {
                auto vreq = compiled_vision.create_infer_request();
                vreq.set_tensor(Qwen3VLVisionIO::kPixelValues, vis.pixel_values);
                vreq.set_tensor(Qwen3VLVisionIO::kGridThw, vis.grid_thw);
                vreq.set_tensor(Qwen3VLVisionIO::kPosEmbeds, vis.pos_embeds);
                vreq.set_tensor(Qwen3VLVisionIO::kRotaryCos, vis.rotary_cos);
                vreq.set_tensor(Qwen3VLVisionIO::kRotarySin, vis.rotary_sin);
                vreq.set_tensor("attention_mask", build_vision_attention_mask(vis.grid_thw));
                vreq.infer();
                VisionOut out;
                out.embeds = vreq.get_tensor(Qwen3VLVisionIO::kVisualEmbeds);
                for (size_t i = 0; i < vl_cfg.vision.deepstack_visual_indexes.size(); ++i)
                    out.ds.push_back(vreq.get_tensor(
                        std::string(Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i)));
                return out;
            };

            ov::Tensor img_embeds, vid_embeds;
            std::vector<ov::Tensor> img_ds, vid_ds;
            ov::Tensor img_grid, vid_grid;

            if (has_image) {
                std::cout << "[Audio+Vision] Processing image: " << image_path.string() << std::endl;
                ov::Tensor image = utils::load_image(image_path);
                auto vis_in_img = preprocessor.preprocess(image, pos_embed);
                auto out = run_vision_encoder(vis_in_img);
                img_embeds = out.embeds;
                img_ds = out.ds;
                img_grid = vis_in_img.grid_thw;
                img_tok = Qwen3OmniVisionPreprocessor::count_visual_tokens(
                    img_grid, vl_cfg.vision.spatial_merge_size);
                std::cout << "[Audio+Vision]   image visual tokens: " << img_tok << std::endl;
            }
            if (has_video) {
                std::cout << "[Audio+Video] Processing video frames: " << video_path.string() << std::endl;
                auto frames = load_video_frames(video_path);
                std::cout << "[Audio+Video]   loaded " << frames.size() << " frames" << std::endl;
                auto vis_in_vid = preprocessor.preprocess_video(frames, pos_embed);
                auto out = run_vision_encoder(vis_in_vid);
                vid_embeds = out.embeds;
                vid_ds = out.ds;
                vid_grid = vis_in_vid.grid_thw;
                vid_tok = Qwen3OmniVisionPreprocessor::count_visual_tokens(
                    vid_grid, vl_cfg.vision.spatial_merge_size);
                std::cout << "[Audio+Video]   video visual tokens: " << vid_tok << std::endl;
            }

            // Combine visual outputs (image first, then video — matching prompt order)
            if (has_image && has_video) {
                visual_embeds = concat_tensors_dim0(img_embeds, vid_embeds);
                for (size_t i = 0; i < img_ds.size(); ++i)
                    ds_embeds.push_back(concat_tensors_dim0(img_ds[i], vid_ds[i]));
                combined_grid_thw = concat_tensors_dim0(img_grid, vid_grid);
            } else if (has_image) {
                visual_embeds = img_embeds;
                ds_embeds = img_ds;
                combined_grid_thw = img_grid;
            } else {
                visual_embeds = vid_embeds;
                ds_embeds = vid_ds;
                combined_grid_thw = vid_grid;
            }
            perf.vision_encode_ms = elapsed_ms(t_vis_start, std::chrono::steady_clock::now());
        }

        auto prompt_str = build_audio_prompt(user_prompt, num_audio_tokens, img_tok, vid_tok);
        auto tokenized = tokenizer.encode(prompt_str, ov::genai::add_special_tokens(false));
        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;

        // Replace video_token_id with image_token_id so the planner recognizes them
        if (has_video) {
            auto* ids = input_ids.data<int64_t>();
            const size_t n = input_ids.get_size();
            for (size_t i = 0; i < n; ++i) {
                if (ids[i] == static_cast<int64_t>(cfg.video_token_id))
                    ids[i] = static_cast<int64_t>(cfg.image_token_id);
            }
        }

        const size_t batch = input_ids.get_shape()[0];
        const size_t seq_len = input_ids.get_shape()[1];
        const size_t hidden = static_cast<size_t>(vl_cfg.text.hidden_size);

        // Step 4: Build audio_pos_mask (where input_ids == audio_token_id)
        const int64_t audio_token_id = cfg.audio_token_id;  // 151675
        audio_pos_mask = ov::Tensor(ov::element::boolean, {batch, seq_len});
        auto* mask_data = audio_pos_mask.data<bool>();
        const auto* id_data = input_ids.data<const int64_t>();
        int64_t audio_pad_count = 0;
        for (size_t i = 0; i < batch * seq_len; ++i) {
            mask_data[i] = (id_data[i] == audio_token_id);
            if (mask_data[i]) ++audio_pad_count;
        }
        std::cout << "[Audio]   audio_pad tokens in prompt: " << audio_pad_count << std::endl;

        // Step 5: Scatter audio features into padded tensor
        audio_features_padded = make_zero_tensor(ov::element::f32, {batch, seq_len, hidden});
        {
            auto* dst = audio_features_padded.data<float>();
            const auto* src = audio_features.data<const float>();
            size_t af_idx = 0;
            for (size_t i = 0; i < batch * seq_len; ++i) {
                if (mask_data[i] && af_idx < static_cast<size_t>(num_audio_tokens)) {
                    std::memcpy(dst + i * hidden, src + af_idx * static_cast<size_t>(audio_hidden),
                                std::min(hidden, static_cast<size_t>(audio_hidden)) * sizeof(float));
                    ++af_idx;
                }
            }
            if (static_cast<int64_t>(af_idx) != num_audio_tokens) {
                std::cout << "[Audio]   WARNING: scattered " << af_idx
                          << " of " << num_audio_tokens << " audio features" << std::endl;
            }
        }

        // Step 6: Build position_ids using Qwen3VLInputPlanner
        Qwen3VLInputPlanner planner(vl_cfg);
        if (has_image || has_video) {
            // Audio+Vision: use grid_thw for mRoPE spatial positions
            auto plan = planner.build_plan(input_ids, &attention_mask, &combined_grid_thw);
            position_ids = plan.position_ids;
            visual_pos_mask = plan.visual_pos_mask;
            rope_deltas = plan.rope_deltas;

            // Scatter visual embeddings into padded tensor
            visual_padded = Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, visual_pos_mask);
            deepstack_padded = Qwen3VLInputPlanner::scatter_deepstack_embeds(ds_embeds, visual_pos_mask);
            std::cout << "[Audio+Vision] Visual embeddings scattered." << std::endl;
        } else {
            // Audio-only: no visual positions
            visual_pos_mask = make_zero_tensor(ov::element::boolean, {batch, seq_len});
            auto plan = planner.build_plan(input_ids, &attention_mask, nullptr);
            position_ids = plan.position_ids;
            rope_deltas = plan.rope_deltas;

            visual_padded = make_zero_tensor(ov::element::f32, {batch, seq_len, hidden});
            deepstack_padded.clear();
            const size_t expected_ds = vl_cfg.vision.deepstack_visual_indexes.size();
            for (size_t i = 0; i < expected_ds; ++i) {
                deepstack_padded.push_back(make_zero_tensor(ov::element::f32, {batch, seq_len, hidden}));
            }
        }

        std::cout << "[Audio] Native C++ audio pipeline complete." << std::endl;
        std::cout << "[Audio]   input_ids shape: [" << batch << ", " << seq_len << "]" << std::endl;
        std::cout << "[Audio]   position_ids shape: " << position_ids.get_shape() << std::endl;
    } else if (has_image || has_video) {
        // ==========================================
        // Vision-only path (image and/or video, no audio)
        // ==========================================
        const auto t_vis_start = std::chrono::steady_clock::now();
        auto compiled_vision = core.compile_model(vision_model, device, props);
        const ov::Tensor pos_embed = source.get_tensor(resolve_pos_embed_name(source));
        Qwen3OmniVisionPreprocessConfig pre_cfg;
        const auto pre_cfg_path = model_dir / "preprocessor_config.json";
        if (std::filesystem::exists(pre_cfg_path))
            pre_cfg = Qwen3OmniVisionPreprocessConfig::from_json_file(pre_cfg_path);
        Qwen3OmniVisionPreprocessor preprocessor(cfg, pre_cfg);

        struct VisionOut { ov::Tensor embeds; std::vector<ov::Tensor> ds; };
        auto run_vision_encoder = [&](const Qwen3OmniVisionInputs& vis) -> VisionOut {
            auto vreq = compiled_vision.create_infer_request();
            vreq.set_tensor(Qwen3VLVisionIO::kPixelValues, vis.pixel_values);
            vreq.set_tensor(Qwen3VLVisionIO::kGridThw, vis.grid_thw);
            vreq.set_tensor(Qwen3VLVisionIO::kPosEmbeds, vis.pos_embeds);
            vreq.set_tensor(Qwen3VLVisionIO::kRotaryCos, vis.rotary_cos);
            vreq.set_tensor(Qwen3VLVisionIO::kRotarySin, vis.rotary_sin);
            vreq.set_tensor("attention_mask", build_vision_attention_mask(vis.grid_thw));
            vreq.infer();
            VisionOut out;
            out.embeds = vreq.get_tensor(Qwen3VLVisionIO::kVisualEmbeds);
            for (size_t i = 0; i < vl_cfg.vision.deepstack_visual_indexes.size(); ++i)
                out.ds.push_back(vreq.get_tensor(
                    std::string(Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i)));
            return out;
        };

        ov::Tensor img_embeds, vid_embeds;
        std::vector<ov::Tensor> img_ds, vid_ds;
        ov::Tensor img_grid, vid_grid;
        int64_t img_tok = 0, vid_tok = 0;

        if (has_image) {
            std::cout << "[Vision] Processing image: " << image_path.string() << std::endl;
            ov::Tensor image = utils::load_image(image_path);
            auto vis_in_img = preprocessor.preprocess(image, pos_embed);
            auto out = run_vision_encoder(vis_in_img);
            img_embeds = out.embeds;
            img_ds = out.ds;
            img_grid = vis_in_img.grid_thw;
            img_tok = Qwen3OmniVisionPreprocessor::count_visual_tokens(
                img_grid, vl_cfg.vision.spatial_merge_size);
            std::cout << "[Vision]   image visual tokens: " << img_tok << std::endl;
        }
        if (has_video) {
            std::cout << "[Vision] Processing video frames: " << video_path.string() << std::endl;
            auto frames = load_video_frames(video_path);
            std::cout << "[Vision]   loaded " << frames.size() << " frames" << std::endl;
            auto vis_in_vid = preprocessor.preprocess_video(frames, pos_embed);
            auto out = run_vision_encoder(vis_in_vid);
            vid_embeds = out.embeds;
            vid_ds = out.ds;
            vid_grid = vis_in_vid.grid_thw;
            vid_tok = Qwen3OmniVisionPreprocessor::count_visual_tokens(
                vid_grid, vl_cfg.vision.spatial_merge_size);
            std::cout << "[Vision]   video visual tokens: " << vid_tok << std::endl;
        }
        perf.vision_encode_ms = elapsed_ms(t_vis_start, std::chrono::steady_clock::now());

        // Combine visual outputs
        ov::Tensor visual_embeds, combined_grid_thw;
        std::vector<ov::Tensor> ds_embeds;
        if (has_image && has_video) {
            visual_embeds = concat_tensors_dim0(img_embeds, vid_embeds);
            for (size_t i = 0; i < img_ds.size(); ++i)
                ds_embeds.push_back(concat_tensors_dim0(img_ds[i], vid_ds[i]));
            combined_grid_thw = concat_tensors_dim0(img_grid, vid_grid);
        } else if (has_image) {
            visual_embeds = img_embeds;
            ds_embeds = img_ds;
            combined_grid_thw = img_grid;
        } else {
            visual_embeds = vid_embeds;
            ds_embeds = vid_ds;
            combined_grid_thw = vid_grid;
        }

        // Build prompt & tokenize
        std::string prompt_str;
        if (has_image && !has_video) {
            prompt_str = build_prompt(user_prompt, img_tok);
        } else if (!has_image && has_video) {
            prompt_str = build_video_prompt(user_prompt, vid_tok);
        } else {
            // Image + Video (no audio): image first, then video
            prompt_str = "<|im_start|>user\n<|vision_start|>";
            for (int64_t i = 0; i < img_tok; ++i) prompt_str += "<|image_pad|>";
            prompt_str += "<|vision_end|><|vision_start|>";
            for (int64_t i = 0; i < vid_tok; ++i) prompt_str += "<|video_pad|>";
            prompt_str += "<|vision_end|>";
            prompt_str += user_prompt;
            prompt_str += "<|im_end|>\n<|im_start|>assistant\n";
        }
        auto tokenized = tokenizer.encode(prompt_str, ov::genai::add_special_tokens(false));
        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;

        // Replace video_token_id with image_token_id so the planner recognizes them
        if (has_video) {
            auto* ids = input_ids.data<int64_t>();
            const size_t n = input_ids.get_size();
            for (size_t i = 0; i < n; ++i) {
                if (ids[i] == static_cast<int64_t>(cfg.video_token_id))
                    ids[i] = static_cast<int64_t>(cfg.image_token_id);
            }
        }

        // Plan positions with mRoPE offsets
        Qwen3VLInputPlanner planner(vl_cfg);
        auto plan = planner.build_plan(input_ids, &attention_mask, &combined_grid_thw);
        position_ids = plan.position_ids;
        visual_pos_mask = plan.visual_pos_mask;
        rope_deltas = plan.rope_deltas;

        // Scatter visual embeddings
        visual_padded = Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, visual_pos_mask);
        deepstack_padded = Qwen3VLInputPlanner::scatter_deepstack_embeds(ds_embeds, visual_pos_mask);
    } else {
        // Text-only mode
        auto tokenized = tokenizer.encode(user_prompt, ov::genai::add_special_tokens(false));
        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;
        position_ids = make_text_position_ids_i64(static_cast<size_t>(input_ids.get_shape()[1]));
    }

    const size_t batch = input_ids.get_shape()[0];
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape()[1]);

    ov::Tensor beam_idx(ov::element::i32, {batch});
    {
        auto* b = beam_idx.data<int32_t>();
        for (size_t i = 0; i < batch; ++i) b[i] = static_cast<int32_t>(i);
    }

    auto treq = compiled_text.create_infer_request();
    treq.reset_state();
    treq.set_tensor(Qwen3OmniTextIO::kInputIds, input_ids);
    treq.set_tensor(Qwen3OmniTextIO::kAttentionMask, attention_mask);
    treq.set_tensor(Qwen3OmniTextIO::kPositionIds, position_ids);
    treq.set_tensor(Qwen3OmniTextIO::kBeamIdx, beam_idx);

    if (has_multimodal) {
        treq.set_tensor(Qwen3OmniTextIO::kVisualEmbeds, visual_padded);
        treq.set_tensor(Qwen3OmniTextIO::kVisualPosMask, visual_pos_mask);
        if (audio_features_padded) {
            treq.set_tensor(Qwen3OmniTextIO::kAudioFeatures, audio_features_padded);
            treq.set_tensor(Qwen3OmniTextIO::kAudioPosMask, audio_pos_mask);
        } else {
            treq.set_tensor(Qwen3OmniTextIO::kAudioFeatures,
                make_zero_tensor(ov::element::f32,
                                 {batch, input_ids.get_shape()[1],
                                  static_cast<size_t>(vl_cfg.text.hidden_size)}));
            treq.set_tensor(Qwen3OmniTextIO::kAudioPosMask,
                make_zero_tensor(ov::element::boolean, {batch, input_ids.get_shape()[1]}));
        }
        for (size_t i = 0; i < deepstack_padded.size(); ++i) {
            treq.set_tensor(
                std::string(Qwen3OmniTextIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i),
                deepstack_padded[i]);
        }
    }

    const auto t_prefill_start = std::chrono::steady_clock::now();
    treq.infer();
    const auto t_prefill_end = std::chrono::steady_clock::now();
    perf.ttft_ms = elapsed_ms(t_prefill_start, t_prefill_end);

    int64_t next_id = argmax_last_token(treq.get_tensor(Qwen3OmniTextIO::kLogits));
    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    generated.push_back(next_id);

    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    {
        auto* m = step_mask.data<int64_t>();
        for (size_t i = 0; i < batch; ++i) m[i] = 1;
    }

    // Prepare decode-step zero tensors for multimodal inputs
    ov::Tensor dec_vis, dec_vis_mask, dec_audio, dec_audio_mask;
    std::vector<ov::Tensor> dec_ds;
    if (has_multimodal) {
        const size_t hs = static_cast<size_t>(vl_cfg.text.hidden_size);
        dec_vis = make_zero_tensor(ov::element::f32, {batch, 1, hs});
        dec_vis_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
        dec_audio = make_zero_tensor(ov::element::f32, {batch, 1, hs});
        dec_audio_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
        for (size_t i = 0; i < deepstack_padded.size(); ++i)
            dec_ds.push_back(make_zero_tensor(ov::element::f32, {batch, 1, hs}));
    }

    int64_t past_len = prompt_len;
    const int64_t eos_id = tokenizer.get_eos_token_id();
    const auto t_decode_start = std::chrono::steady_clock::now();
    for (int step = 1; step < max_new_tokens; ++step) {
        if (eos_id >= 0 && next_id == eos_id) break;
        {
            auto* s = step_ids.data<int64_t>();
            for (size_t i = 0; i < batch; ++i) s[i] = next_id;
        }

        treq.set_tensor(Qwen3OmniTextIO::kInputIds, step_ids);
        treq.set_tensor(Qwen3OmniTextIO::kAttentionMask, step_mask);
        treq.set_tensor(Qwen3OmniTextIO::kBeamIdx, beam_idx);

        if (has_multimodal) {
            treq.set_tensor(Qwen3OmniTextIO::kPositionIds,
                Qwen3VLInputPlanner::build_decode_position_ids(rope_deltas, past_len, 1));
            treq.set_tensor(Qwen3OmniTextIO::kVisualEmbeds, dec_vis);
            treq.set_tensor(Qwen3OmniTextIO::kVisualPosMask, dec_vis_mask);
            treq.set_tensor(Qwen3OmniTextIO::kAudioFeatures, dec_audio);
            treq.set_tensor(Qwen3OmniTextIO::kAudioPosMask, dec_audio_mask);
            for (size_t i = 0; i < dec_ds.size(); ++i)
                treq.set_tensor(
                    std::string(Qwen3OmniTextIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i),
                    dec_ds[i]);
        } else {
            treq.set_tensor(Qwen3OmniTextIO::kPositionIds, make_decode_position_ids_i64(past_len));
        }

        treq.infer();
        next_id = argmax_last_token(treq.get_tensor(Qwen3OmniTextIO::kLogits));
        generated.push_back(next_id);
        past_len += 1;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    const int64_t decode_steps = static_cast<int64_t>(generated.size()) - 1;  // excluding prefill token
    perf.prompt_tokens = prompt_len;
    perf.output_tokens = static_cast<int64_t>(generated.size());
    perf.decode_ms = elapsed_ms(t_decode_start, t_decode_end);
    perf.tpot_ms = decode_steps > 0 ? (perf.decode_ms / static_cast<double>(decode_steps)) : 0.0;
    perf.throughput = (decode_steps > 0 && perf.decode_ms > 0.0)
                          ? (static_cast<double>(decode_steps) * 1000.0 / perf.decode_ms)
                          : 0.0;
    perf.text_gen_total_ms = elapsed_ms(t_start, t_decode_end);

    perf.text = tokenizer.decode(generated, ov::genai::skip_special_tokens(true));
    return perf;
}

struct TtsRunResult {
    std::filesystem::path wav_path;
    size_t samples = 0;
    int sample_rate = 24000;
    std::string backend = "speech_decoder";
    std::string note;
    // Sub-component timing (ms)
    double model_compile_ms = 0.0;     // all model creation + compile
    double talker_prefill_ms = 0.0;    // embedding + prefill infer
    double talker_decode_ms = 0.0;     // AR generation loop (talker decode steps only)
    double code_predictor_ms = 0.0;    // AR generation loop (code predictor steps only)
    double speech_decoder_ms = 0.0;    // speech decoder infer
    int codec_frames = 0;             // number of codec frames generated
};

TtsRunResult synthesize_fallback_tone(const std::filesystem::path& wav_out, const std::string& text) {
    const int sample_rate = 24000;
    const float seconds = 1.5f;
    const size_t sample_count = static_cast<size_t>(sample_rate * seconds);
    std::vector<float> audio(sample_count);
    const int hash = static_cast<int>(std::hash<std::string>{}(text) % 5000);
    const float freq = 220.0f + static_cast<float>(hash % 660);
    const float pi = 3.1415926535f;
    for (size_t i = 0; i < sample_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        audio[i] = 0.2f * std::sin(2.0f * pi * freq * t);
    }
    write_wav(wav_out.string(), audio.data(), audio.size(), sample_rate);

    TtsRunResult result;
    result.wav_path = wav_out;
    result.samples = audio.size();
    result.sample_rate = sample_rate;
    result.backend = "fallback_tone";
    result.note = "speech decoder unavailable; generated deterministic tone for e2e comparison";
    return result;
}

// ---- Weight source wrapper: provides talker.model.text_embedding.weight ----
// In Qwen3-Omni, the talker's text_embedding is tied to the thinker's embed_tokens.
class OmniTalkerWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    OmniTalkerWeightSource(ov::genai::modeling::weights::WeightSource& base)
        : base_(base) {}

    std::vector<std::string> keys() const override {
        auto k = base_.keys();
        // Add virtual text_embedding mapped from thinker embed_tokens
        if (base_.has("thinker.model.embed_tokens.weight")) {
            k.push_back("talker.model.text_embedding.weight");
        }
        return k;
    }
    bool has(const std::string& name) const override {
        if (name == "talker.model.text_embedding.weight") {
            return base_.has("thinker.model.embed_tokens.weight");
        }
        return base_.has(name);
    }
    const ov::Tensor& get_tensor(const std::string& name) const override {
        if (name == "talker.model.text_embedding.weight") {
            return base_.get_tensor("thinker.model.embed_tokens.weight");
        }
        return base_.get_tensor(name);
    }
    void release_tensor(const std::string& name) override {
        if (name == "talker.model.text_embedding.weight") {
            base_.release_tensor("thinker.model.embed_tokens.weight");
            return;
        }
        base_.release_tensor(name);
    }
private:
    ov::genai::modeling::weights::WeightSource& base_;
};

// ---- Token sampling for talker codec generation ----
int64_t sample_codec_token(const float* logits, size_t vocab_size,
                           float temperature, size_t top_k, float top_p,
                           float rep_penalty,
                           const std::vector<int64_t>* history,
                           const std::vector<int64_t>* suppress_tokens,
                           std::mt19937& rng) {
    std::vector<float> adj(logits, logits + vocab_size);

    if (history && rep_penalty > 1.0f) {
        for (int64_t tok : *history) {
            if (tok >= 0 && static_cast<size_t>(tok) < vocab_size) {
                adj[tok] = (adj[tok] > 0) ? adj[tok] / rep_penalty : adj[tok] * rep_penalty;
            }
        }
    }
    if (suppress_tokens) {
        for (int64_t tok : *suppress_tokens) {
            if (tok >= 0 && static_cast<size_t>(tok) < vocab_size) {
                adj[tok] = -1e9f;
            }
        }
    }

    std::vector<size_t> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&adj](size_t a, size_t b) { return adj[a] > adj[b]; });

    size_t eff_k = std::min(top_k, vocab_size);
    float max_l = adj[idx[0]];
    std::vector<float> probs(eff_k);
    float sum = 0;
    for (size_t i = 0; i < eff_k; ++i) {
        probs[i] = std::exp((adj[idx[i]] - max_l) / temperature);
        sum += probs[i];
    }
    for (size_t i = 0; i < eff_k; ++i) probs[i] /= sum;

    float cumsum = 0;
    size_t cutoff = eff_k;
    for (size_t i = 0; i < eff_k; ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }

    float top_sum = 0;
    for (size_t i = 0; i < cutoff; ++i) top_sum += probs[i];

    std::uniform_real_distribution<float> dist(0.0f, top_sum);
    float r = dist(rng);
    cumsum = 0;
    for (size_t i = 0; i < cutoff; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) return static_cast<int64_t>(idx[i]);
    }
    return static_cast<int64_t>(idx[0]);
}

// ---- mRoPE position helper ----
std::vector<int64_t> make_mrope_positions(size_t start, size_t len, size_t batch) {
    std::vector<int64_t> pos(3 * batch * len);
    for (size_t i = 0; i < len; ++i) {
        pos[i] = static_cast<int64_t>(start + i);
        pos[batch * len + i] = static_cast<int64_t>(start + i);
        pos[2 * batch * len + i] = static_cast<int64_t>(start + i);
    }
    return pos;
}
ov::Tensor make_causal_mask(size_t seq_len, size_t batch) {
    ov::Tensor m(ov::element::f32, {batch, 1, seq_len, seq_len});
    float* d = m.data<float>();
    for (size_t b = 0; b < batch; ++b)
        for (size_t i = 0; i < seq_len; ++i)
            for (size_t j = 0; j < seq_len; ++j)
                d[b * seq_len * seq_len + i * seq_len + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
    return m;
}
ov::Tensor make_decode_mask(size_t past_len, size_t batch) {
    size_t total = past_len + 1;
    ov::Tensor m(ov::element::f32, {batch, 1, 1, total});
    std::fill_n(m.data<float>(), batch * total, 0.0f);
    return m;
}

// ---- The full talker + code_predictor + speech_decoder pipeline ----
TtsRunResult run_min_tts(const std::filesystem::path& model_dir,
                         ov::Core& core,
                         ov::genai::modeling::weights::WeightSource& source,
                         ov::genai::modeling::weights::WeightFinalizer& finalizer,
                         const Qwen3OmniConfig& cfg,
                         const std::string& text,
                         const std::filesystem::path& wav_out,
                         const std::string& device,
                         PrecisionMode precision_mode = PrecisionMode::kFP32) {
    try {
        // TTS models (talker, code predictor, speech decoder) always use fp32
        // regardless of the user-specified precision, which only applies to the
        // text generation (thinker) model.  Using reduced precision (e.g. int8 KV)
        // in the talker degrades codec generation quality and produces distorted audio.
        ov::AnyMap tts_props = {{ov::hint::inference_precision.name(), ov::element::f32}};
        std::cerr << "[TTS] precision_mode (text model): " << precision_mode_to_string(precision_mode)
                  << " | TTS models: fp32\n";
        auto talker_cfg = to_qwen3_omni_talker_config(cfg);
        auto cp_cfg = to_qwen3_omni_code_predictor_config(cfg);

        // Wrap source to provide talker.model.text_embedding.weight → thinker embed_tokens
        OmniTalkerWeightSource talker_source(source);

        std::cerr << "[TTS] talker hidden=" << talker_cfg.hidden_size
                  << " layers=" << talker_cfg.num_hidden_layers
                  << " vocab=" << talker_cfg.vocab_size
                  << " text_hidden=" << talker_cfg.text_hidden_size << "\n";

        // --- Create and compile all models ---
        const auto t_compile_start = std::chrono::steady_clock::now();
        std::cerr << "[TTS] Creating talker embedding model...\n";
        auto embed_model = create_qwen3_omni_talker_embedding_model(cfg, talker_source, finalizer);
        auto embed_compiled = core.compile_model(embed_model, device, tts_props);
        auto embed_infer = embed_compiled.create_infer_request();

        std::cerr << "[TTS] Creating talker prefill model...\n";
        auto prefill_model = create_qwen3_omni_talker_prefill_model(cfg, talker_source, finalizer);
        auto prefill_compiled = core.compile_model(prefill_model, device, tts_props);
        auto prefill_infer = prefill_compiled.create_infer_request();

        std::cerr << "[TTS] Creating talker decode model...\n";
        auto decode_model = create_qwen3_omni_talker_decode_model(cfg, talker_source, finalizer);
        auto decode_compiled = core.compile_model(decode_model, device, tts_props);
        auto decode_infer = decode_compiled.create_infer_request();

        std::cerr << "[TTS] Creating talker codec embedding model...\n";
        auto codec_embed_model = create_qwen3_omni_talker_codec_embedding_model(cfg, talker_source, finalizer);
        auto codec_embed_compiled = core.compile_model(codec_embed_model, device, tts_props);
        auto codec_embed_infer = codec_embed_compiled.create_infer_request();

        // Code predictor AR models (15 steps) and single codec embed models
        const int cp_steps = std::max(1, cp_cfg.num_code_groups - 1);
        std::vector<ov::InferRequest> cp_ar_infer;
        std::vector<ov::InferRequest> cp_embed_infer;
        cp_ar_infer.reserve(cp_steps);
        cp_embed_infer.reserve(cp_steps);
        for (int step = 0; step < cp_steps; ++step) {
            std::cerr << "[TTS] Creating code predictor AR step " << step << "...\n";
            auto ar_model = create_qwen3_omni_code_predictor_ar_model(cfg, step, talker_source, finalizer);
            auto ar_c = core.compile_model(ar_model, device, tts_props);
            cp_ar_infer.push_back(ar_c.create_infer_request());

            auto se_model = create_qwen3_omni_code_predictor_single_codec_embed_model(cfg, step, talker_source, finalizer);
            auto se_c = core.compile_model(se_model, device, tts_props);
            cp_embed_infer.push_back(se_c.create_infer_request());
        }

        std::cerr << "[TTS] Creating code predictor codec embedding model...\n";
        auto cp_codec_model = create_qwen3_omni_code_predictor_codec_embed_model(cfg, talker_source, finalizer);
        auto cp_codec_compiled = core.compile_model(cp_codec_model, device, tts_props);
        auto cp_codec_infer = cp_codec_compiled.create_infer_request();

        // Speech decoder (BigVGAN) always runs on CPU: GPU f16 accumulation in
        // SnakeBeta activations (exp/sin²) across 4 upsample stages causes
        // systematic ~10x amplitude loss, producing noise-like audio.
        std::cerr << "[TTS] Creating speech decoder model (always CPU)...\n";
        auto decoder_model = create_qwen3_omni_speech_decoder_model(cfg, source, finalizer);
        auto decoder_compiled = core.compile_model(decoder_model, "CPU", tts_props);
        auto decoder_infer = decoder_compiled.create_infer_request();
        const auto t_compile_end = std::chrono::steady_clock::now();

        // --- Tokenize text ---
        ov::genai::Tokenizer tokenizer(model_dir);
        auto tok_result = tokenizer.encode(text, ov::genai::add_special_tokens(false));
        auto tok_ids_tensor = tok_result.input_ids;
        size_t text_len = tok_ids_tensor.get_shape()[1];
        const int64_t* tok_ptr = tok_ids_tensor.data<int64_t>();
        std::vector<int64_t> text_token_ids(tok_ptr, tok_ptr + text_len);

        // --- Config values ---
        const size_t batch = 1;
        const size_t hidden_size = static_cast<size_t>(talker_cfg.hidden_size);
        const size_t num_layers = static_cast<size_t>(talker_cfg.num_hidden_layers);
        const size_t num_kv_heads = static_cast<size_t>(talker_cfg.num_key_value_heads);
        const size_t head_dim = static_cast<size_t>(talker_cfg.head_dim);
        const size_t vocab_size = static_cast<size_t>(talker_cfg.vocab_size);
        const size_t cp_vocab_size = static_cast<size_t>(cp_cfg.vocab_size);

        const int64_t tts_pad_id = cfg.tts_pad_token_id;
        const int64_t tts_eos_id = cfg.tts_eos_token_id;
        const int64_t codec_bos = talker_cfg.codec_bos_token_id;
        const int64_t codec_eos = talker_cfg.codec_eos_token_id;
        const int64_t codec_pad = talker_cfg.codec_pad_token_id;
        const int64_t codec_nothink = cfg.talker_config_raw.value("codec_nothink_id", 2155);
        const int64_t codec_think_bos = cfg.talker_config_raw.value("codec_think_bos_id", 2156);
        const int64_t codec_think_eos = cfg.talker_config_raw.value("codec_think_eos_id", 2157);

        // --- Pre-compute tts_pad embedding ---
        {
            std::vector<int64_t> tp_text = {tts_pad_id};
            std::vector<int64_t> tp_codec = {0};
            std::vector<float> tp_mask = {0.0f};
            embed_infer.set_tensor("text_input_ids", ov::Tensor(ov::element::i64, {1, 1}, tp_text.data()));
            embed_infer.set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {1, 1}, tp_codec.data()));
            embed_infer.set_tensor("codec_mask", ov::Tensor(ov::element::f32, {1, 1}, tp_mask.data()));
            embed_infer.infer();
        }
        auto tts_pad_embed_tensor = embed_infer.get_tensor("inputs_embeds");
        std::vector<float> tts_pad_embed(hidden_size);
        std::copy(tts_pad_embed_tensor.data<float>(),
                  tts_pad_embed_tensor.data<float>() + hidden_size, tts_pad_embed.begin());

        // --- Resolve speaker ID (default: "f245") ---
        int64_t speaker_id = 2301;  // f245
        if (cfg.talker_config_raw.contains("speaker_id") &&
            cfg.talker_config_raw.at("speaker_id").is_object()) {
            const auto& sid = cfg.talker_config_raw.at("speaker_id");
            if (sid.contains("f245")) speaker_id = sid.at("f245").get<int64_t>();
        }
        const int64_t tts_bos_id = 151672;  // tts_bos token from model tokenizer

        // --- Build prefill input (matching Python talker format) ---
        // Python layout: [role×3, tts_pad×4 + {nothink,think_bos,think_eos,speaker},
        //                 tts_bos + codec_pad, text₀ + codec_bos]  = 9 positions
        std::vector<int64_t> full_text_ids;
        std::vector<int64_t> full_codec_ids;
        std::vector<float> full_codec_mask;

        // 1. Role tokens: <|im_start|>assistant\n (codec_mask=0, text only)
        std::vector<int64_t> role_tokens = {151644, 77091, 198};
        for (auto id : role_tokens) {
            full_text_ids.push_back(id);
            full_codec_ids.push_back(0);
            full_codec_mask.push_back(0.0f);
        }

        // 2. Codec prefix: 4 tokens = {nothink, think_bos, think_eos, speaker_id}
        std::vector<int64_t> codec_prefix = {codec_nothink, codec_think_bos, codec_think_eos, speaker_id};
        for (size_t i = 0; i < codec_prefix.size(); ++i) {
            full_text_ids.push_back(tts_pad_id);
            full_codec_ids.push_back(codec_prefix[i]);
            full_codec_mask.push_back(1.0f);
        }

        // 3. tts_bos + codec_pad
        full_text_ids.push_back(tts_bos_id);
        full_codec_ids.push_back(codec_pad);
        full_codec_mask.push_back(1.0f);

        // 4. First text token + codec_bos (generation start marker)
        full_text_ids.push_back(text_token_ids.empty() ? tts_eos_id : text_token_ids[0]);
        full_codec_ids.push_back(codec_bos);
        full_codec_mask.push_back(1.0f);

        size_t prefill_len = full_text_ids.size();
        std::cerr << "[TTS] Prefill sequence length: " << prefill_len
                  << " (speaker_id=" << speaker_id << " text_tokens=" << text_token_ids.size() << ")\n";

        // --- Pre-compute trailing text embeddings for AR streaming ---
        // Python feeds text₁...textₙ, tts_eos during generation, then tts_pad when exhausted.
        // trailing_text_ids = [text₁, text₂, ..., textₙ, tts_eos]
        std::vector<int64_t> trailing_text_ids;
        for (size_t i = 1; i < text_token_ids.size(); ++i)
            trailing_text_ids.push_back(text_token_ids[i]);
        trailing_text_ids.push_back(tts_eos_id);

        // Pre-compute all trailing text embeddings (embed each token individually)
        std::vector<std::vector<float>> trailing_text_embeds(trailing_text_ids.size());
        for (size_t ti = 0; ti < trailing_text_ids.size(); ++ti) {
            std::vector<int64_t> t_text = {trailing_text_ids[ti]};
            std::vector<int64_t> t_codec = {0};
            std::vector<float> t_mask = {0.0f};  // text-only: codec_mask=0
            embed_infer.set_tensor("text_input_ids", ov::Tensor(ov::element::i64, {1, 1}, t_text.data()));
            embed_infer.set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {1, 1}, t_codec.data()));
            embed_infer.set_tensor("codec_mask", ov::Tensor(ov::element::f32, {1, 1}, t_mask.data()));
            embed_infer.infer();
            auto e = embed_infer.get_tensor("inputs_embeds");
            trailing_text_embeds[ti].assign(e.data<float>(), e.data<float>() + hidden_size);
        }
        std::cerr << "[TTS] Pre-computed " << trailing_text_embeds.size() << " trailing text embeddings\n";

        // --- Get prefill embeddings ---
        const auto t_prefill_start = std::chrono::steady_clock::now();
        {
            ov::Tensor tt(ov::element::i64, {batch, prefill_len}, full_text_ids.data());
            ov::Tensor ct(ov::element::i64, {batch, prefill_len}, full_codec_ids.data());
            ov::Tensor mt(ov::element::f32, {batch, prefill_len}, full_codec_mask.data());
            embed_infer.set_tensor("text_input_ids", tt);
            embed_infer.set_tensor("codec_input_ids", ct);
            embed_infer.set_tensor("codec_mask", mt);
            embed_infer.infer();
        }
        auto prefill_embeds = embed_infer.get_tensor("inputs_embeds");

        // --- Prefill ---
        auto pos_data = make_mrope_positions(0, prefill_len, batch);
        ov::Tensor position_ids(ov::element::i64, {3, batch, prefill_len}, pos_data.data());
        ov::Tensor attn_mask = make_causal_mask(prefill_len, batch);

        // Empty past KV caches
        std::vector<ov::Tensor> past_keys, past_values;
        for (size_t i = 0; i < num_layers; ++i) {
            past_keys.push_back(ov::Tensor(ov::element::f32, {batch, num_kv_heads, 0, head_dim}));
            past_values.push_back(ov::Tensor(ov::element::f32, {batch, num_kv_heads, 0, head_dim}));
        }

        prefill_infer.set_tensor("inputs_embeds", prefill_embeds);
        prefill_infer.set_tensor("position_ids", position_ids);
        prefill_infer.set_tensor("attention_mask", attn_mask);
        for (size_t i = 0; i < num_layers; ++i) {
            prefill_infer.set_tensor("past_key_" + std::to_string(i), past_keys[i]);
            prefill_infer.set_tensor("past_value_" + std::to_string(i), past_values[i]);
        }
        std::cerr << "[TTS] Running talker prefill...\n";
        prefill_infer.infer();
        const auto t_prefill_end = std::chrono::steady_clock::now();

        auto logits_tensor = prefill_infer.get_tensor("logits");
        auto hidden_tensor = prefill_infer.get_tensor("hidden_states");

        std::vector<ov::Tensor> present_keys, present_values;
        for (size_t i = 0; i < num_layers; ++i) {
            present_keys.push_back(prefill_infer.get_tensor("present_key_" + std::to_string(i)));
            present_values.push_back(prefill_infer.get_tensor("present_value_" + std::to_string(i)));
        }

        // --- Build suppress_tokens list ---
        std::vector<int64_t> suppress_tokens;
        int64_t suppress_start = static_cast<int64_t>(vocab_size) - 1024;
        for (int64_t i = suppress_start; i < static_cast<int64_t>(vocab_size); ++i) {
            if (i != codec_eos) suppress_tokens.push_back(i);
        }
        std::vector<int64_t> suppress_with_eos = suppress_tokens;
        suppress_with_eos.push_back(codec_eos);

        // --- Generation loop ---
        std::mt19937 rng(42);
        const float temperature = 0.8f;
        const size_t top_k = 50;
        const float top_p = 0.95f;
        const float rep_penalty = 1.05f;
        // Scale min_frames by text length: ~3 frames per text token, floor 10
        const int min_frames = std::max(10, static_cast<int>(text_token_ids.size()) * 3);
        const int max_frames = 1000;

        std::vector<std::vector<int64_t>> all_layer_tokens(16);

        // Sample first layer0 token from prefill logits
        const float* logits_data = logits_tensor.data<float>() + (prefill_len - 1) * vocab_size;
        int64_t layer0_token = sample_codec_token(logits_data, vocab_size, temperature, top_k, top_p,
                                                   rep_penalty, &all_layer_tokens[0], &suppress_with_eos, rng);
        all_layer_tokens[0].push_back(layer0_token);

        // Get past_hidden (last position)
        const float* hidden_data = hidden_tensor.data<float>() + (prefill_len - 1) * hidden_size;
        std::vector<float> past_hidden(hidden_data, hidden_data + hidden_size);

        // Get layer0 embedding
        {
            std::vector<int64_t> l0_vec = {layer0_token};
            codec_embed_infer.set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {batch, 1}, l0_vec.data()));
            codec_embed_infer.infer();
        }
        auto l0_embed_out = codec_embed_infer.get_tensor("codec_embeds");
        std::vector<float> layer0_embed(l0_embed_out.data<float>(),
                                         l0_embed_out.data<float>() + hidden_size);

        size_t current_seq_len = prefill_len;
        double talker_decode_accum_ms = 0.0;
        double code_predictor_accum_ms = 0.0;

        // Process frames
        for (int frame = 0; frame < max_frames && layer0_token != codec_eos; ++frame) {
            // --- Code predictor: generate layers 1-15 for this frame ---
            const auto t_cp_start = std::chrono::steady_clock::now();
            std::vector<float> ar_seq;
            ar_seq.insert(ar_seq.end(), past_hidden.begin(), past_hidden.end());
            ar_seq.insert(ar_seq.end(), layer0_embed.begin(), layer0_embed.end());

            std::vector<int64_t> current_layer_tokens(cp_steps);
            for (int step = 0; step < cp_steps; ++step) {
                size_t cur_len = ar_seq.size() / hidden_size;
                std::vector<int64_t> pos_ids(cur_len);
                std::iota(pos_ids.begin(), pos_ids.end(), 0);

                ov::Tensor ar_input(ov::element::f32, {batch, cur_len, hidden_size}, ar_seq.data());
                ov::Tensor ar_pos(ov::element::i64, {batch, cur_len}, pos_ids.data());
                cp_ar_infer[step].set_tensor("inputs_embeds", ar_input);
                cp_ar_infer[step].set_tensor("position_ids", ar_pos);
                cp_ar_infer[step].infer();

                auto step_logits = cp_ar_infer[step].get_tensor("logits");
                int64_t layer_token = sample_codec_token(step_logits.data<float>(), cp_vocab_size,
                                                          temperature, top_k, top_p, 1.0f, nullptr, nullptr, rng);
                all_layer_tokens[step + 1].push_back(layer_token);
                current_layer_tokens[step] = layer_token;

                // Get embedding for this token
                std::vector<int64_t> tv = {layer_token};
                cp_embed_infer[step].set_tensor("codec_input", ov::Tensor(ov::element::i64, {batch, 1}, tv.data()));
                cp_embed_infer[step].infer();
                auto le = cp_embed_infer[step].get_tensor("codec_embed");
                ar_seq.insert(ar_seq.end(), le.data<float>(), le.data<float>() + hidden_size);
            }

            // --- Compute combined codec embedding for talker decode input ---
            std::vector<float> codec_sum(hidden_size, 0.0f);
            for (size_t i = 0; i < hidden_size; ++i) codec_sum[i] += layer0_embed[i];

            // Sum embeddings from all 15 code predictor layers
            std::vector<std::vector<int64_t>> layer_tokens_vec(cp_steps);
            std::vector<ov::Tensor> layer_tensors(cp_steps);
            for (int layer = 0; layer < cp_steps; ++layer) {
                layer_tokens_vec[layer] = {current_layer_tokens[layer]};
                layer_tensors[layer] = ov::Tensor(ov::element::i64, {batch, 1}, layer_tokens_vec[layer].data());
                cp_codec_infer.set_tensor("codec_input_" + std::to_string(layer), layer_tensors[layer]);
            }
            cp_codec_infer.infer();
            auto codec_embeds_sum = cp_codec_infer.get_tensor("codec_embeds_sum");
            const float* sum_ptr = codec_embeds_sum.data<float>();
            for (size_t i = 0; i < hidden_size; ++i) codec_sum[i] += sum_ptr[i];

            const auto t_cp_end = std::chrono::steady_clock::now();
            code_predictor_accum_ms += elapsed_ms(t_cp_start, t_cp_end);

            // inputs_embeds = codec_sum + text_conditioning
            // Python streams text tokens: text₁,text₂,...,textₙ,tts_eos, then tts_pad when exhausted
            const auto t_td_start = std::chrono::steady_clock::now();
            const auto& text_cond = (static_cast<size_t>(frame) < trailing_text_embeds.size())
                ? trailing_text_embeds[frame] : tts_pad_embed;
            std::vector<float> step_embed_data(hidden_size);
            for (size_t i = 0; i < hidden_size; ++i)
                step_embed_data[i] = codec_sum[i] + text_cond[i];
            ov::Tensor step_embed(ov::element::f32, {batch, 1, hidden_size}, step_embed_data.data());

            // --- Talker decode step ---
            auto step_pos = make_mrope_positions(current_seq_len, 1, batch);
            ov::Tensor step_positions(ov::element::i64, {3, batch, 1}, step_pos.data());
            ov::Tensor dec_attn_mask = make_decode_mask(current_seq_len, batch);

            decode_infer.set_tensor("inputs_embeds", step_embed);
            decode_infer.set_tensor("position_ids", step_positions);
            decode_infer.set_tensor("attention_mask", dec_attn_mask);
            for (size_t i = 0; i < num_layers; ++i) {
                decode_infer.set_tensor("past_key_" + std::to_string(i), present_keys[i]);
                decode_infer.set_tensor("past_value_" + std::to_string(i), present_values[i]);
            }
            decode_infer.infer();

            logits_tensor = decode_infer.get_tensor("logits");
            hidden_tensor = decode_infer.get_tensor("hidden_states");
            for (size_t i = 0; i < num_layers; ++i) {
                present_keys[i] = decode_infer.get_tensor("present_key_" + std::to_string(i));
                present_values[i] = decode_infer.get_tensor("present_value_" + std::to_string(i));
            }

            logits_data = logits_tensor.data<float>();
            const auto* suppress = (frame < min_frames) ? &suppress_with_eos : &suppress_tokens;
            layer0_token = sample_codec_token(logits_data, vocab_size, temperature, top_k, top_p,
                                               rep_penalty, &all_layer_tokens[0], suppress, rng);
            all_layer_tokens[0].push_back(layer0_token);

            hidden_data = hidden_tensor.data<float>();
            std::copy(hidden_data, hidden_data + hidden_size, past_hidden.begin());

            // Get next layer0 embedding
            {
                std::vector<int64_t> l0v = {layer0_token};
                codec_embed_infer.set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {batch, 1}, l0v.data()));
                codec_embed_infer.infer();
                auto l0e = codec_embed_infer.get_tensor("codec_embeds");
                std::copy(l0e.data<float>(), l0e.data<float>() + hidden_size, layer0_embed.begin());
            }

            current_seq_len++;
            const auto t_td_end = std::chrono::steady_clock::now();
            talker_decode_accum_ms += elapsed_ms(t_td_start, t_td_end);

            if (frame % 20 == 0 || layer0_token == codec_eos) {
                std::string text_info = (static_cast<size_t>(frame) < trailing_text_embeds.size())
                    ? "text_id=" + std::to_string(trailing_text_ids[frame])
                    : "tts_pad";
                std::cerr << "[TTS] Frame " << frame << ": layer0=" << layer0_token
                          << " text_cond=" << text_info
                          << (layer0_token == codec_eos ? " (EOS)" : "") << "\n";
            }
        }

        size_t num_frames = all_layer_tokens[0].size();
        // Layer 0 always has +1 token vs layers 1-15 (sampled after decode but
        // code predictor for the next frame never ran).  Trim to the minimum
        // across all layers so the codes tensor is balanced.
        size_t min_len = num_frames;
        for (int i = 1; i < 16; ++i)
            min_len = std::min(min_len, all_layer_tokens[i].size());
        for (auto& layer : all_layer_tokens)
            while (layer.size() > min_len) layer.pop_back();
        num_frames = min_len;

        // Remove trailing EOS from layer 0 if present
        if (!all_layer_tokens[0].empty() && all_layer_tokens[0].back() == codec_eos) {
            for (auto& layer : all_layer_tokens) {
                if (!layer.empty()) layer.pop_back();
            }
            num_frames = all_layer_tokens[0].size();
        }
        std::cerr << "[TTS] Generation done. Frames: " << num_frames
                  << " (min_frames=" << min_frames << " text_tokens=" << text_token_ids.size() << ")\n";
        // Print first few layer0 tokens for debugging
        if (num_frames > 0) {
            std::cerr << "[TTS] layer0 tokens: ";
            for (size_t i = 0; i < std::min(num_frames, size_t(20)); ++i)
                std::cerr << all_layer_tokens[0][i] << " ";
            if (num_frames > 20) std::cerr << "...";
            std::cerr << "\n";
        }

        if (num_frames == 0) {
            std::cerr << "[TTS] No codec frames generated, falling back to tone.\n";
            return synthesize_fallback_tone(wav_out, text);
        }

        // --- Build codes tensor [1, 16, num_frames] ---
        std::vector<int64_t> codes_flat(16 * num_frames, 0);
        for (int layer = 0; layer < 16; ++layer) {
            const auto& lv = all_layer_tokens[layer];
            for (size_t t = 0; t < std::min(lv.size(), num_frames); ++t) {
                codes_flat[layer * num_frames + t] = lv[t];
            }
        }
        ov::Tensor codes(ov::element::i64, {1, 16, num_frames}, codes_flat.data());

        // --- Run speech decoder ---
        std::cerr << "[TTS] Running speech decoder on " << num_frames << " frames...\n";
        const auto t_decoder_start = std::chrono::steady_clock::now();
        decoder_infer.set_tensor("codes", codes);
        decoder_infer.infer();
        const auto t_decoder_end = std::chrono::steady_clock::now();
        ov::Tensor audio = decoder_infer.get_tensor("audio");

        const float* audio_ptr = audio.data<const float>();
        const size_t sample_count = audio.get_size();
        write_wav(wav_out.string(), audio_ptr, sample_count, 24000);
        std::cerr << "[TTS] Wrote " << sample_count << " audio samples to " << wav_out << "\n";

        TtsRunResult result;
        result.wav_path = wav_out;
        result.samples = sample_count;
        result.sample_rate = 24000;
        result.model_compile_ms = elapsed_ms(t_compile_start, t_compile_end);
        result.talker_prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);
        result.talker_decode_ms = talker_decode_accum_ms;
        result.code_predictor_ms = code_predictor_accum_ms;
        result.speech_decoder_ms = elapsed_ms(t_decoder_start, t_decoder_end);
        result.codec_frames = static_cast<int>(num_frames);
        return result;
    } catch (const std::exception& ex) {
        std::cerr << "[TTS] Speech decoder failed: " << ex.what() << "\n";
        return synthesize_fallback_tone(wav_out, text);
    }
}

}  // namespace

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    // Ensure stdout outputs UTF-8 so Python subprocess can read non-ASCII text correctly
    SetConsoleOutputCP(65001);
#endif
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <CASE_ID> <TEXT_PROMPT> <WAV_OUT> [IMAGE_PATH] [AUDIO_PATH] [DEVICE] [MAX_NEW_TOKENS] [PRECISION] [VIDEO_FRAMES_DIR]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::string case_id = argv[2];
    const std::string text_prompt = argv[3];
    const std::filesystem::path wav_out = argv[4];
    const std::filesystem::path image_path = (argc > 5) ? std::filesystem::path(argv[5]) : std::filesystem::path();
    const std::filesystem::path audio_path = (argc > 6) ? std::filesystem::path(argv[6]) : std::filesystem::path();
    const std::string device = (argc > 7) ? argv[7] : "CPU";
    const int max_new_tokens = (argc > 8) ? std::stoi(argv[8]) : 64;
    const PrecisionMode precision = (argc > 9) ? parse_precision_mode(argv[9]) : PrecisionMode::kFP32;
    const std::filesystem::path video_path = (argc > 10) ? std::filesystem::path(argv[10]) : std::filesystem::path();

    auto st_data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(st_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer;
    Qwen3OmniConfig cfg = Qwen3OmniConfig::from_json_file(model_dir);

    ov::Core core;

    const auto start = std::chrono::high_resolution_clock::now();
    auto text_gen = run_text_generation(model_dir,
                                                    core,
                                                    source,
                                                    finalizer,
                                                    cfg,
                                                    text_prompt,
                                                    image_path,
                                                    audio_path,
                                                    video_path,
                                                    max_new_tokens,
                                                    device,
                                                    precision);
    const auto t_tts_start = std::chrono::high_resolution_clock::now();
    TtsRunResult tts = run_min_tts(model_dir,
                                   core,
                                   source,
                                   finalizer,
                                   cfg,
                                   text_gen.text.empty() ? text_prompt : text_gen.text,
                                   wav_out,
                                   device,
                                   precision);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const auto tts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - t_tts_start).count();

    // Escape newlines so parse_kv_stdout sees TEXT_OUTPUT as a single line
    std::string escaped_text = text_gen.text;
    for (size_t pos = 0; (pos = escaped_text.find('\n', pos)) != std::string::npos; ) {
        escaped_text.replace(pos, 1, "\\n");
        pos += 2;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CASE_ID: " << case_id << "\n";
    std::cout << "DEVICE: " << device << "\n";
    std::cout << "PRECISION_MODE: " << precision_mode_to_string(precision) << "\n";
    std::cout << "TEXT_OUTPUT: " << escaped_text << "\n";
    std::cout << "WAV_OUTPUT: " << wav_out.string() << "\n";
    std::cout << "AUDIO_SAMPLES: " << tts.samples << "\n";
    std::cout << "AUDIO_SAMPLE_RATE: " << tts.sample_rate << "\n";
    std::cout << "TTS_BACKEND: " << tts.backend << "\n";
    std::cout << "TTS_NOTE: " << tts.note << "\n";
    // Performance metrics
    std::cout << "PROMPT_TOKENS: " << text_gen.prompt_tokens << "\n";
    std::cout << "OUTPUT_TOKENS: " << text_gen.output_tokens << "\n";
    std::cout << "MODEL_LOAD_MS: " << text_gen.model_load_ms << "\n";
    std::cout << "AUDIO_ENCODE_MS: " << text_gen.audio_encode_ms << "\n";
    std::cout << "VISION_ENCODE_MS: " << text_gen.vision_encode_ms << "\n";
    std::cout << "TTFT_MS: " << text_gen.ttft_ms << "\n";
    std::cout << "DECODE_MS: " << text_gen.decode_ms << "\n";
    std::cout << "TPOT_MS: " << text_gen.tpot_ms << "\n";
    std::cout << "THROUGHPUT: " << text_gen.throughput << "\n";
    std::cout << "TEXT_GEN_MS: " << text_gen.text_gen_total_ms << "\n";
    std::cout << "TTS_MS: " << tts_ms << "\n";
    // TTS sub-component timing
    std::cout << "TTS_MODEL_COMPILE_MS: " << tts.model_compile_ms << "\n";
    std::cout << "TTS_TALKER_PREFILL_MS: " << tts.talker_prefill_ms << "\n";
    std::cout << "TTS_TALKER_DECODE_MS: " << tts.talker_decode_ms << "\n";
    std::cout << "TTS_CODE_PREDICTOR_MS: " << tts.code_predictor_ms << "\n";
    std::cout << "TTS_SPEECH_DECODER_MS: " << tts.speech_decoder_ms << "\n";
    std::cout << "TTS_CODEC_FRAMES: " << tts.codec_frames << "\n";
    std::cout << "TOTAL_MS: " << total_ms << "\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
}
