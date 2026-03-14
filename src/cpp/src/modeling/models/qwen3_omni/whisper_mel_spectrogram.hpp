// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file whisper_mel_spectrogram.hpp
 *
 * Pure C++ implementation of the WhisperFeatureExtractor mel-spectrogram
 * pipeline (matching HuggingFace transformers).  No Python dependency.
 *
 * Pipeline:
 *   1. Read 16-kHz mono PCM from a WAV file.
 *   2. Reflect-pad → windowed STFT (n_fft=400, hop=160, Hann) → power spectrum.
 *   3. Mel filter-bank (201×128, slaney, 0–8 kHz) → mel spectrogram.
 *   4. log10 → clamp(max-8) → (x+4)/4  normalisation.
 *
 * The output is an ov::Tensor of shape {1, n_mels, T_frames} plus
 * a feature_attention_mask tensor of shape {1, T_frames}.
 */

#include <cstdint>
#include <string>
#include <vector>

#include <openvino/runtime/tensor.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

/// Parameters that mirror preprocessor_config.json for the Whisper feature
/// extractor used in Qwen3-Omni.
struct WhisperFeatureExtractorConfig {
    int32_t sampling_rate = 16000;
    int32_t n_fft         = 400;
    int32_t hop_length    = 160;
    int32_t feature_size  = 128;   // n_mels
    int32_t n_samples     = 480000; // NOT used for padding in our case
    int32_t nb_max_frames = 3000;   // NOT used for padding in our case
    float   padding_value = 0.0f;
    bool    return_attention_mask = true;

    /// Load from preprocessor_config.json (only whisper-relevant fields).
    static WhisperFeatureExtractorConfig from_json_file(const std::filesystem::path& path);
};

/// Result of mel feature extraction.
struct WhisperMelFeatures {
    ov::Tensor input_features;          // f32 {1, n_mels, T_frames}
    ov::Tensor feature_attention_mask;  // i64 {1, T_frames}
    int64_t    audio_feature_length;    // T_frames (valid frames, before any padding)
};

/**
 * Read a 16-kHz mono WAV file and return float32 samples in [-1, 1].
 *
 * Minimal WAV parser — supports PCM 16-bit / 32-bit-float, mono or stereo
 * (stereo is down-mixed).  No dependency on dr_wav.
 */
std::vector<float> read_wav_to_float32(const std::string& path);

/**
 * Compute the Whisper-compatible mel spectrogram for a raw waveform.
 *
 * @param waveform  float32 samples at cfg.sampling_rate (typically 16 kHz)
 * @param cfg       WhisperFeatureExtractorConfig
 * @return          WhisperMelFeatures with input_features and mask
 *
 * This reproduces the Python WhisperFeatureExtractor behaviour:
 *   - reflect-pad waveform by n_fft/2 on each side
 *   - STFT with periodic Hann window, hop_length
 *   - power spectrum |X|^2, drop last time frame
 *   - multiply by slaney mel filter-bank (num_freq_bins × n_mels)
 *   - log10(max(mel, 1e-10))
 *   - clamp(max - 8)
 *   - (x + 4) / 4
 */
WhisperMelFeatures extract_whisper_mel_features(const std::vector<float>& waveform,
                                                const WhisperFeatureExtractorConfig& cfg);

/**
 * Compute the number of audio tokens the audio encoder will produce.
 * Matches the Python ``_get_feat_extract_output_lengths``.
 *
 *   r = T_frames % 100
 *   f = (r - 1) // 2 + 1
 *   N = ((f - 1) // 2 + 1 - 1) // 2 + 1 + (T_frames // 100) * 13
 */
int64_t get_audio_token_count(int64_t T_frames);

/**
 * Compute the CNN-downsampled length after 3× stride-2 convolutions.
 * out = ceil(ceil(ceil(L / 2) / 2) / 2) = ((L + 1) / 2 + 1) / 2 + 1) / 2
 */
int64_t cnn_output_length(int64_t input_length);

/**
 * Split the mel spectrogram into chunks of `chunk_size` frames (default 100),
 * pad to the same length, and prepare batch inputs for the OV audio encoder.
 *
 * @param mel  WhisperMelFeatures with input_features {1, n_mels, T_frames}
 * @param chunk_size  Number of frames per chunk (default: n_window * 2 = 100)
 * @return  Tuple: (chunked_features {num_chunks, n_mels, max_padded_len},
 *                  feature_attention_mask {num_chunks, max_padded_len},
 *                  audio_feature_lengths {num_chunks},
 *                  chunk_lengths - vector of original chunk frame counts,
 *                  total_output_tokens - total expected tokens after encoder)
 */
struct ChunkedMelInput {
    ov::Tensor input_features;           // f32 {num_chunks, n_mels, max_padded_len}
    ov::Tensor feature_attention_mask;   // i64 {num_chunks, max_padded_len}
    ov::Tensor audio_feature_lengths;    // i64 {num_chunks}
    std::vector<int64_t> chunk_lengths;  // original chunk frame counts
    std::vector<int64_t> cnn_output_lengths; // CNN output lengths per chunk
    int64_t total_output_tokens;         // sum of cnn_output_lengths
};

ChunkedMelInput chunk_mel_for_audio_encoder(const WhisperMelFeatures& mel,
                                             int32_t chunk_size = 100);

/**
 * Extract valid audio tokens from batched encoder output.
 *
 * @param encoder_output  f32 {num_chunks, max_seq_after_cnn, hidden_dim}
 * @param cnn_output_lengths  valid token count per chunk
 * @return  f32 {total_tokens, hidden_dim}
 */
ov::Tensor extract_valid_audio_tokens(const ov::Tensor& encoder_output,
                                      const std::vector<int64_t>& cnn_output_lengths);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
