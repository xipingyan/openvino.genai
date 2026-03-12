// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/whisper_mel_spectrogram.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// ============================================================================
// Minimal WAV parser (PCM16 / PCM32float, mono / stereo, 16 kHz).
// ============================================================================

struct WavHeader {
    uint32_t sample_rate   = 0;
    uint16_t num_channels  = 0;
    uint16_t bits_per_sample = 0;
    uint16_t audio_format  = 0;   // 1 = PCM int, 3 = IEEE float
    uint32_t data_size     = 0;
};

template <typename T>
T read_le(std::ifstream& f) {
    T val{};
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

WavHeader parse_wav_header(std::ifstream& f) {
    char riff[4]{}, wave[4]{};
    f.read(riff, 4);
    read_le<uint32_t>(f); // file size
    f.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Not a valid WAV file");
    }

    WavHeader hdr;
    bool got_fmt = false, got_data = false;
    while (f.good() && !(got_fmt && got_data)) {
        char chunk_id[4]{};
        f.read(chunk_id, 4);
        uint32_t chunk_size = read_le<uint32_t>(f);
        std::string id(chunk_id, 4);

        if (id == "fmt ") {
            hdr.audio_format   = read_le<uint16_t>(f);
            hdr.num_channels   = read_le<uint16_t>(f);
            hdr.sample_rate    = read_le<uint32_t>(f);
            read_le<uint32_t>(f); // byte rate
            read_le<uint16_t>(f); // block align
            hdr.bits_per_sample = read_le<uint16_t>(f);
            // skip any extra fmt bytes
            if (chunk_size > 16) {
                f.seekg(chunk_size - 16, std::ios::cur);
            }
            got_fmt = true;
        } else if (id == "data") {
            hdr.data_size = chunk_size;
            got_data = true;
            // current position is start of audio data
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }
    if (!got_fmt || !got_data) {
        throw std::runtime_error("WAV file missing fmt or data chunk");
    }
    return hdr;
}

// ============================================================================
// Slaney mel filter-bank (matches HuggingFace audio_utils.mel_filter_bank).
// ============================================================================

/// Hz → mel (Slaney / O'Shaughnessy scale)
inline float hz_to_mel_slaney(float hz) {
    constexpr float kBreak = 1000.0f;
    constexpr float kLinSlope = 3.0f / 200.0f;  // = 0.015
    if (hz < kBreak) {
        return kLinSlope * hz;
    }
    return 15.0f + std::log(hz / kBreak) * (27.0f / std::log(6.4f));
}

/// mel → Hz (inverse)
inline float mel_to_hz_slaney(float mel) {
    constexpr float kBreak = 1000.0f;
    constexpr float kLinSlope = 3.0f / 200.0f;
    constexpr float kMelBreak = kLinSlope * kBreak;  // = 15.0
    if (mel < kMelBreak) {
        return mel / kLinSlope;
    }
    return kBreak * std::exp((mel - 15.0f) * std::log(6.4f) / 27.0f);
}

/**
 * Build the mel filter-bank matrix of shape (num_freq_bins, num_mel_filters).
 * Slaney norm + Slaney mel scale, matching HuggingFace exactly.
 */
std::vector<float> build_mel_filter_bank(int32_t num_freq_bins,
                                          int32_t num_mel_filters,
                                          float min_frequency,
                                          float max_frequency,
                                          int32_t sampling_rate) {
    const float mel_min = hz_to_mel_slaney(min_frequency);
    const float mel_max = hz_to_mel_slaney(max_frequency);
    const int32_t num_points = num_mel_filters + 2;

    // Evenly-spaced mel points → Hz
    std::vector<float> filter_freqs(static_cast<size_t>(num_points));
    for (int32_t i = 0; i < num_points; ++i) {
        float mel = mel_min + (mel_max - mel_min) * static_cast<float>(i) / static_cast<float>(num_points - 1);
        filter_freqs[static_cast<size_t>(i)] = mel_to_hz_slaney(mel);
    }

    // FFT bin frequencies: linearly spaced 0 .. sr/2
    std::vector<float> fft_freqs(static_cast<size_t>(num_freq_bins));
    const float half_sr = static_cast<float>(sampling_rate) / 2.0f;
    for (int32_t i = 0; i < num_freq_bins; ++i) {
        fft_freqs[static_cast<size_t>(i)] = half_sr * static_cast<float>(i) / static_cast<float>(num_freq_bins - 1);
    }

    // Triangular filter bank  (output: [num_freq_bins, num_mel_filters])
    std::vector<float> bank(static_cast<size_t>(num_freq_bins) * static_cast<size_t>(num_mel_filters), 0.0f);

    for (int32_t m = 0; m < num_mel_filters; ++m) {
        const float f_left   = filter_freqs[static_cast<size_t>(m)];
        const float f_center = filter_freqs[static_cast<size_t>(m + 1)];
        const float f_right  = filter_freqs[static_cast<size_t>(m + 2)];
        const float diff_left  = f_center - f_left;
        const float diff_right = f_right - f_center;

        // Slaney normalisation (area normalisation)
        const float enorm = 2.0f / (f_right - f_left);

        for (int32_t f = 0; f < num_freq_bins; ++f) {
            const float freq = fft_freqs[static_cast<size_t>(f)];
            float up_slope   = (freq - f_left) / std::max(diff_left, 1e-10f);
            float down_slope = (f_right - freq) / std::max(diff_right, 1e-10f);
            float val = std::max(0.0f, std::min(up_slope, down_slope));
            bank[static_cast<size_t>(f) * static_cast<size_t>(num_mel_filters) + static_cast<size_t>(m)] = val * enorm;
        }
    }

    return bank;
}

// ============================================================================
// Periodic Hann window
// ============================================================================

std::vector<float> make_hann_window(int32_t length) {
    // Periodic Hann window (same as torch.hann_window(length))
    std::vector<float> w(static_cast<size_t>(length));
    for (int32_t i = 0; i < length; ++i) {
        w[static_cast<size_t>(i)] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(length)));
    }
    return w;
}

// ============================================================================
// Simple DFT (for small n_fft=400 this is fine; ~80K real multiplies per frame)
// ============================================================================

/**
 * Real-valued DFT of length N.  Returns N/2+1 complex values (the positive
 * frequency half).  Uses the naive O(N^2) algorithm which is perfectly
 * acceptable for N <= 512.
 */
void real_dft(const float* input, int32_t N, std::vector<std::complex<float>>& out) {
    const int32_t K = N / 2 + 1;
    out.resize(static_cast<size_t>(K));
    for (int32_t k = 0; k < K; ++k) {
        float re = 0.0f, im = 0.0f;
        for (int32_t n = 0; n < N; ++n) {
            const float angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(k) * static_cast<float>(n) / static_cast<float>(N);
            re += input[n] * std::cos(angle);
            im -= input[n] * std::sin(angle);
        }
        out[static_cast<size_t>(k)] = {re, im};
    }
}

}  // anonymous namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// ============================================================================
//  WhisperFeatureExtractorConfig
// ============================================================================

WhisperFeatureExtractorConfig WhisperFeatureExtractorConfig::from_json_file(const std::filesystem::path& path) {
    WhisperFeatureExtractorConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) return cfg;
    nlohmann::json j;
    f >> j;
    if (j.contains("sampling_rate"))  cfg.sampling_rate  = j.at("sampling_rate").get<int32_t>();
    if (j.contains("n_fft"))          cfg.n_fft          = j.at("n_fft").get<int32_t>();
    if (j.contains("hop_length"))     cfg.hop_length     = j.at("hop_length").get<int32_t>();
    if (j.contains("feature_size"))   cfg.feature_size   = j.at("feature_size").get<int32_t>();
    if (j.contains("n_samples"))      cfg.n_samples      = j.at("n_samples").get<int32_t>();
    if (j.contains("nb_max_frames"))  cfg.nb_max_frames  = j.at("nb_max_frames").get<int32_t>();
    if (j.contains("padding_value"))  cfg.padding_value  = j.at("padding_value").get<float>();
    if (j.contains("return_attention_mask")) cfg.return_attention_mask = j.at("return_attention_mask").get<bool>();
    return cfg;
}

// ============================================================================
//  read_wav_to_float32
// ============================================================================

std::vector<float> read_wav_to_float32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }
    auto hdr = parse_wav_header(f);

    if (hdr.sample_rate != 16000) {
        throw std::runtime_error("WAV file must be 16 kHz (got " + std::to_string(hdr.sample_rate) + ")");
    }
    if (hdr.num_channels != 1 && hdr.num_channels != 2) {
        throw std::runtime_error("WAV file must be mono or stereo");
    }
    if (hdr.audio_format != 1 && hdr.audio_format != 3) {
        throw std::runtime_error("Unsupported WAV format (must be PCM int16 or IEEE float32)");
    }

    const size_t frame_size = static_cast<size_t>(hdr.num_channels) * (hdr.bits_per_sample / 8);
    const size_t num_frames = static_cast<size_t>(hdr.data_size) / frame_size;

    std::vector<float> samples(num_frames);

    if (hdr.audio_format == 1 && hdr.bits_per_sample == 16) {
        // PCM 16-bit signed
        std::vector<int16_t> raw(num_frames * hdr.num_channels);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(int16_t)));
        if (hdr.num_channels == 1) {
            for (size_t i = 0; i < num_frames; ++i) {
                samples[i] = static_cast<float>(raw[i]) / 32768.0f;
            }
        } else {
            for (size_t i = 0; i < num_frames; ++i) {
                samples[i] = (static_cast<float>(raw[2*i]) + static_cast<float>(raw[2*i+1])) / 65536.0f;
            }
        }
    } else if (hdr.audio_format == 3 && hdr.bits_per_sample == 32) {
        // IEEE float 32
        std::vector<float> raw(num_frames * hdr.num_channels);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
        if (hdr.num_channels == 1) {
            samples = std::move(raw);
        } else {
            for (size_t i = 0; i < num_frames; ++i) {
                samples[i] = (raw[2*i] + raw[2*i+1]) * 0.5f;
            }
        }
    } else {
        throw std::runtime_error("Unsupported bits_per_sample=" + std::to_string(hdr.bits_per_sample) +
                                 " for format=" + std::to_string(hdr.audio_format));
    }

    return samples;
}

// ============================================================================
//  extract_whisper_mel_features
// ============================================================================

WhisperMelFeatures extract_whisper_mel_features(const std::vector<float>& waveform,
                                                const WhisperFeatureExtractorConfig& cfg) {
    const int32_t n_fft      = cfg.n_fft;        // 400
    const int32_t hop_length = cfg.hop_length;    // 160
    const int32_t n_mels     = cfg.feature_size;  // 128
    const int32_t num_freq_bins = n_fft / 2 + 1;  // 201
    const int32_t pad_amount = n_fft / 2;          // 200

    // 1. Reflect-pad waveform
    const size_t orig_len = waveform.size();
    std::vector<float> padded(pad_amount + orig_len + pad_amount);
    // Left reflect pad
    for (int32_t i = 0; i < pad_amount; ++i) {
        padded[static_cast<size_t>(pad_amount - 1 - i)] = waveform[static_cast<size_t>(i + 1)];
    }
    // Copy original
    std::memcpy(padded.data() + pad_amount, waveform.data(), orig_len * sizeof(float));
    // Right reflect pad
    for (int32_t i = 0; i < pad_amount; ++i) {
        padded[pad_amount + orig_len + static_cast<size_t>(i)] =
            waveform[orig_len - 2 - static_cast<size_t>(i)];
    }

    // 2. Compute STFT frames
    const size_t padded_len = padded.size();
    const int32_t num_frames = 1 + static_cast<int32_t>((padded_len - static_cast<size_t>(n_fft)) / static_cast<size_t>(hop_length));

    auto hann = make_hann_window(n_fft);

    // Power spectrum: shape [num_freq_bins, num_frames]
    // We compute it frame-by-frame
    std::vector<float> power_spec(static_cast<size_t>(num_freq_bins) * static_cast<size_t>(num_frames));
    std::vector<float> windowed(static_cast<size_t>(n_fft));
    std::vector<std::complex<float>> fft_out;

    for (int32_t frame = 0; frame < num_frames; ++frame) {
        const size_t start = static_cast<size_t>(frame) * static_cast<size_t>(hop_length);
        // Apply window
        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[static_cast<size_t>(i)] = padded[start + static_cast<size_t>(i)] * hann[static_cast<size_t>(i)];
        }
        // DFT
        real_dft(windowed.data(), n_fft, fft_out);
        // |X|^2
        for (int32_t k = 0; k < num_freq_bins; ++k) {
            const auto& c = fft_out[static_cast<size_t>(k)];
            power_spec[static_cast<size_t>(k) * static_cast<size_t>(num_frames) + static_cast<size_t>(frame)] =
                c.real() * c.real() + c.imag() * c.imag();
        }
    }

    // 3. Drop last time frame (matching Python: log_spec = log_spec[:, :-1])
    const int32_t T_out = num_frames - 1;

    // 4. Mel filter bank
    auto mel_bank = build_mel_filter_bank(num_freq_bins, n_mels, 0.0f, 8000.0f, cfg.sampling_rate);

    // mel_spec[m, t] = sum_f mel_bank[f, m] * power_spec[f, t]
    // Shape: [n_mels, T_out]
    std::vector<float> log_spec(static_cast<size_t>(n_mels) * static_cast<size_t>(T_out));

    for (int32_t m = 0; m < n_mels; ++m) {
        for (int32_t t = 0; t < T_out; ++t) {
            float val = 0.0f;
            for (int32_t f = 0; f < num_freq_bins; ++f) {
                val += mel_bank[static_cast<size_t>(f) * static_cast<size_t>(n_mels) + static_cast<size_t>(m)] *
                       power_spec[static_cast<size_t>(f) * static_cast<size_t>(num_frames) + static_cast<size_t>(t)];
            }
            // Clamp minimum (mel floor of 1e-10)
            val = std::max(val, 1e-10f);
            log_spec[static_cast<size_t>(m) * static_cast<size_t>(T_out) + static_cast<size_t>(t)] = std::log10(val);
        }
    }

    // 5. Dynamic range clamp: log_spec = max(log_spec, log_spec.max() - 8.0)
    float global_max = -1e30f;
    for (float v : log_spec) {
        global_max = std::max(global_max, v);
    }
    const float clamp_min = global_max - 8.0f;
    for (float& v : log_spec) {
        v = std::max(v, clamp_min);
    }

    // 6. Normalise: (x + 4) / 4
    for (float& v : log_spec) {
        v = (v + 4.0f) / 4.0f;
    }

    // 7. Package into ov::Tensors
    //    input_features: {1, n_mels, T_out}
    const size_t T = static_cast<size_t>(T_out);
    ov::Tensor input_features(ov::element::f32, {1, static_cast<size_t>(n_mels), T});
    std::memcpy(input_features.data<float>(), log_spec.data(), log_spec.size() * sizeof(float));

    //    feature_attention_mask: {1, T_out}  — all ones (no padding)
    ov::Tensor feature_attention_mask(ov::element::i64, {1, T});
    auto* mask_ptr = feature_attention_mask.data<int64_t>();
    for (size_t i = 0; i < T; ++i) {
        mask_ptr[i] = 1;
    }

    WhisperMelFeatures result;
    result.input_features         = std::move(input_features);
    result.feature_attention_mask = std::move(feature_attention_mask);
    result.audio_feature_length   = static_cast<int64_t>(T);

    return result;
}

// ============================================================================
//  get_audio_token_count
// ============================================================================

int64_t get_audio_token_count(int64_t T_frames) {
    // Matches Python _get_feat_extract_output_lengths:
    //   r = T_frames % 100
    //   f = (r - 1) / 2 + 1
    //   N = ((f - 1) / 2 + 1 - 1) / 2 + 1 + (T_frames / 100) * 13
    const int64_t r = T_frames % 100;
    const int64_t f = (r - 1) / 2 + 1;
    const int64_t N = ((f - 1) / 2 + 1 - 1) / 2 + 1 + (T_frames / 100) * 13;
    return N;
}

// ============================================================================
//  cnn_output_length
// ============================================================================

int64_t cnn_output_length(int64_t input_length) {
    // After 3× Conv2d with stride 2 (on the time dimension):
    //   L1 = (L + 1) / 2
    //   L2 = (L1 + 1) / 2
    //   L3 = (L2 + 1) / 2
    int64_t out = (input_length + 1) / 2;
    out = (out + 1) / 2;
    out = (out + 1) / 2;
    return std::max<int64_t>(1, out);
}

// ============================================================================
//  chunk_mel_for_audio_encoder
// ============================================================================

ChunkedMelInput chunk_mel_for_audio_encoder(const WhisperMelFeatures& mel,
                                             int32_t chunk_size) {
    const auto mel_shape = mel.input_features.get_shape();
    // mel.input_features: {1, n_mels, T_frames}
    const int32_t n_mels = static_cast<int32_t>(mel_shape[1]);
    const int64_t T = mel.audio_feature_length;

    // Compute chunk boundaries
    const int64_t num_full_chunks = T / chunk_size;
    const int64_t remainder = T % chunk_size;
    const int64_t num_chunks = num_full_chunks + (remainder > 0 ? 1 : 0);

    std::vector<int64_t> chunk_lengths;
    chunk_lengths.reserve(static_cast<size_t>(num_chunks));
    for (int64_t i = 0; i < num_full_chunks; ++i) {
        chunk_lengths.push_back(chunk_size);
    }
    if (remainder > 0) {
        chunk_lengths.push_back(remainder);
    }

    // If remainder is 0 and T is divisible by chunk_size, all chunks have chunk_size length
    // The max padded length is chunk_size (or remainder if only 1 chunk)
    const int64_t max_len = chunk_size;

    // Compute CNN output lengths per chunk
    std::vector<int64_t> cnn_lengths;
    cnn_lengths.reserve(static_cast<size_t>(num_chunks));
    int64_t total_tokens = 0;
    for (auto cl : chunk_lengths) {
        auto cnn_len = cnn_output_length(cl);
        cnn_lengths.push_back(cnn_len);
        total_tokens += cnn_len;
    }

    // Build chunked input features: {num_chunks, n_mels, max_len}
    ov::Tensor chunked_features(ov::element::f32,
                                {static_cast<size_t>(num_chunks),
                                 static_cast<size_t>(n_mels),
                                 static_cast<size_t>(max_len)});
    std::memset(chunked_features.data(), 0, chunked_features.get_byte_size());

    const float* src = mel.input_features.data<const float>();
    float* dst = chunked_features.data<float>();

    for (int64_t c = 0; c < num_chunks; ++c) {
        const int64_t frame_start = c * chunk_size;
        const int64_t frame_count = chunk_lengths[static_cast<size_t>(c)];

        for (int32_t m = 0; m < n_mels; ++m) {
            // Source: mel[0, m, frame_start .. frame_start+frame_count]
            const float* mel_row = src + static_cast<size_t>(m) * static_cast<size_t>(T);
            float* chunk_row = dst + (static_cast<size_t>(c) * static_cast<size_t>(n_mels) + static_cast<size_t>(m)) * static_cast<size_t>(max_len);
            std::memcpy(chunk_row, mel_row + frame_start, static_cast<size_t>(frame_count) * sizeof(float));
        }
    }

    // Build feature_attention_mask: {num_chunks, max_len}
    ov::Tensor feat_mask(ov::element::i64,
                         {static_cast<size_t>(num_chunks), static_cast<size_t>(max_len)});
    auto* mask_data = feat_mask.data<int64_t>();
    std::memset(mask_data, 0, feat_mask.get_byte_size());
    for (int64_t c = 0; c < num_chunks; ++c) {
        for (int64_t f = 0; f < chunk_lengths[static_cast<size_t>(c)]; ++f) {
            mask_data[c * max_len + f] = 1;
        }
    }

    // Build audio_feature_lengths: {num_chunks}
    ov::Tensor afl(ov::element::i64, {static_cast<size_t>(num_chunks)});
    auto* afl_data = afl.data<int64_t>();
    for (int64_t c = 0; c < num_chunks; ++c) {
        afl_data[c] = chunk_lengths[static_cast<size_t>(c)];
    }

    ChunkedMelInput result;
    result.input_features        = std::move(chunked_features);
    result.feature_attention_mask = std::move(feat_mask);
    result.audio_feature_lengths = std::move(afl);
    result.chunk_lengths         = std::move(chunk_lengths);
    result.cnn_output_lengths    = std::move(cnn_lengths);
    result.total_output_tokens   = total_tokens;
    return result;
}

// ============================================================================
//  extract_valid_audio_tokens
// ============================================================================

ov::Tensor extract_valid_audio_tokens(const ov::Tensor& encoder_output,
                                      const std::vector<int64_t>& cnn_output_lengths) {
    const auto shape = encoder_output.get_shape();
    // encoder_output: {num_chunks, max_seq_after_cnn, hidden_dim}
    if (shape.size() != 3) {
        throw std::runtime_error("extract_valid_audio_tokens: expected 3D encoder output, got " +
                                 std::to_string(shape.size()) + "D");
    }
    const size_t num_chunks = shape[0];
    const size_t max_seq = shape[1];
    const size_t hidden = shape[2];

    int64_t total_tokens = 0;
    for (auto len : cnn_output_lengths) total_tokens += len;

    ov::Tensor result(ov::element::f32, {static_cast<size_t>(total_tokens), hidden});
    float* dst = result.data<float>();
    const float* src = encoder_output.data<const float>();

    size_t token_idx = 0;
    for (size_t c = 0; c < num_chunks; ++c) {
        const int64_t valid = cnn_output_lengths[c];
        for (int64_t t = 0; t < valid; ++t) {
            const float* row = src + (c * max_seq + static_cast<size_t>(t)) * hidden;
            std::memcpy(dst + token_idx * hidden, row, hidden * sizeof(float));
            ++token_idx;
        }
    }

    return result;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
