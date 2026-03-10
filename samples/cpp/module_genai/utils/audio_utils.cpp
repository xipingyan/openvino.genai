// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace audio_utils {

// ──────────────────────────────────────────────────────────────────────────────
// Minimal WAV / RIFF parser
//
// Supports:
//   • PCM  8-bit  (unsigned)
//   • PCM 16-bit  (signed, little-endian)
//   • PCM 24-bit  (signed, little-endian, 3 bytes/sample)
//   • PCM 32-bit  (signed, little-endian)
//   • IEEE float  32-bit (little-endian)
//
// Multi-channel audio is reduced to mono by averaging channels.
//
// Output: ov::Tensor [1, num_samples], element type f32, normalised to [-1, 1].
//
// NOTE: No resampling is performed. The Qwen3-Omni audio encoder expects 16 kHz
//       input. Provide a 16 kHz WAV or resample offline before calling this
//       function.
// ──────────────────────────────────────────────────────────────────────────────

namespace {

// ---- helpers to read little-endian integers from a raw byte buffer ----------

inline uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

inline int32_t read_i24_le(const uint8_t* p) {
    // 3-byte signed integer, little-endian
    int32_t v = static_cast<int32_t>(p[0])
              | (static_cast<int32_t>(p[1]) << 8)
              | (static_cast<int32_t>(p[2]) << 16);
    // sign-extend from bit 23
    if (v & 0x800000) v |= static_cast<int32_t>(0xFF000000);
    return v;
}

inline int32_t read_i32_le(const uint8_t* p) {
    return static_cast<int32_t>(read_u32_le(p));
}

// ---- WAV fmt chunk fields ---------------------------------------------------

struct WavFmt {
    uint16_t audio_format;   // 1=PCM, 3=IEEE float
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
};

// ---- find a RIFF chunk by its four-character ID ----------------------------

bool find_chunk(std::ifstream& f, const char id[4], uint32_t& chunk_size) {
    char tag[4];
    char sz[4];
    while (f.read(tag, 4) && f.read(sz, 4)) {
        chunk_size = read_u32_le(reinterpret_cast<const uint8_t*>(sz));
        if (std::memcmp(tag, id, 4) == 0)
            return true;
        // skip this chunk (round up to even byte boundary)
        const uint32_t skip = chunk_size + (chunk_size & 1u);
        f.seekg(skip, std::ios::cur);
    }
    return false;
}

} // anonymous namespace

// ──────────────────────────────────────────────────────────────────────────────

ov::Tensor load_audio(const std::filesystem::path& audio_path) {
    std::ifstream f(audio_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("audio_utils::load_audio: cannot open file: " + audio_path.string());

    // ── RIFF header ──────────────────────────────────────────────────────────
    char riff[4], wave[4];
    char riff_size_buf[4];
    if (!f.read(riff, 4))
        throw std::runtime_error("audio_utils::load_audio: unexpected end of file reading RIFF tag");
    if (std::memcmp(riff, "RIFF", 4) != 0)
        throw std::runtime_error("audio_utils::load_audio: not a RIFF file: " + audio_path.string());
    f.read(riff_size_buf, 4); // ignore total file size
    if (!f.read(wave, 4))
        throw std::runtime_error("audio_utils::load_audio: unexpected end of file reading WAVE tag");
    if (std::memcmp(wave, "WAVE", 4) != 0)
        throw std::runtime_error("audio_utils::load_audio: not a WAVE file: " + audio_path.string());

    // ── fmt chunk ────────────────────────────────────────────────────────────
    uint32_t fmt_size = 0;
    if (!find_chunk(f, "fmt ", fmt_size))
        throw std::runtime_error("audio_utils::load_audio: no 'fmt ' chunk in: " + audio_path.string());
    if (fmt_size < 16)
        throw std::runtime_error("audio_utils::load_audio: 'fmt ' chunk too small");

    std::vector<uint8_t> fmt_buf(fmt_size + (fmt_size & 1u));
    if (!f.read(reinterpret_cast<char*>(fmt_buf.data()), fmt_buf.size()))
        throw std::runtime_error("audio_utils::load_audio: cannot read 'fmt ' chunk");

    WavFmt fmt{};
    fmt.audio_format   = read_u16_le(fmt_buf.data() + 0);
    fmt.num_channels   = read_u16_le(fmt_buf.data() + 2);
    fmt.sample_rate    = read_u32_le(fmt_buf.data() + 4);
    // byte_rate        = read_u32_le(fmt_buf.data() + 8)  // unused
    // block_align      = read_u16_le(fmt_buf.data() + 12) // unused
    fmt.bits_per_sample = read_u16_le(fmt_buf.data() + 14);

    // audio_format == 0xFFFE → extensible; read sub-format GUID (bytes 24-39)
    if (fmt.audio_format == 0xFFFEu && fmt_size >= 40) {
        fmt.audio_format = read_u16_le(fmt_buf.data() + 24); // sub-format
    }

    if (fmt.audio_format != 1 && fmt.audio_format != 3)
        throw std::runtime_error("audio_utils::load_audio: unsupported audio format (only PCM and IEEE float WAV supported)");
    if (fmt.num_channels == 0)
        throw std::runtime_error("audio_utils::load_audio: zero channels");

    // ── data chunk ───────────────────────────────────────────────────────────
    uint32_t data_size = 0;
    if (!find_chunk(f, "data", data_size))
        throw std::runtime_error("audio_utils::load_audio: no 'data' chunk in: " + audio_path.string());

    std::vector<uint8_t> raw(data_size);
    if (!f.read(reinterpret_cast<char*>(raw.data()), data_size))
        throw std::runtime_error("audio_utils::load_audio: cannot read 'data' chunk (file truncated?)");

    // ── convert to float32 mono ───────────────────────────────────────────────
    const uint32_t bytes_per_sample = fmt.bits_per_sample / 8u;
    const uint32_t frame_size       = bytes_per_sample * fmt.num_channels;
    if (frame_size == 0)
        throw std::runtime_error("audio_utils::load_audio: zero frame size");

    const size_t num_frames = data_size / frame_size;
    std::vector<float> samples(num_frames);

    const uint8_t* p = raw.data();
    for (size_t i = 0; i < num_frames; ++i, p += frame_size) {
        // Average channels to produce mono
        double acc = 0.0;

        if (fmt.audio_format == 3) {
            // IEEE float 32-bit
            for (uint16_t ch = 0; ch < fmt.num_channels; ++ch) {
                float v;
                std::memcpy(&v, p + ch * 4, 4);
                acc += static_cast<double>(v);
            }
            acc /= fmt.num_channels;
        } else {
            // PCM – normalise to [-1, 1]
            if (fmt.bits_per_sample == 8) {
                // 8-bit PCM is unsigned [0, 255]
                for (uint16_t ch = 0; ch < fmt.num_channels; ++ch)
                    acc += (static_cast<double>(p[ch]) - 128.0) / 128.0;
            } else if (fmt.bits_per_sample == 16) {
                const double scale = 1.0 / 32768.0;
                for (uint16_t ch = 0; ch < fmt.num_channels; ++ch) {
                    const int16_t v = static_cast<int16_t>(read_u16_le(p + ch * 2));
                    acc += static_cast<double>(v) * scale;
                }
            } else if (fmt.bits_per_sample == 24) {
                const double scale = 1.0 / 8388608.0;
                for (uint16_t ch = 0; ch < fmt.num_channels; ++ch) {
                    acc += static_cast<double>(read_i24_le(p + ch * 3)) * scale;
                }
            } else if (fmt.bits_per_sample == 32) {
                const double scale = 1.0 / 2147483648.0;
                for (uint16_t ch = 0; ch < fmt.num_channels; ++ch) {
                    acc += static_cast<double>(read_i32_le(p + ch * 4)) * scale;
                }
            } else {
                throw std::runtime_error("audio_utils::load_audio: unsupported bits_per_sample: " +
                                         std::to_string(fmt.bits_per_sample));
            }
            acc /= fmt.num_channels;
        }

        samples[i] = static_cast<float>(acc);
    }

    // ── wrap in ov::Tensor [1, num_frames] ────────────────────────────────────
    ov::Tensor tensor(ov::element::f32, {1, num_frames});
    std::memcpy(tensor.data<float>(), samples.data(), num_frames * sizeof(float));
    return tensor;
}

} // namespace audio_utils
