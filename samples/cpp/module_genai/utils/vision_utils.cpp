// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_utils.hpp"

#include <sstream>
#include <set>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <opencv2/opencv.hpp>

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}
#endif  // HAVE_FFMPEG

namespace fs = std::filesystem;

namespace image_utils {

// ============================================================================
// Image Loading Functions
// ============================================================================

ov::Tensor load_image(const std::filesystem::path& image_path) {
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char* image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image) {
        std::stringstream error_message;
        error_message << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{error_message.str()};
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (image && channels * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t, size_t) noexcept {
            stbi_image_free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept { return this == &other; }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desired_channels)},
        SharedImageAllocator{image, desired_channels, y, x});
}

std::vector<ov::Tensor> load_images(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& dir_entry : sorted_images) {
            images.push_back(load_image(dir_entry));
        }
        return images;
    }
    return {load_image(input_path)};
}

ov::Tensor load_video(const std::filesystem::path& input_path) {
    auto rgbs = load_images(input_path);
    if (rgbs.size() == 0) {
        return {};
    }

    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{rgbs.size(), rgbs[0].get_shape()[1], rgbs[0].get_shape()[2], rgbs[0].get_shape()[3]});
    std::cout << "video.shape = " << video.get_shape() << std::endl;

    auto stride = rgbs[0].get_byte_size();
    std::cout << "stride = " << stride << std::endl;
    auto dst = reinterpret_cast<char*>(video.data());
    int b = 0;
    for (auto rgb : rgbs) {
        std::memcpy(dst + stride * b, rgb.data(), stride);
        b++;
    }
    return video;
}

// ============================================================================
// Video Loading (with OpenCV + optional FFmpeg audio)
// ============================================================================

namespace {

/// Mirrors Python smart_resize (factor=28 grid, maintains aspect ratio)
std::pair<int, int> smart_resize_video(int height, int width,
                                       int factor     = 28,
                                       int min_pixels = 128 * 28 * 28,
                                       int max_pixels = 768 * 28 * 28) {
    auto round_f = [factor](double v) { return std::max(factor, (int)(std::round(v / factor) * factor)); };
    auto floor_f = [factor](double v) { return std::max(factor, (int)(std::floor(v / factor) * factor)); };
    auto ceil_f  = [factor](double v) { return std::max(factor, (int)(std::ceil(v  / factor) * factor)); };

    int h_bar = round_f(height);
    int w_bar = round_f(width);
    if ((long long)h_bar * w_bar > max_pixels) {
        double beta = std::sqrt((double)(height * width) / max_pixels);
        h_bar = floor_f(height / beta);
        w_bar = floor_f(width  / beta);
    } else if ((long long)h_bar * w_bar < min_pixels) {
        double beta = std::sqrt((double)min_pixels / (height * width));
        h_bar = ceil_f(height * beta);
        w_bar = ceil_f(width  * beta);
    }
    return {h_bar, w_bar};
}

/// Mirrors Python smart_nframes
int smart_nframes(int total_frames, double video_fps,
                  float target_fps = 2.0f, int min_frames = 4, int max_frames = 768) {
    constexpr int FRAME_FACTOR = 2;
    auto floor2 = [FRAME_FACTOR](double v) { return (int)(std::floor(v / FRAME_FACTOR) * FRAME_FACTOR); };
    auto ceil2  = [FRAME_FACTOR](double v) { return (int)(std::ceil(v  / FRAME_FACTOR) * FRAME_FACTOR); };

    int min_f   = std::max(FRAME_FACTOR, ceil2(min_frames));
    int max_f   = std::max(FRAME_FACTOR, floor2(std::min(max_frames, total_frames)));
    double nf   = total_frames / video_fps * target_fps;
    int nframes = floor2(std::min(std::max(nf, (double)min_f), (double)max_f));
    return std::max(nframes, FRAME_FACTOR);
}

#ifdef HAVE_FFMPEG
/// Decode all audio from a video file and resample to 16 kHz mono f32.
/// Returns empty ov::Tensor if the file has no audio stream.
ov::Tensor extract_audio_from_video(const std::filesystem::path& video_path) {
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, video_path.string().c_str(), nullptr, nullptr) < 0)
        throw std::runtime_error("extract_audio_from_video: cannot open: " + video_path.string());

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("extract_audio_from_video: cannot find stream info");
    }

    // Find first audio stream
    int audio_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_idx = (int)i;
            break;
        }
    }
    if (audio_idx < 0) {
        avformat_close_input(&fmt_ctx);
        return ov::Tensor{};  // no audio stream
    }

    AVCodecParameters* par   = fmt_ctx->streams[audio_idx]->codecpar;
    const AVCodec*     codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("extract_audio_from_video: no decoder for audio codec");
    }
    AVCodecContext* dec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec_ctx, par);
    avcodec_open2(dec_ctx, codec, nullptr);

    // Build SwrContext: source format → 16 kHz mono f32
    SwrContext* swr = nullptr;
    int swr_ret = -1;

#if LIBAVUTIL_VERSION_MAJOR >= 57
    AVChannelLayout in_layout{};
    AVChannelLayout out_layout{};
    av_channel_layout_default(&out_layout, 1);

#if LIBAVCODEC_VERSION_MAJOR >= 59
    // FFmpeg 5+: AVCodecContext exposes ch_layout.
    if (av_channel_layout_copy(&in_layout, &dec_ctx->ch_layout) < 0) {
        av_channel_layout_default(&in_layout, dec_ctx->ch_layout.nb_channels > 0 ? dec_ctx->ch_layout.nb_channels : 1);
    }
#else
    // Compatibility path for older codec API with new channel layout type.
    uint64_t in_mask = dec_ctx->channel_layout
                         ? static_cast<uint64_t>(dec_ctx->channel_layout)
                         : static_cast<uint64_t>(av_get_default_channel_layout(dec_ctx->channels));
    if (av_channel_layout_from_mask(&in_layout, in_mask) < 0) {
        av_channel_layout_default(&in_layout, dec_ctx->channels > 0 ? dec_ctx->channels : 1);
    }
#endif

#if LIBSWRESAMPLE_VERSION_MAJOR >= 4
    swr_ret = swr_alloc_set_opts2(&swr,
                                  &out_layout, AV_SAMPLE_FMT_FLT, 16000,
                                  &in_layout,  dec_ctx->sample_fmt, dec_ctx->sample_rate,
                                  0, nullptr);
#else
    swr_ret = -1;
#endif

    av_channel_layout_uninit(&in_layout);
    av_channel_layout_uninit(&out_layout);
#else
    // FFmpeg 4.x: legacy channel_layout/channel API + swr_alloc_set_opts.
    const int64_t out_layout = static_cast<int64_t>(av_get_default_channel_layout(1));
    const int64_t in_layout = dec_ctx->channel_layout
                                ? static_cast<int64_t>(dec_ctx->channel_layout)
                                : static_cast<int64_t>(av_get_default_channel_layout(dec_ctx->channels));
    swr = swr_alloc_set_opts(nullptr,
                             out_layout, AV_SAMPLE_FMT_FLT, 16000,
                             in_layout,  dec_ctx->sample_fmt, dec_ctx->sample_rate,
                             0, nullptr);
    swr_ret = swr ? 0 : -1;
#endif

    if (swr_ret < 0 || !swr) {
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("extract_audio_from_video: cannot create SwrContext");
    }
    if (swr_init(swr) < 0) {
        swr_free(&swr);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("extract_audio_from_video: cannot init SwrContext");
    }

    std::vector<float> samples;
    AVPacket* pkt   = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();

    auto do_convert = [&](int nb_input_samples, const uint8_t** input_data) {
        int out_count = (int)av_rescale_rnd(
            swr_get_delay(swr, dec_ctx->sample_rate) + nb_input_samples,
            16000, dec_ctx->sample_rate, AV_ROUND_UP);
        std::vector<float> buf(out_count);
        uint8_t* out_ptr = reinterpret_cast<uint8_t*>(buf.data());
        int converted = swr_convert(swr, &out_ptr, out_count, input_data, nb_input_samples);
        if (converted > 0) {
            samples.insert(samples.end(), buf.begin(), buf.begin() + converted);
        }
    };

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == audio_idx) {
            if (avcodec_send_packet(dec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(dec_ctx, frame) == 0) {
                    do_convert(frame->nb_samples,
                               const_cast<const uint8_t**>(frame->data));
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(pkt);
    }
    // Flush resampler
    do_convert(0, nullptr);

    av_frame_free(&frame);
    av_packet_free(&pkt);
    swr_free(&swr);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    if (samples.empty())
        return ov::Tensor{};

    ov::Tensor audio(ov::element::f32, {1, samples.size()});
    std::memcpy(audio.data<float>(), samples.data(), samples.size() * sizeof(float));
    return audio;
}
#endif  // HAVE_FFMPEG

}  // anonymous namespace

VideoLoadResult load_video_with_audio(const std::filesystem::path& video_path,
                                      bool                         use_audio_in_video,
                                      const VideoLoadOptions&      opts) {
    // ── Open video with OpenCV ──────────────────────────────────────────────
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened())
        throw std::runtime_error("load_video_with_audio: cannot open: " + video_path.string());

    int    total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double video_fps    = cap.get(cv::CAP_PROP_FPS);
    int    height_orig  = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int    width_orig   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);

    if (total_frames <= 0 || video_fps <= 0.0)
        throw std::runtime_error("load_video_with_audio: invalid video metadata");

    // ── Compute how many / which frames to sample ───────────────────────────
    // Budget-aware max_pixels per frame (mirrors Python VIDEO_TOTAL_PIXELS logic)
    constexpr long long VIDEO_TOTAL_PIXELS = (long long)(128000 * 28 * 28 * 0.9);
    constexpr int       VIDEO_MAX_PIXELS   = 768 * 28 * 28;
    constexpr int       VIDEO_MIN_PIXELS   = 128 * 28 * 28;
    constexpr int       FRAME_FACTOR       = 2;

    int nframes = smart_nframes(total_frames, video_fps,
                                opts.fps, opts.min_frames, opts.max_frames);
    float sample_fps = (float)(nframes) / (float)(total_frames) * (float)video_fps;

    // Per-frame pixel budget
    long long max_pixels_per_frame = std::max(
        (long long)std::min((long long)VIDEO_MAX_PIXELS,
                            VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
        (long long)(VIDEO_MIN_PIXELS * 1.05));

    auto [res_h, res_w] = smart_resize_video(height_orig, width_orig,
                                              28,
                                              VIDEO_MIN_PIXELS,
                                              (int)max_pixels_per_frame);

    // Uniformly-spaced frame indices (linspace, clamped)
    std::vector<int> frame_indices(nframes);
    for (int i = 0; i < nframes; ++i) {
        double t = (nframes == 1) ? 0.0
                                  : (double)i / (nframes - 1) * (total_frames - 1);
        frame_indices[i] = std::min((int)std::round(t), total_frames - 1);
    }

    // ── Read and resize selected frames ────────────────────────────────────
    ov::Tensor frames_tensor(ov::element::u8,
                             {(size_t)nframes, (size_t)res_h, (size_t)res_w, 3});
    uint8_t* dst = frames_tensor.data<uint8_t>();
    size_t frame_bytes = (size_t)res_h * res_w * 3;

    cv::Mat bgr, rgb, resized;
    int prev_idx = -1;
    for (int i = 0; i < nframes; ++i) {
        int idx = frame_indices[i];
        if (idx != prev_idx) {
            cap.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
            if (!cap.read(bgr) || bgr.empty())
                throw std::runtime_error("load_video_with_audio: failed to decode frame " +
                                         std::to_string(idx));
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            cv::resize(rgb, resized, cv::Size(res_w, res_h),
                       0, 0, cv::INTER_CUBIC);
            prev_idx = idx;
        }
        std::memcpy(dst + i * frame_bytes, resized.data, frame_bytes);
    }
    cap.release();

    // ── Optionally extract audio ────────────────────────────────────────────
    ov::Tensor audio_tensor;
#ifdef HAVE_FFMPEG
    if (use_audio_in_video) {
        audio_tensor = extract_audio_from_video(video_path);
        if (!audio_tensor.get_size()) {
            throw std::runtime_error(
                "load_video_with_audio: use_audio_in_video=true but '" +
                video_path.string() + "' has no audio stream");
        }
    }
#else
    if (use_audio_in_video) {
        throw std::runtime_error(
            "load_video_with_audio: audio extraction requires FFmpeg. "
            "Rebuild with -DHAVE_FFMPEG=ON.");
    }
#endif

    return VideoLoadResult{std::move(frames_tensor), std::move(audio_tensor), sample_fps};
}

ov::Tensor create_countdown_frames() {
    int frames_count = 5, height = 240, width = 360;
    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{(size_t)frames_count, (size_t)height, (size_t)width, 3});

    for (int i = frames_count; i > 0; i--) {
        cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);
        std::string text = std::to_string(i);

        int baseline = 0;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 3.0;
        int thickness = 4;

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

        int text_width = textSize.width;
        int text_height = textSize.height;
        int text_x = (width - text_width) / 2;
        int text_y = (height + text_height) / 2;

        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::Point org(text_x, text_y);

        cv::putText(frame, text, org, fontFace, fontScale, color, thickness, cv::LINE_AA);

        int idx = frames_count - i;
        std::memcpy((char*)video.data() + idx * height * width * 3, frame.data, height * width * 3);
    }
    return video;
}

// ============================================================================
// Image Saving Functions
// ============================================================================

bool save_image_bmp(const std::string& filename, const ov::Tensor& image, bool convert_rgb2bgr) {
    try {
        ov::Shape shape = image.get_shape();

        size_t height, width, channels;
        const uint8_t* data = image.data<const uint8_t>();

        if (shape.size() == 4) {
            if (shape[3] == 3) {
                height = shape[1];
                width = shape[2];
                channels = shape[3];
            } else if (shape[1] == 3) {
                std::cerr << "[ERROR] NCHW format not supported for BMP save" << std::endl;
                return false;
            } else {
                std::cerr << "[ERROR] Unknown 4D tensor format" << std::endl;
                return false;
            }
        } else if (shape.size() == 3) {
            height = shape[0];
            width = shape[1];
            channels = shape[2];
        } else {
            std::cerr << "[ERROR] Unsupported tensor shape for image save" << std::endl;
            return false;
        }

        if (channels != 3) {
            std::cerr << "[ERROR] Expected 3 channels, got " << channels << std::endl;
            return false;
        }

        unsigned char file_header[14] = {
            'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0
        };

        unsigned char info_header[40] = {
            40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        };

        int row_padding = (4 - (width * 3) % 4) % 4;
        int data_size = static_cast<int>((width * 3 + row_padding) * height);
        int file_size = 54 + data_size;

        file_header[2] = file_size & 0xFF;
        file_header[3] = (file_size >> 8) & 0xFF;
        file_header[4] = (file_size >> 16) & 0xFF;
        file_header[5] = (file_size >> 24) & 0xFF;

        info_header[4] = width & 0xFF;
        info_header[5] = (width >> 8) & 0xFF;
        info_header[6] = (width >> 16) & 0xFF;
        info_header[7] = (width >> 24) & 0xFF;

        int32_t neg_height = -static_cast<int32_t>(height);
        info_header[8] = neg_height & 0xFF;
        info_header[9] = (neg_height >> 8) & 0xFF;
        info_header[10] = (neg_height >> 16) & 0xFF;
        info_header[11] = (neg_height >> 24) & 0xFF;

        info_header[20] = data_size & 0xFF;
        info_header[21] = (data_size >> 8) & 0xFF;
        info_header[22] = (data_size >> 16) & 0xFF;
        info_header[23] = (data_size >> 24) & 0xFF;

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Failed to open file: " << filename << std::endl;
            return false;
        }

        file.write(reinterpret_cast<char*>(file_header), 14);
        file.write(reinterpret_cast<char*>(info_header), 40);

        unsigned char padding[3] = {0, 0, 0};

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = (y * width + x) * 3;
                if (convert_rgb2bgr) {
                    file.put(static_cast<char>(data[idx + 2]));
                    file.put(static_cast<char>(data[idx + 1]));
                    file.put(static_cast<char>(data[idx]));
                } else {
                    file.write(reinterpret_cast<const char*>(data + idx), 3);
                }
            }
            file.write(reinterpret_cast<char*>(padding), row_padding);
        }

        file.close();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to save image: " << e.what() << std::endl;
        return false;
    }
}

std::string generate_output_filename(const std::string& prefix, const std::string& suffix) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    std::ostringstream oss;
    oss << prefix << "_"
        << std::put_time(&tm_now, "%Y%m%d_%H%M%S")
        << suffix;
    return oss.str();
}

} // namespace image_utils
