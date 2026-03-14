// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone tool: extract sampled video frames using OpenCV.
// Replaces the Python extract_video_frames.py with zero Python dependency.
//
// Usage:
//   extract_video_frames --video path/to/video.mp4 --output-dir ./frames
//   extract_video_frames --video path/to/video.mp4 --output-dir ./frames --target-fps 2.0 --max-frames 8

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// smart_nframes: mirrors qwen_omni_utils/vision_process.py
// ---------------------------------------------------------------------------
static constexpr int FRAME_FACTOR = 2;
static constexpr double DEFAULT_FPS = 2.0;
static constexpr int DEFAULT_MIN_FRAMES = 4;
static constexpr int DEFAULT_MAX_FRAMES = 768;

static int smart_nframes(int total_frames,
                         double video_fps,
                         double target_fps,
                         int min_frames,
                         int max_frames) {
    double nf = static_cast<double>(total_frames) / video_fps * target_fps;
    int nframes = static_cast<int>(std::round(nf / FRAME_FACTOR)) * FRAME_FACTOR;
    nframes = std::max(FRAME_FACTOR, nframes);
    nframes = std::min(nframes, total_frames);

    min_frames = ((min_frames + FRAME_FACTOR - 1) / FRAME_FACTOR) * FRAME_FACTOR;
    if (min_frames < FRAME_FACTOR) min_frames = FRAME_FACTOR;

    max_frames = std::min(max_frames, total_frames);
    max_frames = (max_frames / FRAME_FACTOR) * FRAME_FACTOR;
    if (max_frames < FRAME_FACTOR) max_frames = FRAME_FACTOR;

    nframes = std::max(min_frames, std::min(nframes, max_frames));
    return nframes;
}

// Return `count` uniformly-spaced indices in [0, total-1].
static std::vector<int> linspace_indices(int total, int count) {
    std::vector<int> idx(count);
    if (count == 1) {
        idx[0] = 0;
        return idx;
    }
    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / (count - 1) * (total - 1);
        idx[i] = static_cast<int>(std::round(t));
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Minimal argument parser
// ---------------------------------------------------------------------------
struct Args {
    std::string video;
    std::string output_dir;
    double target_fps = DEFAULT_FPS;
    int min_frames = DEFAULT_MIN_FRAMES;
    int max_frames = DEFAULT_MAX_FRAMES;
    std::string format = "png";
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --video PATH --output-dir PATH"
                 " [--target-fps F] [--min-frames N] [--max-frames N] [--format png|jpg]\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--video")         args.video = next();
        else if (a == "--output-dir") args.output_dir = next();
        else if (a == "--target-fps") args.target_fps = std::stod(next());
        else if (a == "--min-frames") args.min_frames = std::stoi(next());
        else if (a == "--max-frames") args.max_frames = std::stoi(next());
        else if (a == "--format")     args.format = next();
        else { std::cerr << "Unknown option: " << a << "\n"; return false; }
    }
    if (args.video.empty() || args.output_dir.empty()) {
        std::cerr << "Error: --video and --output-dir are required.\n";
        return false;
    }
    if (args.format != "png" && args.format != "jpg") {
        std::cerr << "Error: --format must be png or jpg.\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!fs::exists(args.video)) {
        std::cerr << "Error: video file not found: " << args.video << "\n";
        return 1;
    }

    fs::create_directories(args.output_dir);

    // --- Open video ---
    cv::VideoCapture cap(args.video);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video: " << args.video << "\n";
        return 1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 24.0;

    std::cout << "[extract_video_frames] backend=opencv, total_frames=" << total_frames
              << ", fps=" << video_fps << "\n";

    // --- Compute sample indices ---
    int n_sample = smart_nframes(total_frames, video_fps, args.target_fps,
                                 args.min_frames, args.max_frames);
    auto indices = linspace_indices(total_frames, n_sample);

    std::cout << "[extract_video_frames] sampling " << n_sample << " frames: indices=[";
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << indices[i];
    }
    std::cout << "]\n";

    // --- Extract & save frames ---
    std::string ext = "." + args.format;
    int saved = 0;
    cv::Mat frame;
    for (int seq = 0; seq < static_cast<int>(indices.size()); ++seq) {
        cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(indices[seq]));
        if (!cap.read(frame) || frame.empty())
            continue;
        char fname[64];
        snprintf(fname, sizeof(fname), "frame_%04d%s", seq, ext.c_str());
        std::string out_path = (fs::path(args.output_dir) / fname).string();
        cv::imwrite(out_path, frame);
        ++saved;
    }
    cap.release();

    std::cout << "[extract_video_frames] saved " << saved << " frames to " << args.output_dir << "\n";

    // --- Write metadata ---
    std::string meta_path = (fs::path(args.output_dir) / "video_meta.txt").string();
    std::ofstream meta(meta_path);
    meta << "total_frames=" << total_frames << "\n";
    meta << "fps=" << video_fps << "\n";
    meta << "sampled=" << n_sample << "\n";
    meta << "backend=opencv\n";
    meta.close();
    std::cout << "[extract_video_frames] metadata written to " << meta_path << "\n";

    return 0;
}
