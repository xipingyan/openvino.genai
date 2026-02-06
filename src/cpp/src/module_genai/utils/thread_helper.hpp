// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <future>
#include <thread>

#include "openvino/runtime/compiled_model.hpp"
#include "profiler.hpp"

namespace ov::genai::module::thread_utils {

#ifndef DISABLE_THREAD
#define DISABLE_THREAD 1
#endif

inline std::future<bool> load_model_weights_async(ov::CompiledModel& compiled_model, ov::InferRequest& infer_request) {
#if DISABLE_THREAD
    PROFILE(pm, "load_model_weights sync");
    compiled_model.load_model_weights();
    infer_request = compiled_model.create_infer_request();
    return std::async(std::launch::deferred, []() -> bool { return true; });
#else
    auto load_fun = [&]() -> bool {
        PROFILE(pm, "load_model_weights async");
        compiled_model.load_model_weights();
        infer_request = compiled_model.create_infer_request();
        return true;
    };
    return std::async(std::launch::async, load_fun);
#endif
}

inline std::future<bool> release_model_weights_async(ov::CompiledModel& compiled_model, ov::InferRequest& infer_request) {
#if DISABLE_THREAD
    PROFILE(pm, "release_model_weights sync");
    compiled_model.release_model_weights();
    // release infer request to release the reference to the model weights
    infer_request = ov::InferRequest();
    return std::async(std::launch::deferred, []() -> bool { return true; });
#else
    auto load_fun = [&]() -> bool {
        PROFILE(pm, "release_model_weights async");
        compiled_model.release_model_weights();
        infer_request = ov::InferRequest();  // reset infer request to release the reference to the model weights
        return true;
    };
    return std::async(std::launch::async, load_fun);
#endif
}

}  // namespace ov::genai::module::thread_utils