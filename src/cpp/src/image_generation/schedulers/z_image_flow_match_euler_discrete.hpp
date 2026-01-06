// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flow_match_euler_discrete.hpp"

namespace ov::genai {

class ZImageFlowMatchEulerDiscreteScheduler : public FlowMatchEulerDiscreteScheduler {
public:
    explicit ZImageFlowMatchEulerDiscreteScheduler(const std::filesystem::path& scheduler_config_path)
        : FlowMatchEulerDiscreteScheduler(scheduler_config_path) {}
    explicit ZImageFlowMatchEulerDiscreteScheduler(const Config& scheduler_config)
        : FlowMatchEulerDiscreteScheduler(scheduler_config) {}

    void set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) override;
    void set_sigma_min(float sigma_min);

private:
    // Config m_config;

    // std::vector<float> m_sigmas;
    // std::vector<float> m_timesteps, m_schedule_timesteps;

    // float m_sigma_min, m_sigma_max;
    // float m_strength;
    // size_t m_step_index, m_begin_index;
    // size_t m_num_inference_steps;

};

}
