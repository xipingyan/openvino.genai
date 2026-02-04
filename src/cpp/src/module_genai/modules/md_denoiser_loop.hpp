// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"
#include "module_genai/transformer_config.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "unipc_multistep_scheduler.hpp"
#include "circular_buffer_queue.hpp"
#include <memory>

namespace ov {
namespace genai {

class IScheduler;

namespace module {

class OV_Unsqueeze {
private:
    OV_Unsqueeze(const std::string& device, const int32_t& axes);
    OV_Unsqueeze(const ov::RemoteContext& context, const int32_t& axes);

    std::string m_device;
    int32_t m_axes = 0;
    ov::RemoteContext m_context;
    ov::InferRequest m_request;

    ov::Shape m_last_shape;
    bool shape_changed(const ov::Shape& in_shape);
public:
    ~OV_Unsqueeze() = default;
    using PTR = std::shared_ptr<OV_Unsqueeze>;
    static PTR create(const std::string& device, const int32_t& axes) {
        return PTR(new OV_Unsqueeze(device, axes));
    }

    static PTR create(const ov::RemoteContext& context, const int32_t& axes) {
        return PTR(new OV_Unsqueeze(context, axes));
    }

    ov::Tensor infer(const ov::Tensor& input_tensor);
};

class DenoiserLoopModule : public IBaseModule {
    DeclareModuleConstructor(DenoiserLoopModule);

private:
    bool initialize();
    // Image generation
    ov::Tensor run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config);
    // Video generation
    ov::Tensor run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        int num_inference_steps,
        float guidance_scale);
    DiffusionModelType m_model_type;
    std::variant<std::shared_ptr<IScheduler>, std::shared_ptr<UniPCMultistepScheduler>> m_scheduler;
    ov::InferRequest m_request;
    bool m_is_multi_prompts {false};
    float m_cfg_truncation;
    bool m_cfg_normalization;

    ov::CompiledModel m_compiled_model;
    OV_Unsqueeze::PTR m_ov_unsqueeze = nullptr;
};

}  // namespace module
}  // namespace genai
}  // namespace ov