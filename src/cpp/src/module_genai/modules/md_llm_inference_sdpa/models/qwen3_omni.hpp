// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "module_genai/modules/md_llm_inference_sdpa/md_llm_inference_sdpa.hpp"

#if defined(ENABLE_MODELING_PRIVATE)

#    include "modeling/models/qwen3_omni/processing_qwen3_omni.hpp"

namespace ov::genai::module {

// Placeholder. The SDPA module is instantiated through the pipeline factory.
// Keeping this file buildable avoids breaking globbed source lists.
class LLMInferenceSDPAImpl_Qwen3Omni : public LLMInferenceSDPAModule {
public:
    LLMInferenceSDPAImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc, const VLMModelType& model_type);

    void run() override;
protected:
    // Model config
    ov::genai::modeling::models::Qwen3OmniProcessingConfig m_model_config;

    struct InputsParamsQwen3Omni : public InputsParams {
    public:
        DeclareClass_PTR_Create(InputsParamsQwen3Omni);
        ov::Tensor position_ids;
        ov::Tensor rope_deltas;
        ov::Tensor attention_mask;
        std::optional<ov::Tensor> visual_embeds = std::nullopt;
        std::optional<ov::Tensor> audio_embeds = std::nullopt;
        std::optional<ov::Tensor> audio_pos_mask = std::nullopt;
        std::optional<ov::Tensor> visual_pos_mask = std::nullopt;
        std::optional<std::vector<ov::Tensor>> deepstack_embeds = std::nullopt;
        std::vector<ov::Tensor> grid_thw;
    };
    InputsParams::PTR parse_inputs(InputsParams::PTR inputs_params = nullptr) override;

    std::string run_qwen3_omni_decode(const ov::Tensor& input_ids,
                                      const ov::Tensor& attention_mask,
                                      const ov::Tensor& position_ids,
                                      const ov::Tensor& rope_deltas,
                                      const std::optional<ov::Tensor>& visual_embeds = std::nullopt,
                                      const std::optional<ov::Tensor>& visual_pos_mask = std::nullopt,
                                      const std::optional<std::vector<ov::Tensor>>& deepstack_embeds = std::nullopt,
                                      const std::optional<ov::Tensor>& audio_embeds = std::nullopt,
                                      const std::optional<ov::Tensor>& audio_pos_mask = std::nullopt);
};

}  // namespace ov::genai::module
#else   // ENABLE_MODELING_PRIVATE not defined
namespace ov::genai::module {
class LLMInferenceSDPAImpl_Qwen3Omni : public LLMInferenceSDPAModule {
public:
    LLMInferenceSDPAImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                                   const PipelineDesc::PTR& pipeline_desc,
                                   const VLMModelType& model_type)
        : LLMInferenceSDPAModule(desc, pipeline_desc, model_type) {
        OPENVINO_THROW("LLMInferenceSDPAImpl_Qwen3Omni is not implemented in open source build");
    }
    void run() override {
        OPENVINO_THROW("LLMInferenceSDPAImpl_Qwen3Omni is not implemented in open source build");
    }
};
}  // namespace ov::genai::module
#endif  // ENABLE_MODELING_PRIVATE
