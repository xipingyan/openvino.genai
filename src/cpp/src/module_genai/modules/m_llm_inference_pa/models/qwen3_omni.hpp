// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "module_genai/modules/m_llm_inference_pa/md_llm_inference_pa.hpp"

namespace ov::genai::module {


class LLMInferencePAModule_Qwen3Omni : public LLMInferencePAModule {
public:
    LLMInferencePAModule_Qwen3Omni(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc, const VLMModelType& model_type);

    void run() override;

protected:
    struct InputsParamsQwen3_OMNI : public InputsParams {
    public:
        DeclareClass_PTR_Create(InputsParamsQwen3_OMNI);
        ov::Tensor attention_mask;

        ov::Tensor visual_embeds;
        ov::Tensor visual_pos_mask;
        std::vector<ov::Tensor> grid_thw;
        ov::Tensor position_ids;
        ov::Tensor rope_deltas;
        std::optional<std::vector<ov::Tensor>> deepstack_embeds = std::nullopt;
    };
    InputsParams::PTR parse_inputs(InputsParams::PTR inputs_params = nullptr) override;

    bool initialize() override;
private:
    ov::InferRequest m_merge_embeds_infer_request;
};

namespace LLMInferencePAModule_Utils {

}  // namespace LLMInferencePAModule_Utils

}  // namespace ov::genai::module
