// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "module_genai/modules/md_llm_inference_sdpa/md_llm_inference_sdpa.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

// #ifndef ENABLE_OPENVINO_NEW_ARCH
// #    define ENABLE_OPENVINO_NEW_ARCH 1
// #endif

#ifdef ENABLE_OPENVINO_NEW_ARCH

namespace ov::genai::module {

// Placeholder. The SDPA module is instantiated through the pipeline factory.
// Keeping this file buildable avoids breaking globbed source lists.
class LLMInferenceSDPAImpl_Qwen3_5 : public LLMInferenceSDPAModule {
public:
    LLMInferenceSDPAImpl_Qwen3_5(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc, const VLMModelType& model_type);

    void run() override;

protected:
    // Model config
    ov::genai::modeling::models::Qwen3_5Config m_model_config;

    struct InputsParamsQwen3_5 : public InputsParams {
    public:
        DeclareClass_PTR_Create(InputsParamsQwen3_5);
        ov::Tensor attention_mask;

        bool is_vl = false;
        ov::Tensor visual_embeds;
        ov::Tensor visual_pos_mask;
        std::vector<ov::Tensor> grid_thw;
        ov::Tensor position_ids;
        ov::Tensor rope_deltas;
        std::optional<std::vector<ov::Tensor>> deepstack_embeds = std::nullopt;

        // text
        ov::genai::modeling::models::Qwen3_5InputPlan plan;
    };
    InputsParams::PTR parse_inputs(InputsParams::PTR inputs_params = nullptr) override;

private:
    // --- Text-only decode (no vision inputs) ---
    std::string run_text_decode(const ov::Tensor& input_ids,
                                const ov::Tensor& attention_mask,
                                const ov::Tensor& position_ids,
                                const ov::Tensor& rope_deltas);

    // --- VL decode (with visual embeddings) ---
    std::string run_vl_decode(const ov::Tensor& input_ids,
                              const ov::Tensor& attention_mask,
                              const ov::Tensor& position_ids,
                              const ov::Tensor& rope_deltas,
                              const ov::Tensor& visual_embeds,
                              const ov::Tensor& visual_pos_mask,
                              const std::optional<std::vector<ov::Tensor>>& deepstack_embeds = std::nullopt);
};

namespace LLMInferenceSDPAModule_Utils {
std::string quant_suffix();
static bool has_ir_pair(const std::filesystem::path& xml, const std::filesystem::path& bin);
static bool has_model_input(const std::shared_ptr<ov::Model>& m, const std::string& name);
}  // namespace LLMInferenceSDPAModule_Utils

}  // namespace ov::genai::module

#else   // ENABLE_OPENVINO_NEW_ARCH not defined
namespace ov::genai::module {
class LLMInferenceSDPAImpl_Qwen3_5 : public LLMInferenceSDPAModule {
public:
    LLMInferenceSDPAImpl_Qwen3_5(const IBaseModuleDesc::PTR& desc,
                                 const PipelineDesc::PTR& pipeline_desc,
                                 const VLMModelType& model_type)
        : LLMInferenceSDPAModule(desc, pipeline_desc, model_type) {
        OPENVINO_THROW("LLMInferenceSDPAImpl_Qwen3_5 is not implemented in open source build");
    }
    void run() override {}
};
}  // namespace ov::genai::module
#endif  // ENABLE_OPENVINO_NEW_ARCH