// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "module_genai/pipeline/module.hpp"
#include "module_genai/pipeline/module_type.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "module_genai/utils/com_utils.hpp"

namespace ov::genai::module {

/// @brief LLM inference module using SDPA (stateful) backend.
///
/// Supports two modes driven by the available inputs:
///   - **Text mode**: receives "input_ids" (OVTensor)
///     → builds attention_mask internally, runs stateful prefill + decode,
///     outputs "generated_text".
///   - **VL mode**: additionally receives "visual_embeds" (OVTensor),
///     "visual_pos_mask" (OVTensor), and "grid_thw" (OVTensor) → computes 3D MRoPE
///     position_ids via build_plan, scatters visual embeddings, then runs stateful
///     prefill + decode, outputs "generated_text".
///
/// Unlike LLMInferenceModule (PA / ContinuousBatching), this module compiles the
/// text model directly via ov::Core and drives a single InferRequest with
/// explicit KV-cache state management – i.e. the SDPA attention path.
class LLMInferenceSDPAModule : public IBaseModule {
protected:
    LLMInferenceSDPAModule() = delete;
    LLMInferenceSDPAModule(const IBaseModuleDesc::PTR& desc,
                           const PipelineDesc::PTR& pipeline_desc,
                           const VLMModelType& model_type);

public:
    ~LLMInferenceSDPAModule();

    using PTR = std::shared_ptr<LLMInferenceSDPAModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc);
    static void print_static_config();

    void run() override;

protected:
    // Abstracted common implementation for different model types.
    struct InputsParams {
    public:
        DeclareClass_PTR_Create(InputsParams);
        ov::Tensor input_ids;
    };
    virtual InputsParams::PTR parse_inputs(InputsParams::PTR inputs_params = nullptr);

    virtual void prefill() {
        // No separate prefill step needed for this module, as it handles prefill logic internally in run().
    };

    virtual void decode() {
        // No separate stateful decode step needed for this module, as it handles decode logic internally in run().
    };

protected:    
    bool initialize(const VLMModelType& model_type);

    // Compiled text model + infer request
    std::optional<ov::CompiledModel> m_compiled_text;
    bool m_text_uses_vl_ir = false;

    // Stop token tracking
    std::set<int64_t> m_stop_ids;

    // Tokenizer (for text mode and decoding)
    std::unique_ptr<ov::genai::Tokenizer> m_tokenizer;

    // Max tokens to generate (default 256, overridden by params)
    size_t m_max_new_tokens = 256;

    // Device used for the text model (for profiling output)
    std::string m_device = "CPU";
    VLMModelType m_model_type;
    std::filesystem::path m_models_ir;
    std::filesystem::path m_models_path;
};

REGISTER_MODULE_CONFIG(LLMInferenceSDPAModule);

namespace LLMInferenceSDPAModule_Utils {
ov::Tensor build_attention_mask(const ov::Tensor& input_ids);

ov::Tensor make_zeros(ov::element::Type t, ov::Shape shape);
ov::Tensor make_beam_idx(size_t batch);
int64_t argmax_last(const ov::Tensor& logits);

inline bool dump_performance_enabled() {
    static const bool enabled = utils::check_env_variable("DUMP_PERFORMANCE");
    return enabled;
}

inline double elapsed_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace LLMInferenceSDPAModule_Utils

}  // namespace ov::genai::module
