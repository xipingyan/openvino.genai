// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/modules/m_llm_inference_pa/md_llm_inference_pa.hpp"

namespace ov::genai::module {

class LLMInferencePAModule_DeepSeekR1DistillQwen1_5B : public LLMInferencePAModule {
public:
    LLMInferencePAModule_DeepSeekR1DistillQwen1_5B(const IBaseModuleDesc::PTR& desc,
                                                   const PipelineDesc::PTR& pipeline_desc,
                                                   const VLMModelType& model_type);

protected:
    bool initialize() override;
};

}  // namespace ov::genai::module