// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>

#include "module_genai/modules/md_text_to_speech/md_text_to_speech.hpp"

#if defined(ENABLE_MODELING_PRIVATE)

#    include "modeling_private/models/qwen3_omni/processing_qwen3_omni.hpp"

namespace ov::genai::module {

class TextToSpeechImpl_Qwen3Omni : public TextToSpeechModule {
public:
	TextToSpeechImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
							   const PipelineDesc::PTR& pipeline_desc,
							   const VLMModelType& model_type);

	void run() override;

private:
	bool initialize();
	std::pair<ov::Tensor, int> qwen3_omni_text_to_speech(const std::string& text);

private:
	modeling::models::Qwen3OmniProcessingConfig m_config;
};

}  // namespace ov::genai::module

#else

namespace ov::genai::module {

class TextToSpeechImpl_Qwen3Omni : public TextToSpeechModule {
public:
	TextToSpeechImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
							   const PipelineDesc::PTR& pipeline_desc,
							   const VLMModelType& model_type)
		: TextToSpeechModule(desc, pipeline_desc, model_type) {
		OPENVINO_THROW("TextToSpeechImpl_Qwen3Omni is not implemented in open source build");
	}

	void run() override {
		OPENVINO_THROW("TextToSpeechImpl_Qwen3Omni is not implemented in open source build");
	}
};

}  // namespace ov::genai::module

#endif  // ENABLE_MODELING_PRIVATE
