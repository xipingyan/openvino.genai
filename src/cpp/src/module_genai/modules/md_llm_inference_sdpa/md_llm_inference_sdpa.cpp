// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference_sdpa.hpp"

#include "module_genai/pipeline/module_factory.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/chat_history.hpp"
#include "module_genai/utils/com_utils.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
// #include "modeling/models/qwen3_omni/processing_qwen3_omni.hpp"
#include "module_genai/utils/profiler.hpp"

#include "module_genai/modules/md_llm_inference_sdpa/models/qwen3_omni.hpp"
#include "module_genai/modules/md_llm_inference_sdpa/models/qwen3_5.hpp"

namespace ov::genai::module {

GENAI_REGISTER_MODULE_SAME(LLMInferenceSDPAModule);

LLMInferenceSDPAModule::PTR LLMInferenceSDPAModule::create(const IBaseModuleDesc::PTR& desc,
                                                           const PipelineDesc::PTR& pipeline_desc) {
    // Model type.
    VLMModelType model_type = to_vlm_model_type(desc->model_type);
    switch (model_type)
    {
    case VLMModelType::QWEN3_VL:
    case VLMModelType::QWEN3_OMNI:
        return std::make_shared<ov::genai::module::LLMInferenceSDPAImpl_Qwen3Omni>(desc, pipeline_desc, model_type);
    case VLMModelType::QWEN3_5:
        return std::make_shared<ov::genai::module::LLMInferenceSDPAImpl_Qwen3_5>(desc, pipeline_desc, model_type);
    default:
        break;
    }
    GENAI_INFO("Model type '" + desc->model_type + "' falls back to default LLMInferenceSDPAModule implementation.");
    return PTR(new LLMInferenceSDPAModule(desc, pipeline_desc, model_type));
}

// ============================================================================
// Static YAML config description (for --print-config)
// ============================================================================

void LLMInferenceSDPAModule::print_static_config() {
    std::cout << R"(
global_context:
  model_type: "qwen3_5"
pipeline_modules:
  llm_inference_sdpa:
    type: "LLMInferenceSDPAModule"
    description: "LLM module using SDPA (stateful) backend — supports text & VL modes"
    device: "CPU"
    inputs:
      # ---- Text mode inputs (required) ----
      - name: "input_ids"            # Tokenized input ids
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      # ---- VL mode inputs (additional, optional) ----
      - name: "visual_embeds"        # [Optional] visual embeddings from vision encoder
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "visual_pos_mask"      # [Optional] boolean mask marking visual token positions
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "grid_thw"             # [Optional] grid dimensions [N,3] for 3D MRoPE position ids
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "deepstack_embeds"     # [Optional] deepstack visual embeddings for Qwen3-Omni
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "audio_embeds"      # [Optional] audio embeddings from audio preprocessor for Qwen3-Omni
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "audio_pos_mask"        # [Optional] boolean mask marking audio token positions for Qwen3-Omni
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "generated_text"
        type: "String"
    params:
      model_path: "model_path"              # Directory containing config.json + IR files
      cache_dir: "cache_dir"                # Optional directory for caching compiled models, e.g. for GPU device, recommended to set this to a local disk path for better performance
      model_cfg_path: "model_config.json"   # Optional fallback when model_path is not provided
      max_new_tokens: "256"
      text_device: ""                       # Override device for text model (e.g. CPU for VL TDR avoidance)
    )" << std::endl;
}

// ============================================================================
// Construction / Destruction
// ============================================================================
LLMInferenceSDPAModule::LLMInferenceSDPAModule(const IBaseModuleDesc::PTR& desc,
                                               const PipelineDesc::PTR& pipeline_desc,
                                               const VLMModelType& model_type)
    : IBaseModule(desc, pipeline_desc),
      m_model_type(model_type) {
    if (!initialize(model_type)) {
        GENAI_ERR("Failed to initialize LLMInferenceSDPAModule");
    }
}

LLMInferenceSDPAModule::~LLMInferenceSDPAModule() {}

// ============================================================================
// Initialization
// ============================================================================

bool LLMInferenceSDPAModule::initialize(const VLMModelType& model_type) {
    const auto& params = module_desc->params;

    // Required model_path param.
    std::filesystem::path models_path = get_param("model_path");
    if (models_path.extension() == ".xml") {
        if (!std::filesystem::exists(models_path)) {
            GENAI_ERR("Specified model_path XML file does not exist: " + models_path.string());
            return false;
        }
        m_models_ir = models_path;
        models_path = models_path.parent_path();
    } else {
        if (models_path.empty()) {
            models_path = get_param("model_cfg_path");
        }
        if (models_path.empty() || !std::filesystem::is_directory(models_path)) {
            GENAI_ERR("LLMInferenceSDPAModule: model_path is required and must be an existing directory");
            return false;
        }
    }
    m_models_path = models_path;
    
    check_cache_dir();

    // Override max_new_tokens from params
    auto max_new_tokens_param = get_optional_param("max_new_tokens");
    if (!max_new_tokens_param.empty()) {
        m_max_new_tokens = str_to_size_t(max_new_tokens_param);
    }

    // Resolve text device
    auto text_device_param = get_optional_param("text_device");
    if (!text_device_param.empty()) {
        m_device = text_device_param;
    }
    GENAI_INFO("LLMInferenceSDPAModule: compiling LLM with device: " + m_device);

    // Load tokenizer.
    try {
        m_tokenizer = std::make_unique<ov::genai::Tokenizer>(models_path);
    } catch (const std::exception& e) {
        GENAI_ERR("LLMInferenceSDPAModule: tokenizer init failed: " + std::string(e.what()));
        // Non-fatal — upstream provides input_ids directly
    }

    // Collect stop token ids from all available sources:
    //   1) Tokenizer EOS token
    //   2) config.json text_config.eos_token_id (Qwen3.5 = 248044)
    if (m_tokenizer) {
        try {
            auto eid = m_tokenizer->get_eos_token_id();
            if (eid >= 0) m_stop_ids.insert(eid);
        } catch (...) {}
    }

    GENAI_INFO("LLMInferenceSDPAModule initialised (vl_ir=" +
               std::string(m_text_uses_vl_ir ? "true" : "false") +
               ", device=" + m_device + ")");
    return true;
}

// ============================================================================
// Text decode (no vision inputs) — stateful prefill + greedy decode
// ============================================================================

LLMInferenceSDPAModule::InputsParams::PTR LLMInferenceSDPAModule::parse_inputs(LLMInferenceSDPAModule::InputsParams::PTR inputs_params) {
    if (inputs_params == nullptr) {
        inputs_params = InputsParams::create();
    }

    // input_ids
    inputs_params->input_ids = get_input("input_ids").as<ov::Tensor>();
    return inputs_params;
};

void LLMInferenceSDPAModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    OPENVINO_THROW("LLMInferenceSDPAModule is an abstract base class and cannot be run directly");

    // prepare_inputs();
    // auto inputs_params = parse_inputs();

    // prefill();
    // decode();
}

namespace LLMInferenceSDPAModule_Utils {
ov::Tensor build_attention_mask(const ov::Tensor& input_ids) {
    const size_t batch = input_ids.get_shape()[0];
    const size_t seq_len = input_ids.get_shape()[1];
    ov::Tensor attention_mask(ov::element::i64, {batch, seq_len});
    std::fill_n(attention_mask.data<int64_t>(), batch * seq_len, int64_t{1});
    return attention_mask;
}

ov::Tensor make_zeros(ov::element::Type t, ov::Shape shape) {
    ov::Tensor out(t, shape);
    std::memset(out.data(), 0, out.get_byte_size());
    return out;
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor t(ov::element::i32, {batch});
    for (size_t i = 0; i < batch; ++i)
        t.data<int32_t>()[i] = static_cast<int32_t>(i);
    return t;
}

int64_t argmax_last(const ov::Tensor& logits) {
    const auto s = logits.get_shape();
    if (s.size() != 3 || s[0] != 1)
        throw std::runtime_error("logits must be [1,S,V]");
    const size_t offset = (s[1] - 1) * s[2];

    auto argmax = [&](auto* data, size_t n) -> int64_t {
        auto best = data[0];
        size_t idx = 0;
        for (size_t i = 1; i < n; ++i)
            if (data[i] > best) {
                best = data[i];
                idx = i;
            }
        return static_cast<int64_t>(idx);
    };
    if (logits.get_element_type() == ov::element::f16)
        return argmax(logits.data<const ov::float16>() + offset, s[2]);
    if (logits.get_element_type() == ov::element::bf16)
        return argmax(logits.data<const ov::bfloat16>() + offset, s[2]);
    if (logits.get_element_type() == ov::element::f32)
        return argmax(logits.data<const float>() + offset, s[2]);
    throw std::runtime_error("Unsupported logits dtype");
}

}  // namespace LLMInferenceSDPAModule_Utils

}  // namespace ov::genai::modules
