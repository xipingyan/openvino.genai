// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_r1_distill_qwen_1_5b.hpp"

#include <fstream>

#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace ov::genai::module {

LLMInferencePAModule_DeepSeekR1DistillQwen1_5B::LLMInferencePAModule_DeepSeekR1DistillQwen1_5B(
    const IBaseModuleDesc::PTR& desc,
    const PipelineDesc::PTR& pipeline_desc,
    const VLMModelType& model_type)
    : LLMInferencePAModule(desc, pipeline_desc, model_type) {}

bool LLMInferencePAModule_DeepSeekR1DistillQwen1_5B::initialize() {
    m_device = module_desc->device.empty() ? "CPU" : module_desc->device;

    const auto device_param = get_optional_param("device");
    if (!device_param.empty()) {
        m_device = device_param;
    }

    std::filesystem::path models_path = get_param("model_path");
    OPENVINO_ASSERT(!models_path.empty(), "LLMInferencePAModule: model_path is empty");
    OPENVINO_ASSERT(std::filesystem::is_directory(models_path),
                    "LLMInferencePAModule: model_path must be an existing directory: ",
                    models_path.string());
    check_cache_dir();

    const auto max_new_tokens_param = get_optional_param("max_new_tokens");
    if (!max_new_tokens_param.empty()) {
        m_max_new_tokens = str_to_size_t(max_new_tokens_param);
    }
    OPENVINO_ASSERT(m_max_new_tokens > 0, "LLMInferencePAModule: max_new_tokens must be > 0");

    m_generation_config = ov::genai::GenerationConfig{};
    m_generation_config.max_new_tokens = m_max_new_tokens;

    ov::AnyMap properties{};
    if (!m_cache_dir.empty()) {
        properties.insert({ov::cache_dir.name(), m_cache_dir});
    }

    const auto scheduler_config = ov::genai::utils::get_latency_oriented_scheduler_config();

    const auto llm_model_xml_path = models_path / "openvino_model.xml";
    const auto llm_model_bin_path = models_path / "openvino_model.bin";

    OPENVINO_ASSERT(std::filesystem::exists(llm_model_xml_path),
                    "LLMInferencePAModule: model XML not found: ",
                    llm_model_xml_path.string());
    OPENVINO_ASSERT(std::filesystem::exists(llm_model_bin_path),
                    "LLMInferencePAModule: model BIN not found: ",
                    llm_model_bin_path.string());

    auto read_file_to_string = [](const std::filesystem::path& path) -> std::string {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        OPENVINO_ASSERT(file.is_open(), "Failed to open file: ", path.string());
        const std::streamsize size = file.tellg();
        OPENVINO_ASSERT(size > 0, "Empty file: ", path.string());
        std::string buffer(static_cast<size_t>(size), '\0');
        file.seekg(0, std::ios::beg);
        OPENVINO_ASSERT(file.read(&buffer[0], size).good(), "Failed to read file: ", path.string());
        return buffer;
    };

    auto read_weights_tensor = [](const std::filesystem::path& path) -> ov::Tensor {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        OPENVINO_ASSERT(file.is_open(), "Failed to open weights file: ", path.string());
        const std::streamsize size = file.tellg();
        OPENVINO_ASSERT(size > 0, "Empty weights file: ", path.string());
        ov::Tensor weights(ov::element::u8, {static_cast<size_t>(size)});
        file.seekg(0, std::ios::beg);
        OPENVINO_ASSERT(file.read(reinterpret_cast<char*>(weights.data()), size).good(),
                        "Failed to read weights file: ", path.string());
        return weights;
    };

    const std::string llm_model_str = read_file_to_string(llm_model_xml_path);
    ov::Tensor llm_weights = read_weights_tensor(llm_model_bin_path);

    ModelsMap models_map;
    models_map["language"] = std::make_pair(llm_model_str, std::move(llm_weights));

    const ov::genai::Tokenizer tokenizer(models_path);
    m_pipeline = std::make_unique<ov::genai::ContinuousBatchingPipeline>(
        models_map,
        tokenizer,
        scheduler_config,
        m_device,
        std::nullopt,
        properties,
        m_generation_config);

    const auto eos_id = tokenizer.get_eos_token_id();
    if (eos_id >= 0) {
        m_stop_ids.insert(eos_id);
    }

    return true;
}

}  // namespace ov::genai::module