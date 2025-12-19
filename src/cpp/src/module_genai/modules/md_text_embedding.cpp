// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_text_embedding.hpp"
#include "visual_language/embedding_model.hpp"
#include "circular_buffer_queue.hpp"

namespace ov {
namespace genai {
namespace module {

void TextEmbeddingModule::print_static_config() {
    std::cout << R"(
  text_embedding:                     # Module Name
    type: "TextEmbeddingModule"
    description: "Compute text embeddings from token ids."
    device: "GPU"
    inputs:
      - name: "input_ids"
        type: "OVTensor"              # Token ids tensor from TextEncoderModule
        source: "ParentModuleName.input_ids"
    outputs:
      - name: "input_embedding"
        type: "OVTensor"              # Or OVRemoteTensor depending on device / use case
    params:
      model_path: "path/to/text_embeddings_model_dir"
      scale_emb: "1.0"
    )" << std::endl;
}

TextEmbeddingModule::TextEmbeddingModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    if (!initialize()) {
        std::cerr << "Failed to initiate TextEmbeddingModule" << std::endl;
    }
}

bool TextEmbeddingModule::initialize() {
    const auto& params = module_desc->params;
    auto it_dir = params.find("model_path");
    if (it_dir == params.end()) {
        std::cerr << "TextEmbeddingModule[" << module_desc->name << "]: 'model_path' not found in params" << std::endl;
        return false;
    }

    std::filesystem::path model_dir = it_dir->second;

    float scale_emb = 1.0f;
    auto it_scale = params.find("scale_emb");
    if (it_scale != params.end()) {
        try {
            scale_emb = std::stof(it_scale->second);
        } catch (const std::exception&) {
            std::cerr << "TextEmbeddingModule[" << module_desc->name << "]: invalid 'scale_emb' value, fallback to 1.0" << std::endl;
        }
    }

    std::string device = module_desc->device.empty() ? "GPU" : module_desc->device;
    ov::AnyMap properties{};
    m_embedding_model = EmbeddingsModel::create(model_dir, scale_emb, device, properties);

    return true;
}

void TextEmbeddingModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;

    OPENVINO_ASSERT(m_embedding_model, "TextEmbeddingModule is not initialized. Call initialize() first.");

    prepare_inputs();
    OPENVINO_ASSERT(this->inputs.count("input_ids") != 0,
                    "TextEmbeddingModule expects 'input_ids' in its inputs.");
    ov::Tensor input_ids = this->inputs["input_ids"].data.as<ov::Tensor>();

    bool return_remote_tensor = true;
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding_model->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();

    ov::Tensor text_embeds = m_embedding_model->infer(req, input_ids, return_remote_tensor);

    this->outputs["input_embedding"].data = text_embeds;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
