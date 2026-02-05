// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "utils.hpp"

namespace ov::genai::module {

class CSplittedModelInfer {
private:
    CSplittedModelInfer() = delete;
    CSplittedModelInfer(const std::string& model_path,
                        const std::string& device,
                        const bool& dynamic_load_model_weights = true,
                        const ov::AnyMap& properties = {});

    bool m_dynamic_load_model_weights;
    std::string m_device;

    void get_splitted_model_paths(const std::string& model_path);
    void load_model(const std::string& model_path);

    std::vector<std::string> m_splitted_model_paths;
    std::vector<ov::CompiledModel> m_compiled_models;
    std::vector<ov::InferRequest> m_infer_requests;
public:
    ~CSplittedModelInfer();
    using PTR = std::shared_ptr<CSplittedModelInfer>;
    static PTR create(const std::string& model_path,
                      const std::string& device,
                      const bool& dynamic_load_model_weights = true,
                      const ov::AnyMap& properties = {}) {
        return std::shared_ptr<CSplittedModelInfer>(new CSplittedModelInfer(model_path, device, dynamic_load_model_weights, properties));
    }

    void infer(const ov::AnyMap& inputs);

    ov::Tensor get_output_tensor(const size_t& index = 0);
};
}  // namespace ov::genai::module