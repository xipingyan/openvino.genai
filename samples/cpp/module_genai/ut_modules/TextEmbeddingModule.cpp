// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"
#include "visual_language/embedding_model.hpp"
#include "circular_buffer_queue.hpp"

class TextEmbeddingModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(TextEmbeddingModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "input_ids"
        type: "OVTensor"

  text_embedding:
    type: "TextEmbeddingModule"
    device: "GPU"
    inputs:
      - name: "input_ids"
        type: "OVTensor"
        source: "pipeline_params.input_ids"
    outputs:
      - name: "input_embedding"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
      scale_emb: "1.0"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "input_embedding"
        type: "OVTensor"
        source: "text_embedding.input_embedding"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* data_ptr = input_ids.data<int64_t>();
        std::vector<int64_t> values = {1986, 374, 264, 6077, 9934, 13};
        std::copy(values.begin(), values.end(), data_ptr);
        
        inputs["input_ids"] = input_ids;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("input_embedding").as<ov::Tensor>();

        std::string path = "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/";
        std::string device = "GPU";
        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* data_ptr = input_ids.data<int64_t>();
        std::vector<int64_t> values = {1986, 374, 264, 6077, 9934, 13};
        std::copy(values.begin(), values.end(), data_ptr);
        auto embed = ov::genai::EmbeddingsModel::create(path, 1.0, device, {});
        bool return_remote_tensor = false;
        ov::genai::CircularBufferQueueElementGuard<ov::genai::EmbeddingsRequest> embeddings_request_guard(embed->get_request_queue().get());
        ov::genai::EmbeddingsRequest& req = embeddings_request_guard.get();
        ov::Tensor expected_text_embeds = embed->infer(req, input_ids, return_remote_tensor);

        CHECK(compare_tensors(output, expected_text_embeds), "input_embedding do not match expected values");
    }
};

REGISTER_MODULE_TEST(TextEmbeddingModuleTest);
