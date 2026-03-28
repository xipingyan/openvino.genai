#include <openvino/openvino.hpp>
#include <map>
#include <memory>
#include <string>

#include "utils/split_text_model.hpp"

int main() {
    ov::Core core;
    core.get_versions("CPU");

    std::string text_model_path_1 = "/mnt/xiping/mygithub/modular_genai/openvino.genai/tests/module_genai/cpp/test_models/Qwen3-Omni-4B-Instruct-multilingual-int4/qwen3_omni_text_model.xml";
    // Load original model, in real case, it should be loaded from file.
    auto original_model = core.read_model(text_model_path_1);

    // Slipt the original text model into sub-models based on input_ids, visual_embeds and audio_embeds.
    // Just for compatable with ContinueBatching inference.
    auto t1 = std::chrono::high_resolution_clock::now();
    auto sub_models = ov::genai::modeling::samples::split_text_model(*original_model, "input_ids", "visual_embeds", "audio_features");
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Split text model time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    // Print the sub-models' names.
    for (const auto& [name, model] : sub_models) {
        std::cout << "Sub-model name: " << name << std::endl;
        // Print the sub-model's parameters' names.
        for (const auto& parameter : model->get_parameters()) {
            std::cout << " - Parameter name: " << parameter->get_friendly_name() << std::endl;
        }
        for (const auto& result : model->get_results()) {
            std::cout << " - Result name: " << result->get_friendly_name() << std::endl;
        }
        // Serialize the sub-models to files, in real case, it can be used for inference directly.
        std::cout << "Serialize sub-model: " << name << std::endl;
        ov::serialize(model, name + ".xml", name + ".bin");
    }


    return 0;
}