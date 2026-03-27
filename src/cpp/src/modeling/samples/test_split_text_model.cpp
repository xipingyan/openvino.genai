#include <openvino/openvino.hpp>
#include <map>
#include <memory>
#include <string>

#include "utils/split_text_model.hpp"

int main() {
    ov::Core core;

    std::string text_model_path_1 = "/mnt/xiping/mygithub/modular_genai/openvino.genai/tests/module_genai/cpp/test_models/Qwen3-Omni-4B-Instruct-multilingual-int4/qwen3_omni_text_model.xml";
    // Load original model, in real case, it should be loaded from file.
    auto original_model = core.read_model(text_model_path_1);

    // Call the split_text_model function with dummy node names, in real case, it should be the actual node names in the model.
    auto sub_models = ov::genai::modeling::samples::split_text_model(*original_model, "input_ids", "visual_embeds", "audio_features");

    // Print the sub-models' names.
    for (const auto& [name, model] : sub_models) {
        std::cout << "Sub-model name: " << name << std::endl;
    }

    return 0;
}