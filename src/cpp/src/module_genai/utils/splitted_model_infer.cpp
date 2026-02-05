#include "splitted_model_infer.hpp"

#include <regex>

namespace ov::genai::module {

CSplittedModelInfer::CSplittedModelInfer(const std::string& model_path,
                                         const std::string& device,
                                         const bool& dynamic_load_model_weights,
                                         const ov::AnyMap& properties)
    : m_dynamic_load_model_weights(dynamic_load_model_weights),
      m_device(device) {
    // parse all splitted model paths, model_path is the directory that contains all splitted models
    get_splitted_model_paths(model_path);

}

void CSplittedModelInfer::get_splitted_model_paths(const std::string& model_path) {
    // each splitted model should be named as **_l{index}_*.xml. For example: model_l0_.xml, model_l1_.xml
    m_splitted_model_paths.clear();

    std::regex pattern(R"(.*_l(\d+)_.*\.xml)");
    std::map<int, std::string> sorted_paths;

    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".xml") {
            std::string filename = entry.path().filename().string();
            std::smatch match;

            if (std::regex_search(filename, match, pattern)) {
                // check only contains one pattern is matched to avoid wrong file name like model_l0_l1_.xml
                OPENVINO_ASSERT(match.size() == 2,
                                "Invalid model file name: " + filename +
                                    ". Expected format: **_l{index}_*.xml, and only one index pattern is allowed.");

                int index = std::stoi(match[1].str());
                sorted_paths[index] = entry.path().string();
            }
        }
    }

    for (int i = 0; i < static_cast<int>(sorted_paths.size()); ++i) {
        auto it = sorted_paths.find(i);
        OPENVINO_ASSERT(
            it != sorted_paths.end(),
            "Model partitions should be continuous and start from index 0. Missing index: " + std::to_string(i));
        m_splitted_model_paths.push_back(it->second);
    }

    OPENVINO_ASSERT(!m_splitted_model_paths.empty(), "No splitted models found in " + model_path);
}

void CSplittedModelInfer::load_model(const std::string& model_path) {}

CSplittedModelInfer::~CSplittedModelInfer() {}

void CSplittedModelInfer::infer(const ov::AnyMap& inputs) {
    
}

ov::Tensor CSplittedModelInfer::get_output_tensor(const size_t& index) {
    return ov::Tensor();
}
}  // namespace ov::genai::module