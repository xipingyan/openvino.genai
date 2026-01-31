#include <iostream>
#include <openvino/genai/module_genai/pipeline.hpp>

#include <stdexcept>

#include "utils/vision_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "utils/utils.hpp"

inline ov::AnyMap parse_inputs_from_yaml_cfg_for_vlm(const std::filesystem::path& cfg_yaml_path,
                                                     const std::string& prompt = std::string{},
                                                     const std::string& image_path = std::string{},
                                                     const std::string& video_path = std::string{}) {
    ov::AnyMap inputs;
    YAML::Node input_params = utils::find_param_module_in_yaml(cfg_yaml_path);

    // Loop input_params to find "prompt", "image", "video"
    for (const auto& param : input_params) {
        std::string param_name = param.first.as<std::string>();

        if (utils::contain_key(param_name, {"prompt"})) {
            inputs[param_name] = prompt;
            if (prompt.empty()) {
                throw std::runtime_error("Prompt string is empty.");
            }
        } else if (utils::contain_key(param_name, {"image", "img"})) {
            if (image_path.empty()) {
                throw std::runtime_error("Image path is empty.");
            }
            ov::Tensor image_tensor = image_utils::load_image(image_path);
            inputs[param_name] = image_tensor;
        } else if (utils::contain_key(param_name, {"video"})) {
            if (video_path.empty()) {
                throw std::runtime_error("Video path is empty.");
            }
            ov::Tensor video_tensor = image_utils::load_video(video_path);
            inputs[param_name] = video_tensor;
        }
    }
    return inputs;
}

int main(int argc, char* argv[]) {
    try {
        if (argc <= 2 || argc >= 6) {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                     " <Config_Yaml> <Prompt> <Image_Path:Optional> <Video_Path:Optional>");
        }

        std::filesystem::path config_path = argv[1];
        std::string prompt = argv[2];
        std::string img_path = argc > 3 ? argv[3] : std::string{};
        std::string video_path = argc > 4 ? argv[4] : std::string{};

        ov::AnyMap inputs = parse_inputs_from_yaml_cfg_for_vlm(config_path, prompt, img_path, video_path);

        ov::genai::module::ModulePipeline pipe(config_path);

        pipe.generate(inputs);

        std::cout << "Generation Result: " << pipe.get_output("generated_text").as<std::string>() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}