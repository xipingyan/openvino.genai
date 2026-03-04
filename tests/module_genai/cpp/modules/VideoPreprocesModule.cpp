// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"

// Tuple: (is_single_video, device, pair<models_type, models_path>)
using test_params = std::tuple<bool, std::string, std::pair<std::string, std::string>>;
using namespace ov::genai::module;

class VideoPreprocessModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    bool _is_single_video;
    std::string _models_path;
    std::string _models_type;

    bool is_single_video() const {
        return _is_single_video;
    }
    float _threshold = 1e-2;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& is_single_video = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        std::string result;
        result += (is_single_video) ? "SingleVideo_" : "BatchVideo_";
        result += device;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::map<std::string, std::string> models_map;
        std::tie(_is_single_video, _device, models_map) = GetParam();
        _models_type = models_map.begin()->first;
        _models_path = models_map.begin()->second;
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = _models_type;

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node video_preprocessor;
        video_preprocessor["type"] = "VideoPreprocessModule";
        video_preprocessor["device"] = _device;
        video_preprocessor["description"] = "Video preprocessing.";

        YAML::Node inputs;
        YAML::Node input_image;
        input_image["name"] = (is_single_video()) ? "video" : "videos";
        input_image["type"] = "OVTensor";
        inputs.push_back(input_image);
        video_preprocessor["inputs"] = inputs;

        YAML::Node outputs;
        YAML::Node pixel_values;
        pixel_values["name"] = "pixel_values";
        pixel_values["type"] = "OVTensor";
        outputs.push_back(pixel_values);
        YAML::Node grid_thw;
        grid_thw["name"] = "grid_thw";
        grid_thw["type"] = "OVTensor";
        outputs.push_back(grid_thw);
        YAML::Node pos_embeds;
        pos_embeds["name"] = "pos_embeds";
        pos_embeds["type"] = "OVTensor";
        outputs.push_back(pos_embeds);
        YAML::Node rotary_cos;
        rotary_cos["name"] = "rotary_cos";
        rotary_cos["type"] = "OVTensor";
        outputs.push_back(rotary_cos);
        YAML::Node rotary_sin;
        rotary_sin["name"] = "rotary_sin";
        rotary_sin["type"] = "OVTensor";
        outputs.push_back(rotary_sin);
        video_preprocessor["outputs"] = outputs;
    
        YAML::Node model_path;
        model_path["model_path"] = _models_path;
        video_preprocessor["params"] = model_path;
        pipeline_modules["video_preprocessor"] = video_preprocessor;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto img1 = utils::load_image(_image_paths[0]);
        EXPECT_TRUE(img1) << "Failed to load test image: " + _image_paths[0];
        if (!img1) return inputs;
        inputs["image"] = img1;
        return inputs;
    }

    const std::vector<float> expected_pixel_values = {
        0.968627, 0.968627, 0.968627, 0.968627, 0.968627, 0.968015, 0.964338, 0.960539, 0.953186, 0.945833, 0.935172, 0.924142, 0.913113, 0.902083, 0.891054, 0.880025, 0.968627, 0.968627, 0.968627, 0.968403
    };
    const ov::Shape expected_pixel_values_shape = {256, 3, 2, 16, 16};

    const std::vector<int64_t> expected_grid_thw = {
        1, 16, 16
    };
    const ov::Shape expected_grid_thw_shape = {1, 3};

    const std::vector<float> expected_pos_embeds = {
        -0.026367, -0.320312, 0.045410, -0.069336, -0.250000, 0.028809, 0.153320, 0.125977, 0.104004, 0.156250, -0.351562, 0.160156, 0.080566, -0.015747, -0.375000, 0.011719, -0.016235, -0.022705, 0.108887, 0.048828
    };
    const ov::Shape expected_pos_embeds_shape = {256, 768};

    const std::vector<float> expected_rotary_cos = {
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
    };
    const ov::Shape expected_rotary_cos_shape = {256, 64};

    const std::vector<float> expected_rotary_sin = {
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    };
    const ov::Shape expected_rotary_sin_shape = {256, 64};

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto pixel_values = pipe.get_output("pixel_values").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(pixel_values.get_shape(), expected_pixel_values_shape))
            << "pixel_values's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(pixel_values, expected_pixel_values, _threshold))
            << "pixel_values do not match expected values";

        auto grid_thw = pipe.get_output("grid_thw").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(grid_thw.get_shape(), expected_grid_thw_shape))
            << "grid_thw's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor<int64_t>(grid_thw, expected_grid_thw, _threshold))
            << "grid_thw do not match expected values";

        auto pos_embeds = pipe.get_output("pos_embeds").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(pos_embeds.get_shape(), expected_pos_embeds_shape))
            << "pos_embeds's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(pos_embeds, expected_pos_embeds, _threshold))
            << "pos_embeds do not match expected values";

        auto rotary_cos = pipe.get_output("rotary_cos").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(rotary_cos.get_shape(), expected_rotary_cos_shape))
            << "rotary_cos's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(rotary_cos, expected_rotary_cos, _threshold))
            << "rotary_cos do not match expected values";

        auto rotary_sin = pipe.get_output("rotary_sin").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(rotary_sin.get_shape(), expected_rotary_sin_shape))
            << "rotary_sin's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(rotary_sin, expected_rotary_sin, _threshold))
            << "rotary_sin do not match expected values";
    }
};

TEST_P(VideoPreprocessModuleTest, ModuleTest) {
    run();
}

auto test_image_3 = std::vector<std::string>{TEST_DATA::img_dog_120_120()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         VideoPreprocessModuleTest,
                         ::testing::Combine(::testing::Values(test_image_3),
                                            ::testing::ValuesIn(test_devices)),
                         VideoPreprocessModuleTest::get_test_case_name);