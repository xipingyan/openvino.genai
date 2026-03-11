// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"

using namespace ov::genai::module;

namespace AudioEncoderModuleTest {

struct AudioEncoderModuleTestData {
    std::string model_type;
    std::string model_path;
    std::vector<ov::Tensor> input_features;
    std::vector<ov::Tensor> feature_attention_mask;

    std::vector<float> expected_audio_features;
    ov::Shape expected_audio_features_shape;
    std::vector<int64_t> expected_audio_feature_lengths;
    ov::Shape expected_audio_feature_lengths_shape;
};

using test_params = std::tuple<AudioEncoderModuleTestData, std::string>;

AudioEncoderModuleTestData audio_encoder_test_data() {
    AudioEncoderModuleTestData data;
    data.model_type = "qwen3_omni";
    data.model_path = TEST_MODEL::Qwen3_Omni_4B_Instruct_Multilingual() + "/qwen3_omni_audio_encoder.xml";
    ov::Tensor input_features = ModuleTestBase::ut_randn_tensor(ov::Shape{128, 290}, 42);
    ov::Tensor feature_attention_mask = ov::Tensor(ov::element::i32, ov::Shape{290});
    {
        auto* mask_data = feature_attention_mask.data<int32_t>();
        for (size_t i = 0; i < 290; ++i) {
            mask_data[i] = 1;
        }
    }
    data.input_features = {input_features};
    data.feature_attention_mask = {feature_attention_mask};
    data.expected_audio_features = {
        -0.0298741, 0.0159347, 0.010697, 0.00146015, -0.0165195, -0.00119845, -0.0512646, -0.00313947, 0.0045964, -0.0175942, -0.0104786, -0.0280771, 0.018191, -0.0118281, 0.000679443, -0.010354, -0.0319885, 0.0144178, 0.0178998, 0.00494952
    };
    data.expected_audio_features_shape = ov::Shape{37, 2560};
    data.expected_audio_feature_lengths = {
        290
    };
    data.expected_audio_feature_lengths_shape = ov::Shape{1};
    return data;
}

class AudioEncoderModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    AudioEncoderModuleTestData m_test_data;
    float m_threshold = 1e-1;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        auto test_data = std::get<0>(obj.param);
        auto device = std::get<1>(obj.param);

        std::filesystem::path model_path(test_data.model_path);
        return test_data.model_type + "_" + model_path.filename().replace_extension("").string() + "_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = m_test_data.model_type;

        YAML::Node pipeline_modules;
        YAML::Node audio_encoder;
        audio_encoder["type"] = "AudioEncoderModule";
        audio_encoder["device"] = m_device;
        audio_encoder["description"] = "Audio encoder for Qwen 3-Omni.";
        
        // Define inputs
        YAML::Node inputs;
        inputs.push_back(input_node("input_features", "VecOVTensor"));
        inputs.push_back(input_node("feature_attention_mask", "VecOVTensor"));
        audio_encoder["inputs"] = inputs;

        // Define outputs
        YAML::Node outputs;
        outputs.push_back(output_node("audio_features", "VecOVTensor"));
        outputs.push_back(output_node("audio_feature_lengths", "OVTensor"));
        audio_encoder["outputs"] = outputs;

        // Define parameters
        YAML::Node params;
        params["model_path"] = m_test_data.model_path;
        audio_encoder["params"] = params;

        pipeline_modules["audio_encoder"] = audio_encoder;
        config["pipeline_modules"] = pipeline_modules;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["input_features"] = m_test_data.input_features;
        inputs["feature_attention_mask"] = m_test_data.feature_attention_mask;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto audio_features = pipe.get_output("audio_features").as<std::vector<ov::Tensor>>();
        auto audio_feature_lengths = pipe.get_output("audio_feature_lengths").as<ov::Tensor>();

        EXPECT_TRUE(compare_shape(audio_features[0].get_shape(), m_test_data.expected_audio_features_shape)) << "audio_features shape does not match expected shape.";
        EXPECT_TRUE(compare_big_tensor(audio_features[0], m_test_data.expected_audio_features, m_threshold)) << "audio_features values do not match expected values.";

        EXPECT_TRUE(compare_shape(audio_feature_lengths.get_shape(), m_test_data.expected_audio_feature_lengths_shape)) << "audio_feature_lengths shape does not match expected shape.";
        EXPECT_TRUE(compare_big_tensor<int64_t>(audio_feature_lengths, m_test_data.expected_audio_feature_lengths, 0)) << "audio_feature_lengths values do not match expected values.";
    }
};

TEST_P(AudioEncoderModuleTest, ModuleTest) {
    run();
}

auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};
auto test_datas = std::vector<AudioEncoderModuleTestData>{audio_encoder_test_data()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                            AudioEncoderModuleTest,
                            ::testing::Combine(::testing::ValuesIn(test_datas),
                                                ::testing::ValuesIn(test_devices)),
                            AudioEncoderModuleTest::get_test_case_name);

}