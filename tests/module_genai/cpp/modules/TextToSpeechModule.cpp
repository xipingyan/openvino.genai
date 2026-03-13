// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef ENABLE_OPENVINO_NEW_ARCH

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"

using namespace ov::genai::module;

namespace TextToSpeechTest {

struct TextToSpeechModuleTestData {
    std::string model_type;
    std::string config_path;
    std::string tokenizer_path;
    std::string embedding_model_path;
    std::string prefill_model_path;
    std::string decode_model_path;
    std::string codec_embedding_model_path;
    std::string code_predictor_ar_model_path;
    std::string code_predictor_single_codec_embed_model_path;
    std::string code_predictor_single_codec_embedding_model_path;
    std::string speech_decoder_model_path;

    std::string input_text;

    std::vector<float> expected_audio;
    ov::Shape expected_audio_shape;
    int expected_sample_rate;
};

// Define the test parameters as a tuple of TextToSpeechModuleTestData and device string
using test_params = std::tuple<TextToSpeechModuleTestData, std::string>;

TextToSpeechModuleTestData text_to_speech_test_data() {
    const std::string model_path = TEST_MODEL::Qwen3_Omni_4B_Instruct_Multilingual();
    TextToSpeechModuleTestData data;
    data.model_type = "qwen3_omni";
    data.config_path = model_path + "/config.json";
    data.tokenizer_path = model_path;
    data.embedding_model_path = model_path + "/qwen3_omni_talker_embedding_model.xml";
    data.prefill_model_path = model_path + "/qwen3_omni_talker_prefill_model.xml";
    data.decode_model_path = model_path + "/qwen3_omni_talker_decode_model.xml";
    data.codec_embedding_model_path = model_path + "/qwen3_omni_talker_codec_embedding_model.xml";
    data.code_predictor_ar_model_path = model_path;
    data.code_predictor_single_codec_embed_model_path = model_path;
    data.code_predictor_single_codec_embedding_model_path = model_path + "/qwen3_omni_code_predictor_codec_embed_model.xml";
    data.speech_decoder_model_path = model_path + "/qwen3_omni_speech_decoder_model.xml";
    data.input_text = "Hello, world!";
    // The expected audio values are not verified in this test, so we can leave them empty.
    data.expected_audio = {
        -0.000125509f, -0.00050764f, -0.000696397f, -0.00147474f, -0.0017727f, -0.00151176f, -0.00133568f, -0.000976406f, -0.00114874f, -0.00173928f, -0.00176832f, 0.000312283f, 0.00301303f, 0.00553984f, 0.00677973f, 0.00661683f, 0.00406882f, -0.00176349f, -0.00481414f, -0.00100633f
    };
    data.expected_audio_shape = {1, 34573};  // Shape is also not verified, set to {0} as a placeholder.
    data.expected_sample_rate = 24000;  // We can verify the sample rate since it's deterministic.
    return data;
}

class TextToSpeechModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    TextToSpeechModuleTestData m_test_data;
    float m_threshold = 1e-1;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        auto test_data = std::get<0>(obj.param);
        auto device = std::get<1>(obj.param);
        return test_data.model_type + "_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        auto param = GetParam();
        m_test_data = std::get<0>(param);
        m_device = std::get<1>(param);
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = m_test_data.model_type;
        config["global_context"]["device"] = m_device;
        YAML::Node pipeline_modules;
        YAML::Node tts;
        tts["type"] = "TextToSpeechModule";
        tts["device"] = m_device;
        YAML::Node inputs;
        inputs.push_back(input_node("text", "String"));
        tts["inputs"] = inputs;
        YAML::Node outputs;
        outputs.push_back(output_node("audios", "VecOVTensor"));
        outputs.push_back(output_node("sample_rates", "VecInt"));
        outputs.push_back(output_node("generated_texts", "VecString"));
        tts["outputs"] = outputs;
        YAML::Node params;
        params["config_path"] = m_test_data.config_path;
        params["tokenizer_path"] = m_test_data.tokenizer_path;
        params["embedding_model_path"] = m_test_data.embedding_model_path;
        params["prefill_model_path"] = m_test_data.prefill_model_path;
        params["decode_model_path"] = m_test_data.decode_model_path;
        params["codec_embedding_model_path"] = m_test_data.codec_embedding_model_path;
        params["code_predictor_ar_model_path"] = m_test_data.code_predictor_ar_model_path;
        params["code_predictor_single_codec_embed_model_path"] = m_test_data.code_predictor_single_codec_embed_model_path;
        params["code_predictor_single_codec_embedding_model_path"] = m_test_data.code_predictor_single_codec_embedding_model_path;
        params["speech_decoder_model_path"] = m_test_data.speech_decoder_model_path;
        tts["params"] = params;
        pipeline_modules["text_to_speech"] = tts;
        config["pipeline_modules"] = pipeline_modules;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["text"] = m_test_data.input_text;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto audios = pipe.get_output("audios").as<std::vector<ov::Tensor>>();
        auto sample_rates = pipe.get_output("sample_rates").as<std::vector<int>>();
        auto generated_texts = pipe.get_output("generated_texts").as<std::vector<std::string>>();

        EXPECT_TRUE(compare_shape(audios[0].get_shape(), m_test_data.expected_audio_shape))
            << "audios shape does not match expected shape.";
        EXPECT_TRUE(compare_big_tensor(audios[0], m_test_data.expected_audio, m_threshold))
            << "audios values do not match expected values.";
        EXPECT_EQ(sample_rates[0], m_test_data.expected_sample_rate)
            << "sample rate does not match expected value.";
        EXPECT_EQ(generated_texts[0], m_test_data.input_text)
            << "generated text does not match input text.";
    }
};

TEST_P(TextToSpeechModuleTest, ModuleTest) {
    run();
}

auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};
auto test_datas = std::vector<TextToSpeechModuleTestData>{text_to_speech_test_data()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
    TextToSpeechModuleTest,
    ::testing::Combine(::testing::ValuesIn(test_datas),
    ::testing::ValuesIn(test_devices)),
    TextToSpeechModuleTest::get_test_case_name);


}
#endif
