// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_text_to_speech.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "module_genai/modules/md_text_to_speech/models/qwen3_omni.hpp"
#include "module_genai/pipeline/module_factory.hpp"
#include "utils.hpp"

namespace ov::genai::module {

GENAI_REGISTER_MODULE_SAME(TextToSpeechModule);

TextToSpeechModule::PTR TextToSpeechModule::create(const IBaseModuleDesc::PTR& desc,
                                                   const PipelineDesc::PTR& pipeline_desc) {
    const VLMModelType model_type = to_vlm_model_type(desc->model_type);
    switch (model_type) {
    case VLMModelType::QWEN3_OMNI:
        return std::make_shared<TextToSpeechImpl_Qwen3Omni>(desc, pipeline_desc, model_type);
    default:
        break;
    }

    GENAI_INFO("Model type '" + desc->model_type + "' falls back to default TextToSpeechModule implementation.");
    return PTR(new TextToSpeechModule(desc, pipeline_desc, model_type));
}

void TextToSpeechModule::print_static_config() {
    std::cout << R"(
  text_to_speech:
    type: "TextToSpeechModule"
    device: "GPU"
    inputs:
      - name: "text"
        type: "String"                                                                                        # Support DataType: [String]
        source: "ParentModuleName.OutputPortName"
      - name: "texts"
        type: "VecString"                                                                                     # Support DataType: [VecString]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "audios"
        type: "VecOVTensor"                                                                                   # Support DataType: [VecOVTensor]
      - name: "sample_rates"
        type: "VecInt"                                                                                        # Support DataType: [VecInt]
      - name: "generated_texts"
        type: "VecString"                                                                                     # Support DataType: [VecString]
    params:
        config_path: "config_path"                                                                            # model config JSON file path
        tokenizer_path: "tokenizer_path"                                                                      # tokenizer model path (e.g. SentencePiece model file)
        embedding_model_path: "embedding_model.xml"                                                           # embedding model IR xml path
        prefill_model_path: "prefill_model.xml"                                                               # prefill model IR xml path
        decode_model_path: "decode_model.xml"                                                                 # decode model IR xml path
        codec_embedding_model_path: "codec_embedding_model.xml"                                               # codec embedding model IR xml path
        code_predictor_ar_model_path: "code_predictor_ar_model"                                               # code predictor autoregressive model directory path
        code_predictor_single_codec_embed_model_path: "code_predictor_single_codec_embed_model"               # code predictor single codec embedding model directory path
        code_predictor_single_codec_embedding_model_path: "code_predictor_single_codec_embedding_model.xml"   # code predictor single codec embedding model IR xml path
        speech_decoder_model_path: "speech_decoder_model.xml"                                                 # speech decoder model IR xml path
     )"
              << std::endl;
}

TextToSpeechModule::TextToSpeechModule(const IBaseModuleDesc::PTR& desc,
                                       const PipelineDesc::PTR& pipeline_desc,
                                       const VLMModelType& model_type)
    : IBaseModule(desc, pipeline_desc),
      m_model_type(model_type),
      m_device(desc->device.empty() ? "CPU" : desc->device) {
}

TextToSpeechModule::~TextToSpeechModule() = default;

void TextToSpeechModule::run() {
    OPENVINO_THROW("TextToSpeechModule is an abstract base class and cannot be run directly");
}

std::vector<std::string> TextToSpeechModule::parse_input_texts() {
    std::vector<std::string> texts;
    if (exists_input("text")) {
        texts.push_back(get_input("text").as<std::string>());
    } else if (exists_input("texts")) {
        texts = get_input("texts").as<std::vector<std::string>>();
    } else {
        OPENVINO_THROW("TextToSpeechModule[",
                       module_desc->name,
                       "]: either 'text' or 'texts' input is required");
    }
    return texts;
}

std::optional<std::filesystem::path> TextToSpeechModule::get_model_path(const std::string& param_name) {
    auto it = module_desc->params.find(param_name);
    if (it == module_desc->params.end() || it->second.empty()) {
        return std::nullopt;
    }
    return module_desc->get_full_path(it->second);
}

std::shared_ptr<ov::Model> TextToSpeechModule::load_model(const std::filesystem::path& model_path,
                                                          const std::string& default_model_filename) {
    if (model_path.extension() == ".xml") {
        if (std::filesystem::exists(model_path)) {
            return ::ov::genai::utils::singleton_core().read_model(model_path);
        }
        GENAI_ERR("Model XML " + model_path.string() + " does not exist");
        return nullptr;
    }

    if (default_model_filename.empty()) {
        if (std::filesystem::is_directory(model_path)) {
            GENAI_ERR("Model path \"" + model_path.string() +
                      "\" is a directory, but no default model filename was provided.");
        } else {
            GENAI_ERR("Model path \"" + model_path.string() +
                      "\" is not an .xml file, and no default model filename was provided.");
        }
        return nullptr;
    }

    const auto xml_path = model_path / default_model_filename;
    if (std::filesystem::exists(xml_path)) {
        return ::ov::genai::utils::singleton_core().read_model(xml_path);
    }
    GENAI_ERR("Model XML " + xml_path.string() + " does not exist in directory " + model_path.string());
    return nullptr;
}

std::vector<int64_t> TextToSpeechModule::make_mrope_positions(size_t start, size_t len, size_t batch) {
    std::vector<int64_t> pos(3 * batch * len);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < len; ++i) {
            const size_t base = b * len + i;
            const int64_t value = static_cast<int64_t>(start + i);
            pos[base] = value;
            pos[batch * len + base] = value;
            pos[2 * batch * len + base] = value;
        }
    }
    return pos;
}

ov::Tensor TextToSpeechModule::make_causal_mask(size_t seq_len, size_t batch) {
    ov::Tensor m(ov::element::f32, {batch, 1, seq_len, seq_len});
    float* d = m.data<float>();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                d[b * seq_len * seq_len + i * seq_len + j] =
                    (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
    }
    return m;
}

int64_t TextToSpeechModule::sample_codec_token(const float* logits,
                                               size_t vocab_size,
                                               float temperature,
                                               size_t top_k,
                                               float top_p,
                                               float rep_penalty,
                                               const std::vector<int64_t>* history,
                                               const std::vector<int64_t>* suppress_tokens,
                                               std::mt19937& rng) {
    std::vector<float> adjusted_logits(logits, logits + vocab_size);

    if (history && rep_penalty > 1.0f) {
        for (const int64_t token : *history) {
            if (token >= 0 && static_cast<size_t>(token) < vocab_size) {
                adjusted_logits[token] = (adjusted_logits[token] > 0) ? adjusted_logits[token] / rep_penalty
                                                                       : adjusted_logits[token] * rep_penalty;
            }
        }
    }

    if (suppress_tokens) {
        for (const int64_t token : *suppress_tokens) {
            if (token >= 0 && static_cast<size_t>(token) < vocab_size) {
                adjusted_logits[token] = -1e9f;
            }
        }
    }

    std::vector<size_t> index(vocab_size);
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&adjusted_logits](size_t lhs, size_t rhs) {
        return adjusted_logits[lhs] > adjusted_logits[rhs];
    });

    const size_t effective_top_k = std::min(top_k, vocab_size);
    const float max_logit = adjusted_logits[index[0]];
    std::vector<float> probs(effective_top_k);

    float probs_sum = 0.0f;
    for (size_t i = 0; i < effective_top_k; ++i) {
        probs[i] = std::exp((adjusted_logits[index[i]] - max_logit) / temperature);
        probs_sum += probs[i];
    }
    for (size_t i = 0; i < effective_top_k; ++i) {
        probs[i] /= probs_sum;
    }

    float cumulative_sum = 0.0f;
    size_t cutoff = effective_top_k;
    for (size_t i = 0; i < effective_top_k; ++i) {
        cumulative_sum += probs[i];
        if (cumulative_sum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    float top_sum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        top_sum += probs[i];
    }

    std::uniform_real_distribution<float> distribution(0.0f, top_sum);
    const float random_value = distribution(rng);
    cumulative_sum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        cumulative_sum += probs[i];
        if (cumulative_sum >= random_value) {
            return static_cast<int64_t>(index[i]);
        }
    }

    return static_cast<int64_t>(index[0]);
}

ov::Tensor TextToSpeechModule::make_decode_mask(size_t past_len, size_t batch) {
    const size_t total = past_len + 1;
    ov::Tensor mask(ov::element::f32, {batch, 1, 1, total});
    std::fill_n(mask.data<float>(), batch * total, 0.0f);
    return mask;
}

std::pair<ov::Tensor, int> TextToSpeechModule::synthesize_fallback_tone(const std::string& text) {
    constexpr int sample_rate = 24000;
    constexpr float seconds = 1.5f;
    const size_t sample_count = static_cast<size_t>(sample_rate * seconds);

    ov::Tensor audio(ov::element::f32, {1, sample_count});
    float* audio_data = audio.data<float>();

    const int hash = static_cast<int>(std::hash<std::string>{}(text) % 5000);
    const float freq = 220.0f + static_cast<float>(hash % 660);
    constexpr float pi = 3.1415926535f;

    for (size_t i = 0; i < sample_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        audio_data[i] = 0.2f * std::sin(2.0f * pi * freq * t);
    }

    return {audio, sample_rate};
}

}  // namespace ov::genai::module
