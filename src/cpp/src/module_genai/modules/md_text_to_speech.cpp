// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef ENABLE_OPENVINO_NEW_ARCH

#include "md_text_to_speech.hpp"
#include "module_genai/pipeline/module_factory.hpp"
#include "module_genai/utils/profiler.hpp"
#include "utils.hpp"
#include <random>

namespace ov::genai::module {

GENAI_REGISTER_MODULE_SAME(TextToSpeechModule);

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
     )" << std::endl;
}

TextToSpeechModule::TextToSpeechModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        OPENVINO_THROW("Failed to initialize TextToSpeechModule");
    }
}

TextToSpeechModule::~TextToSpeechModule() = default;

bool TextToSpeechModule::initialize() {
    const auto& model_type = to_vlm_model_type(module_desc->model_type);
    std::string device = module_desc->device.empty() ? "CPU" : module_desc->device;
    if (model_type == VLMModelType::QWEN3_OMNI) {
        std::optional<std::filesystem::path> config_path = get_model_path("config_path");
        if (!config_path.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: 'config_path' param is required for Qwen3-Omni");
            return false;
        }
        m_config = Qwen3OmniProcessingConfig::from_json_file(config_path.value());

        // All TTS models must run at fp32 precision regardless of device.
        // Using reduced precision (fp16/bf16) for the talker causes audio quality
        // degradation. The speech decoder always runs on CPU for the same reason
        // (GPU fp16 SnakeBeta accumulation causes ~10x amplitude loss → noise).
        ov::AnyMap tts_props = {{ov::hint::inference_precision.name(), ov::element::f32}};

        std::optional<std::filesystem::path> embedding_model_path = get_model_path("embedding_model_path");
        if (!embedding_model_path.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'embedding_model_path' param is required for Qwen3-Omni");
            return false;
        }
        std::shared_ptr<ov::Model> embedding_model = load_model(embedding_model_path.value());
        if (!embedding_model) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load embedding model");
            return false;
        }
        auto compiled_embedding_model = ::ov::genai::utils::singleton_core().compile_model(embedding_model, device, tts_props);
        m_embedding_infer = std::make_unique<ov::InferRequest>(compiled_embedding_model.create_infer_request());

        std::optional<std::filesystem::path> prefill_model_path = get_model_path("prefill_model_path");
        if (!prefill_model_path.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'prefill_model_path' param is required for Qwen3-Omni");
            return false;
        }
        std::shared_ptr<ov::Model> prefill_model = load_model(prefill_model_path.value());
        if (!prefill_model) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load prefill model");
            return false;
        }
        auto compiled_prefill_model = ::ov::genai::utils::singleton_core().compile_model(prefill_model, device, tts_props);
        m_prefill_infer = std::make_unique<ov::InferRequest>(compiled_prefill_model.create_infer_request());

        std::optional<std::filesystem::path> decode_model_path = get_model_path("decode_model_path");
        if (!decode_model_path.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'decode_model_path' param is required for Qwen3-Omni");
            return false;
        }
        std::shared_ptr<ov::Model> decode_model = load_model(decode_model_path.value());
        if (!decode_model) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load decode model");
            return false;
        }
        auto compiled_decode_model = ::ov::genai::utils::singleton_core().compile_model(decode_model, device, tts_props);
        m_decode_infer = std::make_unique<ov::InferRequest>(compiled_decode_model.create_infer_request());

        std::optional<std::filesystem::path> codec_embedding_model_path = get_model_path("codec_embedding_model_path");
        if (!codec_embedding_model_path.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'codec_embedding_model_path' param is required for Qwen3-Omni");
            return false;
        }
        std::shared_ptr<ov::Model> codec_embedding_model = load_model(codec_embedding_model_path.value());
        if (!codec_embedding_model) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load codec embedding model");
            return false;
        }
        auto compiled_codec_embedding_model =
            ::ov::genai::utils::singleton_core().compile_model(codec_embedding_model, device, tts_props);
        m_codec_embedding_infer =
            std::make_unique<ov::InferRequest>(compiled_codec_embedding_model.create_infer_request());

        std::optional<std::filesystem::path> ar_dir_param = get_model_path("code_predictor_ar_model_path");
        if (!ar_dir_param.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'code_predictor_ar_model_path' param is required for Qwen3-Omni");
            return false;
        }
        std::filesystem::path ar_dir(ar_dir_param.value());
        for (int step = 0;; ++step) {
            auto xml = ar_dir / ("qwen3_omni_code_predictor_ar_model_step_" + std::to_string(step) + ".xml");
            if (!std::filesystem::exists(xml))
                break;
            auto model = ::ov::genai::utils::singleton_core().read_model(xml);
            auto compiled = ::ov::genai::utils::singleton_core().compile_model(model, device, tts_props);
            m_code_predictor_ar_infers.push_back(std::make_unique<ov::InferRequest>(compiled.create_infer_request()));
        }
        if (m_code_predictor_ar_infers.empty()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: No AR step models found in " + ar_dir.string());
            return false;
        }
        GENAI_INFO("TextToSpeechModule[" + module_desc->name + "]: Loaded " +
                   std::to_string(m_code_predictor_ar_infers.size()) + " AR step model(s)");

        std::optional<std::filesystem::path> sce_dir_param = get_model_path("code_predictor_single_codec_embed_model_path");
        if (!sce_dir_param.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'code_predictor_single_codec_embed_model_path' param is required for Qwen3-Omni");
            return false;
        }
        std::filesystem::path sce_dir(sce_dir_param.value());
        for (int step = 0;; ++step) {
            auto xml =
                sce_dir / ("qwen3_omni_code_predictor_single_codec_embed_model_step_" + std::to_string(step) + ".xml");
            if (!std::filesystem::exists(xml))
                break;
            auto model = ::ov::genai::utils::singleton_core().read_model(xml);
            auto compiled = ::ov::genai::utils::singleton_core().compile_model(model, device, tts_props);
            m_code_predictor_single_codec_embed_infers.push_back(
                std::make_unique<ov::InferRequest>(compiled.create_infer_request()));
        }
        if (m_code_predictor_single_codec_embed_infers.empty()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: No single-codec-embed step models found in " +
                      sce_dir.string());
            return false;
        }
        GENAI_INFO("TextToSpeechModule[" + module_desc->name + "]: Loaded " +
                   std::to_string(m_code_predictor_single_codec_embed_infers.size()) +
                   " single-codec-embed step model(s)");

        std::optional<std::filesystem::path> sce_emb_param = get_model_path("code_predictor_single_codec_embedding_model_path");
        if (!sce_emb_param.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'code_predictor_single_codec_embedding_model_path' param is required for Qwen3-Omni");
            return false;
        }
        auto sce_emb_model = load_model(sce_emb_param.value());
        if (!sce_emb_model) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load single-codec-embedding model");
            return false;
        }
        auto compiled_sce_emb = ::ov::genai::utils::singleton_core().compile_model(sce_emb_model, device, tts_props);
        m_code_predictor_single_codec_embedding_infer =
            std::make_unique<ov::InferRequest>(compiled_sce_emb.create_infer_request());

        std::optional<std::filesystem::path> sd_param = get_model_path("speech_decoder_model_path");
        if (!sd_param.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'speech_decoder_model_path' param is required for Qwen3-Omni");
            return false;
        }
        auto sd_model = load_model(sd_param.value());
        if (!sd_model) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load speech decoder model");
            return false;
        }
        auto compiled_sd = ::ov::genai::utils::singleton_core().compile_model(sd_model, "CPU", tts_props);
        m_speech_decoder_infer = std::make_unique<ov::InferRequest>(compiled_sd.create_infer_request());

        try {
            std::optional<std::filesystem::path> tokenizer_path = get_model_path("tokenizer_path");
            if (!tokenizer_path.has_value()) {
                GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                          "]: 'tokenizer_path' param is required for Qwen3-Omni");
                return false;
            }
            m_tokenizer = std::make_unique<Tokenizer>(tokenizer_path.value());
        } catch (const std::exception& e) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: tokenizer init failed: " + std::string(e.what()));
            return false;
        }
    } else {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Unsupported model type: " + module_desc->model_type);
        return false;
    }
    return true;
}

void TextToSpeechModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    auto model_type = to_vlm_model_type(module_desc->model_type);

    if (model_type == VLMModelType::QWEN3_OMNI) {
        std::vector<std::string> texts;
        if (exists_input("text")) {
            std::string text = inputs["text"].data.as<std::string>();
            texts.push_back(text);
        } else if (exists_input("texts")) {
            texts = inputs["texts"].data.as<std::vector<std::string>>();
        } else {
            OPENVINO_THROW("TextToSpeechModule[" + module_desc->name + "]: Either 'text' or 'texts' input is required for Qwen3-Omni");
        }
        std::vector<ov::Tensor> audios;
        std::vector<int> sample_rates;
        audios.reserve(texts.size());
        sample_rates.reserve(texts.size());
        for (const auto& text : texts) {
            auto [audio, sample_rate] = qwen3_omni_text_to_speech(text);
            audios.push_back(audio);
            sample_rates.push_back(sample_rate);
        }
        outputs["audios"].data = audios;
        outputs["sample_rates"].data = sample_rates;
        outputs["generated_texts"].data = texts;
    } else {
        OPENVINO_THROW("TextToSpeechModule[" + module_desc->name + "]: Unsupported model type: " + module_desc->model_type);
    }
}

std::optional<std::filesystem::path> TextToSpeechModule::get_model_path(const std::string& param_name) {
    auto it = module_desc->params.find(param_name);
    if (it == module_desc->params.end() || it->second.empty()) {
        return std::nullopt;
    }
    return it->second;
}

std::shared_ptr<ov::Model> TextToSpeechModule::load_model(const std::filesystem::path& model_path,
                                                          const std::string& default_model_filename) {
    if (model_path.extension() == ".xml") {
        if (std::filesystem::exists(model_path)) {
            return ::ov::genai::utils::singleton_core().read_model(model_path);
        } else {
            GENAI_ERR("Model XML " + model_path.string() + " does not exist");
            return nullptr;
        }
    } else {
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
        auto xml_path = model_path / default_model_filename;
        if (std::filesystem::exists(xml_path)) {
            return ::ov::genai::utils::singleton_core().read_model(xml_path);
        } else {
            GENAI_ERR("Model XML " + xml_path.string() + " does not exist in directory " + model_path.string());
            return nullptr;
        }
    }
}

std::pair<ov::Tensor, int> TextToSpeechModule::qwen3_omni_text_to_speech(const std::string& text) {
    Qwen3OmniProcessingConfig cfg = std::get<Qwen3OmniProcessingConfig>(m_config);
    auto talker_cfg = to_qwen3_omni_talker_config(cfg);
    auto cp_cfg = to_qwen3_omni_code_predictor_config(cfg);
    auto speech_decoder_cfg = to_qwen3_omni_speech_decoder_config(cfg);
    const int cp_steps = std::max(1, cp_cfg.num_code_groups - 1);

    // --- Tokenize text ---
    auto tok_result = m_tokenizer->encode(text, ov::genai::add_special_tokens(false));
    auto tok_ids_tensor = tok_result.input_ids;
    size_t text_len = tok_ids_tensor.get_shape()[1];
    const int64_t* tok_ptr = tok_ids_tensor.data<int64_t>();
    std::vector<int64_t> text_token_ids(tok_ptr, tok_ptr + text_len);

    // --- Config values ---
    const size_t batch = 1;
    const size_t hidden_size = static_cast<size_t>(talker_cfg.hidden_size);
    const size_t num_layers = static_cast<size_t>(talker_cfg.num_hidden_layers);
    const size_t num_kv_heads = static_cast<size_t>(talker_cfg.num_key_value_heads);
    const size_t head_dim = static_cast<size_t>(talker_cfg.head_dim);
    const size_t vocab_size = static_cast<size_t>(talker_cfg.vocab_size);
    const size_t cp_vocab_size = static_cast<size_t>(cp_cfg.vocab_size);

    const int64_t tts_pad_id = cfg.tts_pad_token_id;
    const int64_t tts_eos_id = cfg.tts_eos_token_id;
    const int64_t codec_bos = talker_cfg.codec_bos_token_id;
    const int64_t codec_eos = talker_cfg.codec_eos_token_id;
    const int64_t codec_pad = talker_cfg.codec_pad_token_id;
    const int64_t codec_nothink = cfg.talker_config_raw.value("codec_nothink_id", 2155);
    const int64_t codec_think_bos = cfg.talker_config_raw.value("codec_think_bos_id", 2156);
    const int64_t codec_think_eos = cfg.talker_config_raw.value("codec_think_eos_id", 2157);

    // --- Pre-compute tts_pad embedding ---
    {
        std::vector<int64_t> tp_text = {tts_pad_id};
        std::vector<int64_t> tp_codec = {0};
        std::vector<float> tp_mask = {0.0f};
        m_embedding_infer->set_tensor("text_input_ids", ov::Tensor(ov::element::i64, {1, 1}, tp_text.data()));
        m_embedding_infer->set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {1, 1}, tp_codec.data()));
        m_embedding_infer->set_tensor("codec_mask", ov::Tensor(ov::element::f32, {1, 1}, tp_mask.data()));
        {
            PROFILE(pm, "embedding_model infer");
            m_embedding_infer->infer();
        }
    }
    auto tts_pad_embed_tensor = m_embedding_infer->get_tensor("inputs_embeds");
    std::vector<float> tts_pad_embed(hidden_size);
    std::copy(tts_pad_embed_tensor.data<float>(),
              tts_pad_embed_tensor.data<float>() + hidden_size,
              tts_pad_embed.begin());

    // --- Resolve speaker ID (default: "f245") ---
    int64_t speaker_id = 2301;  // f245
    if (cfg.talker_config_raw.contains("speaker_id") && cfg.talker_config_raw.at("speaker_id").is_object()) {
        const auto& sid = cfg.talker_config_raw.at("speaker_id");
        if (sid.contains("f245"))
            speaker_id = sid.at("f245").get<int64_t>();
    }
    const int64_t tts_bos_id = 151672;  // tts_bos token from model tokenizer

    // --- Build prefill input (matching Python talker format) ---
    // Python layout: [role×3, tts_pad×4 + {nothink,think_bos,think_eos,speaker},
    //                 tts_bos + codec_pad, text₀ + codec_bos]  = 9 positions
    std::vector<int64_t> full_text_ids;
    std::vector<int64_t> full_codec_ids;
    std::vector<float> full_codec_mask;

    // 1. Role tokens: <|im_start|>assistant\n (codec_mask=0, text only)
    std::vector<int64_t> role_tokens = {151644, 77091, 198};
    for (auto id : role_tokens) {
        full_text_ids.push_back(id);
        full_codec_ids.push_back(0);
        full_codec_mask.push_back(0.0f);
    }

    // 2. Codec prefix: 4 tokens = {nothink, think_bos, think_eos, speaker_id}
    std::vector<int64_t> codec_prefix = {codec_nothink, codec_think_bos, codec_think_eos, speaker_id};
    for (size_t i = 0; i < codec_prefix.size(); ++i) {
        full_text_ids.push_back(tts_pad_id);
        full_codec_ids.push_back(codec_prefix[i]);
        full_codec_mask.push_back(1.0f);
    }

    // 3. tts_bos + codec_pad
    full_text_ids.push_back(tts_bos_id);
    full_codec_ids.push_back(codec_pad);
    full_codec_mask.push_back(1.0f);

    // 4. First text token + codec_bos (generation start marker)
    full_text_ids.push_back(text_token_ids.empty() ? tts_eos_id : text_token_ids[0]);
    full_codec_ids.push_back(codec_bos);
    full_codec_mask.push_back(1.0f);

    size_t prefill_len = full_text_ids.size();

    // --- Pre-compute trailing text embeddings for AR streaming ---
    // Python feeds text₁...textₙ, tts_eos during generation, then tts_pad when exhausted.
    // trailing_text_ids = [text₁, text₂, ..., textₙ, tts_eos]
    std::vector<int64_t> trailing_text_ids;
    for (size_t i = 1; i < text_token_ids.size(); ++i)
        trailing_text_ids.push_back(text_token_ids[i]);
    trailing_text_ids.push_back(tts_eos_id);

    // Pre-compute all trailing text embeddings in a single batched inference
    std::vector<std::vector<float>> trailing_text_embeds(trailing_text_ids.size());
    if (!trailing_text_ids.empty()) {
        const size_t trailing_len = trailing_text_ids.size();
        // Prepare batched inputs: shape [1, trailing_len], codec_mask = 0 (text-only)
        std::vector<int64_t> batched_codec_ids(trailing_len, 0);
        std::vector<float> batched_codec_mask(trailing_len, 0.0f);
        ov::Tensor t_text_ids(ov::element::i64, {1, trailing_len}, trailing_text_ids.data());
        ov::Tensor t_codec_ids(ov::element::i64, {1, trailing_len}, batched_codec_ids.data());
        ov::Tensor t_codec_mask(ov::element::f32, {1, trailing_len}, batched_codec_mask.data());
        m_embedding_infer->set_tensor("text_input_ids", t_text_ids);
        m_embedding_infer->set_tensor("codec_input_ids", t_codec_ids);
        m_embedding_infer->set_tensor("codec_mask", t_codec_mask);
        {
            PROFILE(pm, "embedding_model infer");
            m_embedding_infer->infer();
        }
        auto e = m_embedding_infer->get_tensor("inputs_embeds");
        const float* e_data = e.data<float>();
        // Determine hidden size from tensor shape if not already known
        const auto& e_shape = e.get_shape();
        const size_t trailing_hidden_size = e_shape.back();
        for (size_t ti = 0; ti < trailing_len; ++ti) {
            const float* token_ptr = e_data + ti * trailing_hidden_size;
            trailing_text_embeds[ti].assign(token_ptr, token_ptr + trailing_hidden_size);
        }
    }

    // --- Get prefill embeddings ---
    {
        ov::Tensor tt(ov::element::i64, {batch, prefill_len}, full_text_ids.data());
        ov::Tensor ct(ov::element::i64, {batch, prefill_len}, full_codec_ids.data());
        ov::Tensor mt(ov::element::f32, {batch, prefill_len}, full_codec_mask.data());
        m_embedding_infer->set_tensor("text_input_ids", tt);
        m_embedding_infer->set_tensor("codec_input_ids", ct);
        m_embedding_infer->set_tensor("codec_mask", mt);
        {
            PROFILE(pm, "embedding_model infer");
            m_embedding_infer->infer();
        }
    }
    auto prefill_embeds = m_embedding_infer->get_tensor("inputs_embeds");

    // --- Prefill ---
    auto pos_data = make_mrope_positions(0, prefill_len, batch);
    ov::Tensor position_ids(ov::element::i64, {3, batch, prefill_len}, pos_data.data());
    ov::Tensor attn_mask = make_causal_mask(prefill_len, batch);

    // Empty past KV caches
    std::vector<ov::Tensor> past_keys, past_values;
    for (size_t i = 0; i < num_layers; ++i) {
        past_keys.push_back(ov::Tensor(ov::element::f32, {batch, num_kv_heads, 0, head_dim}));
        past_values.push_back(ov::Tensor(ov::element::f32, {batch, num_kv_heads, 0, head_dim}));
    }

    m_prefill_infer->set_tensor("inputs_embeds", prefill_embeds);
    m_prefill_infer->set_tensor("position_ids", position_ids);
    m_prefill_infer->set_tensor("attention_mask", attn_mask);
    for (size_t i = 0; i < num_layers; ++i) {
        m_prefill_infer->set_tensor("past_key_" + std::to_string(i), past_keys[i]);
        m_prefill_infer->set_tensor("past_value_" + std::to_string(i), past_values[i]);
    }
    {
        PROFILE(pm, "prefill infer");
        m_prefill_infer->infer();
    }

    auto logits_tensor = m_prefill_infer->get_tensor("logits");
    auto hidden_tensor = m_prefill_infer->get_tensor("hidden_states");

    std::vector<ov::Tensor> present_keys, present_values;
    for (size_t i = 0; i < num_layers; ++i) {
        present_keys.push_back(m_prefill_infer->get_tensor("present_key_" + std::to_string(i)));
        present_values.push_back(m_prefill_infer->get_tensor("present_value_" + std::to_string(i)));
    }

    // --- Build suppress_tokens list ---
    std::vector<int64_t> suppress_tokens;
    int64_t suppress_start = static_cast<int64_t>(vocab_size) - 1024;
    for (int64_t i = suppress_start; i < static_cast<int64_t>(vocab_size); ++i) {
        if (i != codec_eos)
            suppress_tokens.push_back(i);
    }
    std::vector<int64_t> suppress_with_eos = suppress_tokens;
    suppress_with_eos.push_back(codec_eos);

    // --- Generation loop ---
    std::mt19937 rng(42);
    const float temperature = 0.8f;
    const size_t top_k = 50;
    const float top_p = 0.95f;
    const float rep_penalty = 1.05f;
    // Precise EOS suppression: the model is fed text₁…textₙ then tts_eos as
    // conditioning during decode (one token per frame). It cannot possibly have
    // synthesized all content before exhausting that stream, so suppress EOS
    // until frame >= trailing_text_embeds.size(). A small grace window (+5)
    // lets the final syllable complete before we allow EOS.
    const int min_frames = static_cast<int>(trailing_text_embeds.size()) + 5;
    const int max_frames = 1000;

    const int num_layers_total = cp_steps + 1;  // layer0 + cp_steps code-predictor layers
    std::vector<std::vector<int64_t>> all_layer_tokens(num_layers_total);

    // Sample first layer0 token from prefill logits
    const float* logits_data = logits_tensor.data<float>() + (prefill_len - 1) * vocab_size;
    int64_t layer0_token = sample_codec_token(logits_data,
                                              vocab_size,
                                              temperature,
                                              top_k,
                                              top_p,
                                              rep_penalty,
                                              &all_layer_tokens[0],
                                              &suppress_with_eos,
                                              rng);
    all_layer_tokens[0].push_back(layer0_token);

    // Get past_hidden (last position)
    const float* hidden_data = hidden_tensor.data<float>() + (prefill_len - 1) * hidden_size;
    std::vector<float> past_hidden(hidden_data, hidden_data + hidden_size);

    // Get layer0 embedding
    {
        std::vector<int64_t> l0_vec = {layer0_token};
        m_codec_embedding_infer->set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {batch, 1}, l0_vec.data()));
        {
            PROFILE(pm, "codec_embedding_model infer");
            m_codec_embedding_infer->infer();
        }
    }
    auto l0_embed_out = m_codec_embedding_infer->get_tensor("codec_embeds");
    std::vector<float> layer0_embed(l0_embed_out.data<float>(), l0_embed_out.data<float>() + hidden_size);

    size_t current_seq_len = prefill_len;

    // Process frames
    for (int frame = 0; frame < max_frames && layer0_token != codec_eos; ++frame) {
        // --- Code predictor: generate layers 1-15 for this frame ---
        std::vector<float> ar_seq;
        ar_seq.insert(ar_seq.end(), past_hidden.begin(), past_hidden.end());
        ar_seq.insert(ar_seq.end(), layer0_embed.begin(), layer0_embed.end());
        if (m_code_predictor_ar_infers.size() < static_cast<std::size_t>(cp_steps) ||
            m_code_predictor_single_codec_embed_infers.size() < static_cast<std::size_t>(cp_steps)) {
            OPENVINO_THROW("TextToSpeechModule: insufficient code predictor steps loaded for code predictor models");
        }

        std::vector<int64_t> current_layer_tokens(cp_steps);
        for (int step = 0; step < cp_steps; ++step) {
            size_t cur_len = ar_seq.size() / hidden_size;
            std::vector<int64_t> pos_ids(cur_len);
            std::iota(pos_ids.begin(), pos_ids.end(), 0);

            ov::Tensor ar_input(ov::element::f32, {batch, cur_len, hidden_size}, ar_seq.data());
            ov::Tensor ar_pos(ov::element::i64, {batch, cur_len}, pos_ids.data());
            m_code_predictor_ar_infers[step]->set_tensor("inputs_embeds", ar_input);
            m_code_predictor_ar_infers[step]->set_tensor("position_ids", ar_pos);
            {
                PROFILE(pm, "code_predictor_ar_model_step_" + std::to_string(step) + " infer");
                m_code_predictor_ar_infers[step]->infer();
            }

            auto step_logits = m_code_predictor_ar_infers[step]->get_tensor("logits");
            int64_t layer_token = sample_codec_token(step_logits.data<float>(),
                                                     cp_vocab_size,
                                                     temperature,
                                                     top_k,
                                                     top_p,
                                                     1.0f,
                                                     nullptr,
                                                     nullptr,
                                                     rng);
            if (step + 1 < num_layers_total)
                all_layer_tokens[step + 1].push_back(layer_token);
            current_layer_tokens[step] = layer_token;

            // Get embedding for this token
            std::vector<int64_t> tv = {layer_token};
            m_code_predictor_single_codec_embed_infers[step]->set_tensor("codec_input", ov::Tensor(ov::element::i64, {batch, 1}, tv.data()));
            {
                PROFILE(pm, "code_predictor_single_codec_embed_model_step_" + std::to_string(step) + " infer");
                m_code_predictor_single_codec_embed_infers[step]->infer();
            }
            auto le = m_code_predictor_single_codec_embed_infers[step]->get_tensor("codec_embed");
            ar_seq.insert(ar_seq.end(), le.data<float>(), le.data<float>() + hidden_size);
        }

        // --- Compute combined codec embedding for talker decode input ---
        std::vector<float> codec_sum(hidden_size, 0.0f);
        for (size_t i = 0; i < hidden_size; ++i)
            codec_sum[i] += layer0_embed[i];

        // Sum embeddings from all 15 code predictor layers
        std::vector<std::vector<int64_t>> layer_tokens_vec(cp_steps);
        std::vector<ov::Tensor> layer_tensors(cp_steps);
        for (int layer = 0; layer < cp_steps; ++layer) {
            layer_tokens_vec[layer] = {current_layer_tokens[layer]};
            layer_tensors[layer] = ov::Tensor(ov::element::i64, {batch, 1}, layer_tokens_vec[layer].data());
            m_code_predictor_single_codec_embedding_infer->set_tensor("codec_input_" + std::to_string(layer), layer_tensors[layer]);
        }
        {
            PROFILE(pm, "code_predictor_single_codec_embedding infer");
            m_code_predictor_single_codec_embedding_infer->infer();
        }
        auto codec_embeds_sum = m_code_predictor_single_codec_embedding_infer->get_tensor("codec_embeds_sum");
        const float* sum_ptr = codec_embeds_sum.data<float>();
        for (size_t i = 0; i < hidden_size; ++i)
            codec_sum[i] += sum_ptr[i];

        // code_predictor_accum_ms += elapsed_ms(t_cp_start, t_cp_end);

        // inputs_embeds = codec_sum + text_conditioning
        // Python streams text tokens: text₁,text₂,...,textₙ,tts_eos, then tts_pad when exhausted
        const auto& text_cond =
            (static_cast<size_t>(frame) < trailing_text_embeds.size()) ? trailing_text_embeds[frame] : tts_pad_embed;
        std::vector<float> step_embed_data(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i)
            step_embed_data[i] = codec_sum[i] + text_cond[i];
        ov::Tensor step_embed(ov::element::f32, {batch, 1, hidden_size}, step_embed_data.data());

        // --- Talker decode step ---
        auto step_pos = make_mrope_positions(current_seq_len, 1, batch);
        ov::Tensor step_positions(ov::element::i64, {3, batch, 1}, step_pos.data());
        ov::Tensor dec_attn_mask = make_decode_mask(current_seq_len, batch);

        m_decode_infer->set_tensor("inputs_embeds", step_embed);
        m_decode_infer->set_tensor("position_ids", step_positions);
        m_decode_infer->set_tensor("attention_mask", dec_attn_mask);
        for (size_t i = 0; i < num_layers; ++i) {
            m_decode_infer->set_tensor("past_key_" + std::to_string(i), present_keys[i]);
            m_decode_infer->set_tensor("past_value_" + std::to_string(i), present_values[i]);
        }
        {
                PROFILE(pm, "decode_model infer");
                m_decode_infer->infer();
        }

        logits_tensor = m_decode_infer->get_tensor("logits");
        hidden_tensor = m_decode_infer->get_tensor("hidden_states");
        for (size_t i = 0; i < num_layers; ++i) {
            present_keys[i] = m_decode_infer->get_tensor("present_key_" + std::to_string(i));
            present_values[i] = m_decode_infer->get_tensor("present_value_" + std::to_string(i));
        }

        logits_data = logits_tensor.data<float>();
        const auto* suppress = (frame < min_frames) ? &suppress_with_eos : &suppress_tokens;
        layer0_token = sample_codec_token(logits_data,
                                          vocab_size,
                                          temperature,
                                          top_k,
                                          top_p,
                                          rep_penalty,
                                          &all_layer_tokens[0],
                                          suppress,
                                          rng);
        all_layer_tokens[0].push_back(layer0_token);

        hidden_data = hidden_tensor.data<float>();
        std::copy(hidden_data, hidden_data + hidden_size, past_hidden.begin());

        // Get next layer0 embedding
        {
            std::vector<int64_t> l0v = {layer0_token};
            m_codec_embedding_infer->set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {batch, 1}, l0v.data()));
            {
                PROFILE(pm, "codec_embedding_model infer");
                m_codec_embedding_infer->infer();
            }
            auto l0e = m_codec_embedding_infer->get_tensor("codec_embeds");
            std::copy(l0e.data<float>(), l0e.data<float>() + hidden_size, layer0_embed.begin());
        }

        current_seq_len++;
        // talker_decode_accum_ms += elapsed_ms(t_td_start, t_td_end);
    }

    size_t num_frames = all_layer_tokens[0].size();
    // Layer 0 always has +1 token vs layers 1-15 (sampled after decode but
    // code predictor for the next frame never ran).  Trim to the minimum
    // across all layers so the codes tensor is balanced.
    size_t min_len = num_frames;
    for (size_t i = 1; i < all_layer_tokens.size(); ++i)
        min_len = std::min(min_len, all_layer_tokens[i].size());
    for (auto& layer : all_layer_tokens)
        while (layer.size() > min_len)
            layer.pop_back();
    num_frames = min_len;

    // Remove trailing EOS from layer 0 if present
    if (!all_layer_tokens[0].empty() && all_layer_tokens[0].back() == codec_eos) {
        for (auto& layer : all_layer_tokens) {
            if (!layer.empty())
                layer.pop_back();
        }
        num_frames = all_layer_tokens[0].size();
    }

    constexpr size_t N_TAIL = 4;
    const size_t content_frames = num_frames;
    for (size_t tp = 0; tp < N_TAIL; ++tp)
        for (auto& layer : all_layer_tokens)
            layer.push_back(codec_pad);
    num_frames += N_TAIL;
    // Print first few layer0 tokens for debugging

    if (num_frames == 0) {
        return synthesize_fallback_tone(text);
    }

    // --- Build codes tensor [1, num_layers_total, num_frames] ---
    const size_t nl = all_layer_tokens.size();
    std::vector<int64_t> codes_flat(nl * num_frames, 0);
    for (size_t layer = 0; layer < nl; ++layer) {
        const auto& lv = all_layer_tokens[layer];
        for (size_t t = 0; t < std::min(lv.size(), num_frames); ++t) {
            codes_flat[layer * num_frames + t] = lv[t];
        }
    }
    ov::Tensor codes(ov::element::i64, {1, nl, num_frames}, codes_flat.data());

    // --- Run speech decoder ---
    const auto t_decoder_start = std::chrono::steady_clock::now();
    m_speech_decoder_infer->set_tensor("codes", codes);
    {
        PROFILE(pm, "speech_decoder infer");
        m_speech_decoder_infer->infer();
    }
    const auto t_decoder_end = std::chrono::steady_clock::now();
    ov::Tensor audio = m_speech_decoder_infer->get_tensor("audio");

    return {audio, 24000};
}

std::vector<int64_t> TextToSpeechModule::make_mrope_positions(size_t start, size_t len, size_t batch) {
    std::vector<int64_t> pos(3 * batch * len);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < len; ++i) {
            size_t base = b * len + i;
            int64_t value = static_cast<int64_t>(start + i);
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
    for (size_t b = 0; b < batch; ++b)
        for (size_t i = 0; i < seq_len; ++i)
            for (size_t j = 0; j < seq_len; ++j)
                d[b * seq_len * seq_len + i * seq_len + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
    return m;
}

int64_t TextToSpeechModule::sample_codec_token(const float* logits, size_t vocab_size,
                           float temperature, size_t top_k, float top_p,
                           float rep_penalty,
                           const std::vector<int64_t>* history,
                           const std::vector<int64_t>* suppress_tokens,
                           std::mt19937& rng) {
    std::vector<float> adj(logits, logits + vocab_size);

    if (history && rep_penalty > 1.0f) {
        for (int64_t tok : *history) {
            if (tok >= 0 && static_cast<size_t>(tok) < vocab_size) {
                adj[tok] = (adj[tok] > 0) ? adj[tok] / rep_penalty : adj[tok] * rep_penalty;
            }
        }
    }
    if (suppress_tokens) {
        for (int64_t tok : *suppress_tokens) {
            if (tok >= 0 && static_cast<size_t>(tok) < vocab_size) {
                adj[tok] = -1e9f;
            }
        }
    }

    std::vector<size_t> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&adj](size_t a, size_t b) { return adj[a] > adj[b]; });

    size_t eff_k = std::min(top_k, vocab_size);
    float max_l = adj[idx[0]];
    std::vector<float> probs(eff_k);
    float sum = 0;
    for (size_t i = 0; i < eff_k; ++i) {
        probs[i] = std::exp((adj[idx[i]] - max_l) / temperature);
        sum += probs[i];
    }
    for (size_t i = 0; i < eff_k; ++i) probs[i] /= sum;

    float cumsum = 0;
    size_t cutoff = eff_k;
    for (size_t i = 0; i < eff_k; ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }

    float top_sum = 0;
    for (size_t i = 0; i < cutoff; ++i) top_sum += probs[i];

    std::uniform_real_distribution<float> dist(0.0f, top_sum);
    float r = dist(rng);
    cumsum = 0;
    for (size_t i = 0; i < cutoff; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) return static_cast<int64_t>(idx[i]);
    }
    return static_cast<int64_t>(idx[0]);
}

ov::Tensor TextToSpeechModule::make_decode_mask(size_t past_len, size_t batch) {
    size_t total = past_len + 1;
    ov::Tensor m(ov::element::f32, {batch, 1, 1, total});
    std::fill_n(m.data<float>(), batch * total, 0.0f);
    return m;
}

std::pair<ov::Tensor, int> TextToSpeechModule::synthesize_fallback_tone(const std::string& text) {
    const int sample_rate = 24000;
    const float seconds = 1.5f;
    const size_t sample_count = static_cast<size_t>(sample_rate * seconds);
    ov::Tensor audio(ov::element::f32, {1, sample_count});
    auto audio_data = audio.data<float>();
    // std::vector<float> audio(sample_count);
    const int hash = static_cast<int>(std::hash<std::string>{}(text) % 5000);
    const float freq = 220.0f + static_cast<float>(hash % 660);
    const float pi = 3.1415926535f;
    for (size_t i = 0; i < sample_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        audio_data[i] = 0.2f * std::sin(2.0f * pi * freq * t);
    }

    return {audio, sample_rate};
}

}  // namespace ov::genai::module

#endif
