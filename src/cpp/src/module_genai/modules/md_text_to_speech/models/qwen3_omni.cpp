// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni.hpp"

#if defined(ENABLE_MODELING_PRIVATE)

#    include <openvino/op/add.hpp>
#    include <openvino/op/broadcast.hpp>
#    include <openvino/op/concat.hpp>
#    include <openvino/op/constant.hpp>
#    include <openvino/op/gather.hpp>
#    include <openvino/op/parameter.hpp>
#    include <openvino/op/range.hpp>
#    include <openvino/op/reshape.hpp>
#    include <openvino/op/result.hpp>
#    include <openvino/op/scatter_nd_update.hpp>
#    include <openvino/op/shape_of.hpp>
#    include <openvino/op/slice.hpp>
#    include <openvino/op/squeeze.hpp>
#    include <openvino/op/topk.hpp>
#    include <openvino/op/unsqueeze.hpp>

#    include "module_genai/utils/profiler.hpp"
#    include "utils.hpp"

namespace ov::genai::module {

TextToSpeechImpl_Qwen3Omni::TextToSpeechImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                                                       const PipelineDesc::PTR& pipeline_desc,
                                                       const VLMModelType& model_type)
    : TextToSpeechModule(desc, pipeline_desc, model_type) {
    if (!initialize()) {
        OPENVINO_THROW("Failed to initialize TextToSpeechImpl_Qwen3Omni");
    }
}

bool TextToSpeechImpl_Qwen3Omni::initialize() {
    const std::optional<std::filesystem::path> config_path = get_model_path("config_path");
    if (!config_path.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: 'config_path' param is required for Qwen3-Omni");
        return false;
    }
    m_config = modeling::models::Qwen3OmniProcessingConfig::from_json_file(config_path.value());

    // Configs
    m_talker_cfg = to_qwen3_omni_talker_config(m_config);
    m_cp_cfg = to_qwen3_omni_code_predictor_config(m_config);
    m_cp_steps = std::max(1, m_cp_cfg.num_code_groups - 1);

    // All TTS models must run at fp32 precision regardless of device.
    // Using reduced precision (fp16/bf16) for the talker causes audio quality
    // degradation. The speech decoder always runs on CPU for the same reason
    // (GPU fp16 SnakeBeta accumulation causes ~10x amplitude loss → noise).
    const ov::AnyMap tts_props = {{ov::hint::inference_precision.name(), ov::element::f32}};

    const std::optional<std::filesystem::path> embedding_model_path = get_model_path("embedding_model_path");
    if (!embedding_model_path.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                  "]: 'embedding_model_path' param is required for Qwen3-Omni");
        return false;
    }
    const std::shared_ptr<ov::Model> embedding_model = load_model(embedding_model_path.value());
    if (!embedding_model) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load embedding model");
        return false;
    }
    auto compiled_embedding_model =
        ::ov::genai::utils::singleton_core().compile_model(embedding_model, m_device, tts_props);
    m_embedding_infer = std::make_unique<ov::InferRequest>(compiled_embedding_model.create_infer_request());

    const std::optional<std::filesystem::path> prefill_model_path = get_model_path("prefill_model_path");
    if (!prefill_model_path.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                  "]: 'prefill_model_path' param is required for Qwen3-Omni");
        return false;
    }
    const std::shared_ptr<ov::Model> prefill_model = load_model(prefill_model_path.value());
    if (!prefill_model) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load prefill model");
        return false;
    }
    auto compiled_prefill_model =
        ::ov::genai::utils::singleton_core().compile_model(prefill_model, m_device, tts_props);
    m_prefill_infer = std::make_unique<ov::InferRequest>(compiled_prefill_model.create_infer_request());

    const std::optional<std::filesystem::path> decode_model_path = get_model_path("decode_model_path");
    if (!decode_model_path.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                  "]: 'decode_model_path' param is required for Qwen3-Omni");
        return false;
    }
    const std::shared_ptr<ov::Model> decode_model = load_model(decode_model_path.value());
    if (!decode_model) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load decode model");
        return false;
    }
    auto compiled_decode_model = ::ov::genai::utils::singleton_core().compile_model(decode_model, m_device, tts_props);
    m_decode_infer = std::make_unique<ov::InferRequest>(compiled_decode_model.create_infer_request());

    const std::optional<std::filesystem::path> codec_embedding_model_path =
        get_model_path("codec_embedding_model_path");
    if (!codec_embedding_model_path.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                  "]: 'codec_embedding_model_path' param is required for Qwen3-Omni");
        return false;
    }
    const std::shared_ptr<ov::Model> codec_embedding_model = load_model(codec_embedding_model_path.value());
    if (!codec_embedding_model) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load codec embedding model");
        return false;
    }
    auto compiled_codec_embedding_model =
        ::ov::genai::utils::singleton_core().compile_model(codec_embedding_model, m_device, tts_props);
    m_codec_embedding_infer = std::make_unique<ov::InferRequest>(compiled_codec_embedding_model.create_infer_request());

    load_code_predictor_models(tts_props);

    const std::optional<std::filesystem::path> sce_emb_param =
        get_model_path("code_predictor_single_codec_embedding_model_path");
    if (!sce_emb_param.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                  "]: 'code_predictor_single_codec_embedding_model_path' param is required for Qwen3-Omni");
        return false;
    }
    const auto sce_emb_model = load_model(sce_emb_param.value());
    if (!sce_emb_model) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load single-codec-embedding model");
        return false;
    }
    auto compiled_sce_emb = ::ov::genai::utils::singleton_core().compile_model(sce_emb_model, m_device, tts_props);
    m_code_predictor_single_codec_embedding_infer =
        std::make_unique<ov::InferRequest>(compiled_sce_emb.create_infer_request());

    const std::optional<std::filesystem::path> speech_decoder_param = get_model_path("speech_decoder_model_path");
    if (!speech_decoder_param.has_value()) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                  "]: 'speech_decoder_model_path' param is required for Qwen3-Omni");
        return false;
    }
    const auto speech_decoder_model = load_model(speech_decoder_param.value());
    if (!speech_decoder_model) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load speech decoder model");
        return false;
    }
    auto compiled_speech_decoder =
        ::ov::genai::utils::singleton_core().compile_model(speech_decoder_model, "CPU", tts_props);
    m_speech_decoder_infer = std::make_unique<ov::InferRequest>(compiled_speech_decoder.create_infer_request());

    try {
        const std::optional<std::filesystem::path> tokenizer_path = get_model_path("tokenizer_path");
        if (!tokenizer_path.has_value()) {
            GENAI_ERR("TextToSpeechModule[" + module_desc->name +
                      "]: 'tokenizer_path' param is required for Qwen3-Omni");
            return false;
        }
        m_tokenizer = std::make_unique<Tokenizer>(tokenizer_path.value());
    } catch (const std::exception& error) {
        GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: tokenizer init failed: " + std::string(error.what()));
        return false;
    }

    // --- Pre-compute tts_pad embedding ---
    calc_tts_pad_embed();

    return true;
}

void TextToSpeechImpl_Qwen3Omni::load_code_predictor_models(const ov::AnyMap& tts_props) {
    const std::optional<std::filesystem::path> ar_dir_param = get_model_path("code_predictor_ar_model_path");
    OPENVINO_ASSERT(ar_dir_param.has_value(), "code_predictor_ar_model_path param is required for Qwen3-Omni");

    std::vector<std::shared_ptr<ov::Model>> ar_models;
    const std::filesystem::path ar_dir(ar_dir_param.value());
    for (int step = 0;; ++step) {
        const auto xml = ar_dir / ("qwen3_omni_code_predictor_ar_model_step_" + std::to_string(step) + ".xml");
        if (!std::filesystem::exists(xml)) {
            break;
        }
        auto model = ::ov::genai::utils::singleton_core().read_model(xml);
        auto compiled = ::ov::genai::utils::singleton_core().compile_model(model, m_device, tts_props);
        m_code_predictor_ar_infers.push_back(std::make_unique<ov::InferRequest>(compiled.create_infer_request()));
        ar_models.push_back(model);
    }
    OPENVINO_ASSERT(!m_code_predictor_ar_infers.empty(), "No AR step models found in " + ar_dir.string());

    const std::optional<std::filesystem::path> sce_dir_param = get_model_path("code_predictor_single_codec_embed_model_path");
    OPENVINO_ASSERT(sce_dir_param.has_value(), "code_predictor_single_codec_embed_model_path param is required for Qwen3-Omni");

    std::vector<std::shared_ptr<ov::Model>> sce_models;
    const std::filesystem::path sce_dir(sce_dir_param.value());
    for (int step = 0;; ++step) {
        const auto xml =
            sce_dir / ("qwen3_omni_code_predictor_single_codec_embed_model_step_" + std::to_string(step) + ".xml");
        if (!std::filesystem::exists(xml)) {
            break;
        }
        auto model = ::ov::genai::utils::singleton_core().read_model(xml);
        auto compiled = ::ov::genai::utils::singleton_core().compile_model(model, m_device, tts_props);
        m_code_predictor_single_codec_embed_infers.push_back(
            std::make_unique<ov::InferRequest>(compiled.create_infer_request()));
        sce_models.push_back(model);
    }
    OPENVINO_ASSERT(!m_code_predictor_single_codec_embed_infers.empty(), "No single-codec-embed step models found in " + sce_dir.string());

    if (m_sample_codec_token_greedy_search && m_merge_ar_and_sce_ov_models) {
        GENAI_INFO("TextToSpeechModule[" + module_desc->name +
                   "]: sample_codec_token_greedy_search is enabled, will use greedy decoding in sample_codec_token");
        merge_code_predictor_ov_models(ar_models, sce_models);
    }
}

// Todo
// layer_tokens: with shape=[batch_size, token_num].
// update_value: with shape=[batch_size, 1], value per batch, indicating the token id to be updated at current step for all tokens in input.
// output: with shape=[batch_size, 1], and each value in output is the token id to be predicted at current step for all tokens in input.
ov::Output<ov::Node> func_update_layer_token_ids(const ov::Output<ov::Node>& layer_tokens,
                                           const int& step,
                                           const ov::Output<ov::Node>& update_value) {
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(layer_tokens);

    auto const_zero = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto const_zero_scalar = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    auto const_one = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto const_one_scalar = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    auto pos_id = ov::op::v0::Constant::create(element::i64, Shape{1}, {step});

    auto batch_size = std::make_shared<ov::op::v8::Gather>(input_shape, const_zero, const_zero);
    auto batch_size_scalar = std::make_shared<ov::op::v0::Squeeze>(batch_size, const_zero);

    auto row_idx = std::make_shared<ov::op::v4::Range>(const_zero_scalar, batch_size_scalar, const_one_scalar, element::i64);  // shape=[batch]
    auto row_idx_2d = std::make_shared<ov::op::v0::Unsqueeze>(row_idx, const_one_scalar);  // shape=[batch,1]

    auto col_idx = std::make_shared<ov::op::v3::Broadcast>(pos_id, batch_size);  // shape=[batch]
    col_idx->set_friendly_name("col_idx");
    auto col_idx_2d = std::make_shared<ov::op::v0::Unsqueeze>(col_idx, const_one_scalar);  // shape=[batch,1]

    auto indices = std::make_shared<ov::op::v0::Concat>(OutputVector{row_idx_2d, col_idx_2d}, 1);  // shape=[batch,2]

    auto update_value_1d = std::make_shared<ov::op::v0::Squeeze>(update_value, const_one);  // shape=[batch]

    auto scatter_nd_update_node = std::make_shared<ov::op::v15::ScatterNDUpdate>(layer_tokens, indices, update_value_1d);  // shape=[batch, token_num]
    return scatter_nd_update_node->output(0);
}

// TODO:
// 1: For input_embeds with shape [batch_size, token_num, feature_dim], output position_ids with shape [batch_size, token_num].
// 2: For position_ids, each row is [0, 1, 2, ..., token_num-1].
ov::Output<ov::Node> build_position_ids(const ov::Output<ov::Node>& inputs_embeds) {
    auto embeds_shape = std::make_shared<ov::op::v3::ShapeOf>(inputs_embeds);
    auto batch_dim = std::make_shared<ov::op::v8::Gather>(
        embeds_shape,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),  // indices 0
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0})   // axis 0
    );
    auto token_num = std::make_shared<ov::op::v8::Gather>(
        embeds_shape,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),  // indices 1
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0})   // axis 0
    );

    const int start_value = 0;

    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {start_value});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    ov::Output<ov::Node> stop;
    if (start_value == 0) {
        // Range stop is exclusive. With start=0, use stop=token_num to keep length == token_num.
        stop = token_num;
    } else {
        // Range stop is exclusive. With start=1, use stop=token_num+start to keep length == token_num.
        stop = std::make_shared<ov::op::v1::Add>(token_num, start);
    }
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);

    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto batch_dim_1d = std::make_shared<ov::op::v0::Unsqueeze>(batch_dim, axis0);
    auto layer_tokens_1d = std::make_shared<ov::op::v0::Unsqueeze>(token_num, axis0);
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_dim_1d, layer_tokens_1d}, 0);

    auto position_ids = std::make_shared<ov::op::v3::Broadcast>(range, target_shape);

    return position_ids;
}

// TODO: Get the token id with max value in logits at current step.
// Input logits with shape=[batch, seq, vocab_size].
// Output token_ids with shape=[batch, 1].
ov::Output<ov::Node> get_max_token_ids(const ov::Output<ov::Node>& logits) {
    // Get last element in axis=1, shape=[batch, fea]
    auto gather =
        std::make_shared<ov::op::v8::Gather>(logits,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}),
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));

    // Greedy search: get the token id with max value in logits at current step.
    auto max_token = std::make_shared<ov::op::v11::TopK>(gather,
                                                         ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),  // k=1
                                                         1,
                                                         ov::op::v11::TopK::Mode::MAX,
                                                         ov::op::v11::TopK::SortType::NONE,
                                                         ov::element::i64);
    // Max token id, shape=[batch, 1]
    auto layer_token_id = max_token->output(1);
    return layer_token_id;
}

// Merge AR and SCE(single_codec_embed_model) models to a model.
// Model AR input: inputs_embeds[b, token_num, feature_dim], position_ids[b, token_num]
// Model AR output: logits[b, token_num, vocab_size] (from where we can get the predicted token id)
// Model SCE input: codec_input[b, token_num] (the predicted token ids from AR)
// Model SCE output: codec_embed[b, token_num, feature_dim] (the embedding of predicted token)
// Merged model return: codec_embed[batch, token_num, feature_dim], all_layer_token_id[batch, 15] (the predicted token id at all step)
std::shared_ptr<ov::Model> merge_ar_sce_model(std::shared_ptr<ov::Model>& ar_model, std::shared_ptr<ov::Model>& sce_model, const int& step) {
    auto inputs_embeds = ar_model->get_parameters().at(0);
    const ov::PartialShape& inputs_embeds_shape = inputs_embeds->get_partial_shape();
    OPENVINO_ASSERT(inputs_embeds_shape.rank().is_static(),
                    "AR inputs_embeds rank must be static for merged model compilation");
    OPENVINO_ASSERT(inputs_embeds_shape.rank().get_length() == 3,
                    "AR inputs_embeds rank must be 3 for merged model compilation");

    const ov::PartialShape current_layer_tokens_shape{inputs_embeds_shape[0], inputs_embeds_shape[1]};
    auto current_layer_tokens =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i64, current_layer_tokens_shape);

    auto all_token_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    auto model_index = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});

    // Remove the position_ids input of AR model and replace with generated position_ids.
    auto position_ids = build_position_ids(inputs_embeds->output(0));
    ar_model->inputs()[1].replace(position_ids);

    // output logits shape[batch, seq, fea]
    auto logits = ar_model->get_results()[0]->input_value(0);

    // layer_token_id, shape=[batch, 1]
    auto layer_token_id = get_max_token_ids(logits);

    // Get indices for scatter update -> shape=[batch_id, step_id]
    auto new_layer_tokens = func_update_layer_token_ids(current_layer_tokens, step, layer_token_id);

    sce_model->inputs()[0].replace(layer_token_id);

    auto layer_tokens_result = std::make_shared<ov::op::v0::Result>(new_layer_tokens);
    return std::make_shared<ov::Model>(ov::ResultVector{sce_model->get_results()[0], layer_tokens_result},
                                       ov::ParameterVector{inputs_embeds, current_layer_tokens},
                                       "merged_model");
}

std::shared_ptr<ov::Model> merge_neighbor_models(std::shared_ptr<ov::Model>& model_1, std::shared_ptr<ov::Model>& model_2) {
    // 2 inputs: inputs_embeds[batch, seq, fea], current_layer_tokens[batch, seq]
    // 2 outputs: Embeddings[batch, seq, fea], current_layer_tokens[batch, seq] after append new token.
    // model_1's output[Embeddings] -> model_2's input[inputs_embeds]
    // model_1's output[current_layer_tokens] -> model_2's input[current_layer_tokens]

    auto model_1_inputs = model_1->get_parameters();

    auto model_1_inputs_embeds = model_1_inputs.at(0);
    auto model_1_input_current_layer_tokens = model_1_inputs.at(1);
    auto model_1_output_embeddings = model_1->get_results()[0]->input_value(0);
    auto model_1_output_layer_tokens = model_1->get_results()[1]->input_value(0);

    // Append model_1's output embeddings (all previous steps) to inputs_embeds for model_2.
    auto merged_inputs_embeds = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{model_1_inputs_embeds->output(0), model_1_output_embeddings},
        1);

    model_2->inputs()[0].replace(merged_inputs_embeds);
    model_2->inputs()[1].replace(model_1_output_layer_tokens);

    auto model_2_output_embeddings = model_2->get_results()[0]->input_value(0);
    auto merged_output_embeddings = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{model_1_output_embeddings, model_2_output_embeddings},
        1);

    auto merged_output_embeddings_result = std::make_shared<ov::op::v0::Result>(merged_output_embeddings);

    return std::make_shared<ov::Model>(ov::ResultVector{merged_output_embeddings_result, model_2->get_results()[1]},
                                       ov::ParameterVector{model_1_inputs_embeds, model_1_input_current_layer_tokens},
                                       "merged_model");
};

void TextToSpeechImpl_Qwen3Omni::merge_code_predictor_ov_models(std::vector<std::shared_ptr<ov::Model>>& ar_models,
                                                                std::vector<std::shared_ptr<ov::Model>>& sce_models) {
    if (ar_models.size() < 2) {
        GENAI_WARN("TextToSpeechModule[" + module_desc->name + "]: Not enough AR models to merge (found " +
                   std::to_string(ar_models.size()) + "), will skip merging and use separate AR/SCE infer requests");
        return;
    }
    OPENVINO_ASSERT(ar_models.size() == sce_models.size(), "Number of AR and SCE models must be the same for merging");

    std::shared_ptr<ov::Model> merged_model = merge_ar_sce_model(ar_models[0], sce_models[0], 0);
    for (size_t i = 1; i < ar_models.size(); ++i) {
        GENAI_INFO("Merging AR model step " + std::to_string(i) + " with SCE model step " + std::to_string(i));
        auto tmp_model = merge_ar_sce_model(ar_models[i], sce_models[i], i);
        merged_model = merge_neighbor_models(merged_model, tmp_model);
    }

    m_merged_infer_request = std::make_unique<ov::InferRequest>(
        ::ov::genai::utils::singleton_core().compile_model(merged_model, m_device).create_infer_request());
    m_enable_merge_ov_models = true;
    m_cp_steps = ar_models.size();
    GENAI_INFO("Finished merging code predictor AR and SCE models into one OV model with " +
               std::to_string(m_cp_steps) + " steps. Will use merged model for inference.");
}

void TextToSpeechImpl_Qwen3Omni::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    const std::vector<std::string> texts = parse_input_texts();

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
}

ov::Tensor TextToSpeechImpl_Qwen3Omni::text_embedding(ov::Tensor text_input_ids,
                                                      ov::Tensor codec_input_ids,
                                                      ov::Tensor codec_mask) {
    m_embedding_infer->set_tensor("text_input_ids", text_input_ids);
    m_embedding_infer->set_tensor("codec_input_ids", codec_input_ids);
    m_embedding_infer->set_tensor("codec_mask", codec_mask);
    {
        PROFILE(pm, "TextToSpeechImpl_Qwen3Omni::embedding_model infer");
        m_embedding_infer->infer();
    }
    return m_embedding_infer->get_tensor("inputs_embeds");
}

void TextToSpeechImpl_Qwen3Omni::calc_tts_pad_embed() {
    const int64_t tts_pad_id = m_config.tts_pad_token_id;
    std::vector<int64_t> tp_text = {tts_pad_id};
    std::vector<int64_t> tp_codec = {0};
    std::vector<float> tp_mask = {0.0f};

    ov::Tensor embed = text_embedding(ov::Tensor(ov::element::i64, {1, 1}, tp_text.data()),
                                      ov::Tensor(ov::element::i64, {1, 1}, tp_codec.data()),
                                      ov::Tensor(ov::element::f32, {1, 1}, tp_mask.data()));
    m_tts_pad_embed.resize(embed.get_shape()[2]);
    std::copy(embed.data<float>(), embed.data<float>() + embed.get_shape()[2], m_tts_pad_embed.begin());
}

std::vector<int64_t> TextToSpeechImpl_Qwen3Omni::code_predictor_ar_infers_merged_ov(
    int cp_steps,
    std::vector<float>& autoregressive_sequence,
    size_t batch,
    size_t hidden_size,
    size_t cp_vocab_size,
    std::vector<std::vector<int64_t>>& all_layer_tokens,
    int num_layers_total) {
    std::vector<int64_t> current_layer_tokens(batch * cp_steps, 0);  // shape=[batch*cp_steps]
    ov::Tensor current_layer_tokens_tensor(ov::element::i64,
                                           {batch, static_cast<size_t>(cp_steps)},
                                           current_layer_tokens.data());

    const size_t current_length = autoregressive_sequence.size() / hidden_size;
    ov::Tensor ar_input(ov::element::f32, {batch, current_length, hidden_size}, autoregressive_sequence.data());

    m_merged_infer_request->set_input_tensor(0, ar_input);
    m_merged_infer_request->set_input_tensor(1, current_layer_tokens_tensor);
    {
        PROFILE(pm, "m_merged_infer_request infer");
        m_merged_infer_request->infer();
    }

    auto merged_outputs = m_merged_infer_request->get_output_tensor(0);  // shape=[batch, cp_steps, hidden_size]
    auto layer_tokens_output = m_merged_infer_request->get_output_tensor(1);  // shape=[batch, cp_steps]

    // Return layer_tokens_output to current_layer_tokens
    OPENVINO_ASSERT(layer_tokens_output.get_shape()[0] == batch &&
                        layer_tokens_output.get_shape()[1] == static_cast<size_t>(cp_steps),
                    "Merged model output shape mismatch. Expected [batch, cp_steps], got " +
                        std::to_string(layer_tokens_output.get_shape()[0]) + ", " +
                        std::to_string(layer_tokens_output.get_shape()[1]));
    const int64_t* layer_tokens_ptr = layer_tokens_output.data<int64_t>();
    std::copy(layer_tokens_ptr, layer_tokens_ptr + batch * cp_steps, current_layer_tokens.begin());

    // Return to all_layer_tokens
    for (size_t b = 0; b < batch; ++b) {
        for (int step = 0; step < cp_steps; ++step) {
            int64_t layer_token = current_layer_tokens[b * cp_steps + step];
            if (step + 1 < num_layers_total) {
                all_layer_tokens[step + 1].push_back(layer_token);
            }
        }
    }

    // Return to autoregressive_sequence
    const float* merged_emb_ptr = merged_outputs.data<float>();
    autoregressive_sequence.insert(autoregressive_sequence.end(),
                                   merged_emb_ptr,
                                   merged_emb_ptr + batch * cp_steps * hidden_size);

    return current_layer_tokens;
}

std::vector<int64_t> TextToSpeechImpl_Qwen3Omni::code_predictor_ar_infers(
    int cp_steps,
    std::vector<float>& autoregressive_sequence,
    size_t batch,
    size_t hidden_size,
    size_t cp_vocab_size,
    float temperature,
    size_t top_k,
    float top_p,
    std::mt19937& rng,
    std::vector<std::vector<int64_t>>& all_layer_tokens,
    int num_layers_total) {
    if (m_enable_merge_ov_models) {
        return code_predictor_ar_infers_merged_ov(cp_steps,
                                                  autoregressive_sequence,
                                                  batch,
                                                  hidden_size,
                                                  cp_vocab_size,
                                                  all_layer_tokens,
                                                  num_layers_total);
    }
    // Fallback to origianl cpp implementation if merged OV model is not available for inference.
    return code_predictor_ar_infers_cpp(cp_steps,
                                        autoregressive_sequence,
                                        batch,
                                        hidden_size,
                                        cp_vocab_size,
                                        temperature,
                                        top_k,
                                        top_p,
                                        rng,
                                        all_layer_tokens,
                                        num_layers_total);
}

std::vector<int64_t> TextToSpeechImpl_Qwen3Omni::code_predictor_ar_infers_cpp(
    int cp_steps,
    std::vector<float>& autoregressive_sequence,
    size_t batch,
    size_t hidden_size,
    size_t cp_vocab_size,
    float temperature,
    size_t top_k,
    float top_p,
    std::mt19937& rng,
    std::vector<std::vector<int64_t>>& all_layer_tokens,
    int num_layers_total) {
    std::vector<int64_t> current_layer_tokens(cp_steps);

    for (int step = 0; step < cp_steps; ++step) {
        const size_t current_length = autoregressive_sequence.size() / hidden_size;
        std::vector<int64_t> position_ids_vector(current_length);
        std::iota(position_ids_vector.begin(), position_ids_vector.end(), 0);
        ov::Tensor ar_input(ov::element::f32, {batch, current_length, hidden_size}, autoregressive_sequence.data());
        ov::Tensor ar_pos(ov::element::i64, {batch, current_length}, position_ids_vector.data());

        m_code_predictor_ar_infers[step]->set_tensor("inputs_embeds", ar_input);
        m_code_predictor_ar_infers[step]->set_tensor("position_ids", ar_pos);
        {
            PROFILE(pm, "code_predictor_ar_model_step_" + std::to_string(step) + " infer");
            m_code_predictor_ar_infers[step]->infer();
        }

        auto step_logits = m_code_predictor_ar_infers[step]->get_tensor("logits");

        int64_t layer_token;
        if (m_sample_codec_token_greedy_search) {
            layer_token = sample_codec_token_greedy(step_logits.data<float>(), cp_vocab_size);
        } else {
            layer_token = sample_codec_token(step_logits.data<float>(),
                                             cp_vocab_size,
                                             temperature,
                                             top_k,
                                             top_p,
                                             1.0f,
                                             nullptr,
                                             nullptr,
                                             rng);
        }

        if (step + 1 < num_layers_total) {
            all_layer_tokens[step + 1].push_back(layer_token);
        }
        current_layer_tokens[step] = layer_token;

        std::vector<int64_t> token_vector = {layer_token};
        m_code_predictor_single_codec_embed_infers[step]->set_tensor(
            "codec_input",
            ov::Tensor(ov::element::i64, {batch, 1}, token_vector.data()));
        {
            PROFILE(pm, "code_predictor_single_codec_embed_model_step_" + std::to_string(step) + " infer");
            m_code_predictor_single_codec_embed_infers[step]->infer();
        }

        auto layer_embed = m_code_predictor_single_codec_embed_infers[step]->get_tensor("codec_embed");
        // std::cout << "Step " << step << " : layer_embed shape: " << layer_embed.get_shape() << std::endl;
        autoregressive_sequence.insert(autoregressive_sequence.end(),
                                       layer_embed.data<float>(),
                                       layer_embed.data<float>() + hidden_size);
    }
	return current_layer_tokens;
}

std::pair<ov::Tensor, int> TextToSpeechImpl_Qwen3Omni::qwen3_omni_text_to_speech(const std::string& text) {
    // --- Tokenize text ---
    auto tok_result = m_tokenizer->encode(text, ov::genai::add_special_tokens(false));
    auto tok_ids_tensor = tok_result.input_ids;
    size_t text_len = tok_ids_tensor.get_shape()[1];
    const int64_t* tok_ptr = tok_ids_tensor.data<int64_t>();
    std::vector<int64_t> text_token_ids(tok_ptr, tok_ptr + text_len);

    // --- Config values ---
    constexpr size_t batch = 1;
    const size_t hidden_size = static_cast<size_t>(m_talker_cfg.hidden_size);
    const size_t num_layers = static_cast<size_t>(m_talker_cfg.num_hidden_layers);
    const size_t num_kv_heads = static_cast<size_t>(m_talker_cfg.num_key_value_heads);
    const size_t head_dim = static_cast<size_t>(m_talker_cfg.head_dim);
    const size_t vocab_size = static_cast<size_t>(m_talker_cfg.vocab_size);
    const size_t cp_vocab_size = static_cast<size_t>(m_cp_cfg.vocab_size);

    const int64_t tts_pad_id = m_config.tts_pad_token_id;
    const int64_t tts_eos_id = m_config.tts_eos_token_id;
    const int64_t codec_bos = m_talker_cfg.codec_bos_token_id;
    const int64_t codec_eos = m_talker_cfg.codec_eos_token_id;
    const int64_t codec_pad = m_talker_cfg.codec_pad_token_id;
    const int64_t codec_nothink = m_config.talker_config_raw.value("codec_nothink_id", 2155);
    const int64_t codec_think_bos = m_config.talker_config_raw.value("codec_think_bos_id", 2156);
    const int64_t codec_think_eos = m_config.talker_config_raw.value("codec_think_eos_id", 2157);

    // --- Resolve speaker ID (default: "f245") ---
    int64_t speaker_id = 2301;  // f245
    if (m_config.talker_config_raw.contains("speaker_id") && m_config.talker_config_raw.at("speaker_id").is_object()) {
        const auto& sid = m_config.talker_config_raw.at("speaker_id");
        if (sid.contains("f245")) {
            speaker_id = sid.at("f245").get<int64_t>();
        }
    }
    const int64_t tts_bos_id = 151672;  // tts_bos token from model tokenizer

    // --- Build prefill input (matching Python talker format) ---
    std::vector<int64_t> full_text_ids;
    std::vector<int64_t> full_codec_ids;
    std::vector<float> full_codec_mask;

    std::vector<int64_t> role_tokens = {151644, 77091, 198};
    for (const int64_t id : role_tokens) {
        full_text_ids.push_back(id);
        full_codec_ids.push_back(0);
        full_codec_mask.push_back(0.0f);
    }

    std::vector<int64_t> codec_prefix = {codec_nothink, codec_think_bos, codec_think_eos, speaker_id};
    for (size_t i = 0; i < codec_prefix.size(); ++i) {
        full_text_ids.push_back(tts_pad_id);
        full_codec_ids.push_back(codec_prefix[i]);
        full_codec_mask.push_back(1.0f);
    }

    full_text_ids.push_back(tts_bos_id);
    full_codec_ids.push_back(codec_pad);
    full_codec_mask.push_back(1.0f);

    full_text_ids.push_back(text_token_ids.empty() ? tts_eos_id : text_token_ids[0]);
    full_codec_ids.push_back(codec_bos);
    full_codec_mask.push_back(1.0f);

    const size_t prefill_len = full_text_ids.size();

    // --- Pre-compute trailing text embeddings for AR streaming ---
    std::vector<int64_t> trailing_text_ids;
    for (size_t i = 1; i < text_token_ids.size(); ++i) {
        trailing_text_ids.push_back(text_token_ids[i]);
    }
    trailing_text_ids.push_back(tts_eos_id);

    std::vector<std::vector<float>> trailing_text_embeds(trailing_text_ids.size());
    if (!trailing_text_ids.empty()) {
        const size_t trailing_len = trailing_text_ids.size();
        std::vector<int64_t> batched_codec_ids(trailing_len, 0);
        std::vector<float> batched_codec_mask(trailing_len, 0.0f);
        ov::Tensor t_text_ids(ov::element::i64, {1, trailing_len}, trailing_text_ids.data());
        ov::Tensor t_codec_ids(ov::element::i64, {1, trailing_len}, batched_codec_ids.data());
        ov::Tensor t_codec_mask(ov::element::f32, {1, trailing_len}, batched_codec_mask.data());
        auto embeddings = text_embedding(t_text_ids, t_codec_ids, t_codec_mask);
        const float* embeddings_data = embeddings.data<float>();
        const size_t trailing_hidden_size = embeddings.get_shape().back();
        for (size_t token_index = 0; token_index < trailing_len; ++token_index) {
            const float* token_ptr = embeddings_data + token_index * trailing_hidden_size;
            trailing_text_embeds[token_index].assign(token_ptr, token_ptr + trailing_hidden_size);
        }
    }

    // --- Get prefill embeddings ---
    ov::Tensor prefill_embeds;
    {
        ov::Tensor text_tensor(ov::element::i64, {batch, prefill_len}, full_text_ids.data());
        ov::Tensor codec_tensor(ov::element::i64, {batch, prefill_len}, full_codec_ids.data());
        ov::Tensor mask_tensor(ov::element::f32, {batch, prefill_len}, full_codec_mask.data());
        prefill_embeds = text_embedding(text_tensor, codec_tensor, mask_tensor);
    }

    auto pos_data = make_mrope_positions(0, prefill_len, batch);
    ov::Tensor position_ids(ov::element::i64, {3, batch, prefill_len}, pos_data.data());
    ov::Tensor attn_mask = make_causal_mask(prefill_len, batch);

    std::vector<ov::Tensor> past_keys;
    std::vector<ov::Tensor> past_values;
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

    std::vector<ov::Tensor> present_keys;
    std::vector<ov::Tensor> present_values;
    for (size_t i = 0; i < num_layers; ++i) {
        present_keys.push_back(m_prefill_infer->get_tensor("present_key_" + std::to_string(i)));
        present_values.push_back(m_prefill_infer->get_tensor("present_value_" + std::to_string(i)));
    }

    std::vector<int64_t> suppress_tokens;
    const int64_t suppress_start = static_cast<int64_t>(vocab_size) - 1024;
    for (int64_t i = suppress_start; i < static_cast<int64_t>(vocab_size); ++i) {
        if (i != codec_eos) {
            suppress_tokens.push_back(i);
        }
    }
    std::vector<int64_t> suppress_with_eos = suppress_tokens;
    suppress_with_eos.push_back(codec_eos);

    std::mt19937 rng(42);
    constexpr float temperature = 0.8f;
    constexpr size_t top_k = 50;
    constexpr float top_p = 0.95f;
    constexpr float rep_penalty = 1.05f;
    const int min_frames = static_cast<int>(trailing_text_embeds.size()) + 5;
    constexpr int max_frames = 1000;

    const int num_layers_total = m_cp_steps + 1;
    std::vector<std::vector<int64_t>> all_layer_tokens(num_layers_total);

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

    const float* hidden_data = hidden_tensor.data<float>() + (prefill_len - 1) * hidden_size;
    std::vector<float> past_hidden(hidden_data, hidden_data + hidden_size);

    {
        std::vector<int64_t> layer0_vector = {layer0_token};
        m_codec_embedding_infer->set_tensor("codec_input_ids",
                                            ov::Tensor(ov::element::i64, {batch, 1}, layer0_vector.data()));
        {
            PROFILE(pm, "codec_embedding_model infer");
            m_codec_embedding_infer->infer();
        }
    }
    auto layer0_embed_out = m_codec_embedding_infer->get_tensor("codec_embeds");
    std::vector<float> layer0_embed(layer0_embed_out.data<float>(), layer0_embed_out.data<float>() + hidden_size);

    size_t current_seq_len = prefill_len;

    for (int frame = 0; frame < max_frames && layer0_token != codec_eos; ++frame) {
        std::vector<float> autoregressive_sequence;
        autoregressive_sequence.insert(autoregressive_sequence.end(), past_hidden.begin(), past_hidden.end());
        autoregressive_sequence.insert(autoregressive_sequence.end(), layer0_embed.begin(), layer0_embed.end());

        if (m_code_predictor_ar_infers.size() < static_cast<size_t>(m_cp_steps) ||
            m_code_predictor_single_codec_embed_infers.size() < static_cast<size_t>(m_cp_steps)) {
            OPENVINO_THROW("TextToSpeechModule: insufficient code predictor steps loaded for code predictor models");
        }

        // Run AR infer for each step to get intermediate hidden states.
        std::vector<int64_t> current_layer_tokens = code_predictor_ar_infers(m_cp_steps,
                                                                             autoregressive_sequence,
                                                                             batch,
                                                                             hidden_size,
                                                                             cp_vocab_size,
                                                                             temperature,
                                                                             top_k,
                                                                             top_p,
                                                                             rng,
                                                                             all_layer_tokens,
                                                                             num_layers_total);

        std::vector<float> codec_sum(hidden_size, 0.0f);
        for (size_t i = 0; i < hidden_size; ++i) {
            codec_sum[i] += layer0_embed[i];
        }

        std::vector<std::vector<int64_t>> layer_tokens_vec(m_cp_steps);
        std::vector<ov::Tensor> layer_tensors(m_cp_steps);
        for (int layer = 0; layer < m_cp_steps; ++layer) {
            layer_tokens_vec[layer] = {current_layer_tokens[layer]};
            layer_tensors[layer] = ov::Tensor(ov::element::i64, {batch, 1}, layer_tokens_vec[layer].data());
            m_code_predictor_single_codec_embedding_infer->set_tensor("codec_input_" + std::to_string(layer),
                                                                      layer_tensors[layer]);
        }

        {
            PROFILE(pm, "code_predictor_single_codec_embedding infer");
            m_code_predictor_single_codec_embedding_infer->infer();
        }

        auto codec_embeds_sum = m_code_predictor_single_codec_embedding_infer->get_tensor("codec_embeds_sum");
        const float* sum_ptr = codec_embeds_sum.data<float>();
        for (size_t i = 0; i < hidden_size; ++i) {
            codec_sum[i] += sum_ptr[i];
        }

        const auto& text_conditioning =
            (static_cast<size_t>(frame) < trailing_text_embeds.size()) ? trailing_text_embeds[frame] : m_tts_pad_embed;
        std::vector<float> step_embed_data(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            step_embed_data[i] = codec_sum[i] + text_conditioning[i];
        }
        ov::Tensor step_embed(ov::element::f32, {batch, 1, hidden_size}, step_embed_data.data());

        auto step_pos = make_mrope_positions(current_seq_len, 1, batch);
        ov::Tensor step_positions(ov::element::i64, {3, batch, 1}, step_pos.data());
        ov::Tensor decode_attention_mask = make_decode_mask(current_seq_len, batch);

        m_decode_infer->set_tensor("inputs_embeds", step_embed);
        m_decode_infer->set_tensor("position_ids", step_positions);
        m_decode_infer->set_tensor("attention_mask", decode_attention_mask);
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

        {
            std::vector<int64_t> layer0_vector = {layer0_token};
            m_codec_embedding_infer->set_tensor("codec_input_ids",
                                                ov::Tensor(ov::element::i64, {batch, 1}, layer0_vector.data()));
            {
                PROFILE(pm, "codec_embedding_model infer");
                m_codec_embedding_infer->infer();
            }
            auto layer0_embedding = m_codec_embedding_infer->get_tensor("codec_embeds");
            std::copy(layer0_embedding.data<float>(),
                      layer0_embedding.data<float>() + hidden_size,
                      layer0_embed.begin());
        }

        current_seq_len++;
    }

    size_t num_frames = all_layer_tokens[0].size();
    size_t min_len = num_frames;
    for (size_t i = 1; i < all_layer_tokens.size(); ++i) {
        min_len = std::min(min_len, all_layer_tokens[i].size());
    }
    for (auto& layer : all_layer_tokens) {
        while (layer.size() > min_len) {
            layer.pop_back();
        }
    }
    num_frames = min_len;

    if (!all_layer_tokens[0].empty() && all_layer_tokens[0].back() == codec_eos) {
        for (auto& layer : all_layer_tokens) {
            if (!layer.empty()) {
                layer.pop_back();
            }
        }
        num_frames = all_layer_tokens[0].size();
    }

    constexpr size_t N_TAIL = 4;
    for (size_t tail_position = 0; tail_position < N_TAIL; ++tail_position) {
        for (auto& layer : all_layer_tokens) {
            layer.push_back(codec_pad);
        }
    }
    num_frames += N_TAIL;

    if (num_frames == 0) {
        return synthesize_fallback_tone(text);
    }

    const size_t num_layers_total_size = all_layer_tokens.size();
    std::vector<int64_t> codes_flat(num_layers_total_size * num_frames, 0);
    for (size_t layer = 0; layer < num_layers_total_size; ++layer) {
        const auto& layer_values = all_layer_tokens[layer];
        for (size_t t = 0; t < std::min(layer_values.size(), num_frames); ++t) {
            codes_flat[layer * num_frames + t] = layer_values[t];
        }
    }
    ov::Tensor codes(ov::element::i64, {1, num_layers_total_size, num_frames}, codes_flat.data());

    m_speech_decoder_infer->set_tensor("codes", codes);
    {
        PROFILE(pm, "speech_decoder infer");
        m_speech_decoder_infer->infer();
    }
    ov::Tensor audio = m_speech_decoder_infer->get_tensor("audio");

    return {audio, 24000};
}

}  // namespace ov::genai::module

#endif  // ENABLE_MODELING_PRIVATE
