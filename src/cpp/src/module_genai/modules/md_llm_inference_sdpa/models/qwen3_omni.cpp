// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni.hpp"

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/openvino.hpp>

#if defined(ENABLE_MODELING_PRIVATE)

#    include "modeling/models/qwen3_omni/processing_qwen3_omni.hpp"
#    include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#    include "module_genai/utils/com_utils.hpp"
#    include "module_genai/utils/profiler.hpp"
#    include "openvino/genai/chat_history.hpp"
#    include "utils.hpp"

namespace ov::genai::module {
LLMInferenceSDPAImpl_Qwen3Omni::LLMInferenceSDPAImpl_Qwen3Omni(const IBaseModuleDesc::PTR& desc,
                                                               const PipelineDesc::PTR& pipeline_desc,
                                                               const VLMModelType& model_type)
    : LLMInferenceSDPAModule(desc, pipeline_desc, model_type) {
    m_model_config = ov::genai::modeling::models::Qwen3OmniProcessingConfig::from_json_file(m_models_path);
    OPENVINO_ASSERT(!m_stop_ids.empty(),
                    "LLMInferenceSDPAModule: no stop token ids found — "
                    "decoding will run for the full max_new_tokens budget");
    if (m_models_ir.empty()) {
        m_models_ir = m_models_path / "qwen3_omni_text_model.xml";
    }

    ov::AnyMap
        properties;  // = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY), ov::hint::num_requests(1)};
    if (!m_cache_dir.empty()) {
        properties.insert({ov::cache_dir.name(), m_cache_dir});
    }

    auto llm_model = ov::genai::utils::singleton_core().read_model(m_models_ir);
    m_compiled_text = ov::genai::utils::singleton_core().compile_model(llm_model, m_device, properties);
}

LLMInferenceSDPAModule::InputsParams::PTR LLMInferenceSDPAImpl_Qwen3Omni::parse_inputs(
    LLMInferenceSDPAModule::InputsParams::PTR inputs_params) {
    InputsParamsQwen3Omni::PTR cur_inputs;
    if (inputs_params == nullptr) {
        cur_inputs = InputsParamsQwen3Omni::create();
    } else {
        cur_inputs = std::static_pointer_cast<InputsParamsQwen3Omni>(inputs_params);
    }
    cur_inputs = std::dynamic_pointer_cast<InputsParamsQwen3Omni>(LLMInferenceSDPAModule::parse_inputs(cur_inputs));

    // position_ids required.
    cur_inputs->position_ids = get_input("position_ids").as<ov::Tensor>();
    // rope_deltas required.
    cur_inputs->rope_deltas = get_input("rope_delta").as<ov::Tensor>();

    if (exists_input("attention_mask")) {
        cur_inputs->attention_mask = inputs["attention_mask"].data.as<ov::Tensor>();
    } else {
        cur_inputs->attention_mask = LLMInferenceSDPAModule_Utils::build_attention_mask(cur_inputs->position_ids);
    }
    if (exists_input("visual_embeds") && exists_input("visual_pos_mask")) {
        cur_inputs->visual_embeds = inputs["visual_embeds"].data.as<ov::Tensor>();
        cur_inputs->visual_pos_mask = inputs["visual_pos_mask"].data.as<ov::Tensor>();
    }
    if (exists_input("audio_embeds") && exists_input("audio_pos_mask")) {
        cur_inputs->audio_embeds = inputs["audio_embeds"].data.as<ov::Tensor>();
        cur_inputs->audio_pos_mask = inputs["audio_pos_mask"].data.as<ov::Tensor>();
    }
    if (exists_input("deepstack_embeds")) {
        cur_inputs->deepstack_embeds = inputs["deepstack_embeds"].data.as<std::vector<ov::Tensor>>();
    }
    // cast to return type
    return std::static_pointer_cast<InputsParams>(cur_inputs);
}

void LLMInferenceSDPAImpl_Qwen3Omni::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();
    auto inputs_params = std::dynamic_pointer_cast<InputsParamsQwen3Omni>(parse_inputs());

    std::string generated_text = run_qwen3_omni_decode(inputs_params->input_ids,
                                                       inputs_params->attention_mask,
                                                       inputs_params->position_ids,
                                                       inputs_params->rope_deltas,
                                                       inputs_params->visual_embeds,
                                                       inputs_params->visual_pos_mask,
                                                       inputs_params->deepstack_embeds,
                                                       inputs_params->audio_embeds,
                                                       inputs_params->audio_pos_mask);
    GENAI_INFO("LLM output: " + generated_text);
    this->outputs["generated_text"].data = generated_text;
}

std::string LLMInferenceSDPAImpl_Qwen3Omni::run_qwen3_omni_decode(
    const ov::Tensor& input_ids,
    const ov::Tensor& attention_mask,
    const ov::Tensor& position_ids,
    const ov::Tensor& rope_deltas,
    const std::optional<ov::Tensor>& visual_embeds,
    const std::optional<ov::Tensor>& visual_pos_mask,
    const std::optional<std::vector<ov::Tensor>>& deepstack_embeds,
    const std::optional<ov::Tensor>& audio_embeds,
    const std::optional<ov::Tensor>& audio_pos_mask) {
    using TIO = ov::genai::modeling::models::Qwen3OmniTextIO;
    const auto& model_config = m_model_config;

    const size_t batch = input_ids.get_shape()[0];
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape()[1]);

    auto beam_idx = LLMInferenceSDPAModule_Utils::make_beam_idx(batch);
    auto text_req = m_compiled_text->create_infer_request();
    text_req.reset_state();

    // --- Prefill ---
    text_req.set_tensor(TIO::kInputIds, input_ids);
    text_req.set_tensor(TIO::kAttentionMask, attention_mask);
    text_req.set_tensor(TIO::kPositionIds, position_ids);
    text_req.set_tensor(TIO::kBeamIdx, beam_idx);
    if (visual_embeds.has_value() && visual_pos_mask.has_value()) {
        text_req.set_tensor(TIO::kVisualEmbeds, visual_embeds.value());
        text_req.set_tensor(TIO::kVisualPosMask, visual_pos_mask.value());
    } else {
        text_req.set_tensor(
            TIO::kVisualEmbeds,
            LLMInferenceSDPAModule_Utils::make_zeros(
                ov::element::f32,
                {batch, static_cast<size_t>(prompt_len), static_cast<size_t>(model_config.thinker.text.hidden_size)}));
        text_req.set_tensor(
            TIO::kVisualPosMask,
            LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, static_cast<size_t>(prompt_len)}));
    }
    if (deepstack_embeds.has_value()) {
        for (size_t i = 0; i < deepstack_embeds->size(); i++) {
            const std::string name =
                std::string(ov::genai::modeling::models::Qwen3OmniVisionIO::kDeepstackEmbedsPrefix) + "." +
                std::to_string(i);
            text_req.set_tensor(name, deepstack_embeds.value()[i]);
        }
    } else {
        for (size_t i = 0; i < model_config.thinker.vision.deepstack_visual_indexes.size(); i++) {
            const std::string name =
                std::string(ov::genai::modeling::models::Qwen3OmniVisionIO::kDeepstackEmbedsPrefix) + "." +
                std::to_string(i);
            text_req.set_tensor(
                name,
                LLMInferenceSDPAModule_Utils::make_zeros(ov::element::f32,
                                                         {batch,
                                                          static_cast<size_t>(prompt_len),
                                                          static_cast<size_t>(model_config.thinker.text.hidden_size)}));
        }
    }
    if (audio_embeds.has_value() && audio_pos_mask.has_value()) {
        text_req.set_tensor(TIO::kAudioFeatures, audio_embeds.value());
        text_req.set_tensor(TIO::kAudioPosMask, audio_pos_mask.value());
    } else {
        ov::Tensor prefill_audio_features(
            ov::element::f32,
            {batch, input_ids.get_shape()[1], static_cast<size_t>(model_config.thinker.text.hidden_size)});
        std::memset(prefill_audio_features.data(), 0, prefill_audio_features.get_byte_size());
        text_req.set_tensor(TIO::kAudioFeatures, prefill_audio_features);
        ov::Tensor prefill_audio_pos_mask(ov::element::boolean, {batch, input_ids.get_shape()[1]});
        std::memset(prefill_audio_pos_mask.data(), 0, prefill_audio_pos_mask.get_byte_size());
        text_req.set_tensor(TIO::kAudioPosMask, prefill_audio_pos_mask);
    }

    const auto t_prefill0 = std::chrono::steady_clock::now();
    {
        PROFILE(pm, "LLMInferenceSDPAModule::run_text_decode prefill infer");
        text_req.infer();
    }
    const auto t_prefill1 = std::chrono::steady_clock::now();
    int64_t next_id = LLMInferenceSDPAModule_Utils::argmax_last(text_req.get_tensor(TIO::kLogits));

    // --- Decode loop ---
    std::vector<int64_t> generated{next_id};
    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::i64, {batch, 1});
    for (size_t b = 0; b < batch; ++b)
        step_mask.data<int64_t>()[b] = 1;

    ov::Tensor dec_vis = LLMInferenceSDPAModule_Utils::make_zeros(
        ov::element::f32,
        {batch, 1, static_cast<size_t>(model_config.thinker.text.hidden_size)});
    ov::Tensor dec_vis_mask = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, 1});
    ov::Tensor decode_audio_features = LLMInferenceSDPAModule_Utils::make_zeros(
        ov::element::f32,
        {batch, 1, static_cast<size_t>(model_config.thinker.text.hidden_size)});
    ov::Tensor decode_audio_pos_mask = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, 1});
    std::vector<ov::Tensor> decode_deepstack;
    decode_deepstack.reserve(model_config.thinker.vision.deepstack_visual_indexes.size());
    for (size_t i = 0; i < model_config.thinker.vision.deepstack_visual_indexes.size(); ++i) {
        decode_deepstack.push_back(LLMInferenceSDPAModule_Utils::make_zeros(
            ov::element::f32,
            {batch, 1, static_cast<size_t>(model_config.thinker.text.hidden_size)}));
    }

    int64_t past_len = prompt_len;

    size_t decode_steps = 0;
    const auto t_dec0 = std::chrono::steady_clock::now();

    for (size_t step = 1; step < m_max_new_tokens; ++step) {
        if (!m_stop_ids.empty() && m_stop_ids.count(next_id))
            break;

        for (size_t b = 0; b < batch; ++b)
            step_ids.data<int64_t>()[b] = next_id;

        ov::Tensor pos =
            ov::genai::modeling::models::Qwen3OmniInputPlanner::build_decode_position_ids(rope_deltas, past_len, 1);

        text_req.set_tensor(TIO::kInputIds, step_ids);
        text_req.set_tensor(TIO::kAttentionMask, step_mask);
        text_req.set_tensor(TIO::kPositionIds, pos);
        text_req.set_tensor(TIO::kBeamIdx, beam_idx);
        text_req.set_tensor(TIO::kVisualEmbeds, dec_vis);
        text_req.set_tensor(TIO::kVisualPosMask, dec_vis_mask);
        text_req.set_tensor(TIO::kAudioFeatures, decode_audio_features);
        text_req.set_tensor(TIO::kAudioPosMask, decode_audio_pos_mask);
        for (size_t i = 0; i < decode_deepstack.size(); ++i) {
            const std::string name = std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) +
                                     "." + std::to_string(i);
            text_req.set_tensor(name, decode_deepstack[i]);
        }

        {
            PROFILE(pm, "LLMInferenceSDPAModule::run_qwen3_omni_decode step infer");
            text_req.infer();
        }

        next_id = LLMInferenceSDPAModule_Utils::argmax_last(text_req.get_tensor(TIO::kLogits));
        generated.push_back(next_id);
        ++decode_steps;
        ++past_len;
    }

    const auto t_dec1 = std::chrono::steady_clock::now();

    if (LLMInferenceSDPAModule_Utils::dump_performance_enabled()) {
        const double ttft_ms = LLMInferenceSDPAModule_Utils::elapsed_ms(t_prefill0, t_prefill1);
        const double decode_ms = LLMInferenceSDPAModule_Utils::elapsed_ms(t_dec0, t_dec1);
        const double tpot_ms = decode_steps > 0 ? decode_ms / static_cast<double>(decode_steps) : 0.0;
        const double throughput =
            decode_steps > 0 && decode_ms > 0.0 ? static_cast<double>(decode_steps) * 1000.0 / decode_ms : 0.0;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Mode: sdpa / vl\n"
                  << "Device: " << m_device << "\n"
                  << "Prompt token size: " << prompt_len << "\n"
                  << "Output token size: " << generated.size() << "\n"
                  << "TTFT: " << ttft_ms << " ms\n"
                  << "Decode time: " << decode_ms << " ms\n";
        if (decode_steps > 0) {
            std::cout << "TPOT: " << tpot_ms << " ms/token\n"
                      << "Throughput: " << throughput << " tokens/s\n";
        } else {
            std::cout << "TPOT: N/A\nThroughput: N/A\n";
        }
    }

    // Decode tokens to text
    if (m_tokenizer) {
        return m_tokenizer->decode(generated, ov::genai::skip_special_tokens(true));
    } else {
        std::ostringstream oss;
        for (auto id : generated)
            oss << id << ' ';
        return oss.str();
    }
}

}  // namespace ov::genai::module

#endif  // ENABLE_MODELING_PRIVATE