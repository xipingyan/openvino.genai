// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_5.hpp"
#include "utils.hpp"
#include "module_genai/utils/profiler.hpp"
#ifdef ENABLE_OPENVINO_NEW_ARCH
namespace ov::genai::module {

LLMInferenceSDPAModule::InputsParams::PTR LLMInferenceSDPAImpl_Qwen3_5::parse_inputs(
    LLMInferenceSDPAModule::InputsParams::PTR inputs_params) {
    InputsParamsQwen3_5::PTR cur_inputs;
    if (inputs_params == nullptr) {
        cur_inputs = InputsParamsQwen3_5::create();
    } else {
        cur_inputs = std::static_pointer_cast<InputsParamsQwen3_5>(inputs_params);
    }
    cur_inputs = std::dynamic_pointer_cast<InputsParamsQwen3_5>(LLMInferenceSDPAModule::parse_inputs(cur_inputs));

    cur_inputs->attention_mask = LLMInferenceSDPAModule_Utils::build_attention_mask(cur_inputs->position_ids);

    cur_inputs->is_vl = (exists_input("visual_embeds") && exists_input("visual_pos_mask") && exists_input("grid_thw") &&
                         exists_input("position_ids") && exists_input("rope_delta"));
    if (cur_inputs->is_vl) {
        // ---- VL mode ----
        cur_inputs->visual_embeds = inputs["visual_embeds"].data.as<ov::Tensor>();
        cur_inputs->visual_pos_mask = inputs["visual_pos_mask"].data.as<ov::Tensor>();
        cur_inputs->grid_thw = inputs["grid_thw"].data.as<std::vector<ov::Tensor>>();
        cur_inputs->position_ids = inputs["position_ids"].data.as<ov::Tensor>();
        cur_inputs->rope_deltas = inputs["rope_delta"].data.as<ov::Tensor>();
        if (exists_input("deepstack_embeds")) {
            cur_inputs->deepstack_embeds = inputs["deepstack_embeds"].data.as<std::vector<ov::Tensor>>();
        }
    } else {
        // ---- Text mode ----
        ov::genai::modeling::models::Qwen3_5InputPlanner planner(m_model_config);
        cur_inputs->plan = planner.build_plan(cur_inputs->input_ids, &cur_inputs->attention_mask, nullptr);
    }

    return std::static_pointer_cast<InputsParams>(cur_inputs);
}

LLMInferenceSDPAImpl_Qwen3_5::LLMInferenceSDPAImpl_Qwen3_5(const IBaseModuleDesc::PTR& desc,
                                                           const PipelineDesc::PTR& pipeline_desc,
                                                           const VLMModelType& model_type)
    : LLMInferenceSDPAModule(desc, pipeline_desc, model_type) {
    //
    m_model_config = ov::genai::modeling::models::Qwen3_5Config::from_json_file(m_models_path);

    if (m_model_config.text.eos_token_id > 0) {
        m_stop_ids.insert(m_model_config.text.eos_token_id);
    }

    OPENVINO_ASSERT(!m_stop_ids.empty(),
                    "LLMInferenceSDPAModule: no stop token ids found — "
                    "decoding will run for the full max_new_tokens budget");

    std::shared_ptr<ov::Model> llm_model;

    // Resolve IR paths
    if (m_models_ir.empty()) {
        const std::string qs = LLMInferenceSDPAModule_Utils::quant_suffix();
        GENAI_INFO("LLMInferenceSDPAModule: quant suffix = " + qs);

        const auto text_xml = m_models_path / ("qwen3_5_text" + qs + ".xml");
        const auto text_bin = m_models_path / ("qwen3_5_text" + qs + ".bin");
        const auto text_vl_xml = m_models_path / ("qwen3_5_text_vl" + qs + ".xml");
        const auto text_vl_bin = m_models_path / ("qwen3_5_text_vl" + qs + ".bin");

        // Prefer VL IR (supports both text and VL modes at runtime).
        // Fall back to text-only IR when VL IR is not available.
        std::filesystem::path chosen_text_xml, chosen_text_bin;
        if (LLMInferenceSDPAModule_Utils::has_ir_pair(text_vl_xml, text_vl_bin)) {
            chosen_text_xml = text_vl_xml;
            chosen_text_bin = text_vl_bin;
            m_text_uses_vl_ir = true;
        } else if (LLMInferenceSDPAModule_Utils::has_ir_pair(text_xml, text_bin)) {
            chosen_text_xml = text_xml;
            chosen_text_bin = text_bin;
            m_text_uses_vl_ir = false;
            GENAI_INFO("VL text IR not found; using text-only IR (VL mode will not be available)");
        } else {
            GENAI_ERR("No text IR found. Expected: " + text_vl_xml.string() + " or " + text_xml.string());
        }

        // Compile the text model
        GENAI_INFO("LLMInferenceSDPAModule: loading text IR: " + chosen_text_xml.string());
        llm_model = ov::genai::utils::singleton_core().read_model(chosen_text_xml.string(), chosen_text_bin.string());
    }
    else {
        GENAI_INFO("LLMInferenceSDPAModule: using user-provided IR: " + m_models_ir.string());
        llm_model = ov::genai::utils::singleton_core().read_model(m_models_ir);
    }

    // Verify VL inputs when using VL IR
    if (m_text_uses_vl_ir) {
        using IO = ov::genai::modeling::models::Qwen3_5TextIO;
        OPENVINO_ASSERT((LLMInferenceSDPAModule_Utils::has_model_input(llm_model, IO::kVisualEmbeds) &&
                         LLMInferenceSDPAModule_Utils::has_model_input(llm_model, IO::kVisualPosMask)),
                        "LLM model should have VL inputs (visual_embeds / visual_pos_mask)");
    }

    ov::AnyMap properties = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY), ov::hint::num_requests(1)};
    if (!m_cache_dir.empty()) {
        properties.insert({ov::cache_dir.name(), m_cache_dir});
    }

    m_compiled_text = ov::genai::utils::singleton_core().compile_model(llm_model, m_device, properties);
}

void LLMInferenceSDPAImpl_Qwen3_5::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    std::string generated_text;

    prepare_inputs();
    auto inputs_params = std::dynamic_pointer_cast<InputsParamsQwen3_5>(parse_inputs());

    if (inputs_params->is_vl) {
        generated_text = run_vl_decode(inputs_params->input_ids,
                                       inputs_params->attention_mask,
                                       inputs_params->position_ids,
                                       inputs_params->rope_deltas,
                                       inputs_params->visual_embeds,
                                       inputs_params->visual_pos_mask,
                                       inputs_params->deepstack_embeds);

    } else {
        // ---- Text mode ----
        generated_text = run_text_decode(inputs_params->input_ids,
                                         inputs_params->attention_mask,
                                         inputs_params->plan.position_ids,
                                         inputs_params->plan.rope_deltas);
    }
    GENAI_INFO("LLM output: " + generated_text);
    this->outputs["generated_text"].data = generated_text;
}

std::string LLMInferenceSDPAImpl_Qwen3_5::run_text_decode(const ov::Tensor& input_ids,
                                                          const ov::Tensor& attention_mask,
                                                          const ov::Tensor& position_ids,
                                                          const ov::Tensor& rope_deltas) {
    using TIO = ov::genai::modeling::models::Qwen3_5TextIO;

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

    if (m_text_uses_vl_ir) {
        // Feed zero visual inputs for text-only usage of VL IR
        text_req.set_tensor(
            TIO::kVisualEmbeds,
            LLMInferenceSDPAModule_Utils::make_zeros(ov::element::f32,
                       {batch, static_cast<size_t>(prompt_len), static_cast<size_t>(m_model_config.text.hidden_size)}));
        text_req.set_tensor(TIO::kVisualPosMask,
                            LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, static_cast<size_t>(prompt_len)}));
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

    ov::Tensor dec_vis, dec_vis_mask;
    if (m_text_uses_vl_ir) {
        dec_vis = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::f32, {batch, 1, static_cast<size_t>(m_model_config.text.hidden_size)});
        dec_vis_mask = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, 1});
    }

    int64_t past_len = prompt_len;

    size_t decode_steps = 0;
    const auto t_dec0 = std::chrono::steady_clock::now();

    for (size_t step = 1; step < m_max_new_tokens; ++step) {
        if (!m_stop_ids.empty() && m_stop_ids.count(next_id))
            break;

        for (size_t b = 0; b < batch; ++b)
            step_ids.data<int64_t>()[b] = next_id;

        auto pos =
            ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(rope_deltas, past_len, 1);

        text_req.set_tensor(TIO::kInputIds, step_ids);
        text_req.set_tensor(TIO::kAttentionMask, step_mask);
        text_req.set_tensor(TIO::kPositionIds, pos);
        text_req.set_tensor(TIO::kBeamIdx, beam_idx);
        if (m_text_uses_vl_ir) {
            text_req.set_tensor(TIO::kVisualEmbeds, dec_vis);
            text_req.set_tensor(TIO::kVisualPosMask, dec_vis_mask);
        }

        {
            PROFILE(pm, "LLMInferenceSDPAModule::run_text_decode step infer");
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
        std::cout << "Mode: sdpa / text\n"
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
        // Fallback: return token ids as text
        std::ostringstream oss;
        for (auto id : generated)
            oss << id << ' ';
        return oss.str();
    }
}

// ============================================================================
// VL decode (with visual embeddings) — stateful prefill + greedy decode
// ============================================================================

std::string LLMInferenceSDPAImpl_Qwen3_5::run_vl_decode(
    const ov::Tensor& input_ids,
    const ov::Tensor& attention_mask,
    const ov::Tensor& position_ids,
    const ov::Tensor& rope_deltas,
    const ov::Tensor& visual_embeds,
    const ov::Tensor& visual_pos_mask,
    const std::optional<std::vector<ov::Tensor>>& deepstack_embeds) {
    using TIO = ov::genai::modeling::models::Qwen3_5TextIO;

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
    text_req.set_tensor(TIO::kVisualEmbeds, visual_embeds);
    text_req.set_tensor(TIO::kVisualPosMask, visual_pos_mask);
    if (deepstack_embeds.has_value()) {
        for (size_t i = 0; i < deepstack_embeds->size(); i++) {
            const std::string name = std::string(ov::genai::modeling::models::Qwen3_5TextIO::kDeepstackEmbedsPrefix) +
                                     "." + std::to_string(i);
            text_req.set_tensor(name, deepstack_embeds.value()[i]);
        }
        ov::Tensor prefill_audio_features(
            ov::element::f32,
            {batch, input_ids.get_shape()[1], static_cast<size_t>(m_model_config.text.hidden_size)});
        std::memset(prefill_audio_features.data(), 0, prefill_audio_features.get_byte_size());
        text_req.set_tensor("audio_features", prefill_audio_features);
        ov::Tensor prefill_audio_pos_mask(ov::element::boolean, {batch, input_ids.get_shape()[1]});
        std::memset(prefill_audio_pos_mask.data(), 0, prefill_audio_pos_mask.get_byte_size());
        text_req.set_tensor("audio_pos_mask", prefill_audio_pos_mask);
    }

    const auto t_prefill0 = std::chrono::steady_clock::now();
    {
        PROFILE(pm, "LLMInferenceSDPAModule::run_vl_decode prefill infer");
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

    ov::Tensor dec_vis = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::f32, {batch, 1, static_cast<size_t>(m_model_config.text.hidden_size)});
    ov::Tensor dec_vis_mask = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, 1});
    ov::Tensor decode_audio_features =
        LLMInferenceSDPAModule_Utils::make_zeros(ov::element::f32, {batch, 1, static_cast<size_t>(m_model_config.text.hidden_size)});
    ov::Tensor decode_audio_pos_mask = LLMInferenceSDPAModule_Utils::make_zeros(ov::element::boolean, {batch, 1});
    std::vector<ov::Tensor> decode_deepstack;
    if (deepstack_embeds.has_value()) {
        decode_deepstack.reserve(deepstack_embeds.value().size());
        for (size_t i = 0; i < deepstack_embeds.value().size(); ++i) {
            decode_deepstack.push_back(
                LLMInferenceSDPAModule_Utils::make_zeros(ov::element::f32, {batch, 1, static_cast<size_t>(m_model_config.text.hidden_size)}));
        }
    }

    int64_t past_len = prompt_len;

    size_t decode_steps = 0;
    const auto t_dec0 = std::chrono::steady_clock::now();

    for (size_t step = 1; step < m_max_new_tokens; ++step) {
        if (!m_stop_ids.empty() && m_stop_ids.count(next_id))
            break;

        for (size_t b = 0; b < batch; ++b)
            step_ids.data<int64_t>()[b] = next_id;

        ov::Tensor pos;
        if (!deepstack_embeds.has_value()) {
            pos = ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(rope_deltas, past_len, 1);
        } else {
            pos = ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(rope_deltas, past_len, 1);
            text_req.set_tensor("audio_features", decode_audio_features);
            text_req.set_tensor("audio_pos_mask", decode_audio_pos_mask);
            for (size_t i = 0; i < decode_deepstack.size(); ++i) {
                const std::string name =
                    std::string(ov::genai::modeling::models::Qwen3_5TextIO::kDeepstackEmbedsPrefix) + "." +
                    std::to_string(i);
                text_req.set_tensor(name, decode_deepstack[i]);
            }
        }

        text_req.set_tensor(TIO::kInputIds, step_ids);
        text_req.set_tensor(TIO::kAttentionMask, step_mask);
        text_req.set_tensor(TIO::kPositionIds, pos);
        text_req.set_tensor(TIO::kBeamIdx, beam_idx);
        text_req.set_tensor(TIO::kVisualEmbeds, dec_vis);
        text_req.set_tensor(TIO::kVisualPosMask, dec_vis_mask);

        {
            PROFILE(pm, "LLMInferenceSDPAModule::run_vl_decode step infer");
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

namespace LLMInferenceSDPAModule_Utils {

// ============================================================================
// Helpers (ported from md_qwen3_5_modeling.cpp)
// ============================================================================

std::string quant_suffix() {
    using namespace ov::genai::modeling::weights;
    auto cfg = parse_quantization_config_from_env();
    if (!cfg.enabled()) {
        cfg.mode = QuantizationConfig::Mode::INT4_ASYM;
        cfg.backup_mode = QuantizationConfig::Mode::INT4_ASYM;
        cfg.group_size = 128;
    }
    auto tok = [](QuantizationConfig::Mode m) -> std::string {
        switch (m) {
        case QuantizationConfig::Mode::INT4_SYM:
            return "4s";
        case QuantizationConfig::Mode::INT4_ASYM:
            return "4a";
        case QuantizationConfig::Mode::INT8_SYM:
            return "8s";
        case QuantizationConfig::Mode::INT8_ASYM:
            return "8a";
        default:
            return "n";
        }
    };
    return "_q" + tok(cfg.mode) + "_b" + tok(cfg.backup_mode) + "_g" + std::to_string(cfg.group_size);
}

bool has_ir_pair(const std::filesystem::path& xml, const std::filesystem::path& bin) {
    return std::filesystem::is_regular_file(xml) && std::filesystem::is_regular_file(bin);
}

bool has_model_input(const std::shared_ptr<ov::Model>& m, const std::string& name) {
    for (const auto& inp : m->inputs())
        if (inp.get_names().count(name))
            return true;
    return false;
}

}  // namespace LLMInferenceSDPAModule_Utils

}  // namespace ov::genai::module

#endif  // ENABLE_OPENVINO_NEW_ARCH
