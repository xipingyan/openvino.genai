// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_omni.hpp"

#if defined(ENABLE_MODELING_PRIVATE)

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
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: 'prefill_model_path' param is required for Qwen3-Omni");
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
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: 'decode_model_path' param is required for Qwen3-Omni");
		return false;
	}
	const std::shared_ptr<ov::Model> decode_model = load_model(decode_model_path.value());
	if (!decode_model) {
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: Failed to load decode model");
		return false;
	}
	auto compiled_decode_model =
		::ov::genai::utils::singleton_core().compile_model(decode_model, m_device, tts_props);
	m_decode_infer = std::make_unique<ov::InferRequest>(compiled_decode_model.create_infer_request());

	const std::optional<std::filesystem::path> codec_embedding_model_path = get_model_path("codec_embedding_model_path");
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

	const std::optional<std::filesystem::path> ar_dir_param = get_model_path("code_predictor_ar_model_path");
	if (!ar_dir_param.has_value()) {
		GENAI_ERR("TextToSpeechModule[" + module_desc->name +
				  "]: 'code_predictor_ar_model_path' param is required for Qwen3-Omni");
		return false;
	}
	const std::filesystem::path ar_dir(ar_dir_param.value());
	for (int step = 0;; ++step) {
		const auto xml = ar_dir / ("qwen3_omni_code_predictor_ar_model_step_" + std::to_string(step) + ".xml");
		if (!std::filesystem::exists(xml)) {
			break;
		}
		auto model = ::ov::genai::utils::singleton_core().read_model(xml);
		auto compiled = ::ov::genai::utils::singleton_core().compile_model(model, m_device, tts_props);
		m_code_predictor_ar_infers.push_back(std::make_unique<ov::InferRequest>(compiled.create_infer_request()));
	}
	if (m_code_predictor_ar_infers.empty()) {
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: No AR step models found in " + ar_dir.string());
		return false;
	}

	const std::optional<std::filesystem::path> sce_dir_param = get_model_path("code_predictor_single_codec_embed_model_path");
	if (!sce_dir_param.has_value()) {
		GENAI_ERR("TextToSpeechModule[" + module_desc->name +
				  "]: 'code_predictor_single_codec_embed_model_path' param is required for Qwen3-Omni");
		return false;
	}
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
	}
	if (m_code_predictor_single_codec_embed_infers.empty()) {
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: No single-codec-embed step models found in " +
				  sce_dir.string());
		return false;
	}

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
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: 'speech_decoder_model_path' param is required for Qwen3-Omni");
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
			GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: 'tokenizer_path' param is required for Qwen3-Omni");
			return false;
		}
		m_tokenizer = std::make_unique<Tokenizer>(tokenizer_path.value());
	} catch (const std::exception& error) {
		GENAI_ERR("TextToSpeechModule[" + module_desc->name + "]: tokenizer init failed: " + std::string(error.what()));
		return false;
	}

	return true;
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

std::pair<ov::Tensor, int> TextToSpeechImpl_Qwen3Omni::qwen3_omni_text_to_speech(const std::string& text) {
	const auto talker_cfg = to_qwen3_omni_talker_config(m_config);
	const auto cp_cfg = to_qwen3_omni_code_predictor_config(m_config);
	const int cp_steps = std::max(1, cp_cfg.num_code_groups - 1);

	// --- Tokenize text ---
	auto tok_result = m_tokenizer->encode(text, ov::genai::add_special_tokens(false));
	auto tok_ids_tensor = tok_result.input_ids;
	size_t text_len = tok_ids_tensor.get_shape()[1];
	const int64_t* tok_ptr = tok_ids_tensor.data<int64_t>();
	std::vector<int64_t> text_token_ids(tok_ptr, tok_ptr + text_len);

	// --- Config values ---
	constexpr size_t batch = 1;
	const size_t hidden_size = static_cast<size_t>(talker_cfg.hidden_size);
	const size_t num_layers = static_cast<size_t>(talker_cfg.num_hidden_layers);
	const size_t num_kv_heads = static_cast<size_t>(talker_cfg.num_key_value_heads);
	const size_t head_dim = static_cast<size_t>(talker_cfg.head_dim);
	const size_t vocab_size = static_cast<size_t>(talker_cfg.vocab_size);
	const size_t cp_vocab_size = static_cast<size_t>(cp_cfg.vocab_size);

	const int64_t tts_pad_id = m_config.tts_pad_token_id;
	const int64_t tts_eos_id = m_config.tts_eos_token_id;
	const int64_t codec_bos = talker_cfg.codec_bos_token_id;
	const int64_t codec_eos = talker_cfg.codec_eos_token_id;
	const int64_t codec_pad = talker_cfg.codec_pad_token_id;
	const int64_t codec_nothink = m_config.talker_config_raw.value("codec_nothink_id", 2155);
	const int64_t codec_think_bos = m_config.talker_config_raw.value("codec_think_bos_id", 2156);
	const int64_t codec_think_eos = m_config.talker_config_raw.value("codec_think_eos_id", 2157);

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
		m_embedding_infer->set_tensor("text_input_ids", t_text_ids);
		m_embedding_infer->set_tensor("codec_input_ids", t_codec_ids);
		m_embedding_infer->set_tensor("codec_mask", t_codec_mask);
		{
			PROFILE(pm, "embedding_model infer");
			m_embedding_infer->infer();
		}
		auto embeddings = m_embedding_infer->get_tensor("inputs_embeds");
		const float* embeddings_data = embeddings.data<float>();
		const size_t trailing_hidden_size = embeddings.get_shape().back();
		for (size_t token_index = 0; token_index < trailing_len; ++token_index) {
			const float* token_ptr = embeddings_data + token_index * trailing_hidden_size;
			trailing_text_embeds[token_index].assign(token_ptr, token_ptr + trailing_hidden_size);
		}
	}

	// --- Get prefill embeddings ---
	{
		ov::Tensor text_tensor(ov::element::i64, {batch, prefill_len}, full_text_ids.data());
		ov::Tensor codec_tensor(ov::element::i64, {batch, prefill_len}, full_codec_ids.data());
		ov::Tensor mask_tensor(ov::element::f32, {batch, prefill_len}, full_codec_mask.data());
		m_embedding_infer->set_tensor("text_input_ids", text_tensor);
		m_embedding_infer->set_tensor("codec_input_ids", codec_tensor);
		m_embedding_infer->set_tensor("codec_mask", mask_tensor);
		{
			PROFILE(pm, "embedding_model infer");
			m_embedding_infer->infer();
		}
	}
	auto prefill_embeds = m_embedding_infer->get_tensor("inputs_embeds");

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

	const int num_layers_total = cp_steps + 1;
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
		m_codec_embedding_infer->set_tensor("codec_input_ids", ov::Tensor(ov::element::i64, {batch, 1}, layer0_vector.data()));
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

		if (m_code_predictor_ar_infers.size() < static_cast<size_t>(cp_steps) ||
			m_code_predictor_single_codec_embed_infers.size() < static_cast<size_t>(cp_steps)) {
			OPENVINO_THROW("TextToSpeechModule: insufficient code predictor steps loaded for code predictor models");
		}

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
			int64_t layer_token = sample_codec_token(step_logits.data<float>(),
													 cp_vocab_size,
													 temperature,
													 top_k,
													 top_p,
													 1.0f,
													 nullptr,
													 nullptr,
													 rng);

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
			autoregressive_sequence.insert(
				autoregressive_sequence.end(),
				layer_embed.data<float>(),
				layer_embed.data<float>() + hidden_size);
		}

		std::vector<float> codec_sum(hidden_size, 0.0f);
		for (size_t i = 0; i < hidden_size; ++i) {
			codec_sum[i] += layer0_embed[i];
		}

		std::vector<std::vector<int64_t>> layer_tokens_vec(cp_steps);
		std::vector<ov::Tensor> layer_tensors(cp_steps);
		for (int layer = 0; layer < cp_steps; ++layer) {
			layer_tokens_vec[layer] = {current_layer_tokens[layer]};
			layer_tensors[layer] = ov::Tensor(ov::element::i64, {batch, 1}, layer_tokens_vec[layer].data());
			m_code_predictor_single_codec_embedding_infer->set_tensor(
				"codec_input_" + std::to_string(layer),
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
			(static_cast<size_t>(frame) < trailing_text_embeds.size()) ? trailing_text_embeds[frame] : tts_pad_embed;
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
			m_codec_embedding_infer->set_tensor(
				"codec_input_ids",
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
