// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_tts_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

#include "nlohmann/json.hpp"

#include "modeling_qwen3_tts.hpp"
#include "modeling_qwen3_tts_talker.hpp"
#include "modeling_qwen3_tts_code_predictor.hpp"
#include "modeling_qwen3_tts_speech_decoder.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

using json = nlohmann::json;
using namespace ov::genai::safetensors;

// =============================================================================
// Custom Weight Source that pre-computes codebook embeddings for Speech Decoder
// =============================================================================
class SpeechDecoderWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    explicit SpeechDecoderWeightSource(SafetensorsData data) : m_data(std::move(data)) {
        // Build canonical names and pre-compute codebooks
        for (const auto& [hf_name, info] : m_data.tensor_infos) {
            // Convert HF names to canonical (layers.N -> layers[N])
            std::string canonical = hf_name;
            size_t pos = 0;
            while ((pos = canonical.find("layers.", pos)) != std::string::npos) {
                size_t dot_pos = canonical.find('.', pos + 7);
                if (dot_pos != std::string::npos) {
                    std::string num = canonical.substr(pos + 7, dot_pos - pos - 7);
                    bool is_num = !num.empty() && std::all_of(num.begin(), num.end(), ::isdigit);
                    if (is_num) {
                        canonical = canonical.substr(0, pos) + "layers[" + num + "]" + canonical.substr(dot_pos);
                    }
                }
                pos++;
            }
            m_canonical_to_hf[canonical] = hf_name;
            m_keys.push_back(canonical);
        }
        precompute_codebooks();
    }
    
    std::vector<std::string> keys() const override { return m_keys; }
    
    bool has(const std::string& name) const override {
        if (m_precomputed.find(name) != m_precomputed.end()) return true;
        return m_canonical_to_hf.find(name) != m_canonical_to_hf.end();
    }
    
    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto precomp_it = m_precomputed.find(name);
        if (precomp_it != m_precomputed.end()) return precomp_it->second;
        
        auto it = m_canonical_to_hf.find(name);
        if (it == m_canonical_to_hf.end()) throw std::runtime_error("Weight not found: " + name);
        const std::string& hf_name = it->second;
        
        auto cache_it = m_tensor_cache.find(name);
        if (cache_it != m_tensor_cache.end()) return cache_it->second;
        
        const auto& info = m_data.tensor_infos.at(hf_name);
        auto mmap_it = m_data.tensor_mmap_info.find(hf_name);
        if (mmap_it != m_data.tensor_mmap_info.end()) {
            const auto& mmap_info = mmap_it->second;
            const uint8_t* data = mmap_info.holder->data_buffer() + mmap_info.offset;
            
            if (info.dtype == ov::element::bf16) {
                ov::Tensor f32_tensor(ov::element::f32, info.shape);
                const ov::bfloat16* src = reinterpret_cast<const ov::bfloat16*>(data);
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                m_tensor_cache[name] = std::move(f32_tensor);
            } else if (info.dtype == ov::element::f16) {
                ov::Tensor f32_tensor(ov::element::f32, info.shape);
                const ov::float16* src = reinterpret_cast<const ov::float16*>(data);
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                m_tensor_cache[name] = std::move(f32_tensor);
            } else {
                ov::Tensor copy_tensor(info.dtype, info.shape);
                std::memcpy(copy_tensor.data(), data, mmap_info.size);
                m_tensor_cache[name] = std::move(copy_tensor);
            }
            return m_tensor_cache.at(name);
        }
        
        auto tensor_it = m_data.tensors.find(hf_name);
        if (tensor_it != m_data.tensors.end()) {
            const auto& src_tensor = tensor_it->second;
            if (src_tensor.get_element_type() == ov::element::bf16) {
                ov::Tensor f32_tensor(ov::element::f32, src_tensor.get_shape());
                const ov::bfloat16* src = src_tensor.data<ov::bfloat16>();
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                m_tensor_cache[name] = std::move(f32_tensor);
            } else if (src_tensor.get_element_type() == ov::element::f16) {
                ov::Tensor f32_tensor(ov::element::f32, src_tensor.get_shape());
                const ov::float16* src = src_tensor.data<ov::float16>();
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                m_tensor_cache[name] = std::move(f32_tensor);
            } else {
                m_tensor_cache[name] = src_tensor;
            }
            return m_tensor_cache.at(name);
        }
        throw std::runtime_error("Tensor data not available: " + name);
    }

private:
    void precompute_codebooks() {
        // Precompute first codebook (semantic)
        std::string sum_name = "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
        std::string usage_name = "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage";
        if (m_data.tensor_infos.find(sum_name) != m_data.tensor_infos.end()) {
            auto embed = compute_codebook_embedding(sum_name, usage_name);
            std::string key = "decoder.quantizer.rvq_first.vq.layers[0]._codebook.embed";
            m_precomputed[key] = std::move(embed);
            m_keys.push_back(key);
            m_canonical_to_hf[key] = sum_name;
        }
        // Precompute rest codebooks (acoustic)
        for (int i = 0; i < 15; ++i) {
            std::string prefix = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(i);
            std::string sum_name_i = prefix + "._codebook.embedding_sum";
            std::string usage_name_i = prefix + "._codebook.cluster_usage";
            if (m_data.tensor_infos.find(sum_name_i) != m_data.tensor_infos.end()) {
                auto embed = compute_codebook_embedding(sum_name_i, usage_name_i);
                std::string key = "decoder.quantizer.rvq_rest.vq.layers[" + std::to_string(i) + "]._codebook.embed";
                m_precomputed[key] = std::move(embed);
                m_keys.push_back(key);
                m_canonical_to_hf[key] = sum_name_i;
            }
        }
    }
    
    ov::Tensor compute_codebook_embedding(const std::string& sum_name, const std::string& usage_name) {
        auto sum_tensor = load_tensor_from_mmap(sum_name);
        auto usage_tensor = load_tensor_from_mmap(usage_name);
        const auto& shape = sum_tensor.get_shape();
        ov::Tensor codebook(ov::element::f32, shape);
        const float* sum_data = sum_tensor.data<float>();
        const float* usage_data = usage_tensor.data<float>();
        float* cb_data = codebook.data<float>();
        size_t num_codes = shape[0], dim = shape[1];
        for (size_t i = 0; i < num_codes; ++i) {
            float u = std::max(usage_data[i], 1e-5f);
            for (size_t d = 0; d < dim; ++d) cb_data[i * dim + d] = sum_data[i * dim + d] / u;
        }
        return codebook;
    }
    
    ov::Tensor load_tensor_from_mmap(const std::string& name) {
        const auto& info = m_data.tensor_infos.at(name);
        auto mmap_it = m_data.tensor_mmap_info.find(name);
        if (mmap_it != m_data.tensor_mmap_info.end()) {
            const auto& mmap_info = mmap_it->second;
            const uint8_t* data = mmap_info.holder->data_buffer() + mmap_info.offset;
            if (info.dtype == ov::element::bf16) {
                ov::Tensor f32_tensor(ov::element::f32, info.shape);
                const ov::bfloat16* src = reinterpret_cast<const ov::bfloat16*>(data);
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                return f32_tensor;
            } else if (info.dtype == ov::element::f16) {
                ov::Tensor f32_tensor(ov::element::f32, info.shape);
                const ov::float16* src = reinterpret_cast<const ov::float16*>(data);
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                return f32_tensor;
            } else {
                ov::Tensor tensor(info.dtype, info.shape);
                std::memcpy(tensor.data(), data, mmap_info.size);
                return tensor;
            }
        }
        auto tensor_it = m_data.tensors.find(name);
        if (tensor_it != m_data.tensors.end()) {
            const auto& src_tensor = tensor_it->second;
            if (src_tensor.get_element_type() == ov::element::bf16) {
                ov::Tensor f32_tensor(ov::element::f32, src_tensor.get_shape());
                const ov::bfloat16* src = src_tensor.data<ov::bfloat16>();
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                return f32_tensor;
            } else if (src_tensor.get_element_type() == ov::element::f16) {
                ov::Tensor f32_tensor(ov::element::f32, src_tensor.get_shape());
                const ov::float16* src = src_tensor.data<ov::float16>();
                float* dst = f32_tensor.data<float>();
                for (size_t i = 0; i < f32_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                return f32_tensor;
            } else if (src_tensor.get_element_type() == ov::element::f32) {
                ov::Tensor copy(ov::element::f32, src_tensor.get_shape());
                std::memcpy(copy.data(), src_tensor.data(), src_tensor.get_byte_size());
                return copy;
            } else {
                ov::Tensor copy(src_tensor.get_element_type(), src_tensor.get_shape());
                std::memcpy(copy.data(), src_tensor.data(), src_tensor.get_byte_size());
                return copy;
            }
        }
        throw std::runtime_error("No tensor data for: " + name);
    }
    
    SafetensorsData m_data;
    std::vector<std::string> m_keys;
    std::unordered_map<std::string, std::string> m_canonical_to_hf;
    mutable std::unordered_map<std::string, ov::Tensor> m_tensor_cache;
    std::unordered_map<std::string, ov::Tensor> m_precomputed;
};

namespace ov {
namespace genai {
namespace modeling {
namespace models {

namespace {

// Parse talker config from JSON
Qwen3TTSTalkerConfig parse_talker_config(const json& config) {
    Qwen3TTSTalkerConfig cfg;
    auto talker_cfg = config.at("talker_config");
    cfg.hidden_size = talker_cfg.value("hidden_size", 2048);
    cfg.num_attention_heads = talker_cfg.value("num_attention_heads", 16);
    cfg.num_key_value_heads = talker_cfg.value("num_key_value_heads", 8);
    cfg.head_dim = talker_cfg.value("head_dim", 128);
    cfg.intermediate_size = talker_cfg.value("intermediate_size", 6144);
    cfg.num_hidden_layers = talker_cfg.value("num_hidden_layers", 28);
    cfg.vocab_size = talker_cfg.value("vocab_size", 3072);
    cfg.text_vocab_size = talker_cfg.value("text_vocab_size", 151936);
    cfg.text_hidden_size = talker_cfg.value("text_hidden_size", 896);
    cfg.rms_norm_eps = talker_cfg.value("rms_norm_eps", 1e-6f);
    cfg.rope_theta = talker_cfg.value("rope_theta", 1000000.0f);
    cfg.mrope_interleaved = talker_cfg.value("mrope_interleaved", true);
    if (talker_cfg.contains("mrope_section")) {
        cfg.mrope_section.clear();
        for (auto& v : talker_cfg["mrope_section"]) {
            cfg.mrope_section.push_back(v.get<int32_t>());
        }
    }
    cfg.codec_eos_token_id = talker_cfg.value("codec_eos_token_id", 2150);
    cfg.codec_bos_token_id = talker_cfg.value("codec_bos_token_id", 2149);
    cfg.codec_pad_token_id = talker_cfg.value("codec_pad_token_id", 2148);
    return cfg;
}

// Parse code predictor config from JSON
Qwen3TTSCodePredictorConfig parse_code_predictor_config(const json& config) {
    Qwen3TTSCodePredictorConfig cfg;
    auto talker_cfg = config.at("talker_config");
    auto cp_cfg = talker_cfg.at("code_predictor_config");
    cfg.hidden_size = cp_cfg.value("hidden_size", 1024);
    cfg.num_attention_heads = cp_cfg.value("num_attention_heads", 16);
    cfg.num_key_value_heads = cp_cfg.value("num_key_value_heads", 8);
    cfg.head_dim = cp_cfg.value("head_dim", 128);
    cfg.intermediate_size = cp_cfg.value("intermediate_size", 3072);
    cfg.num_hidden_layers = cp_cfg.value("num_hidden_layers", 5);
    cfg.vocab_size = cp_cfg.value("vocab_size", 2048);
    cfg.num_code_groups = cp_cfg.value("num_code_groups", 16);
    cfg.rms_norm_eps = cp_cfg.value("rms_norm_eps", 1e-6f);
    cfg.rope_theta = cp_cfg.value("rope_theta", 1000000.0f);
    cfg.talker_hidden_size = talker_cfg.value("hidden_size", 2048);
    return cfg;
}

}  // namespace

Qwen3TTSPipeline::Qwen3TTSPipeline(const std::filesystem::path& models_path,
                                   const std::string& device,
                                   const ov::AnyMap& properties) 
    : m_rng(42) {
    init_model_config(models_path);
    load_models(models_path, device, properties);
    
    // Initialize tokenizer
    m_tokenizer = std::make_unique<Tokenizer>(models_path);
}

void Qwen3TTSPipeline::init_model_config(const std::filesystem::path& root_dir) {
    const std::filesystem::path config_path = root_dir / "config.json";
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);
    
    json config = json::parse(file);
    
    // Parse special tokens
    m_tts_bos_token_id = config.value("tts_bos_token_id", 151672);
    m_tts_eos_token_id = config.value("tts_eos_token_id", 151673);
    m_tts_pad_token_id = config.value("tts_pad_token_id", 151671);
    
    auto talker_cfg = config.at("talker_config");
    m_codec_bos_id = talker_cfg.value("codec_bos_token_id", 2149);
    m_codec_eos_id = talker_cfg.value("codec_eos_token_id", 2150);
    m_codec_pad_id = talker_cfg.value("codec_pad_token_id", 2148);
    m_codec_nothink_id = talker_cfg.value("codec_nothink_id", 2155);
    m_codec_think_bos_id = talker_cfg.value("codec_think_bos_id", 2156);
    m_codec_think_eos_id = talker_cfg.value("codec_think_eos_id", 2157);
    
    // Talker config
    m_hidden_size = talker_cfg.value("hidden_size", 2048);
    m_num_layers = talker_cfg.value("num_hidden_layers", 28);
    m_num_kv_heads = talker_cfg.value("num_key_value_heads", 8);
    m_head_dim = talker_cfg.value("head_dim", 128);
    m_vocab_size = talker_cfg.value("vocab_size", 3072);
    
    // Code Predictor config
    if (talker_cfg.contains("code_predictor_config")) {
        auto cp_cfg = talker_cfg.at("code_predictor_config");
        m_cp_hidden_size = cp_cfg.value("hidden_size", 1024);
        m_cp_vocab_size = cp_cfg.value("vocab_size", 2048);
    }
}

void Qwen3TTSPipeline::load_models(const std::filesystem::path& models_path,
                                   const std::string& device,
                                   const ov::AnyMap& properties) {
    ov::Core core;
    
    // Load config
    std::ifstream config_file(models_path / "config.json");
    OPENVINO_ASSERT(config_file.is_open(), "Failed to open config.json");
    json config = json::parse(config_file);
    
    auto talker_cfg = parse_talker_config(config);
    auto cp_cfg = parse_code_predictor_config(config);
    SpeechDecoderConfig decoder_cfg;  // Use defaults
    
    // Load main model weights
    auto st_data = ::ov::genai::safetensors::load_safetensors(models_path);
    ::ov::genai::safetensors::SafetensorsWeightSource weight_source(std::move(st_data));
    ::ov::genai::safetensors::SafetensorsWeightFinalizer finalizer;
    
    // Create and compile Embedding model
    auto embed_model = create_qwen3_tts_embedding_model(talker_cfg, weight_source, finalizer);
    auto embed_compiled = core.compile_model(embed_model, device, properties);
    m_embed_infer = embed_compiled.create_infer_request();
    
    // Create and compile Talker Prefill model
    auto talker_prefill_model = create_qwen3_tts_talker_prefill_model(talker_cfg, weight_source, finalizer);
    auto talker_prefill_compiled = core.compile_model(talker_prefill_model, device, properties);
    m_talker_prefill_infer = talker_prefill_compiled.create_infer_request();
    
    // Create and compile Talker Decode model
    auto talker_decode_model = create_qwen3_tts_talker_decode_model(talker_cfg, weight_source, finalizer);
    auto talker_decode_compiled = core.compile_model(talker_decode_model, device, properties);
    m_talker_decode_infer = talker_decode_compiled.create_infer_request();
    
    // Create and compile Talker codec embedding model
    auto talker_codec_model = create_qwen3_tts_codec_embedding_model(talker_cfg, weight_source, finalizer);
    auto talker_codec_compiled = core.compile_model(talker_codec_model, device, properties);
    m_talker_codec_infer = talker_codec_compiled.create_infer_request();
    
    // Create 15 AR Code Predictor models
    m_cp_ar_infer.reserve(15);
    for (int step = 0; step < 15; ++step) {
        auto ar_model = create_qwen3_tts_code_predictor_ar_model(cp_cfg, step, weight_source, finalizer);
        auto ar_compiled = core.compile_model(ar_model, device, properties);
        m_cp_ar_infer.push_back(ar_compiled.create_infer_request());
    }
    
    // Create 15 single codec embedding models
    m_cp_embed_infer.reserve(15);
    for (int layer = 0; layer < 15; ++layer) {
        auto embed_model = create_qwen3_tts_code_predictor_single_codec_embed_model(cp_cfg, layer, weight_source, finalizer);
        auto embed_compiled = core.compile_model(embed_model, device, properties);
        m_cp_embed_infer.push_back(embed_compiled.create_infer_request());
    }
    
    // Create Code Predictor codec embedding model
    auto cp_codec_model = create_qwen3_tts_code_predictor_codec_embed_model(cp_cfg, weight_source, finalizer);
    auto cp_codec_compiled = core.compile_model(cp_codec_model, device, properties);
    m_cp_codec_infer = cp_codec_compiled.create_infer_request();
    
    // Load speech decoder weights from speech_tokenizer subdirectory
    std::filesystem::path tokenizer_path = models_path / "speech_tokenizer";
    auto st_decoder_data = ::ov::genai::safetensors::load_safetensors(tokenizer_path);
    SpeechDecoderWeightSource decoder_weight_source(std::move(st_decoder_data));
    ::ov::genai::safetensors::SafetensorsWeightFinalizer decoder_finalizer;
    
    // Create and compile Speech Decoder model
    auto decoder_model = create_qwen3_tts_speech_decoder_model(decoder_cfg, decoder_weight_source, decoder_finalizer);
    auto decoder_compiled = core.compile_model(decoder_model, device, properties);
    m_decoder_infer = decoder_compiled.create_infer_request();
    
    // Pre-compute tts_pad embedding for decode phase
    size_t hidden_size = static_cast<size_t>(m_hidden_size);
    std::vector<int64_t> tts_pad_vec = {m_tts_pad_token_id};
    std::vector<int64_t> zero_codec = {0};
    std::vector<float> no_codec_mask = {0.0f};
    
    ov::Tensor tts_pad_text(ov::element::i64, {1, 1}, tts_pad_vec.data());
    ov::Tensor tts_pad_codec(ov::element::i64, {1, 1}, zero_codec.data());
    ov::Tensor tts_pad_mask(ov::element::f32, {1, 1}, no_codec_mask.data());
    
    m_embed_infer.set_tensor("text_input_ids", tts_pad_text);
    m_embed_infer.set_tensor("codec_input_ids", tts_pad_codec);
    m_embed_infer.set_tensor("codec_mask", tts_pad_mask);
    m_embed_infer.infer();
    
    auto tts_pad_embed_tensor = m_embed_infer.get_tensor("inputs_embeds");
    m_tts_pad_embed.resize(hidden_size);
    std::copy(tts_pad_embed_tensor.data<float>(),
              tts_pad_embed_tensor.data<float>() + hidden_size,
              m_tts_pad_embed.begin());
}

Qwen3TTSResult Qwen3TTSPipeline::generate(const std::string& text,
                                          const Qwen3TTSGenerationConfig& config) {
    Qwen3TTSResult result;
    result.sample_rate = m_sample_rate;
    
    // Set random seed
    m_rng.seed(config.seed);
    
    // Tokenize text
    auto tokenized = m_tokenizer->encode(text, ov::AnyMap{{"add_special_tokens", false}});
    auto input_ids_tensor = tokenized.input_ids;
    size_t text_len = input_ids_tensor.get_shape()[1];
    const int64_t* ids = input_ids_tensor.data<int64_t>();
    std::vector<int64_t> text_token_ids(ids, ids + text_len);
    
    // Generate codec tokens
    auto gen_start = std::chrono::high_resolution_clock::now();
    auto all_layer_tokens = generate_codec_tokens(text_token_ids, config);
    auto gen_end = std::chrono::high_resolution_clock::now();
    result.generation_time_ms = std::chrono::duration<float, std::milli>(gen_end - gen_start).count();
    result.num_frames = static_cast<int64_t>(all_layer_tokens[0].size());
    
    // Decode to audio
    auto decode_start = std::chrono::high_resolution_clock::now();
    result.audio = decode_to_audio(all_layer_tokens);
    auto decode_end = std::chrono::high_resolution_clock::now();
    result.decode_time_ms = std::chrono::duration<float, std::milli>(decode_end - decode_start).count();
    
    return result;
}

std::vector<std::vector<int64_t>> Qwen3TTSPipeline::generate_codec_tokens(
    const std::vector<int64_t>& text_token_ids,
    const Qwen3TTSGenerationConfig& config) {
    
    const size_t batch_size = 1;
    const size_t hidden_size = static_cast<size_t>(m_hidden_size);
    const size_t num_layers = static_cast<size_t>(m_num_layers);
    const size_t num_kv_heads = static_cast<size_t>(m_num_kv_heads);
    const size_t head_dim = static_cast<size_t>(m_head_dim);
    const size_t vocab_size = static_cast<size_t>(m_vocab_size);
    const size_t cp_vocab_size = static_cast<size_t>(m_cp_vocab_size);
    
    std::vector<std::vector<int64_t>> all_layer_tokens(16);
    
    // Build input sequence following official format (non_streaming_mode)
    std::vector<int64_t> full_text_ids;
    std::vector<int64_t> full_codec_ids;
    std::vector<float> codec_mask;
    
    // 1. Role tokens (codec_id=0 with mask=0)
    for (auto id : m_role_tokens) {
        full_text_ids.push_back(id);
        full_codec_ids.push_back(0);
        codec_mask.push_back(0.0f);
    }
    
    // 2. Codec prefix with tts_pad
    std::vector<int64_t> codec_prefix = {m_codec_nothink_id, m_codec_think_bos_id, m_codec_think_eos_id};
    for (size_t i = 0; i < 3; ++i) {
        full_text_ids.push_back(m_tts_pad_token_id);
        full_codec_ids.push_back(codec_prefix[i]);
        codec_mask.push_back(1.0f);
    }
    
    // 3. Text tokens + codec_pad
    for (auto id : text_token_ids) {
        full_text_ids.push_back(id);
        full_codec_ids.push_back(m_codec_pad_id);
        codec_mask.push_back(1.0f);
    }
    
    // 4. tts_eos + codec_pad
    full_text_ids.push_back(m_tts_eos_token_id);
    full_codec_ids.push_back(m_codec_pad_id);
    codec_mask.push_back(1.0f);
    
    // 5. tts_pad + codec_bos
    full_text_ids.push_back(m_tts_pad_token_id);
    full_codec_ids.push_back(m_codec_bos_id);
    codec_mask.push_back(1.0f);
    
    size_t prefill_len = full_text_ids.size();
    
    // Get embeddings
    ov::Tensor text_tensor(ov::element::i64, {batch_size, prefill_len}, full_text_ids.data());
    ov::Tensor codec_tensor(ov::element::i64, {batch_size, prefill_len}, full_codec_ids.data());
    ov::Tensor mask_tensor(ov::element::f32, {batch_size, prefill_len}, codec_mask.data());
    
    m_embed_infer.set_tensor("text_input_ids", text_tensor);
    m_embed_infer.set_tensor("codec_input_ids", codec_tensor);
    m_embed_infer.set_tensor("codec_mask", mask_tensor);
    m_embed_infer.infer();
    auto embed_output = m_embed_infer.get_tensor("inputs_embeds");
    
    // Create position IDs
    auto pos_data = create_mrope_positions(0, prefill_len, batch_size);
    ov::Tensor position_ids(ov::element::i64, {3, batch_size, prefill_len}, pos_data.data());
    
    // Create causal attention_mask: [batch, 1, seq_len, seq_len] float mask
    // 0.0 = attend, -inf = mask (matches working version)
    ov::Tensor attention_mask = create_causal_mask(prefill_len, batch_size);
    
    // Create empty past KV caches (seq_len = 0) for prefill
    std::vector<ov::Tensor> past_keys, past_values;
    for (size_t i = 0; i < num_layers; ++i) {
        past_keys.push_back(ov::Tensor(ov::element::f32, {batch_size, num_kv_heads, 0, head_dim}));
        past_values.push_back(ov::Tensor(ov::element::f32, {batch_size, num_kv_heads, 0, head_dim}));
    }
    
    // Run prefill with past KV cache inputs (even though seq_len=0)
    m_talker_prefill_infer.set_tensor("inputs_embeds", embed_output);
    m_talker_prefill_infer.set_tensor("position_ids", position_ids);
    m_talker_prefill_infer.set_tensor("attention_mask", attention_mask);
    for (size_t i = 0; i < num_layers; ++i) {
        m_talker_prefill_infer.set_tensor("past_key_" + std::to_string(i), past_keys[i]);
        m_talker_prefill_infer.set_tensor("past_value_" + std::to_string(i), past_values[i]);
    }
    m_talker_prefill_infer.infer();
    
    auto logits_tensor = m_talker_prefill_infer.get_tensor("logits");
    auto hidden_tensor = m_talker_prefill_infer.get_tensor("hidden_states");
    
    // Get KV caches from prefill (output names are present_key_*, present_value_*)
    std::vector<ov::Tensor> present_keys, present_values;
    for (size_t i = 0; i < num_layers; ++i) {
        present_keys.push_back(m_talker_prefill_infer.get_tensor("present_key_" + std::to_string(i)));
        present_values.push_back(m_talker_prefill_infer.get_tensor("present_value_" + std::to_string(i)));
    }
    
    // Build suppress_tokens list (suppress [vocab_size - 1024, vocab_size) except EOS)
    std::vector<int64_t> suppress_tokens;
    int64_t suppress_start = static_cast<int64_t>(vocab_size) - 1024;
    for (int64_t i = suppress_start; i < static_cast<int64_t>(vocab_size); ++i) {
        if (i != m_codec_eos_id) {
            suppress_tokens.push_back(i);
        }
    }
    std::vector<int64_t> suppress_tokens_with_eos = suppress_tokens;
    suppress_tokens_with_eos.push_back(m_codec_eos_id);
    
    // Sample first token (suppress EOS for min_new_tokens)
    const float* logits_data = logits_tensor.data<float>();
    int64_t layer0_token = sample_token(logits_data + (prefill_len - 1) * vocab_size, vocab_size,
                                        config.temperature, config.top_k, config.top_p, config.repetition_penalty,
                                        &all_layer_tokens[0], &suppress_tokens_with_eos);
    
    // Get past_hidden
    const float* hidden_data = hidden_tensor.data<float>();
    std::vector<float> past_hidden(hidden_size);
    std::copy(hidden_data + (prefill_len - 1) * hidden_size,
              hidden_data + prefill_len * hidden_size, past_hidden.begin());
    
    // Get layer0 embedding
    std::vector<int64_t> layer0_vec = {layer0_token};
    ov::Tensor layer0_tensor(ov::element::i64, {batch_size, 1}, layer0_vec.data());
    m_talker_codec_infer.set_tensor("codec_input_ids", layer0_tensor);
    m_talker_codec_infer.infer();
    auto layer0_embed_out = m_talker_codec_infer.get_tensor("codec_embeds");
    std::vector<float> layer0_embed(hidden_size);
    std::copy(layer0_embed_out.data<float>(),
              layer0_embed_out.data<float>() + hidden_size,
              layer0_embed.begin());
    
    all_layer_tokens[0].push_back(layer0_token);
    
    // AR Code Predictor for Frame 0
    std::vector<float> ar_sequence;
    ar_sequence.insert(ar_sequence.end(), past_hidden.begin(), past_hidden.end());
    ar_sequence.insert(ar_sequence.end(), layer0_embed.begin(), layer0_embed.end());
    
    std::vector<int64_t> current_layer_tokens(15);
    
    // Generate layers 1-15 for Frame 0
    for (int step = 0; step < 15; ++step) {
        size_t cur_len = ar_sequence.size() / hidden_size;
        
        std::vector<int64_t> pos_ids(cur_len);
        for (size_t i = 0; i < cur_len; ++i) {
            pos_ids[i] = static_cast<int64_t>(i);
        }
        
        ov::Tensor ar_input(ov::element::f32, {batch_size, cur_len, hidden_size}, ar_sequence.data());
        ov::Tensor ar_pos(ov::element::i64, {batch_size, cur_len}, pos_ids.data());
        
        m_cp_ar_infer[step].set_tensor("inputs_embeds", ar_input);
        m_cp_ar_infer[step].set_tensor("position_ids", ar_pos);
        m_cp_ar_infer[step].infer();
        
        auto step_logits = m_cp_ar_infer[step].get_tensor("logits");
        
        int64_t layer_token = sample_token(step_logits.data<float>(), cp_vocab_size,
                                           config.temperature, config.top_k, config.top_p, 1.0f,
                                           nullptr, nullptr);
        all_layer_tokens[step + 1].push_back(layer_token);
        current_layer_tokens[step] = layer_token;
        
        // Get embedding for this token
        std::vector<int64_t> token_vec = {layer_token};
        ov::Tensor token_tensor(ov::element::i64, {batch_size, 1}, token_vec.data());
        m_cp_embed_infer[step].set_tensor("codec_input", token_tensor);
        m_cp_embed_infer[step].infer();
        auto layer_embed_out = m_cp_embed_infer[step].get_tensor("codec_embed");
        
        const float* embed_ptr = layer_embed_out.data<float>();
        ar_sequence.insert(ar_sequence.end(), embed_ptr, embed_ptr + hidden_size);
    }
    
    // Frame 0 Talker decode
    {
        std::vector<float> codec_sum(hidden_size, 0.0f);
        for (size_t i = 0; i < hidden_size; ++i) {
            codec_sum[i] += layer0_embed[i];
        }
        
        std::vector<std::vector<int64_t>> layer_tokens(15);
        std::vector<ov::Tensor> layer_tensors(15);
        for (int layer = 0; layer < 15; ++layer) {
            layer_tokens[layer] = {current_layer_tokens[layer]};
            layer_tensors[layer] = ov::Tensor(ov::element::i64, {batch_size, 1}, layer_tokens[layer].data());
            m_cp_codec_infer.set_tensor("codec_input_" + std::to_string(layer), layer_tensors[layer]);
        }
        m_cp_codec_infer.infer();
        
        // The model outputs the sum of all 15 codec embeddings
        auto codec_embeds_sum = m_cp_codec_infer.get_tensor("codec_embeds_sum");
        const float* sum_ptr = codec_embeds_sum.data<float>();
        
        for (size_t i = 0; i < hidden_size; ++i) {
            codec_sum[i] += sum_ptr[i];
        }
        
        std::vector<float> step_embed_data(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            step_embed_data[i] = codec_sum[i] + m_tts_pad_embed[i];
        }
        ov::Tensor step_embed(ov::element::f32, {batch_size, 1, hidden_size}, step_embed_data.data());
        
        auto step_pos = create_mrope_positions(prefill_len, 1, batch_size);
        ov::Tensor step_positions(ov::element::i64, {3, batch_size, 1}, step_pos.data());
        
        // Create attention_mask for decode: [batch, 1, 1, total_len] all 0.0 (attend to all)
        size_t decode_kv_len = prefill_len + 1;  // Frame 0 decode
        ov::Tensor decode_attention_mask = create_decode_mask(prefill_len, batch_size);
        
        m_talker_decode_infer.set_tensor("inputs_embeds", step_embed);
        m_talker_decode_infer.set_tensor("position_ids", step_positions);
        m_talker_decode_infer.set_tensor("attention_mask", decode_attention_mask);
        for (size_t i = 0; i < num_layers; ++i) {
            m_talker_decode_infer.set_tensor("past_key_" + std::to_string(i), present_keys[i]);
            m_talker_decode_infer.set_tensor("past_value_" + std::to_string(i), present_values[i]);
        }
        
        m_talker_decode_infer.infer();
        
        logits_tensor = m_talker_decode_infer.get_tensor("logits");
        hidden_tensor = m_talker_decode_infer.get_tensor("hidden_states");
        
        for (size_t i = 0; i < num_layers; ++i) {
            present_keys[i] = m_talker_decode_infer.get_tensor("present_key_" + std::to_string(i));
            present_values[i] = m_talker_decode_infer.get_tensor("present_value_" + std::to_string(i));
        }
        
        logits_data = logits_tensor.data<float>();
        
        layer0_token = sample_token(logits_data, vocab_size,
                                    config.temperature, config.top_k, config.top_p, config.repetition_penalty,
                                    &all_layer_tokens[0], &suppress_tokens_with_eos);
        all_layer_tokens[0].push_back(layer0_token);
        
        std::cout << "Frame 0: layer0_token=" << layer0_token 
                  << (layer0_token == m_codec_eos_id ? " (EOS)" : "") << std::endl;
        
        hidden_data = hidden_tensor.data<float>();
        std::copy(hidden_data, hidden_data + hidden_size, past_hidden.begin());
    }
    
    // Decode phase
    size_t current_seq_len = prefill_len + 1;
    
    // Determine effective max frames
    int effective_max = config.max_new_tokens;
    if (config.force_stop_after_frames > 0) {
        effective_max = std::min(effective_max, config.force_stop_after_frames);
        std::cout << "[INFO] Using force_stop_after_frames=" << config.force_stop_after_frames << std::endl;
    }
    
    for (int frame = 1; frame < effective_max && layer0_token != m_codec_eos_id; ++frame) {
/*         if (frame % 10 == 0 || frame < 5) {
            std::cout << "Frame " << frame << ": layer0_token=" << layer0_token << std::endl;
        } */
        // Get layer0 embedding
        layer0_vec[0] = layer0_token;
        ov::Tensor layer0_tensor_dec(ov::element::i64, {batch_size, 1}, layer0_vec.data());
        m_talker_codec_infer.set_tensor("codec_input_ids", layer0_tensor_dec);
        m_talker_codec_infer.infer();
        auto layer0_embed_out_dec = m_talker_codec_infer.get_tensor("codec_embeds");
        std::copy(layer0_embed_out_dec.data<float>(),
                  layer0_embed_out_dec.data<float>() + hidden_size,
                  layer0_embed.begin());
        
        // Reset AR sequence
        ar_sequence.clear();
        ar_sequence.insert(ar_sequence.end(), past_hidden.begin(), past_hidden.end());
        ar_sequence.insert(ar_sequence.end(), layer0_embed.begin(), layer0_embed.end());
        
        // Generate layers 1-15
        for (int step = 0; step < 15; ++step) {
            size_t cur_len = ar_sequence.size() / hidden_size;
            
            std::vector<int64_t> pos_ids(cur_len);
            for (size_t i = 0; i < cur_len; ++i) {
                pos_ids[i] = static_cast<int64_t>(i);
            }
            
            ov::Tensor ar_input(ov::element::f32, {batch_size, cur_len, hidden_size}, ar_sequence.data());
            ov::Tensor ar_pos(ov::element::i64, {batch_size, cur_len}, pos_ids.data());
            
            m_cp_ar_infer[step].set_tensor("inputs_embeds", ar_input);
            m_cp_ar_infer[step].set_tensor("position_ids", ar_pos);
            m_cp_ar_infer[step].infer();
            
            auto step_logits = m_cp_ar_infer[step].get_tensor("logits");
            int64_t layer_token = sample_token(step_logits.data<float>(), cp_vocab_size,
                                               config.temperature, config.top_k, config.top_p, 1.0f,
                                               nullptr, nullptr);
            all_layer_tokens[step + 1].push_back(layer_token);
            current_layer_tokens[step] = layer_token;
            
            std::vector<int64_t> token_vec = {layer_token};
            ov::Tensor token_tensor(ov::element::i64, {batch_size, 1}, token_vec.data());
            m_cp_embed_infer[step].set_tensor("codec_input", token_tensor);
            m_cp_embed_infer[step].infer();
            auto layer_embed_out = m_cp_embed_infer[step].get_tensor("codec_embed");
            
            const float* embed_ptr = layer_embed_out.data<float>();
            ar_sequence.insert(ar_sequence.end(), embed_ptr, embed_ptr + hidden_size);
        }
        
        // Compute inputs_embeds
        std::vector<float> codec_sum(hidden_size, 0.0f);
        for (size_t i = 0; i < hidden_size; ++i) {
            codec_sum[i] += layer0_embed[i];
        }
        
        std::vector<std::vector<int64_t>> layer_tokens(15);
        std::vector<ov::Tensor> layer_tensors(15);
        for (int layer = 0; layer < 15; ++layer) {
            layer_tokens[layer] = {current_layer_tokens[layer]};
            layer_tensors[layer] = ov::Tensor(ov::element::i64, {batch_size, 1}, layer_tokens[layer].data());
            m_cp_codec_infer.set_tensor("codec_input_" + std::to_string(layer), layer_tensors[layer]);
        }
        m_cp_codec_infer.infer();
        
        // The model outputs the sum of all 15 codec embeddings
        auto codec_embeds_sum = m_cp_codec_infer.get_tensor("codec_embeds_sum");
        const float* sum_ptr = codec_embeds_sum.data<float>();
        for (size_t i = 0; i < hidden_size; ++i) {
            codec_sum[i] += sum_ptr[i];
        }
        
        std::vector<float> step_embed_data(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            step_embed_data[i] = codec_sum[i] + m_tts_pad_embed[i];
        }
        ov::Tensor step_embed(ov::element::f32, {batch_size, 1, hidden_size}, step_embed_data.data());
        
        // Talker decode
        auto step_pos = create_mrope_positions(current_seq_len, 1, batch_size);
        ov::Tensor step_positions(ov::element::i64, {3, batch_size, 1}, step_pos.data());
        
        // Create attention_mask for decode: [batch, 1, 1, total_len] all 0.0 (attend to all)
        ov::Tensor loop_decode_attention_mask = create_decode_mask(current_seq_len, batch_size);
        
        m_talker_decode_infer.set_tensor("inputs_embeds", step_embed);
        m_talker_decode_infer.set_tensor("position_ids", step_positions);
        m_talker_decode_infer.set_tensor("attention_mask", loop_decode_attention_mask);
        for (size_t i = 0; i < num_layers; ++i) {
            m_talker_decode_infer.set_tensor("past_key_" + std::to_string(i), present_keys[i]);
            m_talker_decode_infer.set_tensor("past_value_" + std::to_string(i), present_values[i]);
        }
        
        m_talker_decode_infer.infer();
        
        logits_tensor = m_talker_decode_infer.get_tensor("logits");
        hidden_tensor = m_talker_decode_infer.get_tensor("hidden_states");
        
        for (size_t i = 0; i < num_layers; ++i) {
            present_keys[i] = m_talker_decode_infer.get_tensor("present_key_" + std::to_string(i));
            present_values[i] = m_talker_decode_infer.get_tensor("present_value_" + std::to_string(i));
        }
        
        logits_data = logits_tensor.data<float>();
        
        const std::vector<int64_t>* suppress = (frame < config.min_new_tokens) ? &suppress_tokens_with_eos : &suppress_tokens;
        layer0_token = sample_token(logits_data, vocab_size,
                                    config.temperature, config.top_k, config.top_p, config.repetition_penalty,
                                    &all_layer_tokens[0], suppress);
        all_layer_tokens[0].push_back(layer0_token);
        
/*         if (frame % 10 == 0 || layer0_token == m_codec_eos_id) {
            std::cout << "Frame " << frame << " end: layer0_token=" << layer0_token 
                      << (layer0_token == m_codec_eos_id ? " (EOS!)" : "") << std::endl;
        } */
        
        hidden_data = hidden_tensor.data<float>();
        std::copy(hidden_data, hidden_data + hidden_size, past_hidden.begin());
        
        current_seq_len++;
    }
    
    std::cout << "Generation finished. Total frames: " << all_layer_tokens[0].size() << std::endl;
    
    return all_layer_tokens;
}

ov::Tensor Qwen3TTSPipeline::decode_to_audio(const std::vector<std::vector<int64_t>>& all_layer_tokens) {
    int64_t seq_len = static_cast<int64_t>(all_layer_tokens[0].size());
    
    std::vector<int64_t> codes_flat(16 * seq_len);
    for (int layer = 0; layer < 16; ++layer) {
        for (int64_t t = 0; t < seq_len; ++t) {
            codes_flat[layer * seq_len + t] = all_layer_tokens[layer][t];
        }
    }
    
    ov::Tensor codes_tensor(ov::element::i64, {1, 16, static_cast<size_t>(seq_len)}, codes_flat.data());
    
    m_decoder_infer.set_tensor("codes", codes_tensor);
    m_decoder_infer.infer();
    
    return m_decoder_infer.get_tensor("audio");
}

ov::Tensor Qwen3TTSPipeline::create_causal_mask(size_t seq_len, size_t batch_size) {
    ov::Tensor mask(ov::element::f32, {batch_size, 1, seq_len, seq_len});
    float* data = mask.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                float val = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
                data[b * seq_len * seq_len + i * seq_len + j] = val;
            }
        }
    }
    return mask;
}

ov::Tensor Qwen3TTSPipeline::create_decode_mask(size_t past_len, size_t batch_size) {
    size_t total_len = past_len + 1;
    ov::Tensor mask(ov::element::f32, {batch_size, 1, 1, total_len});
    float* data = mask.data<float>();
    
    for (size_t i = 0; i < batch_size * total_len; ++i) {
        data[i] = 0.0f;
    }
    return mask;
}

std::vector<int64_t> Qwen3TTSPipeline::create_mrope_positions(size_t start, size_t len, size_t batch) {
    std::vector<int64_t> pos(3 * batch * len);
    for (size_t i = 0; i < len; ++i) {
        pos[0 * batch * len + i] = static_cast<int64_t>(start + i);
        pos[1 * batch * len + i] = static_cast<int64_t>(start + i);
        pos[2 * batch * len + i] = static_cast<int64_t>(start + i);
    }
    return pos;
}

int64_t Qwen3TTSPipeline::sample_token(const float* logits, size_t vocab_size,
                                       float temperature, size_t top_k, float top_p,
                                       float rep_penalty,
                                       const std::vector<int64_t>* history,
                                       const std::vector<int64_t>* suppress_tokens) {
    std::vector<float> adj_logits(logits, logits + vocab_size);
    
    // Apply repetition penalty
    if (history && rep_penalty > 1.0f) {
        for (int64_t token : *history) {
            if (token >= 0 && static_cast<size_t>(token) < vocab_size) {
                if (adj_logits[token] > 0) {
                    adj_logits[token] /= rep_penalty;
                } else {
                    adj_logits[token] *= rep_penalty;
                }
            }
        }
    }
    
    // Suppress tokens
    if (suppress_tokens) {
        for (int64_t token : *suppress_tokens) {
            if (token >= 0 && static_cast<size_t>(token) < vocab_size) {
                adj_logits[token] = -1e9f;
            }
        }
    }
    
    // Sort for top-k
    std::vector<size_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&adj_logits](size_t a, size_t b) { return adj_logits[a] > adj_logits[b]; });
    
    size_t effective_k = std::min(top_k, vocab_size);
    
    // Apply temperature and compute probabilities
    std::vector<float> probs(effective_k);
    float max_logit = adj_logits[indices[0]];
    
    float sum = 0.0f;
    for (size_t i = 0; i < effective_k; ++i) {
        probs[i] = std::exp((adj_logits[indices[i]] - max_logit) / temperature);
        sum += probs[i];
    }
    
    for (size_t i = 0; i < effective_k; ++i) {
        probs[i] /= sum;
    }
    
    // Apply top-p
    float cumsum = 0.0f;
    size_t cutoff = effective_k;
    for (size_t i = 0; i < effective_k; ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    
    float top_p_sum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        top_p_sum += probs[i];
    }
    
    // Sample
    std::uniform_real_distribution<float> dist(0.0f, top_p_sum);
    float r = dist(m_rng);
    
    cumsum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) {
            return static_cast<int64_t>(indices[i]);
        }
    }
    
    return static_cast<int64_t>(indices[0]);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
