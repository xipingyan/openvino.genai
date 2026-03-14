// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_code_predictor.hpp"
#include "modeling/models/qwen3_tts/modeling_qwen3_tts_speech_decoder.hpp"
#include "modeling/models/qwen3_tts/modeling_qwen3_tts_talker.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/weights/weight_loader.hpp"

#include <unordered_map>

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::Tensor make_zero_tensor(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
    return tensor;
}

ov::Tensor make_linear_identity_weight(size_t out_dim, size_t in_dim, float scale = 1.0f) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{out_dim, in_dim});
    auto* data = tensor.data<float>();
    std::fill_n(data, tensor.get_size(), 0.0f);
    const size_t diag = std::min(out_dim, in_dim);
    for (size_t i = 0; i < diag; ++i) {
        data[i * in_dim + i] = scale;
    }
    return tensor;
}

ov::Tensor make_conv1d_identity_weight(size_t out_ch, size_t in_ch, size_t kernel, float scale = 1.0f) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{out_ch, in_ch, kernel});
    auto* data = tensor.data<float>();
    std::fill_n(data, tensor.get_size(), 0.0f);
    const size_t center = kernel / 2;
    const size_t diag = std::min(out_ch, in_ch);
    for (size_t i = 0; i < diag; ++i) {
        data[(i * in_ch + i) * kernel + center] = scale;
    }
    return tensor;
}

class OmniCode2WavMappedWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    OmniCode2WavMappedWeightSource(ov::genai::modeling::weights::WeightSource& source,
                                   const ov::genai::modeling::models::SpeechDecoderConfig& cfg)
        : source_(source) {
        const size_t dim = static_cast<size_t>(cfg.latent_dim);
        constexpr float mean_scale = 1.0f / 16.0f;

        // Synthetic identity weights — keys use model-parameter naming (with "decoder." prefix)
        synthetic_["decoder.quantizer.rvq_first.output_proj.weight"] =
            make_conv1d_identity_weight(dim, dim, 1, mean_scale);
        synthetic_["decoder.quantizer.rvq_rest.output_proj.weight"] =
            make_conv1d_identity_weight(dim, dim, 1, mean_scale);

        synthetic_["decoder.pre_conv.conv.weight"] = make_conv1d_identity_weight(dim, dim, 5, 1.0f);
        synthetic_["decoder.pre_conv.conv.bias"] = make_zero_tensor(ov::Shape{dim});

        synthetic_["decoder.pre_transformer.input_proj.weight"] = make_linear_identity_weight(dim, dim, 1.0f);
        synthetic_["decoder.pre_transformer.input_proj.bias"] = make_zero_tensor(ov::Shape{dim});
        synthetic_["decoder.pre_transformer.output_proj.weight"] = make_linear_identity_weight(dim, dim, 1.0f);
        synthetic_["decoder.pre_transformer.output_proj.bias"] = make_zero_tensor(ov::Shape{dim});
    }

    // Return keys using model-parameter naming so load_model() can match them.
    // Source keys get "decoder." prefix prepended, and layers.N → layers[N].
    std::vector<std::string> keys() const override {
        auto src_keys = source_.keys();
        std::vector<std::string> names;
        names.reserve(src_keys.size() + 28);
        for (const auto& k : src_keys) {
            names.push_back(to_model_name(k));
        }
        // Codebook entries (model parameter names)
        names.push_back("decoder.quantizer.rvq_first.vq.layers[0]._codebook.embed");
        for (int i = 0; i < 15; ++i) {
            names.push_back("decoder.quantizer.rvq_rest.vq.layers[" + std::to_string(i) + "]._codebook.embed");
        }
        // Synthetic entries
        for (const auto& kv : synthetic_) {
            names.push_back(kv.first);
        }
        return names;
    }

    bool has(const std::string& name) const override {
        if (synthetic_.find(name) != synthetic_.end()) {
            return true;
        }
        if (is_quantizer_codebook(name) && source_.has("code_embedding.weight")) {
            return true;
        }
        return source_.has(to_source_name(name));
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = synthetic_.find(name);
        if (it != synthetic_.end()) {
            return it->second;
        }
        if (is_quantizer_codebook(name) && source_.has("code_embedding.weight")) {
            return source_.get_tensor("code_embedding.weight");
        }
        std::string src = to_source_name(name);
        OPENVINO_ASSERT(source_.has(src), "Missing mapped code2wav tensor: ", name, " (source: ", src, ")");
        return source_.get_tensor(src);
    }

    void release_tensor(const std::string& name) override {
        if (synthetic_.find(name) != synthetic_.end()) {
            return;
        }
        if (is_quantizer_codebook(name) && source_.has("code_embedding.weight")) {
            source_.release_tensor("code_embedding.weight");
            return;
        }
        std::string src = to_source_name(name);
        if (source_.has(src)) {
            source_.release_tensor(src);
        }
    }

private:
    // Model param name → source key name.
    // Strip "decoder." prefix added by SpeechDecoderModel module name,
    // and convert brackets to dots: layers[0] → layers.0
    static std::string to_source_name(const std::string& model_name) {
        std::string s = model_name;
        // Strip "decoder." prefix
        if (s.rfind("decoder.", 0) == 0) {
            s = s.substr(8);
        }
        // Convert brackets to dots: layers[0] → layers.0
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '[') {
                out.push_back('.');
            } else if (s[i] == ']') {
                // skip closing bracket
            } else {
                out.push_back(s[i]);
            }
        }
        return out;
    }

    // Source key name → model param name.
    // Prepend "decoder." and convert layers.N → layers[N]
    static std::string to_model_name(const std::string& src_name) {
        std::string s = "decoder." + src_name;
        // Convert "layers." followed by digits then "." → "layers[digits]."
        std::string result;
        result.reserve(s.size() + 8);
        size_t i = 0;
        while (i < s.size()) {
            if (i + 7 <= s.size() && s.substr(i, 7) == "layers.") {
                result += "layers[";
                i += 7;
                // Copy digits
                while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                    result += s[i++];
                }
                result += ']';
                // The next char (dot or end) is preserved naturally
            } else {
                result += s[i++];
            }
        }
        return result;
    }

    static bool is_quantizer_codebook(const std::string& name) {
        return name.find("decoder.quantizer.rvq_first.vq.layers[0]._codebook.embed") == 0 ||
               name.find("decoder.quantizer.rvq_rest.vq.layers[") == 0;
    }

    ov::genai::modeling::weights::WeightSource& source_;
    std::unordered_map<std::string, ov::Tensor> synthetic_;
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

std::shared_ptr<ov::Model> create_qwen3_omni_talker_embedding_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto talker_cfg = to_qwen3_omni_talker_config(cfg);

    VocabEmbedding text_embedding(ctx, "talker.model.text_embedding", nullptr);
    VocabEmbedding codec_embedding(ctx, "talker.model.codec_embedding", nullptr);
    Qwen3TTSTextProjection text_projection(ctx, "talker.text_projection", talker_cfg, nullptr);

    auto lenient = ov::genai::modeling::weights::LoadOptions::lenient();
    ov::genai::modeling::weights::load_model(text_embedding, source, finalizer, lenient);
    ov::genai::modeling::weights::load_model(codec_embedding, source, finalizer, lenient);
    ov::genai::modeling::weights::load_model(text_projection, source, finalizer, lenient);

    auto text_input_ids = ctx.parameter("text_input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto codec_input_ids = ctx.parameter("codec_input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto codec_mask = ctx.parameter("codec_mask", ov::element::f32, ov::PartialShape{-1, -1});

    auto text_embed = text_embedding.forward(text_input_ids);
    auto text_projected = text_projection.forward(text_embed);
    auto codec_embed = codec_embedding.forward(codec_input_ids);
    auto mask_expanded = codec_mask.unsqueeze({2});
    auto masked_codec_embed = codec_embed * mask_expanded;
    auto inputs_embeds = text_projected + masked_codec_embed;

    auto result = std::make_shared<ov::op::v0::Result>(inputs_embeds.output());
    set_name(result, "inputs_embeds");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_omni_talker_codec_embedding_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    VocabEmbedding codec_embedding(ctx, "talker.model.codec_embedding", nullptr);
    ov::genai::modeling::weights::load_model(codec_embedding,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    auto codec_input_ids = ctx.parameter("codec_input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto codec_embed = codec_embedding.forward(codec_input_ids);

    auto result = std::make_shared<ov::op::v0::Result>(codec_embed.output());
    set_name(result, "codec_embeds");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_omni_talker_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto talker_cfg = to_qwen3_omni_talker_config(cfg);
    Qwen3TTSTalkerForConditionalGeneration model(ctx, talker_cfg);
    ov::genai::modeling::weights::load_model(model,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, talker_cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{3, -1, -1});

    auto outputs = model.forward_no_cache(inputs_embeds, position_ids);

    auto logits_result = std::make_shared<ov::op::v0::Result>(outputs.first.output());
    set_name(logits_result, "logits");
    auto hidden_result = std::make_shared<ov::op::v0::Result>(outputs.second.output());
    set_name(hidden_result, "hidden_states");

    return ctx.build_model({logits_result->output(0), hidden_result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_omni_talker_prefill_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto talker_cfg = to_qwen3_omni_talker_config(cfg);
    Qwen3TTSTalkerForConditionalGeneration model(ctx, talker_cfg);
    ov::genai::modeling::weights::load_model(model,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, talker_cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{3, -1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::f32, ov::PartialShape{-1, 1, -1, -1});

    std::vector<Tensor> past_keys;
    std::vector<Tensor> past_values;
    for (int i = 0; i < talker_cfg.num_hidden_layers; ++i) {
        auto past_key = ctx.parameter("past_key_" + std::to_string(i),
                                      ov::element::f32,
                                      ov::PartialShape{-1, talker_cfg.num_key_value_heads, 0, talker_cfg.head_dim});
        auto past_value = ctx.parameter("past_value_" + std::to_string(i),
                                        ov::element::f32,
                                        ov::PartialShape{-1, talker_cfg.num_key_value_heads, 0, talker_cfg.head_dim});
        past_keys.push_back(past_key);
        past_values.push_back(past_value);
    }

    auto result = model.forward_with_cache(inputs_embeds, position_ids, attention_mask, past_keys, past_values);

    std::vector<ov::Output<ov::Node>> outputs;

    auto logits_result = std::make_shared<ov::op::v0::Result>(result.logits.output());
    set_name(logits_result, "logits");
    outputs.push_back(logits_result->output(0));

    auto hidden_result = std::make_shared<ov::op::v0::Result>(result.hidden_states.output());
    set_name(hidden_result, "hidden_states");
    outputs.push_back(hidden_result->output(0));

    for (size_t i = 0; i < result.key_caches.size(); ++i) {
        auto key_result = std::make_shared<ov::op::v0::Result>(result.key_caches[i].output());
        set_name(key_result, "present_key_" + std::to_string(i));
        outputs.push_back(key_result->output(0));

        auto value_result = std::make_shared<ov::op::v0::Result>(result.value_caches[i].output());
        set_name(value_result, "present_value_" + std::to_string(i));
        outputs.push_back(value_result->output(0));
    }

    return ctx.build_model(outputs);
}

std::shared_ptr<ov::Model> create_qwen3_omni_talker_decode_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto talker_cfg = to_qwen3_omni_talker_config(cfg);
    Qwen3TTSTalkerForConditionalGeneration model(ctx, talker_cfg);
    ov::genai::modeling::weights::load_model(model,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, 1, talker_cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{3, -1, 1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::f32, ov::PartialShape{-1, 1, 1, -1});

    std::vector<Tensor> past_keys;
    std::vector<Tensor> past_values;
    past_keys.reserve(static_cast<size_t>(talker_cfg.num_hidden_layers));
    past_values.reserve(static_cast<size_t>(talker_cfg.num_hidden_layers));

    for (int32_t i = 0; i < talker_cfg.num_hidden_layers; ++i) {
        auto past_key = ctx.parameter("past_key_" + std::to_string(i),
                                      ov::element::f32,
                                      ov::PartialShape{-1, talker_cfg.num_key_value_heads, -1, talker_cfg.head_dim});
        auto past_value = ctx.parameter("past_value_" + std::to_string(i),
                                        ov::element::f32,
                                        ov::PartialShape{-1, talker_cfg.num_key_value_heads, -1, talker_cfg.head_dim});
        past_keys.push_back(past_key);
        past_values.push_back(past_value);
    }

    auto result = model.forward_with_cache(inputs_embeds, position_ids, attention_mask, past_keys, past_values);

    std::vector<ov::Output<ov::Node>> outputs;

    auto logits_result = std::make_shared<ov::op::v0::Result>(result.logits.output());
    set_name(logits_result, "logits");
    outputs.push_back(logits_result->output(0));

    auto hidden_result = std::make_shared<ov::op::v0::Result>(result.hidden_states.output());
    set_name(hidden_result, "hidden_states");
    outputs.push_back(hidden_result->output(0));

    for (size_t i = 0; i < result.key_caches.size(); ++i) {
        auto key_result = std::make_shared<ov::op::v0::Result>(result.key_caches[i].output());
        set_name(key_result, "present_key_" + std::to_string(i));
        outputs.push_back(key_result->output(0));

        auto value_result = std::make_shared<ov::op::v0::Result>(result.value_caches[i].output());
        set_name(value_result, "present_value_" + std::to_string(i));
        outputs.push_back(value_result->output(0));
    }

    return ctx.build_model(outputs);
}

std::shared_ptr<ov::Model> create_qwen3_omni_code_predictor_ar_model(
    const Qwen3OmniConfig& cfg,
    int generation_step,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto cp_cfg = to_qwen3_omni_code_predictor_config(cfg);
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cp_cfg);
    ov::genai::modeling::weights::load_model(model,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    auto inputs_embeds =
        ctx.parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, cp_cfg.talker_hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto logits = model.forward_no_cache(inputs_embeds, position_ids, generation_step);
    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_omni_code_predictor_codec_embed_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto cp_cfg = to_qwen3_omni_code_predictor_config(cfg);
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cp_cfg);
    ov::genai::modeling::weights::load_model(model,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    std::vector<Tensor> codec_inputs;
    codec_inputs.reserve(15);
    for (int i = 0; i < 15; ++i) {
        auto input = ctx.parameter("codec_input_" + std::to_string(i), ov::element::i64, ov::PartialShape{-1, -1});
        codec_inputs.push_back(input);
    }

    auto embeds_sum = model.get_codec_embeds_sum(codec_inputs);
    auto result = std::make_shared<ov::op::v0::Result>(embeds_sum.output());
    set_name(result, "codec_embeds_sum");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_omni_code_predictor_single_codec_embed_model(
    const Qwen3OmniConfig& cfg,
    int codec_layer,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    auto cp_cfg = to_qwen3_omni_code_predictor_config(cfg);
    Qwen3TTSCodePredictorForConditionalGeneration model(ctx, cp_cfg);
    ov::genai::modeling::weights::load_model(model,
                                             source,
                                             finalizer,
                                             ov::genai::modeling::weights::LoadOptions::lenient());

    auto codec_input = ctx.parameter("codec_input", ov::element::i64, ov::PartialShape{-1, -1});
    auto codec_embed = model.get_codec_embed(codec_input, codec_layer);

    auto result = std::make_shared<ov::op::v0::Result>(codec_embed.output());
    set_name(result, "codec_embed");

    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_omni_speech_decoder_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    PrefixMappedWeightSource code2wav_source(source, "code2wav.");
    auto decoder_cfg = to_qwen3_omni_speech_decoder_config(cfg);
    OmniCode2WavMappedWeightSource mapped_source(code2wav_source, decoder_cfg);

    BuilderContext ctx;
    SpeechDecoderModel model(ctx, decoder_cfg);

    ov::genai::modeling::weights::LoadOptions opts;
    opts.allow_unmatched = true;
    opts.allow_missing = true;
    ov::genai::modeling::weights::load_model(model, mapped_source, finalizer, opts);

    auto codes = ctx.parameter("codes", ov::element::i64, ov::PartialShape{-1, decoder_cfg.num_quantizers, -1});

    std::vector<int64_t> offset_values;
    offset_values.reserve(static_cast<size_t>(decoder_cfg.num_quantizers));
    for (int32_t i = 0; i < decoder_cfg.num_quantizers; ++i) {
        offset_values.push_back(static_cast<int64_t>(i) * static_cast<int64_t>(decoder_cfg.codebook_size));
    }
    ov::Tensor offset_tensor(ov::element::i64, ov::Shape{1, static_cast<size_t>(decoder_cfg.num_quantizers), 1});
    std::copy(offset_values.begin(), offset_values.end(), offset_tensor.data<int64_t>());
    auto code_offset = ops::constant(offset_tensor, codes.context());
    auto offset_codes = codes + code_offset;

    auto audio = model.forward(offset_codes);

    auto result = std::make_shared<ov::op::v0::Result>(audio.output());
    set_name(result, "audio");
    ov::OutputVector outputs;
    outputs.push_back(result->output(0));
    return ctx.build_model(outputs);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
