// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>

#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_audio.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vl.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vision.hpp"

namespace {

std::string build_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits must have shape [1, S, V]");
    }

    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>() + offset;
        ov::float16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }

    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>() + offset;
        ov::bfloat16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }

    if (logits.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Unsupported logits dtype");
    }

    const auto* data = logits.data<const float>() + offset;
    float max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

ov::Tensor build_vision_attention_mask(const ov::Tensor& grid_thw) {
    if (grid_thw.get_element_type() != ov::element::i64 || grid_thw.get_shape().size() != 2 ||
        grid_thw.get_shape()[1] != 3) {
        throw std::runtime_error("grid_thw must be i64 tensor with shape [N,3]");
    }

    const auto* g = grid_thw.data<const int64_t>();
    const size_t rows = grid_thw.get_shape()[0];
    std::vector<size_t> segments;
    segments.reserve(rows * 2);
    size_t total_tokens = 0;
    for (size_t i = 0; i < rows; ++i) {
        const int64_t t = g[i * 3 + 0];
        const int64_t h = g[i * 3 + 1];
        const int64_t w = g[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            throw std::runtime_error("grid_thw contains non-positive values");
        }
        const size_t hw = static_cast<size_t>(h * w);
        for (int64_t f = 0; f < t; ++f) {
            segments.push_back(hw);
            total_tokens += hw;
        }
    }

    ov::Tensor mask(ov::element::f32, {1, 1, total_tokens, total_tokens});
    auto* data = mask.data<float>();
    std::fill_n(data, mask.get_size(), -std::numeric_limits<float>::infinity());

    size_t start = 0;
    for (size_t len : segments) {
        const size_t end = start + len;
        for (size_t r = start; r < end; ++r) {
            float* row = data + r * total_tokens;
            std::fill(row + start, row + end, 0.0f);
        }
        start = end;
    }

    return mask;
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "thinker.model.visual.pos_embed.weight",
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight",
    };
    for (const auto& name : candidates) {
        if (source.has(name)) {
            return name;
        }
    }

    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) {
            return name;
        }
    }
    throw std::runtime_error("Failed to locate visual pos_embed.weight in safetensors");
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string dtype_tag(const ov::element::Type& type) {
    if (type == ov::element::i64) {
        return "i64";
    }
    if (type == ov::element::i32) {
        return "i32";
    }
    if (type == ov::element::boolean) {
        return "bool";
    }
    if (type == ov::element::f32) {
        return "f32";
    }
    if (type == ov::element::f16) {
        return "f16";
    }
    if (type == ov::element::bf16) {
        return "bf16";
    }
    return type.to_string();
}

void dump_tensor_bin(const std::filesystem::path& dump_dir,
                     const std::string& name,
                     const ov::Tensor& tensor) {
    std::filesystem::create_directories(dump_dir);
    const auto bin_path = dump_dir / (name + ".bin");
    const auto meta_path = dump_dir / (name + ".meta");

    std::ofstream bout(bin_path, std::ios::binary);
    if (!bout.is_open()) {
        throw std::runtime_error("Failed to open dump file: " + bin_path.string());
    }
    bout.write(static_cast<const char*>(tensor.data()), static_cast<std::streamsize>(tensor.get_byte_size()));
    bout.close();

    std::ofstream mout(meta_path);
    if (!mout.is_open()) {
        throw std::runtime_error("Failed to open dump meta file: " + meta_path.string());
    }
    mout << "dtype=" << dtype_tag(tensor.get_element_type()) << "\n";
    mout << "shape=";
    const auto shape = tensor.get_shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        mout << shape[i];
        if (i + 1 < shape.size()) {
            mout << ",";
        }
    }
    mout << "\n";
    mout << "bytes=" << tensor.get_byte_size() << "\n";
}

bool has_dump_tensor(const std::filesystem::path& dump_dir, const std::string& name) {
    return std::filesystem::exists(dump_dir / (name + ".bin")) &&
           std::filesystem::exists(dump_dir / (name + ".meta"));
}

ov::Tensor load_dump_tensor(const std::filesystem::path& dump_dir, const std::string& name) {
    const auto meta_path = dump_dir / (name + ".meta");
    const auto bin_path = dump_dir / (name + ".bin");

    if (!std::filesystem::exists(meta_path) || !std::filesystem::exists(bin_path)) {
        throw std::runtime_error("Missing dump tensor files for: " + name);
    }

    std::ifstream min(meta_path);
    if (!min.is_open()) {
        throw std::runtime_error("Failed to open dump meta file: " + meta_path.string());
    }

    std::string dtype;
    ov::Shape shape;
    std::string line;
    while (std::getline(min, line)) {
        if (line.rfind("dtype=", 0) == 0) {
            dtype = line.substr(6);
        } else if (line.rfind("shape=", 0) == 0) {
            std::string s = line.substr(6);
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    shape.push_back(static_cast<size_t>(std::stoull(item)));
                }
            }
        }
    }

    ov::element::Type et;
    if (dtype == "i64") {
        et = ov::element::i64;
    } else if (dtype == "i32") {
        et = ov::element::i32;
    } else if (dtype == "bool") {
        et = ov::element::boolean;
    } else if (dtype == "f32") {
        et = ov::element::f32;
    } else if (dtype == "f16") {
        et = ov::element::f16;
    } else if (dtype == "bf16") {
        et = ov::element::bf16;
    } else {
        throw std::runtime_error("Unsupported dump dtype: " + dtype);
    }

    ov::Tensor tensor(et, shape);
    std::ifstream bin(bin_path, std::ios::binary);
    if (!bin.is_open()) {
        throw std::runtime_error("Failed to open dump bin file: " + bin_path.string());
    }
    bin.read(static_cast<char*>(tensor.data()), static_cast<std::streamsize>(tensor.get_byte_size()));
    if (!bin.good() && !bin.eof()) {
        throw std::runtime_error("Failed reading dump bin file: " + bin_path.string());
    }
    return tensor;
}

ov::Tensor load_image_with_qwen_omni_bridge(const std::filesystem::path& image_path) {
    nlohmann::json conversation = nlohmann::json::array({
        {
            {"role", "user"},
            {"content", nlohmann::json::array({
                {
                    {"type", "image"},
                    {"image", image_path.string()},
                },
            })},
        },
    });

    auto result = ov::genai::modeling::models::Qwen3OmniVisionProcess::process_vision_info_via_python(
        conversation,
        false);
    if (!result.is_array() || result.empty() || !result[0].is_array() || result[0].empty()) {
        throw std::runtime_error("Unexpected vision bridge output: no image payload");
    }

    const auto& image_node = result[0][0];
    if (!image_node.is_object()) {
        throw std::runtime_error("Unexpected vision bridge image payload type");
    }
    const std::string kind = image_node.value("kind", "");
    if (kind != "image" && kind != "ndarray") {
        throw std::runtime_error("Unexpected vision bridge image payload kind: " + kind);
    }

    if (!image_node.contains("shape") || !image_node.at("shape").is_array() || image_node.at("shape").size() != 3) {
        throw std::runtime_error("Vision bridge image payload missing shape=[H,W,C]");
    }

    const auto& shape = image_node.at("shape");
    const size_t h = static_cast<size_t>(shape[0].get<int64_t>());
    const size_t w = static_cast<size_t>(shape[1].get<int64_t>());
    const size_t c = static_cast<size_t>(shape[2].get<int64_t>());
    if (c != 3) {
        throw std::runtime_error("Vision bridge image channels must be 3");
    }

    const auto& data = image_node.at("data");
    if (!data.is_array() || data.size() != h) {
        throw std::runtime_error("Vision bridge image payload invalid data rows");
    }

    ov::Tensor image(ov::element::u8, ov::Shape{1, h, w, c});
    auto* dst = image.data<uint8_t>();
    for (size_t yy = 0; yy < h; ++yy) {
        const auto& row = data[yy];
        if (!row.is_array() || row.size() != w) {
            throw std::runtime_error("Vision bridge image payload invalid row width");
        }
        for (size_t xx = 0; xx < w; ++xx) {
            const auto& pix = row[xx];
            if (!pix.is_array() || pix.size() != c) {
                throw std::runtime_error("Vision bridge image payload invalid pixel channels");
            }
            const size_t base = (yy * w + xx) * c;
            for (size_t cc = 0; cc < c; ++cc) {
                dst[base + cc] = static_cast<uint8_t>(pix[cc].get<int32_t>());
            }
        }
    }
    return image;
}

template <typename T>
void flatten_json_values(const nlohmann::json& data, std::vector<T>& out) {
    if (data.is_array()) {
        for (const auto& item : data) {
            flatten_json_values(item, out);
        }
        return;
    }
    out.push_back(data.get<T>());
}

ov::Tensor tensor_from_bridge_json(const nlohmann::json& node) {
    if (!node.is_object()) {
        throw std::runtime_error("Bridge tensor node must be an object");
    }
    const std::string dtype = node.value("dtype", "");
    const auto& shape_json = node.at("shape");
    ov::Shape shape;
    shape.reserve(shape_json.size());
    for (const auto& d : shape_json) {
        shape.push_back(static_cast<size_t>(d.get<int64_t>()));
    }

    if (dtype == "float32" || dtype == "f32") {
        std::vector<float> values;
        values.reserve(ov::shape_size(shape));
        flatten_json_values(node.at("data"), values);
        ov::Tensor t(ov::element::f32, shape);
        std::memcpy(t.data(), values.data(), values.size() * sizeof(float));
        return t;
    }
    if (dtype == "int64" || dtype == "i64") {
        std::vector<int64_t> values;
        values.reserve(ov::shape_size(shape));
        flatten_json_values(node.at("data"), values);
        ov::Tensor t(ov::element::i64, shape);
        std::memcpy(t.data(), values.data(), values.size() * sizeof(int64_t));
        return t;
    }
    if (dtype == "bool") {
        std::vector<uint8_t> values;
        values.reserve(ov::shape_size(shape));
        flatten_json_values(node.at("data"), values);
        ov::Tensor t(ov::element::boolean, shape);
        std::memcpy(t.data(), values.data(), values.size() * sizeof(uint8_t));
        return t;
    }

    throw std::runtime_error("Unsupported bridge tensor dtype: " + dtype);
}

enum class PrecisionMode {
    kMixed,
    kDefault,
    kFP32,
    kInfFp32KvInt8,
    kInfFp32KvInt4,
    kInfFp16KvInt8,
    kInfFp16KvInt4,
    kInfFp32KvFp32WInt8,
    kInfFp32KvFp32WInt4Asym,
    kInfFp32KvInt8WInt4Asym,
    kInfFp16KvInt8WInt4Asym,
};

std::string to_lower_copy(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

PrecisionMode parse_precision_mode(const std::string& value) {
    const std::string mode = to_lower_copy(value);
    if (mode == "mixed") {
        return PrecisionMode::kMixed;
    }
    if (mode == "default") {
        return PrecisionMode::kDefault;
    }
    if (mode == "fp32") {
        return PrecisionMode::kFP32;
    }
    if (mode == "inf_fp32_kv_int8" || mode == "fp32_kv8") {
        return PrecisionMode::kInfFp32KvInt8;
    }
    if (mode == "inf_fp32_kv_int4" || mode == "fp32_kv4") {
        return PrecisionMode::kInfFp32KvInt4;
    }
    if (mode == "inf_fp16_kv_int8" || mode == "fp16_kv8") {
        return PrecisionMode::kInfFp16KvInt8;
    }
    if (mode == "inf_fp16_kv_int4" || mode == "fp16_kv4") {
        return PrecisionMode::kInfFp16KvInt4;
    }
    if (mode == "inf_fp32_kv_fp32_w_int8") {
        return PrecisionMode::kInfFp32KvFp32WInt8;
    }
    if (mode == "inf_fp32_kv_fp32_w_int4_asym") {
        return PrecisionMode::kInfFp32KvFp32WInt4Asym;
    }
    if (mode == "inf_fp32_kv_int8_w_int4_asym") {
        return PrecisionMode::kInfFp32KvInt8WInt4Asym;
    }
    if (mode == "inf_fp16_kv_int8_w_int4_asym") {
        return PrecisionMode::kInfFp16KvInt8WInt4Asym;
    }

    if (mode == "1" || mode == "true" || mode == "on") {
        return PrecisionMode::kFP32;
    }
    if (mode == "0" || mode == "false" || mode == "off") {
        return PrecisionMode::kMixed;
    }

    throw std::runtime_error(
        "Invalid PRECISION_MODE value: " + value +
        " (expected mixed/default/fp32/inf_fp32_kv_int8/inf_fp32_kv_int4/inf_fp16_kv_int8/inf_fp16_kv_int4/"
        "inf_fp32_kv_fp32_w_int8/inf_fp32_kv_fp32_w_int4_asym/inf_fp32_kv_int8_w_int4_asym/inf_fp16_kv_int8_w_int4_asym)");
}

std::string precision_mode_to_string(PrecisionMode mode) {
    switch (mode) {
    case PrecisionMode::kMixed:
        return "mixed";
    case PrecisionMode::kDefault:
        return "default";
    case PrecisionMode::kFP32:
        return "fp32";
    case PrecisionMode::kInfFp32KvInt8:
        return "inf_fp32_kv_int8";
    case PrecisionMode::kInfFp32KvInt4:
        return "inf_fp32_kv_int4";
    case PrecisionMode::kInfFp16KvInt8:
        return "inf_fp16_kv_int8";
    case PrecisionMode::kInfFp16KvInt4:
        return "inf_fp16_kv_int4";
    case PrecisionMode::kInfFp32KvFp32WInt8:
        return "inf_fp32_kv_fp32_w_int8";
    case PrecisionMode::kInfFp32KvFp32WInt4Asym:
        return "inf_fp32_kv_fp32_w_int4_asym";
    case PrecisionMode::kInfFp32KvInt8WInt4Asym:
        return "inf_fp32_kv_int8_w_int4_asym";
    case PrecisionMode::kInfFp16KvInt8WInt4Asym:
        return "inf_fp16_kv_int8_w_int4_asym";
    }
    return "unknown";
}

ov::genai::modeling::weights::QuantizationConfig quant_config_for_mode(PrecisionMode mode) {
    using QMode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    ov::genai::modeling::weights::QuantizationConfig config;

    switch (mode) {
    case PrecisionMode::kInfFp32KvFp32WInt8:
        config.mode = QMode::INT8_ASYM;
        config.backup_mode = QMode::INT8_ASYM;
        config.selection.verbose = false;
        break;
    case PrecisionMode::kInfFp32KvFp32WInt4Asym:
    case PrecisionMode::kInfFp32KvInt8WInt4Asym:
    case PrecisionMode::kInfFp16KvInt8WInt4Asym:
        config.mode = QMode::INT4_ASYM;
        config.backup_mode = QMode::INT4_ASYM;
        config.selection.verbose = false;
        break;
    default:
        break;
    }
    return config;
}

std::string quant_mode_cache_token(ov::genai::modeling::weights::QuantizationConfig::Mode mode) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    switch (mode) {
    case Mode::INT4_SYM:
        return "int4_sym";
    case Mode::INT4_ASYM:
        return "int4_asym";
    case Mode::INT8_SYM:
        return "int8_sym";
    case Mode::INT8_ASYM:
        return "int8_asym";
    case Mode::NONE:
    default:
        return "none";
    }
}

std::string quant_cache_suffix(const ov::genai::modeling::weights::QuantizationConfig& cfg) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    if (cfg.mode == Mode::NONE && cfg.backup_mode == Mode::NONE) {
        return "";
    }
    return "_q" + quant_mode_cache_token(cfg.mode) + "_b" + quant_mode_cache_token(cfg.backup_mode) + "_g" +
           std::to_string(cfg.group_size);
}

struct SampleOptions {
    std::optional<std::filesystem::path> model_dir;
    std::optional<std::filesystem::path> image_path;
    std::optional<std::string> user_prompt;
    std::optional<std::string> device;
    std::optional<int> max_new_tokens;
    std::filesystem::path dump_dir;
    std::filesystem::path py_ref_dump_dir;
    std::optional<PrecisionMode> precision_mode;
    std::filesystem::path dump_ir_dir;
};

int parse_i32(const std::string& raw, const char* option_name) {
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + option_name + ": " + raw);
    }
}

void print_usage(const char* argv0) {
    std::cerr
        << "Qwen3-Omni Modeling Sample\n"
        << "========================\n\n"
    << "Named options:\n"
        << "  " << argv0 << " --model-dir PATH --image PATH [--prompt TEXT] [--device NAME] [--output-tokens N]\n"
        << "      [--dump-dir PATH] [--py-ref-dump-dir PATH] [--precision MODE] [--dump-ir-dir PATH]\n\n"
        << "  --dump-ir-dir (IR dump directory)\n\n"
        << "PRECISION_MODE / --precision:\n"
        << "  mixed | default | fp32 | inf_fp32_kv_int8 | inf_fp32_kv_int4 | inf_fp16_kv_int8 | inf_fp16_kv_int4\n"
        << "  | inf_fp32_kv_fp32_w_int8 | inf_fp32_kv_fp32_w_int4_asym | inf_fp32_kv_int8_w_int4_asym | inf_fp16_kv_int8_w_int4_asym\n"
        << "  aliases: fp32_kv8/fp32_kv4/fp16_kv8/fp16_kv4 ; legacy 0/1 also supported\n";
}

SampleOptions parse_cli(int argc, char* argv[]) {
    SampleOptions opts;
    if (argc <= 1) {
        print_usage(argv[0]);
        std::exit(1);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            throw std::runtime_error("Positional arguments are not supported. Use --help and pass named options.");
        }

        auto take_value = [&](const char* option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + option_name);
            }
            return argv[++i];
        };

        if (arg == "--model-dir") {
            opts.model_dir = std::filesystem::path(take_value("--model-dir"));
        } else if (arg == "--image") {
            opts.image_path = std::filesystem::path(take_value("--image"));
        } else if (arg == "--prompt") {
            opts.user_prompt = take_value("--prompt");
        } else if (arg == "--device") {
            opts.device = take_value("--device");
        } else if (arg == "--output-tokens") {
            opts.max_new_tokens = parse_i32(take_value("--output-tokens"), "--output-tokens");
            if (opts.max_new_tokens <= 0) {
                throw std::runtime_error("--output-tokens must be a positive integer");
            }
        } else if (arg == "--dump-dir") {
            opts.dump_dir = std::filesystem::path(take_value("--dump-dir"));
        } else if (arg == "--py-ref-dump-dir") {
            opts.py_ref_dump_dir = std::filesystem::path(take_value("--py-ref-dump-dir"));
        } else if (arg == "--precision") {
            opts.precision_mode = parse_precision_mode(take_value("--precision"));
        } else if (arg == "--dump-ir-dir") {
            opts.dump_ir_dir = std::filesystem::path(take_value("--dump-ir-dir"));
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (!opts.model_dir.has_value() || !opts.image_path.has_value()) {
        print_usage(argv[0]);
        throw std::runtime_error("--model-dir and --image are required");
    }
    return opts;
}

}  // namespace

int main(int argc, char* argv[]) try {
    const SampleOptions opts = parse_cli(argc, argv);
    const std::filesystem::path model_dir = *opts.model_dir;
    const std::filesystem::path image_path = *opts.image_path;
    const std::string user_prompt = opts.user_prompt.value_or("What can you see");
    const std::string device = opts.device.value_or("CPU");
    const int max_new_tokens = opts.max_new_tokens.value_or(64);
    const std::filesystem::path dump_dir = opts.dump_dir;
    const std::filesystem::path py_ref_dump_dir = opts.py_ref_dump_dir;
    const std::filesystem::path dump_ir_dir = opts.dump_ir_dir;
    const PrecisionMode precision_mode = opts.precision_mode.value_or(PrecisionMode::kMixed);

    auto omni_cfg = ov::genai::modeling::models::Qwen3OmniConfig::from_json_file(model_dir);
    auto vl_cfg = ov::genai::modeling::models::to_qwen3_omni_vl_cfg(omni_cfg);
    const auto quant_config = quant_config_for_mode(precision_mode);

    ov::genai::modeling::models::Qwen3OmniVisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3OmniVisionPreprocessConfig::from_json_file(pre_cfg_path);
    }

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant_config);

    // Vision model must NOT be quantized — INT4/INT8 weights cause divide-by-zero
    // in the CPU plugin during vision inference.
    ov::genai::modeling::weights::QuantizationConfig no_quant;
    ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(no_quant);

    auto text_model = ov::genai::modeling::models::create_qwen3_omni_text_model(
        omni_cfg,
        source,
        finalizer,
        false,
        true);

    auto vision_model = ov::genai::modeling::models::create_qwen3_omni_vision_model(omni_cfg, source, vision_finalizer);

    if (precision_mode == PrecisionMode::kFP32 ||
        precision_mode == PrecisionMode::kInfFp32KvInt8 ||
        precision_mode == PrecisionMode::kInfFp32KvInt4 ||
        precision_mode == PrecisionMode::kInfFp32KvFp32WInt8 ||
        precision_mode == PrecisionMode::kInfFp32KvFp32WInt4Asym ||
        precision_mode == PrecisionMode::kInfFp32KvInt8WInt4Asym) {
        text_model->set_rt_info(ov::element::f32, {"runtime_options", ov::hint::kv_cache_precision.name()});
    } else if (precision_mode == PrecisionMode::kDefault) {
        text_model->set_rt_info(ov::element::bf16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    } else if (precision_mode == PrecisionMode::kInfFp16KvInt8 ||
               precision_mode == PrecisionMode::kInfFp16KvInt4 ||
               precision_mode == PrecisionMode::kInfFp16KvInt8WInt4Asym) {
        text_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }

    if (precision_mode == PrecisionMode::kInfFp32KvInt8 ||
        precision_mode == PrecisionMode::kInfFp16KvInt8 ||
        precision_mode == PrecisionMode::kInfFp32KvInt8WInt4Asym ||
        precision_mode == PrecisionMode::kInfFp16KvInt8WInt4Asym) {
        text_model->set_rt_info(ov::element::u8, {"runtime_options", ov::hint::kv_cache_precision.name()});
    } else if (precision_mode == PrecisionMode::kInfFp32KvInt4 ||
               precision_mode == PrecisionMode::kInfFp16KvInt4) {
        throw std::runtime_error(
            "Requested KV cache INT4 mode, but current OpenVINO kv_cache_precision supports f16/f32/i8/u8; "
            "INT4 is not available in this path.");
    }

    ov::Core core;
    ov::AnyMap compile_properties;

    if (precision_mode == PrecisionMode::kFP32 ||
        precision_mode == PrecisionMode::kInfFp32KvInt8 ||
        precision_mode == PrecisionMode::kInfFp32KvInt4 ||
        precision_mode == PrecisionMode::kInfFp32KvFp32WInt8 ||
        precision_mode == PrecisionMode::kInfFp32KvFp32WInt4Asym ||
        precision_mode == PrecisionMode::kInfFp32KvInt8WInt4Asym) {
        compile_properties.emplace(ov::hint::inference_precision.name(), ov::element::f32);
    } else if (precision_mode == PrecisionMode::kDefault) {
        compile_properties.emplace(ov::hint::inference_precision.name(), ov::element::bf16);
    } else if (precision_mode == PrecisionMode::kInfFp16KvInt8 ||
               precision_mode == PrecisionMode::kInfFp16KvInt4 ||
               precision_mode == PrecisionMode::kInfFp16KvInt8WInt4Asym) {
        compile_properties.emplace(ov::hint::inference_precision.name(), ov::element::f16);
    }

    if (precision_mode == PrecisionMode::kInfFp32KvInt8 ||
        precision_mode == PrecisionMode::kInfFp16KvInt8 ||
        precision_mode == PrecisionMode::kInfFp32KvInt8WInt4Asym ||
        precision_mode == PrecisionMode::kInfFp16KvInt8WInt4Asym) {
        compile_properties.emplace(ov::hint::kv_cache_precision.name(), ov::element::u8);
    }

    if (!opts.dump_ir_dir.empty()) {
        std::filesystem::create_directories(opts.dump_ir_dir);
        std::string text_ir_stem = "qwen3_omni_text" + quant_cache_suffix(quant_config);
        std::string vision_ir_stem = "qwen3_omni_vision" + quant_cache_suffix(quant_config);
        const std::string full_text_ir_path = (opts.dump_ir_dir / (text_ir_stem + ".xml")).string();
        const std::string full_vision_ir_path = (opts.dump_ir_dir / (vision_ir_stem + ".xml")).string();

        ov::serialize(text_model, full_text_ir_path);
        ov::serialize(vision_model, full_vision_ir_path);

        std::cout << "[DEBUG] Saved text model IR to " << full_text_ir_path << std::endl;
        std::cout << "[DEBUG] Saved vision model IR to " << full_vision_ir_path << std::endl;
    }

    auto compiled_vision = core.compile_model(vision_model, device, compile_properties);
    auto compiled_text = core.compile_model(text_model, device, compile_properties);

    std::unordered_set<std::string> vision_output_names;
    for (const auto& out : compiled_vision.outputs()) {
        for (const auto& name : out.get_names()) {
            vision_output_names.insert(name);
        }
    }

    ov::genai::modeling::models::Qwen3OmniVisionInputs vision_inputs;
    auto preprocess_start = std::chrono::steady_clock::now();
    auto preprocess_end = preprocess_start;

    const bool has_ref_vision_inputs = !py_ref_dump_dir.empty() &&
                                       has_dump_tensor(py_ref_dump_dir, "vision_pixel_values") &&
                                       has_dump_tensor(py_ref_dump_dir, "vision_grid_thw") &&
                                       has_dump_tensor(py_ref_dump_dir, "vision_pos_embeds") &&
                                       has_dump_tensor(py_ref_dump_dir, "vision_rotary_cos") &&
                                       has_dump_tensor(py_ref_dump_dir, "vision_rotary_sin");

    if (has_ref_vision_inputs) {
        vision_inputs.pixel_values = load_dump_tensor(py_ref_dump_dir, "vision_pixel_values");
        vision_inputs.grid_thw = load_dump_tensor(py_ref_dump_dir, "vision_grid_thw");
        vision_inputs.pos_embeds = load_dump_tensor(py_ref_dump_dir, "vision_pos_embeds");
        vision_inputs.rotary_cos = load_dump_tensor(py_ref_dump_dir, "vision_rotary_cos");
        vision_inputs.rotary_sin = load_dump_tensor(py_ref_dump_dir, "vision_rotary_sin");
    } else {
        ov::Tensor image;
        try {
            image = load_image_with_qwen_omni_bridge(image_path);
        } catch (const std::exception&) {
            image = utils::load_image(image_path);
        }
        const std::string pos_embed_name = resolve_pos_embed_name(source);
        const ov::Tensor pos_embed_weight = source.get_tensor(pos_embed_name);

        ov::genai::modeling::models::Qwen3OmniVisionPreprocessor preprocessor(omni_cfg, pre_cfg);
        preprocess_start = std::chrono::steady_clock::now();
        vision_inputs = preprocessor.preprocess(image, pos_embed_weight);
        preprocess_end = std::chrono::steady_clock::now();
    }

    auto vision_request = compiled_vision.create_infer_request();
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPixelValues, vision_inputs.pixel_values);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kGridThw, vision_inputs.grid_thw);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPosEmbeds, vision_inputs.pos_embeds);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotaryCos, vision_inputs.rotary_cos);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotarySin, vision_inputs.rotary_sin);
    vision_request.set_tensor("attention_mask", build_vision_attention_mask(vision_inputs.grid_thw));
    const auto vision_start = std::chrono::steady_clock::now();
    vision_request.infer();
    const auto vision_end = std::chrono::steady_clock::now();

    ov::Tensor visual_embeds = vision_request.get_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kVisualEmbeds);
    std::vector<ov::Tensor> deepstack_embeds;
    deepstack_embeds.reserve(vl_cfg.vision.deepstack_visual_indexes.size());
    for (size_t i = 0; i < vl_cfg.vision.deepstack_visual_indexes.size(); ++i) {
        const std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        deepstack_embeds.push_back(vision_request.get_tensor(name));
    }

    auto try_get_vision_tensor = [&](ov::Tensor& tensor,
                                     const std::string& primary,
                                     const std::string& fallback = std::string()) {
        if (vision_output_names.count(primary)) {
            tensor = vision_request.get_tensor(primary);
            return true;
        }
        if (!fallback.empty() && vision_output_names.count(fallback)) {
            tensor = vision_request.get_tensor(fallback);
            return true;
        }
        return false;
    };

    std::vector<std::pair<int32_t, ov::Tensor>> vision_block_hiddens;
    vision_block_hiddens.reserve(static_cast<size_t>(vl_cfg.vision.depth));
    for (int32_t i = 0; i < vl_cfg.vision.depth; ++i) {
        ov::Tensor hidden;
        const std::string primary = std::string("vision_block_hidden.") + std::to_string(i);
        const std::string fallback = std::string("vision_block_hidden_") + std::to_string(i);
        if (try_get_vision_tensor(hidden, primary, fallback)) {
            vision_block_hiddens.emplace_back(i, hidden);
        }
    }

    const bool has_block25_substages = (vl_cfg.vision.depth > 25) &&
                                       (vision_output_names.count("vision_block25.norm1") ||
                                        vision_output_names.count("vision_block25_norm1"));
    const bool has_block26_substages = (vl_cfg.vision.depth > 26) &&
                                       (vision_output_names.count("vision_block26.norm1") ||
                                        vision_output_names.count("vision_block26_norm1"));
    const bool has_preblock0_stages = (vl_cfg.vision.depth > 0) &&
                                      (vision_output_names.count("vision_preblock0_patch_embed_input_5d") ||
                                       vision_output_names.count("vision_preblock0_patch_embed_conv_5d") ||
                                       vision_output_names.count("vision_preblock0_patch_embed") ||
                                       vision_output_names.count("vision_preblock0_with_pos"));
    const bool has_block0_substages = (vl_cfg.vision.depth > 0) &&
                                      (vision_output_names.count("vision_block0.input") ||
                                       vision_output_names.count("vision_block0_input"));
    const bool has_block16_substages = (vl_cfg.vision.depth > 16) &&
                                       (vision_output_names.count("vision_block16.input") ||
                                        vision_output_names.count("vision_block16_input"));
    ov::Tensor vision_preblock0_patch_embed_input_5d;
    ov::Tensor vision_preblock0_patch_embed_conv_5d;
    ov::Tensor vision_preblock0_patch_embed;
    ov::Tensor vision_preblock0_with_pos;
    ov::Tensor vision_block0_input;
    ov::Tensor vision_block0_norm1;
    ov::Tensor vision_block0_qkv;
    ov::Tensor vision_block0_attn_out;
    ov::Tensor vision_block0_resid1;
    ov::Tensor vision_block0_norm2;
    ov::Tensor vision_block0_mlp_out;
    ov::Tensor vision_block0_mlp_fc1;
    ov::Tensor vision_block0_mlp_act;
    ov::Tensor vision_block0_mlp_fc2;
    ov::Tensor vision_block16_input;
    ov::Tensor vision_block16_norm1;
    ov::Tensor vision_block16_qkv;
    ov::Tensor vision_block25_norm1;
    ov::Tensor vision_block25_qkv;
    ov::Tensor vision_block25_attn_out;
    ov::Tensor vision_block25_resid1;
    ov::Tensor vision_block25_norm2;
    ov::Tensor vision_block25_mlp_out;
    ov::Tensor vision_block25_mlp_fc1;
    ov::Tensor vision_block25_mlp_act;
    ov::Tensor vision_block25_mlp_fc2;
    ov::Tensor vision_block26_norm1;
    ov::Tensor vision_block26_qkv;
    ov::Tensor vision_block26_attn_out;
    ov::Tensor vision_block26_resid1;
    ov::Tensor vision_block26_norm2;
    ov::Tensor vision_block26_mlp_out;
    ov::Tensor vision_block26_mlp_fc1;
    ov::Tensor vision_block26_mlp_act;
    ov::Tensor vision_block26_mlp_fc2;
    bool has_block25_norm1 = false;
    bool has_block25_qkv = false;
    bool has_block25_attn_out = false;
    bool has_block25_resid1 = false;
    bool has_block25_norm2 = false;
    bool has_block25_mlp_out = false;
    bool has_block25_mlp_fc1 = false;
    bool has_block25_mlp_act = false;
    bool has_block25_mlp_fc2 = false;
    bool has_block26_norm1 = false;
    bool has_block26_qkv = false;
    bool has_block26_attn_out = false;
    bool has_block26_resid1 = false;
    bool has_block26_norm2 = false;
    bool has_block26_mlp_out = false;
    bool has_block26_mlp_fc1 = false;
    bool has_block26_mlp_act = false;
    bool has_block26_mlp_fc2 = false;
    bool has_preblock0_patch_embed_input_5d = false;
    bool has_preblock0_patch_embed_conv_5d = false;
    bool has_preblock0_patch_embed = false;
    bool has_preblock0_with_pos = false;
    bool has_block0_input = false;
    bool has_block0_norm1 = false;
    bool has_block0_qkv = false;
    bool has_block0_attn_out = false;
    bool has_block0_resid1 = false;
    bool has_block0_norm2 = false;
    bool has_block0_mlp_out = false;
    bool has_block0_mlp_fc1 = false;
    bool has_block0_mlp_act = false;
    bool has_block0_mlp_fc2 = false;
    bool has_block16_input = false;
    bool has_block16_norm1 = false;
    bool has_block16_qkv = false;
    if (has_preblock0_stages) {
        has_preblock0_patch_embed_input_5d =
            try_get_vision_tensor(vision_preblock0_patch_embed_input_5d,
                                  "vision_preblock0_patch_embed_input_5d",
                                  "vision_preblock0_patch_embed_input_5d");
        has_preblock0_patch_embed_conv_5d =
            try_get_vision_tensor(vision_preblock0_patch_embed_conv_5d,
                                  "vision_preblock0_patch_embed_conv_5d",
                                  "vision_preblock0_patch_embed_conv_5d");
        has_preblock0_patch_embed =
            try_get_vision_tensor(vision_preblock0_patch_embed,
                                  "vision_preblock0_patch_embed",
                                  "vision_preblock0_patch_embed");
        has_preblock0_with_pos =
            try_get_vision_tensor(vision_preblock0_with_pos,
                                  "vision_preblock0_with_pos",
                                  "vision_preblock0_with_pos");
    }

    if (has_block0_substages) {
        has_block0_input = try_get_vision_tensor(vision_block0_input, "vision_block0.input", "vision_block0_input");
        has_block0_norm1 = try_get_vision_tensor(vision_block0_norm1, "vision_block0.norm1", "vision_block0_norm1");
        has_block0_qkv = try_get_vision_tensor(vision_block0_qkv, "vision_block0.qkv", "vision_block0_qkv");
        has_block0_attn_out =
            try_get_vision_tensor(vision_block0_attn_out, "vision_block0.attn_out", "vision_block0_attn_out");
        has_block0_resid1 =
            try_get_vision_tensor(vision_block0_resid1, "vision_block0.resid1", "vision_block0_resid1");
        has_block0_norm2 = try_get_vision_tensor(vision_block0_norm2, "vision_block0.norm2", "vision_block0_norm2");
        has_block0_mlp_out =
            try_get_vision_tensor(vision_block0_mlp_out, "vision_block0.mlp_out", "vision_block0_mlp_out");
        has_block0_mlp_fc1 =
            try_get_vision_tensor(vision_block0_mlp_fc1, "vision_block0.mlp_fc1", "vision_block0_mlp_fc1");
        has_block0_mlp_act =
            try_get_vision_tensor(vision_block0_mlp_act, "vision_block0.mlp_act", "vision_block0_mlp_act");
        has_block0_mlp_fc2 =
            try_get_vision_tensor(vision_block0_mlp_fc2, "vision_block0.mlp_fc2", "vision_block0_mlp_fc2");
    }

    if (has_block16_substages) {
        has_block16_input = try_get_vision_tensor(vision_block16_input, "vision_block16.input", "vision_block16_input");
        has_block16_norm1 = try_get_vision_tensor(vision_block16_norm1, "vision_block16.norm1", "vision_block16_norm1");
        has_block16_qkv = try_get_vision_tensor(vision_block16_qkv, "vision_block16.qkv", "vision_block16_qkv");
    }

    if (has_block25_substages) {
        has_block25_norm1 = try_get_vision_tensor(vision_block25_norm1, "vision_block25.norm1", "vision_block25_norm1");
        has_block25_qkv = try_get_vision_tensor(vision_block25_qkv, "vision_block25.qkv", "vision_block25_qkv");
        has_block25_attn_out = try_get_vision_tensor(vision_block25_attn_out, "vision_block25.attn_out", "vision_block25_attn_out");
        has_block25_resid1 = try_get_vision_tensor(vision_block25_resid1, "vision_block25.resid1", "vision_block25_resid1");
        has_block25_norm2 = try_get_vision_tensor(vision_block25_norm2, "vision_block25.norm2", "vision_block25_norm2");
        has_block25_mlp_out = try_get_vision_tensor(vision_block25_mlp_out, "vision_block25.mlp_out", "vision_block25_mlp_out");
        has_block25_mlp_fc1 = try_get_vision_tensor(vision_block25_mlp_fc1, "vision_block25.mlp_fc1", "vision_block25_mlp_fc1");
        has_block25_mlp_act = try_get_vision_tensor(vision_block25_mlp_act, "vision_block25.mlp_act", "vision_block25_mlp_act");
        has_block25_mlp_fc2 = try_get_vision_tensor(vision_block25_mlp_fc2, "vision_block25.mlp_fc2", "vision_block25_mlp_fc2");
    }

    if (has_block26_substages) {
        has_block26_norm1 = try_get_vision_tensor(vision_block26_norm1, "vision_block26.norm1", "vision_block26_norm1");
        has_block26_qkv = try_get_vision_tensor(vision_block26_qkv, "vision_block26.qkv", "vision_block26_qkv");
        has_block26_attn_out = try_get_vision_tensor(vision_block26_attn_out, "vision_block26.attn_out", "vision_block26_attn_out");
        has_block26_resid1 = try_get_vision_tensor(vision_block26_resid1, "vision_block26.resid1", "vision_block26_resid1");
        has_block26_norm2 = try_get_vision_tensor(vision_block26_norm2, "vision_block26.norm2", "vision_block26_norm2");
        has_block26_mlp_out = try_get_vision_tensor(vision_block26_mlp_out, "vision_block26.mlp_out", "vision_block26_mlp_out");
        has_block26_mlp_fc1 = try_get_vision_tensor(vision_block26_mlp_fc1, "vision_block26.mlp_fc1", "vision_block26_mlp_fc1");
        has_block26_mlp_act = try_get_vision_tensor(vision_block26_mlp_act, "vision_block26.mlp_act", "vision_block26_mlp_act");
        has_block26_mlp_fc2 = try_get_vision_tensor(vision_block26_mlp_fc2, "vision_block26.mlp_fc2", "vision_block26_mlp_fc2");
    }

    std::vector<int32_t> token_debug_blocks = {16, 20, 24, 26};

    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    ov::Tensor position_ids;
    ov::Tensor visual_pos_mask;
    ov::Tensor rope_deltas;
    ov::Tensor visual_padded;
    ov::Tensor ref_audio_features;
    ov::Tensor ref_audio_pos_mask;
    bool has_ref_audio_inputs = false;
    bool has_bridge_visual_inputs = false;
    const char* audio_path_env = std::getenv("QWEN3_OMNI_AUDIO_PATH");
    const bool has_audio_env = audio_path_env && std::strlen(audio_path_env) > 0;
    std::vector<ov::Tensor> deepstack_padded;

    const bool has_ref_text_inputs = !py_ref_dump_dir.empty() &&
                                     has_dump_tensor(py_ref_dump_dir, "input_ids") &&
                                     has_dump_tensor(py_ref_dump_dir, "attention_mask") &&
                                     has_dump_tensor(py_ref_dump_dir, "position_ids") &&
                                     has_dump_tensor(py_ref_dump_dir, "visual_pos_mask") &&
                                     has_dump_tensor(py_ref_dump_dir, "rope_deltas") &&
                                     has_dump_tensor(py_ref_dump_dir, "vision_visual_embeds_padded");

    ov::genai::Tokenizer tokenizer(model_dir);
    if (has_ref_text_inputs) {
        input_ids = load_dump_tensor(py_ref_dump_dir, "input_ids");
        attention_mask = load_dump_tensor(py_ref_dump_dir, "attention_mask");
        position_ids = load_dump_tensor(py_ref_dump_dir, "position_ids");
        visual_pos_mask = load_dump_tensor(py_ref_dump_dir, "visual_pos_mask");
        rope_deltas = load_dump_tensor(py_ref_dump_dir, "rope_deltas");
        visual_padded = load_dump_tensor(py_ref_dump_dir, "vision_visual_embeds_padded");
        if (has_dump_tensor(py_ref_dump_dir, "audio_features") && has_dump_tensor(py_ref_dump_dir, "audio_pos_mask")) {
            ref_audio_features = load_dump_tensor(py_ref_dump_dir, "audio_features");
            ref_audio_pos_mask = load_dump_tensor(py_ref_dump_dir, "audio_pos_mask");
            has_ref_audio_inputs = true;
        }

        const size_t expected_deepstack = vl_cfg.vision.deepstack_visual_indexes.size();
        deepstack_padded.reserve(expected_deepstack);
        for (size_t i = 0; i < expected_deepstack; ++i) {
            const std::string name = std::string("vision_deepstack_padded_") + std::to_string(i);
            if (has_dump_tensor(py_ref_dump_dir, name)) {
                deepstack_padded.push_back(load_dump_tensor(py_ref_dump_dir, name));
            } else {
                deepstack_padded.push_back(
                    make_zero_tensor(ov::element::f32,
                                     {input_ids.get_shape()[0], input_ids.get_shape()[1], static_cast<size_t>(vl_cfg.text.hidden_size)}));
            }
        }
    } else if (has_audio_env) {
        nlohmann::json conversation = nlohmann::json::array({
            {
                {"role", "user"},
                {"content", nlohmann::json::array({
                    {{"type", "image"}, {"image", image_path.string()}},
                    {{"type", "audio"}, {"audio", std::string(audio_path_env)}},
                    {{"type", "text"}, {"text", user_prompt}},
                })},
            },
        });
        auto bridge_info = ov::genai::modeling::models::Qwen3OmniAudioProcess::process_audio_features_via_python(
            conversation,
            model_dir.string(),
            true);
        if (!bridge_info.is_object()) {
            throw std::runtime_error("audio feature bridge output must be an object");
        }

        input_ids = tensor_from_bridge_json(bridge_info.at("input_ids"));
        attention_mask = tensor_from_bridge_json(bridge_info.at("attention_mask"));
        position_ids = tensor_from_bridge_json(bridge_info.at("position_ids"));
        visual_pos_mask = tensor_from_bridge_json(bridge_info.at("visual_pos_mask"));
        rope_deltas = tensor_from_bridge_json(bridge_info.at("rope_deltas"));

        ref_audio_features = tensor_from_bridge_json(bridge_info.at("audio_features"));
        ref_audio_pos_mask = tensor_from_bridge_json(bridge_info.at("audio_pos_mask"));
        has_ref_audio_inputs = true;
        visual_padded = tensor_from_bridge_json(bridge_info.at("visual_embeds_padded"));
        has_bridge_visual_inputs = true;

        deepstack_padded.clear();
        if (bridge_info.contains("deepstack_padded") && bridge_info.at("deepstack_padded").is_array()) {
            for (const auto& item : bridge_info.at("deepstack_padded")) {
                deepstack_padded.push_back(tensor_from_bridge_json(item));
            }
        }
        const size_t expected_deepstack = vl_cfg.vision.deepstack_visual_indexes.size();
        while (deepstack_padded.size() < expected_deepstack) {
            deepstack_padded.push_back(
                make_zero_tensor(ov::element::f32,
                                 {input_ids.get_shape()[0], input_ids.get_shape()[1], static_cast<size_t>(vl_cfg.text.hidden_size)}));
        }
    } else {
        const int64_t image_tokens =
            ov::genai::modeling::models::Qwen3OmniVisionPreprocessor::count_visual_tokens(
                vision_inputs.grid_thw,
                vl_cfg.vision.spatial_merge_size);
        const std::string prompt = build_prompt(user_prompt, image_tokens);
        auto tokenized = tokenizer.encode(prompt, ov::genai::add_special_tokens(false));

        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;

        ov::genai::modeling::models::Qwen3VLInputPlanner planner(vl_cfg);
        auto plan = planner.build_plan(input_ids, &attention_mask, &vision_inputs.grid_thw);
        position_ids = plan.position_ids;
        visual_pos_mask = plan.visual_pos_mask;
        rope_deltas = plan.rope_deltas;

        visual_padded =
            ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, visual_pos_mask);
        deepstack_padded =
            ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_deepstack_embeds(deepstack_embeds, visual_pos_mask);
    }

    const size_t batch = input_ids.get_shape().at(0);
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

    if (!dump_dir.empty()) {
        dump_tensor_bin(dump_dir, "vision_pixel_values", vision_inputs.pixel_values);
        dump_tensor_bin(dump_dir, "vision_pos_embeds", vision_inputs.pos_embeds);
        dump_tensor_bin(dump_dir, "vision_rotary_cos", vision_inputs.rotary_cos);
        dump_tensor_bin(dump_dir, "vision_rotary_sin", vision_inputs.rotary_sin);
        dump_tensor_bin(dump_dir, "input_ids", input_ids);
        dump_tensor_bin(dump_dir, "attention_mask", attention_mask);
        dump_tensor_bin(dump_dir, "position_ids", position_ids);
        dump_tensor_bin(dump_dir, "visual_pos_mask", visual_pos_mask);
        dump_tensor_bin(dump_dir, "rope_deltas", rope_deltas);
        if (has_ref_audio_inputs) {
            dump_tensor_bin(dump_dir, "audio_features", ref_audio_features);
            dump_tensor_bin(dump_dir, "audio_pos_mask", ref_audio_pos_mask);
        }
        if (!has_bridge_visual_inputs) {
            dump_tensor_bin(dump_dir, "vision_visual_embeds_compact", visual_embeds);
        }
        dump_tensor_bin(dump_dir, "vision_visual_embeds_padded", visual_padded);
        dump_tensor_bin(dump_dir, "vision_grid_thw", vision_inputs.grid_thw);
        for (const auto& block_hidden : vision_block_hiddens) {
            dump_tensor_bin(dump_dir,
                            std::string("vision_block_hidden_") + std::to_string(block_hidden.first),
                            block_hidden.second);
        }
        if (has_preblock0_stages) {
            if (has_preblock0_patch_embed_input_5d) {
                dump_tensor_bin(dump_dir, "vision_preblock0_patch_embed_input_5d", vision_preblock0_patch_embed_input_5d);
            }
            if (has_preblock0_patch_embed_conv_5d) {
                dump_tensor_bin(dump_dir, "vision_preblock0_patch_embed_conv_5d", vision_preblock0_patch_embed_conv_5d);
            }
            if (has_preblock0_patch_embed) {
                dump_tensor_bin(dump_dir, "vision_preblock0_patch_embed", vision_preblock0_patch_embed);
            }
            if (has_preblock0_with_pos) {
                dump_tensor_bin(dump_dir, "vision_preblock0_with_pos", vision_preblock0_with_pos);
            }
        }
        if (has_block0_substages) {
            if (has_block0_input) {
                dump_tensor_bin(dump_dir, "vision_block0_input", vision_block0_input);
            }
            if (has_block0_norm1) {
                dump_tensor_bin(dump_dir, "vision_block0_norm1", vision_block0_norm1);
            }
            if (has_block0_qkv) {
                dump_tensor_bin(dump_dir, "vision_block0_qkv", vision_block0_qkv);
            }
            if (has_block0_attn_out) {
                dump_tensor_bin(dump_dir, "vision_block0_attn_out", vision_block0_attn_out);
            }
            if (has_block0_resid1) {
                dump_tensor_bin(dump_dir, "vision_block0_resid1", vision_block0_resid1);
            }
            if (has_block0_norm2) {
                dump_tensor_bin(dump_dir, "vision_block0_norm2", vision_block0_norm2);
            }
            if (has_block0_mlp_out) {
                dump_tensor_bin(dump_dir, "vision_block0_mlp_out", vision_block0_mlp_out);
            }
            if (has_block0_mlp_fc1) {
                dump_tensor_bin(dump_dir, "vision_block0_mlp_fc1", vision_block0_mlp_fc1);
            }
            if (has_block0_mlp_act) {
                dump_tensor_bin(dump_dir, "vision_block0_mlp_act", vision_block0_mlp_act);
            }
            if (has_block0_mlp_fc2) {
                dump_tensor_bin(dump_dir, "vision_block0_mlp_fc2", vision_block0_mlp_fc2);
            }
        }
        if (has_block16_substages) {
            if (has_block16_input) {
                dump_tensor_bin(dump_dir, "vision_block16_input", vision_block16_input);
            }
            if (has_block16_norm1) {
                dump_tensor_bin(dump_dir, "vision_block16_norm1", vision_block16_norm1);
            }
            if (has_block16_qkv) {
                dump_tensor_bin(dump_dir, "vision_block16_qkv", vision_block16_qkv);
            }
        }
        if (has_block25_substages) {
            if (has_block25_norm1) {
                dump_tensor_bin(dump_dir, "vision_block25_norm1", vision_block25_norm1);
            }
            if (has_block25_qkv) {
                dump_tensor_bin(dump_dir, "vision_block25_qkv", vision_block25_qkv);
            }
            if (has_block25_attn_out) {
                dump_tensor_bin(dump_dir, "vision_block25_attn_out", vision_block25_attn_out);
            }
            if (has_block25_resid1) {
                dump_tensor_bin(dump_dir, "vision_block25_resid1", vision_block25_resid1);
            }
            if (has_block25_norm2) {
                dump_tensor_bin(dump_dir, "vision_block25_norm2", vision_block25_norm2);
            }
            if (has_block25_mlp_out) {
                dump_tensor_bin(dump_dir, "vision_block25_mlp_out", vision_block25_mlp_out);
            }
            if (has_block25_mlp_fc1) {
                dump_tensor_bin(dump_dir, "vision_block25_mlp_fc1", vision_block25_mlp_fc1);
            }
            if (has_block25_mlp_act) {
                dump_tensor_bin(dump_dir, "vision_block25_mlp_act", vision_block25_mlp_act);
            }
            if (has_block25_mlp_fc2) {
                dump_tensor_bin(dump_dir, "vision_block25_mlp_fc2", vision_block25_mlp_fc2);
            }
        }

        if (has_block26_substages) {
            if (has_block26_norm1) {
                dump_tensor_bin(dump_dir, "vision_block26_norm1", vision_block26_norm1);
            }
            if (has_block26_qkv) {
                dump_tensor_bin(dump_dir, "vision_block26_qkv", vision_block26_qkv);
            }
            if (has_block26_attn_out) {
                dump_tensor_bin(dump_dir, "vision_block26_attn_out", vision_block26_attn_out);
            }
            if (has_block26_resid1) {
                dump_tensor_bin(dump_dir, "vision_block26_resid1", vision_block26_resid1);
            }
            if (has_block26_norm2) {
                dump_tensor_bin(dump_dir, "vision_block26_norm2", vision_block26_norm2);
            }
            if (has_block26_mlp_out) {
                dump_tensor_bin(dump_dir, "vision_block26_mlp_out", vision_block26_mlp_out);
            }
            if (has_block26_mlp_fc1) {
                dump_tensor_bin(dump_dir, "vision_block26_mlp_fc1", vision_block26_mlp_fc1);
            }
            if (has_block26_mlp_act) {
                dump_tensor_bin(dump_dir, "vision_block26_mlp_act", vision_block26_mlp_act);
            }
            if (has_block26_mlp_fc2) {
                dump_tensor_bin(dump_dir, "vision_block26_mlp_fc2", vision_block26_mlp_fc2);
            }
        }

        auto dump_token_debug = [&](int32_t block_idx, const std::string& suffix) {
            ov::Tensor tensor;
            const std::string primary = std::string("vision_block") + std::to_string(block_idx) + ".token267." + suffix;
            const std::string fallback = std::string("vision_block") + std::to_string(block_idx) + "_token267_" + suffix;
            if (try_get_vision_tensor(tensor, primary, fallback)) {
                const std::string dump_name = std::string("vision_block") + std::to_string(block_idx) + "_token267_" + suffix;
                dump_tensor_bin(dump_dir, dump_name, tensor);
            }
        };

        for (int32_t block_idx : token_debug_blocks) {
            dump_token_debug(block_idx, "q_heads");
            dump_token_debug(block_idx, "k_heads");
            dump_token_debug(block_idx, "attn_prob");
            dump_token_debug(block_idx, "context");
        }
        for (size_t i = 0; i < deepstack_embeds.size(); ++i) {
            dump_tensor_bin(dump_dir,
                            std::string("vision_deepstack_compact_") + std::to_string(i),
                            deepstack_embeds[i]);
        }
        for (size_t i = 0; i < deepstack_padded.size(); ++i) {
            dump_tensor_bin(dump_dir,
                            std::string("vision_deepstack_padded_") + std::to_string(i),
                            deepstack_padded[i]);
        }
    }

    auto beam_idx = make_beam_idx(batch);

    ov::Tensor prefill_audio_features =
        make_zero_tensor(ov::element::f32, {batch, input_ids.get_shape()[1], static_cast<size_t>(vl_cfg.text.hidden_size)});
    ov::Tensor prefill_audio_pos_mask = make_zero_tensor(ov::element::boolean, {batch, input_ids.get_shape()[1]});
    if (has_ref_audio_inputs) {
        const ov::Shape expected_features_shape = {batch, input_ids.get_shape()[1], static_cast<size_t>(vl_cfg.text.hidden_size)};
        const ov::Shape expected_mask_shape = {batch, input_ids.get_shape()[1]};
        if (ref_audio_features.get_shape() != expected_features_shape ||
            ref_audio_pos_mask.get_shape() != expected_mask_shape) {
            throw std::runtime_error("python reference audio tensor shape mismatch with current text inputs");
        }
        prefill_audio_features = ref_audio_features;
        prefill_audio_pos_mask = ref_audio_pos_mask;
    }

    auto text_request = compiled_text.create_infer_request();
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kInputIds, input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kAttentionMask, attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kPositionIds, position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualEmbeds, visual_padded);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualPosMask, visual_pos_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kAudioFeatures, prefill_audio_features);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kAudioPosMask, prefill_audio_pos_mask);
    for (size_t i = 0; i < deepstack_padded.size(); ++i) {
        const std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        text_request.set_tensor(name, deepstack_padded[i]);
    }

    const auto prefill_start = std::chrono::steady_clock::now();
    text_request.infer();
    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
    int64_t next_id = argmax_last_token(logits);
    const auto prefill_end = std::chrono::steady_clock::now();

    if (!dump_dir.empty()) {
        dump_tensor_bin(dump_dir, "prefill_logits", logits);
        ov::Tensor next_token(ov::element::i64, {1});
        next_token.data<int64_t>()[0] = next_id;
        dump_tensor_bin(dump_dir, "prefill_next_token", next_token);
    }

    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    generated.push_back(next_id);

    const int64_t eos_token_id = tokenizer.get_eos_token_id();
    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }

    ov::Tensor decode_visual =
        make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(vl_cfg.text.hidden_size)});
    ov::Tensor decode_visual_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
    ov::Tensor decode_audio_features =
        make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(vl_cfg.text.hidden_size)});
    ov::Tensor decode_audio_pos_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
    std::vector<ov::Tensor> decode_deepstack;
    decode_deepstack.reserve(deepstack_padded.size());
    for (size_t i = 0; i < deepstack_padded.size(); ++i) {
        decode_deepstack.push_back(
            make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(vl_cfg.text.hidden_size)}));
    }

    int64_t past_len = prompt_len;
    size_t decode_steps = 0;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int step = 1; step < max_new_tokens; ++step) {
        if (eos_token_id >= 0 && next_id == eos_token_id) {
            break;
        }

        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            step_data[b] = next_id;
        }

        auto position_ids =
            ov::genai::modeling::models::Qwen3VLInputPlanner::build_decode_position_ids(rope_deltas, past_len, 1);

        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kInputIds, step_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kPositionIds, position_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kBeamIdx, beam_idx);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualEmbeds, decode_visual);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualPosMask, decode_visual_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kAudioFeatures, decode_audio_features);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kAudioPosMask, decode_audio_pos_mask);
        for (size_t i = 0; i < decode_deepstack.size(); ++i) {
            const std::string name =
                std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) + "." +
                std::to_string(i);
            text_request.set_tensor(name, decode_deepstack[i]);
        }

        text_request.infer();
        logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
        next_id = argmax_last_token(logits);
        generated.push_back(next_id);
        decode_steps += 1;
        past_len += 1;
    }
    const auto decode_end = std::chrono::steady_clock::now();

    std::string output = tokenizer.decode(generated, ov::genai::skip_special_tokens(true));

    const double preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);
    const double vision_ms = elapsed_ms(vision_start, vision_end);
    const double ttft_ms = elapsed_ms(prefill_start, prefill_end);
    const double decode_ms = elapsed_ms(decode_start, decode_end);
    const double tpot_ms = decode_steps > 0 ? (decode_ms / static_cast<double>(decode_steps)) : 0.0;
    const double throughput = decode_steps > 0 && decode_ms > 0.0
                                  ? (static_cast<double>(decode_steps) * 1000.0 / decode_ms)
                                  : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Precision mode: " << precision_mode_to_string(precision_mode) << std::endl;
    std::cout << "Prompt token size: " << prompt_len << std::endl;
    std::cout << "Output token size: " << generated.size() << std::endl;
    std::cout << "Preprocess time: " << preprocess_ms << " ms" << std::endl;
    std::cout << "Vision encode time: " << vision_ms << " ms" << std::endl;
    std::cout << "TTFT: " << ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << decode_ms << " ms" << std::endl;
    if (decode_steps > 0) {
        std::cout << "TPOT: " << tpot_ms << " ms/token" << std::endl;
        std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    } else {
        std::cout << "TPOT: N/A" << std::endl;
        std::cout << "Throughput: N/A" << std::endl;
    }
    std::cout << output << std::endl;

    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
