// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "json_utils.hpp"
#include "md_vae_tiling.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace module {

void VaeDecoderTilingModule::print_static_config() {
    std::cout << R"(
  vae_decoder_tiling:
    type: "VaeDecoderTilingModule"
    device: "CPU"
    inputs:
      - name: "latent"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "latents"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "latent"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
      - name: "latents"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
    params:
      tile_overlap_factor: "0.25"
      model_path: "model"

    )" << std::endl;
}

VaeDecoderTilingModule::VaeDecoderTilingModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    m_model_type = to_image_generation_model_type(desc->model_type);
    if (m_model_type != ImageGenerationModelType::ZIMAGE) {
        GENAI_ERR("TransformerModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
        return;
    }
    if (!initialize()) {
        GENAI_ERR("Failed to initiate TransformerModule");
    }
}

VaeDecoderTilingModule::~VaeDecoderTilingModule() {}

bool VaeDecoderTilingModule::init_tile_params(conststd::filesystem::path& model_path) {
    const auto& params = module_desc->params;
    auto it_tile_overlap = params.find("tile_overlap_factor");
    if (it_tile_overlap != params.end()) {
        m_tile_overlap_factor = std::stof(it_tile_overlap->second);
    }

    auto config_path = model_path / "vae_decoder/config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream vae_config(config_path);
        nlohmann::json parsed = nlohmann::json::parse(vae_config);
        utils::read_json_param(parsed, "sample_size", m_sample_size);
        m_tile_sample_min_size = m_sample_size;

        // get block_out_channels: "block_out_channels": [128,256,512,512]
        std::vector<int> block_out_channels;
        utils::read_json_param(parsed, "block_out_channels", block_out_channels);

        // m_tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        m_tile_latent_min_size = m_sample_size / std::pow(2, block_out_channels.size() - 1);
    } else {
        OPENVINO_ASSERT(false,
                        "VaeDecoderTilingModule[" + module_desc->name + "]: vae_decoder config file not found at " +
                            config_path.string());
    }

    return true;
}

bool VaeDecoderTilingModule::initialize() {
    const auto& params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("VaeDecoderTilingModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
    init_tile_params(model_path);
    return true;
}

void VaeDecoderTilingModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    std::vector<ov::Tensor> latents;
    if (this->inputs.find("latent") != this->inputs.end()) {
        latents.push_back(this->inputs["latent"].data.as<ov::Tensor>());
    } else if (this->inputs.find("latents") != this->inputs.end()) {
        latents = this->inputs["latents"].data.as<std::vector<ov::Tensor>>();
    } else {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: 'latent' or 'latents' input not found");
        return;
    }

    // Process batch of latents
    for (const auto& latent : latents) {
        OPENVINO_ASSERT(latent.get_shape().size() == 4,
                        "VaeDecoderTilingModule[" + module_desc->name + "]: latent tensor must be 4D.");

        if (m_enable_tiling && (latent[3] > m_tile_latent_min_size || latent[2] > m_tile_latent_min_size)) {
            // Tiling decode
            ov::Tensor output_latent;
            tile_decode(latent, output_latent);
        } else {
            // Non-tiling decode
        }
    }

    std::vector<ov::Tensor> output_latents;
    this->outputs["image"].data = output_latents.size() == 1 ? ov::Any(output_latents[0]) : ov::Any(output_latents);
}

ov::Tensor VaeDecoderTilingModule::decoder(const ov::Tensor& tile) {
    // VAE decoder model inference (not implemented here)
    const float coeff = 2.6666666666666665f;
    ov::Tensor decoded_tile =
        ov::Tensor(ov::element::f32,
                   ov::Shape{1,
                             3,
                             static_cast<size_t>(coeff * tile.get_shape()[2]),
                             static_cast<size_t>(coeff * tile.get_shape()[3])});  // Placeholder for decoded tile
    return decoded_tile;
}

void VaeDecoderTilingModule::tile_decode(const ov::Tensor& latent, ov::Tensor& output_latent) {
    // Tiling decode implementation
    size_t overlap_size = m_tile_latent_min_size * (1 - m_tile_overlap_factor);
    size_t blend_extent = m_tile_sample_min_size * m_tile_overlap_factor;
    size_t row_limit = m_tile_sample_min_size - blend_extent;

    size_t height = latent.get_shape()[2];
    size_t width = latent.get_shape()[3];

    std::vector<std::vector<ov::Tensor>> rows;
    for (size_t h = 0; h < height; h += overlap_size) {
        const size_t& h_start = h;
        const size_t& h_end = std::min(h + m_tile_sample_min_size, height);
        std::vector<ov::Tensor> row;
        for (size_t w = 0; w < width; w += overlap_size) {
            const size_t& w_start = w;
            const size_t& w_end = std::min(w + m_tile_sample_min_size, width);

            // Get ROI tile from latent
            ov::Tensor tile = ov::genai::module::tensor_utils::slice_tensor(
                latent,
                {0, 0, h_start, w_start},
                {latent.get_shape()[0], latent.get_shape()[1], h_end - h_start, w_end - w_start});

            ov::Tensor decoded_tile = decoder(tile);
            row.push_back(decoded_tile);
        }
        rows.push_back(row);
    }

    std::vector<ov::Tensor> result_rows;
    for (size_t i = 0; i < rows.size(); ++i) {
        std::vector<ov::Tensor> result_row;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            ov::Tensor tile = rows[i][j];
            if (i > 0) {
                tile = blend_v(rows[i - 1][j], tile, blend_extent);
            }
            if (j > 0) {
                tile = blend_h(result_row[j - 1], tile, blend_extent);
            }
            result_row.push_back(tensor_utils::slice_tensor(tile,
                                                            {0, 0, 0, 0},
                                                            {tile.get_shape()[0],
                                                             tile.get_shape()[1],
                                                             std::min(tile.get_shape()[2], row_limit),
                                                             std::min(tile.get_shape()[3], row_limit)}));
        }
        result_rows.push_back(tensor_utils::concat_tensors(result_row));
    }

    // Combine result_rows into output_latent (not implemented here)
    auto dec = tensor_utils::concat_tensors(result_rows, 2);
    return dec;
}

ov::Tensor VaeDecoderTilingModule::blend_v(ov::Tensor& tile1, ov::Tensor& tile2, size_t blend_extent) {

}

ov::Tensor VaeDecoderTilingModule::blend_h(ov::Tensor& tile1, ov::Tensor& tile2, size_t blend_extent) {

}

}  // namespace module
}  // namespace genai
}  // namespace ov