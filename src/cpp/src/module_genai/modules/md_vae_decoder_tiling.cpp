// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_vae_decoder_tiling.hpp"

#include "module_genai/module_factory.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/utils/blend_utils.hpp"
#include "utils.hpp"

#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/multiply.hpp"
#include "module_genai/utils/profiler.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(VAEDecoderTilingModule);

void VAEDecoderTilingModule::print_static_config() {
    std::cout << R"(
  vae_decoder_tiling:
    type: "VAEDecoderTilingModule"
    device: "CPU"
    inputs:
      - name: "latent"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "latents"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "image"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
      - name: "images"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
    params:
      tile_overlap_factor: "0.25"   # [Optional] float, default is 0.25
      sample_size: "1024"           # [Optional] int, tiling size. default is 1024, 
      model_path: "model"
      sub_module_name: "sub_modules: name"  # sub-pipeline module name

    )" << std::endl;
}

VAEDecoderTilingModule::VAEDecoderTilingModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    m_model_type = to_diffusion_model_type(desc->model_type);
    if (m_model_type != DiffusionModelType::ZIMAGE) {
        GENAI_ERR("TransformerModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
        return;
    }
    if (!initialize()) {
        GENAI_ERR("Failed to initiate TransformerModule");
    }
}

VAEDecoderTilingModule::~VAEDecoderTilingModule() {}

bool VAEDecoderTilingModule::init_tile_params(const std::filesystem::path& model_path) {
    auto factor = get_optional_param("tile_overlap_factor");
    if (!factor.empty()) {
        m_tile_overlap_factor = std::stof(factor);
    }

    bool have_sample_size = false;
    auto sample_size = get_optional_param("sample_size");
    if (!sample_size.empty()) {
        m_sample_size = std::stoi(sample_size);
        m_tile_sample_min_size = m_sample_size;
        have_sample_size = true;
    }


    auto config_path = model_path / "vae_decoder/config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream vae_config(config_path);
        nlohmann::json parsed = nlohmann::json::parse(vae_config);
        if (!have_sample_size) {
            ov::genai::utils::read_json_param(parsed, "sample_size", m_sample_size);
            m_tile_sample_min_size = m_sample_size;
        }

        // get block_out_channels: "block_out_channels": [128,256,512,512]
        std::vector<int> block_out_channels;
        ov::genai::utils::read_json_param(parsed, "block_out_channels", block_out_channels);

        // m_tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        m_tile_latent_min_size = m_sample_size / std::pow(2, block_out_channels.size() - 1);
    } else {
        OPENVINO_ASSERT(false,
                        "VAEDecoderTilingModule[" + module_desc->name + "]: vae_decoder config file not found at " +
                            config_path.string());
    }

    return true;
}

bool VAEDecoderTilingModule::init_post_process() {
    std::string device = module_desc->device.empty() ? "CPU" : module_desc->device;

    // Construct post-processing OV model here.
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, -1, -1});
    auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
    auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
    auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(input, constant_0_5);
    auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
    auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
    auto multiplied = std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
    auto result = std::make_shared<ov::op::v0::Result>(multiplied);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.output().postprocess().convert_element_type(ov::element::u8);
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");
    ppp.build();

    auto compiled_model = ov::genai::utils::singleton_core().compile_model(model, device);
    pp_infer_request = compiled_model.create_infer_request();

    return true;
}

bool VAEDecoderTilingModule::initialize() {
    const auto& params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
    init_tile_params(model_path);

    auto it_sub_module_name = module_desc->params.find("sub_module_name");
    if (it_sub_module_name == params.end()) {
        GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name + "]: 'sub_module_name' not found in params");
        return false;
    }

    m_sub_pipeline_impl = init_sub_pipeline(it_sub_module_name->second, pipeline_desc, module_desc);
    if (!m_sub_pipeline_impl) {
        return false;
    }

    if (!init_post_process()) {
        return false;
    }

    m_slice_infer_request = tensor_utils::init_slice_request(module_desc->device.empty() ? "CPU" : module_desc->device);

    return true;
}

void VAEDecoderTilingModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    std::vector<ov::Tensor> latents;
    if (this->inputs.find("latent") != this->inputs.end()) {
        latents.push_back(this->inputs["latent"].data.as<ov::Tensor>());
    } else if (this->inputs.find("latents") != this->inputs.end()) {
        if (this->inputs["latents"].data.is<ov::Tensor>()) {
            latents.push_back(this->inputs["latents"].data.as<ov::Tensor>());
        } else {
            latents = this->inputs["latents"].data.as<std::vector<ov::Tensor>>();
        }
    } else {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: 'latent' or 'latents' input not found");
        return;
    }

    // Process batch of latents
    std::vector<ov::Tensor> output_latents;
    for (const auto& latent : latents) {
        ov::Tensor cur_latent = latent;
        if (latent.get_shape().size() == 3u) {
            // unsqueeze batch dimension
            cur_latent = ov::Tensor(latent.get_element_type(),
                                    ov::Shape{1, latent.get_shape()[0], latent.get_shape()[1], latent.get_shape()[2]},
                                    latent.data());
        }
        OPENVINO_ASSERT(cur_latent.get_shape().size() == 4u,
                        "VAEDecoderTilingModule[" + module_desc->name + "]: cur_latent tensor must be 4D. Got shape: " +
                            ov::genai::module::tensor_utils::shape_to_string(cur_latent.get_shape()));

        ov::Tensor output_latent;
        if (m_enable_tiling &&
            (cur_latent.get_shape()[3] > m_tile_latent_min_size || cur_latent.get_shape()[2] > m_tile_latent_min_size)) {
            // Tiling decode
            tile_decode_4d(cur_latent, output_latent);
        } else {
            // Non-tiling decode
            GENAI_WARN("VAEDecoderTilingModule[" + module_desc->name + "]: Latent size w,h [" +
                       std::to_string(cur_latent.get_shape()[3]) + "," + std::to_string(cur_latent.get_shape()[2]) +
                       "] is smaller than tile size[" + std::to_string(m_tile_latent_min_size) +
                       "], using non-tiling decode.");
            output_latent = decoder(cur_latent);
        }

        // Post-process
        pp_infer_request.set_input_tensor(output_latent);
        {
            PROFILE(pm, "post-process infer");
            pp_infer_request.infer();
        }
        
        
        output_latent = pp_infer_request.get_output_tensor();
        ov::Tensor pp_out_tensor = ov::Tensor(output_latent.get_element_type(), output_latent.get_shape());
        output_latent.copy_to(pp_out_tensor);

        output_latents.push_back(pp_out_tensor);
    }

    if (output_latents.size() == 1) {
        this->outputs["image"].data = output_latents[0];
    } else {
        this->outputs["images"].data = output_latents;
    }
}

ov::Tensor VAEDecoderTilingModule::decoder(const ov::Tensor& tile) {
    ov::AnyMap inputs;
    inputs["latents"] = tile;

    m_sub_pipeline_impl->generate(inputs);

    // Retrieve output tensor from sub-pipeline
    ov::Any output = m_sub_pipeline_impl->get_output("image");
    if (output.is<ov::Tensor>()) {
        auto output_tensor = output.as<ov::Tensor>();
        auto clone_tensor = ov::Tensor(output_tensor.get_element_type(), output_tensor.get_shape());
        output_tensor.copy_to(clone_tensor);
        return clone_tensor;
    }

    GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name +
              "]: Sub-pipeline output 'image' is not of type ov::Tensor");
    return ov::Tensor();
}

void VAEDecoderTilingModule::tile_decode_4d(const ov::Tensor& latent, ov::Tensor& output_latent) {
    // Tiling decode implementation
    size_t overlap_size = m_tile_latent_min_size * (1 - m_tile_overlap_factor);
    size_t blend_extent = m_tile_sample_min_size * m_tile_overlap_factor;
    size_t row_limit = m_tile_sample_min_size - blend_extent;

    size_t height = latent.get_shape()[2];
    size_t width = latent.get_shape()[3];

    std::vector<std::vector<ov::Tensor>> rows;
    for (size_t h = 0; h < height; h += overlap_size) {
        const size_t h_start = h;
        const size_t h_end = std::min(h + m_tile_latent_min_size, height);
        std::vector<ov::Tensor> row;
        for (size_t w = 0; w < width; w += overlap_size) {
            const size_t w_start = w;
            const size_t w_end = std::min(w + m_tile_latent_min_size, width);

            // Get ROI tile from latent
            ov::Tensor tile = ov::genai::module::tensor_utils::slice_tensor(
                latent,
                {0, 0, h_start, w_start},
                {latent.get_shape()[0], latent.get_shape()[1], h_end, w_end});

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
                blend_utils::blend_v_4d(rows[i - 1][j], tile, blend_extent);
            }
            if (j > 0) {
                blend_utils::blend_h_4d(rows[i][j - 1], tile, blend_extent);
            }
            const auto dst_shape = tile.get_shape();
            result_row.push_back(tensor_utils::slice_tensor(
                tile,
                {0, 0, 0, 0},
                {dst_shape[0], dst_shape[1], std::min(dst_shape[2], row_limit), std::min(dst_shape[3], row_limit)}));
        }
        result_rows.push_back(tensor_utils::concat_tensors(result_row, 3));
    }

    // Combine result_rows into output_latent (not implemented here)
    output_latent = tensor_utils::concat_tensors(result_rows, 2);
}


}  // namespace module
}  // namespace genai
}  // namespace ov