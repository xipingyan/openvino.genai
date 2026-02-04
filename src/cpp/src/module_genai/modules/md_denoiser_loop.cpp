// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_denoiser_loop.hpp"

#include "module_genai/module_factory.hpp"
#include "module_genai/transformer_config.hpp"
#include "utils.hpp"
#include "image_generation/schedulers/z_image_flow_match_euler_discrete.hpp"
#include "json_utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "image_generation/numpy_utils.hpp"
#include <fstream>
#include "module_genai/utils/profiler.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(DenoiserLoopModule);

void DenoiserLoopModule::print_static_config() {
    std::cout << R"(
  denoiser_loop:
    type: "DenoiserLoopModule"
    device: "GPU"
    inputs:
      - name: "latents"                                    
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embed"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embeds"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embed_negative"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embeds_negative"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "num_inference_steps"                        # [optional]
        type: "Int"                                        # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "guidance_scale"                             # [optional]
        type: "Float"                                      # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "latents"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
    params:
      model_path: "model"
    )" << std::endl;
}

DenoiserLoopModule::DenoiserLoopModule(const IBaseModuleDesc::PTR& desc,
                                                   const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc), m_cfg_truncation(1.0f), m_cfg_normalization(false) {
    m_model_type = to_diffusion_model_type(desc->model_type);
    if (m_model_type == DiffusionModelType::UNKNOWN) {
        GENAI_ERR("TransformerModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
        return;
    }
    if (!initialize()) {
        GENAI_ERR("Failed to initiate TransformerModule");
    }
}

DenoiserLoopModule::~DenoiserLoopModule() {}

bool DenoiserLoopModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
    auto transformer_model_path = model_path / "transformer/openvino_model.xml";
    if (m_model_type == DiffusionModelType::ZIMAGE) {
        m_scheduler = std::make_shared<ZImageFlowMatchEulerDiscreteScheduler>(model_path / "scheduler/scheduler_config.json", module_desc->device);
    } else if (m_model_type == DiffusionModelType::WAN_2_1) {
        // Force to use CPU for scheduler, since the Inverse(opset14) is not supported on GPU
        m_scheduler = std::make_shared<UniPCMultistepScheduler>(model_path / "scheduler/scheduler_config.json", "CPU");
        transformer_model_path = model_path / "transformer/transformer.xml";
    } else {
        OPENVINO_THROW("Unsupported '", module_desc->model_type, "' Transformer model type");
    }
    if (!std::filesystem::exists(transformer_model_path)) {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: model file not found at " + transformer_model_path.string());
        return false;
    }
    auto model = utils::singleton_core().read_model(transformer_model_path);
    m_compiled_model = utils::singleton_core().compile_model(model, module_desc->device, ov::AnyMap{});

    m_ov_unsqueeze = OV_Unsqueeze::create(m_compiled_model.get_context(), 2);
    OPENVINO_ASSERT(m_ov_unsqueeze != nullptr, "Failed to create OV_Unsqueeze instance");

    m_request = m_compiled_model.create_infer_request();
    return true;
}

void DenoiserLoopModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    std::vector<ov::Tensor> prompt_embeds;
    std::vector<ov::Tensor> negative_prompt_embeds;
    ov::Tensor latents;
    if (this->inputs.find("latents") != this->inputs.end()) {
        latents = this->inputs["latents"].data.as<ov::Tensor>();
    } else {
        OPENVINO_THROW("input latents should not be empty");
    }
    if (this->inputs.find("prompt_embed") != this->inputs.end()) {
        prompt_embeds.push_back(this->inputs["prompt_embed"].data.as<ov::Tensor>());
        m_is_multi_prompts = false;
    } else if (this->inputs.find("prompt_embeds") != this->inputs.end()) {
        prompt_embeds = this->inputs["prompt_embeds"].data.as<std::vector<ov::Tensor>>();
        m_is_multi_prompts = true;
    } else {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: 'prompt_embed' or 'prompt_embeds' input not found");
        return;
    }
    if (this->inputs.find("prompt_embed_negative") != this->inputs.end()) {
        negative_prompt_embeds.push_back(this->inputs["prompt_embed_negative"].data.as<ov::Tensor>());
    } else if (this->inputs.find("prompt_embeds_negative") != this->inputs.end()) {
        negative_prompt_embeds = this->inputs["prompt_embeds_negative"].data.as<std::vector<ov::Tensor>>();
    } else {
        // empty negative prompt embeds
    }

    if (module_desc->device == "GPU") {
        PROFILE(p, "DenoiserLoopModule::run - Copy latents to RemoteTensor");
        auto context = m_compiled_model.get_context();
        auto latents_remote_tensor = context.create_tensor(latents.get_element_type(), latents.get_shape());
        latents_remote_tensor.copy_from(latents);
        latents = latents_remote_tensor;
    }

    ImageGenerationConfig generation_config{};
    if (exists_input("num_inference_steps")) {
        generation_config.num_inference_steps = this->inputs["num_inference_steps"].data.as<int>();
    } else {
        generation_config.num_inference_steps = 10;
    }
    // TODO: temporary guidance_scale is fixed to 0.0
    generation_config.guidance_scale = 0.0f;
    if (exists_input("cfg_truncation")) {
        m_cfg_truncation = this->inputs["cfg_truncation"].data.as<float>();
    } else {
        m_cfg_truncation = 1.0f;
    }
    if (exists_input("cfg_normalization")) {
        std::string cfg_normalization_str = this->inputs["cfg_normalization"].data.as<std::string>();
        std::transform(cfg_normalization_str.begin(),
                       cfg_normalization_str.end(),
                       cfg_normalization_str.begin(),
                       [](unsigned char c) {
                           return (char)std::tolower(c);
                       });
        if (cfg_normalization_str == "true" || cfg_normalization_str == "1") {
            m_cfg_normalization = true;
        } else {
            m_cfg_normalization = false;
        }
    } else {
        m_cfg_normalization = false;
    }
    ov::Tensor output_latents;
    if (m_model_type == DiffusionModelType::ZIMAGE) {
        output_latents = run(latents, prompt_embeds, negative_prompt_embeds, generation_config);
        
    } else {
        int num_inference_steps = 10;
        if (exists_input("num_inference_steps")) {
            num_inference_steps = this->inputs["num_inference_steps"].data.as<int>();
        }
        float guidance_scale = 7.5f;
        if (exists_input("guidance_scale")) {
            guidance_scale = this->inputs["guidance_scale"].data.as<float>();
        }
        output_latents = run(latents, prompt_embeds, negative_prompt_embeds, num_inference_steps, guidance_scale);
    }

    if (output_latents.is<ov::RemoteTensor>()) {
        PROFILE(p, "DenoiserLoopModule::run - Copy output_latents to HostTensor");
        ov::Tensor host_tensor(output_latents.get_element_type(), output_latents.get_shape());
        output_latents.copy_to(host_tensor);
        output_latents = host_tensor;
    }

    this->outputs["latents"].data = output_latents;
}

// Image generation
ov::Tensor DenoiserLoopModule::run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config) {
    // TODO: Add multi image support and negative prompt support
    size_t batch_size = prompt_embeds.size();
    
    std::vector<ov::Tensor> processed_embeds = prompt_embeds;
    std::vector<ov::Tensor> processed_negative_embeds = negative_prompt_embeds;
    if (generation_config.num_images_per_prompt > 1) {
        std::vector<ov::Tensor> expanded_prompt_embeds;
        for (const auto& embed : prompt_embeds) {
            for (int i = 0; i < generation_config.num_images_per_prompt; ++i) {
                expanded_prompt_embeds.push_back(embed);
            }
        }
        processed_embeds.assign(expanded_prompt_embeds.begin(), expanded_prompt_embeds.end());
        if (generation_config.guidance_scale > 1.0f) {
            std::vector<ov::Tensor> expanded_negative_embeds;
            for (const auto& embed : negative_prompt_embeds) {
                for (int i = 0; i < generation_config.num_images_per_prompt; ++i) {
                    expanded_negative_embeds.push_back(embed);
                }
            }
            processed_negative_embeds.assign(expanded_negative_embeds.begin(), expanded_negative_embeds.end());
        }
    }

    auto actual_batch_size = batch_size * generation_config.num_images_per_prompt;
    auto image_seq_len = (latents.get_shape()[2] / 2) * (latents.get_shape()[3] / 2);
    std::dynamic_pointer_cast<ZImageFlowMatchEulerDiscreteScheduler>(std::get<std::shared_ptr<IScheduler>>(m_scheduler))->set_sigma_min(0.0f);
    std::get<std::shared_ptr<IScheduler>>(m_scheduler)->set_timesteps(
        image_seq_len,
        generation_config.num_inference_steps,
         generation_config.strength);
    std::vector<float> timesteps = std::get<std::shared_ptr<IScheduler>>(m_scheduler)->get_float_timesteps();
    ov::Tensor prompt_tensor = tensor_utils::stack(processed_embeds);
    std::vector<ov::Tensor> all_prompt_embeds;
    ov::Tensor all_prompt_tensor;
    if (generation_config.guidance_scale > 1.0f) {
        all_prompt_embeds.reserve(processed_embeds.size() + processed_negative_embeds.size());
        all_prompt_embeds.insert(all_prompt_embeds.end(), processed_embeds.begin(), processed_embeds.end());
        all_prompt_embeds.insert(all_prompt_embeds.end(), processed_negative_embeds.begin(), processed_negative_embeds.end());
        all_prompt_tensor = tensor_utils::stack(all_prompt_embeds);
    }
    for (size_t i = 0; i < timesteps.size(); i++) {
        timesteps[i] = (1000.0f - timesteps[i]) / 1000.0f;
    }
    for (size_t inference_step = 0; inference_step < timesteps.size(); inference_step++) {
        float t_norm = timesteps[inference_step];
        float current_guidance_scale = generation_config.guidance_scale;
        if (generation_config.guidance_scale > 1.0f &&
            (m_cfg_truncation < 1.0f || std::fabs(m_cfg_truncation - 1.0f) < 1e-6)) {
            if (t_norm > m_cfg_truncation) {
                current_guidance_scale = 0.0f;
            }
        }
        bool apply_cfg = generation_config.guidance_scale > 1.0f && current_guidance_scale > 0.0f;
        if (apply_cfg) {
            ov::Tensor tmp_latents = numpy_utils::repeat(latents, 2);
            std::vector<ov::Tensor> prompt_embeds;
            ov::Tensor timestep(ov::element::f32, {1}, &timesteps[inference_step]);
            timestep = numpy_utils::repeat(timestep, 2);
            tmp_latents = tensor_utils::unsqueeze(tmp_latents, 2);
            m_request.set_tensor("hidden_states", tmp_latents);
            m_request.set_tensor("timestep", timestep);
            m_request.set_tensor("encoder_hidden_states", all_prompt_tensor);
        } else {
            // ov::Tensor unsqueezed_latents = tensor_utils::unsqueeze(latents, 2);
            ov::Tensor unsqueezed_latents = m_ov_unsqueeze->infer(latents);

            ov::Tensor timestep(ov::element::f32, {1}, &timesteps[inference_step]);
            m_request.set_tensor("hidden_states", unsqueezed_latents);
            m_request.set_tensor("timestep", timestep);
            m_request.set_tensor("encoder_hidden_states", prompt_tensor);
        }

        // Set output to remote tensor to avoid unnecessary data copy
        if (module_desc->device == "GPU") {
            auto context = m_compiled_model.get_context();
            ov::Tensor remote_output_tensor = context.create_tensor(m_request.get_output_tensor().get_element_type(),
                                                                    m_request.get_output_tensor().get_shape());
            m_request.set_output_tensor(0, remote_output_tensor);
        }

        {
            PROFILE(pm, "infer");
            m_request.infer();
        }
        
        ov::Tensor model_output = m_request.get_output_tensor();
        ov::Tensor noise_pred;
        if (apply_cfg) {
            std::vector<ov::Tensor> noise_preds;
            for (int j = 0; j < actual_batch_size; j++) {
                ov::Shape pred_shape = model_output.get_shape();
                pred_shape.erase(pred_shape.begin());
                ov::Tensor pred(model_output.get_element_type(), pred_shape);
                float *pred_data = pred.data<float>();
                const float *model_output_data = model_output.data<const float>();
                for (size_t k = 0; k < pred.get_size(); k++) {
                    float positive_pred = model_output_data[j * pred.get_size() + k];
                    float negative_pred = model_output_data[(actual_batch_size + j) * pred.get_size() + k];
                    pred_data[k] = positive_pred + current_guidance_scale * (positive_pred - negative_pred);
                }
                if (m_cfg_normalization) {
                    float orig_positive_norm = tensor_utils::calculate_l2_norm(model_output, j * pred.get_size(), (j + 1) * pred.get_size());
                    float new_positive_norm = tensor_utils::calculate_l2_norm(pred, 0, pred.get_size());
                    float max_new_norm = orig_positive_norm * static_cast<float>(m_cfg_normalization);
                    if (new_positive_norm > max_new_norm) {
                        for (size_t k = 0; k < pred.get_size(); k++) {
                            pred_data[k] = pred_data[k] * (max_new_norm / new_positive_norm);
                        }
                    }
                }
                noise_preds.push_back(pred);
            }
            noise_pred = tensor_utils::stack(noise_preds);
        } else {
            noise_pred = m_request.get_output_tensor();
        }

        auto scheduler_step_result = std::get<std::shared_ptr<IScheduler>>(m_scheduler)->step(
            noise_pred, latents, inference_step, generation_config.generator);
        latents = scheduler_step_result["latent"];
    }
    return latents;
}

ov::Tensor DenoiserLoopModule::run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        int num_inference_steps,
        float guidance_scale) {
    ov::Tensor prompt_tensor = tensor_utils::stack(prompt_embeds);
    std::optional<ov::Tensor> negative_prompt_tensor = std::nullopt;
    if (!negative_prompt_embeds.empty()) {
        negative_prompt_tensor = tensor_utils::stack(negative_prompt_embeds);
    }
    std::get<std::shared_ptr<UniPCMultistepScheduler>>(m_scheduler)->set_timesteps(num_inference_steps);
    auto timesteps = std::get<std::shared_ptr<UniPCMultistepScheduler>>(m_scheduler)->get_timesteps();
    ov::Tensor noise_pred(latents.get_element_type(),
                         latents.get_shape());
    ov::Tensor noise_uncond(latents.get_element_type(),
                         latents.get_shape());
    auto noise_pred_data = noise_pred.data<float>();
    auto noise_uncond_data = noise_uncond.data<float>();
    for (size_t i = 0; i < timesteps.size(); i++) {
        int64_t t = timesteps[i];
        ov::Tensor timestep(ov::element::f32, {latents.get_shape()[0]});
        auto *timestep_data = timestep.data<float>();
        for (size_t j = 0; j < timestep.get_size(); j++) {
            timestep_data[j] = static_cast<float>(t);
        }
        m_request.set_tensor("hidden_states", latents);
        m_request.set_tensor("timestep", timestep);
        m_request.set_tensor("encoder_hidden_states", prompt_tensor);
        m_request.set_output_tensor(0, noise_pred);
        m_request.infer();

        if (guidance_scale > 1.0f && negative_prompt_tensor.has_value()) {
            m_request.set_tensor("hidden_states", latents);
            m_request.set_tensor("timestep", timestep);
            m_request.set_tensor("encoder_hidden_states", negative_prompt_tensor.value());
            m_request.set_output_tensor(0, noise_uncond);
            m_request.infer();

            for (size_t j = 0; j < noise_pred.get_size(); j++) {
                noise_pred_data[j] = noise_uncond_data[j] + guidance_scale * (noise_pred_data[j] - noise_uncond_data[j]);
            }
        }

        latents = std::get<std::shared_ptr<UniPCMultistepScheduler>>(m_scheduler)->step(
            noise_pred, t, latents)["prev_sample"];
    }
    return latents;
}

OV_Unsqueeze::OV_Unsqueeze(const std::string& device, const int32_t& axes) {
    m_device = device;
    m_axes = axes;
    // Initialize OpenVINO model for Unsqueeze operation
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    input->set_friendly_name("input");
    // auto axis = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {axes});
    axis_const->set_friendly_name("axes");

    auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(input, axis_const);
    unsqueezed->set_friendly_name("unsqueeze");
    auto result = std::make_shared<ov::op::v0::Result>(unsqueezed);
    result->set_friendly_name("output");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    auto compiled_model = ov::genai::utils::singleton_core().compile_model(model, device);
    m_context = compiled_model.get_context();
    m_request = compiled_model.create_infer_request();
}

OV_Unsqueeze::OV_Unsqueeze(const ov::RemoteContext& context, const int32_t& axes) {
    m_axes = axes;
    m_context = context;

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    input->set_friendly_name("input");
    auto axis_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {axes});
    axis_const->set_friendly_name("axes");

    auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(input, axis_const);
    unsqueezed->set_friendly_name("unsqueeze");
    auto result = std::make_shared<ov::op::v0::Result>(unsqueezed);
    result->set_friendly_name("output");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    // Compile the helper model on the SAME remote context as the main pipeline.
    // This ensures remote input stays on-device and allows remote output allocation.
    auto compiled_model = ov::genai::utils::singleton_core().compile_model(model, context);
    m_request = compiled_model.create_infer_request();
}

bool OV_Unsqueeze::shape_changed(const ov::Shape& in_shape) {
    if (m_last_shape != in_shape) {
        m_last_shape = in_shape;
        return true;
    }
    return false;
}

ov::Tensor OV_Unsqueeze::infer(const ov::Tensor& input_tensor) {
    std::cout << "OV_Unsqueeze::infer input_tensor is remotetensor: " << input_tensor.is<ov::RemoteTensor>() << std::endl;
    m_request.set_input_tensor(0, input_tensor);

    if (input_tensor.is<ov::RemoteTensor>() && shape_changed(input_tensor.get_shape())) {
        ov::Shape out_shape = input_tensor.get_shape();
        const int64_t out_rank = static_cast<int64_t>(out_shape.size()) + 1;
        int64_t axis = static_cast<int64_t>(m_axes);
        if (axis < 0) {
            axis += out_rank;
        }
        OPENVINO_ASSERT(axis >= 0 && axis <= static_cast<int64_t>(out_shape.size()),
                        "OV_Unsqueeze: invalid axis=", m_axes, " for input rank=", out_shape.size());
        out_shape.insert(out_shape.begin() + static_cast<size_t>(axis), 1);

        // Allocate output on the same remote context so output remains a RemoteTensor.
        ov::Tensor remote_out = m_context.create_tensor(input_tensor.get_element_type(), out_shape);
        m_request.set_output_tensor(0, remote_out);
    }

    m_request.infer();
    auto output = m_request.get_output_tensor();
    std::cout << "OV_Unsqueeze::infer output is remotetensor: " << output.is<ov::RemoteTensor>() << std::endl;
    return output;
}

}  // namespace module
}  // namespace genai
}  // namespace ov