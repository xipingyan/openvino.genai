#include "md_vision_encoder.hpp"
#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <vector>
#include "circular_buffer_queue.hpp"
#include "module_genai/module_base.hpp"
#include "utils.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vl_sdpa_transformations.hpp"

namespace ov {
namespace genai {
namespace module {

void VisionEncoderModule::print_static_config() {
    std::cout << R"(
  vision_encoder:
    type: "VisionEncoderModule"
    device: "GPU"
    inputs:
      - name: "preprocessed_image"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "source_size"
        type: "VecInt"                                     # Support DataType: [VecInt]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "image_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
      - name: "video_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
    params:
      model_path: "model"
    )" << std::endl;
}

VisionEncoderModule::VisionEncoderModule(const IBaseModuleDesc::PTR &desc) : IBaseModule(desc) {
    if (!initialize()) {
        std::cerr << "Failed to initiate VisionEncoderModule" << std::endl;
    }
}

bool VisionEncoderModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        std::cerr << "VisionEncoderModule[" << module_desc->name << "]: 'model_path' not found in params" << std::endl;
        return false;
    }

    std::filesystem::path model_path = it_path->second;

    if (!std::filesystem::exists(model_path / "openvino_vision_embeddings_merger_model.xml")) {
        std::cerr << "VisionEncoderModule[" << module_desc->name << "]: model file not found at "
                  << (model_path / "openvino_vision_embeddings_merger_model.xml") << std::endl;
        return false;
    }

    auto model = utils::singleton_core().read_model(
        model_path / "openvino_vision_embeddings_merger_model.xml");
    utils::request_vl_sdpa_transformations(model);

    auto compiled_model = utils::singleton_core().compile_model(
        model, 
        module_desc->device, {});
    m_with_cu_seqlens_input = utils::check_vl_sdpa_transformations(compiled_model);
    ov::genai::utils::print_compiled_model_properties(compiled_model,
        m_with_cu_seqlens_input ? "VLM vision embeddings merger model with VLSDPA optimization ENABLED" :
        "VLM vision embeddings merger model with VLSDPA optimization DISABLED");

    m_ireq_queue_vision_embeddings_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }
    );
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_path, "config.json");
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(model_path, "preprocessor_config.json");
    m_merge_length = std::pow(m_processor_config.merge_size, 2);
    return true;
}

void VisionEncoderModule::run() {
    prepare_inputs();
    if (this->inputs.find("preprocessed_image") == this->inputs.end() || this->inputs["preprocessed_image"].data == nullptr) {
        std::cerr << "VisionEncoderModule[" << module_desc->name << "]: 'preprocessed_image' input not found" << std::endl;
        return;
    }
    if (this->inputs.find("source_size") == this->inputs.end() || this->inputs["source_size"].data.as<std::vector<int>>().empty()) {
        std::cerr << "VisionEncoderModule[" << module_desc->name << "]: 'source_size' input not found" << std::endl;
        return;
    }
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
    ov::Tensor image_embedding;
    ov::Tensor video_embedding;
    EncodedImage encoded;
    encoded.resized_source_size.height = this->inputs["source_size"].data.as<std::vector<int>>()[0];
    encoded.resized_source_size.width = this->inputs["source_size"].data.as<std::vector<int>>()[1];
    encoded.resized_source = this->inputs["preprocessed_image"].data.as<ov::Tensor>();
    std::tie(video_embedding, image_embedding) = embed(encoded);
    
    this->outputs["image_embedding"].data = image_embedding;
    this->outputs["video_embedding"].data = video_embedding;
}


std::pair<ov::Tensor, ov::Tensor> VisionEncoderModule::embed(const EncodedImage &image) {
    OPENVINO_ASSERT(m_ireq_queue_vision_embeddings_merger, "VisionEncoderModule is not initialized. Call initialize() first.");

    std::vector<size_t> images_sequence = {0};
    auto [reordered_image_embeds, reordered_images_grid_thw] = qwen2_vl_utils::reorder_image_embeds_and_grid_thw({image}, images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] = qwen2_vl_utils::reorder_video_embeds_and_grid_thw({}, {});

    ov::Tensor concatenated_embeds = qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);

    std::vector<std::array<size_t, 3>> reordered_vision_grid_thw;
    reordered_vision_grid_thw.reserve(reordered_videos_grid_thw.size() + reordered_images_grid_thw.size());
    reordered_vision_grid_thw.insert(reordered_vision_grid_thw.end(), reordered_videos_grid_thw.begin(), reordered_videos_grid_thw.end());
    reordered_vision_grid_thw.insert(reordered_vision_grid_thw.end(), reordered_images_grid_thw.begin(), reordered_images_grid_thw.end());

    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_vision_grid_thw);

    auto [window_index, cu_window_seqlens] = qwen2_5_vl_utils::get_window_index(
        reordered_vision_grid_thw,
        m_processor_config,
        m_vlm_config
    );

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    if (m_with_cu_seqlens_input) {
        ov::Tensor cu_seq_lens = qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw);
        ov::Tensor t_cu_window_seqlens = qwen2_5_vl_utils::get_cu_window_seqlens(cu_window_seqlens);
        vision_embeddings_merger.set_tensor("cu_seq_lens", cu_seq_lens);
        vision_embeddings_merger.set_tensor("cu_window_seqlens", t_cu_window_seqlens);
    }
    else {
        ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw);
        size_t hidden_states_size = attention_mask.get_shape().at(1);
        ov::Tensor window_attention_mask = qwen2_5_vl_utils::get_window_attention_mask(hidden_states_size, cu_window_seqlens);
        vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
        vision_embeddings_merger.set_tensor("window_attention_mask", window_attention_mask);
    }
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.set_tensor("window_index", window_index);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    auto out_vision_shape = processed_vision_embeds.get_shape();

    // Split Video and Image's features.
    auto video_fea_num = calc_vec_tokens_num(reordered_videos_grid_thw);
    auto image_fea_num = calc_vec_tokens_num(reordered_images_grid_thw);
    size_t video_fea_count = 0;
    if ((video_fea_num + image_fea_num) != 0) {
        video_fea_count = out_vision_shape.at(0) * video_fea_num / (video_fea_num + image_fea_num);
    }

    ov::Shape video_fea_shape = ov::Shape({video_fea_count, out_vision_shape.at(1)});
    ov::Tensor res_video = ov::Tensor(processed_vision_embeds.get_element_type(), video_fea_shape);
    OPENVINO_ASSERT(processed_vision_embeds.get_byte_size() >= res_video.get_byte_size(), "Vision embeds size should >= video embeds size.");
    std::memcpy(res_video.data(), processed_vision_embeds.data(), res_video.get_byte_size());

    ov::Shape image_fea_shape({out_vision_shape.at(0) - video_fea_count, out_vision_shape.at(1)});
    ov::Tensor res_image(processed_vision_embeds.get_element_type(), image_fea_shape);
    OPENVINO_ASSERT(processed_vision_embeds.get_byte_size() == res_image.get_byte_size() + res_video.get_byte_size(),
                    "Vision embeds size should == image + video embeds size.");
    std::memcpy(res_image.data(),
                reinterpret_cast<uint8_t*>(processed_vision_embeds.data()) + res_video.get_byte_size(),
                res_image.get_byte_size());
    return {res_video, res_image};
}

ov::Tensor VisionEncoderModule::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) {
    const size_t spatial_merge_size = m_processor_config.merge_size;

    std::vector<std::vector<size_t>> all_pos_ids;
    size_t max_grid_size = 0;

    for (const auto& grid_thw : grids_thw) {
        size_t t = grid_thw.at(0);
        size_t h = grid_thw.at(1);
        size_t w = grid_thw.at(2);

        max_grid_size = std::max({max_grid_size, h, w});

        // According to spatial merge size, create height & width position IDs
        std::vector<size_t> hpos_ids;
        std::vector<size_t> wpos_ids;
        size_t h_blocks = h / spatial_merge_size;
        size_t w_blocks = w / spatial_merge_size;
        hpos_ids.reserve(h * w);
        wpos_ids.reserve(h * w);

        for (size_t hb = 0; hb < h_blocks; ++hb) {
            for (size_t wb = 0; wb < w_blocks; ++wb) {
                for (size_t hs = 0; hs < spatial_merge_size; ++hs) {
                    for (size_t ws = 0; ws < spatial_merge_size; ++ws) {
                        hpos_ids.push_back(hb * spatial_merge_size + hs);
                        wpos_ids.push_back(wb * spatial_merge_size + ws);
                    }
                }
            }
        }

        // Stack and repeat for each t
        for (size_t i = 0; i < t; ++i) {
            for (size_t j = 0; j < hpos_ids.size(); ++j) {
                all_pos_ids.push_back({hpos_ids[j], wpos_ids[j]});
            }
        }
    }

    // Calculate rotary embeddings for max_grid_size
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    const size_t dim = vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    const float theta = 10000.0f;
    
    std::vector<float> inv_freq(dim / 2);
    for (size_t i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(dim / 2));
    }

    std::vector<std::vector<float>> freqs(max_grid_size);
    for (size_t i = 0; i < max_grid_size; ++i) {
        freqs[i].resize(dim / 2);
        for (size_t j = 0; j < dim / 2; ++j) {
            freqs[i][j] = static_cast<float>(i) * inv_freq[j];
        }
    }

    ov::Tensor rotary_pos_emb(ov::element::f32, {all_pos_ids.size(), dim});
    float* output_data = rotary_pos_emb.data<float>();

    for (size_t i = 0; i < all_pos_ids.size(); ++i) {
        const auto& pos = all_pos_ids.at(i);
        size_t h_idx = pos.at(0);
        size_t w_idx = pos.at(1);
        std::copy_n(freqs[h_idx].begin(), dim / 2, output_data + i * dim);
        std::copy_n(freqs[w_idx].begin(), dim / 2, output_data + i * dim + dim / 2);
    }
    return rotary_pos_emb;
}

size_t VisionEncoderModule::calc_vec_tokens_num(const std::vector<std::array<size_t, 3UL>>& vec_grid_thw) const {
    size_t token_num = 0;
    for (auto grid_thw : vec_grid_thw) {
        token_num += calc_tokens_num(grid_thw[0], grid_thw[1], grid_thw[2]);
    }
    return token_num;
}

size_t VisionEncoderModule::calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const {
    return grid_t * grid_h * grid_w / m_merge_length;
}

}
}
}