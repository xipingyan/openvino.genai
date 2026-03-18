#pragma once

#include <yaml-cpp/yaml.h>
#include <memory>
#include <openvino/runtime/tensor.hpp>
#include "module_genai/pipeline/module.hpp"
#include "module_genai/pipeline/module_type.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

#if defined(ENABLE_MODELING_PRIVATE)

#include "module_genai/modules/models/qwen3_5/qwen3_5config.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni.hpp"
#include "models/qwen3_omni/qwen3_omni_config.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

namespace ov::genai::module {

class VisionEncoderModule : public IBaseModule {
    DeclareModuleConstructor(VisionEncoderModule);

private:
    bool initialize();
    std::pair<ov::Tensor, ov::Tensor> embed(const EncodedImage &image, const std::vector<int>& images_sequence, const ov::Tensor& input_ids);
    Qwen3_5VisionEmbeddingResult embed(
        const std::vector<ov::Tensor> &pixel_values,
        const std::vector<ov::Tensor> &grid_thw,
        const std::vector<ov::Tensor> &pos_embeds,
        const std::vector<ov::Tensor> &rotary_cos,
        const std::vector<ov::Tensor> &rotary_sin,
        const ov::Tensor &input_ids,
        const ov::Tensor &attention_mask);
// #ifdef ENABLE_OPENVINO_NEW_ARCH
    // Qwen 3-Omni
    Qwen3OmniVisionEmbeddingResult embed(
        const ov::Tensor &input_ids,
        const ov::Tensor &attention_mask,
        std::optional<Qwen3OmniVisionInput> &vision_image_input,
        std::optional<Qwen3OmniVisionInput> &vision_video_input,
        std::optional<Qwen3OmniAudioInput> &audio_input);
// #endif
    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw);
    size_t calc_vec_tokens_num(const std::vector<std::array<size_t, 3UL>>& vec_grid_thw) const;
    size_t calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const;
    ov::Tensor create_position_ids(const ov::Tensor& input_ids_tensor,
                                   const std::vector<std::array<size_t, 3>>& images_grid_thw,
                                   const std::vector<size_t>& images_sequence,
                                   const size_t image_id,
                                   const std::vector<std::array<size_t, 3>>& videos_grid_thw,
                                   const std::vector<size_t>& videos_sequence,
                                   const size_t video_id,
                                   const int64_t vision_start_token_id,
                                   const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count);
    ov::Tensor build_vision_attention_mask(const ov::Tensor& grid_thw);

    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_request_queue;
    bool m_with_cu_seqlens_input { false };
    VLMConfig m_vlm_config;
    ProcessorConfig m_processor_config;
    size_t m_merge_length;

    ov::Tensor m_position_ids;
    int64_t m_rope_delta = 0;
    int64_t m_vision_start_token_id = 0;
    int64_t m_image_pad_token_id = 0;
    int64_t m_video_pad_token_id = 0;
// #ifdef ENABLE_OPENVINO_NEW_ARCH
    std::variant<modeling::models::Qwen3OmniProcessingConfig, modeling::models::Qwen3_5Config> m_config;
    std::optional<std::variant<modeling::models::Qwen3OmniInputPlanner, modeling::models::Qwen3_5InputPlanner>> m_input_planner;
// #endif
};

REGISTER_MODULE_CONFIG(VisionEncoderModule) ;

}   // namespace ov::genai::module

#endif  // ENABLE_MODELING_PRIVATE