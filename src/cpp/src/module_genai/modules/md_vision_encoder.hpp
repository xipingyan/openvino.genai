#pragma once

#include <yaml-cpp/yaml.h>
#include <memory>
#include <openvino/runtime/tensor.hpp>
#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"


namespace ov {
namespace genai {
namespace module {

class VisionEncoderModule : public IBaseModule {
protected:
    VisionEncoderModule() = delete;
    VisionEncoderModule(const IBaseModuleDesc::PTR &desc);

public:
    ~VisionEncoderModule() {}

    void run() override;

    using PTR = std::shared_ptr<VisionEncoderModule>;
    static PTR create(const IBaseModuleDesc::PTR &desc) {
        return PTR(new VisionEncoderModule(desc));
    }
    static void print_static_config();

private:
    bool initialize();
    std::pair<ov::Tensor, ov::Tensor> embed(const EncodedImage &image, const std::vector<int>& images_sequence);
    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw);
    size_t calc_vec_tokens_num(const std::vector<std::array<size_t, 3UL>>& vec_grid_thw) const;
    size_t calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const;

    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_merger;
    bool m_with_cu_seqlens_input { false };
    VLMConfig m_vlm_config;
    ProcessorConfig m_processor_config;
    size_t m_merge_length;
};

REGISTER_MODULE_CONFIG(VisionEncoderModule) ;

}
}
}