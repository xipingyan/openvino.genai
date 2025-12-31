// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "continuous_batching/model_runner.hpp"
#include "sampling/sampler.hpp"
#include "continuous_batching/threaded_streamer.hpp"

#include "visual_language/continuous_batching_adapter.hpp"

namespace ov {
namespace genai {
namespace module {

enum class ModelInputType {
    TOKENS,
    EMBEDDINGS
};

/// @brief Unified generation module supporting multiple modalities
/// Supports: LLM, VLM
class LLMInferenceModule : public IBaseModule {
protected:
    LLMInferenceModule() = delete;
    LLMInferenceModule(const IBaseModuleDesc::PTR& desc);

public:
    ~LLMInferenceModule() {}

    void run() override;

    using PTR = std::shared_ptr<LLMInferenceModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new LLMInferenceModule(desc));
    }
    static void print_static_config();

private:
    
    bool initialize();
    bool load_generation_config(const std::string& config_path);
    std::vector<EncodedGenerationResult> generate(
        const std::vector<ov::Tensor>& input_ids,
        const std::vector<GenerationConfig>& sampling_params,
        const StreamerVariant& streamer,
        const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
        const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids_list = std::nullopt);
    bool has_non_finished_requests();
    void set_adapters(const std::optional<AdapterConfig>& adapters);
    void _prepare_rotation_data_storage(const SchedulerConfig& normalized_config, size_t embedding_size);
    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt);
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
    void step();
    void drop_requests();
    void stream_tokens(const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr, const GenerationHandle& handle);
    void _pull_awaiting_requests();
    void _register_step_cache_usage(float step_cache_usage);
    void _compute_cache_rotation_data(const std::vector<SequenceGroup::Ptr>& sequence_groups, const Scheduler::Output& scheduler_output);
    void _free_non_running_requests();
    void _maybe_evict_cache_blocks(const SchedulerConfig& sched_config, const Scheduler::Output& scheduler_output);
    void _fill_prompt_log_probs(std::vector<SequenceGroup::Ptr>& sequence_groups, ov::Tensor& logits);
    void _notify_requests_dropped_by_handle();

    ModelInputType m_model_input_type = ModelInputType::TOKENS;
    // Generation configurations
    ov::genai::GenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<ModelRunner> m_model_runner;
    std::optional<AdapterController> m_adapter_controller;
    std::shared_ptr<Sampler> m_sampler;
    // current requests to process
    std::vector<SequenceGroup::Ptr> m_requests;
    // requests added to the pipeline that will be added to m_requests in the next iteration
    std::vector<SequenceGroup::Ptr> m_awaiting_requests;
    // Mutex protecting access to m_awaiting_requests, so add_request and step methods can be called from different threads
    std::mutex m_awaiting_requests_mutex;
    size_t m_num_decoder_layers = 0;
    size_t m_block_size = 0;
    // Pre-allocated per-layer storages for the per-token cache re-rotation deltas used in cache eviction case
    std::vector<ov::Tensor> m_rotation_deltas_stores;

    std::map<size_t, std::vector<std::set<size_t>>> m_previous_evicted_block_logical_indices_per_sequence;
    std::map<size_t, size_t> m_previous_num_blocks_before_eviction_per_sequence;

    std::vector<std::map<size_t, std::vector<size_t>>> m_current_step_rotated_block_indices_per_sequence;
    std::vector<ov::Tensor> m_current_step_rotation_deltas;

    std::shared_ptr<ov::genai::CacheRotationCalculator> m_cache_rotation_calculator;
    static const size_t AVG_CACHE_USAGE_WINDOW_SIZE_IN_STEPS = 1000;
    std::deque<float> m_previous_step_cache_usages;
    std::map<size_t, CacheEvictionAlgorithm> m_seq_group_id_to_cache_eviction_algo_map;
    size_t m_batch_size = 0;
    bool m_is_validation_mode_enabled = false;

};

REGISTER_MODULE_CONFIG(LLMInferenceModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
