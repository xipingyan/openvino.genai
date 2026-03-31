#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>

#include <openvino/openvino.hpp>

namespace ov::genai::modeling::samples {

// Helper function to split text ov model into multiple sub-models.
// For example, for a text model with input: [input_ids, visual_embeds, or audio_embeds] and output: [logits], we can
// split it into 3 sub-models:
//  - Sub-model 1: input_ids -> text_embeds
//  - Sub-model 2: text_embeds + visual_embeds/audio_embeds -> merged_embeds
//  - Sub-model 3: merged_embeds -> logits
// This function will take the original model and the split points, and return the sub-models with submodel's names.
// The split points can be defined by the user, or can be automatically determined by analyzing the model graph.
// The sub-models can then be used for more flexible inference, such as pre-computing text embeddings, or fusing
// visual/audio embeddings at different stages. Parameters:
// Parameters:
//  - original_model: the original ov model to be split.
//  - input_ids_node_name: the name of the input node for input_ids, which is used to find the text embedding branch in the graph.
//  - visual_embeds_node_name: the name of the input node for visual_embeds, which is used to find the visual embedding branch in the graph. It can be empty if the model does not have visual embeddings.
//  - audio_embeds_node_name: the name of the input node for audio_embeds, which is used to find the audio embedding branch in the graph. It can be empty if the model does not have audio embeddings.
std::map<std::string, std::shared_ptr<ov::Model>> split_text_model(const ov::Model& original_model,
                                                                   const std::string& input_ids_node_name,
                                                                   const std::string& visual_embeds_node_name = std::string(),
                                                                   const std::string& audio_embeds_node_name = std::string());
}  // namespace ov::genai::modeling::samples