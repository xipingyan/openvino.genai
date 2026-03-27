#include "split_text_model.hpp"

namespace ov::genai::modeling::samples {

// DFS search for the selece node.
std::shared_ptr<ov::Node> dfs_search_select_node(const std::shared_ptr<ov::Node>& current_node, int max_search_depth) {
    if (max_search_depth < 0) {
        return nullptr;
    }
    // Name is not fix, just check the type is "Select"
    if (current_node->get_type_name() == std::string("Select")) {
        return current_node;
    }
    for (const auto& output : current_node->outputs()) {
        for (const auto& target_input : output.get_target_inputs()) {
            const auto* next_node = target_input.get_node();
            auto result =
                dfs_search_select_node(std::const_pointer_cast<ov::Node>(next_node->shared_from_this()),
                                       max_search_depth - 1);
            if (result != nullptr) {
                return result;
            }
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Node> dfs_search_gather_node(const std::shared_ptr<ov::Node>& current_node, int max_search_depth) {
    if (max_search_depth < 0) {
        std::cout << "DFS search gather node reach max search depth, current node: " << current_node->get_friendly_name() << std::endl;
        return nullptr;
    }
    // Name is not fix, just check the type is "Gather"
    if (current_node->get_type_name() == std::string("Gather")) {
        return current_node;
    }
    // Towards son node.
    for (const auto& output : current_node->outputs()) {
        for (const auto& target_input : output.get_target_inputs()) {
            const auto* next_node = target_input.get_node();
            auto result = dfs_search_gather_node(std::const_pointer_cast<ov::Node>(next_node->shared_from_this()),
                                                 max_search_depth - 1);
            if (result != nullptr) {
                return result;
            }
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Parameter> find_parameter_node_by_name(const ov::Model& model,
                                                                    const std::string& parameter_name) {
    std::shared_ptr<ov::op::v0::Parameter> parameter_node = nullptr;
    for (const auto& parameter : model.get_parameters()) {
        if (parameter->get_friendly_name() == parameter_name) {
            parameter_node = parameter;
            break;
        }
    }
    return parameter_node;
}

// Get last "select" node.
// For example: select node1 -> select node2 -> select node3, return select node3 as the split point.
std::shared_ptr<ov::Node> find_last_select_node(std::shared_ptr<ov::Node> select_node) {
    std::shared_ptr<ov::Node> last_select_node = select_node;
    while (true) {
        bool found_next_select = false;
        for (const auto& output : last_select_node->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                const auto* next_node = target_input.get_node();
                if (next_node->get_type_name() == std::string("Select")) {
                    last_select_node = std::const_pointer_cast<ov::Node>(next_node->shared_from_this());
                    found_next_select = true;
                    break;
                }
            }
            if (found_next_select) {
                break;
            }
        }
        if (!found_next_select) {
            break;
        }
    }
    return last_select_node;
}

// Automatic split point finding can be implemented by analyzing the original model graph,  and finding the node
// that matches the expected pattern. Pattern 1: Type is select node, if exists 2 neighbors select node, choose the
// last one as the split point. Pattern 2: Inputs: [input_ids, visual_embeds/audio_embeds], the 3 inputs should link
// to the same select node. Pattern 3: If "input_ids" has 2 sons in the graph, if one of the sons is a "shapeof"
// node. The "gather" node should be between the other son and the select node. Pattern 4: Search max deep should be
// less than 10, to avoid wrong split point from other branches.
std::shared_ptr<ov::Node> search_split_point(const std::shared_ptr<ov::Node>& input_node,
                                             const int max_search_depth = 10) {
    // Step 1: DFS search for the select node with the expected pattern
    auto select_node = dfs_search_select_node(input_node, max_search_depth);
    if (select_node == nullptr) {
        return nullptr;
    }
    auto last_select_node = find_last_select_node(select_node);

    return last_select_node;
}

// Find all parameters node of the split node. From the split node, traverse the graph backward to find all parameters
// nodes. To avoid duplicate nodes, use a set to store the visited nodes.
// For example:
// A   B       C
//  \ /        |
//  select1  gather
//    |   /
//  select2
// From this graph, if split node is select2, the parameters nodes are A, B and C. If split node is select1, the
// parameters nodes are A and B.
std::set<std::shared_ptr<ov::Node>> find_parameters(const std::shared_ptr<ov::Node>& node) {
    std::set<std::shared_ptr<ov::Node>> parameters;
    std::set<std::shared_ptr<ov::Node>> visited;
    std::vector<std::shared_ptr<ov::Node>> stack;
    stack.push_back(node);
    while (!stack.empty()) {
        auto current_node = stack.back();
        stack.pop_back();
        if (visited.count(current_node) > 0) {
            continue;
        }
        visited.insert(current_node);
        if (current_node->get_type_name() == std::string("Parameter")) {
            parameters.insert(current_node);
        }
        for (const auto& input : current_node->inputs()) {
            const auto next_node = input.get_source_output().get_node_shared_ptr();
            stack.push_back(next_node);
        }
    }
    return parameters;
}

// Get embedding merge model.
// Inputs: input text embedding, visual embedding and audio embedding.
// Output: merged embedding.
std::shared_ptr<ov::Model> find_embedding_merge_model(const ov::Model& original_model,
                                                      std::shared_ptr<ov::Node> split_node,
                                                      std::shared_ptr<ov::Node> input_ids_node,
                                                      const std::shared_ptr<ov::Node>& gather_node) {
    OPENVINO_ASSERT(gather_node != nullptr, "find_embedding_merge_model: gather_node is null");
    OPENVINO_ASSERT(split_node != nullptr, "find_embedding_merge_model: split_node is null");
    OPENVINO_ASSERT(input_ids_node != nullptr, "find_embedding_merge_model: input_ids_node is null");

    // gather_node's output as new model's input.
    auto text_embedding_input = std::make_shared<ov::op::v0::Parameter>(gather_node->get_element_type(),
                                                                        gather_node->get_output_partial_shape(0));
    text_embedding_input->set_friendly_name("text_embedding_input");
    ov::ParameterVector new_parameters{text_embedding_input};
    // Keep all original parameters except input_ids to preserve graph dependencies.
    for (const auto& parameter : original_model.get_parameters()) {
        if (parameter->get_friendly_name() == input_ids_node->get_friendly_name()) {
            continue;
        }
        new_parameters.push_back(parameter);
    }

    // replace gather_node output consumers with text_embedding_input.
    for (const auto& output : gather_node->outputs()) {
        for (auto target_input : output.get_target_inputs()) {
            target_input.replace_source_output(text_embedding_input->output(0));
        }
    }

    // Debug: show all parameters of the new model.
    std::cout << "New model parameters: " << std::endl;
    for (const auto& parameter : new_parameters) {
        std::cout << " - Parameter name: " << parameter->get_friendly_name() << std::endl;
    }

    // split_node's output as new model's output.
    ov::ResultVector new_results{std::make_shared<ov::op::v0::Result>(split_node->output(0))};
    auto embedding_merge_model = std::make_shared<ov::Model>(new_results, new_parameters, "embedding_merge_model");
    ov::serialize(embedding_merge_model, "embedding_merge_model.xml", "embedding_merge_model.bin");
    return embedding_merge_model;
}

// Split ov::model into text embed model, embedding merge model, and LLM model with embedding input.
std::map<std::string, std::shared_ptr<ov::Model>> split_text_model(const ov::Model& original_model,
                                                                   const std::string& input_ids_node_name,
                                                                   const std::string& visual_embeds_node_name,
                                                                   const std::string& audio_embeds_node_name) {
    std::map<std::string, std::shared_ptr<ov::Model>> result;

    // Find split point based on input_ids.
    auto input_ids_node = find_parameter_node_by_name(original_model, input_ids_node_name);
    auto input_ids_select_node = search_split_point(input_ids_node, 10);
    if (input_ids_select_node == nullptr) {
        std::cout << "Cannot find split point based on input_ids node: " << input_ids_node_name << std::endl;
        return {};
    }

    // Find split point based on visual_embeds.
    if (!visual_embeds_node_name.empty()) {
        auto visual_embeds_node = find_parameter_node_by_name(original_model, visual_embeds_node_name);
        auto visual_embeds_select_node = search_split_point(visual_embeds_node, 5);
        if (visual_embeds_select_node == nullptr) {
            std::cout << "Cannot find select node based on visual_embeds node: " << visual_embeds_node_name << std::endl;
            return {};
        }
        // Check if input_ids_select_node and visual_embeds_select_node are the same node, if not, return empty.
        if (input_ids_select_node->get_friendly_name() != visual_embeds_select_node->get_friendly_name()) {
            std::cout << "The select node based on input_ids and visual_embeds are different, input_ids_select_node: "
                      << input_ids_select_node->get_friendly_name()
                      << ", visual_embeds_select_node: " << visual_embeds_select_node->get_friendly_name() << std::endl;
            return {};
        }
    }

    // Find split point based on audio_embeds.
    if (!audio_embeds_node_name.empty()) {
        auto audio_embeds_node = find_parameter_node_by_name(original_model, audio_embeds_node_name);
        auto audio_embeds_select_node = search_split_point(audio_embeds_node, 5);
        if (audio_embeds_select_node == nullptr) {
            std::cout << "Cannot find select node based on audio_embeds node: " << audio_embeds_node_name << std::endl;
            return {};
        }
        // Check if input_ids_select_node and audio_embeds_select_node are the same node, if not, return empty.
        if (input_ids_select_node->get_friendly_name() != audio_embeds_select_node->get_friendly_name()) {
            std::cout << "The select node based on input_ids and audio_embeds are different, input_ids_select_node: "
                      << input_ids_select_node->get_friendly_name()
                      << ", audio_embeds_select_node: " << audio_embeds_select_node->get_friendly_name() << std::endl;
            return {};
        }
    }

    auto split_node = input_ids_select_node;

    // Find all parameters of the split node.
    auto parameters = find_parameters(split_node);

    // Find text embed model.
    std::shared_ptr<ov::Node> gather_node = nullptr;
    {
        // DFS search for the gather node.
        gather_node = dfs_search_gather_node(input_ids_node, 5);
        if (gather_node == nullptr) {
            std::cout << "Cannot find gather node based on input_ids node: " << input_ids_node_name << std::endl;
            return {};
        }

        // Create a new model with gather node as output and input_ids node as input.
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_node->output(0))};
        auto text_embed_model = std::make_shared<ov::Model>(results,
                                                            ov::ParameterVector{input_ids_node},
                                                            "input_ids_embed_model");
        result["text_embed_model"] = text_embed_model;
    }

    // Find merger model.
    auto embedding_merge_model = find_embedding_merge_model(original_model, split_node, input_ids_node, gather_node);
    result["embedding_merge_model"] = embedding_merge_model;

    // Get LLM model with embedding input.
    {
        // Replace original model's input with split node's output, and replace original model's output with split
        // node's output.
        ov::ParameterVector new_parameters;
        for (const auto& parameter : original_model.get_parameters()) {
            if (parameter->get_friendly_name() == input_ids_node->get_friendly_name()) {
                continue;
            }
            new_parameters.push_back(std::make_shared<ov::op::v0::Parameter>(parameter->get_element_type(),
                                                                             parameter->get_output_partial_shape(0)));
        }

        auto embedding_input = std::make_shared<ov::op::v0::Parameter>(split_node->get_element_type(),
                                        split_node->get_output_partial_shape(0));
        new_parameters.push_back(embedding_input);

        // Replace split_node's output's input with embedding_input.
        for (const auto& output : split_node->outputs()) {
            for (auto target_input : output.get_target_inputs()) {
                target_input.replace_source_output(embedding_input->output(0));
            }
        }

        // If input_ids_node have 2 outputs, the one is shapeof node, replace shapeof node's input with embedding_input
        // as well.
        if (input_ids_node->outputs().size() == 2) {
            for (const auto& output : input_ids_node->outputs()) {
                for (auto target_input : output.get_target_inputs()) {
                    const auto& next_node = target_input.get_node();
                    if (next_node->get_type_name() == std::string("ShapeOf")) {
                        target_input.replace_source_output(embedding_input->output(0));
                    }
                }
            }
        }

        auto llm_model = std::make_shared<ov::Model>(original_model.get_results(), new_parameters, "llm_model");
        result["llm_model"] = llm_model;
    }

    return result;
}
}  // namespace ov::genai::modeling::samples