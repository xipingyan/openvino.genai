// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <variant>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// Use ordered_json to preserve insertion order
using json = nlohmann::ordered_json;

// ============================================================================
// Forward Declarations
// ============================================================================

class ComfyUIJsonParser;
class WorkflowToApiConverter;

// ============================================================================
// Common Types
// ============================================================================

// Input type information
struct InputTypeInfo {
    std::string type;  // INT, FLOAT, STRING, BOOLEAN, or custom types
    std::string category;  // "required" or "optional"
    std::map<std::string, std::variant<int, double, std::string, std::vector<std::string>>> extra_info;
};

// Node definition
struct Node {
    std::string class_type;
    std::map<std::string, json> inputs;
    json meta;
    std::string node_id_str;
};

// Validation error
struct ValidationError {
    std::string type;
    std::string message;
    std::string details;
    std::map<std::string, json> extra_info;
};

// Validation result for a node
struct NodeValidationResult {
    bool valid;
    std::vector<ValidationError> errors;
    std::string node_id;
};

// Node class information (moved from ComfyUIJsonParser private section)
struct NodeClassInfo {
    std::map<std::string, InputTypeInfo> required_inputs;
    std::map<std::string, InputTypeInfo> optional_inputs;
    std::vector<std::string> return_types;
    bool is_output_node;
};

// Prompt validation result
struct PromptValidationResult {
    bool success;
    ValidationError error;  // Main error if validation failed
    std::vector<std::pair<std::string, std::vector<ValidationError>>> output_errors;
    std::map<std::string, json> node_errors;
};

// ============================================================================
// ComfyUIJsonParser - Parse and validate ComfyUI JSON files
// ============================================================================

/**
 * @brief ComfyUI JSON Parser and Validator
 *
 * Parses ComfyUI API JSON files and validates the node connections and inputs.
 * Also supports workflow JSON format by automatically converting to API format.
 */
class ComfyUIJsonParser {
public:
    ComfyUIJsonParser();
    ~ComfyUIJsonParser();

    // Load and parse JSON file
    bool load_json_file(const std::filesystem::path& file_path);

    // Parse JSON from string
    bool parse_json_string(const std::string& json_str);

    // Validate the loaded prompt
    PromptValidationResult validate_prompt(const std::string& prompt_id = "");

    // Get the parsed nodes
    const std::map<std::string, Node>& get_nodes() const { return nodes_; }

    // Get the parsed API JSON (after workflow->API conversion if needed)
    const json& get_api_json() const { return prompt_; }

    // Get the API JSON as string
    std::string get_api_json_string() const { return prompt_.dump(2); }

    // Register a node class type with its input types
    void register_node_class(const std::string& class_name,
                            const std::map<std::string, InputTypeInfo>& required_inputs,
                            const std::map<std::string, InputTypeInfo>& optional_inputs,
                            const std::vector<std::string>& return_types,
                            bool is_output_node = false);

    // Get validation errors as string
    std::string get_validation_errors_string() const;

    // Check if JSON is workflow format (has "nodes" array) or API format
    bool is_workflow_json(const json& j) const;

    // Get the source JSON format type
    enum class JsonFormat {
        UNKNOWN,
        WORKFLOW,  // Has "nodes" array
        API        // Direct node dictionary
    };
    JsonFormat get_json_format() const { return json_format_; }

private:
    // Process parsed JSON: detect format, convert if needed, and populate nodes_
    bool process_json(const json& source_json);

    // Validate individual node inputs
    NodeValidationResult validate_node_inputs(
        const std::string& prompt_id,
        const std::string& node_id,
        std::map<std::string, NodeValidationResult>& validated);

    // Validate input value type
    bool validate_input_type(const json& value, const std::string& expected_type);

    // Check if linked node output type matches input type
    bool validate_node_input(const std::string& received_type, const std::string& input_type);

private:
    json prompt_;  // The parsed JSON prompt
    std::map<std::string, Node> nodes_;  // Parsed nodes
    JsonFormat json_format_;  // Source JSON format

    // Node class mappings: class_type -> (required_inputs, optional_inputs, return_types, is_output_node)
    std::map<std::string, NodeClassInfo> node_class_mappings_;

    std::vector<ValidationError> errors_;
};

// ============================================================================
// WorkflowToApiConverter - Convert workflow JSON to API JSON
// ============================================================================

/**
 * @brief Custom comparator for node IDs to maintain proper order
 * Regular nodes (e.g., "9", "58") come before subgraph nodes (e.g., "57:11")
 * Within each group, sort numerically
 */
struct NodeIdComparator {
    bool operator()(const std::string& a, const std::string& b) const {
        bool a_is_composite = a.find(':') != std::string::npos;
        bool b_is_composite = b.find(':') != std::string::npos;

        // Regular nodes come before composite nodes
        if (!a_is_composite && b_is_composite) return true;
        if (a_is_composite && !b_is_composite) return false;

        // Both are the same type, compare numerically
        if (!a_is_composite) {
            // Both are regular nodes
            return std::stoi(a) < std::stoi(b);
        } else {
            // Both are composite nodes (e.g., "57:11" vs "57:13")
            size_t colon_a = a.find(':');
            size_t colon_b = b.find(':');

            int parent_a = std::stoi(a.substr(0, colon_a));
            int parent_b = std::stoi(b.substr(0, colon_b));

            if (parent_a != parent_b) {
                return parent_a < parent_b;
            }

            int child_a = std::stoi(a.substr(colon_a + 1));
            int child_b = std::stoi(b.substr(colon_b + 1));
            return child_a < child_b;
        }
    }
};

/**
 * @brief Workflow JSON to API JSON converter
 *
 * Converts ComfyUI workflow JSON format to API JSON format.
 *
 * Workflow JSON structure (from ComfyUI UI):
 * {
 *   "nodes": [
 *     {
 *       "id": 9,
 *       "type": "SaveImage",
 *       "inputs": [{"name": "images", "type": "IMAGE", "link": 54}],
 *       "widgets_values": ["ComfyUI"]
 *     }
 *   ],
 *   "links": [[54, 28, 0, 9, 0, "IMAGE"]],
 *   ...
 * }
 *
 * API JSON structure (for execution):
 * {
 *   "9": {
 *     "inputs": {
 *       "filename_prefix": "ComfyUI",
 *       "images": ["28", 0]
 *     },
 *     "class_type": "SaveImage"
 *   }
 * }
 */
class WorkflowToApiConverter {
public:
    /**
     * @brief Convert workflow JSON to API JSON format
     * @param workflow_json Workflow JSON object
     * @return API JSON object
     */
    static json convert(const nlohmann::json& workflow_json);

    /**
     * @brief Convert workflow to list of nodes (preserves insertion order)
     * Regular nodes come first (in workflow order), then subgraph nodes
     */
    static std::vector<std::pair<std::string, json>> convert_to_node_map(const nlohmann::json& workflow_json);

    /**
     * @brief Load workflow JSON from file and convert to API JSON
     * @param workflow_path Path to workflow JSON file
     * @param api_path Path to save API JSON file
     * @return true if successful
     */
    static bool convert_file(const std::string& workflow_path, const std::string& api_path);

private:
    /**
     * @brief Check if a string is a UUID (contains hyphens in UUID format)
     */
    static bool is_uuid(const std::string& str);

    /**
     * @brief Expand subgraph nodes into node_list
     */
    static void expand_subgraph(
        const nlohmann::json& parent_node,
        const nlohmann::json& subgraph,
        std::vector<std::pair<std::string, json>>& node_list,
        const std::unordered_map<int, nlohmann::json>& link_map,
        const std::unordered_map<int, int>& bypass_map
    );

    /**
     * @brief Get widget input names for a given node type
     */
    static std::vector<std::string> get_widget_input_names(const std::string& node_type, size_t widget_count);
};
}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
