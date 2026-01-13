// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <variant>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// Use ordered_json to preserve insertion order
using json = nlohmann::ordered_json;
class ComfyUIJsonParser;

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

// ComfyUI JSON Parser and Validator
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

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
