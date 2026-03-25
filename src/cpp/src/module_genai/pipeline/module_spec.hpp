// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "module_genai/pipeline/module_data_type.hpp"

namespace ov {
namespace genai {
namespace module {

class ModuleSpec final {
public:
    struct PortSpec {
        std::string name;
        std::vector<DataType> supported_types;
        bool optional = false;
    };

    struct ParamSpec {
        std::string name;
        std::string example_value;
        bool optional = false;
        std::string comment;  // e.g. "[Optional], default true."
    };

    explicit ModuleSpec(std::string module_type_string, std::string example_name = {})
        : m_module_type_string(std::move(module_type_string)),
          m_example_name(std::move(example_name)) {
        OPENVINO_ASSERT(!m_module_type_string.empty(), "ModuleSpec: module_type_string must not be empty");
    }

    const std::string& module_type_string() const {
        return m_module_type_string;
    }

    const std::string& example_name() const {
        return m_example_name;
    }

    ModuleSpec& add_input(std::string name, std::initializer_list<DataType> supported_types, bool optional = false) {
        OPENVINO_ASSERT(!name.empty(), "ModuleSpec: input name must not be empty");
        PortSpec spec;
        spec.name = std::move(name);
        spec.supported_types.assign(supported_types.begin(), supported_types.end());
        spec.optional = optional;
        m_inputs.emplace_back(std::move(spec));
        return *this;
    }

    ModuleSpec& add_output(std::string name, std::initializer_list<DataType> supported_types, bool optional = false) {
        OPENVINO_ASSERT(!name.empty(), "ModuleSpec: output name must not be empty");
        PortSpec spec;
        spec.name = std::move(name);
        spec.supported_types.assign(supported_types.begin(), supported_types.end());
        spec.optional = optional;
        m_outputs.emplace_back(std::move(spec));
        return *this;
    }

    ModuleSpec& add_param(std::string name,
                          std::string example_value,
                          bool optional = false,
                          std::string comment = {}) {
        OPENVINO_ASSERT(!name.empty(), "ModuleSpec: param name must not be empty");
        ParamSpec spec;
        spec.name = std::move(name);
        spec.example_value = std::move(example_value);
        spec.optional = optional;
        spec.comment = std::move(comment);
        m_params.emplace_back(std::move(spec));
        return *this;
    }

    const std::vector<PortSpec>& inputs() const {
        return m_inputs;
    }

    const std::vector<PortSpec>& outputs() const {
        return m_outputs;
    }

    const std::vector<ParamSpec>& params() const {
        return m_params;
    }

    // Generate YAML template text similar to existing print_static_config outputs.
    // This does not validate user YAML; it is only a doc/template generator.
    std::string to_yaml_template_string(const std::string& name_override = {}) const {
        const std::string module_name = !name_override.empty() ? name_override : m_example_name;
        OPENVINO_ASSERT(!module_name.empty(), "ModuleSpec: module name must not be empty");

        std::ostringstream os;
        os << "\n";
        os << "  - name: \"" << module_name << "\"\n";
        os << "    type: \"" << m_module_type_string << "\"\n";

        os << "    inputs:\n";
        for (const auto& input : m_inputs) {
            os << "      - name: \"" << input.name << "\"\n";
            os << "        type: \"" << render_type_list_as_single_string(input.supported_types) << "\"\n";
        }

        os << "    outputs:\n";
        for (const auto& output : m_outputs) {
            os << "      - name: \"" << output.name << "\"\n";
            os << "        type: \"" << render_type_list_as_single_string(output.supported_types) << "\"\n";
        }

        if (!m_params.empty()) {
            os << "    params:\n";
            for (const auto& param : m_params) {
                os << "      " << param.name << ": \"" << param.example_value << "\"";
                if (!param.comment.empty()) {
                    os << "    # " << param.comment;
                }
                os << "\n";
            }
        }

        return os.str();
    }

private:
    static std::string render_type_list_as_single_string(const std::vector<DataType>& types) {
        OPENVINO_ASSERT(!types.empty(), "ModuleSpec: supported_types must not be empty");
        if (types.size() == 1u) {
            return DataTypeConverter::toString(types[0]);
        }

        std::ostringstream os;
        for (size_t i = 0; i < types.size(); ++i) {
            if (i != 0u) {
                os << "|";
            }
            os << DataTypeConverter::toString(types[i]);
        }
        return os.str();
    }

    std::string m_module_type_string;
    std::string m_example_name;
    std::vector<PortSpec> m_inputs;
    std::vector<PortSpec> m_outputs;
    std::vector<ParamSpec> m_params;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
