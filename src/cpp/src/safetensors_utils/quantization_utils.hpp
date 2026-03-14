// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <string>
#include <iostream>
#include "safetensors_weight_finalizer.hpp"

inline ov::genai::modeling::weights::QuantizationConfig create_quantization_config(
    std::string quant_mode, 
    int group_size, 
    std::string backup_mode) {

    ov::genai::modeling::weights::QuantizationConfig config;
    
    // Normalize to lowercase
    std::transform(quant_mode.begin(), quant_mode.end(), quant_mode.begin(), ::tolower);
    std::transform(backup_mode.begin(), backup_mode.end(), backup_mode.begin(), ::tolower);

    // Parse mode string: check for specific suffixes to avoid substring matching issues
    // E.g., "int4_asym" contains "sym" but should be ASYM, not SYM
    if (quant_mode == "int4_sym") {
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
    } else if (quant_mode == "int4_asym" || quant_mode == "int4") {
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    } else if (quant_mode == "int8_sym") {
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
    } else if (quant_mode == "int8_asym" || quant_mode == "int8") {
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    } else {
        // config.enabled = false; // "enabled" is a method, not a field. Setting mode to NONE disables it.
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
        return config;
    }

    config.group_size = group_size;

    // Backup mode (supports INT4/INT8/NONE)
    if (backup_mode == "int4_sym") {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
    } else if (backup_mode == "int4_asym" || backup_mode == "int4") {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    } else if (backup_mode == "int8_sym") {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
    } else if (backup_mode == "int8_asym" || backup_mode == "int8") {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    } else if (backup_mode.empty() || backup_mode == "none") {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    }
    
    return config;
}
