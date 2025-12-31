// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "module_base.hpp"
#include "logger.hpp"

namespace ov {
namespace genai {

namespace module {

using PipelineModuleInstance = std::vector<IBaseModule::PTR>;

using PipelineModuleDesc = std::unordered_map<std::string, IBaseModuleDesc::PTR>;

struct PipelineDesc {
    // global_context;
    std::string model_type;

    // main pipeline desc
    PipelineModuleDesc main_pipeline_desc;
    // sub-pipeline name -> sub-pipeline desc
    std::vector<std::pair<std::string, PipelineModuleDesc>> sub_pipeline_descs;
};

void construct_pipeline(const PipelineDesc& pipeline_desc, PipelineModuleInstance& pipeline_instance);

PipelineModuleInstance sort_pipeline(PipelineModuleInstance& pipeline_instrance);

}  // namespace module
}  // namespace genai
}  // namespace ov