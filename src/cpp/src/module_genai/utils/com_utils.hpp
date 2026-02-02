// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ov::genai::module {
namespace utils {

bool check_env_variable(const std::string& var_name);

}  // namespace utils
}  // namespace ov::genai::module