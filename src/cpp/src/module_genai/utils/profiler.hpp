// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace ov {
namespace genai {
namespace module {

class Profile
{
public:
	Profile() : _active(false) {}
	Profile(const std::string& name,
		const std::vector<std::pair<std::string, std::string>>& args =
		std::vector<std::pair<std::string, std::string>>());
	~Profile();

	static bool enabled();

private:
	std::string _name;
	uint64_t _ts1;
	std::vector<std::pair<std::string, std::string>> _args;
	bool _active = true;
};

/* Usage example:
 * ==========================================
 * Profile(var, "fun_name")
 *
#define PROFILE(VAR, NAME) \
	auto VAR = ::ov::genai::module::Profile::enabled() \
				   ? ::ov::genai::module::Profile(std::string(NAME) + std::string(":") + std::to_string(__LINE__)) \
				   : ::ov::genai::module::Profile()

#define PROFILE_ARGS(VAR, NAME, ...) \
	auto VAR = ::ov::genai::module::Profile::enabled() \
				   ? ::ov::genai::module::Profile(std::string(NAME) + std::string(":") + std::to_string(__LINE__), __VA_ARGS__) \
				   : ::ov::genai::module::Profile()
 *    Profile(p, "fun_name")
 *    func()
 * }
 *
 * Usage 3: specific scope with some params.
 * {
 *    Profile(p2, "fun_name", {{"arg1", "sleep 30 ms"}});
 *    func()
 * }
 */
#define PROFILE(VAR, NAME) auto VAR = ov::genai::module::Profile::enabled() ? ov::genai::module::Profile(NAME + std::string(":") + std::to_string(__LINE__)) : ov::genai::module::Profile()
#define PROFILE_ARGS(VAR, NAME, ...) auto VAR = ov::genai::module::Profile::enabled() ? ov::genai::module::Profile(NAME + std::string(":") + std::to_string(__LINE__), __VA_ARGS__) : ov::genai::module::Profile()

}  // namespace module
}  // namespace genai
}  // namespace ov