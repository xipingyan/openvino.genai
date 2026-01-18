#pragma once

#include <thread>

#include "module_genai/module_base.hpp"

namespace ov {
namespace genai {
namespace module {

// A specific Module can't be registered multiple times. So we use this interface to register different
// DummyModule implementations for different test modules via register table(g_dummy_impl_instances_map).

class DummyModuleInterface {
public:
    DummyModuleInterface() = default;
    virtual ~DummyModuleInterface() = default;

    // Initialize dummy module with IBaseModule pointer.
    virtual void init(IBaseModule* p_base_module) = 0;

    // Run function called by DummyModuleImpl.
    virtual void run(std::map<std::string, IBaseModule::InputModule>& inputs, std::map<std::string, IBaseModule::OutputModule>& outputs) = 0;
    using PTR = std::shared_ptr<DummyModuleInterface>;

protected:
    // Passed from init function. Don't own the pointer.
    IBaseModule* m_base_module = nullptr;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
