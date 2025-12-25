// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/module_genai/pipeline.hpp"
#include "py_utils.hpp"
#include "bindings_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;


auto module_generate_docstring = R"(
    Generates sequences or image, video for Modular LLM.

    :param prompt: input prompt
    :type prompt: str
    The prompt can contain <ov_genai_image_i> with i replaced with
    

    :param inputs: Any inputs.
    :type inputs: dict

    :return: return results
    :rtype: dict
)";

ov::AnyMap kwargs_to_intputs(const py::kwargs& kwargs) {
    ov::AnyMap params = {};

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (pyutils::py_object_is_any_map(value)) {
            auto map = pyutils::py_object_to_any_map(value);
            params.insert(map.begin(), map.end());
        } else if (py::isinstance<ov::Tensor>(value)) {
            params[key] = value.cast<ov::Tensor>();
        } else if (py::isinstance<py::list>(value)) {
            auto list = value.cast<py::list>();
            if (list.size() > 0 && py::isinstance<ov::Tensor>(list[0])) {
                params[key] = value.cast<std::vector<ov::Tensor>>();
            } else {
                std::cout << "Error: Input unsupported data type in list with key: " << key << std::endl;
            }
        } else {
            std::cout << "Error: Input unsupported data type with key: " << key << std::endl;
        }
    }
    return params;
}

void call_module_generate(
    ov::genai::module::ModulePipeline& pipe,
    const py::kwargs& kwargs
) {
    auto inputs = kwargs_to_intputs(kwargs);
    {
        py::gil_scoped_release rel;
        pipe.generate(inputs);
    }
}

void init_module_pipeline(py::module_& m) {
    py::class_<ov::genai::module::ModulePipeline>(m,
                                                  "ModulePipeline",
                                                  "This class is used for generation with ModulePipeline")
        .def(py::init([](const std::filesystem::path& models_path, const py::kwargs& kwargs) {
                 return std::make_unique<ov::genai::module::ModulePipeline>(models_path);
             }),
             py::arg("models_path"),
             "folder with exported model files",
             R"(
            Module Pipeline class constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
        )")

        .def("start_chat", &ov::genai::module::ModulePipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ov::genai::module::ModulePipeline::finish_chat)
        .def(
            "generate",
            [](ov::genai::module::ModulePipeline& pipe, const py::kwargs& inputs) -> void {
                return call_module_generate(pipe, inputs);
            },
            (std::string("generate(**kwargs) -> None\n\n") + "Accepts arbitrary keyword arguments as AnyMap.\n\n" +
             module_generate_docstring)
                .c_str());
}
