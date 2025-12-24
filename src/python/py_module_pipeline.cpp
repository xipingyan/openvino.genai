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
namespace common_utils = ov::genai::common_bindings::utils;


auto module_generate_docstring = R"(
    Generates sequences for VLMs.

    :param prompt: input prompt
    :type prompt: str
    The prompt can contain <ov_genai_image_i> with i replaced with
    

    :param inputs: Any inputs.
    :type inputs: dict

    :return: return results
    :rtype: dict
)";

py::object call_module_generate(
    ov::genai::module::ModulePipeline& pipe,
    const py::kwargs& kwargs
) {
    {
        py::gil_scoped_release rel;
        
        pipe.generate();
    }
}

void init_vlm_pipeline(py::module_& m) {

    py::class_<ov::genai::module::ModulePipeline>(m, "ModulePipeline", "This class is used for generation with ModulePipeline")
        .def(py::init([](
            const std::filesystem::path& models_path,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::module::ModulePipeline>(models_path);
        }),
        py::arg("models_path"), "folder with exported model files",
        R"(
            VLMPipeline class constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
        )")

        .def("start_chat", &ov::genai::module::ModulePipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ov::genai::module::ModulePipeline::finish_chat)
        .def(
            "generate",
            [](ov::genai::module::ModulePipeline& pipe,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::VLMDecodedResults> {
                return call_vlm_generate(pipe, prompt, images, videos, generation_config, streamer, kwargs);
            },
            py::arg("inputs"), "AnyMap",
            (module_generate_docstring + std::string(" \n ")).c_str()
        );
}
