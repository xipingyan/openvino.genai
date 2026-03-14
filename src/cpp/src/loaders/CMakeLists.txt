# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Multi-format model loaders module
# This module provides a unified interface for loading models from different formats:
# - GGUF (llama.cpp format)
# - Safetensors (HuggingFace format)

set(LOADERS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/loader_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/weight_name_mapper.cpp
)

set(LOADERS_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/loader_registry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_loader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_builder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/weight_name_mapper.hpp
)

# GGUF loader (conditional)
if(ENABLE_GGUF)
    list(APPEND LOADERS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/gguf/gguf_loader.cpp
    )
    list(APPEND LOADERS_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/gguf/gguf_loader.hpp
    )
endif()

# Safetensors loader (conditional)
if(ENABLE_SAFETENSORS)
    list(APPEND LOADERS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/safetensors/safetensors_loader.cpp
    )
    list(APPEND LOADERS_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/safetensors/safetensors_loader.hpp
    )
endif()

# Export sources to parent scope
set(LOADERS_SRC ${LOADERS_SRC} PARENT_SCOPE)
set(LOADERS_HEADERS ${LOADERS_HEADERS} PARENT_SCOPE)
