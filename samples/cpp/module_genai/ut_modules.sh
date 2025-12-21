SCRIPT_DIR_EXAMPLE_OV_CPP_RUN="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

source ../python-env/bin/activate
# Based on myown openvino
source ../../../../openvino_toolkit_ubuntu24_2025.4.0.20398.8fdad55727d_x86_64/setupvars.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

./build/module_genai_app ./config_pipeline/config_qwen2_5_vl.yaml
