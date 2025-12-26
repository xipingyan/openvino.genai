SCRIPT_DIR_GENAI_MODULE_PY="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_GENAI_MODULE_PY}

source ../../../../python-env/bin/activate
source ../../../../source_ov.sh

GENAI_ROOT_DIR=${SCRIPT_DIR_GENAI_MODULE_PY}/../../../../openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

cd ${SCRIPT_DIR_GENAI_MODULE_PY}

img_fn=${SCRIPT_DIR_GENAI_MODULE_PY}/../../cpp/module_genai/ut_test_data/cat_120_100.png
model_dir=${SCRIPT_DIR_GENAI_MODULE_PY}/../../cpp/module_genai/ut_pipelines/Qwen2.5-VL-3B-Instruct/torch/

python module_pipeline_with_torch_llm.py ${img_fn} ${model_dir}
