SCRIPT_DIR_UNIT_TEST_CPP="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_UNIT_TEST_CPP}

source ../../../../source_ov.sh
cd ${SCRIPT_DIR_UNIT_TEST_CPP}

export DATA_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/test_data
# export MODEL_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/test_models
export MODEL_DIR=${SCRIPT_DIR_UNIT_TEST_CPP}/../../../samples/cpp/module_genai/ut_pipelines/
# export DEVICE=GPU # Specific device for testing, default is CPU

app=../../../build/tests/module_genai/cpp/genai_modules_test

# All tests
# $app

# All ModuleTest
$app --gtest_filter="ModuleTestSuite*cat_120_100_dog_120_120*"

# All PipelineTest
# $app --gtest_filter="PipelineTest*"