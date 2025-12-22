SCRIPT_DIR_EXAMPLE_OV_CPP="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP}

source ../python-env/bin/activate

UBUNTU_VER=$(lsb_release -rs | cut -d. -f1)
source ../../../../openvino_toolkit_ubuntu${UBUNTU_VER}_2025.4.0.20398.8fdad55727d_x86_64/setupvars.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP}

mkdir -p build
cd build

export CMAKE_PREFIX_PATH="../../../../install/runtime/cmake/"
export OpenVINOGenAI_DIR="../../../../build"

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
