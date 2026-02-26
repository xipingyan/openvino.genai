SCRIPT_DIR_BUILD_GENAI="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_BUILD_GENAI}

source ./python-env/bin/activate

# source OV
USE_NIGHT_OV="1" # download from nightly build.
if [ $USE_NIGHT_OV = "1" ]; then
    echo "-------------- USE_NIGHTLY_OV"
    UBUNTU_VER=$(lsb_release -rs | cut -d. -f1)
    source ../openvino_toolkit_ubuntu${UBUNTU_VER}_2026.0.0.dev20260117_x86_64/setupvars.sh
else
    echo "-------------- Use my build OV"
    source ../openvino/build/install/setupvars.sh
fi

echo $SCRIPT_DIR_BUILD_GENAI

BUILD_DEBUG="1" # Default build debug
if [ $BUILD_DEBUG = "1" ]; then
    cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build/
    cmake --build ./build/ --config Debug -j 200
    cmake --install ./build/ --config Debug --prefix ./install
else
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release -j 200
    cmake --install ./build/ --config Release --prefix ./install
fi