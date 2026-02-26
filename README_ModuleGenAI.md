# ModuleGenAI

Typically, VLM (Vision-Language Models), image or video generation models, are composed of many modules. However, GENAI itself is a whole system, and for those unfamiliar with the GENAI architecture, adding new functionalities or modules between different components is difficult. This is why a modular GENAI is needed.

# How to build

#### Dependencies

``1: `` ENV
```
python -m venv python-env
source python-env/bin/activate
pip install numpy
```

``2:`` OpenVINO
```
<!-- OV -->
git clone https://github.com/openvinotoolkit/openvino.git --branch 2025.4.0
cd openvino && mkdir build && cd build
git submodule update --init
cmake -DCMAKE_INSTALL_PREFIX=install ..
make -j20 && make install

<!-- OR download 2025.4 (default) -->
<!-- ubuntu22 -->
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4/linux/openvino_toolkit_ubuntu22_2025.4.0.20398.8fdad55727d_x86_64.tgz
tar -xf openvino_toolkit_ubuntu22_2025.4.0.20398.8fdad55727d_x86_64.tgz
<!-- ubuntu24 -->
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4/linux/openvino_toolkit_ubuntu24_2025.4.0.20398.8fdad55727d_x86_64.tgz
tar -xf openvino_toolkit_ubuntu24_2025.4.0.20398.8fdad55727d_x86_64.tgz

<!-- Update source_ov.sh based on your OV. -->
```

``3:`` Build Module GenAI
```
sudo apt-get install libyaml-cpp-dev

git clone https://github.com/xipingyan/openvino.genai.git
cd openvino.genai
git checkout -b master_modular_genai remotes/origin/master_modular_genai
git submodule update --init
./build_genai.sh
```

# Samples

# Unit test

# How to enable a new module