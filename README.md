# dl_acceleration

# Installation
## Build from source
```
git clone https://gitlab.gnomondigital.com/fzyuan/dl_acceleration.git
DLACC_HOME = ./dl_acceleration
PYTHONPATH=$DLACC_HOME:${PYTHONPATH}
```
```
export TOKENIZERS_PARALLELISM=false
```
## Install via pip
```
pip install dlacc
```

# Python SDK 
```
python setup.py sdist bdist_wheel
cd dist
pip install dlacc-1.0-py3-none-any.whl
```

# Features
- Automatic Optimization
- Benchmark with various metrics (mean inference time, improvement compare, ..)
- Output optimized models
- Save tuning log
- Support pytorch and onnx models, for tensorflow models, see https://github.com/onnx/tensorflow-onnx
# Usage
## Command line
```python
python3.9 main.py --config example1.json
```
## Python script
View getting_started.ipynb 

## Supported Targets
```
['aocl', 'hybrid', 'nvptx', 'sdaccel', 'opencl', 'metal', 'hexagon', 'aocl_sw_emu', 'rocm', 'webgpu', 'llvm', 'cuda', 'vulkan', 'ext_dev', 'c']
```
Specifying the correct target can have a huge impact on the performance of the compiled module, as it can take advantage of hardware features available on the target. For more information, please refer to Auto-tuning a convolutional network for x86 CPU. We recommend identifying which CPU you are running, along with optional features, and set the target appropriately. For example, for some processors target = "llvm -mcpu=skylake", or target = "llvm -mcpu=skylake-avx512" for processors with the AVX-512 vector instruction set.

Notes: 

Generally: 
- Use 'cuda' for GPU backend;
- Use 'llvm' for CPU backend.

specify num_measure_trials=20000 for best performance tuning for optimum.run() method call.
