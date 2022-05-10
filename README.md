# dl_acceleration

# Installation
## Build from source
```
git clone https://gitlab.gnomondigital.com/fzyuan/dl_acceleration.git
```
```
export TOKENIZERS_PARALLELISM=false
```
# Features
- Import HuggingFace Models
- Automatic Optimization
- Benchmark
# Usage
```python
tokenizer, model = from_hf_pretrained("sentence-transformers/all-MiniLM-L6-v2")
example_batch_input = ["This is an example sentence", "Each sentence is converted"]
encoded_input = tokenizer(
    example_batch_input, padding=True, truncation=True, return_tensors="pt"
)
optimum = Optimum(network_name)
target = "llvm"
optimum.run(encoded_input, target)
optimized_model = optimum.get_best_model()
output = optimized_model(encoded_input)
```

To run benchmark:
```
nohup python3.9 dl_acceleration/benchmark.py --target "llvm -mcpu=skylake-avx512" --batchsize_range 10 21 10 > execution_log.txt
```

## Supported Target List
```
['aocl', 'hybrid', 'nvptx', 'sdaccel', 'opencl', 'metal', 'hexagon', 'aocl_sw_emu', 'rocm', 'webgpu', 'llvm', 'cuda', 'vulkan', 'ext_dev', 'c']
```
Specifying the correct target can have a huge impact on the performance of the compiled module, as it can take advantage of hardware features available on the target. For more information, please refer to Auto-tuning a convolutional network for x86 CPU. We recommend identifying which CPU you are running, along with optional features, and set the target appropriately. For example, for some processors target = "llvm -mcpu=skylake", or target = "llvm -mcpu=skylake-avx512" for processors with the AVX-512 vector instruction set.


Notes: 

Generally: 
- Use 'cuda' for GPU backend;
- Use 'llvm' for CPU backend.

specify num_measure_trials=20000 for best performance tunning for optimum.run() method call.