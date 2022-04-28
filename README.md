# dl_acceleration

# Installation
## Build from source
```
git clone https://gitlab.gnomondigital.com/fzyuan/dl_acceleration.git

```
export TOKENIZERS_PARALLELISM=false
```
# Features
- Import HuggingFace Models
- 
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