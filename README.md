# dl_acceleration

# Usage
```python
tokenizer, model = from_hf_pretrained("sentence-transformers/all-MiniLM-L6-v2")
example_batch_input = ["This is an example sentence", "Each sentence is converted"] 
encoded_input = tokenizer(
    example_batch_input, padding=True, truncation=True, return_tensors="pt"
)
optimum = Optimum(model, tokenizer)
target = tvm.target.arm_cpu()
device = tvm.cpu(0)   
optimum = Optimum(model, tokenizer)
optimum.run(encoded_input, target, device)
model = optimum.get_best_model()
output = model(encoded_input)
```