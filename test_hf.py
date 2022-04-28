import tvm
from utils import from_hf_pretrained
from hf_optimum import Optimum
from tvm.testing import assert_allclose
import torch

network_list = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "camembert-base",
    "facebook/bart-base",
    "roberta-base",
    "distilgpt2",
    "bert-base-uncased",
]
example_batch_input = ["This is an example sentence", "Each sentence is converted"]
target = tvm.target.arm_cpu()


def test_all():
    for network_name in network_list:
        tokenizer, model = from_hf_pretrained(network_name)
        encoded_input = tokenizer(
            example_batch_input, padding=True, truncation=True, return_tensors="pt"
        )
        optimum = Optimum(model, network_name)
        optimum.run(encoded_input, target, num_measure_trials=20)
        optimized_model = optimum.get_best_model()
        output = optimized_model(encoded_input)
        with torch.no_grad():
            original_output = model(**encoded_input)
        output = list(output.values())
        for i in range(len(original_output)):
            assert_allclose(original_output[i], output[i], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_all()
