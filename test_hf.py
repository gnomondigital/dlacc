import tvm
from utils import from_hf_pretrained
from hf_optimum import Optimum

network_list = ["sentence-transformers/all-MiniLM-L6-v2"]
example_batch_input = ["This is an example sentence", "Each sentence is converted"]
target = tvm.target.arm_cpu()
device = tvm.cpu(0)


def test_all():
    for network_name in network_list:
        tokenizer, model = from_hf_pretrained(network_name)
        encoded_input = tokenizer(
            example_batch_input, padding=True, truncation=True, return_tensors="pt"
        )
        optimum = Optimum(model, tokenizer)
        optimum.run(encoded_input, target, device)
        model = optimum.get_best_model()
        output = model(encoded_input)
