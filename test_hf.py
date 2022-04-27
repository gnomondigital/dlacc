import tvm
from utils import from_hf_pretrained
from hf_optimum import Optimum

network_list = ["sentence-transformers/all-MiniLM-L6-v2"]
example_batch_input = ["This is an example sentence", "Each sentence is converted"]
target = "llvm"
device = tvm.device(str(target), 0)


def test_all():
    for network_name in network_list:
        tokenizer, model = from_hf_pretrained(network_name)
        encoded_input = tokenizer(
            example_batch_input,
            padding=True,
            truncation=True,
        )
        optimum = Optimum(model, tokenizer)
        optimum.run(example_batch_input, target, device)
        model = optimum.get_best_model()
        output = model(encoded_input)
        
