from hf_optimum import Optimum
from tvm.testing import assert_allclose
import torch
from transformers import AutoModel
from testings import network_list, get_encoded_input, default_target


def test_all(target, num_measure_trials):
    for network_name in network_list[:1]:
        model = AutoModel.from_pretrained(network_name, return_dict=False)
        optimum = Optimum(network_name)
        encoded_input = get_encoded_input(2, network_name)
        optimum.run(target, model, encoded_input, num_measure_trials=num_measure_trials)
        optimized_model = optimum.get_model()
        output = optimized_model(encoded_input)
        with torch.no_grad():
            original_output = model(**encoded_input)
        output = list(output.values())
        for i in range(len(original_output)):
            assert_allclose(original_output[i], output[i], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_all(default_target, 10)
