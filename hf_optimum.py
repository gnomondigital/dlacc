from ansor_engine import AnsorEngine
from base_object import BaseClass

from utils import get_traced_model, optimize_model
from graph_module import GraphModuleWrapper
import torch


class Optimum(BaseClass):
    """
    tokenizer, model = from_hf_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    example_batch_input = ["This is an example sentence", "Each sentence is converted"]
    encoded_input = tokenizer(
        example_batch_input, padding=True, truncation=True, return_tensors="pt"
    )
    optimum = Optimum(network_name)
    target = tvm.target.arm_cpu()
    optimum.run(encoded_input, target)
    model = optimum.get_model()
    output = model(encoded_input)
    """

    def __init__(self, network_name="default_network_name", framework_type="pt"):
        self.network_name = network_name
        self.framework_type = framework_type

    def run(
        self,
        target,
        batch_size,
        shape_dict,
        traced_model=None,
        num_measure_trials=1000,
        log_file=None,
        traced_model_file=None,
    ):
        if traced_model_file:
            self._print("Load traced model from %s" % traced_model_file)
            self.traced_model = torch.jit.load(traced_model_file)
        else:
            if traced_model:
                self.traced_model = traced_model
            else:
                self.raise_default_error(
                    "Traced model file is missing."
                )
        if log_file:
            self.ansor_engine = AnsorEngine(self.network_name)
        if self.framework_type == "pt":
                self.ansor_engine = optimize_model(
                    self.traced_model,
                    self.network_name,
                    shape_dict,
                    target,
                    batch_size,
                    framework_type=self.framework_type,
                    num_measure_trials=num_measure_trials,
                    log_file=log_file,
                )
        elif self.framework_type == 'onnx':
            self.ansor_engine = optimize_model(
                self.traced_model, 
                self.network_name,
                shape_dict,
                target, 
                batch_size,
                framework_type=self.framework_type,
                num_measure_trials=num_measure_trials,
                log_file=log_file,               
            )
        else:
            raise NotImplementedError
        if self.ansor_engine.module:
            self.module = self.ansor_engine.module
        else:
            self.raise_default_error()

    def get_model(self):
        return GraphModuleWrapper(self.module, self.ansor_engine.device)

    def load_model(self, input_path):
        self.ansor_engine.load(input_path)
        return GraphModuleWrapper(self.ansor_engine.module, self.ansor_engine.device)
