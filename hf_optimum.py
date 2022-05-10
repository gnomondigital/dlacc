from ansor_engine import AnsorEngine
from base_object import BaseClass

from utils import get_jit_traced_model, optimize_model
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
    model = optimum.get_best_model()
    output = model(encoded_input)
    """

    def __init__(self, network_name="default_network_name", framework_type="pt"):
        self.network_name = network_name
        self.framework_type = framework_type

    def run(
        self,
        target,
        model=None,
        encoded_input=None,
        mode="ansor",
        num_measure_trials=1000,
        log_file=None,
        traced_model_file=None,
    ):

        if log_file:
            self.ansor_engine = AnsorEngine(self.network_name)
        if self.framework_type == "pt":
            if traced_model_file:
                self._print("Load traced model from %s" % traced_model_file)
                self.traced_model = torch.jit.load(traced_model_file)
            else:
                if model and encoded_input:
                    self.traced_model = get_jit_traced_model(
                        model,
                        tuple(encoded_input.values()),
                        save_path="jit_traced_models/",
                        model_name=self.network_name,
                    ).eval()
                else:
                    self.raise_default_error(
                        "Must specify model and encoded_input when traced model file is absent."
                    )
            if mode == "ansor":
                shape_list = [
                    (i.debugName().split(".")[0], i.type().sizes())
                    for i in list(self.traced_model.graph.inputs())[1:]
                ]
                for t in shape_list:
                    if not t[1]:
                        self.raise_default_error(
                            "Input shape shouldn't have None value : %s" % shape_list
                        )
                batch_size = shape_list[0][1][0]
                self.ansor_engine = optimize_model(
                    self.traced_model,
                    self.network_name,
                    shape_list,
                    target,
                    batch_size,
                    framework_type=self.framework_type,
                    mode=mode,
                    num_measure_trials=num_measure_trials,
                    log_file=log_file,
                )
                if self.ansor_engine.module:
                    self.module = self.ansor_engine.module
                else:
                    self.raise_default_error()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_model(self):
        return GraphModuleWrapper(self.module, self.ansor_engine.device)

    def load_model(self, input_path):
        self.ansor_engine.load(input_path)
        return GraphModuleWrapper(self.ansor_engine.module, self.ansor_engine.device)
