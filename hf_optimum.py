from base_object import BaseClass

from utils import get_jit_traced_model, optimize_model
from graph_module import GraphModuleWrapper


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

    def __init__(self, model, network_name, framework_type="pt"):
        self.network_name = network_name
        self.origin_model = model
        self.framework_type = framework_type

    def run(
        self, encoded_input, target, mode="ansor", num_measure_trials=500, log_file=None
    ):
        if self.framework_type == "pt":
            if mode == "ansor":
                jit_traced_model = get_jit_traced_model(
                    self.origin_model, tuple(encoded_input.values())
                ).eval()
                shape_list = [
                    (i.debugName().split(".")[0], i.type().sizes())
                    for i in list(jit_traced_model.graph.inputs())[1:]
                ]
                batch_size = shape_list[0][1][0]
                self.ansor_engine = optimize_model(
                    jit_traced_model,
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

    def get_best_model(self):
        return GraphModuleWrapper(self.module, self.ansor_engine.device)
