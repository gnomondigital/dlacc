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
    optimum = Optimum(model, tokenizer)
    target = tvm.target.arm_cpu()
    device = tvm.cpu(0)   
    optimum = Optimum(model, tokenizer)
    optimum.run(encoded_input, target, device)
    model = optimum.get_best_model()
    output = model(encoded_input)
    """

    def __init__(self, model, tokenizer, framework_type="pt"):
        self.network_name = model.name_or_path
        self.origin_model = model
        self.tokenizer = tokenizer
        self.framework_type = framework_type

    def run(self, encoded_input, target, device, mode="ansor"):
        if self.framework_type == "pt":
            if mode == "ansor":
                jit_traced_model = get_jit_traced_model(
                    self.origin_model, tuple(encoded_input.values())
                ).eval()
                shape_list = [
                    (i.debugName().split(".")[0], i.type().sizes())
                    for i in list(jit_traced_model.graph.inputs())[1:]
                ]
                ansor_engine = optimize_model(
                    jit_traced_model,
                    self.network_name,
                    shape_list,
                    target,
                    device,
                    framework_type=self.framework_type,
                    mode=mode,
                )
                if ansor_engine.module:
                    self.module = ansor_engine.module
                else:
                    self.raise_default_error()
            else:
                raise not NotImplementedError
        else:
            raise not NotImplementedError

    def get_best_model(self):
        return GraphModuleWrapper(self.module)
