import numpy as np
import timeit


class GraphModuleWrapper:
    def __init__(self, module, device):
        self.module = module
        self.device = device

    def __call__(self, inputs_dict):
        self.module.set_input(**inputs_dict)
        self.module.run()
        num_outputs = self.module.get_num_outputs()
        tvm_outputs = {}
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            tvm_outputs[output_name] = self.module.get_output(i).numpy()
        return tvm_outputs
