class GraphModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __call__(self, **inputs):
        self.module.set_input(**inputs)
        self.module.run()
        num_outputs = self.module.get_num_outputs()
        tvm_outputs = {}
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            tvm_outputs[output_name] = self.module.get_output(i).numpy()
        return tvm_outputs
