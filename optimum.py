from ansor_engine import AnsorEngine
from base_class import BaseClass
from graph_module import GraphModuleWrapper
from utils import infer_platform_type, platformType, download_from_gcp, input_prefix


class Optimum(BaseClass):
    def __init__(self, model_name, onnx_model, config):
        self.onnx_model = onnx_model
        self.model_name = model_name
        self.config = config

    def run(
        self,
        target,
        num_measure_trials,
        mode,
        out_json,
        input_shape=None,
        input_dtype=None,
        log_file=None,
    ):
        if mode == "ansor":
            ae = AnsorEngine(
                self.model_name,
                self.onnx_model,
                target,
                input_shape,
                input_dtype,
                out_json,
            )
            if log_file != "":
                print(
                    "Historical configuration file %s found, tuning will not be executed."
                    % log_file
                )
                ae.ansor_compile(log_file=log_file)
            else:
                ae.ansor_run_tuning(
                    num_measure_trials=num_measure_trials,
                )
        elif mode == "autotvm":
            raise NotImplementedError
        self.ansor_engine = ae

    def get_model(self):
        return GraphModuleWrapper(self.onnx_model, self.ansor_engine.device)

    def load_model(self, input_path):
        platform_type = infer_platform_type(input_path)
        if platform_type == platformType.GOOGLESTORAGE:
            path = download_from_gcp(input_path, input_prefix, "config.json")
        self.ansor_engine.load(input_path)
        return GraphModuleWrapper(self.ansor_engine.module, self.ansor_engine.device)
