"""! @brief Defines the Optimum class."""
##
# @file optimum.py
#
# @brief Defines the Optimum class.
#
# @section author_sensors Author(s)
# - Created by Gnomondigital on 02/06/2022.
# - Modified by Gnomondigital on 02/06/2022.
#
# Copyright (c) 2022 Gnomondigital.  All rights reserved.

from tvm.contrib import graph_executor
import tvm
from ansor_engine import AnsorEngine
from base_class import BaseClass
from utils import infer_platform_type, platformType, download_file_from_gcp
import onnx


class GraphModuleWrapper:
    """A wrapper class for graph module. It is the final object for prediction."""    
    def __init__(self, module: tvm.contrib.graph_executor.GraphModule):
        self.module = module

    def __call__(self, inputs_dict: dict) -> dict:
        """Run prediction.

        Parameters
        ----------
        inputs_dict : dict
            The input dictionnary. The keys are input names, values are their numeric values.

        Returns
        -------
        dict
            Output dictionnary with output names and values.
        """
        self.module.set_input(**inputs_dict)
        self.module.run()
        num_outputs = self.module.get_num_outputs()
        tvm_outputs = {}
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            tvm_outputs[output_name] = self.module.get_output(i).numpy()
        return tvm_outputs

    def predict(self, inputs_dict) -> dict:
        """Run prediction.

        Parameters
        ----------
        inputs_dict : dict
            The input dictionnary. The keys are input names, values are their numeric values.

        Returns
        -------
        dict
            Output dictionnary with output names and values.
        """        
        return self.__call__(inputs_dict)


class Optimum(BaseClass):
    """Optimization entry class.

    Parameters
    ----------
    model_name : str
        The name of model.
    """
    def __init__(self, model_name: str):
        """Initializer.

        Parameters
        ----------
        model_name : str
            The name of model.
        """        
        self.model_name = model_name

    def run(self, onnx_model, config: dict):
        """Run tuning process.

        Parameters
        ----------
        onnx_model : onnx.onnx_ml_pb2.ModelProto
                Onnx model object.
        config: dict
            The output json dictionnary which contains information about runtime.

        Returns
        -------
        None
        """
        return self._run(
            onnx_model,
            config["target"],
            config["tuning_config"]["num_measure_trials"],
            config["tuning_config"]["mode"],
            config,
            log_file=config["tuned_log"],
            input_shape=config["model_config"]["input_shape"],
            input_dtype=config["model_config"]["input_dtype"],
            verbose=config["tuning_config"]["verbose_print"],
        )

    def _run(
        self,
        onnx_model: onnx.onnx_ml_pb2.ModelProto,
        target: str,
        num_measure_trials: int,
        mode: str,
        out_json: dict,
        input_shape: list[int]= None,
        input_dtype: list[str]=None,
        log_file: str = None,
        verbose: int = 0,
    ):
        """Entrypoint for AnsorEngine. See comments in related methodes in AnsorEngine."""    
        if mode == "ansor":
            ae = AnsorEngine(
                self.model_name,
                onnx_model,
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
                    num_measure_trials=num_measure_trials, verbose=verbose
                )
        elif mode == "autotvm":
            raise NotImplementedError
        self.ansor_engine = ae
        self.onnx_model = onnx_model

    def get_model(self):
        return GraphModuleWrapper(self.ansor_engine.module)

    def load_model(self, input_path: str, target: str) -> GraphModuleWrapper:
        """load model from a path. 

        Parameters
        ----------
        input_path : str
            The path can be a local path or remote url (Google Storage). If remote, files will be downloaded in the same folder. The directory containing only 3 files: deploy_graph.json, deploy_lib.tar, deploy_param.params.
        target : str
            Target string.

        Returns
        -------
        GraphModuleWrapper
            Runnable prediction module object.
        """
        platform_type = infer_platform_type(input_path)
        if platform_type == platformType.GOOGLESTORAGE:
            # TODO: download a directory
            download_file_from_gcp(
                input_path + "deploy_graph.json", "./download", "deploy_graph.json"
            )
            download_file_from_gcp(
                input_path + "deploy_lib.tar", "./download", "deploy_lib.tar"
            )
            download_file_from_gcp(
                input_path + "deploy_param.params", "./download", "deploy_param.params"
            )
            input_path = "./download"
        device = tvm.device(str(target), 0)
        self._print("Load module from %s" % input_path)
        loaded_json = open(input_path + "/deploy_graph.json").read()
        loaded_lib = tvm.runtime.load_module(input_path + "/deploy_lib.tar")
        loaded_params = bytearray(
            open(input_path + "/deploy_param.params", "rb").read()
        )
        module = graph_executor.create(loaded_json, loaded_lib, device)
        module.load_params(loaded_params)
        self._print("Load success.")
        return GraphModuleWrapper(module)
