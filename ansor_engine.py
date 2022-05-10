from black import out
from base_object import BaseClass
import os
import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.search_task import TuningOptions
import tvm.relay as relay
from tvm.contrib import graph_executor

# from tvm.contrib.debugger import debug_executor as graph_executor

from pathlib import Path

import logging

DEBUG_MODE = False


class AnsorEngine(BaseClass):
    def __init__(self, network_name) -> None:
        self.network_name = network_name.replace("/", "_")

    def ansor_call_pt(
        self, jit_traced_model, input_infos, default_dtype, batch_size, target
    ):
        mod, params = tvm.relay.frontend.from_pytorch(
            jit_traced_model, input_infos, default_dtype=default_dtype
        )
        self.mod = mod
        self.batch_size = batch_size
        self.target = target
        self.params = params
        return self

    def ansor_run_tuning(
        self,
        jit_traced_model=None,
        input_infos=None,
        default_dtype=None,
        batch_size=None,
        target=None,
        num_measure_trials=500,
        output_path=".",
    ):
        self.ansor_call_pt(
            jit_traced_model, input_infos, default_dtype, batch_size, target
        )
        self._print("Run tuning for network=%s" % self.network_name)
        self.log_file = (
            "./tuning_log/network_name=%s--target=%s--num_measure_trials=%d--batch_size=%d.json"
            % (self.network_name, str(self.target), num_measure_trials, self.batch_size)
        )
        self._print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            self.mod["main"], self.params, self.target
        )

        self._print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(
                repeat=10, enable_cpu_cache_flush=True, timeout=40
            ),
            early_stopping=300,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
        )
        use_sparse = False
        if use_sparse:
            from tvm.topi.sparse.utils import sparse_sketch_rules

            search_policy = [
                auto_scheduler.SketchPolicy(
                    task,
                    program_cost_model=auto_scheduler.XGBModel(),
                    init_search_callbacks=sparse_sketch_rules(),
                )
                for task in tasks
            ]

            tuner.tune(tune_option, search_policy=search_policy)
        else:
            tuner.tune(tune_option)
        # mark log as finished
        p = Path(self.log_file)
        name_without_extension = p.stem
        ext = p.suffix
        new_file_name = f"{name_without_extension}_finished"
        p.rename(Path(p.parent, new_file_name + ext))
        self.log_file = str(p.parent) + new_file_name + ext
        self._print("Tuning Success, configuration file saved at %s" % self.log_file)

        self.ansor_compile(self.log_file, output_path)
        return self

    def ansor_compile(self, log_file=None, output_path="./optimized_models"):
        # Compile with the history best
        if log_file:
            self.log_file = log_file
        self._print("Compile from %s" % self.log_file)
        with auto_scheduler.ApplyHistoryBest(self.log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                graph, lib, graph_params = relay.build(
                    self.mod, target=self.target, params=self.params
                )
        if output_path:
            p = output_path + "/" + self.network_name
            os.makedirs(p, exist_ok=True)
            self._save(p, lib, graph, graph_params)
        self.device = tvm.device(str(self.target), 0)
        if DEBUG_MODE:
            self.module = graph_executor.create(
                graph, lib, self.device, dump_root="./tvmdbg"
            )
        else:
            self.module = graph_executor.create(graph, lib, self.device)
        self._print("Compile success.")
        return self

    def _save(self, output_path, lib, graph, params):
        lib.export_library(output_path + "/deploy_lib.tar")
        with open(output_path + "/deploy_graph.json", "w") as fo:
            fo.write(graph)
        with open(output_path + "/deploy_param.params", "wb") as fo:
            fo.write(relay.save_param_dict(params))

    def load(self, input_path):
        self._print("Load from %s." % input_path)
        loaded_json = open(input_path + "deploy_graph.json").read()
        loaded_lib = tvm.module.load(input_path + "/deploy_lib.tar")
        loaded_params = bytearray(
            open(input_path + "/deploy_param.params", "rb").read()
        )
        self.module = graph_executor.create(loaded_json, loaded_lib, self.device)
        self.module.load_params(loaded_params)
        self._print("Compile success.")
