from base_object import BaseObject

import tvm
from tvm import auto_scheduler
import tvm.relay as relay
from tvm.contrib import graph_executor


class AnsorEngine(BaseObject):
    def __init__(self, network_name, target, device) -> None:
        self.network_name = network_name
        self.target = target
        self.device = device

    def _print(self, content):
        print("[Ansor]", end=" ")
        print(content)

    def ansor_call_pt(self, jit_traced_model, input_infos, default_dtype):
        mod, params = tvm.relay.frontend.from_pytorch(
            jit_traced_model, input_infos, default_dtype=default_dtype
        )
        self.mod = mod
        self.params = params
        return self

    def ansor_run_tuning(self, num_measure_trials=500):
        # Extract tasks from the network
        self._print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            self.mod["main"], self.params, self.device
        )
        self.log_file = (
            "ansor--tuning--{self.network_name}--{self.target}-{self.device}.json"
        )
        self._print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(repeat=20, enable_cpu_cache_flush=True),
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
        self._print("Tuning Success.")
        return self

    def ansor_compile(self):
        # Compile with the history best
        self._print("Compile...")
        with auto_scheduler.ApplyHistoryBest(self.log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(self.mod, target=self.target, params=self.params)
        self.module = graph_executor.GraphModule(lib["default"](self.device))
        return self
