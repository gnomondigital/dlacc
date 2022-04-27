import tvm
from tvm import auto_scheduler
import tvm.relay as relay
from transformers import AutoTokenizer, AutoModel
import torch
from tvm.testing import assert_allclose
from tvm.contrib import graph_executor

# Load model from HuggingFace Hub
network = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(
    network, TOKENIZERS_PARALLELISM=False
)  # uggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
model = AutoModel.from_pretrained(network, return_dict=False)
# Sentences we want sentence embeddings for
sentences = ["This is an example sentence", "Each sentence is converted"]
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
print("dummy_inputs", tuple(encoded_input.values()))
print(type(model))
scripted_model = torch.jit.trace(model, tuple(encoded_input.values())).eval()
torch.jit.save(scripted_model, "traced_bert.pt")
shape_list = [
    (i.debugName().split(".")[0], i.type().sizes())
    for i in list(scripted_model.graph.inputs())[1:]
]
print(shape_list)

mod_bert, params_bert = tvm.relay.frontend.from_pytorch(
    scripted_model, shape_list, default_dtype="int32"
)

# Extract tasks from the network
target = tvm.target.arm_cpu()
dev = tvm.cpu(0)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(
    mod_bert["main"], params_bert, target
)

log_file = "%s--%s.json" % (network, target.kind.name)


def run_tuning():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=500,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=20, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
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


run_tuning()

# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    ):
        lib = relay.build(mod_bert, target=target, params=params_bert)


module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input_ids", encoded_input["input_ids"])
module.set_input("token_type_ids", encoded_input["token_type_ids"])
module.set_input("attention_mask", encoded_input["attention_mask"])
module.run()

num_outputs = module.get_num_outputs()
tvm_outputs = {}
for i in range(num_outputs):
    output_name = "output_{}".format(i)
    tvm_outputs[output_name] = module.get_output(i).numpy()

# Compute token embeddings
with torch.no_grad():
    orginial_model_output = model(**encoded_input)


for i in range(len(orginial_model_output)):
    assert_allclose(
        orginial_model_output[i], list(tvm_outputs.values())[i], atol=1e-4, rtol=1e-4
    )

print("assert_allclose success.")

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

with tvm.transform.PassContext(
    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
):
    lib = relay.build(mod_bert, target=target, params=params_bert)

not_tunned_module = graph_executor.GraphModule(lib["default"](dev))

print(not_tunned_module.benchmark(dev, repeat=3, min_repeat_ms=500))
