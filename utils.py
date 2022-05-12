from transformers import AutoTokenizer, AutoModel
import torch
from ansor_engine import AnsorEngine

def get_traced_model(origin_model, example_inputs, save_path=None, model_name="default_network_name"):
    print("Generate jit traced model...")
    example_inputs = tuple(example_inputs.values())
    model_name = networkname_to_path(model_name)
    traced_model = torch.jit.trace(
        origin_model, example_inputs=example_inputs
    ).eval()
    if save_path:
        path = save_path + "jit_traced_%s.pt" % (model_name)
        torch.jit.save(traced_model, path)
        print("%s saved." % path)
    print("Jit traced model generation success.")
    return traced_model

def get_input_info_hf(traced_model):
    shape_list = [
    (i.debugName().split(".")[0], i.type().sizes())
        for i in list(traced_model.graph.inputs())[1:]
    ]
    batch_size = shape_list[0][1][0]
    return batch_size, shape_list

def optimize_model(
    traced_model,
    network_name,
    input_infos,
    target,
    batch_size,
    log_file=None,
    framework_type="pt",
    num_measure_trials=200000,
):
    ae = AnsorEngine(network_name)
    if log_file:
        print(
            "Historical configuration file %s found, tuning will not be executed."
            % log_file
        )
        return ae.ansor_call(
            traced_model, input_infos, "int32", batch_size, target, framework_type=framework_type
        ).ansor_compile(log_file)
    else:
        return ae.ansor_run_tuning(
            traced_model,
            input_infos,
            "int32",
            batch_size,
            target,
            num_measure_trials=num_measure_trials,
        )



def from_hf_pretrained(network_name):
    tokenizer = AutoTokenizer.from_pretrained(
        network_name, TOKENIZERS_PARALLELISM=False
    )  # uggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
    model = AutoModel.from_pretrained(network_name, return_dict=False)
    return tokenizer, model


def networkname_to_path(network_name):
    return network_name.replace("/", "_")

