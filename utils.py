from transformers import AutoTokenizer, AutoModel
import torch
from ansor_engine import AnsorEngine


def get_jit_traced_model(origin_model, example_inputs, save_path=None, model_name=None):
    print("Generate jit traced model...")
    jit_traced_model = torch.jit.trace(
        origin_model, example_inputs=example_inputs
    ).eval()
    if save_path:
        torch.jit.save(jit_traced_model, "jit_traced_{model_name}.pt")
        print("jit_traced_{model_name}.pt saved.")
    print("Jit traced model generation success.")
    return jit_traced_model


def optimize_model(
    traced_model,
    network_name,
    input_infos,
    target,
    log_file=None,
    framework_type="pt",
    mode="ansor",
    num_measure_trials=500
):
    if framework_type == "pt":
        if mode == "ansor":
            ae = AnsorEngine(network_name, target)
            if log_file:
                print("Historical configuration file %s found, tuning will not be executed."% log_file)
                return ae.ansor_call_pt(traced_model, input_infos, "int64").ansor_compile(log_file)
            else:
                return ae.ansor_call_pt(
                    traced_model, input_infos, "int64"
                ).ansor_run_tuning(num_measure_trials=num_measure_trials).ansor_compile()
                
        if mode == "autotvm":
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def from_hf_pretrained(network_name):
    tokenizer = AutoTokenizer.from_pretrained(
        network_name, TOKENIZERS_PARALLELISM=False
    )  # uggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
    model = AutoModel.from_pretrained(network_name, return_dict=False)
    return tokenizer, model
