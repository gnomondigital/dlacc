from transformers import AutoTokenizer, AutoModel
import torch
from ansor_engine import AnsorEngine


def get_jit_traced_model(origin_model, example_inputs, save_path=None, model_name=None):
    print("Generate jit traced model...")
    model_name = networkname_to_path(model_name)
    jit_traced_model = torch.jit.trace(
        origin_model, example_inputs=example_inputs
    ).eval()
    if save_path:
        path = save_path + "jit_traced_%s.pt" % (model_name)
        torch.jit.save(jit_traced_model, path)
        print("%s saved." % path)
    print("Jit traced model generation success.")
    return jit_traced_model


def optimize_model(
    traced_model,
    network_name,
    input_infos,
    target,
    batch_size,
    log_file=None,
    framework_type="pt",
    mode="ansor",
    num_measure_trials=200000,
):
    if framework_type == "pt":
        if mode == "ansor":
            ae = AnsorEngine(network_name)
            if log_file:
                print(
                    "Historical configuration file %s found, tuning will not be executed."
                    % log_file
                )
                return ae.ansor_call_pt(
                    traced_model, input_infos, "int32", batch_size, target
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


def networkname_to_path(network_name):
    return network_name.replace("/", "_")
