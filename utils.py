from transformers import AutoTokenizer, AutoModel
import torch
from ansor_engine import AnsorEngine


def get_jit_traced_model(origin_model, example_inputs, save_path=None, model_name=None):
    print("generate jit traced model...")
    jit_traced_model = torch.jit.trace(
        origin_model, example_inputs=example_inputs
    ).eval()
    if save_path:
        torch.jit.save(jit_traced_model, "jit_traced_{model_name}.pt")
        print("jit_traced_{model_name}.pt saved.")
    return jit_traced_model


def optimize_model(
    traced_model,
    network_name,
    input_infos,
    target,
    device,
    framework_type="pt",
    mode="ansor",
):
    if framework_type == "pt":
        if mode == "ansor":
            ae = AnsorEngine(network_name, target, device)
            ae.ansor_call_pt(
                traced_model, input_infos, "int64"
            ).ansor_run_tuning().ansor_compile()
            return ae
        if mode == "autotvm":
            raise not NotImplementedError
        else:
            raise not NotImplementedError
    else:
        raise not NotImplementedError


def from_hf_pretrained(network_name):
    tokenizer = AutoTokenizer.from_pretrained(
        network_name, TOKENIZERS_PARALLELISM=False
    )  # uggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
    model = AutoModel.from_pretrained(network_name, return_dict=False)
    return tokenizer, model


def dump_object(obj):
    attrs = vars(obj)
    # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
    # now dump this in some way or another
    content = ", ".join("%s: %s" % item for item in attrs.items())
    return content
