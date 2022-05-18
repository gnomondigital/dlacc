from transformers import AutoTokenizer, AutoModel
import torch
from metadata import ModelType, SourceType, input_prefix, output_prefix
from pathlib import Path
import os
from base_class import BaseClass
import json


def get_traced_model(
    origin_model, example_inputs, save_path=None, model_name="default_network_name"
):
    print("Generate jit traced model...")
    example_inputs = tuple(example_inputs.values())
    model_name = networkname_to_path(model_name)
    traced_model = torch.jit.trace(origin_model, example_inputs=example_inputs).eval()
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


def from_hf_pretrained(network_name):
    tokenizer = AutoTokenizer.from_pretrained(
        network_name, TOKENIZERS_PARALLELISM=False
    )  # uggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
    model = AutoModel.from_pretrained(network_name, return_dict=False)
    return tokenizer, model


def plateform_type_infer(model_path: str):
    if model_path.startswith("gs://"):
        return SourceType.GOOGLESTORAGE
    else:
        return SourceType.LOCAL


def networkname_to_path(network_name):
    return network_name.replace("/", "_")


def download_from_gcp(url, folder, rename: str):
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    url_path = Path(url)
    os.system("gsutil -m cp %s %s" % (url, folder))
    return folder + "/" + rename


def upload(url, plateform_type):
    if plateform_type == SourceType.GOOGLESTORAGE:
        os.system("gsutil -m cp -r %s %s" % (output_prefix, url))
        with open("success", "w") as flag_file:
            os.system("gsutil cp %s %s" % (flag_file, url))


def convert2onnx(plateform_type, model_path, model_type):
    file_path = None
    if plateform_type == int(SourceType.LOCAL):
        file_path = model_path
    elif plateform_type == int(SourceType.GOOGLESTORAGE):
        file_path = download_from_gcp(model_path)
    else:
        raise NotImplementedError

    if file_path:
        input_shape = {}
        model = None
        if model_type == int(ModelType.ONNX):
            import onnx

            model = onnx.load(input_prefix + "/model.onnx")
        elif model_type == int(ModelType.PT):
            raise NotImplementedError
        elif model_type == ModelType.TF:
            raise NotImplementedError
        for inp in model.graph.input:
            shape = str(inp.type.tensor_type.shape.dim)
            input_shape[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
    return model, input_shape


class JSONConfig(BaseClass):
    def __init__(self, json_path, plateform_type) -> None:
        self.load(json_path, plateform_type)

    def load(self, json_path, plateform_type):
        path = input_prefix + "/" + json_path
        if plateform_type == SourceType.GOOGLESTORAGE:
            path = download_from_gcp(json_path, input_prefix, "config.json")
        elif plateform_type == SourceType.AWSSTORAGE:
            raise NotImplementedError
        with open(path) as json_file:
            self.meta = json.load(json_file)

    def __getitem__(self, key):
        return self.meta[key]


class JSONOutput(BaseClass):
    def __init__(self, json_config: JSONConfig):
        self.meta = json_config.meta

    def save(self, file_path):
        with open(file_path, "w") as outfile:
            json.dump(self.meta, outfile)

    def __getitem__(self, key):
        return self.meta[key]

    def __setitem__(self, key, value):
        self.meta[key] = value
        self.save(output_prefix + "/output_json.json")
