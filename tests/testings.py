
def convert2onnx(model, domain_type, model_name, model_type, input_shape, input_dtype):
    def contruct_dummy_input(input_shape, input_dtype, tensor_type, device="cuda"):
        import numpy as np

        if tensor_type == "pt":
            import torch

            dummy_input = tuple(
                [
                    torch.randn(*v).type(
                        {
                            "int32": torch.int32,
                            "int64": torch.int64,
                            "float32": torch.float32,
                            "float64": torch.float64,
                        }[input_dtype[k]]
                    )
                    for k, v in input_shape.items()
                ]
            )
        elif tensor_type == "tf":
            import tensorflow as tf

            dummy_input = [
                tf.TensorSpec(
                    v,
                    {
                        "int32": tf.int32,
                        "int64": tf.int64,
                        "float32": tf.float32,
                        "float64": tf.float64,
                    }[input_dtype[k]],
                    name="x",
                )
                for k, v in input_shape.items()
            ]
        else:
            dummy_input = dict(
                [
                    (k, np.random.rand(*v).astype(input_dtype[k]))
                    for k, v in input_shape.items()
                ]
            )
        return dummy_input

    dummy_input = contruct_dummy_input(input_shape, input_dtype, model_type)
    if model_type == "pt":
        import torch

        model.eval()
        # Export the model
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            "/home/mac_yuan/models/%s/%s.onnx" % (domain_type, model_name),
            export_params=True,  # store the trained parameter weights inside the model file
            do_constant_folding=True,  # whether to execute constant folding for optimization
        )


example_dict = {
    "job_id": "100003",
    "status": 0,
    "model_name": "mymodel",
    "model_path": "/home/mac_yuan/repo/dl_acceleration/inputs/model.onnx",
    "platform_type": 0,
    "model_type": 2,
    "target": "llvm -mcpu cascadelake -num-cores 8",
    "model_config": {
        "input_shape": {
            "input_ids": [10, 512],
            "attention_mask": [10, 512],
            "token_type_ids": [10, 512],
        },
        "input_dtype": {
            "input_ids": "int64",
            "attention_mask": "int64",
            "token_type_ids": "int64",
        },
    },
    "tuning_config": {
        "mode": "ansor", 
        "num_measure_trials": 10},
    "tuned_log": "/home/mac_yuan/repo/dl_acceleration/outputs/tuninglog_network_name=sentence-transformers_all-MiniLM-L6-v2--target=llvm -mcpu cascadelake -num-cores 8_finished.json",
    "error_info": "Exception Error Info",
    "need_benchmark": True,
}

example_creaet_vm = "gcloud compute instances create-with-container instance-1000002 --project=gnomondigital-sandbox --zone=europe-west1-b --machine-type=n1-standard-1 --boot-disk-size=100G --container-image=gcr.io/gnomondigital-sandbox/optimizer:0.1 --service-account=tvm-job-runner@gnomondigital-sandbox.iam.gserviceaccount.com --scopes=cloud-platform  --container-command=bash"

example_run = "docker run -ti -v=$HOME/.config/gcloud:/root/.config/gcloud gcr.io/gnomondigital-sandbox/optimizer:0.1 bash"


def generate_json_config(
    example_dict,
    batch_size_start,
    batch_size_end,
    batch_size_step,
    num_measure_trials_start,
    num_measure_trials_end,
    num_measure_trials_step,
):
    import json
    from pathlib import Path
    cpt = 0
    Path("jsonconfigs/").mkdir(exist_ok=True)
    for batch_size in range(batch_size_start, batch_size_end + 1, batch_size_step):
        for num_measure_trials in range(
            num_measure_trials_start,
            num_measure_trials_end + 1,
            num_measure_trials_step,
        ):
            cpt += 1
            job_id = int(example_dict["job_id"])
            job_id += 1
            example_dict["job_id"] = str(job_id)
            for k, v in example_dict["model_config"]["input_shape"].items():
                v[0] = batch_size
                example_dict["model_config"]["input_shape"][k] = v
            example_dict["tuning_config"]["num_measure_trials"] = num_measure_trials
            with open("jsonconfigs/job_id=%s.json" % str(job_id), "w") as output_file:
                json.dump(example_dict, output_file)
    print("generate_json_config generated %d config" % cpt)
            