import torch


def export(model, onnx_model_path, seq_len=384):
    with torch.no_grad():
        inputs = {
            "input_ids": torch.ones(1, seq_len, dtype=torch.int64),
            "segment_ids": torch.ones(1, seq_len, dtype=torch.int64),
            "input_mask": torch.ones(1, seq_len, dtype=torch.int64),
        }

        with torch.no_grad():
            model(*inputs.values())

        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            model,  # model being run
            (
                inputs["input_ids"],  # model input (or a tuple for multiple inputs)
                inputs["segment_ids"],
                inputs["input_mask"],
            ),  # model input (or a tuple for multiple inputs)
            onnx_model_path,  # where to save the model (can be a file or file-like object)
            opset_version=13,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=[
                "input_ids",  # the model's input names
                "segment_ids",
                "input_mask",
            ],
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input_ids": symbolic_names,  # variable length axes
                "segment_ids": symbolic_names,
                "input_mask": symbolic_names,
            },
        )
