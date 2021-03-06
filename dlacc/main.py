import traceback
import argparse
from pathlib import Path

from optimum import Optimum
from utils import (
    get_onnx_model,
    upload_outputs,
    infer_platform_type,
    JSONConfig,
    JSONOutput,
)
from metadata import platformType, output_prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="The path of config file in json format.",
        required=True,
    )

    args = parser.parse_args()

    config = JSONConfig(args.path, infer_platform_type(args.path))
    onnx_model = get_onnx_model(
        config["platform_type"],
        config["model_path"],
        config["model_type"],
        input_shape=config["model_config"]["input_shape"],
        input_dtype=config["model_config"]["input_dtype"],
    )
    Path(output_prefix).mkdir(exist_ok=True)
    out_json = JSONOutput(config)
    out_json["status"] = 1
    try:
        optimum = Optimum(out_json["model_name"])
        optimum.run(onnx_model, out_json)
    except Exception as e:
        traceback.print_exc()
        out_json["error_info"] = str(e)
        out_json["status"] = -1

    if out_json["status"] != -1:
        out_json["status"] = 4
    out_json.save(output_prefix + "/output_json.json")
    if config["need_benchmark"]:
        optimum.ansor_engine.evaluate()

    upload_outputs(
        out_json["output_bucket"],
        "job_id=%s" % config["job_id"],
        platformType.GOOGLESTORAGE,
    )
