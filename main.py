from utils import JSONConfig, JSONOutput
import argparse

from optimum import Optimum
from utils import convert2onnx, source_type_infer, upload
from metadata import SourceType, output_prefix
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path of config file in json format.",
        required=True,
    )
    args = parser.parse_args()
    config = JSONConfig("/" + args.config, source_type_infer(args.config))
    onnx_model, input_shape = convert2onnx(config["source_type"], config["source_value"], config["model_type"])
    Path(output_prefix).mkdir(exist_ok=True)
    out_json = JSONOutput(config)
    out_json["status"] = 1
    try:
        optimum = Optimum(config["model_name"], onnx_model, out_json)
        optimum.run(
            config["target"],
            config["tuning_config"]["num_measure_trials"],
            config["tuning_config"]["mode"],
            out_json,
            log_file=config["history"]["tunned_log"],
            input_shape=config["model_config"]["input_shape"],
            input_dtype=config["model_config"]["input_dtype"]
        )
    except Exception as e:
        print(e)
        out_json["error_info"] = str(e)
        out_json["status"] = 5
    
    if out_json["status"] != 5:
        out_json["status"] = 4
    out_json.save(output_prefix + "/output_json.json")
    optimum.ansor_engine.evaluate()

    upload(
        "gs://gnomondigital-sdx-tvm-turning-job-output/"
        + "job_id=%s" % out_json["job_id"],
        SourceType.GOOGLESTORAGE
    )


