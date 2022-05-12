from hf_optimum import Optimum
from testings import get_encoded_input, default_target
from transformers import AutoModel


def test_export_import(log_file, output_path, encoded_input, traced_model_file=None):
    opt = Optimum()
    model = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", return_dict=False
    )

    opt.run(
        default_target,
        model,
        encoded_input,
        traced_model_file=traced_model_file,
        log_file=log_file,
    )
    model = opt.load_model(output_path)
    result = model(encoded_input)
    print(result)


if __name__ == "__main__":
    batch_size = 10
    encoded_input = get_encoded_input(
        batch_size, "sentence-transformers/all-MiniLM-L6-v2"
    )
    log_file = (
        "./tuning_log/network_name=sentence-transformers_all-MiniLM-L6-v2--target=llvm -mcpu=skylake-avx512--num_measure_trials=%d--batch_size=10_finished.json"
        % (batch_size)
    )
    test_export_import(
        log_file,
        "./optimized_models/sentence-transformers_all-MiniLM-L6-v2",
        encoded_input,
    )
