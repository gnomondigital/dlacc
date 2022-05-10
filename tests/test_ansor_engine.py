from hf_optimum import Optimum
from testings import get_encoded_input, default_target


def test_export_import(log_file, output_path, encoded_input, traced_model_file=None):
    opt = Optimum()
    opt.run(default_target, traced_model_file=traced_model_file, log_file=log_file)
    result = opt.load_model(output_path)(**encoded_input)
    print(result)


if __name__ == "__main__":
    batch_size = 10
    encoded_input = get_encoded_input(
        batch_size, "sentence-transformers/all-MiniLM-L6-v2"
    )
    test_export_import(
        "./tuning_log/network_name=camembert-base--target=llvm -mcpu=skylake-avx512--num_measure_trials=20000--batch_size=%d_finished.json"
        % batch_size,
        "./optimized_models/sentence-transformers_all-MiniLM-L6-v2",
        encoded_input,
        traced_model_file="/home/mac_yuan/repo/dl_acceleration/jit_traced_models/jit_traced_sentence-transformers_all-MiniLM-L6-v2.pt",
    )
