from utils import from_hf_pretrained
from hf_optimum import Optimum
import numpy as np
import argparse
import timeit


def benchmark(network_name, batch_size, target, log_file):
    example_batch_input = ["This is an example sentence", "Each sentence is converted"]
    tokenizer, model = from_hf_pretrained(network_name)
    encoded_input = tokenizer(
        example_batch_input, padding=True, truncation=True, return_tensors="pt"
    )
    optimum = Optimum(model, network_name)
    optimum.run(encoded_input, target, num_measure_trials=500, log_file=log_file)
    optimized_model = optimum.get_best_model()
    ftimer = optimized_model(encoded_input, time_evaluater=True)

    to_comp = (
        np.array(
            timeit.Timer(lambda: model(**encoded_input)).repeat(repeat=3, number=10)
        )
        / 10
    )

    return np.array(ftimer().results) * 1000, to_comp * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="llvm",
        help="The compilation target.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="The log file path.",
    )
    # Benchmark
    networks = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "camembert-base",
        "facebook/bart-base",
        "roberta-base",
        "distilgpt2",
        "bert-base-uncased",
    ]
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    args = parser.parse_args()
    result_messages = []
    for network in networks[:1]:
        for batch_size in [args.batch_size]:
            print("Benchmark %s ..." % network)
            prof_res, to_comp_res = benchmark(
                network, batch_size, args.target, args.log_file
            )
            message = (
                "%-18s %-12s %-19s (%s)"
                % (
                    network,
                    batch_size,
                    "%.2f ms" % np.mean(prof_res),
                    "%.2f ms" % np.std(prof_res),
                ),
                "%-18s %-12s %-19s (%s)"
                % (
                    network,
                    batch_size,
                    "%.2f ms" % np.mean(to_comp_res),
                    "%.2f ms" % np.std(to_comp_res),
                ),
            )
            result_messages.append(message)

    # Print result
    print("-------------------------------------------------------------")
    print(
        "%-18s %-12s %-20s"
        % ("Network Name", "Batch size", "Mean Inference Time (std dev)")
    )
    print("-------------------------------------------------------------")
    for line in result_messages:
        print(line)
    print("-------------------------------------------------------------")
