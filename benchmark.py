from utils import from_hf_pretrained
from hf_optimum import Optimum
import numpy as np
import argparse
import timeit
import datetime
from sklearn.datasets import fetch_20newsgroups
import os
import glob

texts = fetch_20newsgroups(subset="train").data


def logfile_loader(network_name, target, num_measure_trials, batch_size):
    network_name = network_name.replace("/", "_")
    path = (
        "./tuning_log/network_name=%s--target=%s--num_measure_trials=%d--batch_size=%d.json"
        % (network_name, target, num_measure_trials, batch_size)
    )
    file = glob.glob(path)
    return file


def benchmark(network_name, batch_size, target, log_file, num_measure_trials=1000):
    example_batch_input = texts[:batch_size]
    tokenizer, model = from_hf_pretrained(network_name)
    encoded_input = tokenizer(
        example_batch_input, padding=True, truncation=True, return_tensors="pt"
    )
    optimum = Optimum(model, network_name)
    optimum.run(
        encoded_input, target, num_measure_trials=num_measure_trials, log_file=log_file
    )
    optimized_model = optimum.get_best_model()
    time_res = optimized_model(encoded_input, time_evaluator=True)
    to_comp = (
        np.array(
            timeit.Timer(lambda: model(**encoded_input)).repeat(repeat=3, number=10)
        )
        / 10
    )
    return time_res * 1000, to_comp * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="llvm",
        help="The compilation target.",
    )
    parser.add_argument(
        "--logfile",
        default=[],
        nargs="*",
        help="The log file path. A list corresponding to each network.",
    )
    # Benchmark
    networks = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "camembert-base",
        "facebook/bart-base",
        "roberta-base",
        "distilgpt2",
        "bert-base-uncased",
        "xlm-roberta-large-finetuned-conll03-english",
        "nlptown/bert-base-multilingual-uncased-sentiment",
    ]
    parser.add_argument(
        "--batchsize_range",
        type=int,
        nargs="+",
        default=[],
        help="The batch size range, a triple (start, end, step), for example: (100, 1000, 100)",
    )
    args = parser.parse_args()
    result_file_name = "./benchmark/result-%s.txt" % datetime.datetime.now().strftime(
        "%d-%m-%Y-%H:%M:%S"
    )
    os.makedirs(os.path.dirname(result_file_name), exist_ok=True)
    result_file = open(result_file_name, "w")
    num_measure_trials = 10
    result_messages = []
    for i, network in enumerate(networks):
        for batch_size in range(
            args.batchsize_range[0], args.batchsize_range[1], args.batchsize_range[2]
        ):
            now = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            print("[%s]Benchmark %s, batch_size=%d ..." % (now, network, batch_size))
            files = logfile_loader(network, args.target, num_measure_trials, batch_size)
            logfile = files[0] if len(files) else None
            prof_res, to_comp_res = benchmark(
                network,
                batch_size,
                args.target,
                logfile,
                num_measure_trials=num_measure_trials,
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
            print("Performance Comparison:")
            print(message)
            for msg in message:
                result_file.write(msg + "\n")
            result_messages.append(message)

    # # Print result
    # print("-----------------Original Execution--------------------------")
    # print(
    #     "%-18s %-12s %-20s"
    #     % ("Network Name", "Batch size", "Mean Inference Time (std dev)")
    # )
    # for line in result_messages:
    #     print(line[1])
    # print("-------------------------------------------------------------")
    # print("-----------------Optimized Execution--------------------------")
    # print(
    #     "%-18s %-12s %-20s"
    #     % ("Network Name", "Batch size", "Mean Inference Time (std dev)")
    # )
    # for line in result_messages:
    #     print(line[0])
    # print("-------------------------------------------------------------")
