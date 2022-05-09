from utils import from_hf_pretrained
from hf_optimum import Optimum
import numpy as np
import argparse
import timeit
import datetime
from sklearn.datasets import fetch_20newsgroups
import os
import glob
import tvm
import pandas as pd


texts = fetch_20newsgroups(subset="train").data
num_measure_trials = 20000

def describe(np_array):
    return pd.DataFrame(np_array).describe()


def logfile_loader(network_name, target, num_measure_trials, batch_size):
    network_name = network_name.replace("/", "_")
    path = (
        "./tuning_log/network_name=%s--target=%s--num_measure_trials=%d--batch_size=%d_finished.json"
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
    dev = tvm.device(str(target), 0)
    print("Evaluate inference time cost...")
    timing_results = optimum.ansor_engine.module.benchmark(dev, repeat=3, number=10, end_to_end=True)
    to_comp = (
        np.array(
            timeit.Timer(lambda: model(**encoded_input)).repeat(repeat=3, number=10)
        )
        / 10
    ) 
    print("Evaluate inference time finished...")
    return np.array(timing_results.results) * 1000, to_comp * 1000


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
            df_optimized = pd.DataFrame(prof_res).describe()
            df_original = pd.DataFrame(to_comp_res).describe()
            result_df = pd.concat([df_optimized,df_original], axis=1)
            result_df.columns = ["optimized", "original"]
            print(result_df)
            result_file.write("network_name=%s  batch_size=%s:\n" % (network, batch_size))
            result_file.write(str(result_df) + "\n")
            mean_1, mean_2 = result_df.loc["mean"].values[0], result_df.loc["mean"].values[1]
            percent = 0 if mean_1 > mean_2 else (mean_2 - mean_1) / mean_2
            result_file.write("mean improvement = %.2f%" % (percent*100))
            print("Results written to %s" % (result_file_name))
            result_messages.append(result_df)

