from transformers import AutoTokenizer
from sklearn.datasets import fetch_20newsgroups

texts = fetch_20newsgroups(subset="train").data
default_target = "llvm -mcpu=skylake-avx512"
network_list = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "camembert-base",
    "facebook/bart-base",
    "roberta-base",
    "distilgpt2",
    "bert-base-uncased",
]


def get_encoded_input(batch_size, network_name):
    tokenizer = AutoTokenizer.from_pretrained(network_name)
    encoded_input = tokenizer(
        texts[:batch_size], padding=True, truncation=True, return_tensors="pt"
    )
    return encoded_input
