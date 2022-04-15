import imp
from transformers import AlbertConfig, AlbertModel
from transformers.models.albert.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/config.json",
    "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/config.json",
    "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/config.json",
    "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/config.json",
    "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/config.json",
    "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/config.json",
    "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/config.json",
    "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/config.json",
}

# Initializing an ALBERT-xxlarge style configuration
albert_config = AlbertConfig()

# Initializing an ALBERT-base style configuration
albert_base_configuration = AlbertConfig(
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
)

# Initializing a model from the ALBERT-base style configuration
model = AlbertModel(albert_config)

# Accessing the model configuration
configuration = model.config

