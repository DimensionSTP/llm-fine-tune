import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def merge_tokenizer(
    config: DictConfig,
) -> None:
    korean_tokenizer = AutoTokenizer.from_pretrained(config.korean_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    def is_korean(token):
        for char in token:
            if (
                "\uAC00" <= char <= "\uD7A3"
                or "\u1100" <= char <= "\u11FF"
                or "\u3130" <= char <= "\u318F"
            ):
                return True
        return False

    korean_tokenizer_tokens = korean_tokenizer.get_vocab().keys()
    korean_tokens = [token for token in korean_tokenizer_tokens if is_korean(token)]
    korean_tokens = korean_tokens[: config.add_vocab_size]

    tokenizer_tokens = tokenizer.get_vocab().keys()
    new_tokens = [token for token in korean_tokens if token not in tokenizer_tokens]
    tokenizer.add_tokens(new_tokens)

    if not os.path.exists(config.custom_data_encoder_path):
        os.makedirs(
            config.custom_data_encoder_path,
            exist_ok=True,
        )
    tokenizer.save_pretrained(config.custom_data_encoder_path)


if __name__ == "__main__":
    merge_tokenizer()
