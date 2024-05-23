from transformers import AutoTokenizer, AutoModelForCausalLM

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def merge_model(
    config: DictConfig,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.custom_data_encoder_path}/{config.pretrained_model_name}"
    )
    model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(
        f"{config.connected_dir}/data/merged_model/{config.pretrained_model_name}"
    )


if __name__ == "__main__":
    merge_model()
