import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import json

from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from tqdm import tqdm

import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="cpt.yaml",
)
def convert_zero_ckpt(
    config: DictConfig,
) -> None:
    if config.is_tuned == "tuned":
        params = json.load(
            open(
                config.tuned_hparams_path,
                "rt",
                encoding="UTF-8",
            )
        )
        config = OmegaConf.merge(
            config,
            params,
        )
    elif config.is_tuned == "untuned":
        pass
    else:
        raise ValueError(f"Invalid is_tuned argument: {config.is_tuned}")

    if config.strategy.startswith("deepspeed"):
        for root, dirs, _ in os.walk(config.callbacks.model_checkpoint.dirpath):
            for dir_name in tqdm(dirs):
                if dir_name.endswith(".ckpt"):
                    ckpt_path = os.path.join(
                        root,
                        dir_name,
                    )
                    converted_ckpt_path = os.path.join(
                        ckpt_path,
                        "model.pt",
                    )
                    if not os.path.exists(converted_ckpt_path):
                        convert_zero_checkpoint_to_fp32_state_dict(
                            ckpt_path,
                            converted_ckpt_path,
                        )


if __name__ == "__main__":
    convert_zero_ckpt()
