import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="cpt.yaml",
)
def merge_predictions(
    config: DictConfig,
) -> None:
    generation_dfs = []
    per_device_generation_save_path = os.path.join(
        config.per_device_save_path,
        "generations",
    )
    for per_device_file_name in os.listdir(per_device_generation_save_path):
        if per_device_file_name.endswith(".csv"):
            per_device_generation_file_path = os.path.join(
                per_device_generation_save_path,
                per_device_file_name,
            )
            per_device_generation_df = pd.read_csv(per_device_generation_file_path)
            per_device_generation_df.fillna("_")
            generation_dfs.append(per_device_generation_df)

    generation_df_path = os.path.join(
        config.connected_dir,
        "data",
        f"{config.submission_file_name}.csv",
    )
    generation_df = pd.read_csv(generation_df_path)
    combined_generation_df = pd.concat(generation_dfs)
    sorted_generation_df = combined_generation_df.sort_values(by="index").reset_index()
    all_generations = sorted_generation_df[config.target_column_name]
    if len(all_generations) < len(generation_df):
        raise ValueError(
            f"Length of all_generations {len(all_generations)} is shorter than length of predict data {len(generation_df)}."
        )
    if len(all_generations) > len(generation_df):
        all_generations = all_generations[: len(generation_df)]
    generation_df[config.target_column_name] = all_generations

    submission_save_path = os.path.join(
        config.connected_dir,
        "submissions",
    )
    os.makedirs(
        submission_save_path,
        exist_ok=True,
    )

    submission_file_path = os.path.join(
        submission_save_path,
        f"{config.submission_name}.csv",
    )
    generation_df.to_csv(
        submission_file_path,
        index=False,
    )


if __name__ == "__main__":
    merge_predictions()
