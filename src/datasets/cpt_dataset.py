from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class StructuralDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        is_sft: bool,
        is_preprocessed: bool,
        instruction_column_name: str,
        data_column_name: str,
        target_column_name: str,
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        left_padding: bool,
        data_max_length: int,
        target_max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.instruction_column_name = instruction_column_name
        self.data_column_name = data_column_name
        self.target_column_name = target_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.pretrained_model_name = pretrained_model_name
        if is_preprocessed:
            data_encoder_path = custom_data_encoder_path
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        if left_padding:
            self.data_encoder.padding_side = "left"
        else:
            self.data_encoder.padding_side = "right"
        dataset = self.get_dataset()
        self.instructions = dataset["instructions"]
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.data_max_length = data_max_length
        self.target_max_length = target_max_length
        self.response_template = "### Response:\n"
        self.response_template_tokens = self.data_encoder(
            self.response_template,
            add_special_tokens=False,
        )["input_ids"]
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt = self.generate_prompt(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.labels[idx],
        )
        encoded = self.encode_text(data=prompt)
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            parquet_path = f"{self.data_path}/train.parquet"
            data = pd.read_parquet(parquet_path)
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            parquet_path = f"{self.data_path}/{self.split}.parquet"
            data = pd.read_parquet(parquet_path)
            data = data.fillna("_")
        elif self.split == "predict":
            parquet_path = f"{self.data_path}/test.parquet"
            data = pd.read_parquet(parquet_path)
            data = data.fillna("_")
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")
        instructions = (
            data[self.instruction_column_name].apply(lambda x: x.strip()).tolist()
        )
        datas = data[self.data_column_name].apply(lambda x: x.strip()).tolist()
        labels = data[self.target_column_name].apply(lambda x: x.strip()).tolist()
        return {
            "instructions": instructions,
            "datas": datas,
            "labels": labels,
        }

    def encode_text(
        self,
        data: str,
    ) -> Dict[str, torch.Tensor]:
        if self.split == "predict":
            max_length = self.data_max_length
        else:
            max_length = self.data_max_length + self.target_max_length
        max_length = self.target_max_length

        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}

        if self.is_sft:
            try:
                response_start_idx = next(
                    i
                    for i in range(
                        len(encoded["input_ids"])
                        - len(self.response_template_tokens)
                        + 1
                    )
                    if encoded["input_ids"][i : i + len(self.response_template_tokens)]
                    == self.response_template_tokens
                )
                encoded["labels"] = encoded["input_ids"].clone()
                encoded["labels"][
                    : response_start_idx + len(self.response_template_tokens)
                ] = self.ignore_index
            except StopIteration:
                encoded["labels"] = encoded["input_ids"].clone()
        return encoded

    def generate_prompt(
        self,
        instruction: str,
        data: str,
        label: str,
    ) -> str:
        if self.split == "predict":
            prompt = f"""### Instruction:
{instruction} 

### Input:
{data}

{self.response_template}"""
        else:
            prompt = f"""### Instruction:
{instruction} 

### Input:
{data}

{self.response_template}{label} """
        return prompt
