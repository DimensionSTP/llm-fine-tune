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
        chosen_column_name: str,
        rejected_column_name: str,
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
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name
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
        self.choices = dataset["choices"]
        self.rejections = dataset["rejections"]
        self.data_max_length = data_max_length
        self.target_max_length = target_max_length

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt_choice = self.generate_prompt(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.choices[idx],
        )
        encoded_choice = self.encode_text(
            data=prompt_choice,
            data_type="data",
        )
        if self.is_sft:
            label_choice = self.encode_text(
                data=self.choices[idx],
                data_type="target",
            )
            encoded_choice["labels"] = label_choice
        if "token_type_ids" in encoded_choice.keys():
            del encoded_choice["token_type_ids"]

        prompt_rejection = self.generate_prompt(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.rejections[idx],
        )
        encoded_rejection = self.encode_text(
            data=prompt_rejection,
            data_type="data",
        )
        if self.is_sft:
            label_rejection = self.encode_text(
                data=self.rejections[idx],
                data_type="target",
            )
            encoded_rejection["labels"] = label_rejection
        if "token_type_ids" in encoded_rejection.keys():
            del encoded_rejection["token_type_ids"]
        return {
            "encoded_choice": encoded_choice,
            "encoded_rejection": encoded_rejection,
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
        choices = data[self.chosen_column_name].apply(lambda x: x.strip()).tolist()
        rejections = data[self.rejected_column_name].apply(lambda x: x.strip()).tolist()
        return {
            "instructions": instructions,
            "datas": datas,
            "choices": choices,
            "rejections": rejections,
        }

    def encode_text(
        self,
        data: str,
        data_type: str,
    ) -> Dict[str, torch.Tensor]:
        if data_type == "data":
            if self.split == "predict":
                max_length = self.data_max_length
            else:
                max_length = self.data_max_length + self.target_max_length
        elif data_type == "target":
            max_length = self.target_max_length
        else:
            raise ValueError(f"Inavalid data_type: {data_type}")
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
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

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction} 

### Input:
{data}

### Response:
{label} """
        return prompt
