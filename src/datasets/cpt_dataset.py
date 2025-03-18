from typing import Dict, Any, List
import os

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
        dataset_name: str,
        dataset_format: str,
        is_sft: bool,
        is_preprocessed: bool,
        instruction_column_name: str,
        data_column_name: str,
        target_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        data_max_length: int,
        target_max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.instruction_column_name = instruction_column_name
        self.data_column_name = data_column_name
        self.target_column_name = target_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
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

        if self.data_encoder.chat_template is None:
            reference_data_encoder = AutoTokenizer.from_pretrained(
                reference_data_encoder_name
            )
            self.data_encoder.chat_template = reference_data_encoder.chat_template

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

        self.response_start_template = "### Start"
        self.response_start_tokens = self.data_encoder(
            self.response_start_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.response_end_template = "### End"
        self.response_end_tokens = self.data_encoder(
            self.response_end_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt = self.apply_chat_template(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.labels[idx],
        )
        encoded = self.encode_text(data=prompt)
        if self.is_sft:
            encoded = self.add_sft_label(encoded=encoded)
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

    def apply_chat_template(
        self,
        instruction: str,
        data: str,
        label: str,
    ) -> str:
        if self.is_sft:
            label = f"{self.response_template}{label}"

        conversation = [
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": data,
            },
        ]
        if self.split != "predict":
            conversation.append(
                {
                    "role": "assistant",
                    "content": label,
                }
            )

        prompt = self.data_encoder.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=False if self.split != "predict" else True,
        )
        return prompt

    def encode_text(
        self,
        data: str,
    ) -> Dict[str, torch.Tensor]:
        max_length = self.data_max_length + self.target_max_length
        if self.split == "predict":
            max_length = self.data_max_length

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

    def find_pattern_indices(
        self,
        input_ids: torch.Tensor,
        pattern: torch.Tensor,
    ) -> List[int]:
        pattern_length = pattern.size(0)

        if pattern_length > input_ids.size(0):
            return []

        indices = []
        for i in range(input_ids.size(0) - pattern_length + 1):
            if torch.equal(input_ids[i : i + pattern_length], pattern):
                indices.append(i)
        return indices

    def add_sft_label(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        input_ids = encoded["input_ids"]
        labels = torch.full_like(
            input_ids,
            self.ignore_index,
        )

        start_indices = self.find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_start_tokens,
        )
        end_indices = self.find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_end_tokens,
        )

        end_idx_pos = 0
        for start_idx in start_indices:
            content_start = start_idx + len(self.response_start_tokens)

            while end_idx_pos < len(end_indices):
                if end_indices[end_idx_pos] > start_idx:
                    content_end = end_indices[end_idx_pos]
                    labels[content_start:content_end] = input_ids[
                        content_start:content_end
                    ]
                    end_idx_pos += 1
                    break
                end_idx_pos += 1

            else:
                labels[content_start:] = input_ids[content_start:]

        encoded["labels"] = labels
        return encoded


class ConversationalDataset(StructuralDataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        is_sft: bool,
        is_preprocessed: bool,
        conversation_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        data_max_length: int,
        target_max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.conversation_column_name = conversation_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
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

        if self.data_encoder.chat_template is None:
            reference_data_encoder = AutoTokenizer.from_pretrained(
                reference_data_encoder_name
            )
            self.data_encoder.chat_template = reference_data_encoder.chat_template

        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        if left_padding:
            self.data_encoder.padding_side = "left"
        else:
            self.data_encoder.padding_side = "right"

        dataset = self.get_dataset()
        self.conversations = dataset["conversations"]
        self.data_max_length = data_max_length
        self.target_max_length = target_max_length

        self.response_start_template = "### Start"
        self.response_start_tokens = self.data_encoder(
            self.response_start_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.response_end_template = "### End"
        self.response_end_tokens = self.data_encoder(
            self.response_end_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt = self.apply_chat_template(
            conversation=self.conversations[idx],
        )
        encoded = self.encode_text(data=prompt)
        if self.is_sft:
            encoded = self.add_sft_label(encoded=encoded)
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val", "test", "predict"]:
            file_name = f"{self.dataset_name}.{self.dataset_format}"
            full_data_path = os.path.join(
                self.data_path,
                file_name,
            )
            data = pd.read_parquet(full_data_path)
            data = data.fillna("_")
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.split == "val":
            _, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            data = val_data

        if self.split == "predict" and self.num_devices > 1:
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

        conversations = data[self.conversation_column_name].tolist()
        return {
            "conversations": conversations,
        }

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
    ) -> str:
        preprocessed_conversation = []
        for turn in conversation:
            if (
                turn[self.role_column_name] == self.assistant_column_name
                and self.is_sft
            ):
                content = f"\n{self.response_start_template}\n{turn[self.content_column_name]}\n{self.response_end_template}\n"
                preprocessed_turn = {
                    self.role_column_name: turn[self.role_column_name],
                    self.content_column_name: content,
                }
                preprocessed_conversation.append(preprocessed_turn)
            else:
                preprocessed_turn = {
                    self.role_column_name: turn[self.role_column_name],
                    self.content_column_name: turn[self.content_column_name],
                }
                preprocessed_conversation.append(preprocessed_turn)

        if self.split == "predict":
            preprocessed_conversation.pop()

        prompt = self.data_encoder.apply_chat_template(
            conversation=preprocessed_conversation,
            tokenize=False,
            add_generation_prompt=False if self.split != "predict" else True,
        )
        return prompt
