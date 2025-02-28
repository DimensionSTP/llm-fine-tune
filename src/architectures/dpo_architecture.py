from typing import Dict, Any
import copy
import os

import pandas as pd

import torch
from torch import optim, nn
from torch.nn import functional as F

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from transformers import AutoTokenizer


class CausalLMArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        pretrained_model_name: str,
        is_sft: bool,
        is_preprocessed: bool,
        custom_data_encoder_path: str,
        left_padding: bool,
        dpo_beta: float,
        strategy: str,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        eta_min_ratio: float,
        interval: str,
        options: Dict[str, Any],
        target_max_length: int,
        target_min_length: int,
        per_device_save_path: str,
        chosen_column_name: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.reference_model = copy.deepcopy(model)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.pretrained_model_name = pretrained_model_name
        self.is_sft = is_sft
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
        self.dpo_beta = dpo_beta
        self.strategy = strategy
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.eta_min_ratio = eta_min_ratio
        self.interval = interval
        self.options = options
        self.target_max_length = target_max_length
        self.target_min_length = target_min_length
        self.per_device_save_path = per_device_save_path
        self.chosen_column_name = chosen_column_name

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
        model: nn.Module,
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        if mode == "train":
            model.train()
            output = model(encoded)
        elif mode == "eval":
            model.eval()
            with torch.no_grad():
                output = model(encoded)
        else:
            raise ValueError(f"Invalid model mode: {mode}")
        return output

    def step(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        encoded_choice = batch["encoded_choice"]
        if not self.is_sft:
            encoded_choice["labels"] = encoded_choice["input_ids"]

        encoded_rejection = batch["encoded_rejection"]
        if not self.is_sft:
            encoded_rejection["labels"] = encoded_rejection["input_ids"]

        label = encoded_choice["labels"]
        index = batch["index"]

        chosen_model_output = self(
            encoded=encoded_choice,
            model=self.model,
            mode=mode,
        )
        rejected_model_output = self(
            encoded=encoded_rejection,
            model=self.model,
            mode=mode,
        )
        chosen_reference_output = self(
            encoded=encoded_choice,
            model=self.reference_model,
            mode="eval",
        )
        rejected_reference_output = self(
            encoded=encoded_rejection,
            model=self.reference_model,
            mode="eval",
        )

        chosen_model_logit = chosen_model_output.logits
        rejected_model_logit = rejected_model_output.logits
        chosen_reference_logit = chosen_reference_output.logits
        rejected_reference_logit = rejected_reference_output.logits

        chosen_model_log_prob = F.log_softmax(
            chosen_model_logit,
            dim=-1,
        )
        rejected_model_log_prob = F.log_softmax(
            rejected_model_logit,
            dim=-1,
        )
        chosen_reference_log_prob = F.log_softmax(
            chosen_reference_logit,
            dim=-1,
        )
        rejected_reference_log_prob = F.log_softmax(
            rejected_reference_logit,
            dim=-1,
        )

        prefered_relative_log_prob = chosen_model_log_prob - chosen_reference_log_prob
        disprefered_relative_log_prob = (
            rejected_model_log_prob - rejected_reference_log_prob
        )

        loss = -F.logsigmoid(
            (prefered_relative_log_prob - disprefered_relative_log_prob) * self.dpo_beta
        ).mean()

        prefered_relative_log_probability = prefered_relative_log_prob.mean()
        disprefered_relative_log_probability = disprefered_relative_log_prob.mean()

        reward_accuracy = (
            (prefered_relative_log_prob > disprefered_relative_log_prob).float().mean()
        )
        reward_margin = (
            prefered_relative_log_prob - disprefered_relative_log_prob
        ).mean()

        pred = torch.argmax(
            chosen_model_logit,
            dim=-1,
        )
        return {
            "loss": loss,
            "prefered_relative_log_probability": prefered_relative_log_probability,
            "disprefered_relative_log_probability": disprefered_relative_log_probability,
            "reward_accuracy": reward_accuracy,
            "reward_margin": reward_margin,
            "logit": chosen_model_logit,
            "pred": pred,
            "label": label,
            "index": index,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif (
            self.strategy == "deepspeed_stage_2_offload"
            or self.strategy == "deepspeed_stage_3_offload"
        ):
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        t_max = total_steps - warmup_steps
        eta_min = self.lr * self.eta_min_ratio

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                main_scheduler,
            ],
            milestones=[
                warmup_steps,
            ],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
            },
        }

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="train",
        )
        loss = output["loss"]
        prefered_relative_log_probability = output["prefered_relative_log_probability"]
        disprefered_relative_log_probability = output[
            "disprefered_relative_log_probability"
        ]
        reward_accuracy = output["reward_accuracy"]
        reward_margin = output["reward_margin"]
        pred = output["pred"]
        label = output["label"]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_prefered_relative_log_probability",
            prefered_relative_log_probability,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_disprefered_relative_log_probability",
            disprefered_relative_log_probability,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_reward_accuracy",
            reward_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_reward_margin",
            reward_margin,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        loss = output["loss"]
        prefered_relative_log_probability = output["prefered_relative_log_probability"]
        disprefered_relative_log_probability = output[
            "disprefered_relative_log_probability"
        ]
        reward_accuracy = output["reward_accuracy"]
        reward_margin = output["reward_margin"]
        pred = output["pred"]
        label = output["label"]
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_prefered_relative_log_probability",
            prefered_relative_log_probability,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_disprefered_relative_log_probability",
            disprefered_relative_log_probability,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_reward_accuracy",
            reward_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_reward_margin",
            reward_margin,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(
            batch=batch,
            mode="eval",
        )
        loss = output["loss"]
        prefered_relative_log_probability = output["prefered_relative_log_probability"]
        disprefered_relative_log_probability = output[
            "disprefered_relative_log_probability"
        ]
        reward_accuracy = output["reward_accuracy"]
        reward_margin = output["reward_margin"]
        pred = output["pred"]
        label = output["label"]
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_prefered_relative_log_probability",
            prefered_relative_log_probability,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_disprefered_relative_log_probability",
            disprefered_relative_log_probability,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_reward_accuracy",
            reward_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_reward_margin",
            reward_margin,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        encoded = batch["encoded_choice"]
        index = batch["index"]
        device_num = self.device.index if self.device.index is not None else 0

        output = self.model.generate(
            encoded=encoded,
            options=self.options,
            target_max_length=self.target_max_length,
            target_min_length=self.target_min_length,
        )
        generation = output.sequences
        input_length = len(encoded["input_ids"][0])
        generation = generation[:, input_length:]

        decoded_generation = self.data_encoder.batch_decode(
            sequences=generation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        index_list = index.tolist()
        cleaned_generation = list(
            map(
                lambda sentence: sentence.replace("\n", " ").replace("\r", " "),
                decoded_generation,
            )
        )
        output = {index_list[i]: cleaned_generation[i] for i in range(len(index_list))}
        os.makedirs(
            f"{self.per_device_save_path}/generations",
            exist_ok=True,
        )
        generation_file = f"{self.per_device_save_path}/generations/device_num={device_num}-batch_idx={batch_idx}.csv"
        df = pd.DataFrame(
            {
                "index": output.keys(),
                self.chosen_column_name: output.values(),
            }
        )
        if not os.path.exists(generation_file):
            df.to_csv(
                generation_file,
                mode="w",
                header=True,
                index=False,
            )
        else:
            raise FileExistsError(f"{generation_file} already exists")

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def on_save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint.pop(
            "reference_model",
            None,
        )
