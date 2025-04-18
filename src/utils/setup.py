from typing import Dict, Union
import os

from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import Dataset, DataLoader

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.data_type = self.config.data_type
        self.num_cpus = os.cpu_count()
        self.num_fit_workers = min(
            self.num_cpus,
            (config.devices * config.workers_ratio),
        )
        self.num_workers = (
            self.num_cpus if config.use_all_workers else self.num_fit_workers
        )

    def get_train_loader(self) -> DataLoader:
        train_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.train,
        )
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_val_loader(self) -> DataLoader:
        val_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.val,
        )
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        test_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.test,
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_predict_loader(self) -> DataLoader:
        predict_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.predict,
        )
        return DataLoader(
            dataset=predict_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_architecture(self) -> LightningModule:
        architecture: LightningModule = instantiate(
            self.config.architecture,
        )
        return architecture

    def get_callbacks(self) -> Dict[str, Union[ModelCheckpoint, EarlyStopping]]:
        model_checkpoint: ModelCheckpoint = instantiate(
            self.config.callbacks.model_checkpoint,
        )
        early_stopping: EarlyStopping = instantiate(
            self.config.callbacks.early_stopping,
        )
        return {
            "model_checkpoint": model_checkpoint,
            "early_stopping": early_stopping,
        }

    def get_wandb_logger(self) -> WandbLogger:
        wandb_logger: WandbLogger = instantiate(
            self.config.logger.wandb,
        )
        return wandb_logger
