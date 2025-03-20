from typing import Dict, Any
import os
import json

from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..architectures import CPTCausalLMArchitecture


class CausalLMTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        tracking_direction: str,
        seed: int,
        num_trials: int,
        hparams_save_path: str,
        architecture_config: DictConfig,
        trainer_config: DictConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: EarlyStopping,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.direction = f"{tracking_direction}imize"
        self.seed = seed
        self.num_trials = num_trials
        self.hparams_save_path = hparams_save_path

        self.architecture_config = architecture_config
        self.trainer_config = trainer_config

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(
            self.optuna_objective,
            n_trials=self.num_trials,
        )
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score: {best_score}")
        print(f"Parameters: {best_params}")

        os.makedirs(
            self.hparams_save_path,
            exist_ok=True,
        )

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(
                best_params,
                json_file,
            )

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        seed_everything(self.seed)

        params = dict()
        params["seed"] = self.seed
        params["architecture"] = dict()
        if self.hparams.lr:
            params["architecture"]["lr"] = trial.suggest_float(
                name="lr",
                low=self.hparams.lr.low,
                high=self.hparams.lr.high,
                log=self.hparams.lr.log,
            )
        if self.hparams.weight_decay:
            params["architecture"]["weight_decay"] = trial.suggest_float(
                name="weight_decay",
                low=self.hparams.weight_decay.low,
                high=self.hparams.weight_decay.high,
                log=self.hparams.weight_decay.log,
            )
        if self.hparams.warmup_ratio:
            params["architecture"]["warmup_ratio"] = trial.suggest_float(
                name="warmup_ratio",
                low=self.hparams.warmup_ratio.low,
                high=self.hparams.warmup_ratio.high,
                log=self.hparams.warmup_ratio.log,
            )
        if self.hparams.eta_min_ratio:
            params["architecture"]["eta_min_ratio"] = trial.suggest_float(
                name="eta_min_ratio",
                low=self.hparams.eta_min_ratio.low,
                high=self.hparams.eta_min_ratio.high,
                log=self.hparams.eta_min_ratio.log,
            )

        architecture: CPTCausalLMArchitecture = instantiate(
            self.architecture_config,
            **params["architecture"],
        )

        self.logger.log_hyperparams(params)

        trainer: Trainer = instantiate(
            self.trainer_config,
            enable_checkpointing=False,
            callbacks=self.callbacks,
            logger=self.logger,
            _convert_="partial",
        )

        try:
            trainer.fit(
                model=architecture,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
            self.logger.experiment.alert(
                title="Tuning Complete",
                text="Tuning process has successfully finished.",
                level="INFO",
            )
        except Exception as e:
            self.logger.experiment.alert(
                title="Tuning Error",
                text="An error occurred during tuning",
                level="ERROR",
            )
            raise e

        return trainer.callback_metrics[self.module_params.monitor].item()
