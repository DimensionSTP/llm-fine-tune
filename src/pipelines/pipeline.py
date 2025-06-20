from typing import Union
import os

from hydra.utils import instantiate
from omegaconf import DictConfig

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from ..utils import SetUp
from ..tuners import *


def train(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    architecture = setup.get_architecture()
    callback_candidates = setup.get_callbacks()
    if config.early_stop:
        callbacks = [
            callback_candidates["model_checkpoint"],
            callback_candidates["early_stopping"],
        ]
    else:
        callbacks = [
            callback_candidates["model_checkpoint"],
        ]
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["eval_batch_size"] = config.eval_batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["step"] = config.step
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    for key, value in config.dataset.items():
        if key not in [
            "_target_",
            "data_path",
            "split",
            "seed",
        ]:
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    try:
        if isinstance(config.resumed_step, int):
            if config.resumed_step == 0:
                trainer.fit(
                    model=architecture,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )
            elif config.resumed_step > 0:
                trainer.fit(
                    model=architecture,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=f"{config.callbacks.model_checkpoint.dirpath}/last.ckpt",
                )
            else:
                raise ValueError(
                    f"Invalid resumed_step argument: {config.resumed_step}"
                )
        else:
            raise TypeError(f"Invalid resumed_step argument: {config.resumed_step}")
        logger.experiment.alert(
            title="Training Complete",
            text=f"Training process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Training Error",
            text=f"An error occurred during training on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e

    if config.strategy.startswith("deepspeed") and config.convert_at_end:
        for root, dirs, _ in os.walk(config.callbacks.model_checkpoint.dirpath):
            for dir_name in dirs:
                if dir_name.endswith(".ckpt"):
                    ckpt_path = os.path.join(
                        root,
                        dir_name,
                    )
                    convert_zero_checkpoint_to_fp32_state_dict(
                        ckpt_path,
                        f"{ckpt_path}/model.pt",
                    )


def test(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    architecture = setup.get_architecture()
    callback_candidates = setup.get_callbacks()
    if config.early_stop:
        callbacks = [
            callback_candidates["model_checkpoint"],
            callback_candidates["early_stopping"],
        ]
    else:
        callbacks = [
            callback_candidates["model_checkpoint"],
        ]
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.eval_batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["step"] = config.step
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    for key, value in config.dataset.items():
        if key not in [
            "_target_",
            "data_path",
            "split",
            "seed",
        ]:
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    if (
        config.strategy == "deepspeed_stage_3"
        or config.strategy == "deepspeed_stage_3_offload"
    ):
        trainer: Trainer = instantiate(
            config.trainer,
            strategy="ddp",
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )
    else:
        trainer: Trainer = instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

    try:
        if (
            config.strategy == "deepspeed_stage_3"
            or config.strategy == "deepspeed_stage_3_offload"
        ):
            trainer.test(
                model=architecture,
                dataloaders=test_loader,
                ckpt_path=f"{config.ckpt_path}/model.pt",
            )
        else:
            trainer.test(
                model=architecture,
                dataloaders=test_loader,
                ckpt_path=config.ckpt_path,
            )
        logger.experiment.alert(
            title="Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e


def predict(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    predict_loader = setup.get_predict_loader()
    architecture = setup.get_architecture()
    callback_candidates = setup.get_callbacks()
    if config.early_stop:
        callbacks = [
            callback_candidates["model_checkpoint"],
            callback_candidates["early_stopping"],
        ]
    else:
        callbacks = [
            callback_candidates["model_checkpoint"],
        ]
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.eval_batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["step"] = config.step
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    for key, value in config.dataset.items():
        if key not in [
            "_target_",
            "data_path",
            "split",
            "seed",
        ]:
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    if (
        config.strategy == "deepspeed_stage_3"
        or config.strategy == "deepspeed_stage_3_offload"
    ):
        trainer: Trainer = instantiate(
            config.trainer,
            strategy="ddp",
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )
    else:
        trainer: Trainer = instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

    try:
        if (
            config.strategy == "deepspeed_stage_3"
            or config.strategy == "deepspeed_stage_3_offload"
        ):
            trainer.predict(
                model=architecture,
                dataloaders=predict_loader,
                ckpt_path=f"{config.ckpt_path}/model.pt",
            )
        else:
            trainer.predict(
                model=architecture,
                dataloaders=predict_loader,
                ckpt_path=config.ckpt_path,
            )
        logger.experiment.alert(
            title="Predicting Complete",
            text=f"Predicting process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Predicting Error",
            text=f"An error occurred during predicting on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e


def tune(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    callback_candidates = setup.get_callbacks()
    callbacks = callback_candidates["early_stopping"]
    logger = setup.get_wandb_logger()

    tuner: Union[CPTCausalLMTuner, DPOCausalLMTuner] = instantiate(
        config.tuner,
        architecture_config=config.architecture,
        trainer_config=config.trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        logger=logger,
        _recursive_=False,
    )
    tuner()
