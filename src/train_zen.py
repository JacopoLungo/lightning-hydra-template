from typing import Any

import lightning as L
import torch
from dotenv import load_dotenv
from hydra_zen import store, zen
from lightning import Callback
from omegaconf import DictConfig, OmegaConf

import configs_zen  # noqa: F401 to register the configs in the hydra store
from src.utils import (
    RankedLogger,
    task_wrapper,
    # validate_callbacks,
)

from typing import Any

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    log_hyperparameters,
    task_wrapper,
)

load_dotenv()

# Set TF32 precision for faster computation on Ampere+ GPUs
# Using new API (old torch.set_float32_matmul_precision will be deprecated after PyTorch 2.9)
torch.backends.cuda.matmul.fp32_precision = 'tf32'  # 'tf32' for high performance, 'ieee' for highest precision
torch.backends.cudnn.conv.fp32_precision = 'tf32'

log = RankedLogger(__name__, rank_zero_only=True)


def count_true_in_dict(d: DictConfig) -> int:
    """
    Iterates through a dictionary and counts the number of values that are True.
    """
    if not isinstance(d, DictConfig):
        return 0
    return sum(1 for value in d.values() if value is True)


# Register the new resolver with OmegaConf under the name "count_true".
# you can use it in your config files like this: ${count_true:your_dict}
OmegaConf.register_new_resolver("count_true", count_true_in_dict)


@task_wrapper
def train(datamodule: LightningDataModule, model: LightningModule, trainer: Trainer, **cfg) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # cfg = types.SimpleNamespace(**cfg)
    # cfg = DictConfig(cfg)
    log.info(f"Datamodule: <{type(datamodule).__name__}>")

    log.info(f"Model <{type(model).__name__}>")

    if trainer.logger:
        log.info(f'Logger: <{type(trainer.logger).__name__}>')
    else:
        log.warning("No logger found! Skipping...")

    if trainer.callbacks:
        log.info(f"Callbacks: {[type(callback).__name__ for callback in trainer.callbacks]}")
    else:
        log.warning("No callbacks found! Skipping..")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg["seed"], workers=True)

    log.info(f"Trainer <{type(trainer).__name__}>")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": trainer.callbacks,
        "logger": trainer.logger,
        "trainer": trainer,
    }

    if trainer.logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def main() -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    # train the model
    # metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    # return metric_value


if __name__ == "__main__":
    store.add_to_hydra_store(overwrite_ok=True)
    zen(train, unpack_kwargs=True).hydra_main(
        # config_path=None,  # We don't use a YAML config path
        config_name="train",  # We use our stored Python config
        version_base="1.3",
    )
