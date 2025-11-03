import logging
from typing import Any

import hydra
import pytorch_lightning as L

# --- Imports from hydra-zen ---
from hydra_zen import builds, make_config, store, zen
from omegaconf import DictConfig

# --- Imports for a typical PL setup ---
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, Logger

# A global logger for the application
log = logging.getLogger(__name__)

# --------------------------------------------------------------------
# 1. DEFINE YOUR PLACEHOLDER CLASSES AND HELPER FUNCTIONS
# (I'm mocking these based on your train function)
# --------------------------------------------------------------------


class MyModel(LightningModule):
    # Your model definition...
    def __init__(self, lr: float = 0.01, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # ...


class MyDataModule(LightningDataModule):
    # Your data definition...
    def __init__(self, batch_size: int = 32, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # ...


# Your train function uses these custom helpers, so we mock them
def instantiate_callbacks(callbacks_cfg: Any) -> list[Callback]:
    """Mocks your callback instantiation."""
    return [hydra.utils.instantiate(cb) for cb in callbacks_cfg]


def instantiate_loggers(logger_cfg: Any) -> list[Logger]:
    """Mocks your logger instantiation."""
    return [hydra.utils.instantiate(lg) for lg in logger_cfg]


def log_hyperparameters(object_dict: dict[str, Any]):
    """Mocks your hyperparameter logging."""
    pass


# Your task wrapper (can be a simple passthrough)
def task_wrapper(fn):
    return fn


# --------------------------------------------------------------------
# 2. YOUR ORIGINAL TRAIN FUNCTION (Unchanged)
# --------------------------------------------------------------------


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    (This is your function, unchanged)
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
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


# --------------------------------------------------------------------
# 3. THE MISSING PIECE: DEFINE AND STORE YOUR CONFIGS
# --------------------------------------------------------------------

# Define configs for your components
ConfigData = builds(MyDataModule, batch_size=64, populate_full_signature=True)
ConfigModel = builds(MyModel, lr=0.001, populate_full_signature=True)

# Define configs for callbacks and loggers (as lists, like your train fn expects)
ConfigCallbacks = [builds(ModelCheckpoint, monitor="val_loss", mode="min", save_top_k=1)]

ConfigLoggers = [builds(CSVLogger, save_dir="logs/")]

# Define config for the Trainer
ConfigTrainer = builds(Trainer, max_epochs=10, populate_full_signature=True)

# Use make_config to create the main "TrainConfig"
# This is the Python object that defines the *entire* config
TrainConfig = make_config(
    # These keys (data, model, etc.) MUST match cfg.data, cfg.model
    # in your train() function
    data=ConfigData(),
    model=ConfigModel(),
    trainer=ConfigTrainer(),
    callbacks=ConfigCallbacks,
    logger=ConfigLoggers,
    # Add other top-level parameters
    seed=42,
    train=True,
    test=True,
    ckpt_path=None,
)

# **This is the most important step you were missing:**
# Add your main TrainConfig to the store with name="train"
store(TrainConfig, name="train")


# --------------------------------------------------------------------
# 4. YOUR CORRECTED MAIN FUNCTION
# --------------------------------------------------------------------


def main() -> float | None:
    """Main entry point for training."""

    # This call now finds `TrainConfig` and registers it
    # with Hydra's Config Store under the name "train"
    store.add_to_hydra_store(overwrite_ok=True)

    zen(train).hydra_main(
        config_name="train",  # This now works!
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
