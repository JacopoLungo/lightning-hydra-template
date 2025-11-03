from typing import Any

# import hydra  # No longer needed for hydra.main or hydra.utils.instantiate
import lightning as L
import torch
from dotenv import load_dotenv
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# --- New hydra-zen imports ---
from hydra_zen import builds, make_config, zen, store, ZenStore

# Import your util functions (assuming they are in src.utils)
from src.utils import (
    RankedLogger,
    extras,  # Note: extras(cfg) is not called in this pattern
    get_metric_value,
    # instantiate_callbacks,  # No longer needed, hydra-zen handles this
    # instantiate_loggers,    # No longer needed, hydra-zen handles this
    log_hyperparameters,
    task_wrapper, # No longer needed, zen() replaces this
)

# --- Imports for placeholder component classes ---
# !! IMPORTANT: Replace these with your actual class imports
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from my_project.data import MyDataModule    # e.g., from src.data.my_datamodule import MyDataModule
from my_project.models import MyModel         # e.g., from src.models.my_model import MyModel


load_dotenv()

torch.set_float32_matmul_precision("high")  # 'medium' or 'high'

log = RankedLogger(__name__, rank_zero_only=True)


def count_true_in_dict(d: DictConfig) -> int:
    """
    Iterates through a dictionary and counts the number of values that are True.
    """
    if not isinstance(d, DictConfig):
        return 0
    return sum(1 for value in d.values() if value is True)


# Register the resolver. We will do this just before running the app.
# OmegaConf.register_new_resolver("count_true", count_true_in_dict)


# @task_wrapper # The zen() wrapper provides similar robustness
def train_task(
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    logger: list[Logger],  # Injected for object_dict
    callbacks: list[Callback],  # Injected for object_dict
    cfg: DictConfig,  # Hydra injects the full config here
    seed: int | None = None,
    do_train: bool = True,
    do_test: bool = True,
    ckpt_path: str | None = None,
    optimized_metric: str | None = None,
) -> float | None:
    """
    The main training task, refactored to accept instantiated objects
    and configuration primitives.
    """
    # set seed for random number generators
    if seed:
        L.seed_everything(seed, workers=True)

    log.info(f"Using datamodule <{datamodule.__class__.__name__}>")
    log.info(f"Using model <{model.__class__.__name__}>")
    log.info(f"Using trainer <{trainer.__class__.__name__}>")

    # The 'cfg' is the raw DictConfig, useful for logging
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
        # This util function still works as it gets the full dict
        log_hyperparameters(object_dict)

    if do_train:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if do_test:
        log.info("Starting testing!")
        # Ensure checkpoint_callback exists before accessing
        ckpt_path_to_test = ckpt_path
        if trainer.checkpoint_callback:
             ckpt_path_to_test = trainer.checkpoint_callback.best_model_path
             if ckpt_path_to_test == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path_to_test = None
        else:
            log.warning("No checkpoint callback found. Using provided ckpt_path for testing.")

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path_to_test)
        log.info(f"Best ckpt path: {ckpt_path_to_test}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # This logic was originally in your `main` function
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=optimized_metric
    )

    return metric_value


# We no longer need the old `train` or `main` functions,
# as `train_task` and the `zen` wrapper have replaced them.

if __name__ == "__main__":
    
    # 1. Create a hydra-zen store
    # This will hold our Python-based configs
    hz_store = ZenStore()

    # 2. --- Define Component Configs ---
    # Use `builds` to create configs from your classes.
    # `populate_full_signature=True` automatically adds all __init__
    # args to the config.
    #
    # !! IMPORTANT: Replace these placeholders with your *actual* classes
    # and default values from your old YAML files.

    DataConfig = builds(
        MyDataModule,  # e.g., src.data.MyDataModule
        batch_size=32,
        num_workers=4,
        populate_full_signature=True,
    )

    ModelConfig = builds(
        MyModel,  # e.g., src.models.MyModel
        lr=0.001,
        populate_full_signature=True,
    )

    # `hydra-zen` can instantiate dicts of configs into lists.
    # This replaces your `instantiate_callbacks` function.
    CallbacksConfig = {
        "model_checkpoint": builds(
            ModelCheckpoint,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best-model",
        ),
        "early_stopping": builds(
            EarlyStopping, monitor="val/loss", mode="min", patience=3
        ),
    }

    # This replaces your `instantiate_loggers` function.
    LoggerConfig = {
        "wandb": builds(WandbLogger, project="my-project", name=None)
    }
    
    # The Trainer config can use string interpolation to refer to other
    # components in the config, which will be injected at instantiation.
    TrainerConfig = builds(
        Trainer,
        accelerator="auto",
        devices=1,
        max_epochs=10,
        callbacks="${callbacks}",  # Injects the instantiated CallbacksConfig
        logger="${logger}",        # Injects the instantiated LoggerConfig
        populate_full_signature=True,
    )

    # 3. --- Assemble Main Task Config ---
    # `make_config` creates the final configuration dataclass.
    # Its parameters match the signature of `train_task`.
    TaskConfig = make_config(
        # Primitives for train_task
        seed=42,
        do_train=True,
        do_test=True,
        ckpt_path=None,
        optimized_metric="val/loss",

        # Components to be instantiated and passed to train_task
        datamodule=DataConfig,
        model=ModelConfig,
        callbacks=CallbacksConfig,
        logger=LoggerConfig,
        trainer=TrainerConfig,  # This will be instantiated last

        # Config settings
        hydra_convert="all",
    )

    # 4. Store the config and register the resolver
    hz_store(TaskConfig, name="train_app")
    OmegaConf.register_new_resolver("count_true", count_true_in_dict)
    hz_store.add_to_hydra_store(overwrite_ok=True)

    # 5. --- Run the Application ---
    # `zen` wraps our task function.
    # `hydra_main` runs the Hydra app.
    # This replaces the need for `@hydra.main` entirely.
    zen(train_task).hydra_main(
        config_path=None,  # We don't use a YAML config path
        config_name="train_app",  # We use our stored Python config
        version_base="1.3",
    )