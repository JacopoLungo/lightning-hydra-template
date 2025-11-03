from hydra_zen import builds, make_config, store
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from omegaconf import MISSING

from src.utils.resource_monitor_callback import ResourceMonitorCallback
from src.utils.wandb_image_logger_callback import WandBImageLogger

# Dataclasses for Callbacks configurations

EarlyStoppingConfig = builds(
    EarlyStopping,
    monitor=MISSING,
    min_delta=0.0,
    patience=3,
    verbose=False,
    mode="min",
    strict=True,
    check_finite=True,
    stopping_threshold=None,
    divergence_threshold=None,
    check_on_train_epoch_end=None,
)

ModelCheckpointConfig = builds(
    ModelCheckpoint,
    dirpath=None,
    filename=None,
    monitor=None,
    verbose=False,
    save_last=None,
    save_top_k=1,
    mode="min",
    auto_insert_metric_name=True,
    save_weights_only=False,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
)

RichModelSummaryConfig = builds(RichModelSummary, max_depth=1, mode="model_summary")

ResourceMonitorConfig = builds(ResourceMonitorCallback)

RichProgressBarConfig = builds(RichProgressBar)

WandBImageLoggerConfig = builds(
    WandBImageLogger,
    log_on_train=False,
    log_on_val_test_pred=True,
    val_test_pred_log_every_n_epochs=1,
    val_test_pred_n_batches_to_visualize=-1,
    val_test_pred_random_batches=False,
    image_log_fn_name="log_visualizations",  # Must match the method in the LitModel
)

# Overwriting of dataclasses

checkpoint_on_val_conf = ModelCheckpointConfig(
    dirpath="${paths.output_dir}/checkpoints",
    filename="epoch_{epoch:03d}",
    monitor="val/acc",
    mode="max",
    save_last=True,
    auto_insert_metric_name=False,
)

# Composite Callback configuration
# NOTE: the groups are not swappable here

CallbackConfig = make_config(
    model_checkpoint=checkpoint_on_val_conf,
    early_stopping=EarlyStoppingConfig(monitor="val/acc", patience=100, mode="max"),
    rich_model_summary=RichModelSummaryConfig(max_depth=-1),
    resource_monitor=ResourceMonitorConfig,
    rich_progress_bar=RichProgressBarConfig,
    wandb_image_logger=WandBImageLoggerConfig,
    hydra_defaults=["_self_"],
)

# Helper function to convert callback config to list
def callbacks_to_list(**kwargs):
    """Convert callback config dict to list of callbacks."""
    return list(kwargs.values())


CallbacksListConfig = builds(
    callbacks_to_list,
    model_checkpoint=checkpoint_on_val_conf,
    early_stopping=EarlyStoppingConfig(monitor="val/acc", patience=100, mode="max"),
    rich_model_summary=RichModelSummaryConfig(max_depth=-1),
    resource_monitor=ResourceMonitorConfig,
    rich_progress_bar=RichProgressBarConfig,
    wandb_image_logger=WandBImageLoggerConfig,
    zen_convert={"dataclass": False},  # Keep as dict for easier overrides
)

# Store configurations in the "callbacks" group
store(CallbacksListConfig, group="trainer/callbacks", name="default")
store([], group="trainer/callbacks", name="none")
