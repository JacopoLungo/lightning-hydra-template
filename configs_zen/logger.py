from hydra_zen import builds, store
from lightning.pytorch.loggers import WandbLogger

logger_store = store(group="trainer/logger")

WandBConfig = builds(
    WandbLogger,
    name="${type_run_name}_${run_name}",
    save_dir="${paths.output_dir}",
    offline=False,
    id=None,
    anonymous=None,
    project="test",
    log_model=False,
    prefix="",
    group="",
    tags=[],
    job_type="",
)

logger_store(WandBConfig, name="wandb")
logger_store([], name="none")
