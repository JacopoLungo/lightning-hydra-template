from hydra_zen import store, builds
from lightning.pytorch.trainer import Trainer


TrainerConfig = builds(Trainer,
    default_root_dir="${paths.output_dir}",
    min_epochs=1,
    max_epochs=10,
    accelerator="cpu",
    devices=1,
    callbacks="${callbacks}",
    logger="${logger}",
    check_val_every_n_epoch=1,
    deterministic=False,
    hydra_defaults=[
        "_self_",
        {"callbacks": "default"},
        {"logger": "wandb"},
    ],
)

store(TrainerConfig, group="trainer", name="default")


# store(
#     Trainer,
#     default_root_dir="${paths.output_dir}",
#     min_epochs=1,
#     max_epochs=10,
#     accelerator="cpu",
#     devices=1,
#     callbacks="${callbacks}",
#     logger="${logger}",
#     check_val_every_n_epoch=1,
#     deterministic=False,
#     group="trainer",
#     name="default",
# )
