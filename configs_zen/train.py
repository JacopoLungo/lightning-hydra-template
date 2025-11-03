from hydra_zen import MISSING, make_config, store

TrainConfig = make_config(
    hydra_defaults=[
        "_self_",
        {"datamodule": "mnist"},
        {"model": "mnist"},
        # {"callbacks": "default"},
        # {"logger": "wandb"},
        {"trainer": "default"},
        {"paths": "default"},
    ],
    datamodule=MISSING,
    model=MISSING,
    # callbacks=MISSING,
    # logger=MISSING,
    trainer=MISSING,
    paths=MISSING,
    task_name="train",
    type_run_name="runs",
    run_name="${now:%Y-%m-%d}_${now:%H-%M-%S}",
    tags=["dev"],
    seed=None,
    train=True,
    test=True,
    ckpt_path=None,
    optimized_metric=None,
)

store(TrainConfig, name="train")
