from hydra_zen import store

from src.data.mnist_datamodule import MNISTDataModule

data_store = store(group="datamodule")

data_store(
    MNISTDataModule,
    data_dir="test",
    batch_size=128,  # Needs to be divisible by the number of devices (e.g., if in a distributed setup)
    train_val_test_split=[55_000, 5_000, 10_000],
    num_workers=0,
    pin_memory=False,
    name="mnist",
)
