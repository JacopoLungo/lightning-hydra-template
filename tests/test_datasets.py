"""Tests for dataset components."""

from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize
from src.data.components.base_dataset import BaseDataset

# List of datasets to test - add new datasets here
DATASETS_TO_TEST = ["dataset_config_name"]
CONFIG_BASE_PATH = "configs/data"

@pytest.mark.requires_data
@pytest.mark.parametrize("dataset_name", DATASETS_TO_TEST)
def test_dataset(dataset_name: str) -> None:
    """Test that dataset can be instantiated from Hydra config.

    This test works for any dataset that inherits from BaseDataset.

    :param dataset_config: Configuration for the dataset.
    """
    if dataset_name not in [p.stem for p in Path(CONFIG_BASE_PATH).glob("*.yaml")]:
        pytest.skip(f"Dataset config for {dataset_name} does not exist")

    with initialize(version_base="1.3", config_path=f"../{CONFIG_BASE_PATH}"):
        cfg = compose(config_name=f"{dataset_name}.yaml")

    # -- Instantiate Dataset Tests --
    # Instantiate dataset using Hydra
    dataset = hydra.utils.instantiate(cfg)

    ##########################################
    # From here on you can test your dataset #
    ##########################################
    
    # Check dataset properties
    assert dataset.num_classes > 0
    assert len(dataset.all_class_names) > 0
    assert len(dataset.class_names_to_encode) > 0
    assert len(dataset) > 0

    # -- Get Item Tests --
    # Get first item
    item = dataset[0]

    # Check item structure
    assert "image" in item
    assert "mask" in item
    assert "image_path" in item
    assert "mask_path" in item

    # Check tensor properties
    assert isinstance(item["image"], torch.Tensor)
    assert isinstance(item["mask"], torch.Tensor)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.uint8

    # Check image shape (C, H, W)
    assert item["image"].ndim == 3
    assert item["image"].shape[0] == 3  # RGB channels

    # Check mask shape (H, W)
    assert item["mask"].ndim == 2

    # -- Dataloader Tests --
    batch_size = 1

    # Create dataloader using the dataset's method
    dataloader = dataset.create_dataloader(batch_size=batch_size, num_workers=0)
    # Get first batch
    batch = next(iter(dataloader))

    # Check batch structure
    assert "image" in batch
    assert "mask" in batch
    assert "image_path" in batch
    assert "mask_path" in batch

    # Check batch dimensions
    expected_batch_size = min(batch_size, len(dataset))
    assert batch["image"].shape[0] == expected_batch_size
    assert batch["mask"].shape[0] == expected_batch_size
    assert len(batch["image_path"]) == expected_batch_size
    assert len(batch["mask_path"]) == expected_batch_size

    # Check batch tensor properties
    assert batch["image"].dtype == torch.float32
    assert batch["mask"].dtype == torch.uint8

    # Check batch image shape (B, C, H, W)
    assert batch["image"].ndim == 4
    assert batch["image"].shape[1] == 3  # RGB channels

    # Check batch mask shape (B, H, W)
    assert batch["mask"].ndim == 3
