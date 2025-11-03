"""Hydra-Zen configs for models.

This module demonstrates how to migrate from YAML configs to typed Python configs using hydra-zen.

Benefits over YAML:
- Full type safety and IDE autocomplete
- Compile-time validation instead of runtime errors
- Easier refactoring (rename/find references work)
- Programmatic config generation
- Can use Python logic and conditionals
"""

from hydra_zen import builds
import torch
from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.mnist_module import MNISTLitModule


# Build configs for network components
SimpleDenseNetConfig = builds(
    SimpleDenseNet,
    input_size=784,
    lin1_size=64,
    lin2_size=128,
    lin3_size=64,
    output_size=10,
)


# Build configs for optimizer (partial=True means it returns a callable)
AdamConfig = builds(
    torch.optim.Adam,
    lr=0.001,
    weight_decay=0.0,
    populate_full_signature=True,  # Include all available parameters
    zen_partial=True,  # Equivalent to _partial_: true in YAML
)


# Build configs for scheduler
ReduceLROnPlateauConfig = builds(
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    mode="min",
    factor=0.1,
    patience=10,
    populate_full_signature=True,
    zen_partial=True,
)


# Build the full model config by composing the components
MNISTModelConfig = builds(
    MNISTLitModule,
    net=SimpleDenseNetConfig,  # Nested config
    optimizer=AdamConfig,  # Nested config
    scheduler=ReduceLROnPlateauConfig,  # Nested config
    compile=False,
    populate_full_signature=True,
)


# You can also create variants easily in Python
MNISTModelConfigLargeLR = builds(
    MNISTLitModule,
    net=SimpleDenseNetConfig,
    optimizer=builds(torch.optim.Adam, lr=0.01, weight_decay=0.0, zen_partial=True),  # Override LR
    scheduler=ReduceLROnPlateauConfig,
    compile=False,
    populate_full_signature=True,
)


# Example: Config with compilation enabled
MNISTModelConfigCompiled = builds(
    MNISTLitModule,
    net=SimpleDenseNetConfig,
    optimizer=AdamConfig,
    scheduler=ReduceLROnPlateauConfig,
    compile=True,  # Enable torch.compile for faster training
    populate_full_signature=True,
)
