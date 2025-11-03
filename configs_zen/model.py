import torch
from hydra_zen import builds, store

from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.mnist_module import MNISTLitModule

model_store = store(group="model")

AdamConfig = builds(torch.optim.Adam, lr=0.001, weight_decay=0.0, zen_partial=True)
SchedulerConfig = builds(
    torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10, zen_partial=True
)
NetConfig = builds(
    SimpleDenseNet, input_size=784, lin1_size=64, lin2_size=128, lin3_size=64, output_size=10, zen_partial=True
)

model_store(
    builds(MNISTLitModule, optimizer=AdamConfig, scheduler=SchedulerConfig, net=NetConfig, compile=False), name="mnist"
)
