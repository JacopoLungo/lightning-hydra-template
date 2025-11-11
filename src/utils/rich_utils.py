from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil
import rich
import rich.syntax
import rich.tree
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import RichProgressBar
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


class RamGpuUsageRichProgressBar(RichProgressBar):
    """A custom RichProgressBar that includes RAM and GPU usage metrics.

    you can use it by instantiating it callback/rich_progress_bar.yaml
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ram_text: str = "N/A"
        self.gpu_text: str = "N/A"

    def get_metrics(self, trainer, model) -> dict[str, Any]:
        """Adds RAM and GPU usage to the default metrics."""
        items = super().get_metrics(trainer, model)

        # Calculate Total system RAM usage for the current process tree
        current_process = psutil.Process()
        total_ram_bytes = current_process.memory_info().rss
        try:
            for child in current_process.children(recursive=True):
                try:
                    total_ram_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            self.ram_text = f"{total_ram_bytes / (1024**3):.2f}GB"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.ram_text = "Error"

        items["RAM"] = self.ram_text

        # GPU memory metrics
        if torch.cuda.is_available() and torch.cuda.is_initialized():  # Ensure CUDA is initialized
            try:
                reserved_bytes = torch.cuda.memory_reserved(0)
                self.gpu_text = f"{reserved_bytes / (1024**3):.2f}GB"
            except Exception:
                self.gpu_text = "Error"
        elif torch.cuda.is_available():
            self.gpu_text = "CUDA init pending"
        else:
            self.gpu_text = "N/A (No CUDA)"

        items["GPU"] = self.gpu_text

        return items
