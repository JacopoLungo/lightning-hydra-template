import lightning.pytorch as lp
import torch
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
from typing import List, Dict, Any, Optional

class WandBImageLogger(lp.Callback):
    def __init__(self,
                visualize_images_every_n_epochs: int = 1,
                n_batches_to_visualize: int = 5,
                random_batches: bool = False,
                image_log_fn_name: str = "prepare_images_for_logging",
                log_on_train_epoch: bool = False,
                log_on_val_epoch: bool = True
                ):
        super().__init__()
        if visualize_images_every_n_epochs < 0:
            raise ValueError("visualize_images_every_n_epochs must be non-negative.")
        if n_batches_to_visualize < -1:
            raise ValueError("n_batches_to_visualize must be >= -1.")

        self.visualize_images_every_n_epochs = visualize_images_every_n_epochs
        self.n_batches_to_visualize = n_batches_to_visualize
        self.random_batches = random_batches
        self.image_log_fn_name = image_log_fn_name
        self.log_on_train_epoch = log_on_train_epoch
        self.log_on_val_epoch = log_on_val_epoch

        self._train_batch_indices_to_log: Optional[np.ndarray] = None
        self._val_batch_indices_to_log: Optional[np.ndarray] = None
        self._do_log_on_current_train_epoch: bool = False
        self._do_log_on_current_val_epoch: bool = False
        
        if not log_on_train_epoch and not log_on_val_epoch and self.visualize_images_every_n_epochs > 0:
            print("WandBImageLogger: Warning! Image logging is enabled but both log_on_train_epoch and log_on_val_epoch are False.")
    
    def _setup_epoch_visualization(self, trainer: lp.Trainer, mode: str):
        
        should_log_this_epoch = (trainer.current_epoch % self.visualize_images_every_n_epochs == 0)
        
        if mode == 'train':
            self._do_log_on_current_train_epoch = should_log_this_epoch
            if not self._do_log_on_current_train_epoch: return
            
            # when using IterableDataset, num_training_batches is float('inf') -> log only first n batches (-1 option not supported)
            if isinstance(trainer.num_training_batches, float) and trainer.num_training_batches == float('inf'):
                if trainer.global_rank == 0:
                    print("WandBImageLogger: Iterable training dataset. Will log first `n_batches_to_visualize` batches if not random.")
                if self.n_batches_to_visualize > 0:
                    self._train_batch_indices_to_log = np.arange(self.n_batches_to_visualize)
                else:
                    self._train_batch_indices_to_log = np.array([], dtype=int)
                return
            
            num_batches = trainer.num_training_batches
            batch_indices_attr = '_train_batch_indices_to_log'
        else: # val
            self._do_log_on_current_val_epoch = should_log_this_epoch
            if not self._do_log_on_current_val_epoch: return
            # when there are no val dataloader
            if not trainer.num_val_batches or len(trainer.num_val_batches) == 0:
                if trainer.global_rank == 0:
                    print(f"WandBImageLogger: No validation batches for dataloader_idx 0. Skipping val image logging for epoch {trainer.current_epoch}.")
                self._do_log_on_current_val_epoch = False
                return
            num_batches = trainer.num_val_batches[0]
            batch_indices_attr = '_val_batch_indices_to_log'
        
        if self.n_batches_to_visualize == -1:
            indices = np.arange(num_batches)
        else:
            num_to_pick = min(self.n_batches_to_visualize, num_batches)
            if self.random_batches:
                indices = np.random.choice(num_batches, size=num_to_pick, replace=False)
            else:
                indices = np.linspace(0, num_batches - 1, num=num_to_pick, dtype=int)
        setattr(self, batch_indices_attr, np.sort(indices))
    
    def on_train_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        if self.log_on_train_epoch and \
        isinstance(trainer.logger, WandbLogger) and \
        self.visualize_images_every_n_epochs != 0 and \
        trainer.num_training_batches > 0 and \
        self.n_batches_to_visualize != 0:
            self._setup_epoch_visualization(trainer, mode='train')
        else:
            self._do_log_on_current_train_epoch = False
    
    def on_validation_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        if self.log_on_val_epoch and \
        isinstance(trainer.logger, WandbLogger) and \
        self.visualize_images_every_n_epochs != 0 and \
        trainer.num_val_batches[0] > 0 and \
        self.n_batches_to_visualize != 0:
            self._setup_epoch_visualization(trainer, mode='val')
        else:
            self._do_log_on_current_val_epoch = False
    
    def on_train_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: Optional[Dict[str, Any]], # This `outputs` is from training_step
        batch: Any,
        batch_idx: int,
    ):
        if not self.log_on_train_epoch or not self._do_log_on_current_train_epoch or \
            not isinstance(trainer.logger, WandbLogger) or \
            self._train_batch_indices_to_log is None or \
            batch_idx not in self._train_batch_indices_to_log:
            return
        
        if not hasattr(pl_module, self.image_log_fn_name):
            if trainer.global_rank == 0:
                print(f"WandBImageLogger: model_lit_module missing '{self.image_log_fn_name}'. Skipping for \'train\' batch {batch_idx}.")
            return
        
        image_log_fn = getattr(pl_module, self.image_log_fn_name)
        
        image_log_fn(batch, outputs, batch_idx, trainer.current_epoch, mode='train')
    
    def on_validation_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: Optional[Dict[str, Any]], # This `outputs` is from validation_step
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if dataloader_idx != 0:
            return
        if not self.log_on_val_epoch or not self._do_log_on_current_val_epoch or \
            not isinstance(trainer.logger, WandbLogger) or \
            self._val_batch_indices_to_log is None or \
            batch_idx not in self._val_batch_indices_to_log:
            return
        
        if not hasattr(pl_module, self.image_log_fn_name):
            if trainer.global_rank == 0:
                print(f"WandBImageLogger: model_lit_module missing '{self.image_log_fn_name}'. Skipping for \'val\' batch {batch_idx}.")
            return
        
        image_log_fn = getattr(pl_module, self.image_log_fn_name)
        image_log_fn(batch, outputs, batch_idx, trainer.current_epoch, mode='val')
