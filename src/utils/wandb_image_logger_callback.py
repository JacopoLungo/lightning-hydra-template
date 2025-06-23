import lightning.pytorch as lp
import torch
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
from typing import List, Dict, Any, Optional

class WandBImageLogger(lp.Callback):
    """
    A Lightning Callback for logging images to Weights & Biases (W&B).
    
    This callback logs images from the training and/or validation loop to a W&B run.
    It is designed to be flexible, allowing for logging at specified epoch or global
    step intervals.
    
    When logging by epoch, you can configure the number of batches to visualize.
    When logging by step, it logs the single batch at the specified global step.
    
    The actual image preparation and logging logic is delegated to a method
    within the LightningModule. This keeps the data processing and visualization
    logic with the model, while this callback handles the triggering and scheduling.
    
    Args:
        log_every_n_epochs (int, optional):
            Frequency of epochs to log images. If 1, logs every epoch. If 0 or None,
            this form of logging is disabled. Defaults to 1.
            Mutually exclusive with `log_every_n_steps`.
        log_every_n_steps (int, optional):
            Frequency of global steps to log images. If set, this will trigger logging
            at every `n` global steps. Defaults to None.
            Mutually exclusive with `log_every_n_epochs`.
        n_batches_to_visualize (int, optional):
            Number of batches to visualize per epoch when using epoch-based logging.
            If -1, all batches are logged. This parameter is ignored if `log_every_n_steps`
            is used. Defaults to 5.
        random_batches (bool, optional):
            If True, randomly selects `n_batches_to_visualize` from the available
            batches for epoch-based logging. If False, batches are chosen at evenly
            spaced intervals. Ignored for step-based logging. Defaults to False.
        image_log_fn_name (str, optional):
            The name of the method in the LightningModule that prepares and logs
            the images. This method should accept `batch`, `outputs`, `batch_idx`,
            `epoch`, and `mode` as arguments. Defaults to "log_visualizations".
        log_on_train_epoch (bool, optional):
            If True, enables logging for the training loop. Defaults to False.
        log_on_val_epoch (bool, optional):
            If True, enables logging for the validation loop. Defaults to True.
    
    Raises:
        ValueError: If both `log_every_n_epochs` and `log_every_n_steps` are specified.
        ValueError: If `log_every_n_epochs` or `log_every_n_steps` is negative.
        ValueError: If `n_batches_to_visualize` is less than -1.
        
    Example Usage in a LightningModule:
    
    .. code-block:: python
        
        class MyLitModule(lp.LightningModule):
            # ... (your model definition) ...
            
            def log_visualizations(self,
                        batch: Any,
                        step_outputs: Dict[str, Any],
                        batch_idx: int,
                        current_epoch: int,
                        mode: str) -> Dict[str, List[Any]]:
                
                pred_name_in_step_outputs = "preds"
                idx_of_img_in_batch = 0
                
                normalized_img = batch[0][idx_of_img_in_batch].detach() #! first element of batch should be images 
                unnormalize_img = unnormalize_image(normalized_img)
                unnormalize_image_np = unnormalize_img.permute(1, 2, 0).cpu().numpy()
                
                lbl = batch[1][idx_of_img_in_batch].detach().cpu() #! second element of batch should be labels
                pred = step_outputs[pred_name_in_step_outputs][idx_of_img_in_batch]
                
                wandb_image = wandb.Image(unnormalize_image_np, caption=f"lbl_{lbl}_pred_{pred}")
                
                log_data = {}
                log_data[f"img/{mode}"] = wandb_image
                
                self.logger.experiment.log({**log_data, "trainer/global_step": self.global_step})
        
        # In your training script:
        
        # For epoch-based logging (log 5 batches every 2 epochs)
        # trainer = lp.Trainer(
        #     logger=WandbLogger(project="my_project"),
        #     callbacks=[WandBImageLogger(log_every_n_epochs=2, n_batches_to_visualize=5)]
        # )
        
        # For step-based logging (log every 100 global steps)
        # trainer = lp.Trainer(
        #     logger=WandbLogger(project="my_project"),
        #     callbacks=[WandBImageLogger(log_every_n_steps=100)]
        # )
        # trainer.fit(model, datamodule)
    """
    def __init__(self,
                log_every_n_epochs: Optional[int] = 1,
                log_every_n_steps: Optional[int] = None,
                n_batches_to_visualize: int = 5,
                random_batches: bool = False,
                image_log_fn_name: str = "log_visualizations",
                log_on_train_epoch: bool = False,
                log_on_val_epoch: bool = True
                ):
        super().__init__()
        
        if log_every_n_epochs is not None and log_every_n_steps is not None:
            raise ValueError("You can only specify one of `log_every_n_epochs` or `log_every_n_steps`, not both.")
        if log_every_n_epochs is not None and log_every_n_epochs < 0:
            raise ValueError("log_every_n_epochs must be non-negative.")
        if log_every_n_steps is not None and log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive.")
        if n_batches_to_visualize < -1:
            raise ValueError("n_batches_to_visualize must be >= -1.")
        
        self.log_every_n_epochs = log_every_n_epochs if log_every_n_epochs is not None else 0
        self.log_every_n_steps = log_every_n_steps
        self.n_batches_to_visualize = n_batches_to_visualize
        self.random_batches = random_batches
        self.image_log_fn_name = image_log_fn_name
        self.log_on_train_epoch = log_on_train_epoch
        self.log_on_val_epoch = log_on_val_epoch
        
        self._train_batch_indices_to_log: Optional[np.ndarray] = None
        self._val_batch_indices_to_log: Optional[np.ndarray] = None
        self._do_log_on_current_train_epoch: bool = False
        self._do_log_on_current_val_epoch: bool = False
        
        is_logging_configured = (self.log_every_n_epochs > 0) or (self.log_every_n_steps is not None)
        if not log_on_train_epoch and not log_on_val_epoch and is_logging_configured:
            print("WandBImageLogger: Warning! Image logging is enabled but both log_on_train_epoch and log_on_val_epoch are False.")
    
    def _setup_epoch_visualization(self, trainer: lp.Trainer, mode: str):
        """Pre-calculates which batch indices to log for epoch-based logging."""
        if self.log_every_n_steps is not None:
            # This setup is only for epoch-based logging
            return
        
        should_log_this_epoch = (self.log_every_n_epochs > 0 and trainer.current_epoch % self.log_every_n_epochs == 0)
        
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
            
            if not trainer.val_dataloaders or not trainer.num_val_batches:
                if trainer.global_rank == 0:
                    print(f"WandBImageLogger: No validation batches detected. Skipping val image logging for epoch {trainer.current_epoch}.")
                self._do_log_on_current_val_epoch = False
                return
            
            num_batches = trainer.num_val_batches[0] # Assuming single val dataloader
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
        if not self.log_on_train_epoch or self.n_batches_to_visualize == 0:
            self._do_log_on_current_train_epoch = False
            return
        self._setup_epoch_visualization(trainer, mode='train')
    
    def on_validation_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        if not self.log_on_val_epoch or self.n_batches_to_visualize == 0:
            self._do_log_on_current_val_epoch = False
            return
        self._setup_epoch_visualization(trainer, mode='val')
    
    def _log_image_batch(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
        mode: str
    ):
        """Central logging logic for both step and epoch-based logging."""
        if not isinstance(trainer.logger, WandbLogger):
            return
        
        # Check if the logging function exists on the model
        if not hasattr(pl_module, self.image_log_fn_name):
            if trainer.global_rank == 0:
                print(f"WandBImageLogger: LightningModule missing '{self.image_log_fn_name}'. Skipping image logging.")
            return
        
        should_log = False
        # --- Step-based logging ---
        if self.log_every_n_steps is not None:
            if trainer.global_step > 0 and trainer.global_step % self.log_every_n_steps == 0:
                should_log = True
        
        # --- Epoch-based logging ---
        else:
            is_epoch_log_batch = False
            if mode == 'train':
                if self._do_log_on_current_train_epoch and self._train_batch_indices_to_log is not None:
                    is_epoch_log_batch = batch_idx in self._train_batch_indices_to_log
            elif mode == 'val':
                if self._do_log_on_current_val_epoch and self._val_batch_indices_to_log is not None:
                    is_epoch_log_batch = batch_idx in self._val_batch_indices_to_log
            
            if is_epoch_log_batch:
                should_log = True
        
        # If we should log, execute the model's logging function
        if should_log:
            image_log_fn = getattr(pl_module, self.image_log_fn_name)
            image_log_fn(batch, outputs, batch_idx, trainer.current_epoch, mode=mode)
    
    def on_train_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
        ):
        if not self.log_on_train_epoch:
            return
        
        self._log_image_batch(trainer, pl_module, outputs, batch, batch_idx, mode='train')
    
    def on_validation_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        ):
        if not self.log_on_val_epoch or dataloader_idx != 0:
            return
        
        self._log_image_batch(trainer, pl_module, outputs, batch, batch_idx, mode='val')