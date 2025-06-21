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
    It is designed to be flexible, allowing for logging at specified epoch intervals
    and for a configurable number of batches.
    
    The actual image preparation and logging logic is delegated to a method
    within the LightningModule. This keeps the data processing and visualization
    logic with the model, while this callback handles the triggering and scheduling.
    
    Args:
        visualize_images_every_n_epochs (int, optional):
            Frequency of epochs to log images. If 1, logs every epoch.
            If 0, logging is disabled. Defaults to 1.
        n_batches_to_visualize (int, optional):
            Number of batches to visualize per epoch. If -1, all batches are logged.
            Defaults to 5.
        random_batches (bool, optional):
            If True, randomly selects `n_batches_to_visualize` from the available
            batches. If False, batches are chosen at evenly spaced intervals.
            Defaults to False.
        image_log_fn_name (str, optional):
            The name of the method in the LightningModule that prepares and logs
            the images. This method should accept `batch`, `outputs`, `batch_idx`,
            `epoch`, and `mode` as arguments. Defaults to "prepare_images_for_logging".
        log_on_train_epoch (bool, optional):
            If True, enables logging for the training loop. Defaults to False.
        log_on_val_epoch (bool, optional):
            If True, enables logging for the validation loop. Defaults to True.
    
    Raises:
        ValueError: If `visualize_images_every_n_epochs` is negative.
        ValueError: If `n_batches_to_visualize` is less than -1.
    
    Example Usage in a LightningModule:
    
    .. code-block:: python
    
        class MyLitModule(lp.LightningModule):
            def __init__(self, *args, **kwargs):
                super().__init__()
                # your model layers
                self.layer = torch.nn.Linear(32, 3)
            
            def forward(self, x):
                return self.layer(x)
            
            # ... training_step, configure_optimizers, etc. ...
            
            def validation_step(self, batch, batch_idx):
                # your validation logic
                x, y = batch
                y_hat = self(x)
                loss = torch.nn.functional.mse_loss(y_hat, y)
                self.log('val_loss', loss)
                # Return model outputs to be used in the logging function
                return {'predictions': y_hat}
            
            def prepare_images_for_logging(self, batch, outputs, batch_idx, epoch, mode='val'):
                '''
                This method is called by the WandBImageLogger callback.
                
                Args:
                    batch (Any): The input batch from the dataloader.
                    outputs (Optional[Dict[str, Any]]): The output from the
                                                    training_step or validation_step.
                    batch_idx (int): The index of the current batch.
                    epoch (int): The current epoch number.
                    mode (str): Either 'train' or 'val'.
                '''
                if self.logger:
                    # Example: Log a plot of predictions vs. ground truth
                    # Assumes batch is a tuple (inputs, labels) and outputs is a dict
                    inputs, labels = batch
                    preds = outputs['predictions']
                    
                    # Convert tensors to numpy for plotting
                    inputs_np = inputs.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    preds_np = preds.cpu().numpy()
                    
                    # Create a wandb.Image
                    # (This is just an example, you can create any W&B object)
                    # For this example, let's assume we are logging a matplotlib plot
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.scatter(labels_np[:, 0], preds_np[:, 0], label="Dimension 1")
                    ax.set_title(f"Epoch {epoch} - Batch {batch_idx}")
                    ax.set_xlabel("Ground Truth")
                    ax.set_ylabel("Predictions")
                    ax.legend()
                    
                    # Log the plot to W&B
                    log_key = f"{mode.capitalize()} Predictions/Epoch_{epoch}"
                    self.logger.experiment.log({
                        log_key: wandb.Image(fig)
                    })
                    plt.close(fig) # Important to close the figure to free memory
        
        # In your training script:
        # trainer = lp.Trainer(
        #     logger=WandbLogger(project="my_project"),
        #     callbacks=[WandBImageLogger(log_on_val_epoch=True)]
        # )
        # trainer.fit(model, datamodule)
    
    """
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
