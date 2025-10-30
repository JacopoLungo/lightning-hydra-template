from typing import Any

import lightning.pytorch as lp
import numpy as np
from lightning.pytorch.loggers import WandbLogger


class WandBImageLogger(lp.Callback):
    """
    A Lightning Callback for logging images to Weights & Biases (W&B).

    This callback logs images from the training, validation, testing, and/or
    prediction loops to a W&B run. It is designed to be flexible, allowing for
    separate logging configurations for the training phase versus the
    validation, testing, and prediction phases.

    Logging can be triggered at specified epoch intervals (logging a certain
    number of batches) or at step/batch intervals (logging every N steps for
    training, or every N batches for other stages).

    The actual image preparation and logging logic is delegated to a method
    within the LightningModule. This keeps the data processing and visualization
    logic with the model, while this callback handles the triggering and scheduling.

    Args:
        train_log_every_n_epochs (int, optional):
            Frequency of epochs to log images during training. If 1, logs every
            epoch. If None, epoch-based logging is disabled for training.
            Mutually exclusive with `train_log_every_n_steps`. Defaults to None.
        train_log_every_n_steps (int, optional):
            Frequency of global steps to log images during training. If set,
            this will trigger logging at every `n` global steps.
            Mutually exclusive with `train_log_every_n_epochs`. Defaults to None.
        train_n_batches_to_visualize (int, optional):
            Number of batches to visualize per epoch during training when using
            epoch-based logging. If -1, all batches are logged. Defaults to 1.
        train_random_batches (bool, optional):
            If True, randomly selects batches for epoch-based logging during
            training. If False, batches are chosen at evenly spaced intervals.
            Defaults to True.
        log_on_train (bool, optional):
            Master switch to enable or disable logging during the training phase.
            Defaults to False.
        val_test_pred_log_every_n_epochs (int, optional):
            Frequency of epochs to log images during validation, testing, and
            prediction. If 1, logs every epoch. If None, epoch-based logging
            is disabled for these stages. Mutually exclusive with
            `val_test_pred_log_every_n_steps`. Defaults to 1.
        val_test_pred_log_every_n_steps (int, optional):
            Frequency of batches to log images during validation, testing, and
            prediction. If set, logs every `n`-th batch within the respective stage.
            Mutually exclusive with `val_test_pred_log_every_n_epochs`.
            Defaults to None.
        val_test_pred_n_batches_to_visualize (int, optional):
            Number of batches to visualize per epoch during validation, testing,
            and prediction when using epoch-based logging. If -1, all batches
            are logged. Defaults to 1.
        val_test_pred_random_batches (bool, optional):
            If True, randomly selects batches for epoch-based logging during
            validation, testing, and prediction. If False, batches are chosen
            at evenly spaced intervals. Defaults to False.
        log_on_val_test_pred (bool, optional):
            Master switch to enable or disable logging during validation, testing,
            and prediction phases. Defaults to True.
        image_log_fn_name (str, optional):
            The name of the method in the LightningModule that prepares and logs
            the images. This method should accept `batch`, `outputs`, `batch_idx`,
            `epoch`, and `mode` as arguments. Defaults to "log_visualizations".

    Raises:
        ValueError: If both `..._log_every_n_epochs` and `..._log_every_n_steps` are
                    specified for the same stage (train or val_test_pred).
        ValueError: If any frequency or batch count arguments are invalid.

    Example Usage in a LightningModule:

    .. code-block:: python

        class MyLitModule(lp.LightningModule):
            # ... (your model definition) ...

            def log_visualizations(self,
                                   batch: Any,
                                   step_outputs: Dict[str, Any],
                                   batch_idx: int,
                                   current_epoch: int,
                                   mode: str):

                # Example: Log the first image of the batch
                # Assumes batch[0] are images and batch[1] are labels
                img = batch[0][0].cpu()
                label = batch[1][0].cpu()
                pred = step_outputs["preds"][0].cpu() # Assumes "preds" are in outputs

                # Create a wandb.Image for logging
                # You might need to unnormalize or reshape the image
                wandb_image = wandb.Image(img, caption=f"Epoch: {current_epoch}, Label: {label}, Pred: {pred}")

                # Log with a dynamic key based on the mode (train/val/test/predict)
                self.logger.experiment.log({
                    f"visualizations/{mode}": wandb_image,
                    "trainer/global_step": self.global_step
                })

        # In your training script:
        # Log 5 random training batches every epoch, and the first batch of val/test/predict
        image_logger = WandBImageLogger(
            log_on_train=True,
            train_log_every_n_epochs=1,
            train_n_batches_to_visualize=5,
            train_random_batches=True,
            log_on_val_test_pred=True,
            val_test_pred_log_every_n_epochs=1,
            val_test_pred_n_batches_to_visualize=1,
            val_test_pred_random_batches=False
        )

        trainer = lp.Trainer(
            logger=WandbLogger(project="my_project"),
            callbacks=[image_logger]
        )
        # trainer.fit(model, datamodule)
        # trainer.test(model, datamodule)
        # trainer.predict(model, datamodule)
    """

    def __init__(
        self,
        log_on_train: bool = False,
        train_log_every_n_epochs: int | None = None,
        train_log_every_n_steps: int | None = None,
        train_n_batches_to_visualize: int = 1,
        train_random_batches: bool = True,
        log_on_val_test_pred: bool = True,
        val_test_pred_log_every_n_epochs: int | None = 1,
        val_test_pred_log_every_n_steps: int | None = None,
        val_test_pred_n_batches_to_visualize: int = 1,
        val_test_pred_random_batches: bool = False,
        image_log_fn_name: str = "log_visualizations",
    ):
        super().__init__()

        # --- Validation for Training Config ---
        if log_on_train:
            if train_log_every_n_epochs is not None and train_log_every_n_steps is not None:
                raise ValueError(
                    "For training, you can only specify one of `train_log_every_n_epochs` or `train_log_every_n_steps`."
                )
            if train_log_every_n_epochs is not None and train_log_every_n_epochs <= 0:
                raise ValueError("train_log_every_n_epochs must be positive.")
            if train_log_every_n_steps is not None and train_log_every_n_steps <= 0:
                raise ValueError("train_log_every_n_steps must be positive.")
            if train_n_batches_to_visualize < -1:
                raise ValueError("train_n_batches_to_visualize must be >= -1.")

        # --- Validation for Val/Test/Predict Config ---
        if log_on_val_test_pred:
            if val_test_pred_log_every_n_epochs is not None and val_test_pred_log_every_n_steps is not None:
                raise ValueError(
                    "For val/test/predict, you can only specify one of `val_test_pred_log_every_n_epochs` or `val_test_pred_log_every_n_steps`."
                )
            if val_test_pred_log_every_n_epochs is not None and val_test_pred_log_every_n_epochs <= 0:
                raise ValueError("val_test_pred_log_every_n_epochs must be positive.")
            if val_test_pred_log_every_n_steps is not None and val_test_pred_log_every_n_steps <= 0:
                raise ValueError("val_test_pred_log_every_n_steps must be positive.")
            if val_test_pred_n_batches_to_visualize < -1:
                raise ValueError("val_test_pred_n_batches_to_visualize must be >= -1.")

        # --- Store configurations ---
        self.log_on_train = log_on_train
        self.train_log_every_n_epochs = train_log_every_n_epochs
        self.train_log_every_n_steps = train_log_every_n_steps
        self.train_n_batches_to_visualize = train_n_batches_to_visualize
        self.train_random_batches = train_random_batches

        self.log_on_val_test_pred = log_on_val_test_pred
        self.val_test_pred_log_every_n_epochs = val_test_pred_log_every_n_epochs
        self.val_test_pred_log_every_n_steps = val_test_pred_log_every_n_steps
        self.val_test_pred_n_batches_to_visualize = val_test_pred_n_batches_to_visualize
        self.val_test_pred_random_batches = val_test_pred_random_batches

        self.image_log_fn_name = image_log_fn_name

        # --- State variables ---
        self._train_batch_indices_to_log: np.ndarray | None = None
        self._val_batch_indices_to_log: np.ndarray | None = None
        self._test_batch_indices_to_log: np.ndarray | None = None
        self._predict_batch_indices_to_log: np.ndarray | None = None

        self._do_log_on_current_train_epoch: bool = False
        self._do_log_on_current_val_epoch: bool = False
        self._do_log_on_current_test_epoch: bool = False
        self._do_log_on_current_predict_epoch: bool = False

    def _setup_epoch_visualization(self, trainer: lp.Trainer, mode: str):
        """Pre-calculates which batch indices to log for epoch-based logging."""
        # Determine config based on mode
        if mode == "train":
            log_on_mode = self.log_on_train
            log_every_n_epochs = self.train_log_every_n_epochs
            log_every_n_steps = self.train_log_every_n_steps
            n_batches_to_visualize = self.train_n_batches_to_visualize
            random_batches = self.train_random_batches
        else:  # 'val', 'test', 'predict'
            log_on_mode = self.log_on_val_test_pred
            log_every_n_epochs = self.val_test_pred_log_every_n_epochs
            log_every_n_steps = self.val_test_pred_log_every_n_steps
            n_batches_to_visualize = self.val_test_pred_n_batches_to_visualize
            random_batches = self.val_test_pred_random_batches

        # This setup is only for epoch-based logging; exit if disabled or using step-based logging
        if not log_on_mode or log_every_n_epochs is None or log_every_n_steps is not None:
            setattr(self, f"_do_log_on_current_{mode}_epoch", False)
            return

        # Check if logging should happen on this epoch
        should_log_this_epoch = trainer.current_epoch % log_every_n_epochs == 0
        setattr(self, f"_do_log_on_current_{mode}_epoch", should_log_this_epoch)
        if not should_log_this_epoch:
            return

        # Determine the number of batches for the current mode
        num_batches = 0
        if mode == "train":
            if isinstance(trainer.num_training_batches, float) and trainer.num_training_batches == float("inf"):
                if trainer.global_rank == 0:
                    print(
                        "WandBImageLogger: Iterable training dataset. Will log first `n_batches_to_visualize` batches if not random."
                    )
                if n_batches_to_visualize > 0:
                    self._train_batch_indices_to_log = np.arange(n_batches_to_visualize)
                else:  # -1 is not supported for iterable datasets
                    self._train_batch_indices_to_log = np.array([], dtype=int)
                return
            num_batches = trainer.num_training_batches
        elif mode == "val":
            if trainer.num_val_batches:
                num_batches = sum(trainer.num_val_batches)
        elif mode == "test":
            if trainer.num_test_batches:
                num_batches = sum(trainer.num_test_batches)
        elif mode == "predict":
            if trainer.num_predict_batches:
                num_batches = sum(trainer.num_predict_batches)

        if num_batches == 0:
            if trainer.global_rank == 0:
                print(
                    f"WandBImageLogger: No {mode} batches detected. Skipping image logging for epoch {trainer.current_epoch}."
                )
            setattr(self, f"_do_log_on_current_{mode}_epoch", False)
            return

        # Calculate and store batch indices to log
        if n_batches_to_visualize == -1:
            indices = np.arange(num_batches)
        else:
            num_to_pick = min(n_batches_to_visualize, num_batches)
            if random_batches:
                indices = np.random.choice(num_batches, size=num_to_pick, replace=False)
            else:
                indices = np.linspace(0, num_batches - 1, num=num_to_pick, dtype=int)

        setattr(self, f"_{mode}_batch_indices_to_log", np.sort(indices))

    # --- Hooks for setting up epoch-based logging ---
    def on_train_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        self._setup_epoch_visualization(trainer, mode="train")

    def on_validation_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        self._setup_epoch_visualization(trainer, mode="val")

    def on_test_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        self._setup_epoch_visualization(trainer, mode="test")

    def on_predict_epoch_start(self, trainer: lp.Trainer, pl_module: lp.LightningModule):
        self._setup_epoch_visualization(trainer, mode="predict")

    def _log_image_batch(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        mode: str,
    ):
        """Central logging logic for all stages."""
        if not isinstance(trainer.logger, WandbLogger):
            return

        if not hasattr(pl_module, self.image_log_fn_name):
            if trainer.global_rank == 0:
                print(f"WandBImageLogger: LightningModule missing '{self.image_log_fn_name}'. Skipping image logging.")
            return

        # Determine config based on mode
        if mode == "train":
            log_on_mode = self.log_on_train
            log_every_n_steps = self.train_log_every_n_steps
            do_log_on_current_epoch = self._do_log_on_current_train_epoch
            batch_indices_to_log = self._train_batch_indices_to_log
        else:  # 'val', 'test', 'predict'
            log_on_mode = self.log_on_val_test_pred
            log_every_n_steps = self.val_test_pred_log_every_n_steps
            do_log_on_current_epoch = getattr(self, f"_do_log_on_current_{mode}_epoch")
            batch_indices_to_log = getattr(self, f"_{mode}_batch_indices_to_log")

        if not log_on_mode:
            return

        should_log = False
        # --- Step/Batch-based logging ---
        if log_every_n_steps is not None:
            # For train, use global_step. For others, use batch_idx within the stage.
            step_to_check = trainer.global_step if mode == "train" else batch_idx
            if step_to_check > 0 and step_to_check % log_every_n_steps == 0:
                should_log = True
        # --- Epoch-based logging ---
        else:
            if do_log_on_current_epoch and batch_indices_to_log is not None:
                if batch_idx in batch_indices_to_log:
                    should_log = True

        # If we should log, execute the model's logging function
        if should_log:
            image_log_fn = getattr(pl_module, self.image_log_fn_name)
            image_log_fn(batch, outputs, batch_idx, trainer.current_epoch, mode=mode)

    # --- Hooks for triggering batch-level logging ---
    def on_train_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ):
        self._log_image_batch(trainer, pl_module, outputs, batch, batch_idx, mode="train")

    def on_validation_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self._log_image_batch(trainer, pl_module, outputs, batch, batch_idx, mode="val")

    def on_test_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self._log_image_batch(trainer, pl_module, outputs, batch, batch_idx, mode="test")

    def on_predict_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self._log_image_batch(trainer, pl_module, outputs, batch, batch_idx, mode="predict")
