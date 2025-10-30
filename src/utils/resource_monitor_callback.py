import psutil
import torch
import lightning.pytorch as lp
from typing import Any

class ResourceMonitorCallback(lp.Callback):
    """
    PyTorch Lightning callback that monitors and logs basic system resources:
    - Total RAM usage of the process tree
    - GPU reserved memory (if CUDA is available)
    
    These metrics are logged to external loggers like Weights & Biases or TensorBoard.
    """
    
    def __init__(self, log_freq: int = 50):
        """
        Initialize the SystemResourceMonitorCallback.
        
        Args:
            log_freq: Frequency of logging in number of batches
        """
        super().__init__()
        self.log_freq = log_freq
    
    def state_dict(self) -> dict[str, Any]:
        """Return callback state dict for checkpointing."""
        return {"log_freq": self.log_freq}
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.log_freq = state_dict.get("log_freq", 50)
    
    def _get_resource_metrics(self) -> dict[str, float]:
        """Collect RAM and GPU usage metrics."""
        metrics = {}
        
        # Calculate Total system RAM usage for the current process tree
        current_process = psutil.Process()
        total_ram_bytes = current_process.memory_info().rss
        try:
            for child in current_process.children(recursive=True):
                try:
                    total_ram_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            metrics["ram_usage_gb"] = total_ram_bytes / (1024**3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            metrics["ram_usage_gb"] = 0.0
        
        # GPU memory metrics
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            try:
                reserved_bytes = torch.cuda.memory_reserved(0)
                metrics["gpu_reserved_gb"] = reserved_bytes / (1024**3)
            except Exception:
                metrics["gpu_reserved_gb"] = 0.0
        else:
            metrics["gpu_reserved_gb"] = 0.0
            
        return metrics
    
    def _log_metrics(self, trainer: lp.Trainer, prefix: str = ""):
        """Log the collected metrics to available loggers."""
        metrics = self._get_resource_metrics()
        
        # Add prefix if provided
        prefixed_metrics = {}
        for key, value in metrics.items():
            if prefix:
                prefixed_metrics[f"{prefix}/{key}"] = value
            else:
                prefixed_metrics[key] = value
        
        # Log to all available loggers
        if trainer.loggers:
            for logger in trainer.loggers:
                if hasattr(logger, 'log_metrics'):
                    # Use Lightning's logging API
                    logger.log_metrics(prefixed_metrics, step=trainer.global_step)
                elif hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                    # For WandB specifically
                    logger.experiment.log(prefixed_metrics, step=trainer.global_step)
                elif hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_scalar'):
                    # For TensorBoard specifically
                    for name, value in prefixed_metrics.items():
                        logger.experiment.add_scalar(name, value, trainer.global_step)
    
    def on_train_batch_end(self, trainer: lp.Trainer, pl_module: lp.LightningModule, 
                        outputs: Any, batch: Any, batch_idx: int) -> None:
        """Log system metrics at the end of training batches based on log_freq."""
        if trainer.global_step % self.log_freq == 0:
            self._log_metrics(trainer, prefix="resources")
    
    def on_validation_epoch_end(self, trainer: lp.Trainer, pl_module: lp.LightningModule) -> None:
        """Log system metrics at the end of validation epoch."""
        self._log_metrics(trainer, prefix="resources")
    
    def on_test_epoch_end(self, trainer: lp.Trainer, pl_module: lp.LightningModule) -> None:
        """Log system metrics at the end of test epoch."""
        self._log_metrics(trainer, prefix="resources")
    
    def on_fit_end(self, trainer: lp.Trainer, pl_module: lp.LightningModule) -> None:
        """Log final system metrics at the end of training."""
        self._log_metrics(trainer, prefix="resources/final")
