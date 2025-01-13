from candle.callbacks import Callback
from typing import Optional
import copy


class EarlyStopping(Callback):
    def __init__(self,
                 basis: str,
                 metric_minimize: bool = True,
                 patience: int = 5,
                 threshold: Optional[float] = None,
                 restore_best_weights: bool = True):
        """
        Early stopping callback that monitors a metric and stops training when it stops improving.

        Args:
            basis: Metric to monitor for improvement
            metric_minimize: If True, training stops when metric stops decreasing
            patience: Number of epochs to wait for improvement before stopping
            threshold: Optional threshold value for the metric
            restore_best_weights: Whether to restore model to best weights when stopped
        """
        super().__init__()
        self.best_epoch = 0
        self.basis = basis
        self.monitor = None
        self.metric_minimize = metric_minimize
        self.patience = patience
        self.initial_patience = patience
        self.threshold = threshold
        self.best_value = float('inf') if metric_minimize else float('-inf')
        self.restore_best_weights = restore_best_weights
        self.best_weights_restored = False
        self.__dashes = "-" * 100

    def before_training_starts(self):
        """Initialize the metric monitor before training starts."""
        self.monitor = self.tracker.create_link(self.basis, split="self")
        self.reset_state()

    def reset_state(self):
        self.patience = self.initial_patience
        self.best_value = float('inf') if self.metric_minimize else float('-inf')
        self.best_epoch = 0

    def on_epoch_end(self):
        """Check metric value after validation and determine if training should stop."""
        current_value = self.monitor.latest
        self.logger.debug(current_value)

        # Check if metric improved
        improved = (self.metric_minimize and current_value < self.best_value) or \
                   (not self.metric_minimize and current_value > self.best_value)

        if improved:
            self.best_value = current_value
            self.trainer.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.trainer.current_epoch
            self.patience = self.initial_patience  # Reset patience
        else:
            # Check if we should reduce patience
            threshold_check = self.threshold is None or \
                              (self.metric_minimize and self.best_value < self.threshold) or \
                              (not self.metric_minimize and self.best_value > self.threshold)

            if threshold_check:
                self.patience -= 1
                # self.trainer.epoch_headline += f" <es-p-{self.patience}>" <-- No use

        # Check stopping conditions
        is_last_epoch = (self.trainer.current_epoch == self.trainer.epochs)
        should_stop = self.patience == 0 or (is_last_epoch and self.trainer.best_state_dict)

        if should_stop:
            self.trainer.STOPPER = True
            if is_last_epoch:
                self.logger.info(f"Stopping at last epoch {self.trainer.current_epoch}")
            else:
                self.logger.info(f"Early-stopping at epoch {self.trainer.current_epoch}, basis : {self.basis}"
                                 f"{'↑' if self.metric_minimize else '↓'}")

            if self.restore_best_weights:
                # Restore best weights
                self.trainer.model.load_state_dict(self.trainer.best_state_dict)
                self.best_weights_restored = True

                # Build summary message
                summary = [
                    "Restoring best weights...",
                    f"Best epoch: {self.best_epoch}",
                    f"Training loss: {self.tracker.metrics['loss'].records[self.best_epoch]:.4f}",
                    f"Validation loss: {self.tracker.metrics['val_loss'].records[self.best_epoch]:.4f}"
                ]

                # Add metric summaries if metrics exist
                if self.trainer.metrics:
                    for metric in self.trainer.metrics:
                        summary.extend([
                            f"Training {metric}: {self.tracker.metrics[metric].records[self.best_epoch]:.4f}",
                            f"Validation {metric}: {self.tracker.metrics[f'val_{metric}'].records[self.best_epoch]:.4f}"
                        ])

                summary = self.__dashes + "\n" + "\n\t".join(summary) + "\n" + self.__dashes
                self.logger.info(summary)

    def after_training_ends(self):
        if self.best_weights_restored:
            res = {"epoch": self.best_epoch}
            for metric_name in self.tracker.metrics:
                res[metric_name] = self.tracker.metrics[metric_name].records[self.best_epoch]
            self.trainer.final_metrics = res
