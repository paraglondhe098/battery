import logging
from typing import Optional, List, Union
from candle.utils.module import Module
import copy


class Callback(Module):

    def __init__(self, priority: float = None, logger: Optional[logging.Logger] = None):
        super().__init__(name="Callback")
        self._trainer = None
        self.priority = priority
        self.__use_trainer_logger = True if logger is None else False

    def set_trainer(self, trainer: 'TrainerModule'):
        self._trainer = trainer
        if self.__use_trainer_logger:
            self.logger = trainer.logger

    @property
    def trainer(self) -> 'TrainerModule':
        return self._trainer

    @property
    def tracker(self) -> 'TrainerModule':
        return self.trainer.tracker

    @property
    def model(self) -> 'torch.models.Module':
        return self.trainer.model

    @property
    def device(self):
        return self.trainer.device

    def before_training_starts(self) -> Optional[str]:
        """Initialize parameters before training begins"""
        pass

    def after_training_ends(self) -> Optional[str]:
        """Carry out any operation after training just ended"""
        pass

    def on_batch_begin(self) -> Optional[str]:
        """A backwards compatibility alias for `on_train_batch_begin`."""
        pass

    def on_batch_end(self) -> Optional[str]:
        """A backwards compatibility alias for `on_train_batch_end`."""
        pass

    def on_epoch_begin(self) -> Optional[str]:
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.
        """
        pass

    def on_epoch_end(self) -> Optional[str]:
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.
        """
        pass

    def on_train_batch_begin(self) -> Optional[str]:
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        """
        # For backwards compatibility.
        self.on_batch_begin()
        pass

    def on_train_batch_end(self) -> Optional[str]:
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        """
        # For backwards compatibility.
        self.on_batch_end()
        pass

    def on_test_batch_begin(self) -> Optional[str]:
        """Called at the beginning of a batch in `evaluate` methods.

        Subclasses should override for any actions to run.

        """
        pass

    def on_test_batch_end(self) -> Optional[str]:
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation datasets is provided.

        Subclasses should override for any actions to run.

        """
        pass

    def on_predict_batch_begin(self) -> Optional[str]:
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        """
        pass

    def on_predict_batch_end(self) -> Optional[str]:
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        """
        pass

    def on_train_begin(self) -> Optional[str]:
        """Called at the beginning of training.

        Subclasses should override for any actions to run.
        """
        pass

    def on_train_end(self) -> Optional[str]:
        """Called at the end of training.

        Subclasses should override for any actions to run.
        """
        pass

    def on_test_begin(self) -> Optional[str]:
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.
        """
        pass

    def on_test_end(self) -> Optional[str]:
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.
        """
        pass

    def on_predict_begin(self) -> Optional[str]:
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.
        """
        pass

    def on_predict_end(self) -> Optional[str]:
        """Called at the end of prediction.

        Subclasses should override for any actions to run.
        """
        pass

    def before_backward_pass(self) -> Optional[str]:
        """Called before loss.backward(can be used for regularization or dynamic loss function modification.).

        Subclasses should override for any actions to run.
        """
        pass


class CallbacksList(Module):
    def __init__(self, callbacks: Optional[List[Callback]], trainer: 'TrainerModule'):

        super().__init__()
        self.trainer = trainer
        self.callbacks = []
        if callbacks:
            for cb in callbacks:
                self.append(cb)

    def append(self, callback: Callback):
        if callback not in self.callbacks:
            if isinstance(callback, Callback):
                callback.set_trainer(self.trainer)
                self.callbacks.append(callback)
            else:
                raise TypeError("callbacks should be inherited from Callback class (from torchtrainer.callbacks)")
        else:
            self.logger.info(f"Callback {callback} is already present")

    def remove(self, callback: Callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def run_all(self, pos: str) -> Optional[List[str]]:
        responses = []
        for cb in self.callbacks:
            try:
                run_at_pos = getattr(cb, pos, None)
                response = run_at_pos()
                if response:
                    responses.append(response)
            except Exception as e:
                self.logger.warning(f"Error in callback {cb} during calling of method '{pos}': {e}")
                raise e
        return responses

    def __len__(self):
        return len(self.callbacks)

    def __str__(self):
        return str(self.callbacks)


class WeightInitializer(Callback):
    def __init__(self):
        super().__init__()


class Regularizer(Callback):
    def __init__(self):
        super().__init__()


class LayerFreezer(Callback):
    def __init__(self):
        super().__init__()


class ImageSaver(Callback):
    def __init__(self):
        super().__init__()


class NotebookLogger(Callback):
    def __init__(self):
        super().__init__()


class TensorBoardLogger(Callback):
    def __init__(self):
        super().__init__()


class CSVLogger(Callback):
    def __init__(self):
        super().__init__()


class LRScheduler(Callback):
    def __init__(self):
        super().__init__()


class StateManager(Callback):
    def __init__(self):
        super().__init__()


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

    def before_training_starts(self) -> Optional[str]:
        """Initialize the metric monitor before training starts."""
        self.monitor = self.tracker.create_link(self.basis, split="self")
        self.reset_state()

    def reset_state(self):
        self.patience = self.initial_patience
        self.best_value = float('inf') if self.metric_minimize else float('-inf')
        self.best_epoch = 0

    def on_epoch_end(self) -> Optional[str]:
        """Check metric value after validation and determine if training should stop."""
        current_value = self.monitor.latest
        self.logger.debug(current_value)

        # Check if metric improved
        improved = (self.metric_minimize and current_value < self.best_value) or \
                   (not self.metric_minimize and current_value > self.best_value)

        if improved:
            self.best_value = current_value
            self.trainer.best_model_weights = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.trainer.current_epoch
            self.patience = self.initial_patience  # Reset patience
        else:
            # Check if we should reduce patience
            threshold_check = self.threshold is None or \
                              (self.metric_minimize and self.best_value < self.threshold) or \
                              (not self.metric_minimize and self.best_value > self.threshold)

            if threshold_check:
                self.patience -= 1
                # self.trainer.epoch_message += f" <es-p-{self.patience}>" <-- No use

        # Check stopping conditions
        is_last_epoch = (self.trainer.current_epoch == self.trainer.epochs)
        should_stop = self.patience == 0 or (is_last_epoch and self.trainer.best_model_weights)

        if should_stop:
            self.trainer.STOPPER = True
            if is_last_epoch:
                self.logger.info(f"Stopping at last epoch {self.trainer.current_epoch}")
            else:
                self.logger.info(f"Early-stopping at epoch {self.trainer.current_epoch}, basis : {self.basis}"
                                 f"{'↑' if self.metric_minimize else '↓'}")

            if self.restore_best_weights:
                # Restore best weights
                self.trainer.model.load_state_dict(self.trainer.best_model_weights)
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

                return_val = ("-" * 100)+"\n"
                return_val += "\n\t".join(summary)
                #
                return return_val

        return None

    def after_training_ends(self) -> Optional[str]:
        if self.best_weights_restored:
            res = {"epoch": self.best_epoch}
            for metric_name in self.tracker.metrics:
                res[metric_name] = self.tracker.metrics[metric_name].records[self.best_epoch]
            self.trainer.final_metrics = res

        return None


class GradientClipping(Callback):
    def __init__(self):
        super().__init__()


class IntraEpochReport(Callback):
    def __init__(self):
        super().__init__()


class MemoryUsageLogger(Callback):
    def __init__(self):
        super().__init__()


class WeightWatcher(Callback):
    def __init__(self):
        super().__init__()


class ReduceLROnPlateau(Callback):
    def __init__(self):
        super().__init__()


class FeatureMapVisualizer(Callback):
    def __init__(self):
        super().__init__()


class RemoteMonitor(Callback):
    def __init__(self):
        super().__init__()


class NoiseInjector(Callback):
    def __init__(self):
        super().__init__()


class LRTracker(Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self):
        self.tracker['lr'] = []

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def on_epoch_end(self):
        self.tracker['lr'].append(self.get_lr(self.trainer.optimizer))
