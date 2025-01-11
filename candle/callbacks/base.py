import logging
from typing import Optional, List
from candle.utils.module import Module


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