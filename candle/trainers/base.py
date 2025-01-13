import logging
from abc import abstractmethod
from candle.utils.module import Module
from candle.callbacks import CallbacksList, Callback
import torch
from typing import Optional, List, Any
from tqdm import tqdm


class ProgressBar:
    def __init__(self, positions, **kwargs):
        self.positions = positions
        self.kwargs = kwargs

    def __call__(self, position, iterable, desc):
        self.kwargs['desc'] = desc
        self.kwargs['iterable'] = iterable
        if position in self.positions:
            return tqdm(**self.kwargs)
        else:
            return iterable


class TrainerModule(Module):
    def __init__(self, name: str,
                 model: Optional[torch.nn.Module] = None,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(name, device, logger)
        self.model = self.to_device(model) if model else None
        self.callbacks = None
        self.epoch_headline = ""
        self.epochs = None
        self.progress_bar = ProgressBar([])

    def set_callbacks(self, callbacks: List[Callback]):
        return CallbacksList(callbacks, trainer=self)

    @abstractmethod
    def init_tracker(self, *args, **kwargs) -> 'Tracker':
        pass

    @abstractmethod
    def fit(self, training_data: Any, validation_data: Any, epochs: int, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class AdversarialTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="Adversarial Trainer")

    def init_tracker(self, *args, **kwargs) -> 'Tracker':
        pass

    def fit(self, training_data: Any, validation_data: Any, epochs: int, **kwargs):
        pass

    def predict(self, X):
        pass


class LLMTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="LLM Trainer")

    def init_tracker(self, *args, **kwargs) -> 'Tracker':
        pass

    def fit(self, training_data: Any, validation_data: Any, epochs: int, **kwargs):
        pass

    def predict(self, X):
        pass


class SemiSupervisedTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="Semi Supervised Trainer")

    def init_tracker(self, *args, **kwargs) -> 'Tracker':
        pass

    def fit(self, training_data: Any, validation_data: Any, epochs: int, **kwargs):
        pass

    def predict(self, X):
        pass
