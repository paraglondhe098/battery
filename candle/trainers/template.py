import logging
from abc import abstractmethod
from candle.callbacks import Callback
from candle.trainers.base import TrainerModule
from candle.utils.tracking import Tracker
import torch
from typing import Optional, List
import copy
from torch.nn import Module


class ModelConfig:
    def __init__(self, model, **kwargs):
        self.model = model
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_requires_grad(self, requires_grad=True):
        for p in self.model.parameters():
            p.requires_grad = requires_grad


class TrainerTemplate(TrainerModule):
    def __init__(self,
                 model: Module,
                 callbacks: Optional[List[Callback]] = None,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):

        super().__init__(model=model, name="SimpleTrainer", device=(device or torch.device('cpu')), logger=logger)

        self.num_batches = None
        self.batch_size = None
        self._current_epoch = 0

        self.clear_cuda_cache = clear_cuda_cache
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.tracker = self.init_tracker()

        self.STOPPER = False
        self.external_events = set()
        self._best_state_dict = None
        self._final_metrics = {}

        self.std_pos = {'on_train_batch_begin', 'on_train_batch_end', 'on_epoch_begin', 'on_epoch_end',
                        'on_test_batch_begin', 'on_test_batch_end', 'on_predict_batch_begin', 'on_predict_batch_end',
                        'on_train_begin', 'on_train_end', 'on_test_begin', 'on_test_end', 'on_predict_begin',
                        'on_predict_end', 'before_training_starts', 'after_training_ends', 'before_backward_pass'}
        self.callbacks = self.set_callbacks(callbacks or [])

    @abstractmethod
    def init_tracker(self) -> Tracker:
        pass

    @abstractmethod
    def training_step(self, inputs, labels):
        pass

    @abstractmethod
    @torch.no_grad()
    def eval_step(self, inputs, labels):
        pass

    @abstractmethod
    @torch.no_grad()
    def prediction_step(self, data):
        return self.model(data)

    @abstractmethod
    def save_progress(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_progress(self, *args, **kwargs):
        pass

    def reset_progress(self):
        self._current_epoch = 0
        self.STOPPER = False
        self._final_metrics = {}
        self.tracker = self.init_tracker()

    def _run_callbacks(self, pos: str) -> List[Optional[str]]:
        return self.callbacks.run_all(pos)

    def train(self, train_loader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        self._run_callbacks(pos="on_train_begin")
        for inputs, labels in self.progress_bar(position='training',
                                                iterable=train_loader,
                                                desc=self.epoch_headline):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self._run_callbacks(pos="on_train_batch_begin")

            self.training_step(inputs, labels)

            self._run_callbacks(pos="on_train_batch_end")
        self._run_callbacks(pos="on_train_end")

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()
        self._run_callbacks(pos="on_test_begin")
        for inputs, labels in self.progress_bar(position='validation', iterable=val_loader, desc="Validation: "):
            self._run_callbacks(pos="on_test_batch_begin")
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.eval_step(inputs, labels)

            self._run_callbacks(pos="on_test_batch_end")
        self._run_callbacks(pos="on_test_end")

    @property
    def current_epoch(self):
        return self._current_epoch

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
            epochs: int = 1, epoch_start: int = 0):
        """
        Trains the model for the specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training datasets.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation datasets.
            epoch_start (int): from what epoch number we should start
            epochs (int): No. of epochs to run for

        Returns:
            None
        """
        self.reset_progress()
        self.epochs = epochs
        self.num_batches = len(train_loader)
        self.batch_size = train_loader.batch_size
        on_gpu = True if self.device.type == 'cuda' else False

        self._run_callbacks(pos="before_training_starts")
        for self._current_epoch in range(epoch_start, epoch_start + self.epochs):
            self._run_callbacks(pos="on_epoch_begin")

            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.train(train_loader)
            self.validate(val_loader)

            self._run_callbacks(pos="on_epoch_end")
            self.tracker.snap_and_reset_all()

            if self.STOPPER:
                break

        self._run_callbacks(pos="after_training_ends")
        return self.tracker.get_history()

    @property
    def final_metrics_(self):
        return self._final_metrics or self.tracker.get_final_values(self.current_epoch)

    @property
    def best_state_dict_(self):
        return self._best_state_dict or copy.deepcopy(self.model.state_dict())

    def predict(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Predicts outputs for the given DataLoader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader providing input datasets for prediction.

        Returns:
            torch.Tensor: Concatenated model predictions for all input batches.
        """
        self.model.eval()
        self._run_callbacks(pos="on_predict_begin")

        all_predictions = []
        for batch_idx, data in self.progress_bar(position="prediction",
                                                 iterable=enumerate(data_loader),
                                                 desc="Processing"):
            self._run_callbacks(pos="on_predict_batch_begin")
            data = data.to(self.device)
            predictions = self.prediction_step(data)
            all_predictions.append(predictions)
            self._run_callbacks(pos="on_predict_batch_end")

        all_predictions = torch.cat(all_predictions, dim=0)
        self._run_callbacks(pos="on_predict_end")
        return all_predictions
