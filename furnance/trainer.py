import time
# from torch.cuda.amp import GradScaler, autocast -> deprecated
from torch.amp import GradScaler, autocast
from abc import abstractmethod
from utils.tracking import Tracker
from callbacks import Callback, CallbacksList
from metrics import Metric
import torch
import copy
from torchsummary import summary
from typing import Tuple, Optional, List, Callable
from tqdm import tqdm
from utils.module import Module


class TrainerModule(Module):
    def __init__(self, name, device=None):
        super().__init__(name, device=device)

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class Trainer(TrainerModule):
    def __init__(self, model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 input_shape: Tuple[int, ...],
                 optimizer: torch.optim.Optimizer,
                 metrics: Optional[List[Metric]] = None,
                 callbacks: Optional[List[Callback]] = None,
                 display_time_elapsed: bool = False,
                 roff: int = 5,
                 report_in_one_line: bool = True,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None):

        super().__init__(name="Simple Trainer", device=(device or torch.device('cpu')))
        self.epochs = None
        self.messages_joiner = "  ||  " if report_in_one_line else "\n"
        self.epoch_message = None

        self.num_batches = None
        self.batch_size = None
        self.__current_epoch = 0
        self.__current_batch = 0

        self.metrics = [metric.name for metric in metrics]
        self.metric_fns = metrics

        self.model = self.to_device(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.clear_cuda_cache = clear_cuda_cache
        self.scaler = GradScaler() if (self.device.type == 'cuda' and use_amp) else None
        self.tracker = self.init_tracker()

        self.roff = roff
        self.input_shape = input_shape
        self.display_time_elapsed = display_time_elapsed

        self.STOPPER = False
        self.external_events = set()
        self.best_model_weights = None

        self.std_pos = {'on_train_batch_begin', 'on_train_batch_end', 'on_epoch_begin', 'on_epoch_end',
                        'on_test_batch_begin', 'on_test_batch_end', 'on_predict_batch_begin', 'on_predict_batch_end',
                        'on_train_begin', 'on_train_end', 'on_test_begin', 'on_test_end', 'on_predict_begin',
                        'on_predict_end', 'before_training_starts', 'after_training_ends', 'before_backward_pass'}

        self.callbacks = CallbacksList(callbacks=callbacks, trainer=self)

    def init_tracker(self):
        temp = self.metrics + ["loss"]
        metrics = []
        for metric in temp:
            metrics.append(metric)
            metrics.append(f"val_{metric}")
        return Tracker(metrics)

    def model_summary(self):
        summary(self.model, self.input_shape)

    def add_callback(self, callback: Callback) -> None:
        """
        Adds a callback to the Trainer.

        Note:
            If you're adding a custom callback function, make sure it's inherited
            from the `Callback` abstract base class and overwrites the `run` method,
            otherwise the callback will not run!

        Args:
            callback (Callback): Callback object to add. Must be an instance of
                                 a class inherited from the `Callback` base class.

        """
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callback) -> None:
        """
        Removes a callback from the Trainer.

        Args:
            callback: Callback object to remove.
        """
        self.callbacks.remove(callback)

    def __run_callbacks(self, pos: str) -> List[Optional[str]]:
        return self.callbacks.run_all(pos)

    def __train_fn(self, train_loader: torch.utils.data.DataLoader) -> None:

        # Set to training mode
        self.model.train()
        self.__run_callbacks(pos="on_train_begin")
        for self.__current_batch, (inputs, labels) in tqdm(enumerate(train_loader), self.epoch_message):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.__run_callbacks(pos="on_train_batch_begin")

            # One Batch Training
            self.optimizer.zero_grad()
            if self.scaler:
                # Mixed precision training
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.__run_callbacks(pos="before_backward_pass")
                loss.backward()
                self.optimizer.step()

            self.tracker.update({"loss": loss.item()})
            with torch.no_grad():
                self.tracker.update({metric: self.metric_fns[metric](labels, outputs) for metric in self.metrics})
            self.__run_callbacks(pos="on_train_batch_end")
        self.__run_callbacks(pos="on_train_end")

    @torch.no_grad()
    def __validation_fn(self, val_loader: torch.utils.data.DataLoader) -> None:
        # Set to the evaluation mode
        self.model.eval()
        self.__run_callbacks(pos="on_test_begin")
        for inputs, labels in val_loader:
            self.__run_callbacks(pos="on_test_batch_begin")
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.scaler:
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    val_loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                val_loss = self.criterion(outputs, labels)

            self.tracker.update({"val_loss": val_loss.item()})
            self.tracker.update(
                {"val_" + metric: self.metric_fns[metric](labels, outputs) for metric in self.metrics})
            self.__run_callbacks(pos="on_test_batch_end")
        self.__run_callbacks(pos="on_test_end")

    @property
    def current_batch(self):
        return self.__current_batch

    @property
    def current_epoch(self):
        return self.__current_epoch

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
        self.epochs = epochs
        self.num_batches = len(train_loader)
        self.batch_size = train_loader.batch_size
        on_gpu = True if self.device.type == 'cuda' else False
        tracker = self.tracker
        # The main Training loop

        start_time = time.time()
        self.__run_callbacks(pos="before_training_starts")
        for self.__current_epoch in range(epoch_start, epoch_start + self.epochs):
            self.__run_callbacks(pos="on_epoch_begin")

            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.epoch_message = f"EPOCH {self.current_epoch}: "

            # Train model
            self.__train_fn(train_loader)
            self.__validation_fn(val_loader)
            epoch_statistics = tracker.message("--> Metrics: ")
            tracker.snap_and_reset_all()

            # Run callbacks
            responses = self.__run_callbacks(pos="on_epoch_end")
            print(epoch_statistics)

            if self.display_time_elapsed:
                end_time = time.time()
                print(f"Time elapsed: {end_time - start_time} s")

            for response in responses:
                print(response)

            if self.STOPPER:
                break
        self.__run_callbacks(pos="after_training_ends")
        if self.best_model_weights is None:
            self.best_model_weights = copy.deepcopy(self.model.state_dict())
        return tracker.get_history()

    @torch.no_grad()
    def predict(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Predicts outputs for the given DataLoader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader providing input datasets for prediction.

        Returns:
            torch.Tensor: Concatenated model predictions for all input batches.
        """
        self.model.eval()
        self.__run_callbacks(pos="on_predict_begin")

        all_predictions = []
        for batch_idx, data in enumerate(data_loader):
            self.__run_callbacks(pos="on_predict_batch_begin")
            data = data.to(self.device)
            predictions = self.model(data)
            all_predictions.append(predictions)
            self.__run_callbacks(pos="on_predict_batch_end")

        all_predictions = torch.cat(all_predictions, dim=0)
        self.__run_callbacks(pos="on_predict_end")
        return all_predictions

    class __CallbackTemplate(Callback):
        def __init__(self):
            super().__init__()

    def add_event(self, pos: str):
        """
        Write a custom callback event without explicitly creating a new callback class.
        """

        def decorator(event: Callable) -> Optional[Callable]:
            # Check if the event is already registered
            if event.__name__ in self.external_events:
                return None  # Do nothing if event already exists

            # Create a new callback template
            ct = self.__CallbackTemplate()

            # Register the event if the position is valid
            if pos in self.std_pos:
                setattr(ct, pos, event)
            else:
                raise AttributeError(f"Invalid method '{pos}'. Must be one of {self.std_pos}.")

            # Add the callback template to the callback list
            self.external_events.add(event.__name__)
            self.add_callback(ct)

            return event

        return decorator


class AdversarialTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="Adversarial Trainer")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


class LLMTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="LLM Trainer")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


class SemiSupervisedTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="Semi Supervised Trainer")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass
