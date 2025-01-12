import logging
import time
# from torch.cuda.amp import GradScaler, autocast -> deprecated
from torch.amp import GradScaler, autocast
from abc import abstractmethod
from candle.utils.tracking import Tracker
from candle.utils.module import Module
from candle.callbacks import Callback, CallbacksList
from candle.metrics import Metric
import torch
import copy
from typing import Optional, List, Callable, Dict
from tqdm import tqdm
from datetime import datetime
import os


class TrainerModule(Module):
    def __init__(self, name: str,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(name, device, logger)

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class TrainerBluePrint(TrainerModule):
    def __init__(self,
                 callbacks: Optional[List[Callback]] = None,
                 display_time_elapsed: bool = False,
                 roff: int = 5,
                 report_in_one_line: bool = True,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 name: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None,
                 display_blueprint_use_warning: bool = True):

        super().__init__(name=name, device=(device or torch.device('cpu')), logger=logger)
        self.best_model_weights = None
        self.final_metrics = None
        self.epochs = None
        self.messages_joiner = "  ||  " if report_in_one_line else "\n"
        self.epoch_message = ""

        self.num_batches = None
        self.batch_size = None
        self.__current_epoch = 0
        self.__current_batch = 0

        self.clear_cuda_cache = clear_cuda_cache
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        self.tracker = self.init_tracker()

        self.roff = roff
        self.display_time_elapsed = display_time_elapsed

        self.STOPPER = False
        self.external_events = set()

        self.std_pos = {'on_train_batch_begin', 'on_train_batch_end', 'on_epoch_begin', 'on_epoch_end',
                        'on_test_batch_begin', 'on_test_batch_end', 'on_predict_batch_begin', 'on_predict_batch_end',
                        'on_train_begin', 'on_train_end', 'on_test_begin', 'on_test_end', 'on_predict_begin',
                        'on_predict_end', 'before_training_starts', 'after_training_ends', 'before_backward_pass'}

        self.callbacks = CallbacksList(callbacks=callbacks, trainer=self)
        if display_blueprint_use_warning:
            self.blueprint_use_warning()

    def blueprint_use_warning(self):
        warning_message = (
            "You have inherited from the 'TrainerBluePrint' class. "
            "Ensure that you have implemented all the required callback executions, "
            "as missing or incorrect implementations can lead to unexpected behavior.\n\n"
            "The following standard callback positions are available:\n"
            f"{', '.join(sorted(self.std_pos))}\n\n"
            "Callbacks allow you to customize and extend the training process at various stages. "
            "You can add your own custom callbacks and specify their positions using these standard positions or by defining new ones.\n\n"
            "To disable this warning in future runs, set the attribute 'display_blueprint_use_warning' to False."
        )
        self.logger.warning(warning_message)

    @abstractmethod
    def init_tracker(self):
        pass

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

    def _run_callbacks(self, pos: str) -> List[Optional[str]]:
        return self.callbacks.run_all(pos)

    @abstractmethod
    def train(self, train_loader: torch.utils.data.DataLoader):
        pass

    @abstractmethod
    def validate(self, val_loader: torch.utils.data.DataLoader):
        pass

    @property
    def current_batch(self):
        return self.__current_batch

    @property
    def current_epoch(self):
        return self.__current_epoch

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
            epochs: int = 1, epoch_start: int = 0) -> Dict:
        """
        Trains the model for the specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training datasets.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation datasets.
            epoch_start (int): from what epoch number we should start
            epochs (int): No. of epochs to run for

        Returns:
            Dict: History of training and validation metrics, and other information.
        """
        self.epochs = epochs
        self.num_batches = len(train_loader)
        self.batch_size = train_loader.batch_size
        on_gpu = True if self.device.type == 'cuda' else False
        tracker = self.tracker
        # The main Training loop

        start_time = time.time()
        self._run_callbacks(pos="before_training_starts")
        for self.__current_epoch in range(epoch_start, epoch_start + self.epochs):
            self._run_callbacks(pos="on_epoch_begin")

            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.epoch_message = f"EPOCH {self.current_epoch}: "

            # Train model
            self.train(train_loader)
            self.validate(val_loader)
            epoch_statistics = tracker.message("--> Metrics: ")
            self.logger.info(epoch_statistics)

            # Run callbacks
            self._run_callbacks(pos="on_epoch_end")

            tracker.snap_and_reset_all()

            if self.display_time_elapsed:
                end_time = time.time()
                self.logger.info(f"Time elapsed: {end_time - start_time} s")
            self.logger.info("-" * 100)
            if self.STOPPER:
                break

        self.final_metrics = self.tracker.get_final_values(self.current_epoch)
        self._run_callbacks(pos="after_training_ends")
        if self.best_model_weights is None:
            self.best_model_weights = copy.deepcopy(self.get_model_weights())
        return tracker.get_history()

    @abstractmethod
    def get_model_weights(self):
        pass

    @abstractmethod
    def predict(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        pass

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


class Trainer(TrainerBluePrint):
    def __init__(self, model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 metrics: Optional[List[Metric]] = None,
                 callbacks: Optional[List[Callback]] = None,
                 display_time_elapsed: bool = False,
                 roff: int = 5,
                 report_in_one_line: bool = True,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):

        self.metrics = [metric.name for metric in metrics]
        self.metric_fns = {metric.name: metric for metric in
                           metrics}
        super().__init__(
            callbacks=callbacks,
            display_time_elapsed=display_time_elapsed,
            roff=roff,
            report_in_one_line=report_in_one_line,
            clear_cuda_cache=clear_cuda_cache,
            use_amp=use_amp,
            name="SimpleTrainer",
            device=(device or torch.device('cpu')),
            logger=logger,
            display_blueprint_use_warning=False
        )
        self.model = self.to_device(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.final_metrics = {}

    def init_tracker(self):
        temp = self.metrics + ["loss"]
        metrics = []
        for metric in temp:
            metrics.append(metric)
            metrics.append(f"val_{metric}")
        tracker = Tracker(metrics)
        tracker.logger = self.logger
        return tracker

    def train(self, train_loader: torch.utils.data.DataLoader) -> None:
        # Set to training mode
        self.model.train()
        self._run_callbacks(pos="on_train_begin")
        for self.__current_batch, (inputs, labels) in tqdm(enumerate(train_loader), self.epoch_message):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self._run_callbacks(pos="on_train_batch_begin")

            # One Batch Training
            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            self._run_callbacks(pos="before_backward_pass")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                self.tracker.update({"loss": loss.item()})
                self.tracker.update({metric: self.metric_fns[metric](labels, outputs) for metric in self.metrics})
            self._run_callbacks(pos="on_train_batch_end")
        self._run_callbacks(pos="on_train_end")

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader) -> None:
        # Set to the evaluation mode
        self.model.eval()
        self._run_callbacks(pos="on_test_begin")
        for inputs, labels in val_loader:
            self._run_callbacks(pos="on_test_batch_begin")
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                val_loss = self.criterion(outputs, labels)

            self.tracker.update({"val_loss": val_loss.item()})
            self.tracker.update(
                {"val_" + metric: self.metric_fns[metric](labels, outputs) for metric in self.metrics})
            self._run_callbacks(pos="on_test_batch_end")
        self._run_callbacks(pos="on_test_end")

    def get_model_weights(self):
        return self.model.state_dict()

    @torch.no_grad()
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
        for batch_idx, data in enumerate(data_loader):
            self._run_callbacks(pos="on_predict_batch_begin")
            data = data.to(self.device)
            predictions = self.model(data)
            all_predictions.append(predictions)
            self._run_callbacks(pos="on_predict_batch_end")

        all_predictions = torch.cat(all_predictions, dim=0)
        self._run_callbacks(pos="on_predict_end")
        return all_predictions

    def save_progress(self, path: str, metric_name: str = "val_loss", save_trainer: bool = False):
        """
        Saves the current progress of training, including the model, optimizer, tracker, and trainer.

        Args:
            path (str): The directory where the progress should be saved.
            metric_name (str): The metric to include in the checkpoint name. Defaults to "val_loss".
            save_trainer (bool): Whether to save the entire Trainer object. Defaults to False.

        Returns:
            None
        """
        if metric_name not in self.tracker.metrics:
            available_metrics = list(self.tracker.metrics.keys())
            self.logger.warning(
                f"Metric '{metric_name}' not found in tracker. Available metrics: {available_metrics}. Cannot save progress."
            )
            return

        def save_trainer_fn(trainer, save_dir):
            model, optimizer, tracker = trainer.model, trainer.optimizer, trainer.tracker
            try:
                trainer.model, trainer.optimizer, trainer.tracker = None, None, None
                torch.save(self, os.path.join(save_dir, "trainer.pt"))
            except Exception as e:
                self.logger.warning(f"Trainer object could not be saved: {e}")
            finally:
                trainer.model, trainer.optimizer, trainer.tracker = model, optimizer, tracker

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            metric_value = self.final_metrics.get(metric_name, None) or self.tracker.metrics[metric_name].latest
            save_dir = os.path.join(path, f"{timestamp}=={metric_name}=={metric_value:.4f}")
            os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
            torch.save(self.tracker, os.path.join(save_dir, "tracker.pt"))
            self.logger.info(f"Successfully saved progress!")

            if save_trainer:
                save_trainer_fn(self, save_dir)

        except Exception as e:
            self.logger.warning(f"Failed to save progress!")
            raise e

    def load_progress(self, saved_path, mode="latest"):
        """
        Loads the training progress, including model, optimizer, and tracker states.

        Args:
            saved_path (str): The directory where saved progress folders are located.
            mode (str): Specifies which progress to load:
                        - "latest": Loads the most recent checkpoint.
                        - "low_metric": Loads the checkpoint with the lowest metric value.
                        - "high_metric": Loads the checkpoint with the highest metric value.

        Raises:
            AttributeError: If the mode is not one of "latest", "low_metric", or "high_metric".
            FileNotFoundError: If the specified path or required files are missing.
        """
        folder_names = os.listdir(saved_path)
        if not folder_names:
            raise FileNotFoundError("No saved progress found in the specified path.")

        def extract_metric_and_timestamp(folder_name):
            try:
                parts = folder_name.split("==")
                timestamp = datetime.strptime(parts[0], "%Y-%m-%d_%H-%M-%S")
                metric_value = float(parts[-1])
                return timestamp, metric_value
            except (ValueError, IndexError):
                return None, None

        progress_info = []
        for folder in folder_names:
            timestamp, metric_value = extract_metric_and_timestamp(folder)
            if timestamp and metric_value is not None:
                progress_info.append((folder, timestamp, metric_value))

        if not progress_info:
            raise FileNotFoundError("No valid progress folders found in the specified path.")

        if mode == "latest":
            folder_name = max(progress_info, key=lambda x: x[1])[0]  # Select folder with latest timestamp
        elif mode == "low_metric":
            folder_name = min(progress_info, key=lambda x: x[2])[0]  # Select folder with lowest metric
        elif mode == "high_metric":
            folder_name = max(progress_info, key=lambda x: x[2])[0]  # Select folder with highest metric
        else:
            raise AttributeError("Invalid mode. Choose from 'latest', 'low_metric', or 'high_metric'.")

        # Construct the path to the selected checkpoint
        selected_path = os.path.join(saved_path, folder_name)
        try:
            self.model.load_state_dict(torch.load(
                os.path.join(selected_path, "model.pt"), map_location=self.device, weights_only=True))
            self.optimizer.load_state_dict(
                torch.load(os.path.join(selected_path, "optimizer.pt"), map_location=self.device, weights_only=True))
            self.tracker = torch.load(os.path.join(
                selected_path, "tracker.pt"), map_location=self.device, weights_only=False)

            self.logger.info(f"Progress successfully loaded!")
        except Exception as e:
            self.logger.warning(f"Failed to load progress!")
            raise e

    class __CallbackTemplate(Callback):
        def __init__(self):
            super().__init__()


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
