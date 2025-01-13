import logging
# from torch.cuda.amp import GradScaler, autocast -> deprecated
from torch.amp import GradScaler, autocast
from candle.utils.tracking import Tracker
from candle.callbacks import Callback, ConsoleLogger
from candle.trainers import ProgressBar
from candle.trainers.base import TrainerBlueprint
from candle.metrics import Metric
import torch
from typing import Optional, List
from datetime import datetime
import os


class BasicTrainer(TrainerBlueprint):
    default_callbacks = [ConsoleLogger()]

    def __init__(self, model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 metrics: Optional[List[Metric]] = None,
                 callbacks: Optional[List[Callback]] = None,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):

        self.metrics = [metric.name for metric in metrics]
        super().__init__(name="SimpleTrainer", callbacks=callbacks,
                         clear_cuda_cache=clear_cuda_cache,
                         use_amp=use_amp, logger=logger,
                         device=(device or torch.device('cpu')),
                         )
        self.metric_fns = {metric.name: metric for metric in
                           metrics}
        self.progress_bar = ProgressBar([])
        self.model = self.to_device(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_state_dict = None

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
        for inputs, labels in self.progress_bar(position='train',
                                                iterable=train_loader,
                                                desc=self.epoch_headline):
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
        for inputs, labels in self.progress_bar(position='validation', iterable=val_loader, desc="Validation: "):
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

    def pre_run(self):
        pass

    def reset(self):
        self.__current_epoch = 0
        self.STOPPER = False
        self.best_state_dict = None
        self.final_metrics = {}
        self.tracker = self.init_tracker()

    def post_run(self):
        pass

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

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
            metric_value = (self.final_metrics.get(metric_name, None) if self.final_metrics else None) or \
                           self.tracker.metrics[metric_name].latest
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
