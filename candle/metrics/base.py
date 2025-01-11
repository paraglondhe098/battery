import torch
from typing import Optional, Callable
from abc import abstractmethod
from candle.utils.module import Module


class Metric(Module):
    """
    Abstract base class for all metric calculations.

    Args:
        name (str): The name of the metric.
        pre_transform (Optional[Callable]): A callable function to preprocess the true and predicted values.
    """
    def __init__(self, name: str, pre_transform: Optional[Callable] = None) -> None:
        super().__init__(name)
        self.pre_trf = pre_transform or self.default_pre_transform

    @staticmethod
    def default_pre_transform(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Default preprocessing function that returns the inputs as-is.

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            tuple: Preprocessed true and predicted values.
        """
        return y_true, y_pred

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calls the metric calculation after preprocessing.

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            float: The calculated metric value.
        """
        y_true, y_pred = self.pre_trf(y_true, y_pred)
        return self.calculate(y_true, y_pred)

    @abstractmethod
    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Abstract method to calculate the metric value.

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            float: The calculated metric value.
        """
        pass
