import torch
from typing import Optional, Union, Callable
from candle.metrics import Metric


class R2Score(Metric):
    """
    Class to calculate the R² score (coefficient of determination).

    Args:
        pre_transform (Optional[Callable]): A callable function to preprocess the true and predicted values.
    """
    def __init__(self, pre_transform: Optional[Callable] = None) -> None:
        super().__init__("r2_score", pre_transform)

    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculates the R² score.

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            float: The R² score.
        """
        targets_mean = torch.mean(y_true, dim=0)
        tss = torch.sum((y_true - targets_mean) ** 2)
        rss = torch.sum((y_true - y_pred) ** 2)
        delta = 1.e-10
        tss += delta
        return (1 - rss / tss).item()