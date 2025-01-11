import torch
from typing import Optional, Union, Callable
from candle.metrics import Metric


class Accuracy(Metric):
    """
    Class to calculate accuracy (both binary and multiclass).

    Args:
        binary_output (bool): Flag to indicate if binary classification is used.
        threshold (float): The threshold for binary classification predictions.
        pre_transform (Optional[Callable]): A callable function to preprocess the true and predicted values.
    """
    def __init__(self, binary_output: bool = True, threshold: float = 0.5, pre_transform: Optional[Callable] = None) -> None:
        super().__init__("accuracy", pre_transform)
        self.threshold = threshold
        self.binary_output = binary_output

    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculates accuracy based on the output type (binary or multiclass).

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            float: Accuracy score.
        """
        if self.binary_output:
            return self.binary_accuracy_score(y_true, y_pred, self.threshold)
        else:
            return self.multiclass_accuracy_score(y_true, y_pred)

    @staticmethod
    def multiclass_accuracy_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        """
        Calculates accuracy for multiclass classification.

        Args:
            labels (torch.Tensor): Ground truth labels.
            raw_predictions (torch.Tensor): Raw predicted outputs (logits).

        Returns:
            float: Accuracy score for multiclass classification.
        """
        predictions = torch.argmax(raw_predictions, dim=-1)
        correct = (predictions == labels).sum().item()
        return correct / labels.size(0)

    @staticmethod
    def binary_accuracy_score(labels: torch.Tensor, raw_predictions: torch.Tensor, threshold: float) -> float:
        """
        Calculates accuracy for binary classification.

        Args:
            labels (torch.Tensor): Ground truth labels (0 or 1).
            raw_predictions (torch.Tensor): Raw predicted outputs (logits).
            threshold (float): Threshold for binary classification.

        Returns:
            float: Accuracy score for binary classification.
        """
        if threshold == 0.5:
            predictions = torch.round(torch.sigmoid(raw_predictions))  # Assuming threshold of 0.5
        else:
            predictions = (torch.sigmoid(raw_predictions) > threshold).float()
        correct = (predictions == labels).sum().item()
        return correct / labels.size(0)


class Precision(Metric):
    """
    Class to calculate precision (both binary and multiclass).

    Args:
        binary_output (bool): Flag to indicate if binary classification is used.
        average_type (str): The type of averaging ('macro', 'micro', 'weighted').
        pre_transform (Optional[Callable]): A callable function to preprocess the true and predicted values.
    """
    def __init__(self, binary_output: bool = True, average_type: str = 'macro', pre_transform: Optional[Callable] = None) -> None:
        super().__init__("precision", pre_transform)
        self.binary_output = binary_output
        self.average_type = average_type

    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculates precision based on the output type (binary or multiclass).

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            float: Precision score.
        """
        if self.binary_output:
            return self.binary_precision_score(y_true, y_pred)
        else:
            return self.multiclass_precision_score(y_true, y_pred, self.average_type)

    @staticmethod
    def multiclass_precision_score(labels: torch.Tensor, raw_predictions: torch.Tensor,
                                   average: str = 'macro') -> Optional[Union[float, torch.tensor]]:
        """
        Calculates precision for multiclass classification.

        Args:
            labels (torch.Tensor): Ground truth labels.
            raw_predictions (torch.Tensor): Raw predicted outputs (logits).
            average (str): The type of averaging ('macro', 'micro', 'weighted').

        Returns:
            float or torch.Tensor: Precision score for multiclass classification.
        """
        y_pred = torch.argmax(raw_predictions, dim=1)
        classes = torch.unique(labels)
        precision_per_class = []

        for cls in classes:
            tp = torch.sum((labels == cls) & (y_pred == cls)).item()
            fp = torch.sum((labels != cls) & (y_pred == cls)).item()
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            precision_per_class.append(precision)

        precision_per_class = torch.tensor(precision_per_class)

        if average == 'macro':
            return precision_per_class.mean().item()
        elif average == 'micro':
            tp_total = torch.sum(labels == y_pred).item()
            fp_total = torch.sum((labels != y_pred) & (y_pred != -1)).item()
            return tp_total / (tp_total + fp_total)
        elif average == 'weighted':
            support = torch.tensor([torch.sum(labels == cls).item() for cls in classes])
            return (precision_per_class * support / support.sum()).sum().item()
        else:
            return precision_per_class

    @staticmethod
    def binary_precision_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        """
        Calculates precision for binary classification.

        Args:
            labels (torch.Tensor): Ground truth labels (0 or 1).
            raw_predictions (torch.Tensor): Raw predicted outputs (logits).

        Returns:
            float: Precision score for binary classification.
        """
        predictions = torch.round(torch.sigmoid(raw_predictions))
        tp = torch.sum((labels == 1) & (predictions == 1)).item()
        fp = torch.sum((labels == 0) & (predictions == 1)).item()
        return tp / (tp + fp) if tp + fp > 0 else 0.0


class Recall(Metric):
    """
    Class to calculate recall (both binary and multiclass).

    Args:
        binary_output (bool): Flag to indicate if binary classification is used.
        average_type (str): The type of averaging ('macro', 'micro', 'weighted').
        pre_transform (Optional[Callable]): A callable function to preprocess the true and predicted values.
    """
    def __init__(self, binary_output: bool = True, average_type: str = 'macro', pre_transform: Optional[Callable] = None) -> None:
        super().__init__("recall", pre_transform)
        self.binary_output = binary_output
        self.average_type = average_type

    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculates recall based on the output type (binary or multiclass).

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            float: Recall score.
        """
        if self.binary_output:
            return self.binary_recall_score(y_true, y_pred)
        else:
            return self.multiclass_recall_score(y_true, y_pred, self.average_type)

    @staticmethod
    def multiclass_recall_score(labels: torch.Tensor, raw_predictions: torch.Tensor,
                                average: str = 'macro') -> Optional[Union[float, torch.tensor]]:
        """
        Calculates recall for multiclass classification.

        Args:
            labels (torch.Tensor): Ground truth labels.
            raw_predictions (torch.Tensor): Raw predicted outputs (logits).
            average (str): The type of averaging ('macro', 'micro', 'weighted').

        Returns:
            float or torch.Tensor: Recall score for multiclass classification.
        """
        y_pred = torch.argmax(raw_predictions, dim=1)
        classes = torch.unique(labels)
        recall_per_class = []

        for cls in classes:
            tp = torch.sum((labels == cls) & (y_pred == cls)).item()
            fn = torch.sum((labels == cls) & (y_pred != cls)).item()
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            recall_per_class.append(recall)

        recall_per_class = torch.tensor(recall_per_class)

        if average == 'macro':
            return recall_per_class.mean().item()
        elif average == 'micro':
            tp_total = torch.sum(labels == y_pred).item()
            fn_total = torch.sum((labels != y_pred) & (y_pred != -1)).item()
            return tp_total / (tp_total + fn_total)
        elif average == 'weighted':
            support = torch.tensor([torch.sum(labels == cls).item() for cls in classes])
            return (recall_per_class * support / support.sum()).sum().item()
        else:
            return recall_per_class

    @staticmethod
    def binary_recall_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
        """
        Calculates recall for binary classification.

        Args:
            labels (torch.Tensor): Ground truth labels (0 or 1).
            raw_predictions (torch.Tensor): Raw predicted outputs (logits).

        Returns:
            float: Recall score for binary classification.
        """
        predictions = torch.round(torch.sigmoid(raw_predictions))
        tp = torch.sum((labels == 1) & (predictions == 1)).item()
        fn = torch.sum((labels == 1) & (predictions == 0)).item()
        return tp / (tp + fn) if tp + fn > 0 else 0.0
