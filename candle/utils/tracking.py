"""
Module for Aggregator and Tracker classes to manage and monitor statistics for machine learning metrics.
"""

import copy
import matplotlib.pyplot as plt
from candle.utils.module import Module
from abc import abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union

Numeric = Union[int, float]


class TrainingVariable(Module):
    """
    Abstract class for training variables.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "TrainingVariable")
        self.records: List[float] = []

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def __getitem__(self, idx) -> Numeric:
        return self.records[idx]

    def __len__(self) -> int:
        return len(self.records)


class NonMetricVariable(TrainingVariable):
    """
    Tracks a non-metric variable.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initializes the NonMetricVariable instance.

        Args:
            name (str, optional): Optional name for the variable. Defaults to "NonMetricVariable".
        """
        super().__init__(name=name or "NonMetricVariable")

    def update(self, value: Any):
        """
        Update the variable with a new value.

        Args:
            value (Any): New value to assign.
        """
        self.records.append(value)

    @property
    def latest(self):
        return self.records[-1] if self.records else None


class Aggregator(TrainingVariable):
    """
    Aggregates and tracks statistics for a specific metric.

    Attributes:
        count (float): The total count of updates.
        avg (float): The running average of the metric.
        sum (float): The cumulative sum of the metric values.
        records (List[float]): History of snapshot averages.
        links (List['Aggregator']): Linked Aggregators for hierarchical updates.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initializes the Aggregator instance.

        Args:
            name (str, optional): Optional name for the Aggregator. Defaults to "Aggregator".
        """
        super().__init__(name=name or "Aggregator")
        self.count, self.avg, self.sum = 0., 0., 0.
        self.links: Optional[List['Aggregator']] = []

    def reset(self):
        """Reset all statistics to their initial state."""
        self.count, self.avg, self.sum = 0., 0., 0.

    def snapshot(self):
        """Take a snapshot of current average if valid"""
        if self.count > 0:
            self.records.append(self.avg)

    def snap_and_reset(self):
        """Take a snapshot and reset all statistics."""
        self.snapshot()
        self.reset()

    def update(self, val: Numeric, count: Numeric = 1):
        """
        Update statistics with new value

        Args:
            val: New value to incorporate
            count: Weight of the new value (default: 1)
        """
        if not isinstance(val, Numeric) or not isinstance(count, Numeric):
            raise TypeError("Values and counts must be numeric")
        val, count = float(val), float(count)
        self.update_links(val, count)

        self.count += count  # For weighing
        self.sum += count * val
        self.avg = self.sum / self.count

    def create_link(self, split: str):
        """
        Create a linked Aggregator with a specified behavior.

        Args:
            split (str): Type of link ("child", "clone", or "self").

        Returns:
            Aggregator: A new linked Aggregator instance.
        """
        if split == "child":  # Can only read values from Parent, (best for multi level averaging)
            new_link = Aggregator(f"{self.name}_child")
            self.links.append(new_link)
            return new_link
        elif split == "clone":  # Can read and modify only records from parent (best for quick access of average values from anywhere)
            clone = copy.copy(self)
            clone.links = None
            clone.name = f"{self.name}_clone"
            return clone
        elif split == "self":  # Can read and modify anything (Best for quick access)
            return self
        else:
            raise ValueError(f"Invalid split format, expected ['child' or 'clone' or 'self'] got {split}")

    def update_links(self, val: Numeric, count: Numeric = 1):
        """
        Propagate updates to linked Aggregators.

        Args:
            val (Number): The new value to propagate.
            count (Number): The weight of the new value. Defaults to 1.
        """
        for link in self.links:
            link.update(val, count)

    @property
    def latest(self):
        """
        Retrieve the most recent snapshot value.

        Returns:
            float: The last recorded snapshot value, or 0 if no records exist.
        """
        return self.avg or (self.records[-1] if self.records else 0.)


class Tracker(Module):
    """
    Tracks multiple metrics using Aggregator instances.

    Attributes:
        metrics (Dict[str, Aggregator]): Dictionary of metrics and their associated Aggregators.
        others (Dict[str, Any]): Additional attributes (e.g., epochs, learning rate).
    """

    def __init__(self, metrics: List[str]):
        """
        Initializes the Tracker with specified metrics.

        Args:
            metrics (List[str]): List of metric names to track.
        """
        super().__init__(name="Tracker")
        self.metrics: Dict[str, Aggregator] = {
            metric_name: Aggregator(f"agg_{metric_name}")
            for metric_name in metrics
        }
        self.others: Dict[str, TrainingVariable] = {}  # epochs, lr, etc.

    @property
    def variables_(self):
        return self.metrics.keys() | self.others.keys()

    def __getitem__(self, key):
        if key in self.metrics:
            return self.metrics[key]
        elif key in self.others:
            return self.others[key]
        else:
            raise KeyError(f"Invalid key: {key}")

    def __len__(self):
        return len(self.metrics) + len(self.others)

    def add_variable(self, var_name: str, exists_ok: bool = False):
        """
        Add a new variable to the tracker.

        Args:
            var_name (str): Name of the variable.
            exists_ok (bool): Whether to ignore if the variable already exists. Defaults to False (raises warning!).
        """
        if var_name in self.metrics or var_name in self.others:
            if exists_ok:
                return
            else:
                raise KeyError(f"Variable '{var_name}' already exists in tracker.")
        self.others[var_name] = NonMetricVariable()

    def update(self, metric_val_pairs: Dict[str, Numeric], count: Numeric = 1):
        """
        Update multiple metrics with new values.

        Args:
            metric_val_pairs (Dict[str, Number]): Dictionary of metric names and their values.
            count (Number): Weight to apply to all updates.
        """
        invalid_metrics = set(metric_val_pairs.keys()) - set(self.metrics.keys())
        if invalid_metrics:
            raise KeyError(f"Invalid metrics: {invalid_metrics}")

        for metric_name, val in metric_val_pairs.items():
            self.metrics[metric_name].update(val, count)

    def create_link(self, metric: str, split: str = "child"):
        """
        Create a link for a specific metric.

        Args:
            metric (str): Metric name.
            split (str): Type of link ("child", "clone", or "self").

        Returns:
            Aggregator: The linked Aggregator instance.
        """
        return self.metrics[metric].create_link(split)

    def create_links(self, metrics: list, split: str = "child"):
        """
        Create links for multiple metrics.

        Args:
            metrics (List[str]): List of metric names.
            split (str): Type of link ("child", "clone", or "self").

        Returns:
            Dict[str, Aggregator]: Dictionary of linked Aggregators.
        """
        links = {}
        for metric in metrics:
            links[metric] = self.create_link(metric, split)
        return links

    def link_all(self, split: str = "child"):
        """
        Link all tracked metrics.

        Args:
            split (str): Type of link ("child", "clone", or "self").

        Returns:
            Dict[str, Aggregator]: Dictionary of linked Aggregators.
        """
        links = {}
        for metric in self.metrics:
            links[metric] = self.create_link(metric, split)
        return links

    def message(self, prefix: str = ""):
        """
        Generate a summary message for the current averages.

        Args:
            prefix (str): Optional prefix for the message.

        Returns:
            str: Formatted message string.
        """
        joiner = " "
        text = f"{prefix} "
        for metric_name, metric in self.metrics.items():
            text += f"{joiner}{metric_name}: {metric.avg:.4f} "
            joiner = ','
        return text.strip()

    def reset_all(self):
        """Reset all tracked metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def snap_and_reset_all(self):
        """Take snapshots and reset all tracked metrics."""
        for metric in self.metrics.values():
            metric.snap_and_reset()

    def get_history(self) -> Dict[str, List[float]]:
        """
        Get the complete history of all metrics.

        Returns:
            Dict[str, List[float]]: Dictionary of metrics and their recorded history.
        """
        history = {
            metric_name: metric.records
            for metric_name, metric in self.metrics.items()
        }
        history.update({
            var_name: var.records
            for var_name, var in self.others.items()
        })
        return history

    def get_final_values(self, epoch):
        res = {"epoch": epoch}
        for metric_name in self.metrics:
            res[metric_name] = self.metrics[metric_name].latest
        return res

    def plot(self, *metrics, figsize: Tuple = (10, 6), colors: Optional[str] = None,
             line_styles: Optional[List[str]] = None, markers: Optional[List[str]] = None):
        """
        Plot metrics over time.

        Args:
            metrics (Tuple[str]): Metric names to plot.
            figsize (Tuple): Size of the plot. Defaults to (10, 6).
            colors (Optional[str]): Colors for the lines.
            line_styles (Optional[List[str]]): Line styles for the metrics.
            markers (Optional[List[str]]): Markers for the lines.
        """
        for metric in metrics:
            if not isinstance(metric, str):
                raise AttributeError(f"metric should be of type str got {type(metric)}")
        # Generate a default list of colors if none are provided
        if colors is None:
            # Generate a list of colors using a color map
            colormap = plt.get_cmap('tab10')  # You can choose other colormaps if needed
            colors = [colormap(i / len(metrics)) for i in range(len(metrics))]
        if line_styles is None:
            line_styles = ['-'] * len(metrics)
        if markers is None:
            markers = [''] * len(metrics)

        plt.figure(figsize=figsize)

        for idx, metric_name in enumerate(metrics):
            if metric_name not in self.metrics:
                self.logger.warning(f"Metric '{metric_name}' not found in metrics.")
                continue

            metric = self.metrics[metric_name].records

            epochs = list(range(len(metric)))
            plt.plot(epochs, metric, color=colors[idx], linestyle=line_styles[idx], marker=markers[idx],
                     label=metric_name)

        plt.title('Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
